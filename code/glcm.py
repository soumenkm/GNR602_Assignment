import torch, datetime
import tqdm
import numpy as np
from typing import List, Tuple, Dict
from dataset import MillionAIDataset 

class EfficientGLCM:
    """
    Calculates Gray Level Co-occurrence Matrices (GLCMs) efficiently
    using a sliding window approach with updates.
    """
    def __init__(self,
                 num_levels: int = 5,
                 window_size: int = 5,
                 distance: int = 1,
                 angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4] # Radians for calculation
                 ):
        """
        Initializes the GLCM calculator parameters.

        Args:
            num_levels (int): The number of gray levels (G) to rescale the image to.
            window_size (int): The side length of the square sliding window (e.g., 5 for 5x5).
            distance (int): The distance (d) between pixel pairs for co-occurrence.
            angles (List[float]): List of angles (in radians) to calculate GLCM for.
                                  Common angles: 0, pi/4, pi/2, 3pi/4.
        """
        if num_levels not in [5, 6, 7]:
            raise ValueError("num_levels must be 5, 6, or 7.")
        if not isinstance(window_size, int) or window_size < 2 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3 for centered windows")
        if not isinstance(distance, int) or distance < 1:
            raise ValueError("distance must be an integer >= 1")

        self.num_levels = num_levels
        self.window_size = window_size
        self.window_radius = window_size // 2
        self.distance = distance
        self.angles = angles

        # Calculate offsets (delta_row, delta_col) for each angle and distance
        # We compute for 8 directions to simplify symmetric updates later
        self.offsets = self._calculate_offsets(distance, angles)

        # Internal state, will be set when process_image is called
        self.original_shape = None
        self.rescaled_image = None
        self.output_glcms = None # Shape: (H, W, num_levels, num_levels) or smaller if valid mode

    def _calculate_offsets(self, distance: int, angles: List[float]) -> List[Tuple[int, int]]:
        """Calculates (dr, dc) offsets for given distances and angles."""
        offsets = []
        for angle in angles:
            dr = round(-distance * np.sin(angle)) # Row offset (negative sin for image coords)
            dc = round(distance * np.cos(angle))  # Col offset
            offsets.append((dr, dc))
            # Add symmetric offset if not already included (e.g., for 0 or pi)
            if (dr, dc) != (-dr, -dc) or angle == 0 or angle == np.pi:
                 offsets.append((-dr, -dc)) # Symmetric pair
        # Remove duplicates if any angle resulted in the same offset as another's symmetric pair
        return list(set(offsets))

    def _rescale_image(self, gray_tensor: torch.Tensor) -> torch.Tensor:
        """
        Rescales the input grayscale tensor (0-255) to the desired number of levels (0 to G-1).

        Args:
            gray_tensor (torch.Tensor): Input grayscale image tensor, shape (1, H, W), dtype uint8.

        Returns:
            torch.Tensor: Rescaled image tensor, shape (H, W), dtype uint8.
        """
        if gray_tensor.dim() == 3 and gray_tensor.shape[0] == 1:
            gray_tensor = gray_tensor.squeeze(0) # Shape (H, W)
        elif gray_tensor.dim() != 2:
             raise ValueError("Input gray_tensor must be (1, H, W) or (H, W)")

        if gray_tensor.dtype != torch.uint8:
             print("Warning: Input tensor is not uint8. Assuming range 0-255 for rescaling.")
             gray_tensor = gray_tensor.byte() # Convert to uint8

        # Perform rescaling: floor((pixel_value / 256.0) * num_levels)
        # Using float division
        rescaled = (gray_tensor.float() / 256.0) * self.num_levels
        # Floor and clamp to ensure values are in [0, num_levels-1]
        rescaled = torch.floor(rescaled).clamp(0, self.num_levels - 1)

        return rescaled.byte() # Convert back to uint8 for efficient indexing

    def _update_glcm(self, glcm: torch.Tensor, center_r: int, center_c: int, sign: int):
        """
        Adds or subtracts pairs associated with a single pixel entering/leaving the window's calculation scope.

        Args:
            glcm (torch.Tensor): The GxG GLCM tensor to update (int64).
            center_r (int): Row index of the pixel whose pairs are being added/subtracted.
            center_c (int): Column index of the pixel.
            sign (int): +1 to add pairs, -1 to subtract pairs.
        """
        H, W = self.rescaled_image.shape
        val1 = self.rescaled_image[center_r, center_c].item() # Value of the center pixel

        for dr, dc in self.offsets:
            neighbor_r, neighbor_c = center_r + dr, center_c + dc

            # Check if the neighbor is within the image bounds
            if 0 <= neighbor_r < H and 0 <= neighbor_c < W:
                # Crucially: Check if this neighbor ALSO falls within the current window's boundary
                # This check depends on how the update functions call this method.
                # Let's assume the calling update function ensures the pair was/is relevant to the window.
                val2 = self.rescaled_image[neighbor_r, neighbor_c].item()
                glcm[val1, val2] += sign

    def _calculate_glcm_for_window_efficient(self, r_start: int, c_start: int) -> torch.Tensor:
        """
        Calculates the GLCM from scratch for a specific window using vectorized operations.

        Args:
            r_start (int): Top row index of the window in the full rescaled image.
            c_start (int): Left column index of the window in the full rescaled image.

        Returns:
            torch.Tensor: The GxG GLCM tensor for the window (int64), guaranteed symmetric.
        """
        # Ensure rescaled_image is available
        if self.rescaled_image is None:
             raise RuntimeError("Rescaled image not available. Call compute_glcms first or set self.rescaled_image.")

        H, W = self.rescaled_image.shape
        G = self.num_levels
        WS = self.window_size

        # 1. Extract the window from the rescaled image
        r_end = r_start + WS
        c_end = c_start + WS
        # Ensure window boundaries are valid (should be handled by calling context, but check doesn't hurt)
        if r_end > H or c_end > W:
             raise ValueError(f"Window [{r_start}:{r_end}, {c_start}:{c_end}] exceeds image bounds ({H}x{W})")
        window = self.rescaled_image[r_start:r_end, c_start:c_end] # Shape: (WS, WS)

        # 2. Initialize the GLCM
        # Use float32 for intermediate bincount accumulation if needed, then convert
        glcm = torch.zeros((G, G), dtype=torch.int64, device=window.device) # Keep on same device

        # 3. Iterate through each offset (vectorize the inner loops)
        for dr, dc in self.offsets:
            # Calculate slices for the two views corresponding to the offset
            # View 1: Origin pixels
            r1_start = max(0, -dr)
            r1_end   = min(WS, WS - dr)
            c1_start = max(0, -dc)
            c1_end   = min(WS, WS - dc)

            # View 2: Neighbor pixels (shifted by offset)
            r2_start = max(0, dr)
            r2_end   = min(WS, WS + dr)
            c2_start = max(0, dc)
            c2_end   = min(WS, WS + dc)

            # Check if the slices are valid (have positive size)
            if r1_start >= r1_end or c1_start >= c1_end:
                continue # This offset doesn't produce valid pairs within the window

            # Extract the overlapping views
            view1 = window[r1_start:r1_end, c1_start:c1_end]
            view2 = window[r2_start:r2_end, c2_start:c2_end]

            # Flatten the views to get lists of values for pairs
            vals1 = view1.flatten() # Shape: (N_pairs,)
            vals2 = view2.flatten() # Shape: (N_pairs,)

            # Ensure they have the same number of elements (should always be true if logic is correct)
            assert vals1.shape == vals2.shape

            # 4. Calculate linear indices for the pairs (val1, val2)
            # linear_index = val1 * num_levels + val2
            # Use long() for indices as bincount expects LongTensor indices
            linear_indices = (vals1.long() * G + vals2.long())

            # 5. Count occurrences of each linear index
            # minlength ensures the output has size G*G, even if some pairs don't occur
            pair_counts = torch.bincount(linear_indices, minlength=G * G)

            # 6. Reshape counts into GxG matrix and add to the total GLCM
            # View as GxG and add to the running total
            glcm += pair_counts.view(G, G)

        # The resulting GLCM is inherently symmetric because we included all symmetric offsets
        # and counted pairs derived from them.
        return glcm

    def _calculate_glcm_for_window(self, r_start: int, c_start: int) -> torch.Tensor:
        """
        Calculates the GLCM from scratch for a specific window.
        Used for initialization and potentially for the first window of each row.
        """
        glcm = torch.zeros((self.num_levels, self.num_levels), dtype=torch.int64)
        H, W = self.rescaled_image.shape
        r_end = r_start + self.window_size
        c_end = c_start + self.window_size

        # Iterate through each pixel *within* the window
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                val1 = self.rescaled_image[r, c].item()

                # Check neighbors based on offsets
                for dr, dc in self.offsets:
                    neighbor_r, neighbor_c = r + dr, c + dc

                    # Check if the neighbor is ALSO within the *same window*
                    if r_start <= neighbor_r < r_end and c_start <= neighbor_c < c_end:
                        val2 = self.rescaled_image[neighbor_r, neighbor_c].item()
                        glcm[val1, val2] += 1
        return glcm
    
    def compute_glcms(self, image_item: Dict) -> torch.Tensor:
        """
        Computes the GLCM for each pixel using a sliding window.
        This recalculates the GLCM from scratch for every window position.

        Args:
            image_item (Dict): A dictionary from MillionAIDataset containing 'gray_pixels'.
                               Expected gray_pixels shape: (1, H, W), dtype uint8, range [0, 255].

        Returns:
            torch.Tensor: A tensor containing the GLCM for each valid pixel center.
                          Shape: (out_H, out_W, num_levels, num_levels), dtype int64.
                          out_H = H - window_size + 1
                          out_W = W - window_size + 1
        """
        if 'gray_pixels' not in image_item:
            raise ValueError("Input dictionary must contain 'gray_pixels' tensor.")

        gray_tensor = image_item['gray_pixels']
        self.original_shape = gray_tensor.shape # Keep original shape if needed
        self.rescaled_image = self._rescale_image(gray_tensor)
        H, W = self.rescaled_image.shape

        # Calculate output dimensions (valid convolution mode)
        out_H = H - self.window_size + 1
        out_W = W - self.window_size + 1

        if out_H <= 0 or out_W <= 0:
             print(f"Warning: Window size ({self.window_size}) is larger than image dimensions ({H}x{W}). No GLCMs computed.")
             # Return tensor with correct dimensions but size 0
             self.output_glcms = torch.zeros((0, 0, self.num_levels, self.num_levels), dtype=torch.int64)
             return self.output_glcms

        # Initialize the output tensor to store GLCMs for each valid center pixel
        glcms_array = torch.zeros((out_H, out_W, self.num_levels, self.num_levels), dtype=torch.int64)

        # --- Sliding Window Calculation ---
        print("Starting GLCM computation...")

        # Iterate through all possible top-left corners (r_start, c_start) of the window
        for r_start in tqdm.trange(out_H, desc="Rows", colour="yellow"):
            for c_start in range(out_W):
                # Calculate GLCM from scratch for the window starting at (r_start, c_start)
                # Using the vectorized version for the single window calculation is still best practice
                glcm = self._calculate_glcm_for_window_efficient(r_start, c_start)

                # Store the computed GLCM in the output array
                # The index [r_start, c_start] corresponds to the window whose top-left is (r_start, c_start)
                glcms_array[r_start, c_start] = glcm

        print("Finished GLCM computation.")
        self.output_glcms = glcms_array # Store result
        return self.output_glcms
    
    def compute_glcms_efficient(self, image_item: Dict) -> torch.Tensor:
        """
        Computes the GLCM for each pixel using a vectorized approach based on unfolding.
        Effectively calculates the GLCM from scratch for every window simultaneously.

        Args:
            image_item (Dict): A dictionary containing 'gray_pixels'.
                               Expected shape: (1, H, W), dtype uint8.

        Returns:
            torch.Tensor: A tensor containing the GLCM for each valid pixel center.
                          Shape: (out_H, out_W, num_levels, num_levels), dtype int64.
        """
        if 'gray_pixels' not in image_item:
            raise ValueError("Input dictionary must contain 'gray_pixels' tensor.")

        gray_tensor = image_item['gray_pixels']
        self.original_shape = gray_tensor.shape
        self.rescaled_image = self._rescale_image(gray_tensor) # Shape (H, W)
        H, W = self.rescaled_image.shape
        G = self.num_levels
        WS = self.window_size
        device = self.rescaled_image.device # Work on the same device

        # Calculate output dimensions
        out_H = H - WS + 1
        out_W = W - WS + 1

        if out_H <= 0 or out_W <= 0:
             print(f"Warning: Window size ({WS}) > image dimensions ({H}x{W}). No GLCMs computed.")
             self.output_glcms = torch.zeros((0, 0, G, G), dtype=torch.int64, device=device)
             return self.output_glcms

        # --- Vectorized Calculation using unfold and scatter_add_ ---
        print("Starting Efficient GLCM computation...")

        # 1. Create sliding window views of the rescaled image
        # Unfold along rows (dimension 0)
        unfolded_rows = self.rescaled_image.unfold(0, WS, 1) # Shape: (out_H, W, WS)
        # Unfold along columns (dimension 1 of the *result* from above)
        # Note: unfold dimension indexes the dimension *of the tensor it's called on*
        all_windows = unfolded_rows.unfold(1, WS, 1) # Shape: (out_H, out_W, WS, WS)

        # Ensure the unfolded view is contiguous in memory for efficiency later if needed
        all_windows = all_windows.contiguous()

        # 2. Initialize the final GLCM tensor
        glcms_array = torch.zeros((out_H, out_W, G, G), dtype=torch.int64, device=device)

        # 3. Iterate through offsets (vectorize calculations *within* each offset)
        for dr, dc in tqdm.tqdm(self.offsets, desc="Vectorized Offsets", colour="blue", leave=False):
            # Calculate slices within the WS x WS window
            r1_start, r1_end = max(0, -dr), min(WS, WS - dr)
            c1_start, c1_end = max(0, -dc), min(WS, WS - dc)
            r2_start, r2_end = max(0, dr), min(WS, WS + dr)
            c2_start, c2_end = max(0, dc), min(WS, WS + dc)

            # Check if the slices are valid
            if r1_start >= r1_end or c1_start >= c1_end:
                continue # No valid pairs for this offset within any window

            # Extract the two views *across all windows* simultaneously
            # all_windows has shape (out_H, out_W, WS, WS)
            all_views1 = all_windows[:, :, r1_start:r1_end, c1_start:c1_end]
            all_views2 = all_windows[:, :, r2_start:r2_end, c2_start:c2_end]
            # Shape of views: (out_H, out_W, H_view, W_view) where H_view/W_view depend on slices

            # Flatten the window dimensions (H_view, W_view) to get pairs for each window
            # Shape becomes (out_H, out_W, N_pairs) where N_pairs = H_view * W_view
            all_vals1 = all_views1.reshape(out_H, out_W, -1)
            all_vals2 = all_views2.reshape(out_H, out_W, -1)

            # Calculate linear indices for all pairs across all windows
            # Shape: (out_H, out_W, N_pairs)
            all_linear_indices = (all_vals1.long() * G + all_vals2.long())

            # Use scatter_add_ for efficient counting across all windows for this offset
            # We need a temporary tensor to accumulate counts for this specific offset
            # Shape: (out_H, out_W, G*G)
            offset_glcm_flat = torch.zeros(out_H, out_W, G * G, dtype=torch.int64, device=device)

            # scatter_add_ needs source tensor of same shape as index, containing values to add (here, all 1s)
            src = torch.ones_like(all_linear_indices, dtype=torch.int64)

            # Perform the scatter operation along the last dimension (dim=2)
            # index=all_linear_indices tells scatter_add_ *which* position in the G*G dimension to increment
            offset_glcm_flat.scatter_add_(dim=2, index=all_linear_indices, src=src)

            # Reshape the flat offset GLCMs back to (out_H, out_W, G, G)
            offset_glcm = offset_glcm_flat.view(out_H, out_W, G, G)

            # Add the contribution from this offset to the total GLCM array
            glcms_array += offset_glcm

        print("Finished Efficient GLCM computation.")
        self.output_glcms = glcms_array # Store result
        return self.output_glcms # (H - WS + 1, W - WS + 1, G, G)
    
# --- Example Usage ---
if __name__ == "__main__":
    # Assume you have a MillionAIDataset instance 'train_dataset'
    # Create a dummy dataset item for testing
    dummy_rgb = torch.randint(0, 256, (3, 512, 512), dtype=torch.uint8)
    dummy_gray = torch.randint(0, 256, (1, 512, 512), dtype=torch.uint8) # Use a dummy grayscale
    dummy_item = {"rgb_pixels": dummy_rgb, "gray_pixels": dummy_gray, "labels": "dummy_label"}

    # --- Configuration ---
    NUM_LEVELS = 5
    WINDOW_SIZE = 7 # Must be odd
    DISTANCE = 1
    ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4] # Common angles

    # --- Initialize GLCM calculator ---
    glcm_calculator = EfficientGLCM(
        num_levels=NUM_LEVELS,
        window_size=WINDOW_SIZE,
        distance=DISTANCE,
        angles=ANGLES
    )

    # --- Process the dummy image ---
    print(f"Processing image with shape: {dummy_item['gray_pixels'].shape}")
    start_time = datetime.datetime.now()
    output_glcm_tensor = glcm_calculator.compute_glcms_efficient(dummy_item)
    end_time = datetime.datetime.now()
    print(f"GLCM computation took: {end_time - start_time}")

    # --- Inspect Results ---
    print(f"Output GLCM tensor shape: {output_glcm_tensor.shape}")
    # Expected shape: (H - WS + 1, W - WS + 1, G, G)
    # e.g., (512 - 7 + 1, 512 - 7 + 1, 5, 5) -> (506, 506, 5, 5)

    if output_glcm_tensor.numel() > 0:
        # Check a sample GLCM (e.g., for the center pixel of the output)
        center_h = output_glcm_tensor.shape[0] // 2
        center_w = output_glcm_tensor.shape[1] // 2
        sample_glcm = output_glcm_tensor[center_h, center_w]
        print(f"\nSample GLCM at output center ({center_h}, {center_w}):")
        print(sample_glcm)

        # Verify symmetry (optional)
        is_symmetric = torch.all(sample_glcm == sample_glcm.t())
        print(f"Is the sample GLCM symmetric? {is_symmetric}") # Should be true
