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
        Computes the GLCM for each pixel using an efficient sliding window.
        Handles boundaries by only computing GLCMs where the full window fits (valid mode).

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
             self.output_glcms = torch.zeros((0, 0, self.num_levels, self.num_levels), dtype=torch.int64)
             return self.output_glcms

        # Initialize the output tensor to store GLCMs for each valid center pixel
        glcms_array = torch.zeros((out_H, out_W, self.num_levels, self.num_levels), dtype=torch.int64)

        # --- Efficient Sliding Window Calculation ---
        print("Starting efficient GLCM computation...")

        # 1. Calculate GLCM for the initial top-left window (0, 0)
        current_glcm = self._calculate_glcm_for_window(0, 0)
        glcms_array[0, 0] = current_glcm.clone() # Store a copy

        # 2. Slide horizontally across the first row
        # print("Processing first row...")
        for c_win in tqdm.trange(1, out_W, desc="First Row", leave=False, colour="cyan"):
            # Window moves from (0, c_win-1) to (0, c_win)
            # Subtract contributions from the leftmost column leaving the window
            leaving_c = c_win - 1
            for r_win in range(self.window_size): # Iterate through rows within the column
                r_img = r_win # Image row index
                # Subtract pairs involving pixels at (r_img, leaving_c)
                # Careful: Need to consider pairs where the neighbor was *inside* the old window
                self._update_glcm_efficient(current_glcm, r_img, leaving_c, -1, 0, c_win-1)

            # Add contributions from the rightmost column entering the window
            entering_c = c_win + self.window_size - 1
            for r_win in range(self.window_size): # Iterate through rows within the column
                r_img = r_win # Image row index
                # Add pairs involving pixels at (r_img, entering_c)
                # Careful: Need to consider pairs where the neighbor is *inside* the new window
                self._update_glcm_efficient(current_glcm, r_img, entering_c, +1, 0, c_win)

            glcms_array[0, c_win] = current_glcm.clone()


        # 3. Slide vertically and horizontally for subsequent rows
        # print("Processing subsequent rows...")
        # We need the GLCM from the window *above* to update vertically
        glcm_above = torch.zeros_like(current_glcm) # Temp storage

        for r_win in tqdm.trange(1, out_H, desc="Rows", colour="magenta"):
            # --- Vertical Update for the first column ---
            # Update from glcms_array[r_win-1, 0] to get glcms_array[r_win, 0]
            glcm_above = glcms_array[r_win-1, 0] # Get the GLCM from the window directly above
            current_glcm = glcm_above.clone() # Start with the GLCM from above

            # Subtract contributions from the top row leaving the window
            leaving_r = r_win - 1
            for c_win_col0 in range(self.window_size): # Iterate through columns within the row
                c_img = c_win_col0 # Image col index
                # Subtract pairs involving pixel (leaving_r, c_img) relevant to the window ABOVE
                self._update_glcm_efficient(current_glcm, leaving_r, c_img, -1, r_win - 1, 0)

            # Add contributions from the bottom row entering the window
            entering_r = r_win + self.window_size - 1
            for c_win_col0 in range(self.window_size): # Iterate through columns within the row
                c_img = c_win_col0 # Image col index
                # Add pairs involving pixel (entering_r, c_img) relevant to the CURRENT window
                self._update_glcm_efficient(current_glcm, entering_r, c_img, +1, r_win, 0)

            glcms_array[r_win, 0] = current_glcm.clone() # Store GLCM for the first column of this row

            # --- Horizontal Updates for the rest of the row ---
            for c_win in range(1, out_W):
                # Update horizontally from glcms_array[r_win, c_win-1]
                # (using the GLCM we just stored or calculated)
                # Subtract contributions from the leftmost column leaving the window
                leaving_c = c_win - 1
                for r_win_row in range(self.window_size):
                    r_img = r_win + r_win_row
                    self._update_glcm_efficient(current_glcm, r_img, leaving_c, -1, r_win, c_win - 1)

                # Add contributions from the rightmost column entering the window
                entering_c = c_win + self.window_size - 1
                for r_win_row in range(self.window_size):
                    r_img = r_win + r_win_row
                    self._update_glcm_efficient(current_glcm, r_img, entering_c, +1, r_win, c_win)

                glcms_array[r_win, c_win] = current_glcm.clone()

        print("Finished GLCM computation.")
        self.output_glcms = glcms_array
        return self.output_glcms

    def _update_glcm_efficient(self, glcm: torch.Tensor, pixel_r: int, pixel_c: int, sign: int, win_r_start: int, win_c_start: int):
        """
        Efficiently updates GLCM by considering pairs involving a specific pixel (pixel_r, pixel_c)
        that is entering (+1) or leaving (-1) the calculation scope, ensuring the pairs
        counted/removed are relevant to the window defined by (win_r_start, win_c_start).

        Args:
            glcm (torch.Tensor): The GxG GLCM tensor to update (int64).
            pixel_r (int): Row index of the specific pixel entering/leaving.
            pixel_c (int): Column index of the specific pixel.
            sign (int): +1 if pixel is entering (add pairs), -1 if leaving (subtract pairs).
            win_r_start (int): Starting row index of the relevant window boundary.
            win_c_start (int): Starting column index of the relevant window boundary.
        """
        val1 = self.rescaled_image[pixel_r, pixel_c].item()
        win_r_end = win_r_start + self.window_size
        win_c_end = win_c_start + self.window_size

        for dr, dc in self.offsets:
            neighbor_r, neighbor_c = pixel_r + dr, pixel_c + dc

            # Check if the neighbor is within the relevant window boundaries
            if win_r_start <= neighbor_r < win_r_end and win_c_start <= neighbor_c < win_c_end:
                 # Boundary check for image limits should implicitly be handled if window is valid
                 # but double check doesn't hurt, though rescaled_image access handles it
                 # if 0 <= neighbor_r < H and 0 <= neighbor_c < W: # H,W from self.rescaled_image.shape
                val2 = self.rescaled_image[neighbor_r, neighbor_c].item()
                glcm[val1, val2] += sign
                # Note: This assumes the GLCM should be symmetric. If we only computed 4 offsets initially,
                # we would need to add glcm[val2, val1] += sign here as well.
                # Since we calculated 8 offsets, symmetry is implicitly handled IF the offset list is symmetric.

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
    output_glcm_tensor = glcm_calculator.compute_glcms(dummy_item)
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

         # Verify counts (optional, harder to predict exactly)
         # Sum of GLCM elements should be the number of pairs counted in that window
         print(f"Sum of sample GLCM elements: {torch.sum(sample_glcm)}")
         # Expected sum = window_size * window_size * num_offsets_used (approx)
         # num_offsets_used depends on how many unique (dr, dc) pairs were generated.
         print(f"Number of offsets used (symmetric): {len(glcm_calculator.offsets)}")

    # You would typically proceed to calculate features (like eigenvalues)
    # from each GxG matrix in the output_glcm_tensor.