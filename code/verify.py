import torch
import numpy as np
from skimage.feature import graycomatrix
from typing import List
from PIL import Image
from torchvision import transforms # For potential tensor conversion if needed

# Import your custom GLCM class (assuming it's in the same directory or path)
from glcm import EfficientGLCM

def verify_single_window(
    image_tensor_gray: torch.Tensor,
    r_start: int,
    c_start: int,
    num_levels: int,
    window_size: int,
    distance: int,
    angles: List[float], # Expects radians
    epsilon: float = 1e-9 # For rescaling consistency if needed by EfficientGLCM
    ) -> bool:
    """
    Verifies if the custom GLCM calculation for a single window matches scikit-image.

    Args:
        image_tensor_gray (torch.Tensor): Input grayscale image tensor (1, H, W) or (H, W), uint8 [0, 255].
        r_start (int): Top row index of the window to verify.
        c_start (int): Left column index of the window to verify.
        num_levels (int): Number of gray levels (G) for rescaling.
        window_size (int): Side length of the window (Ws).
        distance (int): Distance 'd' for GLCM pairs.
        angles (List[float]): List of angles in radians (e.g., [0, np.pi/4, ...]).
        epsilon (float): Epsilon value used in the custom rescale function.

    Returns:
        bool: True if the GLCM matrices match, False otherwise. Prints comparison details.
    """
    print("-" * 40)
    print("Starting Verification for Single Window")
    print(f"Params: G={num_levels}, WS={window_size}, d={distance}, Angles={np.rad2deg(angles)}")
    print(f"Window Top-Left: ({r_start}, {c_start})")

    # --- 1. Prepare Data using Custom Logic ---
    try:
        # Instantiate custom calculator just to use its rescaling method
        temp_glcm_calculator = EfficientGLCM(
            num_levels=num_levels,
            window_size=window_size, # Needed for instantiation, not used directly here
            distance=distance,
            angles=angles,
            epsilon=epsilon
        )

        # Rescale the *entire* image using the exact same logic
        rescaled_image_tensor = temp_glcm_calculator._rescale_image(image_tensor_gray) # Shape (H, W), uint8 [0, G-1]
        H, W = rescaled_image_tensor.shape

        # Extract the specific window
        r_end = r_start + window_size
        c_end = c_start + window_size
        if r_start < 0 or c_start < 0 or r_end > H or c_end > W:
            print(f"Error: Window [{r_start}:{r_end}, {c_start}:{c_end}] is out of bounds for image ({H}x{W}).")
            return False

        window_tensor = rescaled_image_tensor[r_start:r_end, c_start:c_end] # Shape (Ws, Ws)

    except Exception as e:
        print(f"Error during custom data preparation: {e}")
        return False

    # --- 2. Calculate GLCM using Custom Class Method ---
    try:
        # Use the single-window calculation method from your class
        # Needs the full rescaled image to handle offsets correctly relative to window start
        temp_glcm_calculator.rescaled_image = rescaled_image_tensor # Set internal state
        custom_glcm_tensor = temp_glcm_calculator._calculate_glcm_for_window_efficient(r_start, c_start)
        # Convert to NumPy for easier comparison later if needed
        custom_glcm_np = custom_glcm_tensor.cpu().numpy()
        print(f"\nCustom GLCM (Shape: {custom_glcm_np.shape}):\n{custom_glcm_np}")
    except Exception as e:
        print(f"Error running custom GLCM calculation: {e}")
        return False

    # --- 3. Calculate GLCM using scikit-image ---
    try:
        # Convert window tensor to NumPy array suitable for scikit-image
        window_np = window_tensor.cpu().numpy().astype(np.uint8) # Shape (Ws, Ws), uint8 [0, G-1]

        # Calculate GLCM with scikit-image
        # Note: distances needs to be a list
        # levels = num_levels (number of levels AFTER rescaling)
        skimage_glcm = graycomatrix(
            image=window_np,
            distances=[distance], # Must be list
            angles=angles,      # Pass the same list of radians
            levels=num_levels,    # Number of levels in the input window_np
            symmetric=True,       # Match custom class behavior (includes opposite offsets)
            normed=False          # Compare raw counts
        )

        # skimage_glcm shape is (levels, levels, num_distances, num_angles)
        # We summed over offsets/angles implicitly in our custom method.
        # We need to sum over the distance and angle dimensions here.
        if skimage_glcm.shape[2] != 1 or skimage_glcm.shape[3] != len(angles):
             print(f"Warning: Unexpected skimage GLCM shape: {skimage_glcm.shape}")

        # Sum over the last two dimensions (distance and angle)
        skimage_glcm_summed = np.sum(skimage_glcm, axis=(2, 3))

        print(f"\nskimage GLCM (Summed, Shape: {skimage_glcm_summed.shape}):\n{skimage_glcm_summed}")

    except ImportError:
         print("Error: scikit-image is not installed. Cannot perform verification.")
         print("Please install it: pip install scikit-image")
         return False
    except Exception as e:
        print(f"Error running scikit-image GLCM calculation: {e}")
        return False

    # --- 4. Compare Results ---
    try:
        # Ensure both are NumPy arrays of the same integer type for comparison
        match = np.array_equal(custom_glcm_np.astype(np.int64), skimage_glcm_summed.astype(np.int64))

        print("\n--- Comparison ---")
        if match:
            print("✅ SUCCESS: Custom GLCM matches scikit-image GLCM.")
        else:
            print("❌ FAILURE: Custom GLCM does NOT match scikit-image GLCM.")
            # Optional: Print difference
            # diff = custom_glcm_np.astype(np.int64) - skimage_glcm_summed.astype(np.int64)
            # print(f"Difference:\n{diff}")
        print("-" * 40)
        return match
    except Exception as e:
        print(f"Error during comparison: {e}")
        return False


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    NUM_LEVELS = 5
    WINDOW_SIZE = 5 # Keep small for easier manual verification if needed
    DISTANCE = 1
    # Use specific angles for easier comparison
    ANGLES_RAD = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    # ANGLES_RAD = [0] # Test with a single angle

    # Window position to test (top-left corner)
    R_START, C_START = 1, 1

    # --- Create Dummy Image ---
    H_IMG, W_IMG = 10, 10
    # Create an image with some structure
    dummy_gray_np = np.arange(H_IMG * W_IMG).reshape(H_IMG, W_IMG) % 256
    dummy_gray_np = dummy_gray_np.astype(np.uint8)
    # Convert to PyTorch tensor (add channel dim)
    dummy_gray_tensor = torch.from_numpy(dummy_gray_np).unsqueeze(0)
    print("Created Dummy Grayscale Tensor:")
    print(dummy_gray_tensor)

    # --- Run Verification ---
    verification_passed = verify_single_window(
        image_tensor_gray=dummy_gray_tensor,
        r_start=R_START,
        c_start=C_START,
        num_levels=NUM_LEVELS,
        window_size=WINDOW_SIZE,
        distance=DISTANCE,
        angles=ANGLES_RAD,
        epsilon=1e-9 # Match the epsilon used in your class if relevant for rescaling
    )

    if verification_passed:
        print("\nVerification successful!")
    else:
        print("\nVerification failed.")