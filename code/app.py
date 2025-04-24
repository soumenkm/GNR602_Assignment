import streamlit as st
import torch
import numpy as np
from PIL import Image
import io # To handle byte stream from file uploader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import datetime
import time # For simulating work

# Import your custom classes
from glcm import EfficientGLCM
from kmeans import KMeansTextureClassifier
from torchvision import transforms # For PILToTensor

# --- Helper Functions ---

@st.cache_data # Cache image loading and conversion
def load_image_tensors(uploaded_file):
    """Loads uploaded image file and converts to RGB and Grayscale tensors."""
    try:
        image_pil = Image.open(uploaded_file)
        st.write(f"Original image mode: {image_pil.mode}, size: {image_pil.size}")

        # Ensure image is RGB
        if image_pil.mode != 'RGB':
            image_pil_rgb = image_pil.convert('RGB')
        else:
            image_pil_rgb = image_pil

        # Convert RGB PIL to Tensor [0, 255] uint8 (C, H, W)
        rgb_tensor = transforms.PILToTensor()(image_pil_rgb)

        # Convert RGB tensor to Grayscale tensor [0, 255] uint8 (1, H, W)
        if rgb_tensor.shape[0] == 3: # Check if it's indeed RGB
            ntsc_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(3, 1, 1)
            gray_float = torch.sum(rgb_tensor.float() * ntsc_weights, dim=0, keepdim=True)
            gray_tensor = gray_float.round().clamp(0, 255).byte()
        else: # Should not happen if converted to RGB, but handle grayscale input
             gray_tensor = rgb_tensor if rgb_tensor.shape[0]==1 else transforms.PILToTensor()(image_pil_rgb.convert('L'))
             if gray_tensor.dim() == 2: gray_tensor = gray_tensor.unsqueeze(0) # Add channel dim
             # Ensure RGB tensor is derived if input was grayscale
             rgb_tensor = gray_tensor.repeat(3,1,1)


        # Ensure they are uint8
        if rgb_tensor.dtype != torch.uint8:
             rgb_tensor = rgb_tensor.byte()
        if gray_tensor.dtype != torch.uint8:
             gray_tensor = gray_tensor.byte()


        st.write(f"RGB Tensor: shape={rgb_tensor.shape}, dtype={rgb_tensor.dtype}, min={rgb_tensor.min()}, max={rgb_tensor.max()}")
        st.write(f"Grayscale Tensor: shape={gray_tensor.shape}, dtype={gray_tensor.dtype}, min={gray_tensor.min()}, max={gray_tensor.max()}")

        return rgb_tensor, gray_tensor
    except Exception as e:
        st.error(f"Error loading or processing image: {e}")
        return None, None

# --- Visualization functions adapted for Streamlit (return Figure objects) ---

def create_input_eigen_figure(rgb_tensor, gray_tensor, eigenvalue_maps, G, WS, Dist):
    """Creates a matplotlib figure displaying inputs and eigenvalue maps."""
    if eigenvalue_maps is None or eigenvalue_maps.numel() == 0:
        return None # Cannot plot if no eigenvalues

    try:
        rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
        gray_np = gray_tensor.squeeze(0).cpu().numpy()
        eigen_maps_np = eigenvalue_maps.cpu().numpy()
        H_out, W_out, num_eigen_maps = eigen_maps_np.shape # G should equal num_eigen_maps

        num_total_plots = 2 + num_eigen_maps
        cols = int(np.ceil(np.sqrt(num_total_plots)))
        rows = int(np.ceil(num_total_plots / cols))

        fig = plt.figure(figsize=(cols * 4, rows * 4))
        gs = gridspec.GridSpec(rows, cols, figure=fig)
        plot_count = 0

        # Plot RGB
        ax_rgb = fig.add_subplot(gs[plot_count])
        ax_rgb.imshow(rgb_np)
        ax_rgb.set_title("Uploaded RGB")
        ax_rgb.axis('off')
        plot_count += 1

        # Plot Grayscale
        ax_gray = fig.add_subplot(gs[plot_count])
        ax_gray.imshow(gray_np, cmap='gray')
        ax_gray.set_title(f"Derived Grayscale (G={G})")
        ax_gray.axis('off')
        plot_count += 1

        # Plot Eigenvalue Maps
        for k in range(num_eigen_maps):
            if plot_count >= rows * cols: break
            ax_eigen = fig.add_subplot(gs[plot_count])
            eigen_map_k = eigen_maps_np[:, :, k]
            im = ax_eigen.imshow(eigen_map_k, cmap='viridis')
            title = f"Eigenvalue {k+1}"
            if k == 0: title += " (Largest)"
            if k == G - 1: title += " (Smallest)"
            ax_eigen.set_title(title)
            ax_eigen.axis('off')
            plt.colorbar(im, ax=ax_eigen, fraction=0.046, pad=0.04)
            plot_count += 1

        while plot_count < rows * cols:
             fig.add_subplot(gs[plot_count]).axis('off')
             plot_count += 1

        suptitle = f"Input Images & Eigenvalue Maps (G={G}, WS={WS}, Dist={Dist})"
        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    except Exception as e:
        st.error(f"Error creating input/eigenvalue visualization: {e}")
        return None

def create_classification_figure(classification_map, K, rgb_tensor=None, glcm_window_size=None):
    """Creates matplotlib figure for K-Means results, optionally with cropped RGB."""
    if classification_map is None or classification_map.numel() == 0:
        return None

    try:
        map_h, map_w = classification_map.shape
        map_np = classification_map.cpu().numpy()

        num_plots = 1
        plot_rgb = False
        rgb_np = None

        # Prepare cropped RGB if possible
        if rgb_tensor is not None and glcm_window_size is not None:
            if isinstance(rgb_tensor, torch.Tensor) and rgb_tensor.dim() == 3 and rgb_tensor.shape[0] == 3:
                orig_h, orig_w = rgb_tensor.shape[1], rgb_tensor.shape[2]
                margin = glcm_window_size // 2
                if (orig_h >= map_h + 2*margin) and (orig_w >= map_w + 2*margin):
                    cropped_rgb = rgb_tensor[:, margin:margin+map_h, margin:margin+map_w]
                    rgb_np = cropped_rgb.permute(1, 2, 0).cpu().numpy()
                    num_plots = 2
                    plot_rgb = True
                else:
                    st.warning("Original image size incompatible for cropping. Showing only classification.")
            else:
                st.warning("Invalid RGB tensor provided for classification plot.")

        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), squeeze=False)
        axes = axes.ravel()
        plot_idx = 0

        # Plot Cropped RGB
        if plot_rgb:
            axes[plot_idx].imshow(rgb_np)
            axes[plot_idx].set_title(f"Original RGB (Cropped to {map_h}x{map_w})")
            axes[plot_idx].axis('off')
            plot_idx += 1

        # Plot Classification Map
        cmap = plt.get_cmap('viridis', K)
        im = axes[plot_idx].imshow(map_np, cmap=cmap, vmin=-0.5, vmax=K-0.5)
        axes[plot_idx].set_title(f"K-Means Classification (K={K})")
        axes[plot_idx].axis('off')
        cbar = plt.colorbar(im, ax=axes[plot_idx], ticks=np.arange(K), fraction=0.046, pad=0.04)
        cbar.set_label('Cluster ID')

        suptitle = f"K-Means Texture Classification Result (K={K})"
        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    except Exception as e:
        st.error(f"Error creating classification visualization: {e}")
        return None


# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("GLCM Eigenvalue Texture Classification")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("1. Upload Image")
    uploaded_file = st.file_uploader("Choose a PNG or JPG image", type=["png", "jpg", "jpeg"])

    st.header("2. Configure Parameters")
    # GLCM Parameters
    g_level = st.selectbox("Gray Levels (G)", [5, 6, 7], index=0) # Default G=5
    window_size = st.number_input("GLCM Window Size (odd, >=3)", min_value=3, max_value=31, value=7, step=2)
    distance = st.number_input("GLCM Distance (d)", min_value=1, max_value=10, value=1, step=1)
    # K-Means Parameters
    k_clusters = st.number_input("Number of Clusters (K)", min_value=2, max_value=20, value=4, step=1)
    max_iter = st.number_input("K-Means Max Iterations", min_value=10, max_value=500, value=100, step=10)
    tolerance = st.number_input("K-Means Convergence Tolerance", min_value=1e-6, max_value=1e-2, value=1e-4, step=1e-5, format="%.5f")

    st.header("3. Run Analysis")
    process_button = st.button("Process Image")

# --- Main Area for Outputs ---
if uploaded_file is not None:
    # Load and display the uploaded image immediately
    st.subheader("Uploaded Image")
    rgb_tensor, gray_tensor = load_image_tensors(uploaded_file)

    if rgb_tensor is not None:
         # Display RGB using st.image for quick preview
         st.image(rgb_tensor.permute(1, 2, 0).cpu().numpy(), caption="Uploaded RGB Image", use_container_width=True)
    else:
         st.warning("Could not load image tensors.")


    if process_button:
        # --- Input Validation ---
        valid_input = True
        if window_size % 2 == 0:
            st.error("Window size must be an odd number.")
            valid_input = False
        if rgb_tensor is None or gray_tensor is None:
             st.error("Image tensors could not be loaded. Cannot process.")
             valid_input = False
        if gray_tensor.shape[1] < window_size or gray_tensor.shape[2] < window_size:
             st.error(f"Image dimensions ({gray_tensor.shape[1]}x{gray_tensor.shape[2]}) are smaller than window size ({window_size}). Cannot compute GLCM.")
             valid_input = False


        if valid_input:
            st.info(f"Processing with G={g_level}, WS={window_size}, d={distance}, K={k_clusters}...")

            # Add tensors to a dictionary format expected by GLCM class
            # Ensure gray_tensor has channel dim (1, H, W)
            if gray_tensor.dim() == 2: gray_tensor = gray_tensor.unsqueeze(0)
            data_item = {"rgb_pixels": rgb_tensor, "gray_pixels": gray_tensor}

            # --- Instantiate Classes ---
            try:
                glcm_calculator = EfficientGLCM(
                    num_levels=g_level,
                    window_size=window_size,
                    distance=distance,
                    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4] # Keep angles fixed for simplicity
                )

                kmeans_classifier = KMeansTextureClassifier(
                    num_clusters=k_clusters,
                    max_iterations=max_iter,
                    tolerance=tolerance
                )
            except Exception as e:
                st.error(f"Error initializing calculators: {e}")
                st.stop() # Stop execution if setup fails

            # --- Run Pipeline ---
            eigenvalue_tensor = None
            classification_map = None

            try:
                # 1. Compute GLCMs
                with st.spinner("Calculating GLCMs..."):
                    start_glcm = time.time()
                    # Use the faster vectorized method
                    output_glcms = glcm_calculator.compute_glcms_efficient(data_item)
                    end_glcm = time.time()
                    st.write(f"GLCM computation took: {end_glcm - start_glcm:.2f} seconds.")

                if output_glcms is None or output_glcms.numel() == 0:
                    st.warning("GLCM computation resulted in empty output. Check parameters or image size.")
                else:
                    st.write(f"GLCM tensor shape: {output_glcms.shape}")
                    # 2. Extract Eigenvalues
                    with st.spinner("Extracting Eigenvalues..."):
                        start_eigen = time.time()
                        eigenvalue_tensor = glcm_calculator.extract_eigenvalue_features() # Uses self.output_glcms
                        end_eigen = time.time()
                        st.write(f"Eigenvalue extraction took: {end_eigen - start_eigen:.2f} seconds.")
                        st.write(f"Eigenvalue maps shape: {eigenvalue_tensor.shape}")

                    if eigenvalue_tensor is None or eigenvalue_tensor.numel() == 0:
                         st.warning("Eigenvalue extraction failed or resulted in empty output.")
                    else:
                        # 3. Run K-Means
                        with st.spinner(f"Running K-Means (K={k_clusters})..."):
                            start_kmeans = time.time()
                            classification_map = kmeans_classifier.classify(eigenvalue_tensor)
                            end_kmeans = time.time()
                            st.write(f"K-Means classification took: {end_kmeans - start_kmeans:.2f} seconds.")
                            st.write(f"Classification map shape: {classification_map.shape}")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.exception(e) # Show traceback for debugging


            # --- Display Visualizations ---
            st.divider()
            st.subheader("Analysis Results")

            # Plot Inputs and Eigenvalue maps
            fig1 = create_input_eigen_figure(
                rgb_tensor=rgb_tensor,
                gray_tensor=gray_tensor.squeeze(0), # Pass H,W version
                eigenvalue_maps=eigenvalue_tensor,
                G=g_level,
                WS=window_size,
                Dist=distance
            )
            if fig1:
                st.pyplot(fig1)
            else:
                st.warning("Could not generate Input/Eigenvalue visualization.")

            # Plot Classification map
            fig2 = create_classification_figure(
                classification_map=classification_map,
                K=k_clusters,
                rgb_tensor=rgb_tensor,
                glcm_window_size=window_size
            )
            if fig2:
                st.pyplot(fig2)
            else:
                st.warning("Could not generate K-Means classification visualization.")

            st.success("Processing complete!")

elif process_button:
    st.warning("Please upload an image first.")

st.sidebar.markdown("---")
st.sidebar.info("Upload an image and set parameters, then click 'Process Image'.")