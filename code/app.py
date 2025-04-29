import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import datetime
import time

from glcm import EfficientGLCM
from kmeans import KMeansTextureClassifier
from torchvision import transforms

@st.cache_data
def load_image_tensors(uploaded_file):
    """
    Loads an image file uploaded via Streamlit, converts it to RGB PIL Image,
    and generates both RGB and Grayscale PyTorch tensors.

    Ensures output tensors are uint8 with pixel values in the range [0, 255].
    Handles basic image mode conversions.

    Args:
        uploaded_file: The file object received from st.file_uploader.

    Returns:
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple containing:
            - rgb_tensor: Tensor of the image in RGB format (3, H, W), uint8. None on error.
            - gray_tensor: Tensor of the image in Grayscale format (1, H, W), uint8. None on error.
    """
    try:
        image_pil = Image.open(uploaded_file)

        if image_pil.mode != 'RGB':
            image_pil_rgb = image_pil.convert('RGB')
        else:
            image_pil_rgb = image_pil

        rgb_tensor = transforms.PILToTensor()(image_pil_rgb)

        if rgb_tensor.shape[0] == 3:
            ntsc_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(3, 1, 1)
            gray_float = torch.sum(rgb_tensor.float() * ntsc_weights, dim=0, keepdim=True)
            gray_tensor = gray_float.round().clamp(0, 255).byte()
        else:
             gray_tensor = rgb_tensor if rgb_tensor.shape[0]==1 else transforms.PILToTensor()(image_pil_rgb.convert('L'))
             if gray_tensor.dim() == 2: gray_tensor = gray_tensor.unsqueeze(0)
             rgb_tensor = gray_tensor.repeat(3,1,1)

        if rgb_tensor.dtype != torch.uint8:
             rgb_tensor = rgb_tensor.byte()
        if gray_tensor.dtype != torch.uint8:
             gray_tensor = gray_tensor.byte()

        return rgb_tensor, gray_tensor
    except Exception as e:
        st.error(f"Error loading or processing image: {e}")
        return None, None

def create_input_eigen_figure(rgb_tensor,
                              gray_tensor,
                              eigenvalue_maps,
                              G, WS, Dist,
                              include_mean,
                              mean_map,
                              num_eigen_for_kmeans
                             ):
    """
    Creates and returns a Matplotlib figure visualizing input images and feature maps.

    Displays the uploaded RGB image, the derived/rescaled grayscale image, all G
    calculated eigenvalue maps, and optionally the computed mean map. The figure's
    super title indicates the parameters used for the subsequent K-Means step.

    Args:
        rgb_tensor (torch.Tensor): Original RGB image tensor (3, H, W).
        gray_tensor (torch.Tensor): Derived grayscale tensor (1, H, W) or (H, W).
        eigenvalue_maps (torch.Tensor): Calculated eigenvalue maps (H_out, W_out, G).
        G (int): Number of gray levels used.
        WS (int): GLCM window size used.
        Dist (int): GLCM distance used.
        include_mean (bool): Flag indicating if the mean map should be plotted.
        mean_map (Optional[torch.Tensor]): Calculated mean map (H_out, W_out). Required if include_mean is True.
        num_eigen_for_kmeans (int): The number of top eigenvalues selected for K-Means (used for title).

    Returns:
        matplotlib.figure.Figure or None: The generated Matplotlib figure, or None if visualization fails.
    """
    if eigenvalue_maps is None or eigenvalue_maps.numel() == 0:
        st.warning("Eigenvalue maps are empty, cannot visualize.")
        return None
    if include_mean and (mean_map is None or mean_map.numel() == 0):
        st.warning("Mean map requested but is empty, skipping mean map plot.")
        include_mean = False

    try:
        rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
        if gray_tensor.dim() == 3: gray_np = gray_tensor.squeeze(0).cpu().numpy()
        else: gray_np = gray_tensor.cpu().numpy()
        eigen_maps_np = eigenvalue_maps.cpu().numpy()
        H_out, W_out, num_eigen_maps_actual = eigen_maps_np.shape

        mean_map_np = None
        if include_mean:
            mean_map_np = mean_map.cpu().numpy()
            if mean_map_np.shape != (H_out, W_out):
                 st.warning(f"Mean map shape mismatch. Skipping mean map plot.")
                 include_mean = False

        num_plots_shown = 2 + num_eigen_maps_actual + (1 if include_mean else 0)
        cols = int(np.ceil(np.sqrt(num_plots_shown)))
        rows = int(np.ceil(num_plots_shown / cols))

        fig = plt.figure(figsize=(cols * 4, rows * 4))
        gs = gridspec.GridSpec(rows, cols, figure=fig)
        plot_count = 0

        ax_rgb = fig.add_subplot(gs[plot_count]); plot_count += 1
        ax_rgb.imshow(rgb_np); ax_rgb.set_title("Uploaded RGB"); ax_rgb.axis('off')

        ax_gray = fig.add_subplot(gs[plot_count]); plot_count += 1
        ax_gray.imshow(gray_np, cmap='gray'); ax_gray.set_title(f"Derived Grayscale (G={G})"); ax_gray.axis('off')

        if include_mean:
             if plot_count < rows * cols:
                 ax_mean = fig.add_subplot(gs[plot_count]); plot_count += 1
                 im_mean = ax_mean.imshow(mean_map_np, cmap='plasma')
                 ax_mean.set_title("Mean Map (Rescaled Gray)"); ax_mean.axis('off')
                 plt.colorbar(im_mean, ax=ax_mean, fraction=0.046, pad=0.04)

        for k in range(num_eigen_maps_actual):
            if plot_count >= rows * cols: break
            ax_eigen = fig.add_subplot(gs[plot_count]); plot_count += 1
            eigen_map_k = eigen_maps_np[:, :, k]
            im = ax_eigen.imshow(eigen_map_k, cmap='viridis')
            title = f"Eigenvalue {k+1}"
            if k == 0: title += " (Largest)"
            if k == G - 1: title += " (Smallest)"
            ax_eigen.set_title(title); ax_eigen.axis('off')
            plt.colorbar(im, ax=ax_eigen, fraction=0.046, pad=0.04)

        while plot_count < rows * cols:
             fig.add_subplot(gs[plot_count]).axis('off'); plot_count += 1

        kmeans_feat_str = f"Top {num_eigen_for_kmeans} Eigen"
        if include_mean:
             kmeans_feat_str += " + Mean"

        suptitle = (f"Inputs & Feature Maps (G={G}, WS={WS}, D={Dist})\n"
                    f"[K-Means used: {kmeans_feat_str}]")
        fig.suptitle(suptitle, fontsize=14)

        fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        return fig

    except Exception as e:
        st.error(f"Error creating input/feature visualization: {e}")
        st.exception(e)
        return None


def create_classification_figure(classification_map, K, rgb_tensor=None, glcm_window_size=None):
    """
    Creates and returns a Matplotlib figure visualizing the K-Means classification result.

    Optionally displays the original RGB image, cropped to the classified area,
    alongside the classification map.

    Args:
        classification_map (torch.Tensor): The 2D tensor of cluster labels (H_out, W_out).
        K (int): The number of clusters used in K-Means.
        rgb_tensor (Optional[torch.Tensor]): Original RGB image tensor (3, H, W) for comparison display.
        glcm_window_size (Optional[int]): Window size used for GLCM, required for cropping rgb_tensor.

    Returns:
        matplotlib.figure.Figure or None: The generated Matplotlib figure, or None if visualization fails.
    """
    if classification_map is None or classification_map.numel() == 0: return None
    try:
        map_h, map_w = classification_map.shape
        map_np = classification_map.cpu().numpy()
        num_plots = 1; plot_rgb = False; rgb_np = None
        if rgb_tensor is not None and glcm_window_size is not None:
            if isinstance(rgb_tensor, torch.Tensor) and rgb_tensor.dim()==3 and rgb_tensor.shape[0]==3:
                orig_h, orig_w = rgb_tensor.shape[1], rgb_tensor.shape[2]
                margin = glcm_window_size // 2
                if (orig_h >= map_h + 2*margin) and (orig_w >= map_w + 2*margin):
                    cropped_rgb = rgb_tensor[:, margin:margin+map_h, margin:margin+map_w]
                    rgb_np = cropped_rgb.permute(1, 2, 0).cpu().numpy()
                    num_plots = 2; plot_rgb = True
                else: st.warning("Orig size incompatible for cropping.")
            else: st.warning("Invalid RGB tensor for classification plot.")
        fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6), squeeze=False)
        axes = axes.ravel(); plot_idx = 0
        if plot_rgb:
            axes[plot_idx].imshow(rgb_np); axes[plot_idx].set_title(f"Orig RGB (Cropped)"); axes[plot_idx].axis('off'); plot_idx += 1
        cmap = plt.get_cmap('viridis', K)
        im = axes[plot_idx].imshow(map_np, cmap=cmap, vmin=-0.5, vmax=K-0.5)
        axes[plot_idx].set_title(f"K-Means Classification (K={K})"); axes[plot_idx].axis('off')
        cbar = plt.colorbar(im, ax=axes[plot_idx], ticks=np.arange(K), fraction=0.046, pad=0.04)
        cbar.set_label('Cluster ID')
        suptitle = f"K-Means Texture Classification Result (K={K})"
        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
    except Exception as e: st.error(f"Error creating classification viz: {e}"); return None


st.set_page_config(layout="wide")
st.title("GLCM Eigenvalue Texture Classification")

with st.sidebar:
    st.header("1. Upload Image")
    uploaded_file = st.file_uploader("Choose a PNG or JPG image", type=["png", "jpg", "jpeg"])

    st.header("2. Configure Parameters")
    g_level = st.selectbox("Gray Levels (G)", [5, 6, 7], index=0, key="g_level")
    window_size = st.number_input("GLCM Window Size (odd, >=3)", min_value=3, max_value=31, value=7, step=2, key="window_size")
    distance = st.number_input("GLCM Distance (d)", min_value=1, max_value=10, value=1, step=1, key="distance")

    num_top_eigenvalues = st.number_input(
        f"Number of Top Eigenvalues (1 to G={g_level})",
        min_value=1, max_value=g_level, value=g_level, step=1, key="num_eigen"
    )
    include_mean_feature = st.checkbox("Include Mean Feature for K-Means", value=True, key="include_mean")

    k_clusters = st.number_input("Number of Clusters (K)", min_value=2, max_value=20, value=4, step=1, key="k_clusters")
    max_iter = st.number_input("K-Means Max Iterations", min_value=10, max_value=500, value=100, step=10, key="max_iter")
    tolerance = st.number_input("K-Means Convergence Tolerance", min_value=1e-6, max_value=1e-2, value=1e-4, step=1e-5, format="%.5f", key="tolerance")

    st.header("3. Run Analysis")
    process_button = st.button("Process Image", key="process_btn")

if uploaded_file is not None:
    st.subheader("Uploaded Image Preview")
    rgb_tensor, gray_tensor = load_image_tensors(uploaded_file)

    if rgb_tensor is not None:
         st.write(f"Image Dimensions: {rgb_tensor.shape[1]} H x {rgb_tensor.shape[2]} W")
         st.image(rgb_tensor.permute(1, 2, 0).cpu().numpy(), caption="Uploaded RGB Image", use_container_width=False, width=300)
    else:
         st.warning("Could not load image tensors.")


    if process_button:
        valid_input = True
        if window_size % 2 == 0: st.error("Window size must be odd."); valid_input = False
        if rgb_tensor is None or gray_tensor is None: st.error("Image tensors missing."); valid_input = False
        if valid_input and (gray_tensor.shape[-2] < window_size or gray_tensor.shape[-1] < window_size):
             st.error(f"Image dimensions too small for window size {window_size}."); valid_input = False
        if not (1 <= num_top_eigenvalues <= g_level): st.error(f"Num top eigenvalues invalid."); valid_input = False

        if valid_input:
            mean_feat_str = "MeanFeat=True" if include_mean_feature else "MeanFeat=False"
            st.info(f"Processing: G={g_level}, WS={window_size}, d={distance}, TopEigen={num_top_eigenvalues}, {mean_feat_str}, K={k_clusters}...")
            if gray_tensor.dim() == 2: gray_tensor = gray_tensor.unsqueeze(0)
            data_item = {"rgb_pixels": rgb_tensor, "gray_pixels": gray_tensor}

            try:
                glcm_calculator = EfficientGLCM(
                    num_levels=g_level, window_size=window_size, distance=distance,
                    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]
                )
                kmeans_classifier = KMeansTextureClassifier(
                    num_clusters=k_clusters, max_iterations=max_iter, tolerance=tolerance
                )
            except Exception as e: st.error(f"Init error: {e}"); st.stop()

            eigenvalue_tensor = None
            mean_map_tensor = None
            features_for_kmeans = None
            classification_map = None

            try:
                with st.spinner("Calculating GLCMs & Mean Map..."):
                    start_glcm = time.time()
                    output_glcms = glcm_calculator.compute_glcms_efficient(data_item)
                    mean_map_tensor = glcm_calculator.mean_map
                    end_glcm = time.time()
                    st.write(f"GLCM & Mean computation: {end_glcm - start_glcm:.2f}s.")

                if output_glcms is None or output_glcms.numel() == 0: st.warning("GLCM empty."); st.stop()
                if mean_map_tensor is None or mean_map_tensor.numel() == 0: st.warning("Mean map empty."); st.stop()
                st.write(f"GLCM shape: {output_glcms.shape}, Mean map shape: {mean_map_tensor.shape}")

                with st.spinner("Extracting Eigenvalues..."):
                    start_eigen = time.time()
                    eigenvalue_tensor = glcm_calculator.extract_eigenvalue_features()
                    end_eigen = time.time()
                    st.write(f"Eigenvalue extraction: {end_eigen - start_eigen:.2f}s.")

                if eigenvalue_tensor is None or eigenvalue_tensor.numel() == 0: st.warning("Eigenvalues empty."); st.stop()
                st.write(f"ALL Eigenvalue maps shape: {eigenvalue_tensor.shape}")

                if include_mean_feature:
                    st.write(f"Combining top {num_top_eigenvalues} eigenvalues with mean feature...")
                    features_for_kmeans = glcm_calculator.get_combined_features(num_top_eigenvalues=num_top_eigenvalues)
                else:
                    st.write(f"Using top {num_top_eigenvalues} eigenvalues only...")
                    features_for_kmeans = eigenvalue_tensor[..., :num_top_eigenvalues]

                if features_for_kmeans is None or features_for_kmeans.numel() == 0: st.error("Failed to prepare features for K-Means."); st.stop()
                st.write(f"Features shape for K-Means: {features_for_kmeans.shape}")

                with st.spinner(f"Running K-Means..."):
                    start_kmeans = time.time()
                    classification_map = kmeans_classifier.classify(features_for_kmeans)
                    end_kmeans = time.time()
                    st.write(f"K-Means classification: {end_kmeans - start_kmeans:.2f}s.")

                if classification_map is None: st.warning("K-Means failed."); st.stop()
                st.write(f"Classification map shape: {classification_map.shape}")

            except Exception as e: st.error(f"Processing error: {e}"); st.exception(e); st.stop()

            st.divider(); st.subheader("Analysis Results")

            fig1 = create_input_eigen_figure(
                rgb_tensor=rgb_tensor,
                gray_tensor=gray_tensor.squeeze(0),
                eigenvalue_maps=eigenvalue_tensor,
                G=g_level, WS=window_size, Dist=distance,
                include_mean=include_mean_feature,
                mean_map=mean_map_tensor,
                num_eigen_for_kmeans=num_top_eigenvalues
            )
            if fig1: st.pyplot(fig1)
            else: st.warning("Could not generate Input/Feature visualization.")

            fig2 = create_classification_figure(
                classification_map=classification_map,
                K=k_clusters,
                rgb_tensor=rgb_tensor,
                glcm_window_size=window_size
            )
            if fig2: st.pyplot(fig2)
            else: st.warning("Could not generate K-Means classification visualization.")

            st.success("Processing complete!")

elif process_button:
    st.warning("Please upload an image first.")

st.sidebar.markdown("---")
st.sidebar.info("Upload an image, set parameters (incl. Mean Feature option), then click 'Process Image'.")