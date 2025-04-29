import torch
import tqdm
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import datetime
from glcm import EfficientGLCM
from dataset import MillionAIDataset

class KMeansTextureClassifier:
    """
    Performs K-Means clustering on texture feature vectors.

    Implements the standard K-Means algorithm from scratch, including random initialization,
    iterative assignment of data points to nearest centroids based on Euclidean distance,
    and recalculation of centroids as the mean of assigned points. Handles potential
    empty clusters during updates. Designed to work with feature vectors derived from
    image analysis methods like GLCM eigenvalues and local mean intensity. Includes
    functionality to visualize the resulting classification map.

    Attributes:
        K (int): The desired number of clusters.
        max_iterations (int): Maximum number of iterations allowed for convergence.
        tolerance (float): Threshold for centroid movement to determine convergence.
        output_path (Path): Directory path where classification visualizations are saved.
        centroids (Optional[torch.Tensor]): Final computed cluster centroids (K, G_features).
        labels (Optional[torch.Tensor]): 1D tensor of cluster assignments for each input pixel/feature vector.
        classification_map (Optional[torch.Tensor]): 2D tensor representing the classification map (H', W').
        feature_shape (Optional[Tuple]): Original shape of the input feature maps (H', W', G_features).
    """
    def __init__(self,
                 num_clusters: int,
                 max_iterations: int = 100,
                 tolerance: float = 1e-4,
                 output_dir: str = "output/classification"):
        """
        Initializes the KMeansTextureClassifier.

        Args:
            num_clusters (int): The number of clusters (K) to find (must be >= 1).
            max_iterations (int): Maximum iterations for the K-Means algorithm (must be >= 1).
            tolerance (float): Convergence threshold based on squared centroid shift (must be >= 0).
            output_dir (str): Directory path for saving output visualizations.
        """
        if not isinstance(num_clusters, int) or num_clusters < 1:
            raise ValueError("num_clusters (K) must be a positive integer.")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        if not isinstance(tolerance, float) or tolerance < 0:
            raise ValueError("tolerance must be a non-negative float.")

        self.K = num_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.output_path = Path.cwd() / output_dir
        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"Classification output path: {self.output_path}")

        self.centroids: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.classification_map: Optional[torch.Tensor] = None
        self.feature_shape: Optional[Tuple] = None

    def _initialize_centroids(self, data: torch.Tensor) -> torch.Tensor:
        """
        Initializes K centroids by selecting K unique random data points from the input features.

        Args:
            data (torch.Tensor): Input feature vectors, shape (N_pixels, G_features).

        Returns:
            torch.Tensor: Initialized centroids, shape (K, G_features).
        """
        N_pixels, G_features = data.shape
        if self.K > N_pixels:
            raise ValueError(f"Number of clusters ({self.K}) cannot be greater than number of data points ({N_pixels}).")

        indices = torch.randperm(N_pixels, device=data.device)[:self.K]
        centroids = data[indices]
        return centroids

    def _assign_clusters(self, data: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        Assigns each data point to the cluster with the nearest centroid using squared Euclidean distance.

        Args:
            data (torch.Tensor): Feature vectors, shape (N_pixels, G_features).
            centroids (torch.Tensor): Current centroids, shape (K, G_features).

        Returns:
            torch.Tensor: 1D tensor of cluster indices (0 to K-1) for each data point, shape (N_pixels,).
        """
        N, G = data.shape
        K = centroids.shape[0]

        diff = data.unsqueeze(1) - centroids.unsqueeze(0)
        distances_sq = torch.sum(diff ** 2, dim=2)
        assignments = torch.argmin(distances_sq, dim=1)
        return assignments

    def _update_centroids(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Recalculates centroids as the mean of all data points assigned to each cluster.
        Handles empty clusters by re-initializing their centroid randomly.

        Args:
            data (torch.Tensor): Feature vectors, shape (N_pixels, G_features).
            labels (torch.Tensor): Current cluster assignments for each data point, shape (N_pixels,).

        Returns:
            torch.Tensor: Updated centroids, shape (K, G_features).
        """
        N, G = data.shape
        new_centroids = torch.zeros((self.K, G), dtype=data.dtype, device=data.device)

        for k in range(self.K):
            mask = (labels == k)
            points_k = data[mask]
            count = points_k.shape[0]

            if count > 0:
                new_centroids[k] = torch.mean(points_k, dim=0)
            else:
                print(f"Warning: Cluster {k} became empty. Re-initializing centroid randomly.")
                new_centroids[k] = self._initialize_centroids(data)[0]

        return new_centroids

    def classify(self, feature_maps: torch.Tensor):
        """
        Runs the K-Means clustering algorithm on the input feature maps.

        The input tensor typically contains features like GLCM eigenvalues and/or mean intensity
        for each pixel position corresponding to the center of a GLCM window.

        Args:
            feature_maps (torch.Tensor): Input feature tensor, shape (H', W', G_features).
                                         G_features is the number of features per pixel
                                         (e.g., G eigenvalues, or G eigenvalues + 1 mean).

        Returns:
            torch.Tensor: The resulting 2D classification map with cluster labels (0 to K-1),
                          shape (H', W'). Returns None if clustering fails or input is invalid.
        """
        if feature_maps.dim() != 3:
            raise ValueError("feature_maps must have shape (H_out, W_out, G_features).")

        self.feature_shape = feature_maps.shape
        H_out, W_out, G_features = self.feature_shape

        data = feature_maps.reshape(-1, G_features).float()
        N_pixels = data.shape[0]

        print(f"Starting K-Means classification with K={self.K}...")
        print(f"Data shape: {data.shape} (N_pixels={N_pixels}, Features={G_features})")

        if N_pixels == 0:
            print("Error: Cannot perform K-Means on empty feature data.")
            self.classification_map = None
            return None

        centroids = self._initialize_centroids(data)

        for i in range(self.max_iterations):
            old_centroids = centroids.clone()
            labels = self._assign_clusters(data, centroids)
            centroids = self._update_centroids(data, labels)
            centroid_shift = torch.sum((centroids - old_centroids) ** 2)
            if centroid_shift <= self.tolerance:
                print(f"K-Means converged after {i+1} iterations (centroid shift: {centroid_shift:.6f}).")
                break

        if i == self.max_iterations - 1 and centroid_shift > self.tolerance:
             print(f"K-Means reached max iterations ({self.max_iterations}) without full convergence (shift: {centroid_shift:.6f}).")

        self.centroids = centroids
        self.labels = labels
        self.classification_map = labels.reshape(H_out, W_out)

        print("K-Means classification finished.")
        return self.classification_map

    def visualize_classification(self,
                                 index: int,
                                 original_rgb_image: torch.Tensor,
                                 glcm_window_size: int
        ) -> None:
        """
        Visualizes the computed classification map alongside a cropped version of the original RGB image.
        Saves the resulting figure to the designated output directory. Requires the `classify` method
        to have been run successfully first.

        Args:
            index (int): An index associated with the image (used for filename).
            original_rgb_image (torch.Tensor): The original RGB image tensor (3, H, W), uint8.
                                               Used for side-by-side comparison.
            glcm_window_size (int): The window size used during GLCM computation, needed to
                                    correctly crop the original image for display.
        """
        classification_map = self.classification_map
        if classification_map is None or classification_map.numel() == 0:
            print("Error: Cannot visualize. Classification map not available or empty.")
            return

        map_h, map_w = classification_map.shape
        map_np = classification_map.cpu().numpy()

        num_plots = 1
        plot_rgb = False
        rgb_np = None

        if not isinstance(original_rgb_image, torch.Tensor) or original_rgb_image.dim() != 3 or original_rgb_image.shape[0] != 3:
            print("Warning: Invalid original_rgb_image provided for visualization.")
        else:
            orig_h, orig_w = original_rgb_image.shape[1], original_rgb_image.shape[2]
            margin = glcm_window_size // 2
            if (orig_h >= map_h + 2*margin) and (orig_w >= map_w + 2*margin):
                cropped_rgb = original_rgb_image[:, margin:margin+map_h, margin:margin+map_w]
                rgb_np = cropped_rgb.permute(1, 2, 0).cpu().numpy()
                num_plots = 2
                plot_rgb = True
            else:
                print("Warning: Original image size incompatible for cropping. Showing only classification map.")

        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), squeeze=False)
        axes = axes.ravel()
        plot_idx = 0

        if plot_rgb:
            axes[plot_idx].imshow(rgb_np)
            axes[plot_idx].set_title(f"Original RGB (Cropped to {map_h}x{map_w})")
            axes[plot_idx].axis('off')
            plot_idx += 1

        cmap = plt.get_cmap('viridis', self.K)
        im = axes[plot_idx].imshow(map_np, cmap=cmap, vmin=-0.5, vmax=self.K-0.5)
        axes[plot_idx].set_title(f"K-Means Classification (K={self.K})")
        axes[plot_idx].axis('off')

        cbar = plt.colorbar(im, ax=axes[plot_idx], ticks=np.arange(self.K), fraction=0.046, pad=0.04)
        cbar.set_label('Cluster ID')

        title = f"Visualization of RGB Image and Classification Map (K={self.K})"
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95] if title else None)

        save_prefix = f"image_cls_{index}"
        save_filename = f"{save_prefix}_K{self.K}.png"
        final_save_path = self.output_path / save_filename
        try:
            plt.savefig(final_save_path)
            print(f"Classification visualization saved to {final_save_path}")
        except Exception as e:
            print(f"Error saving figure to {final_save_path}: {e}")
        plt.close(fig)


if __name__ == "__main__":
    try:
      from dataset import MillionAIDataset
      train_dataset = MillionAIDataset(frac=0.002, is_train=True, output_dir="output/images")
      index = 3
      data_item = train_dataset[index]
    except ImportError:
      print("Warning: dataset.py not found. Using dummy data for testing kmeans.py.")
      H_img, W_img = 128, 128 # Smaller dummy data
      dummy_rgb = torch.randint(0, 256, (3, H_img, W_img), dtype=torch.uint8)
      dummy_gray = torch.randint(0, 256, (1, H_img, W_img), dtype=torch.uint8)
      data_item = {"rgb_pixels": dummy_rgb, "gray_pixels": dummy_gray, "labels": "dummy_label"}
      index = "dummy_k" # Different index for dummy data


    H_img, W_img = data_item['gray_pixels'].shape[1:3]

    NUM_LEVELS = 7
    WINDOW_SIZE = 11
    DISTANCE = 1
    ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    NUM_CLUSTERS = 6
    NUM_TOP_EIGEN = 5 # Example: use top 5 eigenvalues + mean
    INCLUDE_MEAN = True

    glcm_calculator = EfficientGLCM(
        num_levels=NUM_LEVELS,
        window_size=WINDOW_SIZE,
        distance=DISTANCE,
        angles=ANGLES
    )
    print(f"Image Size: {H_img}x{W_img}, Window: {WINDOW_SIZE}x{WINDOW_SIZE}, Levels: {NUM_LEVELS}")
    output_glcms = glcm_calculator.compute_glcms_efficient(data_item)

    if output_glcms.numel() > 0:
        eigenvalue_tensor = glcm_calculator.extract_eigenvalue_features()
        # Visualize GLCM features (optional, can be commented out)
        # glcm_calculator.visualize_all(index=index)
        print(f"Eigenvalue tensor shape: {eigenvalue_tensor.shape}")

        if INCLUDE_MEAN:
            features_for_kmeans = glcm_calculator.get_combined_features(num_top_eigenvalues=NUM_TOP_EIGEN)
        else:
             features_for_kmeans = eigenvalue_tensor[..., :NUM_TOP_EIGEN] if eigenvalue_tensor is not None else None

        if features_for_kmeans is not None and features_for_kmeans.numel() > 0:
            print(f"\n--- Running K-Means (K={NUM_CLUSTERS}) on features: {features_for_kmeans.shape} ---")
            kmeans_classifier = KMeansTextureClassifier(num_clusters=NUM_CLUSTERS)

            start_time_kmeans = datetime.datetime.now()
            classification_result_map = kmeans_classifier.classify(features_for_kmeans)
            end_time_kmeans = datetime.datetime.now()
            print(f"K-Means classification took: {end_time_kmeans - start_time_kmeans}")

            if classification_result_map is not None:
                print(f"Classification map shape: {classification_result_map.shape}")
                print("\n--- Visualizing K-Means Result ---")
                kmeans_classifier.visualize_classification(
                    index=index,
                    original_rgb_image=data_item['rgb_pixels'],
                    glcm_window_size=WINDOW_SIZE,
                )
            else:
                 print("K-Means classification failed.")
        else:
             print("Skipping K-Means: Feature preparation failed or resulted in empty features.")

    else:
        print("\nSkipping K-Means classification due to empty GLCM tensor.")