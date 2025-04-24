import torch
import tqdm
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # For distinct colors
from pathlib import Path
import datetime
from glcm import EfficientGLCM
from dataset import MillionAIDataset

class KMeansTextureClassifier:
    """
    Performs K-Means clustering on texture features (eigenvalues) derived from GLCMs.
    Implements K-Means from scratch.
    """
    def __init__(self,
                 num_clusters: int,
                 max_iterations: int = 100,
                 tolerance: float = 1e-4,
                 output_dir: str = "output/classification"):
        """
        Initializes the K-Means classifier.

        Args:
            num_clusters (int): The number of clusters (K) to find.
            max_iterations (int): Maximum number of iterations for K-Means.
            tolerance (float): Convergence threshold based on centroid movement.
            output_dir (str): Directory to save classification visualizations.
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

        # Results storage
        self.centroids: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None # 1D labels for flattened data
        self.classification_map: Optional[torch.Tensor] = None # 2D map
        self.feature_shape: Optional[Tuple] = None # Shape of the input eigenvalue maps (H, W, G)

    def _initialize_centroids(self, data: torch.Tensor) -> torch.Tensor:
        """
        Initializes centroids by randomly selecting K unique points from the data.

        Args:
            data (torch.Tensor): The input feature vectors, shape (N_pixels, G).

        Returns:
            torch.Tensor: Initial centroids, shape (K, G).
        """
        N_pixels, G_features = data.shape
        if self.K > N_pixels:
            raise ValueError(f"Number of clusters ({self.K}) cannot be greater than number of data points ({N_pixels}).")

        # Get K unique random indices
        indices = torch.randperm(N_pixels, device=data.device)[:self.K]
        centroids = data[indices]
        return centroids

    def _assign_clusters(self, data: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        Assigns each data point to the nearest centroid (based on squared Euclidean distance).

        Args:
            data (torch.Tensor): Feature vectors, shape (N, G).
            centroids (torch.Tensor): Current centroids, shape (K, G).

        Returns:
            torch.Tensor: Cluster assignments (labels) for each data point, shape (N,).
        """
        N, G = data.shape
        K = centroids.shape[0]

        # Calculate squared Euclidean distances using broadcasting (from scratch logic)
        # data shape: (N, G) -> (N, 1, G)
        # centroids shape: (K, G) -> (1, K, G)
        # Difference shape: (N, K, G)
        diff = data.unsqueeze(1) - centroids.unsqueeze(0)
        # Squared distance shape: (N, K)
        distances_sq = torch.sum(diff ** 2, dim=2)

        # Find the index of the minimum distance (nearest centroid)
        # shape: (N,)
        assignments = torch.argmin(distances_sq, dim=1)
        return assignments

    def _update_centroids(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Updates centroids by calculating the mean of points assigned to each cluster.

        Args:
            data (torch.Tensor): Feature vectors, shape (N, G).
            labels (torch.Tensor): Cluster assignments for each data point, shape (N,).

        Returns:
            torch.Tensor: New centroids, shape (K, G).
        """
        N, G = data.shape
        new_centroids = torch.zeros((self.K, G), dtype=data.dtype, device=data.device)
        points_in_cluster = torch.zeros(self.K, dtype=torch.int64, device=data.device)

        # A more efficient way using scatter_add or looping through K
        for k in range(self.K):
            # Find mask for points belonging to cluster k
            mask = (labels == k)
            points_k = data[mask] # Select points for cluster k
            count = points_k.shape[0]

            if count > 0:
                # Calculate mean for this cluster
                new_centroids[k] = torch.mean(points_k, dim=0)
            else:
                # Handle empty cluster: Re-initialize centroid randomly
                # This is a simple strategy; others exist (e.g., assign farthest point)
                print(f"Warning: Cluster {k} became empty. Re-initializing centroid randomly.")
                new_centroids[k] = self._initialize_centroids(data)[0] # Pick one random point

        return new_centroids

    def classify(self, eigenvalue_maps: torch.Tensor):
        """
        Performs K-Means clustering on the provided eigenvalue feature maps.

        Args:
            eigenvalue_maps (torch.Tensor): The computed eigenvalues tensor,
                shape (H_out, W_out, G).

        Returns:
            torch.Tensor: The 2D classification map with cluster labels,
                          shape (H_out, W_out).
        """
        if eigenvalue_maps.dim() != 3:
            raise ValueError("eigenvalue_maps must have shape (H_out, W_out, G).")

        self.feature_shape = eigenvalue_maps.shape # Store for reshaping later
        H_out, W_out, G_features = self.feature_shape

        # 1. Prepare Data: Reshape feature maps into a 2D array (N_pixels, G)
        # Ensure data is float for distance calculations
        data = eigenvalue_maps.reshape(-1, G_features).float()
        N_pixels = data.shape[0]

        print(f"Starting K-Means classification with K={self.K}...")
        print(f"Data shape: {data.shape} (N_pixels={N_pixels}, Features={G_features})")

        # 2. Initialize Centroids
        centroids = self._initialize_centroids(data)

        # 3. K-Means Iteration Loop
        for i in range(self.max_iterations):
            # Store old centroids for convergence check
            old_centroids = centroids.clone()

            # Assignment Step
            labels = self._assign_clusters(data, centroids)

            # Update Step
            centroids = self._update_centroids(data, labels)

            # Convergence Check
            centroid_shift = torch.sum((centroids - old_centroids) ** 2)
            if centroid_shift <= self.tolerance:
                print(f"K-Means converged after {i+1} iterations (centroid shift: {centroid_shift:.6f}).")
                break
            # elif i == self.max_iterations - 1:
            #      print(f"K-Means reached max iterations ({self.max_iterations}) without full convergence (shift: {centroid_shift:.6f}).")

        if i == self.max_iterations - 1 and centroid_shift > self.tolerance:
             print(f"K-Means reached max iterations ({self.max_iterations}) without full convergence (shift: {centroid_shift:.6f}).")

        # 4. Store Results
        self.centroids = centroids
        self.labels = labels # Shape (N_pixels,)

        # 5. Reshape labels back to 2D map
        self.classification_map = labels.reshape(H_out, W_out)

        print("K-Means classification finished.")
        return self.classification_map

    def visualize_classification(self,
                                 index: int, # Index for the dataset item to visualize
                                 original_rgb_image: torch.Tensor, # Original RGB image (3, H, W) uint8
                                 glcm_window_size: int # Window size for cropping
        ) -> None:
        """
        Visualizes the K-Means classification map, optionally alongside the original RGB image.

        Args:
            index (int): Index of the dataset item to visualize.
            original_rgb_image (torch.Tensor): Original RGB image (3, H, W, uint8)
                to display. It will be cropped to match the classified area.
            glcm_window_size (int): The window size used for GLCM calculation.
                Needed to correctly crop the original image.
        """
        classification_map = self.classification_map
        if classification_map.numel() == 0:
            print("Cannot visualize empty classification map.")
            return

        map_h, map_w = classification_map.shape
        map_np = classification_map.cpu().numpy() # For matplotlib

        num_plots = 2
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), squeeze=False)
        axes = axes.ravel() # Flatten axes array

        plot_idx = 0

        # --- Plot Original Image (Cropped) ---
        if not isinstance(original_rgb_image, torch.Tensor) or original_rgb_image.dim() != 3 or original_rgb_image.shape[0] != 3:
            print("Warning: Invalid original_rgb_image provided. Skipping.")
        else:
            orig_h, orig_w = original_rgb_image.shape[1], original_rgb_image.shape[2]
            # Calculate cropping margins (due to 'valid' mode in GLCM)
            margin = glcm_window_size // 2
            # Crop the original image - careful with slicing end indices
            # Valid region starts at margin, ends at H/W - margin
            if (orig_h >= map_h + 2*margin) and (orig_w >= map_w + 2*margin):
                cropped_rgb = original_rgb_image[:, margin:margin+map_h, margin:margin+map_w]
                rgb_np = cropped_rgb.permute(1, 2, 0).cpu().numpy() # H, W, C
                axes[plot_idx].imshow(rgb_np)
                axes[plot_idx].set_title(f"Original RGB (Cropped to {map_h}x{map_w})")
                axes[plot_idx].set_xticks([])
                axes[plot_idx].set_yticks([])
                plot_idx += 1
            else:
                print("Warning: Original image size incompatible with classification map size and window size for cropping. Skipping original image plot.")

        # --- Plot Classification Map ---
        # Use a discrete colormap
        cmap = plt.get_cmap('viridis', self.K) # Get K distinct colors from cmap

        im = axes[plot_idx].imshow(map_np, cmap=cmap, vmin=-0.5, vmax=self.K-0.5) # Center bins on integers
        axes[plot_idx].set_title(f"K-Means Classification (K={self.K})")
        axes[plot_idx].set_xticks([])
        axes[plot_idx].set_yticks([])

        # Add colorbar with integer ticks
        cbar = plt.colorbar(im, ax=axes[plot_idx], ticks=np.arange(self.K), fraction=0.046, pad=0.04)
        cbar.set_label('Cluster ID')
        plot_idx += 1

        # --- Final Touches & Saving ---
        title = f"Visualization of RGB Image and Classifcication Map (K={self.K})"
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95] if title else None)
        
        # Save the figure
        save_prefix = f"image_cls_{index}"
        save_filename = f"{save_prefix}_K{self.K}.png"
        final_save_path = self.output_path / save_filename
        try:
            plt.savefig(final_save_path)
            print(f"Classification visualization saved to {final_save_path}")
        except Exception as e:
            print(f"Error saving figure to {final_save_path}: {e}")
        plt.close(fig)

# --- Example Usage ---
if __name__ == "__main__":
    # Assume you have a MillionAIDataset instance 'train_dataset'
    train_dataset = MillionAIDataset(frac=0.002, is_train=True, output_dir="output/images")

    # Create a dummy dataset item for testing
    index = 3
    data_item = train_dataset[index]
    H_img, W_img = data_item['gray_pixels'].shape[1:3] # gray_pixels is (C, H, W)
    
    NUM_LEVELS = 5
    WINDOW_SIZE = 7
    DISTANCE = 1
    ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    NUM_CLUSTERS = 4 # Example K value

    # --- GLCM Part ---
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
        glcm_calculator.visualize_all(index=index)
        print(f"Eigenvalue tensor shape: {eigenvalue_tensor.shape}") # (H_out, W_out, G)

        # --- K-Means Part ---
        print(f"\n--- Running K-Means (K={NUM_CLUSTERS}) ---")
        kmeans_classifier = KMeansTextureClassifier(num_clusters=NUM_CLUSTERS)

        start_time_kmeans = datetime.datetime.now()
        classification_result_map = kmeans_classifier.classify(eigenvalue_tensor)
        end_time_kmeans = datetime.datetime.now()
        print(f"K-Means classification took: {end_time_kmeans - start_time_kmeans}")
        print(f"Classification map shape: {classification_result_map.shape}") # (H_out, W_out)

        # --- Visualize Classification ---
        print("\n--- Visualizing K-Means Result ---")
        kmeans_classifier.visualize_classification(
            index=index, # Index for the dataset item to visualize
            original_rgb_image=data_item['rgb_pixels'], # Provide original RGB
            glcm_window_size=WINDOW_SIZE, # Provide window size for cropping
        )

    else:
        print("\nSkipping K-Means classification due to empty GLCM tensor.")