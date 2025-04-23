from datasets import load_dataset
import torch
import tqdm
import os
import sys
import pickle
import datetime
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List, Union
from torchvision import transforms
import matplotlib.pyplot as plt

torch.manual_seed(42)

class MillionAIDataset(Dataset):
    def __init__(self, frac: float, is_train: bool, max_pixels: int = 89478485, output_dir: str = "output/images") -> None:
        super(MillionAIDataset, self).__init__()
        self.is_train = is_train
        self.frac = frac
        self.max_pixels = max_pixels
        # Define NTSC weights for grayscale conversion
        self.ntsc_weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1) # Reshape for broadcasting (C, H, W)

        # The transform pipeline will resize, crop, and convert RGB to tensor [0, 255]
        self.transform_rgb = self._get_rgb_transform()

        # Setup output path
        self.output_path = Path.cwd() / output_dir
        self.output_path.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist
        print(f"Output path for visualized images: {self.output_path}")

        # Load and prepare the dataset subset (train or test)
        self.ds = self._get_dataset()

    def _get_rgb_transform(self) -> transforms.Compose:
        # Transforms applied BEFORE grayscale conversion
        # Outputs RGB tensor [0, 255], uint8
        return transforms.Compose([
            transforms.Resize(512),              # Resize smaller dimension to 512
            transforms.CenterCrop(512),          # Crop center 512x512
            transforms.PILToTensor(),            # Convert PIL Image to Tensor (C, H, W) in [0, 255], uint8
        ])

    def _calculate_grayscale(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Calculates grayscale tensor using NTSC weights."""
        # Ensure input is float for weighted sum
        rgb_float = rgb_tensor.float()
        # Apply NTSC weights: (3, H, W) * (3, 1, 1) -> sum over C -> (1, H, W)
        gray_float = torch.sum(rgb_float * self.ntsc_weights, dim=0, keepdim=True)
        # Convert back to uint8, clamping values just in case of minor floating point inaccuracies
        gray_tensor = gray_float.round().clamp(0, 255).byte() # .byte() is equivalent to .to(torch.uint8)
        return gray_tensor # Shape (1, H, W), dtype uint8

    def _get_dataset(self) -> List[dict]:
        # Load the specified split from Hugging Face datasets
        print("Loading Million-AID dataset from Hugging Face...")
        try:
            full_ds = load_dataset("jonathan-roberts1/Million-AID", split="train")
        except Exception as e:
             print(f"Error loading dataset. Ensure you have internet connection and 'datasets' library is updated.")
             print(f"If you are offline, ensure the dataset is cached.")
             print(f"Error details: {e}")
             sys.exit(1)

        print("Dataset loaded. Processing subset...")
        length = int(len(full_ds) * self.frac)
        print(f"Using {length} images ({self.frac*100:.2f}% of total)...")
        subset_ds = full_ds.select(range(length)).shuffle(seed=42)

        size = int(len(subset_ds) * 0.8) # 80% train, 20% test
        processed_ds = []

        if self.is_train:
            indices = range(size)
            desc = "Preparing train dataset..."
        else:
            indices = range(size, len(subset_ds))
            desc = "Preparing test dataset..."

        with tqdm.tqdm(indices, desc=desc, colour="green", unit="examples") as pbar:
            for i in pbar:
                try:
                    image_pil = subset_ds[i]["image"]
                    if image_pil.mode != 'RGB':
                       image_pil = image_pil.convert('RGB')
                except Exception as e:
                    print(f"\nWarning: Skipping image index {i} due to loading/conversion error: {e}")
                    continue

                if image_pil.size[0] * image_pil.size[1] <= self.max_pixels:
                    # Apply transformations to get RGB tensor
                    rgb_tensor = self.transform_rgb(image_pil) # Shape (3, H, W), uint8 [0, 255]

                    # Calculate grayscale tensor from RGB tensor
                    gray_tensor = self._calculate_grayscale(rgb_tensor) # Shape (1, H, W), uint8 [0, 255]

                    # Append dictionary with both tensors and the label
                    processed_ds.append({
                        "rgb_pixels": rgb_tensor,
                        "gray_pixels": gray_tensor,
                        "labels": subset_ds[i]["label_3"]
                    })
                else:
                    print(f"\nSkipping image index {i} due to exceeding max_pixels ({image_pil.size[0] * image_pil.size[1]} > {self.max_pixels}).")

        print(f"Finished preparing {'train' if self.is_train else 'test'} dataset. Found {len(processed_ds)} valid images.")
        return processed_ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> dict:
        # Returns {'rgb_pixels': tensor, 'gray_pixels': tensor, 'labels': label}
        return self.ds[index]

    def visualize_image(self, index: int):
        """
        Visualizes the RGB and Grayscale images side-by-side for the given index
        and saves the figure to the output directory.
        """
        if index >= len(self.ds):
            print(f"Error: Index {index} out of bounds for dataset size {len(self.ds)}.")
            return

        # Get the data dictionary
        item = self.ds[index]
        rgb_tensor = item["rgb_pixels"]   # Shape (3, H, W), uint8 [0, 255]
        gray_tensor = item["gray_pixels"] # Shape (1, H, W), uint8 [0, 255]
        label = item["labels"]

        # --- Convert tensors for matplotlib display ---
        # RGB: Needs (H, W, C)
        rgb_numpy = rgb_tensor.permute(1, 2, 0).numpy()
        # Grayscale: Needs (H, W)
        gray_numpy = gray_tensor.squeeze(0).numpy() # Remove channel dim

        # --- Create figure with two subplots ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 6)) # 1 row, 2 columns

        # Plot RGB
        axes[0].imshow(rgb_numpy)
        axes[0].set_title("RGB")
        axes[0].axis('off')

        # Plot Grayscale
        axes[1].imshow(gray_numpy, cmap='gray') # Use grayscale colormap
        axes[1].set_title("Grayscale (NTSC)")
        axes[1].axis('off')

        # Add overall title
        plt.suptitle(f"Image Index: {index}\nLabel: {label}", fontsize=12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

        # --- Save the figure ---
        save_path = self.output_path / f"image_{index}_comparison.png"
        try:
            plt.savefig(save_path)
            print(f"Comparison image saved to {save_path}")
        except Exception as e:
            print(f"Error saving figure to {save_path}: {e}")

        # Close the plot figure to free memory, especially if calling this in a loop
        plt.close(fig)


if __name__ == "__main__":
    # Create the training dataset instance (using 0.1% of the data for speed)
    train_dataset = MillionAIDataset(frac=0.001, is_train=True, output_dir="output/images")

    # Check if the dataset was loaded successfully
    if len(train_dataset) > 0:
        # Get the first item
        first_item = train_dataset[0]
        rgb_tensor = first_item["rgb_pixels"]
        gray_tensor = first_item["gray_pixels"]
        label = first_item["labels"]

        # Print information about the first item's tensors
        print("\n--- First Item Details ---")
        print(f"Label: {label}")
        print(f"RGB Tensor shape: {rgb_tensor.shape}, dtype: {rgb_tensor.dtype}, Min: {torch.min(rgb_tensor)}, Max: {torch.max(rgb_tensor)}")
        print(f"Grayscale Tensor shape: {gray_tensor.shape}, dtype: {gray_tensor.dtype}, Min: {torch.min(gray_tensor)}, Max: {torch.max(gray_tensor)}")

        # Visualize the first image (RGB and Grayscale)
        print("\nVisualizing the first image...")
        train_dataset.visualize_image(0)

        # Example: Visualize another image if available
        if len(train_dataset) > 5:
             print("\nVisualizing image at index 5...")
             train_dataset.visualize_image(5)
    else:
        print("Dataset is empty. Check the fraction used or potential loading errors.")

