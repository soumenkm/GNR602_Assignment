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
    """
    PyTorch Dataset class for loading and processing a fraction of the Million-AID dataset.

    Handles loading from Hugging Face datasets library, splitting into train/test subsets,
    applying image transformations (resize, crop), converting images to RGB and grayscale
    PyTorch tensors with pixel values in the [0, 255] range (uint8), and filtering
    out images exceeding a maximum pixel count. Also includes visualization functionality.

    Attributes:
        is_train (bool): If True, loads the training subset, otherwise loads the test subset.
        frac (float): The fraction of the total dataset to load and process.
        max_pixels (int): The maximum allowed number of pixels (width * height) for an image
                          to be included in the dataset. Images larger than this are skipped.
        output_path (Path): Path object representing the directory where visualizations are saved.
        ntsc_weights (torch.Tensor): Tensor containing NTSC weights for grayscale conversion.
        transform_rgb (transforms.Compose): Composition of torchvision transforms applied
                                            to the loaded PIL images to get RGB tensors.
        ds (List[dict]): The final processed list of data items (dictionaries), either
                         for the training or testing split. Each dictionary contains
                         'rgb_pixels', 'gray_pixels', and 'labels'.
    """
    def __init__(self, frac: float, is_train: bool, max_pixels: int = 89478485, output_dir: str = "output/images") -> None:
        """
        Initializes the MillionAIDataset instance.

        Args:
            frac (float): Fraction of the full dataset to use (e.g., 0.01 for 1%).
            is_train (bool): True to prepare the training set, False for the test set.
            max_pixels (int): Maximum number of pixels (width * height) allowed per image.
            output_dir (str): Directory path to save visualized images.
        """
        super(MillionAIDataset, self).__init__()
        self.is_train = is_train
        self.frac = frac
        self.max_pixels = max_pixels
        self.ntsc_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(3, 1, 1)
        self.transform_rgb = self._get_rgb_transform()
        self.output_path = Path.cwd() / output_dir
        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output path for visualized images: {self.output_path}")
        self.ds = self._get_dataset()

    def _get_rgb_transform(self) -> transforms.Compose:
        """
        Defines the sequence of transformations applied to PIL images to obtain
        resized, cropped, and tensorized RGB images.

        Returns:
            transforms.Compose: The transformation pipeline. Resulting tensors are
                                uint8 with shape (3, H, W) and range [0, 255].
        """
        return transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.PILToTensor(),
        ])

    def _calculate_grayscale(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Converts an RGB tensor to a grayscale tensor using NTSC standard weights.

        Args:
            rgb_tensor (torch.Tensor): Input RGB tensor, expected shape (3, H, W) and dtype uint8.

        Returns:
            torch.Tensor: Output grayscale tensor, shape (1, H, W) and dtype uint8, range [0, 255].
        """
        rgb_float = rgb_tensor.float()
        gray_float = torch.sum(rgb_float * self.ntsc_weights, dim=0, keepdim=True)
        gray_tensor = gray_float.round().clamp(0, 255).byte()
        return gray_tensor

    def _get_dataset(self) -> List[dict]:
        """
        Loads the dataset from Hugging Face, selects a fraction, shuffles, splits into
        train/test, applies transformations, converts to grayscale, filters by size,
        and returns the appropriate subset as a list of dictionaries.

        Returns:
            List[dict]: A list containing the processed data items for the specified
                        split (train or test). Each item is a dictionary:
                        {'rgb_pixels': torch.Tensor, 'gray_pixels': torch.Tensor, 'labels': any}.
        """
        print("Loading Million-AID dataset from Hugging Face...")
        try:
            full_ds = load_dataset("jonathan-roberts1/Million-AID", split="train")
        except Exception as e:
             print(f"Error loading dataset. Ensure internet connection and updated 'datasets' library.")
             print(f"If offline, ensure dataset is cached.")
             print(f"Error details: {e}")
             sys.exit(1)

        print("Dataset loaded. Processing subset...")
        length = int(len(full_ds) * self.frac)
        print(f"Using {length} images ({self.frac*100:.2f}% of total)...")
        subset_ds = full_ds.select(range(length)).shuffle(seed=42)

        size = int(len(subset_ds) * 0.8)
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
                    rgb_tensor = self.transform_rgb(image_pil)
                    gray_tensor = self._calculate_grayscale(rgb_tensor)
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
        """
        Returns the number of items in the processed dataset split.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.ds)

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a specific processed data item by index.

        Args:
            index (int): The index of the data item to retrieve.

        Returns:
            dict: A dictionary containing the 'rgb_pixels' tensor, 'gray_pixels' tensor,
                  and the 'labels' for the requested item.
        """
        return self.ds[index]

    def visualize_image(self, index: int):
        """
        Displays the RGB and Grayscale versions of the image at the specified index
        side-by-side using Matplotlib and saves the figure to the output directory.

        Args:
            index (int): The index of the dataset item to visualize.
        """
        if index >= len(self.ds):
            print(f"Error: Index {index} out of bounds for dataset size {len(self.ds)}.")
            return

        item = self.ds[index]
        rgb_tensor = item["rgb_pixels"]
        gray_tensor = item["gray_pixels"]
        label = item["labels"]

        rgb_numpy = rgb_tensor.permute(1, 2, 0).numpy()
        gray_numpy = gray_tensor.squeeze(0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(rgb_numpy)
        axes[0].set_title("RGB")
        axes[0].axis('off')

        axes[1].imshow(gray_numpy, cmap='gray')
        axes[1].set_title("Grayscale (NTSC)")
        axes[1].axis('off')

        plt.suptitle(f"Image Index: {index}\nLabel: {label}", fontsize=12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = self.output_path / f"image_{index}_comparison.png"
        try:
            plt.savefig(save_path)
            print(f"Comparison image saved to {save_path}")
        except Exception as e:
            print(f"Error saving figure to {save_path}: {e}")

        plt.close(fig)


if __name__ == "__main__":
    train_dataset = MillionAIDataset(frac=0.001, is_train=True, output_dir="output/images")

    if len(train_dataset) > 0:
        first_item = train_dataset[0]
        rgb_tensor = first_item["rgb_pixels"]
        gray_tensor = first_item["gray_pixels"]
        label = first_item["labels"]

        print("\n--- First Item Details ---")
        print(f"Label: {label}")
        print(f"RGB Tensor shape: {rgb_tensor.shape}, dtype: {rgb_tensor.dtype}, Min: {torch.min(rgb_tensor)}, Max: {torch.max(rgb_tensor)}")
        print(f"Grayscale Tensor shape: {gray_tensor.shape}, dtype: {gray_tensor.dtype}, Min: {torch.min(gray_tensor)}, Max: {torch.max(gray_tensor)}")

        print("\nVisualizing the first image...")
        train_dataset.visualize_image(0)

        if len(train_dataset) > 5:
             print("\nVisualizing image at index 5...")
             train_dataset.visualize_image(5)
    else:
        print("Dataset is empty. Check the fraction used or potential loading errors.")