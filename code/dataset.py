from datasets import load_dataset
import torch, tqdm, os, sys, pickle, datetime
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List, Union
from torchvision import transforms
import matplotlib.pyplot as plt # Import matplotlib

torch.manual_seed(42)

class MillionAIDataset(Dataset):
    def __init__(self, frac: float, is_train: bool, max_pixels: int = 89478485) -> None:
        super(MillionAIDataset, self).__init__()
        self.is_train = is_train
        self.frac = frac
        self.max_pixels = max_pixels
        # The transform pipeline will resize, crop, and convert to tensor [0, 255]
        self.transform = self._get_transform()
        # Load and prepare the dataset subset (train or test)
        self.ds = self._get_dataset()
        self.output_path = Path.cwd() / "output/image"
        self.output_path.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist

    def _get_transform(self) -> transforms.Compose:
        # Use PILToTensor to keep the range [0, 255] and integer type
        return transforms.Compose([
            transforms.Resize(512),              # Resize smaller dimension to 512
            transforms.CenterCrop(512),          # Crop center 512x512
            transforms.PILToTensor(),            # Convert PIL Image to Tensor (C, H, W) in [0, 255], uint8
        ])

    def _get_dataset(self) -> List[dict]:
        # Load the specified split from Hugging Face datasets
        print("Loading Million-AID dataset from Hugging Face...")
        # Note: Might take time on first download
        try:
            full_ds = load_dataset("jonathan-roberts1/Million-AID", split="train")
        except Exception as e:
             print(f"Error loading dataset. Ensure you have internet connection and 'datasets' library is updated.")
             print(f"If you are offline, ensure the dataset is cached.")
             print(f"Error details: {e}")
             sys.exit(1) # Exit if dataset loading fails

        print("Dataset loaded. Processing subset...")
        # Select a fraction of the dataset
        length = int(len(full_ds) * self.frac)
        print(f"Using {length} images ({self.frac*100:.2f}% of total)...")
        # Select the subset and shuffle
        subset_ds = full_ds.select(range(length)).shuffle(seed=42)

        # Define split point for train/test
        size = int(len(subset_ds) * 0.8) # 80% for training, 20% for testing
        processed_ds = []

        # Determine the range for iteration based on whether it's train or test
        if self.is_train:
            indices = range(size)
            desc = "Preparing train dataset..."
        else:
            indices = range(size, len(subset_ds))
            desc = "Preparing test dataset..."

        # Iterate through the selected indices
        with tqdm.tqdm(indices, desc=desc, colour="green", unit="examples") as pbar:
            for i in pbar:
                # Get the PIL image object
                try:
                    # Accessing the image might trigger download/decompression
                    image_pil = subset_ds[i]["image"]
                    # Convert image mode if necessary (e.g., grayscale 'L' or RGBA to RGB)
                    if image_pil.mode != 'RGB':
                       image_pil = image_pil.convert('RGB')
                except Exception as e:
                    print(f"\nWarning: Skipping image index {i} due to loading/conversion error: {e}")
                    continue # Skip to next image

                # Check pixel count before potentially large operations
                if image_pil.size[0] * image_pil.size[1] <= self.max_pixels:
                    # Apply the transformations (Resize, Crop, PILToTensor)
                    image_tensor = self.transform(image_pil)
                    # Append the processed tensor and label
                    processed_ds.append({"pixels": image_tensor, "labels": subset_ds[i]["label_3"]})
                else:
                    # This message might not appear if initial loading fails for large images
                    print(f"\nSkipping image index {i} due to exceeding max_pixels ({image_pil.size[0] * image_pil.size[1]} > {self.max_pixels}).")

        print(f"Finished preparing {'train' if self.is_train else 'test'} dataset. Found {len(processed_ds)} valid images.")
        return processed_ds

    def __len__(self) -> int:
        # Return the length of the processed dataset (train or test subset)
        return len(self.ds)

    def __getitem__(self, index: int) -> dict:
        # Retrieve the pre-processed item (dictionary of tensor and label)
        return self.ds[index]

    def visualize_image(self, index: int):
        """
        Visualizes the image at the given index using matplotlib.
        """
        if index >= len(self.ds):
            print(f"Error: Index {index} out of bounds for dataset size {len(self.ds)}.")
            return

        # Get the data dictionary
        item = self.ds[index]
        image_tensor = item["pixels"] # Tensor shape (C, H, W), dtype uint8, range [0, 255]
        label = item["labels"]

        # --- Convert tensor for matplotlib display ---
        # Matplotlib expects (H, W, C) for RGB images and values in [0, 255] for uint8
        # or [0.0, 1.0] for float. Our tensor is uint8 [0, 255] but (C, H, W).
        image_numpy = image_tensor.permute(1, 2, 0).numpy() # Permute to (H, W, C) and convert to numpy

        # --- Display the image ---
        plt.figure(figsize=(6, 6))
        plt.imshow(image_numpy)
        plt.title(f"Image Index: {index}\nLabel: {label}", fontsize=10)
        plt.axis('off') # Hide axes
        path = str(self.output_path) + f"/image_{index}.png"
        plt.savefig(path) # Save the figure
        print(f"Image saved as {path}")

if __name__ == "__main__":
    # Create the training dataset instance (using 1% of the data)
    # Set is_train=False for the test set
    train_dataset = MillionAIDataset(frac=0.01, is_train=True)

    # Check if the dataset was loaded successfully
    if len(train_dataset) > 0:
        # Get the first item
        first_item = train_dataset[0]
        image_tensor = first_item["pixels"]
        label = first_item["labels"]

        # Print information about the first item's tensor
        print("\n--- First Item Details ---")
        print(f"Label: {label}")
        print(f"Tensor shape: {image_tensor.shape}")      # Should be [3, 224, 224]
        print(f"Tensor dtype: {image_tensor.dtype}")      # Should be torch.uint8
        print(f"Tensor min value: {torch.min(image_tensor)}") # Should be >= 0
        print(f"Tensor max value: {torch.max(image_tensor)}") # Should be <= 255

        # Visualize the first image
        print("\nVisualizing the first image...")
        train_dataset.visualize_image(0)

        # Example: Visualize another image if available
        if len(train_dataset) > 10:
             print("\nVisualizing image at index 10...")
             train_dataset.visualize_image(10)
    else:
        print("Dataset is empty. Check the fraction used or potential loading errors.")