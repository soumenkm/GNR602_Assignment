# Unsupervised Texture Classification using GLCM Eigenvalues and K-Means

This repository contains the implementation for an unsupervised image texture classification pipeline based on Gray Level Co-occurrence Matrix (GLCM) eigenvalues and K-Means clustering. The project allows users to upload an image, configure parameters, and visualize the texture segmentation results through a Streamlit web application.

## Abstract

Texture analysis is crucial for image understanding. This project implements a methodology for unsupervised segmentation based on local texture features. Input images are rescaled to a reduced number of gray levels ($G \in \{5, 6, 7\}$). GLCMs are computed within a sliding window ($W_s \times W_s$) using specified distance/angle parameters. The $G$ eigenvalues derived from each normalized GLCM serve as texture descriptors. These feature vectors are then clustered using a K-Means algorithm (implemented from scratch) into $K$ distinct texture classes. The final output is a classification map highlighting regions with similar textural characteristics.

## Features

*   **Image Preprocessing:** Converts input RGB images to grayscale and rescales intensity to $G$ levels.
*   **GLCM Computation:** Calculates $G \times G$ GLCMs using a sliding window approach. Includes both naive recalculation and efficient vectorized (`unfold`-based) methods.
*   **Eigenvalue Feature Extraction:** Computes the $G$ eigenvalues from each normalized GLCM and sorts them to form feature vectors.
*   **K-Means Clustering:** Implements K-Means from scratch to cluster pixels based on their eigenvalue features into $K$ classes.
*   **Visualization:** Generates visualizations of:
    *   Original RGB image
    *   Derived Grayscale image (rescaled)
    *   Eigenvalue maps (texture images)
    *   K-Means classification map (optionally alongside cropped original)
*   **Streamlit GUI:** Provides an interactive web interface for uploading images, setting parameters, running the analysis, and viewing results.

## File Structure
```
.
├── .git/ # Git directory
├── code/ # Main source code
│ ├── app.py # Streamlit GUI application script
│ ├── dataset.py # Dataset loading class (if used, e.g., for initial testing)
│ ├── glcm.py # Class implementing GLCM computation and eigenvalue extraction
│ └── kmeans.py # Class implementing K-Means clustering from scratch
├── output/ # Directory for saved outputs (created automatically)
│ ├── classification/ # Saved K-Means classification plots
│ ├── glcms/ # Saved GLCM/Eigenvalue plots (if implemented)
│ └── images/ # Saved input/comparison images (if implemented)
├── .gitignore # Files/directories ignored by Git
└── readme.md # This README file
```

*   **`code/app.py`**: The main script to run the Streamlit web application.
*   **`code/glcm.py`**: Contains the `EfficientGLCM` class responsible for GLCM calculations and eigenvalue feature extraction.
*   **`code/kmeans.py`**: Contains the `KMeansTextureClassifier` class for performing K-Means clustering.
*   **`code/dataset.py`**: (Optional) May contain dataset handling code, like the `MillionAIDataset` class used during development/testing. The Streamlit app loads images directly via upload.
*   **`output/`**: Automatically generated directory where visualization plots are saved by default.

## Requirements

*   Python 3.8+
*   Streamlit
*   PyTorch
*   NumPy
*   Matplotlib
*   Pillow (PIL Fork)
*   tqdm (for progress bars in non-GUI scripts)
*   Torchvision (primarily for `transforms.PILToTensor`)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install streamlit torch numpy matplotlib Pillow tqdm torchvision
    ```
 
## Usage

1.  **Navigate to the code directory:**
    ```bash
    cd code
    ```

2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

3.  **Interact with the GUI:**
    *   A new tab should open in your web browser displaying the application.
    *   Use the sidebar to **upload** a PNG or JPEG image.
    *   Adjust the **GLCM parameters** (Gray Levels G, Window Size, Distance) and **K-Means parameters** (Number of Clusters K, Max Iterations, Tolerance) as needed. Window size must be odd.
    *   Click the "**Process Image**" button.
    *   Wait for the processing steps (GLCM calculation, eigenvalue extraction, K-Means) to complete. Status messages and timings will be displayed.
    *   View the generated visualizations in the main area of the application:
        *   Input Images & Eigenvalue Maps
        *   K-Means Classification Result (potentially alongside the cropped original image)

## Output

*   **Interactive Visualization:** The primary output is displayed directly within the Streamlit application interface.
*   **Saved Figures:** By default, the visualization functions within the classes might save figures to the `output/` subdirectories (e.g., `output/classification/`). Check the specific visualization method implementations in `glcm.py` and `kmeans.py` for details on saving behavior. The Streamlit app itself primarily focuses on displaying the figures interactively using `st.pyplot()`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.