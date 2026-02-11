# 3D Brain Tumor Segmentation (U-Net)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange?logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Dice Score](https://img.shields.io/badge/Best_Dice_Score-0.96-brightgreen)

## Project Overview
This project implements a **3D U-Net** architecture to perform volumetric segmentation of brain tumors from multimodal MRI scans (BraTS 2020 dataset). The model identifies three specific tumor sub-regions, crucial for clinical diagnosis and treatment planning:
* **Necrotic Core**
* **Peritumoral Edema**
* **Enhancing Tumor**

## Project Structure
```text
├── notebooks/
│   ├── 01_data_exploration.ipynb  # MRI visualization & intensity distribution
│   ├── 02_preprocessing.ipynb     # Pipeline execution (Crop -> Normalize -> Save)
│   └── 03_training.ipynb          # Model training & evaluation loop
├── src/
│   ├── architectures/
│   │   ├── blocks.py        # Encoder/Decoder building blocks
│   │   └── unet_3d.py       # Full Model assembly
│   ├── training/
│   │   └── losses.py        # Custom Combined Loss (Dice + Categorical Cross-Entropy)
│   ├── data_loader.py       # BraTSLoader class for NIfTI handling
│   ├── generator.py         # BraTSGenerator for batch data loading
│   ├── preprocessing.py     # Z-score normalization & background removal logic
│   └── visualization.py     # Plotly 3D interactive rendering
├── models/              # Saved model weights and training logs
├── results/             # Output visualizations and predictions
└── README.md
```

## Data Pipeline
The project processes multimodal MRI scans (T1, T1ce, T2, FLAIR) using a custom pipeline designed for high-dimensional medical data.

1.  **Preprocessing & Volume Handling:**
    * **NIfTI Loading:** Utilizes `nibabel` to load 4D volumetric data (240x240x155x4).
    * **Cropping:** Automated extraction of the brain region, reducing volume to **128x128x128** voxels. This optimizes GPU memory usage while preserving the Region of Interest (ROI).
    * **Normalization:** Applied Z-Score normalization per channel to handle intensity variations between patients.

2.  **Efficient 3D Data Loading:**
    * Engineered a custom `BraTSGenerator` (inheriting from `tf.keras.utils.Sequence`) to handle memory-intensive 3D MRI volumes.
    * **Dynamic Processing:** Loads batches on-the-fly, enabling training on consumer GPUs without OOM errors.
    * **One-Hot Encoding:** Converts segmentation masks into 4-channel categorical arrays.

## Architecture & Technical Approach

The model is based on the **3D U-Net** architecture ([Çiçek et al., 2016](https://arxiv.org/abs/1606.06650)), a volumetric extension of the original U-Net ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)), optimized for multimodal brain tumor segmentation.

![3D U-Net Architecture](results/diagram.png)

### Model Structure
* **Encoder (Contraction Path):** 4 hierarchical blocks using `Conv3D` -> `BatchNorm` -> `ReLU`. Downsampling via `MaxPool3D` captures semantic context.
* **Bridge (Bottleneck):** Connects encoder and decoder, processing the most abstract features.
* **Decoder (Expansion Path):** 4 upsampling blocks using `Conv3DTranspose`. Features are concatenated with skip connections from the encoder to recover spatial details.
* **Output Layer:** A `1x1x1 Conv3D` layer with **Softmax** activation maps features to 4 mutually exclusive classes.

### The "Class Imbalance" Challenge
Medical imaging datasets are heavily dominated by background voxels (98%+), leading to model bias. To address this, I engineered a custom **Combined Loss Function**:

$$\mathcal{L}_{combined} = \mathcal{L}_{Dice} + \mathcal{L}_{CCE}$$

1.  **Dice Loss** - Optimizes the overlap of the tumor shape (insensitive to background size):

$$\mathcal{L}_{Dice} = 1 - \frac{2 \sum_{i} p_{i} \cdot g_{i} + \epsilon}{\sum_{i} p_{i} + \sum_{i} g_{i} + \epsilon}$$

2.  **Categorical Cross-Entropy** - Heavily penalizes false positive classifications in the background.

## Key Results
Evaluation performed on the independent test set. The model achieves state-of-the-art performance in detecting the Whole Tumor region.

| Region | Description | Mean Dice Score | Best Case |
| :--- | :--- | :--- | :--- |
| **Whole Tumor (WT)** | Visible Edema + Core + Enhancing | **0.93** | **0.98** |
| **Tumor Core (TC)** | Necrotic Core + Enhancing Tumor | **0.90** | **0.98** |
| **Enhancing (ET)** | Active Enhancing Tumor (Class 3) | **0.79** | **1.00** |

## 🖼Visualizations

### 1. Volumetric Fly-through
![Tumor Segmentation GIF](results/tumor_scan.gif)

### 2. Best Case Prediction (Dice: 0.98)
![Best Prediction](results/best_patient.png)
> *Cyan = Necrotic Core, Yellow = Edema, Red = Enhancing Tumor*

## 🚀 Installation & Usage

### Prerequisites
* Python 3.12+
* TensorFlow 2.16+
* Nibabel, NumPy, Plotly, Scikit-image

### Setup
```bash
# Clone the repository
git clone https://github.com/KacSzy/3d-brain-tumor-segmentation.git
cd 3d-brain-tumor-segmentation

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline
1.  **Data Exploration:** Run `notebooks/01_data_exploration.ipynb` to visualize raw MRI scans and check class balance.
2.  **Preprocessing:** Run `notebooks/02_preprocessing.ipynb`. This will load raw data from `data/01_raw`, apply cropping/normalization, and save `.npy` files to `data/02_processed`.
3.  **Training:** Run `notebooks/03_training.ipynb` to start the training loop.

## References
1.  Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
2.  Çiçek, Ö. et al. (2016). *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.* [arXiv:1606.06650](https://arxiv.org/abs/1606.06650)
3.  **Dataset:** [BraTS 2020 Training & Validation](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
