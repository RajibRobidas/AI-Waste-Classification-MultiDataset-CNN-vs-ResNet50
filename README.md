# ♻️ AI-Driven Waste Classification

### 🧠 Comparative Analysis of Deep Learning Models (Custom CNN & ResNet-50) on Multiple Image Datasets for Smart Waste Management

---

## 📌 Overview

This project presents a **comparative study** of two deep learning models — a **Custom Convolutional Neural Network (CNN)** and a **ResNet-50** (via transfer learning) — for image-based **waste classification**. The analysis spans across **four diverse benchmark datasets**:
- 🗂 TrashNet (GitHub)
- 📦 RealWaste (UCI)
- 🧃 Boarkee (GitHub)
- 🛢 Kaggle Waste Classification (binary)

The complete pipeline is implemented in **PyTorch**, with robust evaluation through:
- Stratified train/validation/test splits
- Advanced data augmentation
- Early stopping
- Per-class accuracy
- Confusion matrix visualization
- Cross-dataset benchmarking

---

## 🧪 Methodology

### 🔁 Pipeline Components

| Step                    | Tool/Library         | Purpose                                     |
|-------------------------|----------------------|---------------------------------------------|
| Data Loading            | `PyTorch`, `PIL`     | Load datasets from structured directories   |
| Data Augmentation       | `torchvision.transforms` | Resize, flip, color jitter, normalization   |
| Stratified Splitting    | `scikit-learn`       | Balanced train/val/test sets                |
| Model Architectures     | `PyTorch`, `torchvision.models` | Custom CNN, ResNet-50 w/ ImageNet weights |
| Training & Early Stop   | `PyTorch`            | Adam optimizer, weight decay, early stop    |
| Evaluation & Visualization | `matplotlib`, `seaborn`, `sklearn` | Confusion matrix, per-class accuracy     |
| Device Management       | `PyTorch`            | GPU acceleration (Tesla T4 / RTX 3050)      |

---

## 📂 Datasets Overview

| Dataset        | Classes | Total Images | Train/Val/Test | Source        |
|----------------|---------|--------------|----------------|---------------|
| TrashNet       | 6       | 2,527        | 1,768 / 253 / 506 | GitHub     |
| RealWaste      | 9       | 4,752        | 3,326 / 475 / 951 | UCI        |
| Boarkee        | 5       | 377          | 263 / 38 / 76   | GitHub       |
| Kaggle Waste   | 2       | 25,077       | 17,553 / 2,508 / 5,016 | Kaggle |

---

## 🏗 Model Architectures

### 1. **Custom CNN**
- Lightweight model for baseline comparison
- Includes Conv2D, BatchNorm, ReLU, MaxPool, Dropout, Dense layers

### 2. **ResNet-50 (Transfer Learning)**
- Pretrained on ImageNet
- Final FC layer replaced for each dataset
- Gradual unfreezing + adaptive learning rate

---

## 📊 Visualizations

- Accuracy vs. Epochs
- Loss vs. Epochs
- Confusion Matrices
- Per-Class Accuracy Plots
- Side-by-side model comparisons across datasets

📸 *All visualizations available inside the Jupyter Notebooks.*

---

## ⚙️ Training Setup

- Framework: PyTorch
- Epochs: Max 50 (early stopping applied)
- Optimizer: Adam (w/ weight decay for ResNet-50)
- Hardware: NVIDIA Tesla T4 / RTX 3050 (GPU)

---

## 📁 Datasets & Trained Models

All **four datasets** and the corresponding **trained model weights** (8 models total: 4 for CNN, 4 for ResNet-50) are available here:

🔗 [Google Drive – Datasets & Trained Models](https://drive.google.com/drive/folders/1UbNcEQ_avXihHdQERskTMr7E7TeG_8w-?usp=sharing)

> Download and extract the folders to the appropriate dataset and model directories before running the notebooks.

---

## 📦 How to Run

1. **Clone this repository**:
   ```bash
   git clone https://github.com/<your-username>/AI-Waste-Classification-MultiDataset-CNN-vs-ResNet50.git
   cd AI-Waste-Classification-MultiDataset-CNN-vs-ResNet50
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. Run training notebooks:
    ```bash
    python -m notebook
4. (Optional) Use GPU:
    - Ensure CUDA is available for training on GPU.

---



