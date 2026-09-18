# Brain Tumor Classification

A deep learning project for multi-class classification of brain tumor types from MRI scan images. Two convolutional architectures — a custom-built CNN and a transfer-learned MobileNetV2 — are trained, compared, and refined through fine-tuning to distinguish between four tumor categories using the Kaggle brain tumor MRI dataset.

> This README was written with the assistance of an LLM to ensure clarity, structural consistency, and precise technical language throughout.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

Brain tumor diagnosis from MRI imaging is a critical and time-sensitive task in clinical practice. Manual interpretation by radiologists is both resource-intensive and subject to human variability. This project explores the degree to which deep learning models can automate the classification of MRI scans into four categories: glioma, meningioma, no tumor, and pituitary tumor.

The pipeline covers dataset acquisition via the Kaggle API, image preprocessing and augmentation, training a custom CNN from scratch, and applying transfer learning with MobileNetV2 followed by selective fine-tuning of its deeper layers. Model outputs are evaluated using per-class classification reports and confusion matrices.

---

## Dataset

**Source:** [Kaggle — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/fahimabrarsami/brain-tumor) (downloaded via `kagglehub`)

The dataset is organized into `Training/` and `Testing/` directories, each containing four subfolders corresponding to the tumor classes. Images are in varying sizes and are resized to 224x224 pixels during the data pipeline.

**Class distribution (approximate):**

| Class | Training | Testing |
|-------|----------|---------|
| glioma | ~1,321 | ~300 |
| meningioma | ~1,339 | ~306 |
| notumor | ~1,595 | ~405 |
| pituitary | ~1,457 | ~300 |

**Total:** ~5,600 training images (4,480 train / 1,120 validation after 80-20 split), 1,600 test images

**Target classes:** `glioma`, `meningioma`, `notumor`, `pituitary`

---

## Project Structure

```
brain-tumor-classification/
├── brain_tumor.ipynb               # Main notebook: data loading, preprocessing, modeling, evaluation
├── brain_tumor_mobilenetv2.pkl     # Serialized fine-tuned MobileNetV2 model
└── README.md
```

The dataset is not stored locally — it is downloaded at runtime via `kagglehub.dataset_download("fahimabrarsami/brain-tumor")`.

---

## Methodology

### 1. Data Loading and Exploration

The dataset is loaded from the Kaggle path using `os` utilities. Class folders are enumerated to verify per-class image counts across training and test splits. Sample images from each class are displayed to visually confirm label integrity and understand image characteristics.

### 2. Preprocessing and Augmentation

- **Resize:** All images are resized to `(224, 224)` to match the input requirement of MobileNetV2 and to standardize the custom CNN's input dimensions.
- **Normalization:** Pixel values are scaled from the `[0, 255]` range to `[0.0, 1.0]` using `layers.Rescaling(1./255)`.
- **Augmentation (training only):** Random horizontal flips, rotations (up to 10%), and zoom (up to 10%) are applied during training to improve generalization and reduce overfitting.
- **Pipeline:** `tf.keras.utils.image_dataset_from_directory` is used to build efficient `tf.data` pipelines for training, validation (80-20 split from Training/), and testing.

### 3. Custom CNN

A convolutional neural network is built from scratch with three Conv2D + MaxPooling2D blocks (32, 64, 128 filters respectively), followed by a Flatten layer, a Dense layer with 128 units and 0.5 dropout, and a softmax output layer for four classes. The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss and trained for 10 epochs.

### 4. Transfer Learning with MobileNetV2

MobileNetV2 pretrained on ImageNet is loaded with `include_top=False` to use only its convolutional feature extractor. The base model weights are frozen entirely and a classification head — GlobalAveragePooling2D, Dense(128, relu), Dropout(0.5), Dense(4, softmax) — is appended and trained independently.

### 5. Fine-Tuning

After the initial transfer learning phase, the last 30 layers of the MobileNetV2 base are unfrozen and the full model is recompiled with a very low learning rate (`1e-5`) to carefully update pretrained weights without destroying learned features. The model is fine-tuned for an additional 5 epochs on the training set.

### 6. Evaluation

Both models are assessed on the held-out test set using:
- Overall test accuracy and loss
- Per-class precision, recall, and F1 score (via `classification_report`)
- Confusion matrix heatmap
- Visual inspection of misclassified images (up to 8 samples shown with actual vs. predicted labels)

---

## Models Evaluated

| # | Model | Approach | Epochs |
|---|-------|----------|--------|
| 1 | Custom CNN | Trained from scratch | 10 |
| 2 | MobileNetV2 (frozen) | Transfer learning, head only | - |
| 3 | MobileNetV2 (fine-tuned) | Last 30 layers unfrozen | 5 |

---

## Results

The fine-tuned MobileNetV2 is the best-performing model and is saved for inference. Specific accuracy values depend on the training run, but the general progression observed is:

- **Custom CNN:** Establishes a baseline; prone to underfitting on complex MRI textures given limited depth and no pretraining.
- **MobileNetV2 (frozen):** Significantly outperforms the custom CNN by leveraging ImageNet-pretrained spatial features.
- **MobileNetV2 (fine-tuned):** Achieves the highest test accuracy by adapting deeper MobileNetV2 layers to the MRI domain through careful low-rate gradient updates.

Misclassification analysis shows the most frequent confusion occurring between `glioma` and `meningioma`, which share overlapping visual morphology in MRI scans — a known challenge in the radiology literature.

> The fine-tuned MobileNetV2 model is serialized to `brain_tumor_mobilenetv2.pkl` for downstream inference.

---

## Installation

**Python 3.8 or higher is required.**

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification
pip install -r requirements.txt
```

A valid Kaggle API token (`kaggle.json`) must be present at `~/.kaggle/kaggle.json` for `kagglehub` to download the dataset automatically at runtime.

---

## Usage

Open and run the notebook sequentially:

```bash
jupyter notebook brain_tumor.ipynb
```

To load the saved model and run inference on a new MRI image:

```python
import pickle
import numpy as np
from PIL import Image

# Load model
with open("brain_tumor_mobilenetv2.pkl", "rb") as f:
    model = pickle.load(f)

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Preprocess image
image = Image.open("scan.jpg").convert("RGB").resize((224, 224))
image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Predict
predictions = model.predict(image_array)
predicted_class = class_names[np.argmax(predictions)]
print("Predicted:", predicted_class)
```

Ensure all input images are RGB, resized to `(224, 224)`, and normalized to `[0, 1]` before passing to the model.

---

## Dependencies

```
tensorflow
kagglehub
numpy
pandas
matplotlib
seaborn
Pillow
scikit-learn
```

Install all at once:

```bash
pip install tensorflow kagglehub numpy pandas matplotlib seaborn Pillow scikit-learn
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.
