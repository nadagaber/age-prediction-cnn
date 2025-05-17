# age-prediction-cnn
Age Prediction from Facial Images using Transfer Learning (Xception, ResNet50) and Custom CNN Models on UTKFace Dataset

# Age Prediction from Facial Images

This project implements an age prediction system using deep learning on the UTKFace dataset. It leverages transfer learning with pre-trained models (Xception, ResNet50) and custom convolutional neural networks (CNNs) to predict human age from facial images.

## Features

- Data preprocessing and augmentation using TensorFlow's ImageDataGenerator
- Age prediction as a regression problem (predicting continuous age values)
- Transfer learning models:
  - Xception (with frozen base layers)
  - ResNet50 (with frozen base layers)
- Custom CNN architectures for comparison
- Model training with early stopping and learning rate scheduling
- Performance evaluation with Mean Absolute Error (MAE) and loss plots
- Visualization of predictions on sample images

## Dataset

- [UTKFace Dataset](https://susanqq.github.io/UTKFace/) — a large-scale face dataset with age, gender, and ethnicity annotations.
- The dataset is expected to be provided as a ZIP archive (`archive.zip`).
- Images are filtered for ages between 10 and 90.

## Usage

### Setup

1. Clone the repo and upload your dataset ZIP file (e.g., `archive.zip`) to your working directory (Google Colab recommended).
2. Mount Google Drive if using Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
