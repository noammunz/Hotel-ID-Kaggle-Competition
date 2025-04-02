# Hotel Classification to Combat Human Trafficking

![Hotel Classification](https://img.shields.io/badge/Competition-Kaggle-blue)
![Deep Learning](https://img.shields.io/badge/AI-Deep%20Learning-orange)

## Overview

This repository contains code for the "Hotel-ID to Combat Human Trafficking" Kaggle competition (2021), developed as part of a university course. The project uses deep learning to classify hotel rooms from images, aiding in human trafficking investigations.

## Repository Structure

- `data_exploration.ipynb`: Exploratory data analysis of the hotel images dataset
- `download_data.ipynb`: Script to download the dataset from Kaggle
- `preprocessing.ipynb`: Data preprocessing pipeline for preparing images
- `train_vgg16.py`: Training script for the VGG16 model
- `train_resnet101.py`: Training script for the ResNet101 model
- `train_vit.py`: Training script for the Vision Transformer model
- `train_arcface.py`: Training script for the ArcFace model
- `ensemble.py`: Implementation of model ensemble combining the trained models
- `report.pdf`: Detailed project report with methodology and results

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- transformers
- timm
- pandas, numpy, matplotlib, OpenCV

### Data Download
```python
# Use Kaggle API to download data
!kaggle competitions download -c hotel-id-2021-fgvc8
!unzip -q hotel-id-2021-fgvc8.zip -d data
```

## Models

The project uses an ensemble of pretrained models:
- VGG16
- ResNet101
- Vision Transformer (ViT)
- ArcFace with ResNet101 backbone

## For More Information

Please refer to `report.pdf` for comprehensive information about:
- Dataset analysis
- Preprocessing steps
- Model architectures
- Training methodology
- Results and performance
- Explainability studies
- Future work
