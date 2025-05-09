# Breast Cancer Detection and Segmentation from Ultrasound Images

This repository contains a deep learning pipeline for detecting and segmenting breast cancer from ultrasound images. It was developed as a group project for the **Machine Learning and Artificial Intelligence** course at Bocconi University.

## Team Members

- Cesare Bergossi  
- Riccardo Carollo
- Emilija Milanovic
- Elia Parolari
- Giulia Pezzani

## Project Overview

The project addresses two main tasks:

1. **Binary Classification**: Classify breast ultrasound images as *benign* or *malignant*.  
2. **Semantic Segmentation**: Localize and segment the tumor region in malignant samples.

The final solution includes:
- Deep learning models for classification and segmentation
- Image preprocessing and augmentation
- A functional web app for image upload and prediction

## File Structure
```bash
project_root/
├── app.py                          # Streamlit app backend
├── app.ipynb                       # Streamlit app notebook version
├── config.py                       # Configuration settings
├── Playground_classification.ipynb         # Baseline classification notebook
├── Playground_classification_with_mask.ipynb  # Classification with ROI masks
├── Playground_segmentation.ipynb           # Segmentation pipeline
├── core/                          # Model and training logic
├── data/                          # Dataset utilities or samples
├── saves/                         # Trained model weights
├── utils/                         # Helper functions
├── sample_image_benign.png        # Example input image
├── web_app_example.png            # Screenshot of the app
├── Breast_Cancer_Detection.pdf    # Final project report
└── README.md                      # Project documentation
```

## About the Data

The dataset contains **780 grayscale ultrasound images** of breast tissue, originally collected at **Baheya Hospital** and published on [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset). Images are categorized into **normal**, **benign**, and **malignant** classes, but only the *benign* and *malignant* samples were used for classification.

Key features:
- Originally ~1100 images → reduced to 780 after preprocessing
- Images cropped to focus on tumor regions
- **Segmentation masks** are provided for benign and malignant cases, drawn manually by radiologists
- Format: 2D grayscale images of varying resolution

> **Note**: The dataset is not included in this repository. You can download it from the Kaggle source above and place it in the `data/` folder.

## Models & Techniques

### Classification
- **Model**: ResNet18 fine-tuned on ultrasound data
- **Input**: Grayscale breast ultrasound images
- **Output**: Benign / Malignant label
- **Enhancements**:
  - ROI masking with segmentation output
  - Data augmentation (flip, rotate, scale)

### Segmentation
- **Model**: U-Net trained on pixel-level masks
- **Output**: Tumor region segmentation map
- **Loss Function**: Dice loss + Binary Cross Entropy

## Web App

Built with **Streamlit**, the app allows users to:
- Upload ultrasound images  
- Run the trained classification model  
- View predictions and segmentation overlays

Run it locally:
```bash
streamlit run app.py
```

## How to Use
1.	Clone the repo
2.	Place your images in a suitable folder
3.	Launch the app or run the notebooks for evaluation
4.	Models are loaded from the saves/ directory


## Report
See Breast_Cancer_Detection.pdf for full methodology, experiments, and results.
