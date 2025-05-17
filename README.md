# Skin Lesion Classification and Segmentation

This repository contains two Jupyter notebooks that implement deep learning models for skin lesion analysis:

1. `skin lesion task-1.ipynb`: Focuses on skin lesion segmentation using U-Net architecture
2. `skin lesion task-3.ipynb`: Implements skin lesion classification using EfficientNet

## Project Overview

This project aims to develop deep learning models for automated analysis of skin lesions, with two main tasks:

### Task 1: Skin Lesion Segmentation
- Implements a U-Net based model for segmenting skin lesions from images
- Uses TensorFlow/Keras for model implementation
- Includes data preprocessing and augmentation
- Implements training with validation and model evaluation

### Task 3: Skin Lesion Classification
- Uses EfficientNetB0 as the base model for classification
- Implements transfer learning approach
- Includes comprehensive data augmentation
- Uses various callbacks for model training optimization

## Requirements

The project requires the following Python packages:
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PIL (Python Imaging Library)

## Usage

1. Clone the repository
2. Install required dependencies
3. Run the notebooks in order:
   - First run `skin lesion task-1.ipynb` for segmentation
   - Then run `skin lesion task-3.ipynb` for classification

## Model Details

### Segmentation Model (Task 1)
- Architecture: U-Net
- Input: Skin lesion images
- Output: Binary segmentation masks
- Training: Uses binary cross-entropy loss
- Evaluation: Uses IoU (Intersection over Union) metric

### Classification Model (Task 3)
- Base Model: EfficientNetB0
- Input: Skin lesion images
- Output: Classification predictions
- Training: Uses categorical cross-entropy loss
- Evaluation: Uses accuracy, precision, recall metrics

## Data

The project uses skin lesion images for both segmentation and classification tasks. The data should be organized in appropriate directories for training and validation.

## Results

The models achieve good performance on both segmentation and classification tasks. Detailed results and visualizations are available in the respective notebooks.

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 