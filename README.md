# **CNN Models for Hackathon: Brain Tumor Classification and COVID-19 Radiography**

This repository contains Notebooks demonstrating deep learning models for two medical image classification tasks: **Brain Tumor Classification** and **COVID-19 Radiography Analysis**. The models utilize **Convolutional Neural Networks (CNNs)**, **Transfer Learning**, and **Data Augmentation** techniques to improve prediction accuracy on medical datasets.

## **Contents**

1. **brain_tumour_classification.ipynb**
   - **Task**: Classification of brain tumors from MRI images.
   - **Techniques**: 
     - Custom CNN architecture.
     - Transfer learning using models like ResNet and EfficientNet.
     - No data augmentation and data augmentation versions.
   - **Models**: 
     - Simple 3-layer CNN.
     - EfficientNet B6 using transfer learning from TensorFlow Hub.
     - ResNetV250 for feature extraction and fine-tuning.
     - Data augmentation with ImageDataGenerator.

2. **covid_radiography.ipynb**
   - **Task**: Classification of chest X-ray images to detect COVID-19.
   - **Techniques**: 
     - Custom CNN architecture.
     - Data augmentation for better generalization.
   - **Models**: 
     - 3-layer CNN with data augmentation.
     - Validation accuracy and loss tracking through TensorBoard callback.

## **Models Overview**

### **Model 1**: Simple CNN (3-layer)
- **Architecture**: 
    - Convolutional layers: 3
    - Pooling layers: 2
    - Dense layer: 1
    - Activation: ReLU and Sigmoid for output
- **Loss**: `categorical_crossentropy`
- **Optimizer**: Adam
- **Metrics**: Accuracy

### **Model 2**: EfficientNet B6 Transfer Learning
- **Source**: TensorFlow Hub
- **Layers**: 
    - EfficientNet B6 for feature extraction (non-trainable)
    - Dense output layer with softmax activation
- **Loss**: `categorical_crossentropy`
- **Optimizer**: Adam
- **Metrics**: Accuracy

### **Model 3**: ResNetV250 Transfer Learning
- **Source**: TensorFlow Hub
- **Layers**: 
    - ResNetV250 for feature extraction (non-trainable)
    - Dense output layer with softmax activation
- **Loss**: `categorical_crossentropy`
- **Optimizer**: Adam
- **Metrics**: Accuracy

### **Model 4**: Data Augmented CNN (COVID-19 Radiography)
- **Data Augmentation**: 
    - Rotation: 20%
    - Shear: 20%
    - Zoom: 20%
    - Width/Height shift: 20%
    - Horizontal flip: True
- **Architecture**: Similar to Model 1 but applied on augmented data.

## **Getting Started**

To run the notebooks in this repository, you will need to set up a Python environment with the necessary libraries.

### **Prerequisites**

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- TensorFlow Hub (for transfer learning models)

You can install the required dependencies using pip:

```bash
pip install tensorflow keras opencv-python matplotlib tensorflow-hub
```
### **Cloning the Rep**

```bash
git clone https://github.com/ifrahaha/CNN-Models---Hackathons.git
cd CNN-Models---Hackathons
```


