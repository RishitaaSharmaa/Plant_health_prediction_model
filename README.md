# Plant Health Prediction Model

## Overview
The **Plant Health Prediction Model** is a deep learning-based system that utilizes **Convolutional Neural Networks (CNNs)** to classify plant health conditions. The model analyzes leaf images to detect and diagnose diseases, ensuring timely intervention for better crop yield and plant care.

## Features
- **Deep Learning-based Classification**: Uses CNN to classify healthy and diseased plant leaves.
- **Multiple Disease Detection**: Supports multiple plant diseases, depending on the dataset used.
- **User-Friendly**: Designed for ease of use with an intuitive interface.
- **Scalable**: Can be trained on different datasets to include more plant species and diseases.

## Dataset
The model is trained using *PlantVillage Dataset* a dataset consisting of labeled images of plant leaves.

## Technologies Used
- **Python**
- **TensorFlow/Keras**
- **Pillow** (for image handling)
- **Streamlit** (for web-based UI)
- **NumPy**

## Model Architecture
The CNN architecture consists of:
1. **Convolutional Layers**: Extract spatial features from input images.
2. **Pooling Layers**: Reduce spatial dimensions and computation.
3. **Fully Connected Layers**: Classify features into respective categories.
4. **Activation Functions**: ReLU for hidden layers, Softmax for output.
5. **Optimization Algorithm**: Adam optimizer for efficient learning.

## Installation & Usage
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow/Keras
- Pillow
- Streamlit
- NumPy

### Web Deployment
Run the Streamlit application:
```sh
streamlit run app.py
```
## Future Enhancements
- Improve dataset diversity for better generalization.
- Implement real-time prediction.
- Enhance interpretability using Grad-CAM visualization.
















