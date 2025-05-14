# mnist-cnn-colab

![MNIST Dataset](https://img.shields.io/badge/Dataset-MNIST-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Keras](https://img.shields.io/badge/API-Keras-red)
![Python](https://img.shields.io/badge/Language-Python-green)
![Google Colab](https://img.shields.io/badge/Environment-Google_Colab-yellow)

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. The model can recognize handwritten digits from 0 to 9 with an accuracy of over 99%. This implementation is designed to run in Google Colab.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Introduction](#dataset-introduction)
- [Model Architecture](#model-architecture)
- [Google Colab Setup](#google-colab-setup)
- [Usage Instructions](#usage-instructions)
- [Training Results](#training-results)
- [Future Improvements](#future-improvements)
- [References](#references)

## 📝 Project Overview

This project builds a deep learning model for recognizing handwritten digits, which is a classic case study for machine learning/deep learning beginners. Implemented using TensorFlow and Keras frameworks, it includes complete data processing, model construction, training, and evaluation workflows.

### Key Features

- CNN architecture for processing image data
- Efficient data preprocessing and transformation
- Training visualization
- Prediction result visualization
- Google Colab integration for free GPU acceleration

## 📊 Dataset Introduction

MNIST is a widely used handwritten digit dataset in machine learning, containing:

- 60,000 training images
- 10,000 test images
- Each image is a 28x28 pixel grayscale image
- 10 classes (digits 0-9)

The dataset is directly accessible through TensorFlow's Keras API, so no manual download is required.

## 🧠 Model Architecture

The CNN model consists of the following layers:

1. **Convolutional Layers**:
   - First layer: 32 3x3 filters with ReLU activation
   - Second layer: 64 3x3 filters with ReLU activation
   - Third layer: 64 3x3 filters with ReLU activation

2. **Pooling Layers**:
   - Two 2x2 max pooling layers to reduce feature map dimensions

3. **Fully Connected Layers**:
   - Dense layer with 64 neurons and ReLU activation
   - Dropout (0.5) to prevent overfitting
   - Output layer with 10 neurons and Softmax activation

## 💻 Google Colab Setup

To run this project in Google Colab:

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook or upload the existing `.ipynb` file
3. Select a GPU runtime for faster training:
   - Click `Runtime` > `Change runtime type` > Select `GPU` from the Hardware accelerator dropdown

### Check GPU Availability (Optional)

```python
import tensorflow as tf
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT available")
```

## 🚀 Usage Instructions

### Clone the Repository (Optional)

If you want to clone this repository to your Colab environment:

```python
!git clone https://github.com/yourusername/mnist-cnn.git
%cd mnist-cnn
```

### Install Dependencies

TensorFlow and other required libraries are pre-installed in Colab, but you can ensure the correct versions:

```python
!pip install -q tensorflow numpy matplotlib
```

### Run the Code

All code can be executed directly in Colab notebook cells. Make sure to run the cells in sequence to ensure proper execution.

### Save Your Work

To save your trained model or results from Colab:

```python
# Save model to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Save the model
model.save('/content/drive/My Drive/mnist_cnn_model')

# Download results or model files
from google.colab import files
files.download('model_results.png')
```

### Customize Parameters

To adjust model parameters, modify the following variables:

```python
epochs = 5  # Number of training epochs
batch_size = 32  # Batch size
```

## 📈 Training Results

Model performance on the test set:
- Accuracy: approximately 99%
- Loss value: approximately 0.03

Training process visualization:
- Accuracy and loss curves
- Comparison of prediction results with actual labels

## 🔮 Future Improvements

1. Implement data augmentation to improve model generalization
2. Add batch normalization to accelerate training
3. Use learning rate scheduler to optimize the training process
4. Try deeper or more complex network architectures
5. Save model checkpoints to Google Drive
6. Deploy model using TensorFlow.js or TensorFlow Lite

## 📚 References

- [TensorFlow Official Documentation](https://www.tensorflow.org/api_docs/python/tf)
- [MNIST Dataset Introduction](http://yann.lecun.com/exdb/mnist/)
- [Introduction to Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)
- [Google Colab Features](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

If you have any questions or suggestions, feel free to open an issue or pull request!
