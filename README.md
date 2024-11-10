# Convolutional-Neural-Network
This project is an advanced image classification model built with TensorFlow, demonstrating the use of a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The model achieves high accuracy through a layered CNN architecture, which is trained, validated, and evaluated on CIFAR-10 images.

## Table of Content
- Overview
- Dataset
- Model Architecture
- Installation
- Usage
- Results
- Contribution
- License

## Overview
The CIFAR-10 dataset is a popular dataset in computer vision, consisting of 60,000 32x32 color images in 10 different classes. This project builds a CNN model with TensorFlow to classify these images into their respective categories with high accuracy.

The following features are implemented:
- Loading and preprocessing CIFAR-10 images.
- Constructing a CNN model with multiple convolutional and pooling layers.
- Training and validating the model.
- Visualizing model performance.
- Displaying sample predictions with correct and incorrect classifications.

## Dataset
The CIFAR-10 dataset consists of:

- 50,000 training images
- 10,000 test images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Each image is 32x32 pixels and in RGB format. This project downloads the dataset automatically using TensorFlow.

## Model Architecture
The CNN model architecture used in this project includes:

1. Convolutional Layers - 3 layers with increasing filters to extract complex features.
2. Pooling Layers - Max pooling layers to reduce the dimensionality.
3. Flatten Layer - Flattens the input for the dense layer.
4. Dense Layers - Fully connected layers for classification.

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.

## Model Summary

|  Layer        | Output Shape          | Parameters   |
|  ------------ |  -------------------- |  ----------- |
|  Conv2D       |  (None, 30, 30, 32)   |  896         |
|  MaxPooling2D |  (None, 15, 15, 32)   |  0           |
|  Conv2D       |  (None, 13, 13, 64)	 |  18496       |
|  MaxPooling2D |  (None, 6, 6, 64)     |  0           |
|  Conv2D	    |  (None, 4, 4, 64)	    |  36928       |
|  Dense        |  (None, 64)	          |  	65600     |
|  Dense        |  (None, 10)           |  	650       |

## Installation

1. Clone the repository:
   
       git clone https://github.com/kunal-1207/Convolutional-Neural-Network.git
       cd Convolutional-Neural-Network
2. Install the required packages:
   
       pip install tensorflow matplotlib

## Usage
Run the main program to train and evaluate the model:

    python CNN.py


### Visualizing Sample Images
The program will display 16 sample images from the dataset to help users familiarize themselves with the classes.

### Training the Model
The CNN model will train on 50,000 images from the CIFAR-10 dataset and validate on 10,000 test images.

### Model Evaluation and Visualization
After training, the model's accuracy and loss will be displayed, and a graph of training vs. validation accuracy will be generated. Predictions on test images will also be displayed, with correct and incorrect classifications marked.

## Results
After training, the model typically achieves an accuracy of around 75-80% on the test set, though results may vary. The following metrics are provided in the program output:

- Training Accuracy
- Validation Accuracy
- Test Accuracy

### Sample Output Plot
A sample accuracy plot will display the training and validation accuracy trends over epochs.

## Contribution 
Contributions are welcome! Please open an issue or submit a pull request if you would like to contribute.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.







