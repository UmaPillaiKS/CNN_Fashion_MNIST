# CNN_Fashion_MNIST

Fashion MNIST Image Classification
Overview
This project consists of Python scripts for building, training, and evaluating Convolutional Neural Network (CNN) models for image classification tasks using the Fashion MNIST dataset.

Dataset
Fashion MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. The dataset serves as a drop-in replacement for the original MNIST dataset.

Project Structure
data_preprocessing.py: This script preprocesses raw image data from the Fashion MNIST dataset stored in a folder. It converts images into NumPy arrays, performs one-hot encoding for labels, and saves the processed data into an NPZ file.
cnn_model.py: This script defines the architecture of the CNN model using Keras. The model is built with convolutional layers, pooling layers, and fully connected layers for image classification.
train_and_test.py: This script handles the training and testing of the CNN model. It loads the preprocessed data, trains the model on the training set, evaluates its performance on the test set, and visualizes the results.
Getting Started
To run this project, follow these steps:

Download the Fashion MNIST Dataset:
Download the Fashion MNIST dataset from this link.
Extract the Dataset:
Extract the downloaded dataset archive to a folder of your choice.
Preprocess the Data:
Run the data_preprocessing.py script with the path to the extracted dataset folder as an argument:
bash
Copy code
python3 data_preprocessing.py /path/to/fashion_mnist_dataset
Build, Train, and Test the CNN Model:
Run the train_and_test.py script with the path to the preprocessed NPZ file as an argument:
Copy code
python3 train_and_test.py fashion_mnist_data.npz
Dependencies
Python 3.x
NumPy
PIL (Python Imaging Library)
Keras
Note
Ensure that you have installed the required dependencies before running the scripts.