import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as inline

def preprocess_data(X, Y):
    """
    function that pre-processes the CIFAR10 dataset as per
    densenet model requirements for input images
    labels are one-hot encoded
    """
    X = K.applications.densenet.preprocess_input(X)
    Y = K.utils.to_categorical(Y)
    return X, Y

#Load the Cifar10 dataset, 50,000 training images and 10,000 test images (here used as validation data)

(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

#preprocess the data using the application's preprocess_input method and
#convert the labels to one-hot encodings

x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)



