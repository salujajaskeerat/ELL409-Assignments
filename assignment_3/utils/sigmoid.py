import numpy as np


def sigmoid(z):
    """
    z is assumed to be vector and then the sigmoid is applied element wise
    """
    return 1.0/(1.0+np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z)*(1.0-sigmoid(z))
