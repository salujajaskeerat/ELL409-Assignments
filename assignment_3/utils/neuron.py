import numpy as np


class sigmoid_neuron:
    @staticmethod
    def f(z):
        """
        z is assumed to be vector and then the sigmoid is applied element wise
        """
        return 1.0/(1.0+np.exp(-z))

    @staticmethod
    def f_prime(z):
        return sigmoid_neuron.f(z)*(1.0-sigmoid_neuron.f(z))


class tanh_neuron:
    @staticmethod
    def f(z: np.ndarray):
        return np.tanh(z)

    @staticmethod
    def f_prime(z: np.ndarray):

        return 1.0 - np.tanh(z)**2


class soft_plus:
    @staticmethod
    def f(z: np.ndarray):
        return np.log(1 + np.exp(z))

    def f_prime(z: np.ndarray):
        return np.exp(z)/(1+np.exp(z))


alpha = 1e-3


class relu_neuron:

    @staticmethod
    def f(z: np.ndarray):
        result = np.where(z < 0.0, alpha*z, z)
        return result

    @staticmethod
    def f_prime(z: np.ndarray):
        result = np.where(z < 0.0, alpha, z)
        result = np.where(result > 0.0, 1.0, result)
        return result
