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
        return np.maximum(z, 0)

    @staticmethod
    def f_prime(z: np.ndarray):
        result = np.where(z < 0.0, 0.0, z)
        result = np.where(result > 0.0, 1.0, result)
        return result


alpha = 0


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
