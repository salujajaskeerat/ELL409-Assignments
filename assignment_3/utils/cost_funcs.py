import numpy as np
from neuron import sigmoid_neuron


class cross_entropy_cost:
    @staticmethod
    def cost(activation_out, y):
        """
        activation_out : y_pred 

        computes : `y_true ln(activation_out)  + (1-y_true) (ln(1-activation_out))`
        """
        return np.sum(np.nan_to_num(-y * np.log(activation_out) - (1-y)*np.log(1-activation_out)))

    @staticmethod
    def delta(z_out, activation_out, y, f_prime):
        """
        Computes the delta for the last layer , i.e for last activation layer 

        Input required are z_out , activation_out and ytrue
        """
        return (activation_out - y)


class quadratic_cost:
    @staticmethod
    def cost(activation_out, y):
        """
        Computes `0.5 * |activation_out-y|^2`
        """
        return 0.5 * (np.linalg.norm(activation_out-y)**2)

    def delta(z_out, activation_out, y, f_prime):
        """
        Computes the delta for the last layer w.r.t to the quadratic loss function

        """
        return (activation_out-y) * f_prime(z_out)
