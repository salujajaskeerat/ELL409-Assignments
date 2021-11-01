import numpy as np
from numpy.linalg import inv


def piv_matrix(phi_x: np.ndarray, lmda=0):
    """"
    returns pseudo penrose inverse
    """
    m = phi_x.shape[1]
    # print(phi_x.shape)
    inv_m = inv(lmda*np.identity(m) + phi_x.T @ phi_x)
    return inv_m @ phi_x.T


def piv(phi_x, T, lmda=0):
    T = np.reshape(T, (-1, 1))
    w_ml = piv_matrix(phi_x, lmda)@T
    return w_ml
