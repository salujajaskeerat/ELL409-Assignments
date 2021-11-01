import numpy as np
from itertools import product


def linear_kernel(x1, x2):
    # x1,x2 are row vector: 1*m vector
    return x1@x2.T


def poly_kernel(x1: np.ndarray, x2: np.ndarray, degree=3, r=1):

    return (x1@x2.T+r)**degree


def gaussian_kernel(x1: np.ndarray, x2: np.ndarray, gamma=None):
    """"
    x1: m*1 vector
    Default value of gamma = 1/number of features in the data
    """
    if(gamma is None):
        gamma = 1/x1.shape[0]

    if x1.ndim == 1 and x2.ndim == 1:
        return np.exp(((- np.linalg.norm(x1-x2) ** 2)*gamma))
    elif (x1.ndim > 1 and x2.ndim == 1) or (x1.ndim == 1 and x2.ndim > 1):
        return np.exp((-np.linalg.norm(x1-x2, axis=1)**2)*gamma)
    elif x1.ndim > 1 and x2.ndim > 1:
        return np.exp((-np.linalg.norm(x1[:, np.newaxis]-x2[np.newaxis, :], order=2, axis=2)**2)*gamma)


def kernel(x1: np.ndarray, x2: np.ndarray, kernel_type=1, degree=3, gamma=None, r=1):
    if(kernel_type == 2):
        return poly_kernel(x1, x2, degree=degree, r=r)
    elif (kernel_type == 3):
        return gaussian_kernel(x1, x2, gamma=gamma)
    else:
        return linear_kernel(x1, x2)


def linear_kernel_matrix(X: np.ndarray):
    return X @ X.T


def poly_kernel_matrix(X: np.ndarray, degree=3, r=1):
    """" Computes (Xi dot Xj + r)^degree """
    m = X.shape[0]
    XX = (X@X.T)
    for (i, j) in product(range(m), range(m)):
        XX[i][j] = (XX[i][j]+r) ** degree

    return XX


def gaussian_kernel_matrix(X: np.ndarray, gamma=None):
    """"
    Default value of gamma = 1/number of features in the data
    """
    if(gamma is None):
        gamma = 1/X.shape[1]
    XX = np.zeros(shape=(X.shape[0], X.shape[0]))

    for (i, j) in product(range(X.shape[0]), range(X.shape[0])):
        XX[i][j] = np.exp((- np.linalg.norm(X[i] - X[j]) ** 2)*gamma)

    return XX
