import numpy as np
from itertools import product


def min_max_normalize(X: np.ndarray):
    """
      X is a feture vector , i.e  each row is a input data
      Each coloumn represent a feature

      return normalized  min and max array of the normalization
      """
    if(X.ndim == 1):
        X = np.reshape(X, (-1, 1))

    if(X.ndim != 2):
        # print("Hi")
        raise ValueError("X shaped passed is wrong")
    XX = np.zeros_like(X)
    min_X = np.array([np.min(X[:, i]) for i in range(X.shape[1])])
    max_X = np.array([np.max(X[:, i]) for i in range(X.shape[1])])

    for (i, j) in product(np.arange(X.shape[0]), np.arange(X.shape[1])):
        XX[i, j] = (X[i, j] - min_X[j])/(max_X[j]-min_X[j])

    return XX, min_X, max_X


def unnormalize_min_max(X: np.ndarray.copy, min_X: np.ndarray, max_X: np.ndarray) -> np.ndarray:
    if(X.ndim == 1):
        X = np.reshape(X, (-1, 1))
    if(X.ndim != 2 or X.shape[1] != min_X.shape[0] or X.shape[1] != max_X.shape[0]):
        raise ValueError("Incompatible types")
    XX = np.zeros_like(X)
    for (i, j) in product(np.arange(X.shape[0]), np.arange(X.shape[1])):

        XX[i, j] = (X[i, j])*(max_X[j]-min_X[j])+min_X[j]
    return XX


def tranform_min_max(X: np.ndarray, min_X: np.ndarray, max_X: np.ndarray):
    """
    Given X , The function return the transformed X on min and max_X

    """
    if(X.ndim == 1):
        X = np.reshape(X, (-1, 1))

    XX = np.zeros_like(X)
    if(X.ndim != 2 or X.shape[1] != min_X.shape[0] or X.shape[1] != max_X.shape[0]):
        raise ValueError("Incompatible types")

    for (i, j) in product(np.arange(X.shape[0]), np.arange(X.shape[1])):
        XX[i, j] = (X[i, j] - min_X[j])/(max_X[j]-min_X[j])
    return XX
