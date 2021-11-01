import numpy as np
from numpy.linalg import norm
from normalize import *


def sq_err(X, T, w, lmda):
    """
    X : n*m matrix
    W : m*1 matrix
    Y =  XW : n*1 matrix
    Squared_ error = np.sum((Y-t)**2) : n*1 matrix
    """
    T = np.reshape(T, (-1, 1))
    w = np.reshape(w, (-1, 1))
    Y = X@w
    sq_error = np.sum(np.square(Y-T))
    return sq_error + lmda*norm(np.ravel(w))


def abs_err(X, T, w, lmda):
    """
    X : n*m matrix
    W : m*1 matrix
    Y =  XW : n*1 matrix
    Squared_ error = np.sum((Y-t)**2) : n*1 matrix
    """
    T = np.reshape(T, (-1, 1))
    w = np.reshape(w, (-1, 1))
    Y = X@w
    absrr = np.sum(np.abs(Y-T))
    return absrr + lmda*norm(np.ravel(w))


def random_split(idx: np.array, batch_size=10):
    """
    Input : index array idx
    batch-size : size of batch
    return indices for random split of X in parts of k size each
    """
    n = idx.shape[0]
    np.random.shuffle(idx)
    batches = np.array_split(idx, [int(n/batch_size)])
    return batches
