import numpy as np
from numpy.linalg import norm
from normalize import *
from utils import *


def gradient(x, w, t, lmda=0):
    x = np.reshape(x, (-1, 1))
    w = np.reshape(w, (-1, 1))
    return ((x.T @ w)-t) * x + lmda*w


def grad_descent(X: np.array, T: np.array,
                 max_iter=1000, steps=1e-2,
                 batch_size=10,
                 lmda=0, w_ini=-1, error_gaps=10):
    """"
    Batch gradient descent :
    At each iteration partition data set into n/k batches on size k each
    Now for each batch ,cacluate the grad_descent and update the w
    """
    errors = []
    n, m = X.shape
    if(type(w_ini) == np.ndarray and len(w_ini) == m):
        w = np.reshape(w_ini, (m, 1))

    else:
        w = np.zeros(shape=(m, 1), dtype=np.float64)
    # err = sq_err(X, T, w, lmda)
    indx = np.arange(0, n)
    dw = np.zeros((m, 1))
    for it in range(0, max_iter):
        batches: np.ndarray = random_split(indx, batch_size)
        for batch in batches:
            for x, t in zip(X[batch], T[batch]):
                dw += gradient(x, w, t, lmda)
            w -= (steps*(dw))
            dw = np.zeros_like(dw)
        if(it % error_gaps == 0 or it == max_iter-1):
            errors.append((it, sq_err(X, T, w, lmda)))

    return w, errors


def moment_grad_descent(X: np.array, T: np.array,
                        max_iter=1000, steps=1e-2,
                        batch_size=10,
                        lmda=0, gmma=0.9,  w_ini=-1,
                        error_gaps=100):
    """
    Apply moment based grad_descent i.e
    update_i = update_i-1 * gmma  + steps*grad
    w(i) = w(i-1) - update(i-1)

    v_w :velocity of w
    """
    errors = []
    n, m = X.shape
    if(type(w_ini) == np.ndarray and len(w_ini) == m):
        # print("here")
        w = np.reshape(w_ini, (m, 1))

    else:

        w = np.zeros(shape=(m, 1), dtype=np.float64)
    # print(w)
    # err = sq_err(X, T, w, lmda)
    # errors.append(sq_err(X, T, w, lmda))
    indx = np.arange(0, n)

    dw, v_w, prev_v_w = np.zeros((m, 1)), np.zeros((m, 1)), np.zeros((m, 1))
    for it in range(max_iter):
        batches: np.ndarray = random_split(indx, batch_size)
        for batch in batches:
            for x, t in zip(X[batch], T[batch]):
                dw += gradient(x, w, t, lmda)
            v_w = gmma*prev_v_w + steps*dw
            w -= v_w
            dw = np.zeros_like(dw)
            prev_v_w = v_w
        # err = sq_err(X, T, w, lmda)
        if(it % error_gaps == 0 or it == max_iter-1):
            errors.append((it, sq_err(X, T, w, lmda)))
        # print("Iteration:  ", it, "err =", err)

    return w, errors
