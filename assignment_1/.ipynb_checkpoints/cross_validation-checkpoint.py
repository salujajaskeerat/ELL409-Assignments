import numpy as np
from numpy.lib.shape_base import split
from polyfit import *


def kfold_cross_validation(X, T, degree, K=10, max_iter=2000,   steps=1e-3, method='mgrad'):
    """
    Fits X ,T using polyfit function
    K: K is a hyper hyper paramters 
    Return cross validation error on fittiing hyperparameters provided
    """

    splits = np.array_split(np.arange(0, len(X)), K)
    cross_val_err = 0

    # print("Hyper paramater config : degree =", degree)
    for i in range(0, K):
        # print("Traning ", i+1, "th fold")
        # print("Training indices", np.ravel(np.delete(splits, i, axis=0)))
        # print("Testing indices", np.ravel(splits[i]))
        model = polyfit(degree, max_iter=max_iter, steps=steps, method=method)
        model.fit(X[np.ravel(np.delete(splits, i, axis=0))],
                  T[np.ravel(np.delete(splits, i, axis=0))])
        score = model.score(X[np.ravel(splits[i])], T[np.ravel(splits[i])])
        # print("Error", score)
        cross_val_err += score
    return (cross_val_err/K)


def shuffled_cvr(X, T, degree, K=10, max_iter=2000, steps=1e-3, method='piv', n_permutations=10):
    cross_val_err = 0
    for i in range(n_permutations):
        indices = np.arange(0, X.shape[0])
        np.random.shuffle(indices)
        cross_val_err += kfold_cross_validation(X[indices],
                                                T[indices], degree=degree, K=K,
                                                max_iter=max_iter, steps=steps, method=method)
    return cross_val_err/n_permutations
