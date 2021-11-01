import numpy as np

from polyfit import *


def kfold_cross_validation(X, T, degree, K, max_iter=3000,   steps=1e-3, method='mgrad', lmda=0):
    """
    Fits X ,T using polyfit function
    K: K is a hyper hyper paramters 
    Return cross validation error on fittiing hyperparameters provided
    handle cases when PIV overflow Just not include that
    """
    X, T = np.ravel(X), np.ravel(T)
    splits = np.array_split(np.arange(0, len(X)), K)
    mse_train = 0.0
    mse_test = 0.0
    i = 0
    # print("Hyper paramater config : degree =", degree)
    for i in range(0, K):
        # print("Traning ", i+1, "th fold")
        # print("Training indices", X[np.ravel(np.delete(splits, i, axis=0))])
        # print("Testing indices", X[np.ravel(splits[i])])
        model = polyfit(degree=degree, max_iter=max_iter,
                        steps=steps, method=method, lmda=lmda)
        X_train, Y_train = X[np.hstack(np.delete(splits, i, axis=0))], T[np.hstack(
            np.delete(splits, i, axis=0))]

        X_test, Y_test = X[np.ravel(splits[i])], T[np.ravel(splits[i])]
        model.fit(X_train, Y_train)
        mse_test += np.sqrt(model.score(X_test, Y_test))/len(X_test)
        mse_train += np.sqrt(model.train_error)/len(X_train)
        # print("Error", score)
    # mse_test=np.sqrt(cross_val_err)/K
    # print((mse_train/K, mse_test/K))
    return (mse_train/K, mse_test/K)


def shuffled_cvr(X, T, degree, K=10, max_iter=2000, steps=1e-3, method='piv', n_permutations=10, lmda=0):
    cross_val_err = 0
    for i in range(n_permutations):
        indices = np.arange(0, X.shape[0])
        np.random.shuffle(indices)
        cross_val_err += kfold_cross_validation(X[indices],
                                                T[indices], degree=degree, K=K,
                                                max_iter=max_iter, steps=steps, method=method, lmda=lmda)
    return cross_val_err/n_permutations


def shuffled_cvr(X, T, degree, K=10, max_iter=2000, method='piv', n_permuatations=20, lmda=0, steps=1e-3):
    train_err_t, test_err_t = 0.0, 0.0

    for i in range(n_permuatations):
        indices = np.arange(0, X.shape[0])
        np.random.shuffle(indices)
        tr, ts = kfold_cross_validation(
            X[indices], T[indices], degree=degree, K=K, max_iter=max_iter, steps=steps, method=method, lmda=lmda)
        train_err_t += tr
        test_err_t += ts

    return (train_err_t/n_permuatations, test_err_t/n_permuatations)
