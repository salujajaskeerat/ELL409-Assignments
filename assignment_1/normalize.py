import numpy as np
from numpy.lib.index_tricks import nd_grid
from numpy.linalg import norm


class norm_data:

    def __init__(self, phi: np.ndarray, method="znormal"):
        self.phi_orginal = phi
        self.phi = np.zeros_like(self.phi_orginal)
        self.method = method
        if(method == "znormal"):

            self.mean = np.mean(self.phi_orginal, axis=0)
            self.std = np.std(self.phi_orginal, axis=0)
            for i in range(0, self.phi.shape[1]):
                if(self.std[i] != 0):
                    self.phi[:, i] = (phi[:, i]-self.mean[i])/self.std[i]
                else:
                    self.phi[:, i] = self.mean[i]
        elif(method == "minmaxnormal"):
            self.min = self.min(self.phi_orginal, axis=0)
            self.max = self.max(self.phi_orginal, axis=0)
            for i in range(0, self.phi.shape[1]):
                if(self.max[i]-self.min != 0):
                    self.phi[:, i] = (phi[:, i]-self.min[i]) / \
                        (self.max[i]-self.min[i])
                else:
                    self.phi[:, i] = 1

    def unnormalize_params(self, w):
        w_n = np.zeros_like(w)
        w_n[0] = w[0]
        if(self.method == "znormal"):
            for i in range(1, len(w)):
                w_n[0] -= (w[i]*self.mean[i])/self.std[i]
                w_n[i] = w[i]/self.std[i]
        elif(self.method == "minmaxnormal"):
            for i in range(1, len(w)):
                w_n[0] -= (w[i]*self.min[i])/(self.max[i]-self.min[i])
                w_n[i] = w[i]/(self.max[i]-self.min[i])
        return w_n


def unnormalize_params(w, mean, std):
    w_n = np.zeros_like(w)
    w_n[0] = w[0]
    for i in range(1, len(w)):
        w_n[0] -= (w[i]*mean[i])/std[i]
        w_n[i] = w[i]/std[i]
    return w_n


def znormal(X: np.ndarray):

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = X.T
    mean.reshape((-1, 1)), std.reshape((-1, 1))
    for i in range(1, X.shape[0]):
        X[i] = (X[i]-mean[i])/std[i]

    return X.T, mean, std


def minmaxnormalize(X: np.ndarray):
    X = X.T

    maxx = np.array([np.max(X[i]) for i in range(0, X.shape[0])])
    minn = np.array([np.min(X[i])for i in range(0, X.shape[0])])

    maxx.reshape((-1, 1)), minn.reshape((-1, 1))
    for i in range(1, X.shape[0]):
        X[i] = (X[i]-minn[i])/(maxx[i]-minn[i])

    return X.T, minn, maxx


def relative_err(x, y):
    """ return realtive norm w.r.t x
    """
    if(norm(x) == 0):
        return norm(y)
    return norm(x-y)/norm(x)


def normalize(v: np.ndarray):
    """
    return the vector norm of v
    """
    v = v.reshape(-1, 1)
    norm = np.linalg.norm(v)
    if(norm == 0):
        return v
    return v/norm
