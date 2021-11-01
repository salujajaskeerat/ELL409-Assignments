import numpy as np
from gradient_descent import *
from normalize import *
from pseudo_inv import *


class polyfit:
    def __init__(self, degree, lmda=0, method='mgrad',
                 batch_size=1, steps=1e-3,
                 max_iter: np.int16 = 5000):
        self.lmda = lmda
        self.method = method
        self.batch_size = batch_size
        self.steps = steps
        self.max_iter = max_iter
        self.degree = degree

    def fit(self, X: np.ndarray, T: np.ndarray, w_ini=-1, error_gaps=100):
        """"
        Given Data input (X,Y) ,it learns the paramter w which fits
        polynomial of degree d with Least squared error.
        The X,T is The training data set
        """
        self.X, self.T = np.reshape(X, (-1, 1)), np.reshape(T, (-1, 1))
        phi = np.power(self.X, np.arange(0, self.degree+1))
        self.data = norm_data(phi)

        if(self.method == 'grad'):
            w, self.train_error_gapped = grad_descent(self.data.phi, self.T,
                                                      batch_size=self.batch_size,
                                                      steps=self.steps, max_iter=self.max_iter,
                                                      lmda=self.lmda, w_ini=w_ini, error_gaps=error_gaps)
            self.train_error = self.train_error_gapped[-1][1]
            self.w = self.data.unnormalize_params(w)
        elif(self.method == 'mgrad'):
            w, self.train_error_gapped = moment_grad_descent(self.data.phi, self.T, batch_size=self.batch_size,
                                                             steps=self.steps, max_iter=self.max_iter,
                                                             lmda=self.lmda, w_ini=w_ini, error_gaps=error_gaps)
            self.train_error = self.train_error_gapped[-1][1]
            self.w = self.data.unnormalize_params(w)

        elif(self.method == 'piv'):
            self.w = np.fliplr(piv(self.data.phi_orginal, self.T, self.lmda))

            self.train_error = np.sum(
                np.square(np.reshape(self.data.phi_orginal@self.w, -1)-T))

    def pred(self, X_test):
        X_test = np.reshape(X_test, (-1, 1))
        phi_test = np.power(X_test, np.arange(0, self.degree+1))
        return (phi_test@self.w)

    def score(self, X_test, Y_test):
        X_test, Y_test = np.reshape(
            X_test, (-1, 1)), np.reshape(Y_test, (-1, 1))
        Y_pred = self.pred(X_test)
        return np.sum((Y_pred-Y_test)**2)
    
    
