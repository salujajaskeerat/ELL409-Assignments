import numpy as np
import time
import cvxopt as cp
from .kernels import *
from .normalize import *


class SVM_CVX:
    def __init__(self, kernel='linear', C=None, gamma=None, degree=3, r_cofficient=0):

        self.gamma = gamma
        self.C = C
        if(self.C is not None):  # regularized SVM
            self.C = float(self.C)
        self.degree = degree
        self.r_cofficient = r_cofficient
        self.kernel_type = {
            'linear': 1,
            'polynomial': 2,
            'gaussian': 3,  # radial basis function
            'rbf': 3
        }[kernel]

    def kernel_matrix(self, X):
        if(self.kernel_type == 2):
            return poly_kernel_matrix(X, self.degree, self.r_cofficient)
        elif(self.kernel_type == 3):
            return gaussian_kernel_matrix(X, self.gamma)

        else:
            return linear_kernel_matrix(X)

    def fit(self, X_train: np.ndarray, Y: np.ndarray, quiet=True):
        """
        Input X: m,n matrix
        Y : m*1 matrix input
        """
        if(Y.ndim == 1):
            Y = Y.reshape((-1, 1))
        m, n = X_train.shape
        start_time = time.time()
        # PreProcessing: Data Normalization
        X, min_X, max_X = min_max_normalize(X_train)
        self.min_X_train = min_X
        self.max_X_train = max_X

        # CVXOPT Problem Formulation
        """
        MIN(alpha)  = alpha.T @ P @ alpha + q.T @ alpha

        P = y@y.T @K #K:Kernel matrix
        q = -[1,1,1.....1].T
        Constrainst

        A @ alpha =b
        G@alpha <=h
        """

        K = self.kernel_matrix(X)  # Kernel matrix
        P = cp.matrix(np.array((Y @ Y.T)*K).astype('float'))  # P
        q = cp.matrix(np.ones(m)*-1)  # Q
        A = cp.matrix(np.reshape(Y.T, (1, m)).astype('float'))
        b = cp.matrix([0.0])

        if(self.C is None):
            G = cp.matrix(np.diag(np.ones(m)) * -1)
            h = cp.matrix(np.zeros(m))

        else:
            g1 = np.diag(np.ones(m)*-1)
            g2 = np.identity(m)
            G = cp.matrix(np.vstack((g1, g2)))
            h1 = np.zeros(m)
            h2 = np.ones(m)*self.C
            h = cp.matrix(np.hstack((h1, h2)))

        # Solve the problem
        cp.solvers.options['show_progress'] = False
        soln = cp.solvers.qp(P, q, G, h, A, b)

        # Optimized value of alpha
        alpha = np.ravel(soln['x'])
        # Support vectors
        sv = alpha > 1e-5

        idx = np.arange(len(alpha))[sv]
        self.alpha = alpha[sv]
        self.sv_x = X[sv]
        self.sv_x_denormalized = unnormalize_min_max(
            self.sv_x, self.min_X_train, self.max_X_train)
        self.sv_y = np.ravel(Y[sv])

        # value of b in W.T @ X + b= 0
        self.b = 0
        for i in range(len(self.alpha)):
            self.b += self.sv_y[i] - \
                np.sum(self.alpha * self.sv_y * K[idx[i], sv])

        self.b /= len(self.alpha)

        # Weight vector : W = summation(alpha(i)*y(i)*x(i))
        if(self.kernel_type == 1):
            self.w = np.zeros(n)
            for i in range(len(self.alpha)):
                self.w += self.alpha[i] * self.sv_y[i]*self.sv_x[i]
        else:
            self.w = None

        finish_time = time.time()

        if not quiet:
            print('Total Training Time = ', finish_time-start_time)
        p_label, self.p_acc = self.score(X, Y)
        self.train_time = finish_time-start_time

    def describe_solution(self):
        kernel = {
            1: 'linear',
            2: 'polynomial',
            3: 'gaussian',  # radial basis function
        }[self.kernel_type]
        print("Kernel:", kernel, "   C :", self.C)

        if(self.w is not None):
            print("W(weight matrix) = ", self.w)
        print("Intercept (b) = ", self.b)

        print("Number of support vector is ", len(self.alpha))

        print("To get the support vector use model.sv_x ")

        print("Model accuracy (training) b(fraction) = ", self.p_acc)
        print("Model training time = ", self.train_time)

    def project(self, X: np.ndarray.copy):

        m, n = X.shape
        # PreProcessing
        # X =
        X = tranform_min_max(
            X, self.min_X_train, self.max_X_train)

        if(self.w is not None):
            return np.sign(X @ self.w+self.b)

        else:
            ypred = np.ones(m)*self.b
            for i in range(m):
                for alpha_i, sv_x, sv_y in zip(self.alpha, self.sv_x, self.sv_y):

                    ypred[i] += alpha_i * sv_y * \
                        kernel(X[i], sv_x, kernel_type=self.kernel_type,
                               degree=self.degree, gamma=self.gamma, r=self.r_cofficient)
        return ypred

    def pred(self, X: np.ndarray):
        return np.sign(self.project(X))

    def fit_predict(self, X: np.ndarray, Y: np.ndarray):
        self.fit(X, Y)
        return self.pred(X)

    def score(self, X_test: np.ndarray, Y_test: np.ndarray,):

        p_label: np.ndarray = self.pred(X_test)
        Y_test = np.reshape(Y_test, (-1,))
        p_acc = len(np.where(Y_test == p_label)[0])/len(Y_test)
        return p_label, p_acc

    def kfold_cross_validation(self, X: np.ndarray, Y: np.ndarray, cv: np.int32 = 5, random_state=False) -> float:
        """
        Return cross validation score on the training data
        Perform K-fold cross validation on model given X and Y
        """
        indexes = np.arange(0, X.shape[0])
        if(random_state):
            np.random.shuffle(indexes)
        splits = np.array_split(indexes, cv)

        test_score, train_score = 0.0, 0.0
        for i in range(0, cv):
            # print(np.hstack(np.delete(splits, i, axis=0)))
            X_train, Y_train = X[np.hstack(np.delete(splits, i, axis=0))], Y[np.hstack(
                np.delete(splits, i, axis=0))]
            X_test, Y_test = X[np.ravel(
                splits[i])], Y[np.ravel(splits[i])]
            self.fit(X_train, Y_train)
            s = self.score(X_test, Y_test)[1]
            test_score += s
            train_score += self.p_acc

        # print(score, cv)

        return test_score/cv, train_score/cv
