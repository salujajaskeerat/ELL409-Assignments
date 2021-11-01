"""
Reference : Code developed from pseudo code in JOHN PLATT PAPER (SMO)
Credits : John C. Platt
"""

import numpy as np
from .kernels import *
from .normalize import *
import time


class SVM_SMO_FULL:
    def __init__(self,  kernel='linear', C=10, gamma=None, degree=3, r_cofficient=0, tol=1e-3, eps=1e-2):

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
        self.b = 0.0

        self.tol = tol
        self.eps = eps

    # Descion function for the model
    def dsfn(self, alphas, y, b):
        return (alphas * y) @ self.K - b

    def kernel_matrix(self, X):
        if(self.kernel_type == 2):
            return poly_kernel_matrix(X, self.degree, self.r_cofficient)
        elif(self.kernel_type == 3):
            return gaussian_kernel_matrix(X, self.gamma)

        else:
            return linear_kernel_matrix(X)

    def kernel(self, x1, x2):
        return kernel(x1, x2, kernel_type=self.kernel_type, degree=self.degree, gamma=self.gamma)

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

        print("Number of support vector is ", len(self.sv_alphas))

        print("To get the support vector use model.sv_x ")

        print("Model training time = ", self.train_time)

    def objective_function(self, alphas, Y):
        return np.sum(alphas) - 0.5 * np.sum((Y * Y) * self.K * (alphas * alphas))

    def takeStep(self, index1, index2):

        if index1 == index2:
            return 0
        E1 = self.errors[index1]
        E2 = self.errors[index2]
        alpha1 = self.alphas[index1]
        alpha2 = self.alphas[index2]
        y1 = self.y[index1]
        y2 = self.y[index2]

        s = y1 * y2

        if (y1 != y2):
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        elif (y1 == y2):
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)
        if (L == H):
            return 0

        if (2 * self.K[index1, index2] - self.K[index1, index1] - self.K[index2, index2] < 0):
            a2 = alpha2 - y2 * \
                (E1 - E2) / (2 * self.K[index1, index2] -
                             self.K[index1, index1] - self.K[index2, index2])
            if L < a2 < H:
                a2 = a2
            elif (a2 <= L):
                a2 = L
            elif (a2 >= H):
                a2 = H

        else:
            """
            SIMPLE SMO DOES NOT HANDLE THIS CASE
            """
            alpha_copy = self.alphas.copy()
            alpha_copy[index2] = H
            # objective a2 = H
            Hobj = self.objective_function(alpha_copy, self.y)
            alpha_copy[index2] = L
            # objective  a2 = L
            Lobj = self.objective_function(alpha_copy, self.y)

            if Lobj > (Hobj + self.eps):
                a2 = L
            elif Lobj < (Hobj - self.eps):
                a2 = H
            else:
                a2 = alpha2

        if a2 < 1e-8:
            a2 = 0.0
        elif a2 > (self.C - 1e-8):
            a2 = self.C

        if (np.abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps)):
            return 0

        a1 = alpha1 + s * (alpha2 - a2)
        b2 = E2 + y1 * (a1 - alpha1) * self.K[index1, index2] + \
            y2 * (a2 - alpha2) * self.K[index2, index2] + self.b
        b1 = E1 + y1 * (a1 - alpha1) * self.K[index1, index1] + \
            y2 * (a2 - alpha2) * self.K[index1, index2] + self.b

        if 0 < a1 and a1 < self.C:
            b = b1
        elif 0 < a2 and a2 < self.C:
            b = b2
        else:
            b = (b1 + b2) * 0.5

        self.alphas[index1] = a1
        self.alphas[index2] = a2
        if 0.0 < alpha1 < self.C:
            self.errors[index1] = 0.0
        if 0.0 < alpha2 < self.C:
            self.errors[index2] = 0.0

        for i in range(self.m):
            if(i != index1 and i != index2):
                self.errors[i] = self.errors[i] + y1 * \
                    (a1-alpha1)*self.kernel(self.X[index1], self.X[i]) + y2*(
                        a2-alpha2)*self.kernel(self.X[index2], self.X[i])+self.b-b
        self.b = b
        return 1

    def examineExample(self, index2):
        """"
        The function finds another index and then optimizes the pair

        returns 0 if not able to optimize the corresponding index else returns 1
        """

        y2 = self.y[index2]
        alph2 = self.alphas[index2]
        E2 = self.errors[index2]
        r2 = E2 * y2

        if ((r2 < -self.tol and alph2 < self.C) or (r2 > self.tol and alph2 > 0)):

            if len(self.alphas[(self.alphas != 0) & (self.alphas != self.C)]) > 1:
                if self.errors[index2] > 0:
                    index1 = np.argmin(self.errors)
                elif self.errors[index2] <= 0:
                    index1 = np.argmax(self.errors)
                result = self.takeStep(index1, index2)
                if result:
                    return 1

            for index1 in np.roll(np.where((self.alphas != 0) & (self.alphas != self.C))[0],
                                  np.random.choice(np.arange(self.m))):
                result = self.takeStep(index1, index2)
                if result:
                    return 1

            for index1 in np.roll(np.arange(self.m), np.random.choice(np.arange(self.m))):
                result = self.takeStep(index1, index2)
                if result:
                    return 1
        return 0

    def fit(self, X: np.ndarray, Y: np.ndarray):

        # Intializing the object
        m, n = X.shape
        X, min_X, max_X = min_max_normalize(X)
        self.min_X_train = min_X
        self.max_X_train = max_X
        self.m = m
        self.n = n
        self.X = X
        self.y = Y
        self.alphas = np.zeros(m)

        self.K = self.kernel_matrix(X)
        self.errors = self.dsfn(
            self.alphas, self.y, self.b)-self.y

        start_time = time.time()
        numChanged = 0
        examineAll = 1

        while(numChanged > 0) or (examineAll):
            numChanged = 0
            if examineAll:

                for i in range(self.m):
                    examine_result = self.examineExample(i)
                    numChanged += examine_result

            else:
                for i in np.where((self.alphas != 0) & (self.alphas != self.C))[0]:
                    examine_result = self.examineExample(i)
                    numChanged += examine_result

            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1

            # Finally model trained
            finish_time = time.time()
            self.train_time = finish_time-start_time
            # Support vectors : sv :support vectors

            sv = self.alphas > 1e-5

            idx = np.arange(len(self.alphas))[sv]
            # self.alpha = self.alpha[sv]
            self.sv_x = self.X[sv]
            self.sv_y = np.ravel(self.y[sv])
            self.sv_alphas = self.alphas[sv]
            # value of b in W.T @ X + b= 0
            self.b = 0
            for i in range(len(self.sv_alphas)):
                self.b += self.sv_y[i] - \
                    np.sum(self.sv_alphas * self.sv_y * self.K[idx[i], sv])

            self.b /= len(self.sv_alphas)

            # Weight vector : W = summation(alpha(i)*y(i)*x(i))
            if(self.kernel_type == 1):
                self.w = np.zeros(self.X.shape[1])
                for i in range(len(self.sv_alphas)):
                    self.w += self.sv_alphas[i] * self.sv_y[i]*self.sv_x[i]
            else:
                self.w = None

    def project(self, X: np.ndarray.copy):
        m, n = X.shape
        X = tranform_min_max(
            X, self.min_X_train, self.max_X_train)

        if(self.w is not None):
            return np.sign(X @ self.w+self.b)

        else:
            ypred = np.ones(m)*self.b
            for i in range(m):
                for alpha_i, sv_x, sv_y in zip(self.alphas, self.sv_x, self.sv_y):

                    ypred[i] += alpha_i * sv_y * \
                        kernel(X[i], sv_x, kernel_type=self.kernel_type,
                               degree=self.degree, gamma=self.gamma, r=self.r_cofficient)
        return ypred

    def pred(self, X: np.ndarray):
        return np.sign(self.project(X))

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

# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([-1, 1, 1, -1])
# model = SVM_SMO_FULL(kernel='gaussian', gamma=10, C=100, tol=1e-3, eps=1e-3)
# model.fit(X, y)

# # print(model.alphas)

# model.describe_solution()
# print(model.score(X, y))
