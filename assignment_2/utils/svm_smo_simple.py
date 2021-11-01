import numpy as np
from .kernels import *
from .normalize import *
import time


class SVM_SMO_SIMPLE:
    def __init__(self, kernel='linear', C=10, gamma=None, degree=3, r_cofficient=0):

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

        # print("Model accuracy (training) b(fraction) = ", self.p_acc)
        print("Model training time = ", self.train_time)

    def kernel_matrix(self, X):
        if(self.kernel_type == 2):
            return poly_kernel_matrix(X, self.degree, self.r_cofficient)
        elif(self.kernel_type == 3):
            return gaussian_kernel_matrix(X, self.gamma)

        else:
            return linear_kernel_matrix(X)

    def clip_alpha(self, alpha, L, H):

        if L < alpha < H:
            alpha = alpha
        elif alpha <= L:
            alpha = L
        else:
            alpha = H
        return alpha

    def fit(self, X: np.ndarray, Y: np.ndarray, tol=1e-3, max_passes: int = 5):
        """
        Input X: m,n matrix
        Y : 1*m array input
        """
        m, n = X.shape
        start_time = time.time()
        # PreProcessing: Data Normalization
        X, min_X, max_X = min_max_normalize(X)
        self.min_X_train = min_X
        self.max_X_train = max_X

        alpha = np.zeros(m)
        b = 0
        E = np.zeros(m)
        total_passes = 0
        eta = 0
        L = 0
        H = 0

        # compute kernel matrix
        K_matrix = self.kernel_matrix(X)

        #  SMO optimzation
        while total_passes < max_passes:
            count_changed_alpha = 0
            for i in range(m):

                # Erro terms
                E[i] = np.sum(alpha*Y*K_matrix[:, i]) - Y[i] + b

                if (Y[i]*E[i] > tol and alpha[i] > 0) or (Y[i]*E[i] < -tol and alpha[i] < self.C):
                    j = np.random.randint(0, m)
                    while j == i:  # Make sure i \neq j
                        j = np.random.randint(0, m)

                    E[j] = sum(alpha*Y*K_matrix[:, j]) - Y[j] + b

                    old_alpha_i = alpha[i]
                    old_alpha_j = alpha[j]

                    # COmputing L and H bounds
                    if Y[i] == Y[j]:
                        L = max(0, alpha[j] + alpha[i] - self.C)
                        H = min(self.C, alpha[j] + alpha[i])
                    else:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])

                    if L == H:
                        continue

                    # Compute eta
                    eta = 2 * K_matrix[i, j] - K_matrix[i, i] - K_matrix[j, j]
                    if eta >= 0:  # FULL-SMO handle this case
                        # continue to next i.
                        continue
                    else:
                        alpha[j] = alpha[j] - (Y[j] * (E[i] - E[j])) / eta
                        # Clip alpha_j
                        alpha[j] = self.clip_alpha(alpha[j], L, H)

                    if abs(alpha[j] - old_alpha_j) < tol:
                        alpha[j] = old_alpha_j
                        # Move to next i
                        continue

                    # Compute alpha_i
                    alpha[i] = alpha[i] + Y[i]*Y[j]*(old_alpha_j - alpha[j])

                    # Compute intercept b
                    b1 = b - E[j] - Y[i] * (alpha[i] - old_alpha_i) * \
                        K_matrix[i, j] - Y[j] * \
                        (alpha[j] - old_alpha_j) * K_matrix[j, j]
                    b2 = b - E[i] - Y[i] * (alpha[i] - old_alpha_i) * \
                        K_matrix[i, i] - Y[j] * \
                        (alpha[j] - old_alpha_j) * K_matrix[i, j]

                    if 0 < alpha[i] and alpha[i] < self.C:
                        b = b2
                    elif 0 < alpha[j] and alpha[j] < self.C:
                        b = b1
                    else:
                        b = (b1+b2)/2

                    count_changed_alpha = count_changed_alpha + 1

            if count_changed_alpha == 0:
                total_passes = total_passes + 1
            else:
                total_passes = 0

        # Support vectors : sv :support vectors
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
                np.sum(self.alpha * self.sv_y * K_matrix[idx[i], sv])

        self.b /= len(self.alpha)

        # Weight vector : W = summation(alpha(i)*y(i)*x(i))
        if(self.kernel_type == 1):
            self.w = np.zeros(n)
            for i in range(len(self.alpha)):
                self.w += self.alpha[i] * self.sv_y[i]*self.sv_x[i]
        else:
            self.w = None

        # Store training time
        self.train_time = time.time() - start_time

    def project(self, X: np.ndarray.copy):
        m, n = X.shape
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

    def score(self, X_test: np.ndarray, Y_test: np.ndarray,):

        p_label: np.ndarray = self.pred(X_test)
        Y_test = np.reshape(Y_test, (-1,))
        p_acc = len(np.where(Y_test == p_label)[0])/len(Y_test)
        return p_label, p_acc
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([-1, -1, -1, 1])
# model = SVM_SMO(kernel='linear')
# model.fit(X, y, max_passes=5)


# model.describe_solution()
