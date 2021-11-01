import numpy as np
import time
from libsvm.svm import svm_parameter, svm_problem
from libsvm.svmutil import svm_predict, svm_train
from libsvm.svmutil import *

from utils.normalize import *


class SVM_LIBSVM:

    def __init__(self, kernel='linear', C=1, gamma=None, degree=3, r_cofficient=1):

        self.gamma = gamma
        if(self.gamma is not None):
            self.gamma = float(self.gamma)
        self.C = C
        if(self.C is not None):  # regularized SVM
            self.C = float(self.C)
        self.degree = degree
        self.r_cofficient = r_cofficient
        self.kernel_type = {
            'linear': 0,
            'polynomial': 1,
            'rbf': 2,  # radial basis function
            'gaussian': 2
        }[kernel]

        if(gamma is None):
            # gamma is auto
            self.paramter = svm_parameter(
                f'-t {self.kernel_type} -c {self.C}   -d {self.degree} -r {self.r_cofficient} -q')
        else:
            self.paramter = svm_parameter(
                f'-t {self.kernel_type} -c {self.C} -g {self.gamma} -d {self.degree} -r {self.r_cofficient} -q')

    def fit(self, X: np.ndarray.copy, Y: np.ndarray, quiet=True):

        if(Y.ndim == 2):
            Y = Y.reshape((-1,))
        s = time.time()
        X, min_X, max_X = min_max_normalize(X)
        probs = svm_problem(Y, X)
        model = svm_train(probs, self.paramter)

        self.model = model
        self.min_X_train = min_X
        self.max_X_train = max_X

        self.p_acc = self.score(X, Y)[1]
        f = time.time()
        if(not quiet):
            print('Total Training Time = ', f-s)

    def pred(self, X: np.ndarray):
        """"
        return the predicted labels for the SVM
        """

        return np.array(self.score(X, [])[0])

    def score(self, X_test: np.ndarray, Y_test: np.ndarray):
        """
        Return classifier score on the classifier 
        """
        if(X_test.shape[0] == 0):
            return 1.0

        X = tranform_min_max(
            X_test, self.min_X_train, self.max_X_train)
        p_labels, p_acc, p_vals = svm_predict(
            y=Y_test, x=X, m=self.model, options='-q')

        # p_acc : (accuracy ,mse_err , squared correlation cofficient)
        return p_labels, p_acc[0]/100

    def fit_predict(self, X: np.ndarray, Y: np.ndarray):
        self.fit(X, Y)
        return self.pred(X, Y)

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
