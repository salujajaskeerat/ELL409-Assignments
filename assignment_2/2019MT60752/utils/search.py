from typing import Dict, Iterable, List, Mapping
from itertools import *

from numpy import ndarray
from kernels import kernel
from svm_cvx import SVM_CVX
from svm_libsvm import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ParamGrid:
    """
    paramgrid is a Map or Iterables
    """

    def __init__(self, paramGrid):
        # Check if the paramGrid is a dictionary
        if (not isinstance(paramGrid, (Mapping, Iterable))):
            raise TypeError("param grid is not a list, or dictionary")

        if(isinstance(paramGrid, Mapping)):
            paramGrid = [paramGrid]
        self.paramGrid = paramGrid

    def iterate(self):
        for grid in self.paramGrid:
            elements = sorted(grid.items())

            if not elements:
                yield {}
            else:
                key, val = zip(*elements)

                for v in product(*val):
                    parameter = dict(zip(key, v))

                    yield parameter


class gridSearchCV_SVM:
    """

    - Tune the hyperparameter of SVM model by passing range of hyperparamter values

    # EG:
    paramGrid : {'kernel':['linear','rbf','polynomial'],'C':[1,10,100,1000,1000],'gamma'=[1e-1,1e-2,1e-3,1e-4,1e-5]}

    # Attributes

    - X:Training data
    - Y:Training labels
    - estimator_type : 1 or 0 , 0 for libsvm ,  1 for svxopt
    - cv : Number of folds used in K-fold Cross validation
    - n_jobs : number of process in paralled, since each cartesian product works independent of other

    # Methods
    - run()
    - best_param() : return paramters with best cross-validation score
    - head(int k) : returns top k best paramters
    """

    def __init__(self, paramGrid,  X: ndarray, Y: ndarray, estimator_type: int = 0, cv: int = 5, n_jobs: int = 1):
        self.paramGrid = ParamGrid(paramGrid)
        indexes = np.arange(X.shape[0])
        np.random.shuffle(indexes)

        self.X = X[indexes]
        self.Y = Y[indexes]
        self.cv = cv
        self.n_jobs = n_jobs
        self.num_features = X.shape[1]

        # SVM_type =0 for LIBSVM implementation , 1 for CVXOPT
        self.svm_type = estimator_type % 2

    def run(self):

        cv_results: List[Dict[str, ]] = []
        for param in self.paramGrid.iterate():

            # Param is a dictionary of hyperparameters
            kernel = 'linear' if(not param.__contains__(
                'kernel')) else param['kernel']
            C = 1 if(not param.__contains__(
                'C')) else param['C']
            gamma = (1/self.num_features) if(not param.__contains__(
                'gamma')) else param['gamma']
            degree = 3 if(not param.__contains__(
                'degree')) else param['degree']
            r_cofficient = 0 if(not param.__contains__(
                'r_cofficient')) else param['r_cofficient']

            if(self.svm_type == 0):
                model = SVM_LIBSVM(kernel=kernel, degree=degree, C=C,
                                   gamma=gamma, r_cofficient=r_cofficient)
            else:
                model = SVM_CVX(kernel=kernel, degree=degree,
                                C=C, gamma=gamma, r_cofficient=r_cofficient)
            test_score, train_score = model.kfold_cross_validation(
                self.X, self.Y, cv=self.cv)

            param['test_score'] = test_score
            param['train_score'] = train_score

            cv_results.append(param)
        cv_results = sorted(
            cv_results, key=lambda x: x['test_score'], reverse=True)
        self.cv_results = cv_results

    def best_param(self):
        return self.cv_results[0]

    def head(self, num: int = 10):
        return self.cv_results[:num]

    def plot_graph(self):
        if self.cv_results is None:
            return

        sns.set_style("darkgrid")
        # plot for Guassian Kernel
        df = pd.DataFrame(self.cv_results)

        # Plot score Vs C  : linear kernel
        if('linear' in df['kernel'].values):

            plt.figure()
            dfl = df.loc[df['kernel'] == 'linear']
            ax = sns.lineplot(x=np.log10(
                dfl['C']), y=dfl['test_score'], label='test score')
            # ax = sns.lineplot(x=np.log10(
            #     dfl['C']), y=dfl['train_score'], label='train score')
            ax.set_title('Linear Kernel : CVR Score vs C'), ax.set_xlabel(
                '$\log_{10}{C}$')
            # ax.set_ylim(0.8, 1)
            plt.show()

        # Plot C vs gamma
        if('gaussian' in df['kernel'].values):
            plt.figure()
            ax = sns.heatmap(df.loc[df['kernel'] == 'gaussian'].pivot('C', 'gamma', 'test_score'),
                             annot=True, cmap='viridis', fmt='.4g')
            ax.set_title('Gaussian Kernel : Cross Validation Scores')
            ax.set_xlabel('Gamma'), ax.invert_yaxis()
            ax.set_ylabel('C (Regularization strength)')
            plt.show()
        if('rbf' in df['kernel'].values):
            plt.figure()
            ax = sns.heatmap(df.loc[df['kernel'] == 'rbf'].pivot('C', 'gamma', 'test_score'),
                             annot=True, cmap='viridis', fmt='.4g')
            ax.set_title('Gaussian Kernel : Cross Validation Scores')
            ax.set_xlabel('Gamma'), ax.invert_yaxis()
            ax.set_ylabel('C (Regularization strength)')
            plt.show()

        # polynomial kernel :
        if('polynomial' in df['kernel'].values):
            plt.figure()
            ax = sns.heatmap(df.loc[df['kernel'] == 'polynomial'].pivot(
                'C', 'degree', 'test_score'), annot=True, cmap='viridis', fmt='.4g')
            ax.set_title('Polynomial Kernel : Cross Validation Scores')
            ax.set_xlabel('Degree of kernel'), ax.invert_yaxis()
            ax.set_ylabel('C (Regularization strength)')
            plt.show()
