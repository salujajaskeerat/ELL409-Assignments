from typing import Dict, Iterable, List, Mapping
from itertools import *

import numpy as np

import matplotlib.pyplot as plt
from numpy.core.defchararray import split
from numpy.core.fromnumeric import size
import pandas as pd
import seaborn as sns

from neuralNet import NeuralNet


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


def cv_neural_net(data):

    pass


def kfolddCV_neural_net(data: np.ndarray, labels: np.ndarray, sizes, lmbda, eta, cost_func, neuron_type, epochs, batch_size, cv=5):

  # Split the data into cv number of folds
    indexes = np.arange(0, data.shape[0])
    np.random.shuffle(indexes)
    splits = np.array_split(indexes, cv)

    net_train_accuracy, net_test_accuracy, net_train_cost, net_test_cost = 0.0, 0.0, 0.0, 0.0
    for i in range(cv):

        # data
        train_indexes = np.hstack(np.delete(splits, i, axis=0))

        # Now modify the train_labels to vectorize them
        train_x = [np.reshape(x, (-1, 1)) for x in data[train_indexes]]
        train_y = [vectorized_result(y) for y in labels[train_indexes]]

        test_x = [np.reshape(x, (-1, 1)) for x in data[splits[i]]]
        test_y = labels[splits[i]]

        train_data = list(zip(train_x, train_y))
        test_data = list(zip(test_x, test_y))

        model = NeuralNet(size=sizes, neuron_type=neuron_type, cost=cost_func)
        train_accuracy, train_cost, test_accuracy, test_cost = model.fit(train_data=train_data, epochs=epochs,
                                                                         mini_batch_size=batch_size, eta=eta, verbose=False, lmda=lmbda, test_data=test_data)

        # We take the best values of train_accuracy ,test_accuracy etc which are given for each epoch
        # Only store the best values

        train_accuracy = np.max(train_accuracy)
        test_accuracy = np.max(test_accuracy)
        train_cost = np.min(train_cost)
        test_cost = np.min(test_cost)

        net_train_accuracy += (train_accuracy/cv)
        net_test_accuracy += (test_accuracy/cv)
        net_train_cost += (train_cost/cv)
        net_test_cost += (test_cost/cv)

    return net_train_accuracy, net_train_cost, net_test_accuracy, net_test_cost

    pass


class gridSearchCV:
    """
    - Tune the hyperparameter of neural net model by passing range of hyperparamter values
    # EG:
    paramGrid : {'lmbda':[1e-1,1e-2,1e-3], 'neuron_type':['sigmoid','relu'],'cost':['cross_entropy_cost',quadratic_cost'],'sizes':[[100,20,10],[100,10]]}

    lmbda
    neuron_type
    cost : Cost function to minimize for the optmization
    sizes : Layer configuration in ANN
    epoch : number of epochs
    eta:learning rate
    batch_size : for the gradient descent


    # Attributes
    - cv : Number of folds used in K-fold Cross validation
    # Methods
    - fit(train_x,train_Y)
    - best_param() : return paramters with best cross-validation score
    - head(int k) : returns top k best paramters
    """

    def __init__(self, paramGrid, cv: int = 5,):
        self.paramGrid = ParamGrid(paramGrid)
        self.cv = cv

    def fit(self, X, Y, verbose=True):

        cv_results: List[Dict[str, ]] = []

        # Caluclate scores for each configuration in the params we have
        for param in self.paramGrid.iterate():

            if(verbose):
                print("Configuration :", param)
            # Param is a dictionary of hyperparameters
            neuron_type = 'sigmoid' if(not param.__contains__(
                'neuron_type')) else param['neuron_type']

            lmbda = 0.0 if(not param.__contains__(
                'lmbda')) else param['lmbda']
            cost = 'quadratic_cost' if(not param.__contains__(
                'cost')) else param['cost']
            sizes = param['sizes']
            epoch = 10 if(not param.__contains__(
                'epoch')) else param['epoch']
            eta = 1e-3 if(not param.__contains__(
                'eta')) else param['eta']
            batch_size = 10 if(not param.__contains__(
                'batch_size')) else param['batch_size']

            train_accuracy, train_cost, test_accuracy, test_cost = kfolddCV_neural_net(
                X, Y, sizes=sizes, lmbda=lmbda,
                eta=eta, cost_func=cost,
                neuron_type=neuron_type,
                epochs=epoch, batch_size=batch_size,
                cv=5)

            param['train_accuracy'] = train_accuracy
            param['train_cost'] = train_cost
            param['test_accuracy'] = test_accuracy
            param['test_cost'] = test_cost

            if verbose:
                print('Scores : ', param)
            cv_results.append(param)

        cv_results = sorted(
            cv_results, key=lambda x: x['test_accuracy'], reverse=True)
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


def vectorized_result(j):

    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
