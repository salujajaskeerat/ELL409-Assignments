import numpy as np


def generate_linearly_separable_data(count=100, mean1=[1, 0], mean2=[0, 1], cov=[[0.8, 0.6], [0.6, 0.8]]):
    X1 = np.random.multivariate_normal(mean1, cov, count)
    Y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, count)
    Y2 = np.ones(len(X2)) * -1

    return X1, Y1, X2, Y2


def generate_non_linearly_separable_data():
    # Generate two sets of linearly separable data
    x11, y11, x12, y12 = generate_linearly_separable_data(
        count=50, mean1=[-3, 3], mean2=[-1, 1])
    x21, y21, x22, y22 = generate_linearly_separable_data(
        count=50, mean1=[1, -1], mean2=[3, -3])
    X1, X2 = np.vstack((x11, x21)), np.vstack((x12, x22))
    Y1, Y2 = np.hstack((y11, y21)), np.hstack((y12, y22))

    return X1, Y1, X2, Y2
