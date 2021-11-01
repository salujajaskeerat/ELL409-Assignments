import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contour(ax, X_train: np.ndarray, Y_train: np.ndarray, model, h=0.5):
    if(X_train.shape[1] != 2):
        raise ValueError('Only 2d data can be plot')
    pos_indices, neg_indices = np.where(
        Y_train == 1)[0], np.where(Y_train == -1)[0]
    X1 = X_train[pos_indices]
    X2 = X_train[neg_indices]

    # Plot contours
    xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1], h=h)
    X = np.array([[x1, x2] for x1, x2 in zip(xx.ravel(), yy.ravel())])
    Z = model.pred(X).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Scatter Plot
    sns.scatterplot(x=X1[:, 0], y=X1[:, 1], label='+1', color='darkred', ax=ax)
    sns.scatterplot(x=X2[:, 0], y=X2[:, 1], label='-1',
                    color='darkblue', ax=ax)
