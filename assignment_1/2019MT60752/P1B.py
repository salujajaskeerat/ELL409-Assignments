# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from cross_validation import *
import seaborn as sbn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polyfit import *
from IPython import get_ipython

# %%
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%


# %%
data = np.genfromtxt('./non_gaussian.csv', delimiter=',')
X, Y = data[:, 0], data[:, 1]
X, Y = zip(*sorted(zip(X, Y)))
X, Y = np.array(X), np.array(Y)


# %%
degrees = np.arange(0, 21)
piv_models = []
for degree in degrees:
    model = polyfit(degree=degree, method='piv')
    model.fit(X, Y)
    piv_models.append(model)


# %%
# After training lets plot all the curves
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))
axes = np.array(axes)

# Plotting curves

for i, ax in zip([8, 9, 10, 11], np.reshape(axes, -1)):
    ax.scatter(X, Y, label='Actual Distrib', color='red')
    ax.plot(np.sort(X), piv_models[i].pred(
        np.sort(X)), label='degree ='+str(piv_models[i].degree))
    ax.legend(['degree ='+str(piv_models[i].degree)])
plt.tight_layout()
plt.show()


# %%
piv_error = [piv_model.train_error for piv_model in piv_models[:16]]
plt.figure(figsize=(15, 8))
plt.plot(np.arange(0, len(piv_error)), (piv_error), '-o')

plt.title("PIV optimization: Training error vs Degree"), plt.xlabel(
    "Degree of polynomial"), plt.ylabel("E(W) training error")
plt.show()

# %% [markdown]
# Clearly polynomial of degree>=9 fits the curve nicely

# %%
X.shape


# %%
# Maximum Likelihood i.e (Without regularization , tuning the degree)
test_err, train_err = [], []
posssible_degree = np.arange(0, 14, 1)
indices = np.arange(0,)
for degree in posssible_degree:
    # print(kfold_cross_validation(X, Y, degree=degree, method='piv', K=10))
    mse_train, mse_test = kfold_cross_validation(
        X, Y, degree=degree, method='piv', K=10)
    test_err.append(mse_test), train_err.append(mse_train)


# %%
plt.plot(posssible_degree, np.log(train_err), '-o')
plt.plot(posssible_degree, np.log(test_err), '-o')
plt.tight_layout()
plt.legend(['Train error', 'Cross Validation Error'])
plt.xlabel('degree'), plt.ylabel('$\log_{10}{E_{rms}}$')
plt.xticks(np.arange(1, 12, 1))
plt.show()

# %% [markdown]
# Using regularization in our model

# %%
X.shape, Y.shape


# %%
dd = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
# df.head()
posssible_degree = np.arange(8, 13, 1)
lmda = [10**i for i in range(-2, -15, -1)]
# lmda.append(0)
for lm in lmda:
    for degree in posssible_degree:
        train_err, test_err = shuffled_cvr(
            X, Y, degree=degree, K=10, lmda=lm, method='piv')
        dd.loc[-1] = {'lmda': (lm), 'degree': degree,
                      'test_err': test_err, 'train_err': train_err}
        dd.index = dd.index + 1
        dd = dd.sort_index()

dd = dd.sort_values('test_err')

dd.head(10)

# %% [markdown]
# degree=11 work great

# %%
lmda = [10**i for i in range(-1, -10, -1)]
mse_train_total, mse_test_total = [], []
for lm in lmda:

    mse_train, mse_test = kfold_cross_validation(
        X[indices], Y[indices], degree=9, method='piv', K=10, lmda=lm)
    mse_test_total.append(mse_test), mse_train_total.append(mse_train)

# %% [markdown]
# degree=11
# lmda=1e-6

# %%
plt.figure(figsize=(7, 7))
model = polyfit(degree=9, method='piv', lmda=1e-7)
model.fit(X, Y)
hx = np.ravel(model.pred(X))
plt.plot(X, model.pred(X), color='green', label='Prediction')
plt.title("Degree =9 , $\lambda=$1e-7")
plt.plot(X, Y, color='red', label='True Label')
plt.legend()
plt.show()

# %% [markdown]
# Thus degree=9 fits nicely, increasing the degreee would only lead to increase in complexity .

# %%
noise = np.ravel(hx)-np.ravel(Y)
print(np.mean(noise))


# %%
np.mean(noise)
fig, ax = plt.subplots(figsize=(7, 7))
sbn.histplot(noise, kde=True, ax=ax)
# plt.xscale(5)
# plt.xticks(np.arange(-22,22,1))
plt.title("Noise distribution")
plt.show()


# %%
normalized_noise = noise/(40)

# %% [markdown]
# Sort of beta distribution
# mean=0
# mode=0
#
#
# mean = (alpha)/(alpha+beta)
# mode=(alpha-1)/(alpha+beta-2)
# since aplha

# %%
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
axes = np.array(axes)

for alpha, ax in zip(np.arange(4, 6, 0.5), np.reshape(axes, -1)):
    # generate points from alpha distribution
    # fig, ax = plt.subplots()
    distrib = (np.random.beta(alpha, alpha, 100000)-0.5)
    sbn.kdeplot(distrib, ax=ax, label="alpha="+str(alpha))
    sbn.kdeplot(normalized_noise, ax=ax, label="normalized noise =(noise/40)")
    ax.legend()
plt.tight_layout()

# %% [markdown]
# Good estimate of the noise in model
# Noice =40 (beta(4.5,4.5)-0.5)
# %% [markdown]
#
