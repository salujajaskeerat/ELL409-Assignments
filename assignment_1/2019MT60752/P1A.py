# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import seaborn as sbn
from polyfit import *
from cross_validation import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython import get_ipython

# %%
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %% [markdown]
# ## Polynomial Fit for first 20 points
#
# The data has gaussian noise with 0 mean i.e $N(h(x)|t,\frac{1}{\beta})$ , where $\frac{1}{\beta}$ is variance of the noise distribution
# %% [markdown]
# $\textbf{Load and plotting data}$

# %%
data = np.genfromtxt('./gaussian.csv', delimiter=',')
X, Y = data[:20, 0], data[:20, 1]
X, Y = zip(*sorted(zip(X, Y)))
X, Y = np.array(X), np.array(Y)
print(X.shape)

# %% [markdown]
# ## Error Minimization on given number of points
#
#
# ### (a) Optimization using **Batch Gradient descent** on given data
#
# Notice that convergence of gradient descent algorithm is quite slow , took around 50,000 iterations on an average to converge.

# %%
degrees = np.arange(0, 20)
models, w_ini = [], np.zeros(1).T
for degree in degrees:
    model = polyfit(degree=degree, max_iter=50000, steps=1e-3)
    model.fit(X, Y, w_ini=w_ini)
    w_ini = np.append(model.w, [[0]], axis=0)
    models.append(model)


# %%
# After training lets plot all the curves

plt.figure(figsize=(18, 8))
# Plotting curves
plt.scatter(X, Y, label='Actual Distrib', color='red')
for i in (0, 4, 8, 12, 16, 19):
    plt.plot(np.sort(X), models[i].pred(np.sort(X)), label='degree ='+str(
        models[i].degree)+(" Train error = ")+str(np.float16(models[i].train_error)))
plt.legend(), plt.title('PolyFit on First 20 Points'), plt.xlabel('X'), plt.ylabel('Y')
plt.show()


# %%
grad_error = [model.train_error for model in models]
plt.figure(figsize=(15, 5))
plt.plot(np.arange(0, len(grad_error)), grad_error, '-o')
plt.title("Gradient Descent: Training error vs Degree"), plt.xlabel(
    "Degree of polynomial"), plt.ylabel("E(W) training error")
plt.show()

# %% [markdown]
# #### Observations:
# 1. **Starting degree 8 onwards** the training error is reduced drastically , hence  $degree\geq 8 $ are good fit to our data set
# 2. Training error reduces as the degree of poynomial is increased ,but this occurs at the expense of the increase complexity of model.
# %% [markdown]
# ### **Variation of error vs iteration  of gradient descent**
#
# From below we can say the gradient descent algorithm converges very fast for the first 5000 iteration , after that the convergence of gradient descent is very slow in successive iteration.
# Here 1 iteration corresponds to 1 pass over the data set , hence in our case 1 iteration implies 1 pass over all 20 points.
#
# Error is monotonically decresing as number of iterations increases
#

# %%
start, gaps, max_iter = 100, 1000, 70000
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
axes = np.array(axes)

for degree, ax in zip([6, 10, 14, 19], axes.reshape(-1)):
    iters = np.arange(start, max_iter, gaps)
    modeli = polyfit(degree=degree, max_iter=max_iter, steps=5e-4)
    modeli.fit(X, Y, error_gaps=gaps)
    iter_errors = modeli.train_error_gapped[1:]
    ax.plot(iters, iter_errors, '-o', color='red')
    ax.set_xlabel('Iterations'), ax.set_ylabel(
        'Least squared Error'), ax.legend(['Degree = '+str(degree)])

# %% [markdown]
# ### Variation of Error $E(W)$ vs batch size in gradient descent
#
# <span class="mark">Since the data set is quite small i.e only 20 points</span> . Also the data was normalized before the grad desccent thus the <span class="mark">batch gradient descent performed similar</span> on all batches

# %%
error_batches = []
max_iter = 1000
for batch_size in np.arange(1, 20, 2):
    model_b = polyfit(degree=5, batch_size=batch_size,
                      max_iter=max_iter, method='grad')
    model_b.fit(X, Y, error_gaps=10)
    error_batches.append(model_b.train_error_gapped)


# %%
plt.figure(figsize=(15, 5))
for eb in error_batches:
    x = [coord[0] for coord in eb]
    y = [coord[1] for coord in eb]
    plt.plot(x, y)
#     plt.ylim(0,10)
#     plt.xlim(0,100)

# %% [markdown]
# ### (b) Optimization using **PIV** (penrose inverse matrix)
#
# Below is polynomial fitting plot of various degree of polynomials optimized using PIV method .
#
# **Observations:**
# Starting $degree \geq 15$ the PIV fails , thus gives poor fitting , as shown below.This is the reason we cannot rely on PIV method for fitting higher degree poynomials on our data set.

# %%
degrees = np.arange(0, 21)
piv_models = []
for degree in degrees:
    model = polyfit(degree=degree, method='piv')
    model.fit(X, Y)
    piv_models.append(model)


# %%
# After training lets plot all the curves
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 10))
axes = np.array(axes)

# Plotting curves

for i, ax in zip([0, 4, 8, 12, 14, 16, 17, 19], np.reshape(axes, -1)):
    ax.scatter(X, Y, label='Actual Distrib', color='red')
    ax.plot(np.sort(X), piv_models[i].pred(
        np.sort(X)), label='degree ='+str(piv_models[i].degree))
    ax.legend(['degree ='+str(piv_models[i].degree)])
plt.tight_layout()
plt.show()


# %%
print(np.ravel(piv_models[9].w))

# %% [markdown]
# ### Error vs degree of poynomial
#
# Since PIV fails for degree>=5 hence the error shoots up

# %%
piv_error = [piv_model.train_error for piv_model in piv_models[:16]]
plt.figure(figsize=(15, 8))
plt.plot(np.arange(0, len(piv_error)), (piv_error), '-o')

plt.title("PIV curve fitting: Error vs Degree"), plt.xlabel(
    "Degree of polynomial"), plt.ylabel("E(W) training error")
plt.show()

# %% [markdown]
# ## Good fit of polynomial (Underfitting , overfitting) (without regularization)
# %% [markdown]
# We know the  $\textbf{train error} -> 0$ as the degree of poynomial M increases but this lead to overfitting of the model over the underlying data set.
#
# **Test validation**
# To test a model we will partition our data i.e 20 points into test and train data . The model is trained on trained data and scored on the test data.
#
# Several models are scored using above schema , one with the best score i.e in our case the least error is chosen.
#
# **I have used K-fold cross validation method to score the models with variable hyper-paramters**.
#
# The k-fold cross validation method is implemented in file cross-validation.py

# %%

test_err, train_err = [], []
posssible_degree = np.arange(0, 14, 1)
for degree in posssible_degree:
    mse_train, mse_test = kfold_cross_validation(
        X, Y, degree=degree, method='piv', K=10)
    test_err.append(mse_test), train_err.append(mse_train)


# %%
fig, axes = plt.subplots()
plt.plot(posssible_degree, np.log10(test_err), label='Cross-validation Error')
plt.plot(posssible_degree, np.log10(train_err), label='Training error')
plt.ylabel("$\log_{10}{E_{RMS}}$"), plt.ylabel("degree")
plt.legend()
plt.title("Cross validation error without regularization")
plt.tight_layout()

# %% [markdown]
# #### Observations:
# 1. The points 20 are too low in number thus we *donot* get the an expected $U$ shaped curved depicting decrese in $(bias)^2$ and increase in $variance$ ,giving an sweet-spot in between.
# 2. The error is contributed from following
#     1. test variance (= variance due to test sample size) : Since the test points are too low in our model thus the test error itsef have a high variance.
#     2. **Model instability** variance due to training sample size
#     3. **testing bias**
# %% [markdown]
# Thus we see at lower degree of polynomial the cross validation is dominated by bias in model ,but due to so low count of data points the bias does-not dominate that well over the data , hence the varince starts dominating soon.
#
# At $degree =9$ there is sudden dip in the bias , hence we get a see spot where the total bias and variance sum is minimized.
# For $degree\geq 10$ , the varince shoot up
#
# ### Hence polynomial of $degree =9$ fits our data well (low train and test error)
#
# ### region ($degree<9$ high training error -> Underfitting)
# ### region ($degree \geq 10 $ high testing error -> overfitting)
# %% [markdown]
# ## Estimate the noise variance

# %%


# %% [markdown]
# # Introduce the regularization (lmda) (Bayesian)(posterior)
# %% [markdown]
# Introducing paramters $\lambda$ to prevent overfitting case , i.e prevent our $w$ to take abrupt values
# $$E(w,\lambda)_trainingError = \sum{(h(x_i)-t_i)^2} + \lambda (||w||)^2$$
#
# We intriduce a regularized error to our model
# %% [markdown]
# # Noise variance estimation

# %%
X, Y = X[:20], Y[:20]
model = polyfit(degree=11, method='piv')

model.fit(X, Y)
score = model.score(X, Y)
variance = 1/score
variance


# %%
beta = []
for degree in np.arange(1, 14, 1):
    model = polyfit(degree=degree, method='piv')
    model.fit(X, Y)
    beta.append(X.shape[0]/model.train_error)


# %%
plt.plot(np.arange(1, 14, 1), beta, '-x')
plt.title('beta (Maximum Likelihood)')
plt.ylabel("beta"), plt.xticks(np.arange(1, 14, 1))
plt.xlabel('degree of poynomial')


# %%
X, Y = X[:20], Y[:20]
model = polyfit(method='piv', degree=9)
model.fit(X, Y)
hx = np.ravel(model.pred(X))
noise = hx-np.ravel(Y)


# %%


# %%
plt.plot(X, hx)
plt.plot(X, Y)
plt.title("Noise visualization")
plt.legend(['Hypothesis degree=9', 'Actual curve'])


# %%
print(np.mean(noise))
sbn.histplot(noise)


# %%
variance = 1/beta[8]
np.sqrt(variance)

# %% [markdown]
# Guaussain distribution with mean=0 and variance = 1/beta(poly=9) should look similar to our distribution

# %%
fig, ax = plt.subplots(figsize=(10, 5))

distrib = (np.random.normal(0, np.sqrt(variance), size=100000))
sbn.kdeplot(distrib, ax=ax, label="Gaussian Distrib (mean= 0,std=0.03558)")
sbn.kdeplot(noise, ax=ax, label="True Noise distrib")
ax.legend()
ax.set_title("Noise Distribution")
plt.tight_layout()

# %% [markdown]
# $gaussianNoise = N(0,(0.03558)^{2})$
# %% [markdown]
# # Hyper Parameter Tuning (regularization approach)

# %%

df = pd.DataFrame(columns=['log10(lmda)', 'degree', 'test_err', 'train_err'])
df.head()


# %%
df = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
df.head()
posssible_degree = np.arange(7, 15, 1)
lmda = [10**i for i in np.arange(1.0, -5.0, -0.2)]
for lm in lmda:
    for degree in posssible_degree:
        train_err, test_err = kfold_cross_validation(
            X, Y, degree=degree, K=10, lmda=lm, method='piv')
        df.loc[-1] = {'lmda': np.log10(lm), 'degree': degree,
                      'test_err': test_err, 'train_err': train_err}
        df.index = df.index + 1
        df = df.sort_index()


# %%
df.sort_values('test_err')


# %%
# NOISE ESTIMATION
model = polyfit(method='piv', degree=14, lmda=10**(-4.6))
model.fit(X, Y)
hx = np.ravel(model.pred(X))
noise = hx-np.ravel(Y)
std = np.sqrt(model.score(X, Y)/X.shape[0])
print(np.ravel(model.w))


# %%
sbn.histplot(noise)
print(std)


# %%
fig, ax = plt.subplots(figsize=(10, 5))

distrib = (np.random.normal(0, std, size=100000))
sbn.kdeplot(distrib, ax=ax,
            label="Gaussian Distrib (mean= 0,std=0.0399403)")
sbn.kdeplot(noise, ax=ax, label="True Noise distrib")
ax.legend()
ax.set_title("Noise Distribution")
plt.tight_layout()


# %%


# %%


# %% [markdown]
# # Repeat for 100 points dataset

# %%
X, Y = data[:, 0], data[:, 1]
X, Y = zip(*sorted(zip(X, Y)))
X, Y = np.array(X), np.array(Y)
print(X.shape)

# %% [markdown]
# ## Error Minimization on given number of points
#
#
# ### (a) Optimization using **Batch Gradient descent** on given data
#
# Notice that convergence of gradient descent algorithm is quite slow , took around 50,000 iterations on an average to converge.

# %%
degrees = np.arange(5, 15)
models = []
for degree in degrees:
    model = polyfit(degree=degree, max_iter=20000, steps=5e-4)
    model.fit(X, Y)
    models.append(model)
    print(degree, "done")


# %%
# After training lets plot all the curves
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 5))
axes = np.array(axes)
# Plotting curves
for i, ax in zip((5, 8, 9, 12), np.reshape(axes, -1)):
    ax.plot(X, Y, label='Actual Distrib', color='red')
    ax.plot((X), models[i-5].pred(X))
    ax.legend(['Actual Distribution', 'degree='+str(i)]),
plt.show()


# %%
grad_error = [model.train_error for model in models]
plt.figure(figsize=(15, 5))
plt.plot(np.arange(0, len(grad_error)), np.log10(grad_error), '-o')
plt.title("Gradient Descent: Training error vs Degree"), plt.xlabel(
    "Degree of polynomial"), plt.ylabel("E(W) training error")
plt.show()

# %% [markdown]
# #### Observations:
# 1. **Starting degree 8 onwards** the training error is reduced drastically , hence  $degree\geq 8 $ are good fit to our data set
# 2. Training error reduces as the degree of poynomial is increased ,but this occurs at the expense of the increase complexity of model.
# %% [markdown]
# ### **Variation of error vs iteration  of gradient descent**
#
# From below we can say the gradient descent algorithm converges very fast for the first 5000 iteration , after that the convergence of gradient descent is very slow in successive iteration.
# Here 1 iteration corresponds to 1 pass over the data set , hence in our case 1 iteration implies 1 pass over all 20 points.
#
# Error is monotonically decresing as number of iterations increases
#

# %%
start, gaps, max_iter = 100, 1000, 20000
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
axes = np.array(axes)

for degree, ax in zip([8, 9, 10, 11], axes.reshape(-1)):
    iters = np.arange(start, max_iter, gaps)
    modeli = polyfit(degree=degree, max_iter=max_iter, steps=1e-3)
    modeli.fit(X, Y, error_gaps=gaps)
    iters = [it[0] for it in modeli.train_error_gapped[1:]]
    iter_errors = [it[1] for it in modeli.train_error_gapped[1:]]
    # print(iter_errors)
    ax.plot(iters, np.log10(iter_errors), '-o', color='red')
    ax.set_xlabel('Iterations'), ax.set_ylabel(
        'LOG10(Least squared Error)'), ax.legend(['Degree = '+str(degree)])
    print(degree, "Done")

# %% [markdown]
# ### Variation of Error $E(W)$ vs batch size in gradient descent
#
# <span class="mark">Since the data set is quite small i.e only 20 points</span> . Also the data was normalized before the grad desccent thus the <span class="mark">batch gradient descent performed similar</span> on all batches

# %%
error_batches = []
max_iter = 1000
for batch_size in np.arange(1, 20, 2):
    model_b = polyfit(degree=5, batch_size=batch_size,
                      max_iter=max_iter, method='grad')
    model_b.fit(X, Y, error_gaps=10)
    error_batches.append(model_b.train_error_gapped)


# %%
plt.figure(figsize=(15, 5))
for eb in error_batches:
    x = [coord[0] for coord in eb]
    y = [coord[1] for coord in eb]
    plt.plot(x, y)
#     plt.ylim(0,10)
#     plt.xlim(0,100)

# %% [markdown]
# ### (b) Optimization using **PIV** (penrose inverse matrix)
#
# Below is polynomial fitting plot of various degree of polynomials optimized using PIV method .
#
# **Observations:**
# Starting $degree \geq 15$ the PIV fails , thus gives poor fitting , as shown below.This is the reason we cannot rely on PIV method for fitting higher degree poynomials on our data set.

# %%
degrees = np.arange(0, 20)
piv_models = []
for degree in degrees:
    model = polyfit(degree=degree, method='piv')
    model.fit(X, Y)
    piv_models.append(model)


# %%
# After training lets plot all the curves
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
axes = np.array(axes)

# Plotting curves

for i, ax in zip([0, 8, 9, 14, 16, 17], np.reshape(axes, -1)):
    ax.scatter(X, Y, label='Actual Distrib', color='red')
    ax.plot(np.sort(X), piv_models[i].pred(
        np.sort(X)), label='degree ='+str(piv_models[i].degree))
    ax.legend(['degree ='+str(piv_models[i].degree)])
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Error vs degree of poynomial
#
# Since PIV fails for degree>=17 hence the error shoots up as PIV method fails.
#
# Degree=9 polynomial fits our curve nicely.

# %%
piv_error = [piv_model.train_error for piv_model in piv_models[:16]]
plt.figure(figsize=(8, 5))
plt.plot(np.arange(0, len(piv_error)), piv_error, '-o')

plt.title("PIV Curve Fitting: Error vs Degree"), plt.xlabel(
    "Degree of polynomial"), plt.ylabel("E(W) training error")
plt.show()

# %% [markdown]
# ## Good fit of polynomial (Underfitting , overfitting)(Maximum Likelihood) (without regularization)

# %%
test_err, train_err = [], []
posssible_degree = np.arange(0, 18, 1)
for degree in posssible_degree:
    mse_train, mse_test = kfold_cross_validation(
        X, Y, degree=degree, method='piv', K=10)
    test_err.append(mse_test), train_err.append(mse_train)


# %%
ig, axes = plt.subplots()
plt.plot(posssible_degree, np.log10(test_err), label='Cross-validation Error')
plt.plot(posssible_degree, np.log10(train_err), label='Training error')
plt.ylabel("$\log_{10}{E_{RMS}}$"), plt.xlabel("degree")
plt.xticks(np.arange(1, 18, 1))
plt.legend()
plt.title("Cross validation error without regularization")
plt.tight_layout()

# %% [markdown]
# ## Estimate the noise variance

# %%
model = polyfit(degree=9, method='piv')
model.fit(X, Y)
hx = np.ravel(model.pred(X))
noise = hx-np.ravel(Y)
std = np.sqrt(model.score(X, Y)/X.shape[0])
std


# %%
plt.plot(X, Y, label="Actual Distribution")
plt.plot(X, hx, label="Hypothesis")
plt.legend()


# %%
sbn.histplot(noise)


# %%
fig, ax = plt.subplots(figsize=(10, 5))

distrib = (np.random.normal(0, std, size=100000))
sbn.kdeplot(distrib, ax=ax,
            label="Gaussian Distrib (mean= 0,std=0.051147)")
sbn.kdeplot(noise, ax=ax, label="True Noise distrib")
ax.legend()
ax.set_title("Noise Distribution")
plt.tight_layout()

# %% [markdown]
# # Introduce the regularization (lmda)(bayesian approach)
# %% [markdown]
# Introducing paramters $\lambda$ to prevent overfitting case , i.e prevent our $w$ to take abrupt values
# $$E(w,\lambda)_trainingError = \sum{(h(x_i)-t_i)^2} + \lambda (||w||)^2$$
#
# We intriduce a regularized error to our model
# %% [markdown]
# # Hyper Parameter Tuning (regularization approach)

# %%
X.shape


# %%
df = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
df.head()


# %%
df = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
df.head()
posssible_degree = np.arange(7, 15, 1)
lmda = [10**i for i in range(-1, -20, -1)]
for lm in lmda:
    for degree in posssible_degree:
        train_err, test_err = kfold_cross_validation(
            X, Y, degree=degree, K=10, lmda=lm, method='piv')
        df.loc[-1] = {'lmda': lm, 'degree': degree,
                      'test_err': test_err, 'train_err': train_err}
        df.index = df.index + 1
        df = df.sort_index()


# %%
df.sort_values('test_err')


# %%
model = polyfit(degree=9, method='piv', lmda=1e-17)
model.fit(X, Y)
hx = np.ravel(model.pred(X))
noise = hx-np.ravel(Y)
std = np.sqrt(model.score(X, Y)/X.shape[0])
std
print(np.ravel(model.w))


# %%
fig, ax = plt.subplots(figsize=(10, 5))

distrib = (np.random.normal(0, std, size=100000))
sbn.kdeplot(distrib, ax=ax,
            label="Gaussian Distrib (mean= 0,std=0.051147)")
sbn.kdeplot(noise, ax=ax, label="True Noise distrib")
ax.legend()
ax.set_title("Noise Distribution")
plt.tight_layout()


# %%
