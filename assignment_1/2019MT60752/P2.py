# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import seaborn as sbn
from cross_validation import *
import pandas as pd
from polyfit import *
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

# %%
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%
df = pd.read_csv('./train.csv', parse_dates=True, index_col='id')
df = df.sort_values(by='id')
df.reset_index(inplace=True)
df.head(3)


# %%
df['year'] = [d.year - 2000 for d in df.id]
df['month'] = [d.month-1 for d in df.id]
df.head()


# %%
x_y = np.array(df['year'])
x_m = np.array(df['month'])
T = np.array(df['value'])
x_t = np.array(((x_y)*12 + x_m), dtype=int)
df['time'] = x_t
df.head()


# %%
df.head()

# %% [markdown]
# ## Multi linear regression models

# %%
# month[i] represent the index of month_i in x3 labels of ith months
months = [[] for i in range(0, 12)]
for i in range(0, len(x_t)):
    months[(x_t[i] % 12)].append(i)
months


# %%

plt.figure(figsize=(15, 5))
for i in range(0, 12):

    plt.plot(x_t[np.array(months[i])], T[np.array(months[i])])
plt.show()


# %%
# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=80)
sbn.boxplot(x='year', y='value', data=df, ax=axes[0])
sbn.boxplot(x='month', y='value', data=df)

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18)
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
plt.show()

# %% [markdown]
# # Lets train the models for each month , since each month has a cyclic trend over the period of time
# %% [markdown]
#  Train all the models separately find best fit for each of the model

# %%
X = np.array(x_t)
T = np.reshape(T, (-1))
X[months[0]]

# %% [markdown]
# ## January

# %%
# Tuning the hyper-paramters -> degree of polynomial
# fig,axes = plt.subplots(nrows=)
df_jan = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
# df.head()
posssible_degree = np.arange(1, 4, 1)
lmda = [10**i for i in range(3, -5, -1)]
# lmda.append(0)
for lm in lmda:
    for degree in posssible_degree:
        # print(degree,lm)
        train_err, test_err = shuffled_cvr(
            X[months[0]], T[months[0]], degree=degree, K=4, lmda=lm, method='piv', n_permuatations=40)
        df_jan.loc[-1] = {'lmda': np.log10(lm), 'degree': degree,
                          'test_err': test_err, 'train_err': train_err}
        df_jan.index = df_jan.index + 1
        df_jan = df_jan.sort_index()

df_jan.sort_values('test_err')

# %% [markdown]
# # Feburary
#

# %%
df_feb = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
# df.head()
posssible_degree = np.arange(1, 6, 1)
lmda = [10**i for i in range(2, -5, -1)]
# lmda.append(0)
for lm in lmda:
    for degree in posssible_degree:
        # print(months[1], months[1])
        x_train = X[months[1]]
        y_train = T[months[1]]
        # print(x_train)
        train_err, test_err = shuffled_cvr(
            x_train, y_train, degree=degree, K=4, lmda=lm, method='piv')
        df_feb.loc[-1] = {'lmda': (lm), 'degree': degree,
                          'test_err': test_err, 'train_err': train_err}
        df_feb.index = df_feb.index + 1
        df_feb = df_feb.sort_index()

df_feb = df_feb.sort_values('test_err')

df_feb.head(10)

# %% [markdown]
# # March

# %%


# %%
df_march = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
# df.head()
posssible_degree = np.arange(1, 6, 1)
lmda = [10**i for i in range(3, -5, -1)]
# lmda.append(0)
for lm in lmda:
    for degree in posssible_degree:
        # print(months[1], months[1])
        x_train = X[months[2]]
        y_train = T[months[2]]
        # print(x_train)
        train_err, test_err = shuffled_cvr(
            x_train, y_train, degree=degree, K=4, lmda=lm, method='piv', n_permuatations=40)
        df_march.loc[-1] = {'lmda': (lm), 'degree': degree,
                            'test_err': test_err, 'train_err': train_err}
        df_march.index = df_march.index + 1
        df_march = df_march.sort_index()

df_march = df_march.sort_values('test_err')

df_march.head(10)

# %% [markdown]
# # April

# %%
dd = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
# df.head()
posssible_degree = np.arange(1, 6, 1)
lmda = [10**i for i in range(1, -5, -1)]
# lmda.append(0)
for lm in lmda:
    for degree in posssible_degree:
        # print(months[1], months[1])
        x_train = X[months[3]]
        y_train = T[months[3]]
        # print(x_train)
        train_err, test_err = shuffled_cvr(
            x_train, y_train, degree=degree, K=4, lmda=lm, method='piv', n_permuatations=40)
        dd.loc[-1] = {'lmda': (lm), 'degree': degree,
                      'test_err': test_err, 'train_err': train_err}
        dd.index = dd.index + 1
        dd = dd.sort_index()

dd = dd.sort_values('test_err')

dd.head(10)

# %% [markdown]
# # May

# %%
dd = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
# df.head()
posssible_degree = np.arange(1, 6, 1)
lmda = [10**i for i in range(2, -5, -1)]
# lmda.append(0)
for lm in lmda:
    for degree in posssible_degree:
        # print(months[1], months[1])
        x_train = X[months[4]]
        y_train = T[months[4]]
        # print(x_train)
        train_err, test_err = shuffled_cvr(
            x_train, y_train, degree=degree, K=5, lmda=lm, method='piv', n_permuatations=40)
        dd.loc[-1] = {'lmda': (lm), 'degree': degree,
                      'test_err': test_err, 'train_err': train_err}
        dd.index = dd.index + 1
        dd = dd.sort_index()

dd = dd.sort_values('test_err')

dd.head(20)

# %% [markdown]
# # June

# %%
dd = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
# df.head()
posssible_degree = np.arange(1, 6, 1)
lmda = [10**i for i in range(1, -5, -1)]
# lmda.append(0)
for lm in lmda:
    for degree in posssible_degree:
        # print(months[1], months[1])
        x_train = X[months[5]]
        y_train = T[months[5]]
        # print(x_train)
        train_err, test_err = shuffled_cvr(
            x_train, y_train, degree=degree, K=5, lmda=lm, method='piv', n_permuatations=40)
        dd.loc[-1] = {'lmda': (lm), 'degree': degree,
                      'test_err': test_err, 'train_err': train_err}
        dd.index = dd.index + 1
        dd = dd.sort_index()

dd = dd.sort_values('test_err')

dd.head(10)

# %% [markdown]
# # July

# %%
dd = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
# df.head()
posssible_degree = np.arange(1, 6, 1)
lmda = [10**i for i in range(1, -5, -1)]
# lmda.append(0)
for lm in lmda:
    for degree in posssible_degree:
        # print(months[1], months[1])
        x_train = X[months[6]]
        y_train = T[months[6]]
        # print(x_train)
        train_err, test_err = shuffled_cvr(
            x_train, y_train, degree=degree, K=4, lmda=lm, method='piv', n_permuatations=40)
        dd.loc[-1] = {'lmda': (lm), 'degree': degree,
                      'test_err': test_err, 'train_err': train_err}
        dd.index = dd.index + 1
        dd = dd.sort_index()

dd = dd.sort_values('test_err')

dd.head(10)

# %% [markdown]
# # August

# %%
dd = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
# df.head()
posssible_degree = np.arange(1, 6, 1)
lmda = [10**i for i in range(1, -5, -1)]
# lmda.append(0)
for lm in lmda:
    for degree in posssible_degree:
        # print(months[1], months[1])
        x_train = X[months[7]]
        y_train = T[months[7]]
        # print(x_train)
        train_err, test_err = shuffled_cvr(
            x_train, y_train, degree=degree, K=5, lmda=lm, method='piv', n_permuatations=40)
        dd.loc[-1] = {'lmda': (lm), 'degree': degree,
                      'test_err': test_err, 'train_err': train_err}
        dd.index = dd.index + 1
        dd = dd.sort_index()

dd = dd.sort_values('test_err')

dd.head(10)

# %% [markdown]
# # September

# %%
dd = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
# df.head()
posssible_degree = np.arange(1, 6, 1)
lmda = [10**i for i in range(4, -5, -1)]
# lmda.append(0)
for lm in lmda:
    for degree in posssible_degree:
        # print(months[1], months[1])
        x_train = X[months[8]]
        y_train = T[months[8]]
        # print(x_train)
        train_err, test_err = shuffled_cvr(
            x_train, y_train, degree=degree, K=5, lmda=lm, method='piv', n_permuatations=40)
        dd.loc[-1] = {'lmda': (lm), 'degree': degree,
                      'test_err': test_err, 'train_err': train_err}
        dd.index = dd.index + 1
        dd = dd.sort_index()

dd = dd.sort_values('test_err')

dd.head(10)

# %% [markdown]
# # October

# %%
dd = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
# df.head()
posssible_degree = np.arange(1, 6, 1)
lmda = [10**i for i in range(5, -8, -1)]
# lmda.append(0)
for lm in lmda:
    for degree in posssible_degree:
        # print(months[1], months[1])
        x_train = X[months[9]]
        y_train = T[months[9]]
        # print(x_train)
        train_err, test_err = shuffled_cvr(
            x_train, y_train, degree=degree, K=5, lmda=lm, method='piv', n_permuatations=40)
        dd.loc[-1] = {'lmda': (lm), 'degree': degree,
                      'test_err': test_err, 'train_err': train_err}
        dd.index = dd.index + 1
        dd = dd.sort_index()

dd = dd.sort_values('test_err')

dd.head(10)

# %% [markdown]
# # November

# %%
dd = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
# df.head()
posssible_degree = np.arange(1, 6, 1)
lmda = [10**i for i in range(5, -5, -1)]
# lmda.append(0)
for lm in lmda:
    for degree in posssible_degree:
        # print(months[1], months[1])
        x_train = X[months[10]]
        y_train = T[months[10]]
        # print(x_train)
        train_err, test_err = shuffled_cvr(
            x_train, y_train, degree=degree, K=5, lmda=lm, method='piv', n_permuatations=40)
        dd.loc[-1] = {'lmda': (lm), 'degree': degree,
                      'test_err': test_err, 'train_err': train_err}
        dd.index = dd.index + 1
        dd = dd.sort_index()

dd = dd.sort_values('test_err')

dd.head(10)

# %% [markdown]
# # December

# %%
dd = pd.DataFrame(columns=['lmda', 'degree', 'test_err', 'train_err'])
# df.head()
posssible_degree = np.arange(1, 6, 1)
lmda = [10**i for i in range(4, -2, -1)]
# lmda.append(0)
for lm in lmda:
    for degree in posssible_degree:
        # print(months[1], months[1])
        x_train = X[months[11]]
        y_train = T[months[11]]
        # print(x_train)
        train_err, test_err = shuffled_cvr(
            x_train, y_train, degree=degree, K=5, lmda=lm, method='piv', n_permuatations=10)
        dd.loc[-1] = {'lmda': (lm), 'degree': degree,
                      'test_err': test_err, 'train_err': train_err}
        dd.index = dd.index + 1
        dd = dd.sort_index()

dd = dd.sort_values('test_err')

dd.head(10)


# %%
month_degrees = [2, 1, 1, 2, 3, 1, 1, 1, 1, 2, 2, 1]
lmda = [1e1, 1e-1, 1e-1, 1e-1, 1e-3, 1e-1, 1e-1, 1e-3, 1e0, 1e4, 1e2, 1e3]


# %%
new_models = []
for i in range(0, 12):
    # train the ith month model by fitting a polyfit
    X_train = x_t[months[i]]
    t = T[months[i]]
    model = polyfit(degree=month_degrees[i],
                    lmda=np.float64(lmda[i]), method='piv')
    model.fit(X_train, t)
    new_models.append(model)


# %%
model_error = [model.score_rms(model.X, model.T) for model in new_models]
# rmse = np.sqrt(np.sum(model_error))/len(x_t)
print(model_error)
# rmse


# %%
df_test = pd.read_csv('./test.csv', parse_dates=True, index_col='id')

df_test.reset_index(inplace=True)
df_test.head(3)


# %%
df_test['year'] = [d.year - 2000 for d in df_test.id]
df_test['month'] = [d.month-1 for d in df_test.id]
df_test.head()


# %%
x_y_t = np.array(df_test['year'])
x_m_t = np.array(df_test['month'])

x_tt = np.array(((x_y_t)*12 + x_m_t), dtype=int)
df_test['time'] = x_tt
df_test.head()


# %%
x_tt


# %%
y_pred = []

for x_test in x_tt:
    k = x_test % 12
    y_pred.append(new_models[k].pred(x_test)[0][0])

y_pred


# %%
df_test['y_pred'] = y_pred

df_test.head(10)


# %%
print("id,value")
for d in df_test.values:
    print(str(d[0].month)+'/'+str(d[0].day) +
          '/'+str(d[0].year-2000)+','+str(d[4]))


# %%
