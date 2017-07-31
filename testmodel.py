# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
sns.set()


def create_data(N=1000):
    Y = np.vstack([np.random.normal([3, 8], [1, 3], (N // 2, 2)),
                   np.random.normal([8, 3], [3, 1], (N - N // 2, 2))])
    X = np.vstack([0.1 * Y[:, 0] * Y[:, 1],
                   np.sin(Y[:, 0]) * np.cos(Y[:, 1]),
                   np.cos(Y[:, 0]) * np.sin(Y[:, 1])]).T
    X = np.random.normal(X, 1.0)
    return X, Y

np.random.seed(0)
X, Y = create_data(1000)

fig, ax = plt.subplots()
ax.plot(Y[:, 0], Y[:, 1], 'o', alpha=0.5)
ax.add_patch(plt.Rectangle((6, 6), 14, 14, color='yellow', alpha=0.2))
ax.set_xlim(0, 20); ax.set_ylim(0, 20)
ax.set_xlabel('$y_1$'); ax.set_ylabel('$y_2$');
plt.show()


from sklearn.cross_validation import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, train_size=0.5)


from sklearn.ensemble import RandomForestRegressor

clf1 = RandomForestRegressor(100).fit(Xtrain, Ytrain)
Ypred1 = clf1.predict(Xtest)

fig, ax = plt.subplots()
ax.plot(Ypred1[:, 0], Ypred1[:, 1], 'o', alpha=0.5)
ax.add_patch(plt.Rectangle((6, 6), 14, 14, color='yellow', alpha=0.2))
ax.set_xlim(2, 12); ax.set_ylim(2, 12)
ax.set_xlabel('$y_1$'); ax.set_ylabel('$y_2$');
plt.show()