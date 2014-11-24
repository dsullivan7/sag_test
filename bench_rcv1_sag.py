import numpy as np
from scipy import io

import matplotlib.pyplot as plt
import math

from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.linear_model import (SGDClassifier, SAGClassifier,
                                  SGDRegressor, SAGRegressor)

data = io.loadmat('rcv1_train.binary.mat')
# data = io.loadmat('covtype.libsvm.binary.mat')

X, y = data['X'], data['y'].ravel()
X, y = shuffle(X, y)
# shuffle to balance the data
# rng = np.random.RandomState(42)
# order = np.argsort(rng.randn(len(X)))
# X, y = X[order], y[order]

# subsample so it's fast
# X, y = X[:20000].copy(), y[:20000].copy()

# cast for sklearn
X = X.astype(np.float64)
y = y.astype(np.int)

# make it -1 and 1 for tracking log loss
# y[y == 1] = -1
# y[y == 2] = 1

# Split data
n_samples, n_features = X.shape
# X_train, y_train, X_test, y_test = \
#     X[:n_samples // 2], y[:n_samples // 2], \
#     X[n_samples // 2:], y[n_samples // 2:]
X_train, y_train, X_test, y_test = X, y, X, y

alpha = .0000001
pobj = []

n_iter_range = list(range(1, 100, 5))
clfs = [
    ("SGDClassifier", SGDClassifier(eta0=.05, alpha=alpha, loss='log', learning_rate='constant'), [], []),
    ("ASGDClassifier", SGDClassifier(eta0=.05, alpha=alpha, loss='log', learning_rate='constant', average=True), [], []),
    ("SAGClassifier", SAGClassifier(eta0='auto', alpha=alpha), [], []),
    ]
plt.close('all')

for name, clf, pobj, score in clfs:
    for n_iter in n_iter_range:
        clf = clone(clf)
        clf.set_params(n_iter=n_iter, random_state=42)
        clf.fit(X_train, y_train)
        w = clf.coef_.ravel()
        this_pobj = np.mean(np.log(1. + np.exp(-y_train * (X_train.dot(w) +
                                                           clf.intercept_))))
        this_pobj += alpha * np.dot(w, w) / 2.
        pobj.append(math.log(this_pobj))
        score.append(clf.score(X_test, y_test))

        print(name + " %1.6f %f" % (clf.score(X_test, y_test), this_pobj))

    print("")
    plt.plot(n_iter_range, pobj, label=name)

plt.legend(loc="upper right")
plt.xlabel("n_iter")
plt.ylabel("pobj")
plt.show()
