import numpy as np
from scipy import io

import matplotlib.pyplot as plt
import math
import time
import pure_sag
from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.linear_model import (SGDClassifier, SAGClassifier,
                                  SGDRegressor, SAGRegressor)

# data = io.loadmat('rcv1_train.binary.mat')
data = io.loadmat('covtype.libsvm.binary.mat')

X, y = data['X'], data['y'].ravel()
X, y = shuffle(X, y, random_state=42)

# subsample so it's fast
X, y = X[:100000].copy(), y[:100000].copy()

# cast for sklearn
X = X.astype(np.float64)
y = y.astype(np.int)

# make it -1 and 1 for tracking log loss
y[y == 1] = -1
y[y == 2] = 1

# Split data
n_samples, n_features = X.shape
training_percent = .7
training_num = int(training_percent * n_samples)
X_train, y_train, X_test, y_test = \
    X[:training_num], y[:training_num], \
    X[training_num:], y[training_num:]
# X_train, y_train, X_test, y_test = X, y, X, y

# alpha = .0000001
# eta = 4.0

alpha = .01
eta = .00000004
pobj = []

# n_iter_range = list(range(1, 50, 5))
n_iter_range = list(range(1, 100, 5))
# n_iter_range = list(range(1, 500, 50))


clfs = [
    ("SGDClassifier", SGDClassifier(eta0=eta, alpha=alpha, loss='log',
     learning_rate='constant'), [], [], [0]),
    ("ASGDClassifier", SGDClassifier(eta0=eta, alpha=alpha, loss='log',
     learning_rate='constant', average=True), [], [], [0]),
    ("SAGClassifier", SAGClassifier(eta0='auto', alpha=alpha, random_state=42), [], [], [0]),
    ]
plt.close('all')

def get_pobj(clf):
    w = clf.coef_.ravel()
    p = np.mean(np.log(1. + np.exp(-y_train * (X_train.dot(w) +
                                               clf.intercept_))))
    p += alpha * np.dot(w, w) / 2.
    return p

print('computing pobj optimal')
pobj_opt = get_pobj(SAGClassifier(eta0='auto',
                                  alpha=alpha,
                                  n_iter=100).fit(X_train, y_train))
print('done !', pobj_opt)
# pobj_opt = 0.0

for name, clf, pobj, score, seconds in clfs:
    for i, n_iter in enumerate(n_iter_range):
        clf = clone(clf)
        clf.set_params(n_iter=n_iter, random_state=42)
        t1 = time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        seconds.append(seconds[i] + t2 - t1)

        this_pobj = get_pobj(clf)

        pobj.append(math.log10(this_pobj - pobj_opt))
        score.append(clf.score(X_test, y_test))

        print(name + " %1.6f %1.16f" % (clf.score(X_test, y_test), this_pobj))

    print("")
    # plt.plot(n_iter_range, pobj, label=name)
    plt.plot(seconds[1:], pobj, label=name)
    # plt.plot(seconds[1:], score, label=name)
    # plt.plot(n_iter_range, score, label=name)

plt.legend(loc="upper right")
plt.xlabel("time (seconds)")
plt.ylabel("log10(pobj - pogj_opt)")
plt.show()
