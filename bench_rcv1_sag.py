import numpy as np
from scipy import io

import matplotlib.pyplot as plt
import math
import time
import pure_sag
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.linear_model import (SGDClassifier, SAGClassifier,
                                  SGDRegressor, SAGRegressor)

# data = io.loadmat('rcv1_train.binary.mat')
data = io.loadmat('covtype.libsvm.binary.mat')

X, y = data['X'], data['y'].ravel()
X, y = shuffle(X, y, random_state=42)

# subsample so it's fast
X, y = X[:1000].copy(), y[:1000].copy()

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
# X_train, y_train, X_test, y_test = \
#     X[:training_num], y[:training_num], \
#     X[training_num:], y[training_num:]
X_train, y_train, X_test, y_test = X, y, X, y

# alpha = .0000001
# eta = 4.0

alpha = .01
eta = .00000004
pobj = []

# n_iter_range = list(range(1, 100, 5))
tol_range = [.01, .001, .0001, .00001]
log_tols = np.log10(tol_range)


clfs = [
    # ("SGDClassifier", SGDClassifier(eta0=eta, alpha=alpha, loss='log',
    #  learning_rate='constant'), [], [], []),
    # ("ASGDClassifier", SGDClassifier(eta0=eta, alpha=alpha, loss='log',
    #  learning_rate='constant', average=True), [], [], []),
    ("SAGClassifier", SAGClassifier(eta0='auto', alpha=alpha, random_state=42,
                                    max_iter=100000, verbose=True), [], [], []),
    ]
plt.close('all')

def get_pobj(clf):
    w = clf.coef_.ravel()
    p = np.mean(np.log(1. + np.exp(-y_train * (X_train.dot(w) +
                                               clf.intercept_))))
    p += alpha * np.dot(w, w) / 2.
    return p

# print('computing pobj optimal')
# pobj_opt = get_pobj(SAGClassifier(eta0='auto',
#                                   alpha=alpha,
#                                   tol=.0000099,
#                                   max_iter=100000,
#                                   verbose=True).fit(X_train, y_train))
# print('done !', pobj_opt)
# pobj_opt = 0.0

for name, clf, pobj, score, seconds in clfs:
    for i, tol in enumerate(tol_range):
        print("tol:", tol)
        clf = clone(clf)
        clf.set_params(tol=tol, random_state=42)
        scores = cross_validation.cross_val_score(clf, X, y, cv=4)
        print("scores:", scores)
        print("std:", scores.std())
        print("")
        print("")
        score.append(scores.mean() + scores.std())

    print("")
for name, clf, pobj, score, seconds in clfs:
    plt.plot(log_tols, score, label=name)
    plt.legend(loc="lower right")
    plt.xlabel("log10(tol)")
    plt.ylabel("mean cv 4 score + std")
plt.show()
plt.close('all')
