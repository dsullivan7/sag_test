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
                                  SGDRegressor, SAGRegressor,
                                  LogisticRegression)

data = io.loadmat('rcv1_train.binary.mat')
# data = io.loadmat('covtype.libsvm.binary.mat')

X, y = data['X'], data['y'].ravel()
X, y = shuffle(X, y, random_state=42)

# subsample so it's fast
# X, y = X[:100000].copy(), y[:100000].copy()
# X, y = X[:10000].copy(), y[:10000].copy()
# X, y = X[:10000].copy(), y[:10000].copy()

# cast for sklearn
X = X.astype(np.float64)
y = y.astype(np.int)

# make it -1 and 1 for tracking log loss
# y[y == 1] = -1
# y[y == 2] = 1

# Split data
n_samples, n_features = X.shape
# training_percent = .7
# training_num = int(training_percent * n_samples)
# X_train, y_train, X_test, y_test = \
#     X[:training_num], y[:training_num], \
#     X[training_num:], y[training_num:]
X_train, y_train = X, y

# alpha = .0000001
# eta = 4.0

alpha = 1 / n_samples

iter_range = list(range(1, 20, 2))
# tol_range = [.01, .001, .0001, .00001]
# tol_range = [.01, .001, 0.0001]
# tol_range = [.01]
# log_tols = np.log10(tol_range)


clfs = [
    # ("SGDClassifier", SGDClassifier(eta0=eta, alpha=alpha, loss='log',
    #  learning_rate='constant'), [], [], [], []),
    ("LogisticRegression", LogisticRegression(C=1.0/alpha, tol=.0000000001), [], [], []),
    ("SAGClassifier", SAGClassifier(eta0='auto', tol=.0000000001, alpha=alpha,
                                    random_state=42,
                                    ), [], [], []),
    ]
plt.close('all')


def get_pobj(clf):
    w = clf.coef_.ravel()
    p = np.mean(np.log(1. + np.exp(-y_train * (X_train.dot(w) +
                                               clf.intercept_))))
    p += alpha * np.dot(w, w) / 2.
    return p

for name, clf, score, std, seconds in clfs:
    for i, itr in enumerate(iter_range):
        clf = clone(clf)
        clf.set_params(max_iter=itr, random_state=42)
        st = time.time()
        scores = cross_validation.cross_val_score(clf, X, y, cv=4)
        end = time.time()
        seconds.append(end - st)
        print("time for cv: %.8f seconds" % (end - st))
        print("score mean:", scores.mean())
        print("std:", scores.std())
        print("")
        print("")
        score.append(scores.mean())
        std.append(scores.std())

    print("")


for name, clf, score, std, seconds in clfs:
    plt.errorbar(seconds, score, std, label=name)
    plt.legend(loc="lower right")
    plt.xlabel("seconds")
    plt.ylabel("mean cv 4 score + std")
plt.show()
plt.close('all')
