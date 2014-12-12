import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_svmlight_file
from sklearn.base import clone
from sklearn.linear_model import (SAGClassifier, LogisticRegression)


def get_pobj(w, intercept, myX, myy):
    w = w.ravel()
    p = np.mean(np.log(1. + np.exp(-myy * (myX.dot(w) + intercept))))
    p += alpha * w.dot(w) / 2.
    return p

X, y = load_svmlight_file("rcv1.train.txt.gz")
n_samples, n_features = X.shape
X_test, y_test = load_svmlight_file("rcv1.test.txt.gz")
y = y.ravel()
y_test = y_test.ravel()

# cast for sklearn
X = X.astype(np.float64)
y = y.astype(np.int)
X_test = X_test.astype(np.float64)
y_test = y_test.astype(np.int)

alpha = 1.0 / n_samples
iter_range = list(range(1, 11, 2))

tol = 1.0e-8
clfs = [
    ("LogisticRegression",
     LogisticRegression(C=1.0 / (n_samples * alpha), tol=tol,
                        solver="newton-cg", fit_intercept=True, verbose=1),
     iter_range, [], [], []),
    ("SAGClassifier",
     SAGClassifier(eta0='auto', tol=tol, alpha=alpha,
                   random_state=42, fit_intercept=True),
     iter_range, [], [], []),
    ]
plt.close('all')

for name, clf, iter_range, seconds, scores, pobj in clfs:
    for i, itr in enumerate(iter_range):
        clf = clone(clf)
        clf.set_params(max_iter=itr, random_state=42)
        st = time.time()
        clf.fit(X, y)
        end = time.time()

        this_pobj = get_pobj(clf.coef_.ravel(), clf.intercept_, X, y)
        pobj.append(this_pobj)

        this_score = clf.score(X_test, y_test)
        scores.append(this_score)

        seconds.append(end - st)
        print("pobj: %.8f" % this_pobj)
        print("score: %.8f" % this_score)
        print("time for fit: %.8f seconds" % (end - st))
        print("")

    print("")

for name, clf, iter_range, seconds, scores, pobj in clfs:
    plt.plot(seconds, scores, label=name)
    plt.legend(loc="lower right")
    plt.xlabel("seconds")
    plt.ylabel("score")
plt.show()
plt.close('all')

for name, clf, iter_range, seconds, scores, pobj in clfs:
    plt.plot(seconds, pobj, label=name)
    plt.legend(loc="upper right")
    plt.xlabel("seconds")
    plt.ylabel("loss")
plt.show()
plt.close('all')
