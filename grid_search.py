import numpy as np
from scipy import io

import pure_sag
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.linear_model import (SGDClassifier, SAGClassifier,
                                  SGDRegressor, SAGRegressor)

# data = io.loadmat('rcv1_train.binary.mat')
data = io.loadmat('covtype.libsvm.binary.mat')

X, y = data['X'], data['y'].ravel()
X, y = shuffle(X, y, random_state=42)

# subsample so it's fast
X, y = X[:20000].copy(), y[:20000].copy()

# cast for sklearn
X = X.astype(np.float64)
y = y.astype(np.int)

# make it -1 and 1 for tracking log loss
y[y == 1] = -1
y[y == 2] = 1

X_train, y_train, X_test, y_test = X, y, X, y

param_grid = [
    {'alpha': [.1, .01, .001],
     'n_iter': [50, 100, 200, 300]},
    ]

scores = ['accuracy']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SAGClassifier(eta0='auto', random_state=42),
                       param_grid, cv=2, scoring=score, verbose=True, n_jobs=2)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
