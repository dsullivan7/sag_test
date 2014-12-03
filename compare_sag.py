import numpy as np
from scipy import io

from sklearn.utils import shuffle
from sklearn.linear_model import (SGDClassifier, SAGClassifier,
                                  RidgeClassifier, LogisticRegression)

from memory_profiler import profile


@profile
def main():

    data = io.loadmat('rcv1_train.binary.mat')
    # data = io.loadmat('covtype.libsvm.binary.mat')
    X, y = data['X'], data['y'].ravel()
    X, y = shuffle(X, y, random_state=42)
    X = X.astype(np.float64)
    y = y.astype(np.int)

    # make it -1 and 1 for tracking log loss
    # y[y == 1] = -1
    # y[y == 2] = 1

    # clf = RidgeClassifier()
    clf = LogisticRegression()
    # clf = SGDClassifier()
    # clf = SGDClassifier(average=True)
    # clf = SAGClassifier()

    clf.fit(X, y)

if __name__ == "__main__":
    main()
