import numpy as np
import math


# this is used for sag classification
def log_dloss(p, y):
    z = p * y
    # approximately equal and saves the computation of the log
    if z > 18.0:
        return math.exp(-z) * -y
    if z < -18.0:
        return -y
    return -y / (math.exp(z) + 1.0)


def sag(X, y, eta, alpha, n_iter=5,
        dloss=log_dloss):
    n_samples, n_features = X.shape[0], X.shape[1]

    weights = np.zeros(X.shape[1])
    sum_gradient = np.zeros(X.shape[1])
    gradient_memory = np.zeros((n_samples, n_features))
    intercept = 0.0
    seen = set()

    for i in range(n_iter * n_samples):
        # idx = int(rng.rand(1) * n_samples)
        idx = i % n_samples
        entry = X[idx]
        seen.add(idx)
        p = np.dot(entry, weights) + intercept
        gradient = dloss(p, y[idx])
        update = entry * gradient + alpha * weights

        # SAG
        sum_gradient += update - gradient_memory[idx]
        gradient_memory[idx] = update
        weights -= eta * sum_gradient / len(seen)

        # SGD
        # weights -= eta * update

        intercept -= eta * gradient

    return weights, intercept
