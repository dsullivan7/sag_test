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


def sag(X, y, eta, alpha, n_iter=5, fit_intercept=True, dloss=log_dloss):
    n_samples, n_features = X.shape[0], X.shape[1]

    weights = np.zeros(n_features)
    sum_gradient = np.zeros(n_features)
    gradient_memory = np.zeros((n_samples, n_features))
    intercept = 0.0
    seen = set()

    rng = np.random.RandomState(42)

    for i in range(n_iter * n_samples):
        idx = int(rng.rand(1) * n_samples)
        # idx = i % n_samples
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

        if fit_intercept:
            intercept -= eta * gradient

    return weights, intercept


def sag_sparse(X, y, eta, alpha, n_iter=5, fit_intercept=True, dloss=log_dloss):
        n_samples, n_features = X.shape[0], X.shape[1]

        weights = np.zeros(n_features)
        sum_gradient = np.zeros(n_features)
        last_updated = np.zeros(n_features, dtype=np.int)
        gradient_memory = np.zeros((n_samples, n_features))
        intercept = 0.0
        wscale = 1.0
        seen = set()
        rng = np.random.RandomState(42)

        c_sum = np.zeros(n_iter * n_samples)

        counter = 0
        for k in range(n_iter):
            for i in range(n_samples):
                idx = int(rng.rand(1) * n_samples)
                entry = X[idx]
                seen.add(idx)

                if counter > 0:
                    for j in range(n_features):
                        if(last_updated[j] == 0):
                            weights[j] -= c_sum[counter - 1] * sum_gradient[j]
                        else:
                            weights[j] -= ((c_sum[counter - 1] -
                                            c_sum[last_updated[j] - 1]) *
                                           sum_gradient[j])
                        last_updated[j] = counter

                p = (wscale * np.dot(entry, weights)) + intercept
                gradient = dloss(p, y[idx])

                update = entry * gradient
                sum_gradient += update - gradient_memory[idx]
                gradient_memory[idx] = update

                wscale *= (1.0 - alpha * eta)
                if counter == 0:
                    c_sum[0] = eta / (wscale * len(seen))
                else:
                    c_sum[counter] = c_sum[counter - 1] + eta / (wscale *
                                                                 len(seen))

                if fit_intercept:
                    intercept -= (eta * gradient) + (alpha * intercept)
                counter += 1

        for k in range(n_features):
            weights[k] -= (c_sum[counter - 1] -
                           c_sum[last_updated[k] - 1]) * sum_gradient[k]
        weights *= wscale

        return weights, intercept
