import numpy as np
import find_best_threshold as fbt


def random_booster(X, y, T):
    rows, cols = X.shape
    p_dist = (1 / rows) * np.ones(rows)

    thetas = np.zeros(T)
    feature_indices = np.zeros(T, dtype='int')
    thresholds = np.zeros(T)

    for t in range(T):
        index_t = int(np.floor(cols * np.random.random()))
        threshold_t = X[int(np.floor(rows * np.random.random())), index_t] + 1e-8 * np.random.random()
        Wplus     = p_dist.T.dot(y * np.sign(X[:, index_t] - threshold_t) == 1)
        Wminus    = p_dist.T.dot(y * np.sign(X[:, index_t] - threshold_t) == -1)
        theta_t   = 0.5 * np.log(Wplus / Wminus)
        thetas[t]  = theta_t
        feature_indices[t] = index_t
        thresholds[t] = threshold_t

        thresholds_per_example = np.repeat(thresholds.T, rows).reshape((rows,T))
        p_dist    = np.exp(-y * (thetas.dot(np.sign(X[:, feature_indices] - thresholds_per_example).T)))
        p_dist    = p_dist / np.sum(p_dist)

    return (thetas, feature_indices, thresholds)
