import numpy as np
import find_best_threshold as fbt


def stump_booster(X, y, T):
    """
    The function ``stump_booster`` Uses boosted decision stumps to train a classifier.
    It performs ``T`` rounds of boosted decision stumps to classify the data ``X``,
    which is an m-by-n matrix of m training examples in dimension n,
    to match ``y``.

    The returned parameters are ``theta``, the parameter vector in ``T`` dimensions,
    the ``feature_inds``, which are indices of the features (a ``T`` dimensional
    vector taking values in ``{1, 2, ..., n}``), and ``thresholds``, which are
    real-valued thresholds. The resulting classifier may be computed on an
    n-dimensional training example by
    ``
    theta' * sign(x(feature_inds) - thresholds).
    ``
    The resulting predictions may be computed simultaneously on an
    n-dimensional dataset, represented as an m-by-n matrix ``X``, by

    ``
    sign(X(:, feature_inds) - repmat(thresholds', m, 1)) * theta.
    ``

    This is an m-vector of the predicted margins.
    """
    rows, cols = X.shape
    p_dist = (1 / rows) * np.ones(rows)
    
    thetas = np.zeros(T)
    feature_indices = np.zeros(T, dtype='int')
    thresholds = np.zeros(T)

    for t in range(T):
        index_t, threshold_t = fbt.find_best_threshold(X, y, p_dist)
        feature_indices[t] = index_t
        thresholds[t] = threshold_t
        Wplus     = p_dist.T.dot(y * np.sign(X[:, index_t] - threshold_t) == 1)
        Wminus    = p_dist.T.dot(y * np.sign(X[:, index_t] - threshold_t) == -1)
        theta_t   = 0.5 * np.log(Wplus / Wminus)
        thetas[t] = theta_t
        thresholds_per_example = np.repeat(thresholds.T, rows).reshape((rows,T))
        p_dist    = np.exp(-y * (thetas.dot(np.sign(X[:, feature_indices] - thresholds_per_example).T)))
        p_dist    = p_dist / np.sum(p_dist)
    

    return (thetas, feature_indices, thresholds)
