import numpy as np
import find_best_threshold as fbt


def random_booster(X, y, T):
    """
    The function ``random_booster`` usses random thresholds and indices to train a 
    classifier. It performs ``T`` rounds of boosted decision stumps to classify 
    the data ``X``, which is an m-by-n matrix of m training examples in dimension n.
    
    The returned parameters are ``theta``, the parameter vector in ``T`` dimensions,
    the feature_inds, which are indices of the features (a ``T``-dimensional vector
    taking values in ``{1, 2, ..., n}``), and ``thresholds``, which are real-valued
    thresholds. The resulting classifier may be computed on an n-dimensional
    ``
    theta' * sgn(x(feature_inds) - thresholds)
    ``
    """
    rows, cols = X.shape
    p_dist = np.ones(rows)
    p_dist = p_dist / np.sum(p_dist)

    thetas = np.zeros(T)
    feature_indices = np.zeros(T, dtype='int')
    thresholds = np.zeros(T)

    # We use floor instead of ceil because we are indexing form zero, 
    # whereas MATLAB indexes from one.
    for t in range(T):
        index_t = int(np.floor(cols * np.random.random()))
        threshold_t = X[int(np.floor(rows * np.random.random())), index_t] + 1e-8 * np.random.random()
        Wplus   = p_dist.T.dot(y * np.sign(X[:, index_t] - threshold_t) == 1)
        Wminus  = p_dist.T.dot(y * np.sign(X[:, index_t] - threshold_t) == -1)
        theta_t = 0.5 * np.log(Wplus / Wminus)
        
        thetas[t] = theta_t
        feature_indices[t] = index_t
        thresholds[t] = threshold_t

        thresholds_per_example = np.repeat(thresholds[:(t+1)].T, rows).reshape((rows,t+1))
        p_dist = np.exp(-y * (thetas[:(t+1)].T.dot(np.sign(X[:, feature_indices[:(t+1)]] - thresholds_per_example).T)))
        p_dist = p_dist / np.sum(p_dist)


    return (thetas, feature_indices, thresholds)
