import numpy as np


def find_best_threshold(X, y, p_dist):
    """
    Finds the best threshold for the given data.

    The function ``find_best_threshold`` returns a threshold
    ``thresh`` and index ``ind`` that gives the best thresholded classifier for the
    weights ``p_dist`` on the training data. That is, the returned index ``ind``
    and threshold ``thresh`` minimize

     ``sum_{i = 1}^m p(i) * 1{sign(X(i, ind) - thresh) ~= y(i)}``

    OR

     ``sum_{i = 1}^m p(i) * 1{sign(thresh - X(i, ind)) ~= y(i)}``.

    We must check both signed directions, as it is possible that the best
    decision stump (coordinate threshold classifier) is of the form
    ``sign(threshold - x_j)`` rather than ``sign(x_j - threshold)``.

    The data matrix ``X`` is of size m-by-n, where m is the training set size
    and n is the dimension.

    The implementation uses efficient sorting and data structures to perform
    this calculation in time ``O(n m log(m))``, where the size of the data matrix
    ``X`` is m-by-n.
    """

    rows, cols = X.shape
    ind = 0
    thresh = 0
    best_err = float('inf')

    for j in range(cols):
        inds = np.argsort(X[:,j])[::-1]
        x_sort = X[inds, j]
        y_sort = y[inds]
        p_sort = p_dist[inds]

        s = x_sort[0] + 1
        possible_thresholds = (x_sort + np.roll(x_sort, 1)) / 2
        possible_thresholds[0] = x_sort[0] + 1

        increments = np.roll(p_sort * y_sort, 1)
        increments[0] = 0

        empirical_errors = np.ones((rows, 1)).T.dot((p_sort.T * (y_sort == 1)))
        empirical_errors = empirical_errors - np.cumsum(increments)

        thresh_index = np.argmin(empirical_errors)
        best_low = empirical_errors[thresh_index]

        thresh_high = np.argmax(empirical_errors)
        best_high = empirical_errors[thresh_high]

        best_high = 1 - best_high
        best_err_j = min(best_high, best_low)

        if best_high < best_low:
            thresh_index = thresh_high

        if best_err_j < best_err:
            ind = j
            thresh = possible_thresholds[thresh_index]
            best_err = best_err_j

    return (ind, thresh)
