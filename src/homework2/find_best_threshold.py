import numpy as np


def find_best_threshold(X, y, p_dist):
    rows, cols = X.shape
    ind = 0
    thresh = 0
    best_err = float('inf')

    for j in range(cols):
        inds = np.argsort(X[:,j])[::-1]
        x_sort = X[inds, j]
        y_sort = y[inds]
        p_sort = p_sort[inds]

        s = x_sort[0] + 1
        possible_thresholds = (x_sort + np.roll(x_sort, 1)) / 2
        possible_thresholds[0] = x_sort[0] + 1

        increments = np.roll(p_sort * y_sort, 1)
        increments[0] = 0

        empirical_errors = np.ones((rows, 1)).dot(p_sort.T.dot(np.where(y_sort == 1)))
        empirical_errors = empirical_errors - np.cumsum(increments)

        thesh_index = np.argmin(empirical_errors)
        best_low = empirical_errors[thresh_index]

        thresh_high = np.argmax(empirical_errors)
        best_high = empirical_errors[thresh_high]

        best_high = 1 - best_high
        best_err_j = min(best_high, best_low)

        if best_high < best_low:
            thresh_index = thresh_high

        if best_err_j < best_err:
            ind = j
            thresh = possible_thresholds[thresh_ind]
            best_err = best_err_j

    return (ind, thresh)
