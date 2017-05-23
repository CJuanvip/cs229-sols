import numpy as np


def compute_errors_over_time(Xtrain, 
                             ytrain, 
                             Xtest, 
                             ytest, 
                             theta, 
                             feature_inds, 
                             thresholds):
    """
    The function ``plt_errors_over_time`` Plots train and test error from boosting.
    It plots the training and testing error of a decision-stump based boosting
    algorithm over iterations of the boosting algorithm.
    """
    num_thresholds = thresholds.shape[0]

    train_errors = np.zeros(num_thresholds)
    test_errors = np.zeros(num_thresholds)

    mtrain = Xtrain.shape[0]
    mtest = Xtest.shape[0]

    # Predicted margins for train and test
    train_predictions = np.zeros(mtrain)
    test_predictions = np.zeros(mtest)

    # Iteratively compute the margin predicted by the thresholded classifier,
    # updating both test and training predictions.
    for i in range(num_thresholds):
        import sys
        train_predictions = train_predictions + \
            theta[i] * np.sign(Xtrain[:, feature_inds[i]] - thresholds[i])
        test_predictions = test_predictions + \
            theta[i] * np.sign(Xtest[:, feature_inds[i]] - thresholds[i])

        train_errors[i] = (1 / mtrain) * np.sum((ytrain * train_predictions) <= 0)
        test_errors[i] = (1 / mtest) * np.sum((ytest * test_predictions) <= 0)

    return train_errors, test_errors
