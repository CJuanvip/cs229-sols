import numpy as np
import pandas as pd


def test(model, df_test):
    # Make y be a vector of +/-1 labels and X be a {0, 1} matrix.
    Xtest = 1 * (df_test.as_matrix()[:,1:] > 0)
    # Assume svm_train.py has just been executed, and the model trained
    # by your classifier is in memory through that execution. You can also assume 
    # that the columns in the test set are arranged in exactly the same way as for the
    # training set (i.e., the j-th column represents the same token in the test data 
    # matrix as in the original training data matrix).

    # Write code below to classify each document in the test set (ie, each row
    # in the current document word matrix) as 1 for SPAM and 0 for NON-SPAM.

    # Note that the predict function for LIBLINEAR uses the sparse matrix 
    # representation of the document word  matrix, which is stored in sparseTestMatrix.
    # Additionally, it expects the labels to be dimension (numTestDocs x 1).

    # Construct the (num_test_docs x 1) vector 'predictions' such that the i-th
    # entry of this vector is positive if the predicted class is 1 and negative if
    # the predicted class is -1 for the i-th email (i-th row in Xtest) in the test
    # set.
    return model.classify(Xtest)


def compute_error(y, predicted_y):
    """
    Compute the empirical error on the test set.
    """
    num_test_docs = y.shape[0]

    return np.sum(y * predicted_y <= 0) / num_test_docs
