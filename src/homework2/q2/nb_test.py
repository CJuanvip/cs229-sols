import numpy as np
import pandas as pd


def test(model, test_matrix):
    # Assume nb_train.m has just been executed, and all the parameters computed/needed
    # by your classifier are in memory through that execution. You can also assume 
    # that the columns in the test set are arranged in exactly the same way as for the
    # training set (i.e., the j-th column represents the same token in the test data 
    # matrix as in the original training data matrix).

    # Write code below to classify each document in the test set (ie, each row
    # in the current document word matrix) as 1 for SPAM and 0 for NON-SPAM.

    # Construct the (num_test_docs x 1) vector 'output' such that the i-th entry 
    # of this vector is the predicted class (1/0) for the i-th  email (i-th row 
    # in testMatrix) in the test set.    
    num_test_docs = test_matrix.shape[0]
    num_tokens = test_matrix.shape[1]
    output = np.zeros(num_test_docs)
    X = test_matrix.iloc[:, 1:].as_matrix()

    for i in range(num_test_docs):
        output[i] = model.classify(X[i, :])

    return output


def compute_error(y, guessed_y):
    """
    Compute the empirical error on the test set.
    """
    num_test_docs = y.shape[0]

    return np.sum(guessed_y != y) / num_test_docs
