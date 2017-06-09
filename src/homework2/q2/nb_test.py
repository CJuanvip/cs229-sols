import numpy as np
import pandas as pd


def test(model, df_test):
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
    num_test_docs = df_test.shape[0]
    output = np.zeros(num_test_docs)
    X = df_test.iloc[:, 1:].as_matrix()

    for i in range(num_test_docs):
        output[i] = model.classify(X[i])

    return output


def compute_error(y, guessed_y):
    """
    Compute the empirical error on the test set.
    """
    num_test_docs = y.shape[0]

    return np.sum(1*(guessed_y != y)) / num_test_docs


def k_most_indicative_words(k, dfp):
    """
    Compute the k most indicate spam words in our dictionary.
    """
    log_dfp = np.log(dfp.as_matrix())
    diff_log_dfp = log_dfp[1] - log_dfp[0]
    sorted_frame = np.argsort(diff_log_dfp)[::-1]
    words = dfp.columns[sorted_frame[:k]]

    return list(words)