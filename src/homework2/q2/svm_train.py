import numpy as np
import pandas as pd


def train(df_train, tau, max_iters):
    Xtrain = df_train.iloc[:, 1:].as_matrix()
    ytrain = df_train.iloc[:, 0].as_matrix()
    classifier = SVM(Xtrain, ytrain, tau, max_iters)

    return classifier


class SVM:
    def __init__(self, X, y, tau, max_iters):
        """
        Xtrain is a (num_train_docs x num_tokens) sparse matrix.
        Each row represents a unique document (email).
        The j-th column of the row $i$ represents if the j-th token appears in
        email i.
        
        tokenlist is a long string containing the list of all tokens (words).
        These tokens are easily known by position in the file TOKENS_LIST
        
        trainCategory is a (1 x numTrainDocs) vector containing the true 
        classifications for the documents just read in. The i-th entry gives the 
        correct class for the i-th email (which corresponds to the i-th row in 
        the document word matrix).
        
        Spam documents are indicated as class 1, and non-spam as class 0.
        For the SVM, we convert these to +1 and -1 to form the numTrainDocs x 1
        vector ytrain.
        
        This vector should be output by this method
        average_alpha = np.zeros(num_train_docs)
        """
        # Make y be a vector of +/-1 labels and X be a {0, 1} matrix.
        Xtrain = 1 * (X > 0)
        ytrain = (2 * y - 1).T

        num_train_docs = Xtrain.shape[0]
        num_tokens     = Xtrain.shape[1]
        Xtrain_squared = np.sum(Xtrain * Xtrain, axis=1).reshape((num_train_docs, 1))
        gram_train     = np.dot(Xtrain, Xtrain.T)
        # Vectorized Ktrain.
        Ktrain = np.exp(-(np.tile(Xtrain_squared,   (1, num_train_docs))
                        + np.tile(Xtrain_squared.T, (num_train_docs, 1))
                        - 2 * gram_train) / (2 * np.power(tau, 2)))

        # lambda
        lam = 1 / (64 * num_train_docs)
        alpha = np.zeros(num_train_docs)
        average_alpha = np.zeros(num_train_docs)

        t = 0
        for _ in range(max_iters * num_train_docs):
            t += 1
            idx = np.random.randint(num_train_docs)
            # margin = ytrain[idx] * Ktrain.T[:, idx] * alpha
            margin = ytrain[idx] * Ktrain[idx, :].dot(alpha)
            grad = -((margin < 1) * ytrain[idx] * Ktrain[:, idx]) + \
                   num_train_docs * lam * (Ktrain[:, idx] * alpha[idx])
            eta = 1.0 / np.sqrt(t)
            alpha = alpha - eta * grad
            average_alpha = average_alpha + alpha

        average_alpha = average_alpha / (max_iters * num_train_docs)

        self.Xtrain = Xtrain
        self.yTrain = ytrain
        self.Ktrain = Ktrain
        self.alpha  = average_alpha
        self.tau    = tau
        self.lam    = lam


    def classify(self, X):
        """
        Classify a matrix of testing data.
        """
        num_test_docs  = X.shape[0]
        num_train_docs = self.Xtrain.shape[0]
        alpha  = self.alpha
        tau    = self.tau
        Xtrain = self.Xtrain
        Xtest  = 1 * (X > 0)
        Xtest_squared  = np.sum(Xtest  * Xtest,  axis=1).reshape((num_test_docs, 1))
        Xtrain_squared = np.sum(Xtrain * Xtrain, axis=1).reshape((num_train_docs, 1))
        gram_test = np.dot(Xtest, Xtrain.T)
        # Vectorized Ktest.
        Ktest = np.exp(-(np.tile(Xtest_squared,   (1, num_train_docs))
                       + np.tile(Xtrain_squared.T, (num_test_docs, 1))
                       - 2 * gram_test) / (2 * np.power(tau, 2)))

        predictions = Ktest.dot(alpha)
        predictions = 2 * (predictions > 0) - 1

        return predictions
