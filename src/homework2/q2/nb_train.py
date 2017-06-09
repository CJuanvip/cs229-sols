import numpy as np
import pandas as pd


def train(df_train):
    X = df_train.iloc[:, 1:].as_matrix()
    y = df_train.iloc[:, 0].as_matrix()
    tokenlist = df_train.columns
    classifier = NaiveBayes(X, y, tokenlist)

    return classifier


class NaiveBayes:
    def __init__(self, X, y, tokenlist):
        m = y.shape[0]
        word_count = X.shape[1]
        absV    = word_count
        denom_0 = np.sum(X[y == 0], axis=(0,1)) + absV
        denom_1 = np.sum(X[y == 1], axis=(0,1)) + absV

        dfp = np.zeros((2, word_count))
        # Compute the probability that a training email is spam.
        prior_0 = np.sum(y == 0) / m
        prior_1 = np.sum(y == 1) / m

        # Compute the conditional probabilities.
        for k in range(word_count):
            numer_k_0 = np.sum(X[y == 0][:, k]) + 1
            numer_k_1 = np.sum(X[y == 1][:, k]) + 1
            phi_k_0 = numer_k_0 / denom_0
            phi_k_1 = numer_k_1 / denom_1
            dfp[0, k] = phi_k_0
            dfp[1, k] = phi_k_1

        self.prior_0 = prior_0
        self.prior_1 = prior_1
        self.dfp = dfp
        self.tokenlist = tokenlist


    def classify(self, x):
        """
        Given a preprocessed email, classify it as SPAM or NOT SPAM.
        """
        # TODO: Convert probabilities to logarithms to prevent numerical underflows.
        prob_y_equals_0 = self.prior_0
        prob_y_equals_1 = self.prior_1

        probs_x_given_0 = self.dfp[0, np.where(x != 0)]
        probs_x_given_0 = probs_x_given_0.reshape(probs_x_given_0.shape[1])
        probs_x_given_1 = self.dfp[1, np.where(x != 0)]
        probs_x_given_1 = probs_x_given_1.reshape(probs_x_given_1.shape[1])
        
        probs_0 = np.power(prob_y_equals_0 * probs_x_given_0, x[x != 0])
        probs_1 = np.power(prob_y_equals_1 * probs_x_given_1, x[x != 0])

        numer_0 = np.product(probs_0)
        numer_1 = np.product(probs_1)
        denom = numer_0 + numer_1

        py_given_x = np.zeros(2)
        # SPAM given x
        py_given_x[1] = numer_1 / denom
        # NOT SPAM given x
        py_given_x[0] = numer_0 / denom
        
        return np.argmax(py_given_x)


    def to_dataframe(self):
        df = pd.DataFrame(np.zeros((self.dfp.shape[0], self.dfp.shape[1]+1)), columns=self.tokenlist)
        df.iloc[0,0] = self.prior_0
        df.iloc[1,0] = self.prior_1
        df.iloc[:, 1:] = self.dfp

        return df

    def __str__(self):
        return str(self.to_dataframe())
