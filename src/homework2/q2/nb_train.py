import numpy as np
import pandas as pd


def train(df):
    classifier = NaiveBayes(df)

    return classifier


class NaiveBayes:
    def __init__(self, df):
        self.dfp = self.nb_train(df)

    def nb_train(self, df):
        category = df.iloc[:, 0].as_matrix()
        X = df.iloc[:, 1:].as_matrix()
        m = df.shape[0]
        word_count = df.shape[1] - 1
        absV = word_count
        denom_0 = np.sum(X[category == 0], axis=(0,1)) + absV
        denom_1 = np.sum(X[category == 1], axis=(0,1)) + absV

        dfp = pd.DataFrame(np.zeros((2, df.shape[1])), columns=df.columns)
        phi_0 = np.sum(category == 0) / m
        phi_1 = np.sum(category == 1) / m
        dfp.iloc[0,0] = phi_0
        dfp.iloc[1,0] = phi_1

        for k in range(1, word_count+1):
            numer_k_0 = np.sum(X[category == 0][:, k-1]) + 1
            numer_k_1 = np.sum(X[category == 1][:, k-1]) + 1
            phi_k_0 = numer_k_0 / denom_0
            phi_k_1 = numer_k_1 / denom_1
            dfp.iloc[0, k] = phi_k_0
            dfp.iloc[1, k] = phi_k_1

        self.dfp = dfp

    def classify(self, df):
        """
        Given a vector of emails on a trained model, classify them.
        """
        pass
