import numpy as np
import pandas as pd

# TODO: Tuck the dataframe manipulation code up here to keep is out of the 
# Naive Bayes code.
def train(df_train):
    classifier = NaiveBayes(df)

    return classifier


class NaiveBayes:
    def __init__(self, df):
        self.dfp = self.train(df)

    def train(self, df):
        category = df.iloc[:, 0].as_matrix()
        X = df.iloc[:, 1:].as_matrix()
        m = df.shape[0]
        word_count = df.shape[1] - 1
        absV = word_count
        denom_0 = np.sum(X[category == 0], axis=(0,1)) + absV
        denom_1 = np.sum(X[category == 1], axis=(0,1)) + absV

        dfp = pd.DataFrame(np.zeros((2, df.shape[1])), columns=df.columns)
        # Compute the probability that a training email is spam.
        phi_0 = np.sum(category == 0) / m
        phi_1 = np.sum(category == 1) / m
        # Pack them into the first column.
        dfp.iloc[0,0] = phi_0
        dfp.iloc[1,0] = phi_1

        for k in range(1, word_count+1):
            numer_k_0 = np.sum(X[category == 0][:, k-1]) + 1
            numer_k_1 = np.sum(X[category == 1][:, k-1]) + 1
            phi_k_0 = numer_k_0 / denom_0
            phi_k_1 = numer_k_1 / denom_1
            dfp.iloc[0, k] = phi_k_0
            dfp.iloc[1, k] = phi_k_1

        return dfp

    def classify(self, x):
        """
        Given a preprocessed email, classify it as SPAM or NOT SPAM.
        """
        prob_y_equals_0 = self.dfp.iloc[0,0]
        prob_y_equals_1 = self.dfp.iloc[1,0]
        probs_x_given_0 = self.dfp.as_matrix()[0, np.where(x != 0)]
        probs_x_given_0 = probs_x_given_0.reshape(probs_x_given_0.shape[1])
        probs_x_given_1 = self.dfp.as_matrix()[1, np.where(x != 0)]
        probs_x_given_1 = probs_x_given_1.reshape(probs_x_given_1.shape[1])
        
        probs_0 = np.power(prob_y_equals_0 * probs_x_given_0, x[x != 0])
        probs_1 = np.power(prob_y_equals_1 * probs_x_given_1, x[x != 0])

        # We call index zero because we are dealingn with a pandas DataFrame.
        # TODO: Remove the pandas dataframes from the ML code and wrap it instead.
        numer_0 = np.product(probs_0)#[0]
        numer_1 = np.product(probs_1)#[0]
        denom = numer_0 + numer_1

        py_given_x = np.zeros(2)
        # SPAM given x
        py_given_x[1] = numer_1 / denom
        # NOT SPAM given x
        py_given_x[0] = numer_0 / denom
        
        return np.argmax(py_given_x)
