import numpy as np
import pandas as pd


def nb_train(df):
    category = df.iloc[:, 0].as_matrix()
    X = df.iloc[:, 1:].as_matrix()
    m = df.shape[0]
    word_count = df.shape[1] - 1
    absV = word_count
    denom_0 = np.sum(X[category == 0], axis=(0,1)) + absV
    denom_1 = np.sum(X[category == 1], axis=(0,1)) + absV

    dfp = pd.DataFrame(np.zeros(2, df.shape[1]), column=df.columns)
    phi_0 = np.sum(X[category == 0]) / m
    phi_1 = np.sum(X[category == 1]) / m
    dfp.iloc[0,0] = phi_0
    dfp.iloc[0,1] = phi_1

    for k in range(1, word_count+1):
        numer_k_0 = np.sum(X[category == 0][:, k]) + 1
        numer_k_1 = np.sum(X[category == 1][:, k]) + 1
        phi_k_0 = numer_k_0 / denom_0
        phi_k_1 = numer_k_1 / denom_1
        dfp.iloc[0, k] = phi_k_0
        dfp.iloc[1, k] = phi_k_1

    return dfp
