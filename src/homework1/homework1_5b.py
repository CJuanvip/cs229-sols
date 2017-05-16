import numpy as np
import numpy.linalg as lin


def weight(tau, Xtrain_i):
    return lambda x: np.exp(-((x - Xtrain_i).T.dot(x - Xtrain_i))/(2*tau**2))


class WeightMatrix():
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, x):
        shape = (len(self.weights), len(self.weights))
        mat = np.zeros(shape)
        for i in range(len(self.weights)):
            mat[i, i] = self.weights[i](x)

        return mat


def weightM(tau, Xtrain):
    size = Xtrain.shape[0]
    weights = {}
    for i in range(0, size):
        weights[i] = weight(tau, Xtrain[i])

    return WeightMatrix(weights)


class LWLRModel():
    def __init__(self, W, Xtrain, ytrain):
        self.W = W
        self.Xtrain = Xtrain
        self.ytrain = ytrain

    def evaluate(self, x):
        W = self.W(x)
        X = self.Xtrain
        y = self.ytrain
        theta = lin.inv((X.T.dot(W).dot(X))).dot(X.T).dot(W).dot(y)

        return np.dot(theta.T, x)
