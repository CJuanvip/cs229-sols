import numpy as np
import numpy.linalg as lin


def weight(tau):
    def go(Xtrain_i):
        return lambda x: np.exp(-((x - Xtrain_i).T.dot(x - Xtrain_i))/(2*tau**2))

    return lambda Xtrain_i: go(Xtrain_i)


class WeightMatrix():
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, x, out=None):
        if out is None:
            mat = np.eye(len(self.weights))
        else:
            mat = out

        for i in range(len(self.weights)):
            mat[i, i] = self.weights[i](x)

        return mat


def weightM(tau):
    def go(Xtrain):
        size = Xtrain.shape[0]
        weights = {}
        for i in range(0, size):
            weights[i] = weight(tau)(Xtrain[i, 1:])

        return WeightMatrix(weights)

    return lambda Xtrain: go(Xtrain)


class LWLRModel():
    def __init__(self, W, Xtrain, ytrain):
        self.W = W
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.vevaluate = np.vectorize(self.__evaluate)
        self.scratch = np.eye(len(W.weights))

    def __evaluate(self, x):
        W = self.W(x, out=self.scratch)
        X = self.Xtrain
        y = self.ytrain
        theta = lin.inv((X.T.dot(W).dot(X))).dot(X.T).dot(W).dot(y)

        return np.dot(theta.T, np.hstack((1, x)))

    def evaluate(self, vec):
        return self.vevaluate(vec)

    def __call__(self, vec):
        return self.evaluate(vec)
