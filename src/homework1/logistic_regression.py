import numpy as np
import numpy.linalg as lin

MAX_ITERS = 20
EPSILON   = 1.0e-7

fileX = 'logistic_x.txt'
fileY = 'logistic_y.txt'


def load_data(fileX, fileY):
    X = np.loadtxt(fileX)
    y = np.loadtxt(fileY)

    return (X, y)


def h(theta, X, y):
    margins = y * X.dot(theta)
    
    return 1 / (1 + np.exp(-margins))


def J(theta, X, y):
    probs = h(theta, X, y)
    mm = probs.size

    return (-1 / mm) * np.sum(np.log(probs))


def gradJ(theta, X, y):
    probs = h(theta, X, -y) * y
    mm = probs.size

    return (-1 / mm) * X.T.dot(probs)


def hessJ(theta, X, y):
    d     = h(theta, X, y)
    probs = d * (1 - d)
    mm    = probs.size
    prob_vec = np.diag(probs)
    hessJ = (1 / mm) * (X.T.dot(prob_vec).dot(X))

    return hessJ


def logistic_regression(X, y, epsilon, max_iters):
    mm = X.shape[0]
    nn = X.shape[1]

    # The cost of the ith iteration of newton-raphson.
    cost = np.zeros(max_iters)
    theta = np.zeros(nn)

    for i in range(0, max_iters):
        cost[i] = J(theta, X, y)
        grad    = gradJ(theta, X, y)
        H       = hessJ(theta, X, y)
        Hinv    = lin.inv(H)
        theta   = theta - Hinv.dot(grad)

    return (theta, cost)


def regression(epsilon=EPSILON, max_iters=MAX_ITERS):
    X, y = load_data(fileX, fileY)
    ones = np.ones((99,1))
    Xsplit = np.split(X, indices_or_sections=[1], axis=1)
    X = np.concatenate([ones, Xsplit[0], Xsplit[1]], axis=1)

    return logistic_regression(X, y, epsilon, max_iters)



def main():
    theta, cost = regression()

    print('theta = {}'.format(theta))
    print('cost = {}'.format(cost))