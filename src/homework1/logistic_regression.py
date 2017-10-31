import numpy as np
import numpy.linalg as lin

MAX_ITERS = 20
EPSILON   = 1.0e-7


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

    # The cost of the ith iteration of Newton-Raphson.
    cost = np.zeros(max_iters)
    theta = np.zeros(nn)

    for i in range(0, max_iters):
        cost[i] = J(theta, X, y)
        grad    = gradJ(theta, X, y)
        H       = hessJ(theta, X, y)
        Hinv    = lin.inv(H)
        theta   = theta - Hinv.dot(grad)

    return (theta, cost)
