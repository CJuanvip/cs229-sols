import numpy as np

KCENTROIDS = 16
MAX_ITERATIONS = 200

def initialize(kcentroids, image):
    height = image.shape[0]
    width = image.shape[1]
    centroids = np.zeros((kcentroids, 3))

    for k in range(kcentroids):
        i = np.random.randint(low=0, high=width)
        j = np.random.randint(low=0, high=height)
        centroids[k] = image[i, j]

    return centroids
