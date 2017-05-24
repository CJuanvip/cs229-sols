import numpy as np

EPSILON = 1e-7
KCENTROIDS = 16
MAX_ITERS = 200

def initialize(image, kcentroids):
    height = image.shape[0]
    width = image.shape[1]
    centroids = np.zeros((kcentroids, 3))

    for k in range(kcentroids):
        i = np.random.randint(low=0, high=width)
        j = np.random.randint(low=0, high=height)
        centroids[k] = image[i, j]

    return centroids


def kmeans(image, kcentroids, epsilon=EPSILON, max_iterations=MAX_ITERS):
    centroids = initialize(kcentroids, image)
    diffs = np.zeros(centroids.shape)
    clusters = np.zeros(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    for iter in range(max_iterations):
        for i in range(height):
            for j in range(width):
                diffs = image[i, j] - centroids
                clusters[i, j] = np.argmin(np.linalg.norm(diffs, axis=1))

        for k in range(centroids.shape[0]):
            total_k = np.sum(np.sign(clusters == k))
            centroids[k] = (1 / total_k) * np.sum(image * (clusters == k))

    return centroids
