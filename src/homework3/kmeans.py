import numpy as np


def initialize(image, kcentroids):
    height = image.shape[0]
    width = image.shape[1]
    centroids = np.zeros((kcentroids, 3))

    for k in range(kcentroids):
        i = np.random.randint(low=0, high=width)
        j = np.random.randint(low=0, high=height)
        centroids[k] = image[i, j]

    return centroids


def kmeans(image, kcentroids, max_iterations):
    centroids = initialize(image, kcentroids)
    diffs = np.zeros(centroids.shape)
    clusters = np.zeros(image.shape, dtype='int')
    height = image.shape[0]
    width = image.shape[1]
    for iter in range(max_iterations):
        # TODO: Vectorize these loops.
        for i in range(height):
            for j in range(width):
                diffs = image[i, j] - centroids
                clusters[i, j] = int(np.argmin(np.linalg.norm(diffs, axis=1)))

        for k in range(kcentroids):
            total_k = np.sum(1*(clusters == k))
            centroids[k] = (1 / total_k) * np.sum(image * (clusters == k))

    return clusters, centroids

def make_image(clusters, centroids):
    pass