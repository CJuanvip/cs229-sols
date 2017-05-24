import numpy as np


def indicator(x):
    return np.abs(np.sign(1*x))

def initialize(image, kcentroids):
    height    = image.shape[0]
    width     = image.shape[1]
    centroids = np.zeros((kcentroids, 3))

    for k in range(kcentroids):
        i = np.random.randint(low=0, high=width)
        j = np.random.randint(low=0, high=height)
        centroids[k] = image[i, j]

    return centroids


def kmeans(image, kcentroids, max_iterations):
    height = image.shape[0]
    width  = image.shape[1]

    centroids = initialize(image, kcentroids)
    diffs     = np.zeros(centroids.shape)
    clusters  = np.zeros((height, width, 1), dtype=int)
    
    print('CENTROIDS INIT = {}'.format(centroids))

    for round in range(max_iterations):
        for i in range(height):
            for j in range(width):
                diffs = image[i, j] - centroids
                clusters[i, j] = np.int(np.argmin(np.linalg.norm(diffs, axis=1)))

        for k in range(kcentroids):
            indicators = indicator(clusters == k)
            total_k = np.sum(indicators)
            centroids[k] = (1 / total_k) * np.sum(indicators * image, axis=(1,0))

        import sys
        print('ROUND = {}\nCENTROIDS = {}'.format(round, centroids))

    centroids = np.uint(np.round(centroids))
    
    return clusters, centroids


def make_image(clusters, centroids):
    new_image = np.zeros((clusters.shape[0], clusters.shape[1], 3))
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            new_image[i, j] = centroids[clusters[i,j,0]]

    return new_image
