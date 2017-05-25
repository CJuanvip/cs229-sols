import numpy as np


def indicator(x):
    return np.abs(np.sign(1*x))

def initialize(image, kcentroids):
    height    = image.shape[0]
    width     = image.shape[1]
    centroids = np.zeros((kcentroids, 3))

    for k in range(kcentroids):
        i = np.random.randint(height)
        j = np.random.randint(width)
        centroids[k] = image[i, j]

    return centroids


def kmeans(image, kcentroids, epsilon, max_iterations):
    height = image.shape[0]
    width  = image.shape[1]

    centroids = initialize(image, kcentroids)
    diffs     = np.zeros(centroids.shape)
    clusters  = np.zeros((height, width, 1), dtype=int)
    new_centroids = np.zeros(centroids.shape)
    centroid_delta = centroids.copy()

    # Initial clustering.
    for i in range(height):
        for j in range(width):
            diffs = image[i, j] - centroids
            clusters[i, j] = np.argmin(np.sum(diffs*diffs, axis=1))

    for _ in range(max_iterations):
        for i in range(height):
            for j in range(width):
                diffs = image[i, j] - centroids
                clusters[i, j] = np.argmin(np.sum(diffs*diffs, axis=1))

        for k in range(kcentroids):
            indicators = indicator(clusters == k)
            total_k = np.sum(indicators)
            new_centroids[k] = (1 / total_k) * np.sum(indicators * image, axis=(1,0))
        
        # Test for convergence.
        centroid_delta = new_centroids - centroids
        if np.all(np.linalg.norm(centroid_delta) < epsilon):
            centroids = new_centroids.copy()
            break
        else:
            centroids = new_centroids.copy()

    centroids = np.uint8(np.round(centroids))
    
    return clusters, centroids


def make_image(clusters, centroids):
    new_image = np.zeros((clusters.shape[0], clusters.shape[1], 3))
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            new_image[i, j,:] = centroids[clusters[i, j]]
    
    return np.uint8(np.round(new_image))

