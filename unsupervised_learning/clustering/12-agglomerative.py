#!/usr/bin/env python3
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt

def agglomerative(X, dist):
    """Function that performs agglomerative clustering on dataset
        X is a numpy.ndarray of shape (n, d) containing the dataset
        dist is the maximum cophenetic distance for all cluseters
        Performs agglomerative clustering with Ward linkage
        Displays the dendogram with each cluster displayed in a different color
            Returns: clss, a numpy.ndarray of shape (n,) containing the cluster
            indices for each data point
        """
    matrix = scipy.cluster.hierarchy.ward(X)

    plt.figure(figsize=(10, 7))

    scipy.cluster.hierarchy.dendrogram(matrix, color_threshold=dist)

    plt.show()

    clss = scipy.cluster.hierarchy.fcluster(matrix, dist, criterion='distance')

    return clss - 1
