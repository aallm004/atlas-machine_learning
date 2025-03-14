#!/usr/bin/env python3
"""module For kmeans with sklearn"""
import sklearn.cluster


def kmeans(X, k):
    """Function that performs K-means on a dataset:
        X is a numpy.ndarray of shape (n, d) containing the dataset
        k is the number of clusters
    Returns: C, clss
        C is a numpy.ndarray of shape (k, d) containing the centroid means For
        each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of the
        cluster in C that each data point belongs to"""

    kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=None)
    clss = kmeans.fit_predict(X)

    C = kmeans.cluster_centers_

    return C, clss
