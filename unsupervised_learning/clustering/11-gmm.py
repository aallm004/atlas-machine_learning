#!/usr/bin/env python3
"""module for gmm"""
import sklearn.mixture


def gmm(X, k):
    """Function that calculates a GMM from a dataset:
            X is a numpy.ndarray of shape (n, d) containing the dataset
            k is the number of clusters
        Returns: pi, m, S, clss, bic
            pi is a numpy.ndarray of shape (k,) containing the cluster priors
            m is a numpy.ndarray of shape (k, d) containing the centroid means
            S is a numpy.ndarray of shape (k, d, d) containing the covariance
            matrices
            clss is a numpy.ndarray of shape (n,) containing the cluster
            indices for each data point
            bic is a numpy.ndarray of shape (kmax - kmin + 1) containing the
            BIC value for each cluster size tested
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=k, random_state=None)
    clss = gmm.fit_predict(X)

    # Get model parameters
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    bic = gmm.bic(X)

    return pi, m, S, clss, bic
