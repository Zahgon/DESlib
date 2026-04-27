# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors

"""
This file contains the implementation of different functions to measure
instance hardness. Instance hardness can be defined as the likelihood that a
given sample will be misclassified by different learning algorithms.

References
----------
Smith, M.R., Martinez, T. and Giraud-Carrier, C., 2014. An instance level
analysis of data complexity.
Machine learning, 95(2), pp.225-256
"""


def hardness_region_competence(neighbors_idx, labels, safe_k):
    """Calculate the Instance hardness of the sample based on its neighborhood.
    The sample is deemed hard to classify when there is overlap between
    different classes in the region of competence. This method does not
    takes into account the target label of the test sample

    This hardness measure is used to select whether use DS or use the KNN for
    the classification of a given query sample

    Parameters
    ----------
    neighbors_idx : array of shape = [n_samples_test, k]
        Indices of the nearest neighbors for each considered sample

    labels : array of shape = [n_samples_train]
        labels associated with each training sample

    safe_k : int
        Number of neighbors used to estimate the hardness of the corresponding
        region

    Returns
    -------
    hardness : array of shape = [n_samples_test]
        The Hardness level associated with each example.

    References
    ----------
    Smith, M.R., Martinez, T. and Giraud-Carrier, C., 2014. An instance level
    analysis of data complexity.
    Machine learning, 95(2), pp.225-256
    """
    pass


def kdn_score(X, y, k):
    """
    Calculates the K-Disagreeing Neighbors score (KDN) of each sample in the
    input dataset.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        The input data.

    y : array of shape (n_samples)
        class labels of each example in X.

    k : int
        Neighborhood size for calculating the KDN score.

    Returns
    -------

    score : array of shape = [n_samples,1]
        KDN score of each sample in X.

    neighbors : array of shape = [n_samples,k]
        Indexes of the k neighbors of each sample in X.


    References
    ----------
    M. R. Smith, T. Martinez, C. Giraud-Carrier, An instance level analysis of
    data complexity,
    Machine Learning 95 (2) (2014) 225-256.

    """
    pass
