"""Implementation of the Dynamic Frienemy Pruning (DFP) algorithm for online
pruning of base classifiers.

References
----------
Oliveira, D.V.R., Cavalcanti, G.D.C. and Sabourin, R., Online Pruning
of Base Classifiers for Dynamic Ensemble Selection,
Pattern Recognition, vol. 72, December 2017, pp 44-58.

Cruz, Rafael MO, Dayvid VR Oliveira, George DC Cavalcanti, and Robert Sabourin.
"FIRE-DES++: Enhanced online pruning of base classifiers for dynamic ensemble
selection." Pattern Recognition 85 (2019): 149-160.
"""

# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause


import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def frienemy_pruning(X_query, X_dsel, y_dsel, ensemble, k):
    """Implements the Online Pruning method (frienemy) which prunes base
    classifiers that do not cross the region of competence of a given instance.
    A classifier crosses the region of competence if it correctly
    classify at least one sample for each different class in the region.

    Parameters
    ----------
    X_query : array-like of shape (n_samples, n_features)
        Test set.
    X_dsel : array-like of shape (n_samples, n_features)
        Dynamic selection set.
    y_dsel : array-like of shape (n_samples,)
        The target values (Dynamic selection set).
    ensemble : list of shape = [n_classifiers]
        The ensemble of classifiers to be pruned.
    k : int
        Number of neighbors used to compute the regions of competence.

    Returns
    -------
    DFP_mask : array-like of shape = [n_samples, n_classifiers]
               Mask containing 1 for the selected base classifier and 0
               otherwise.

    """
    pass


def frienemy_pruning_preprocessed(neighbors, y_val, hit_miss):
    """Implements the Online Pruning method (frienemy) which prunes base
    classifiers that do not cross the region of competence of a given instance.
    A classifier crosses the region of competence if it correctly
    classify at least one sample for each different class in the region.

    Notes
    -----
    This implementation assumes the regions of competence of each query example
    (neighbors) and the predictions for the dynamic selection data (hit_miss)
    were already pre-computed.

    Parameters
    ----------
    neighbors : array-like of shape (n_samples, n_neighbors)
        Indices of the k nearest neighbors.
    y_val : array-like of shape (n_samples,)
        The target values (class labels).
    hit_miss : array-like of shape (n_samples, n_classifiers)
        Matrix containing 1 when the base classifier made the correct
        prediction, 0 otherwise.

    Returns
    -------
    DFP_mask : array-like of shape = [n_samples, n_classifiers]
               Mask containing 1 for the selected base classifier and 0
               otherwise.
    """
    pass
