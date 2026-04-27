import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_X_y
from sklearn.utils import check_array

from deslib.util import faiss_knn_wrapper
from deslib.util.prob_functions import softmax


class KNNE(BaseEstimator):
    """"
    Implementation of the K-Nearest Neighbors-Equality technique.

    This implementation fits a different KNN method for each class, and search
    on each class for the nearest examples.

    Parameters
    ----------

    n_neighbors : int, (default = 7)
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    algorithm : str = ['knn', 'faiss]', (default = 'knn')
        Whether to use scikit-learn or faiss for nearest neighbors estimation.

    References
    ----------
    Sierra, Basilio, Elena Lazkano, Itziar Irigoien, Ekaitz Jauregi,
    and Iñigo Mendialdua. "K nearest neighbor equality: giving equal chance
    to all existing classes."
    Information Sciences 181, no. 23 (2011): 5158-5168.

    Mendialdua, Iñigo, José María Martínez-Otzeta, I. Rodriguez-Rodriguez,
    T. Ruiz-Vazquez, and Basilio Sierra. "Dynamic selection of the best base
    classifier in one versus one." Knowledge-Based Systems 85 (2015): 298-306.

    Cruz, Rafael MO, Dayvid VR Oliveira, George DC Cavalcanti,
    and Robert Sabourin. "FIRE-DES++: Enhanced online pruning of base
    classifiers for dynamic ensemble selection."
    Pattern Recognition 85 (2019): 149-160.
    """

    def __init__(self, n_neighbors=7, knn_classifier='sklearn', **kwargs):

        self.n_neighbors = n_neighbors
        self.knn_classifier = knn_classifier
        self.kwargs = kwargs

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Data used to fit the model.

        y : array of shape (n_samples)
            class labels of each example in X.
        """
        pass

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        n_neighbors : int
            Number of neighbors to get (default is the value
            passed to the constructor).

        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dist : array
            Array representing the lengths to points, only present if
            return_distance=True

        ind : array
            Indices of the nearest points in the population matrix.
        """
        pass

    def predict(self, X):
        """Predict the class label for each sample in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        preds : array, shape (n_samples,)
                Class labels for samples in X.
        """
        pass

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        proba : array of shape (n_samples, n_classes), or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        pass

    def _set_knn_type(self):

        pass

    def _organize_neighbors(self, dists, inds):
        pass

    def _check_n_neighbors(self, n_neighbors):
        pass

    def _handle_n_neighbors(self, n_neighbors):
        pass
