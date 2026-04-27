# coding=utf-8

# Author:  Le Thanh Nguyen-Meidine <nmlethanh91@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted


def is_available():
    try:
        import faiss
        return True
    except ImportError:
        return False


class FaissKNNClassifier:
    """ Scikit-learn wrapper interface for Faiss KNN.

    Parameters
    ----------
    n_neighbors : int (Default = 5)
                Number of neighbors used in the nearest neighbor search.

    n_jobs : int (Default = None)
             The number of jobs to run in parallel for both fit and predict.
              If -1, then the number of jobs is set to the number of cores.

    algorithm : {'brute', 'voronoi'} (Default = 'brute')

        Algorithm used to compute the nearest neighbors:

            - 'brute' will use the :class: `IndexFlatL2` class from faiss.
            - 'voronoi' will use :class:`IndexIVFFlat` class from faiss.
            - 'hierarchical' will use :class:`IndexHNSWFlat` class from faiss.

        Note that selecting 'voronoi' the system takes more time during
        training, however it can significantly improve the search time
        on inference. 'hierarchical' produce very fast and accurate indexes,
        however it has a higher memory requirement. It's recommended when
        you have a lots of RAM or the dataset is small.

        For more information see: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

    n_cells : int (Default = 100)
        Number of voronoi cells. Only used when algorithm=='voronoi'.

    n_probes : int (Default = 1)
        Number of cells that are visited to perform the search. Note that the
        search time roughly increases linearly with the number of probes.
        Only used when algorithm=='voronoi'.

    References
    ----------
    Johnson Jeff, Matthijs Douze, and Hervé Jégou. "Billion-scale similarity
    search with gpus." arXiv preprint arXiv:1702.08734 (2017).
    """

    def __init__(self,
                 n_neighbors=5,
                 n_jobs=None,
                 algorithm='brute',
                 n_cells=100,
                 n_probes=1):

        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.n_cells = n_cells
        self.n_probes = n_probes

        import faiss
        self.faiss = faiss

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

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.

        Parameters
        ----------

        X : array of shape (n_samples, n_features)
            The input data.

        n_neighbors : int
            Number of neighbors to get (default is the value passed to the
            constructor).

        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dists : list of shape = [n_samples, k]
            The distances between the query and each sample in the region of
            competence. The vector is ordered in an ascending fashion.

        idx : list of shape = [n_samples, k]
            Indices of the instances belonging to the region of competence of
            the given query sample.
        """
        pass

    def predict_proba(self, X):
        """Estimates the posterior probabilities for sample in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        preds_proba : array of shape (n_samples, n_classes)
                          Probabilities estimates for each sample in X.
        """
        pass

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

    def _prepare_knn_algorithm(self, X, d):
        pass
