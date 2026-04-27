# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause


import functools
import math
import warnings
from abc import abstractmethod, ABCMeta

import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaseEnsemble, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils.validation import (check_X_y, check_is_fitted, check_array,
                                      check_random_state, validate_data)

from deslib.util import KNNE
from deslib.util import faiss_knn_wrapper
from deslib.util.dfp import frienemy_pruning_preprocessed
from deslib.util.instance_hardness import hardness_region_competence


class BaseDS(BaseEstimator, ClassifierMixin):
    """Base class for a dynamic classifier selection (dcs) and
       dynamic ensemble selection (des) methods.

    All DCS and DES techniques should inherit from this class.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, needs_proba=False,
                 random_state=None, knn_classifier='knn',
                 knn_metric='minkowski', DSEL_perc=0.5, knne=False, n_jobs=-1,
                 voting=None):

        self.pool_classifiers = pool_classifiers
        self.k = k
        self.DFP = DFP
        self.with_IH = with_IH
        self.safe_k = safe_k
        self.IH_rate = IH_rate
        self.needs_proba = needs_proba
        self.random_state = random_state
        self.knn_classifier = knn_classifier
        self.knn_metric = knn_metric
        self.DSEL_perc = DSEL_perc
        self.knne = knne
        self.n_jobs = n_jobs
        self.voting = voting

        # Check optional dependency
        if knn_classifier == 'faiss' and not faiss_knn_wrapper.is_available():
            raise ImportError(
                'Using knn_classifier="faiss" requires that the FAISS library '
                'be installed.Please check the Installation Guide.')

    def fit(self, X, y):
        """Prepare the DS model by setting the KNN algorithm and
        pre-processing the information required to apply the DS
        methods

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.

        y : array of shape (n_samples)
            class labels of each example in X.

        Returns
        -------
        self
        """
        pass

    def get_competence_region(self, query, k=None):
        """Compute the region of competence of the query sample
        using the data belonging to DSEL.

        Parameters
        ----------
        query : array of shape (n_samples, n_features)
                The test examples.

        k : int (Default = self.k)
            The number of neighbors used to in the region of competence.

        Returns
        -------
        dists : array of shape (n_samples, k)
                The distances between the query and each sample in the region
                of competence. The vector is ordered in an ascending fashion.

        idx : array of shape (n_samples, k)
              Indices of the instances belonging to the region of competence of
              the given query sample.
        """
        pass

    @abstractmethod
    def estimate_competence(self, competence_region, distances=None,
                            predictions=None):
        """estimate the competence of each base classifier :math:`c_{i}`
        the classification of the query sample :math:`\\mathbf{x}`.
        Returns an array containing the level of competence estimated
        for each base classifier. The size of the vector is equals to
        the size of the generated_pool of classifiers.

        Parameters
        ----------
        competence_region : array of shape (n_samples, n_neighbors)
                    Indices of the k nearest neighbors according for each
                    test sample.

        distances : array of shape (n_samples, n_neighbors)
                    Distances of the k nearest neighbors according for each
                    test sample.

        predictions : array of shape (n_samples, n_classifiers)
                      Predictions of the base classifiers for all test examples
        Returns
        -------
        competences : array (n_classifiers) containing the competence level
                      estimated for each base classifier
        """
        pass

    @abstractmethod
    def select(self, competences):
        """Select the most competent classifier for
        the classification of the query sample x.
        The most competent classifier (dcs) or an ensemble
        with the most competent classifiers (des) is returned

        Parameters
        ----------
        competences : array of shape (n_samples, n_classifiers)
                      The estimated competence level of each base classifier
                      for test example

        Returns
        -------
        selected_classifiers : array containing the selected base classifiers
                               for each test sample

        """
        pass

    @abstractmethod
    def classify_with_ds(self, predictions, probabilities=None,
                         neighbors=None, distances=None, DFP_mask=None):
        """Predicts the label of the corresponding query sample.
        Returns the predicted label.

        Parameters
        ----------
        predictions : array of shape (n_samples, n_classifiers)
            Predictions of the base classifiers for all test examples

        probabilities : array of shape (n_samples, n_classifiers, n_classes)
            Probabilities estimates of each base classifier for all test
            examples (For methods that always require probabilities from the
            base classifiers)

        neighbors : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors.
        distances : array of shape (n_samples, n_neighbors)
            Distances from the k nearest neighbors to the query

        DFP_mask : array of shape (n_samples, n_classifiers)
            Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_label : array of shape (n_samples)
            The predicted label for each query
        """
        pass

    @abstractmethod
    def predict_proba_with_ds(self, predictions, probabilities,
                              neighbors=None, distances=None, DFP_mask=None):
        """Predicts the posterior probabilities of the corresponding
        query sample. Returns the probability estimates of each class.

        Parameters
        ----------
        predictions : array of shape (n_samples, n_classifiers)
            Predictions of the base classifiers for all test examples

        probabilities : array of shape (n_samples, n_classifiers, n_classes)
            The predictions of each base classifier for all samples (For
            methods that always require probabilities from the base
            classifiers).

        neighbors : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors.
        distances : array of shape (n_samples, n_neighbors)
            Distances from the k nearest neighbors to the query

        DFP_mask : array of shape (n_samples, n_classifiers)
           Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_proba: array of shape (n_samples, n_classes)
            Posterior probabilities estimates for each test example.
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
        predicted_labels : array of shape (n_samples)
                           Predicted class label for each sample in X.
        """
        pass

    def _check_predict(self, X):
        pass

    def predict_proba(self, X):
        """Estimates the posterior probabilities for sample in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        predicted_proba : array of shape (n_samples, n_classes)
                          Probabilities estimates for each sample in X.
        """
        pass

    def _preprocess_predictions(self, X, req_proba):
        pass

    def _split_agreement(self, base_predictions):
        pass

    def _IH_prediction(self, X, ind_disagree, predicted_proba, is_proba=False):
        pass

    def _split_easy_samples(self, neighbors):
        pass

    def _predict_easy_samples(self, X_DS, distances, ind_disagreement,
                              ind_easy, neighbors, predictions, is_proba):
        pass

    def _prepare_indices_DS(self, base_predictions, base_probabilities,
                            ind_disagreement, ind_ds_classifier):
        # Get the real indices_ of the samples that will be classified
        # using a DS algorithm.
        pass

    def _get_DFP_mask(self, neighbors):
        pass

    def _fit_pool_classifiers(self, X, y):
        pass

    def _check_label_encoder(self):
        # Check if base classifiers are not using LabelEncoder (the case for
        # scikit-learn's ensembles):
        pass

    def _compute_highest_possible_IH(self):
        pass

    def _validate_ih(self):
        pass

    def _validate_k(self):
        # validate safe_k
        pass

    def _setup_label_encoder(self, y):
        pass

    def _encode_base_labels(self, y):
        pass

    def _set_dsel(self, X, y):
        """Pre-Process the input X and y data into the dynamic selection
        dataset(DSEL) and get information about the structure of the data
        (e.g., n_classes, n_samples, classes)

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The Input data.

        y : array of shape (n_samples)
            class labels of each sample in X.
        """
        pass

    def _set_region_of_competence_algorithm(self, X):

        pass

    def _preprocess_dsel(self):
        """Compute the prediction of each base classifier for
        all samples in DSEL. Used to speed-up the test phase, by
        not requiring to re-classify training samples during test.

        Returns
        -------
        DSEL_processed_ : array of shape (n_samples, n_classifiers).
                         Each element indicates whether the base classifier
                         predicted the correct label for the corresponding
                         sample (True), otherwise (False).

        BKS_DSEL_ : array of shape (n_samples, n_classifiers)
                   Predicted labels of each base classifier for all samples
                   in DSEL.
        """
        pass

    def _predict_base(self, X):
        """ Get the predictions of each base classifier in the pool for all
            samples in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The test examples.

        Returns
        -------
        predictions : array of shape (n_samples, n_classifiers)
                      The predictions of each base classifier for all samples
                      in X.
        """
        pass

    def _predict_proba_base(self, X):
        """ Get the predictions (probabilities) of each base classifier in the
        pool for all samples in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The test examples.

        Returns
        -------
        probabilities : array of shape (n_samples, n_classifiers, n_classes)
                        Probabilities estimates of each base classifier for all
                        test samples.
        """
        pass

    @staticmethod
    def _all_classifier_agree(predictions):
        """Check whether there is a difference in opinion among the classifiers
        in the generated_pool.

        Parameters
        ----------
        predictions : array of shape (n_samples, n_classifiers)
                      Predictions of the base classifiers for the test examples

        Returns
        -------
        array of shape (classes)
            containing True if all classifiers in the generated_pool agrees
            on the same label, otherwise False.
        """
        pass

    def _validate_parameters(self):
        """Verify if the input parameters are correct (generated_pool and k)
        raises an error if k < 1 or generated_pool is not fitted.
        """
        pass

    def _validate_pool_classifiers(self):
        """ Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute.

        Raises
        -------
        ValueError
            If the pool of classifiers is empty.
        """
        pass

    def _check_predict_proba(self):
        """ Checks if each base classifier in the pool implements the
        predict_proba method.

        Raises
        -------
        ValueError
            If the base classifiers do not implements the predict_proba method.
        """
        pass

    def _check_base_classifier_fitted(self):
        """ Checks if each base classifier in the pool is fitted.

        Raises
        -------
        NotFittedError: If any of the base classifiers is not yet fitted.
        """
        pass
