# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause
import warnings

import numpy as np
from sklearn import metrics
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans

from deslib.base import BaseDS
from deslib.util.aggregation import sum_votes_per_class
from deslib.util.diversity import Q_statistic, ratio_errors, \
    negative_double_fault, compute_pairwise_diversity


class DESClustering(BaseDS):
    """Dynamic ensemble selection-Clustering (DES-Clustering).

    This method selects an ensemble of classifiers taking into account the
    accuracy and diversity of the base classifiers. The K-means algorithm is
    used to define the region of competence. For each cluster, the N most
    accurate classifiers are first selected. Then, the J more diverse
    classifiers from the N most accurate classifiers are selected to
    compose the ensemble.

    Parameters
    ----------
     pool_classifiers : list of classifiers (Default = None)
        The generated_pool of classifiers trained for the corresponding
        classification problem. Each base classifiers should support the method
        "predict". If None, then the pool of classifiers is a bagging
        classifier.

    clustering : sklearn.cluster (Default = None)
        The clustering model used to estimate the region of competence.
        If None, a KMeans with K = 5 is used.

    pct_accuracy : float (Default = 0.5)
                   Percentage of base classifiers selected based on accuracy

    pct_diversity : float (Default = 0.33)
                    Percentage of base classifiers selected based on diversity

    more_diverse : Boolean (Default = True)
                   Whether we select the most or the least diverse classifiers
                   to add to the pre-selected ensemble

    metric_diversity : String (Default = 'df')
        Metric used to estimate the diversity of the base classifiers. Can be
        either the double fault (df), Q-statistics (Q), or error correlation.

    metric_performance : String (Default = 'accuracy_score')
        Metric used to estimate the performance of a base classifier on a
        cluster. Can be either any metric from sklearn.metrics.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    DSEL_perc : float (Default = 0.5)
        Percentage of the input data used to fit DSEL.
        Note: This parameter is only used if the pool of classifier is None or
        unfitted.

    voting : {'hard', 'soft'}, default='hard'
            If 'hard', uses predicted class labels for majority rule voting.
            Else if 'soft', predicts the class label based on the argmax of
            the sums of the predicted probabilities, which is recommended for
            an ensemble of well-calibrated classifiers.

    n_jobs : int, default=-1
        The number of parallel jobs to run. None means 1 unless in
        a joblib.parallel_backend context. -1 means using all processors.
        Doesn’t affect fit method.

    References
    ----------
    Soares, R. G., Santana, A., Canuto, A. M., & de Souto, M. C. P.
    "Using accuracy and more_diverse to select classifiers to build ensembles."
    International Joint Conference on Neural Networks (IJCNN)., 2006.

    Britto, Alceu S., Robert Sabourin, and Luiz ES Oliveira. "Dynamic selection
    of classifiers—a comprehensive review."
    Pattern Recognition 47.11 (2014): 3665-3680.

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.
    """

    def __init__(self, pool_classifiers=None, clustering=None,
                 pct_accuracy=0.5, voting='hard',
                 pct_diversity=0.33, more_diverse=True, metric_diversity='DF',
                 metric_performance='accuracy_score', n_clusters=5,
                 random_state=None, DSEL_perc=0.5, n_jobs=-1):

        super(DESClustering, self).__init__(pool_classifiers=pool_classifiers,
                                            random_state=random_state,
                                            DSEL_perc=DSEL_perc,
                                            n_jobs=n_jobs,
                                            )

        self.metric_diversity = metric_diversity
        self.metric_performance = metric_performance
        self.voting = voting
        self.clustering = clustering
        self.pct_accuracy = pct_accuracy
        self.pct_diversity = pct_diversity
        self.more_diverse = more_diverse
        self.n_clusters = n_clusters

    def fit(self, X, y):
        """ Train the DS model by setting the Clustering algorithm and
        pre-processing the information required to apply the DS
        methods.

        First the data is divided into K clusters. Then, for each cluster,
        the N most accurate classifiers are first selected. Then, the J more
        diverse classifiers from the N most accurate classifiers are selected
        to compose the ensemble of the corresponding cluster. An ensemble of
        classifiers is assigned to each of the K clusters.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Data used to fit the model.

        y : array of shape (n_samples)
            class labels of each example in X.

        Returns
        -------
        self
        """
        pass

    def get_competence_region(self, query, k=None):
        pass

    def _preprocess_clusters(self):
        """Preprocess the competence as well as the average diversity of each
        base classifier for each specific cluster.

        This process makes the test routines faster, since the ensemble of
        classifiers of each cluster is already predefined.

        The class attributes Accuracy_cluster_ and diversity_cluster_ stores
        the accuracy and diversity information respectively of each base
        classifier for each cluster. The attribute indices_ stores the
        pre-selected base classifiers for each cluster.
        """
        pass

    def estimate_competence(self, competence_region, distances=None,
                            predictions=None):
        """Get the competence estimates of each base classifier :math:`c_{i}`
        for the classification of the query sample.

        In this case, the competences were already pre-calculated for each
        cluster. So this method computes the nearest cluster and get the
        pre-calculated competences of the base classifiers for the
        corresponding cluster.

        Parameters
        ----------
        predictions : array of shape (n_samples, n_classifiers)
            Predictions of the base classifiers for all test examples.

        Returns
        -------
        competences : array = [n_samples, n_classifiers]
                      The competence level estimated for each base classifier.
        """
        pass

    def select(self, competences):
        """Select an ensemble with the most accurate and most diverse
        classifier for the classification of the query.

        The ensemble for each cluster was already pre-calculated in the fit
        method. So, this method calculates the closest cluster, and returns
        the ensemble associated to this cluster.

        Parameters
        ----------
        competences : array of shape (n_samples)
            Array containing closest cluster index.

        Returns
        -------
        selected_classifiers : array of shape = [n_samples, self.k]
            Indices of the selected base classifier for each test example.

        """
        pass

    def classify_with_ds(self, predictions, probabilities=None,
                         competence_region=None, distances=None,
                         DFP_mask=None):
        """Predicts the label of the corresponding query sample.

        Parameters
        ----------
        predictions : array of shape (n_samples, n_classifiers)
            Predictions of the base classifiers for all test examples.

        probabilities : array of shape (n_samples, n_classifiers, n_classes)
            Probabilities estimates of each base classifier for all test
            examples.

        competence_region : array of shape (n_samples)
            Indices of the nearest clusters to each sample.

        distances : array of shape (n_samples)
            Distances of the nearest clusters to each sample.

        DFP_mask : array of shape (n_samples, n_classifiers)
            Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_label : array of shape (n_samples)
                          Predicted class label for each test example.
        """
        pass

    def predict_proba_with_ds(self, predictions, probabilities,
                              competence_region=None, distances=None,
                              DFP_mask=None):
        """Predicts the label of the corresponding query sample.

        Parameters
        ----------
        predictions : array of shape (n_samples, n_classifiers)
            Predictions of the base classifiers for all test examples.

        probabilities : array of shape (n_samples, n_classifiers, n_classes)
            Probabilities estimates of each base classifier for all test
            examples.

        competence_region : array of shape (n_samples)
            Indices of the nearest clusters to each sample.

        distances : array of shape (n_samples)
            Distances of the nearest clusters to each sample.

        DFP_mask : array of shape (n_samples, n_classifiers)
            Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_proba : array of shape (n_samples, n_classes)
            Posterior probabilities estimates for each test example.
        """
        pass

    def _check_parameters(self):
        """Check if the parameters passed as argument are correct.

        Raises
        ------
        ValueError
            If the hyper-parameters are incorrect.
        """
        pass

    def get_scores_(self, sample_indices):

        pass

    def _set_diversity_func(self):
        """Set the diversity function to be used according to the
        hyper-parameter metric_diversity

        The diversity_func_ can be either the Double Fault, Q-Statistics
        or Ratio of errors.

        """
        pass
