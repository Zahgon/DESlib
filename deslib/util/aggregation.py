# coding=utf-8
# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from scipy.stats.mstats import mode
from sklearn.utils.validation import check_array

from deslib.util.prob_functions import softmax

"""
This file contains the implementation of different aggregation functions to
combine the outputs of the base
classifiers to give the final decision.

References
----------
Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms.
John Wiley & Sons, 2004.

J. Kittler, M. Hatef, R. P. W. Duin, J. Matas, On combining classifiers, IEEE
Transactions on Pattern Analysis and Machine Intelligence 20 (1998) 226–239.
"""


def majority_voting(classifier_ensemble, X):
    """Apply the majority voting rule to predict the label of each sample in X.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the
        aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    pass


def weighted_majority_voting(classifier_ensemble, weights, X):
    """Apply the weighted majority voting rule to predict the label of each
    sample in X. The size of the weights vector should be equal to the size of
    the ensemble.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    weights : array of shape (n_samples, n_classifiers)
              Weights associated to each base classifier for each sample


    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    pass


def _get_ensemble_votes(classifier_ensemble, X):
    """Calculates the votes obtained by each based classifier in the ensemble
    for sample in X

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    votes : array of shape (n_samples, n_classifiers)
            The votes obtained by each base classifier
    """
    pass


def majority_voting_rule(votes):
    """Applies the majority voting rule to the estimated votes.

    Parameters
    ----------
    votes : array of shape (n_samples, n_classifiers),
        The votes obtained by each classifier for each sample.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    pass


def weighted_majority_voting_rule(votes, weights, labels_set=None):
    """Applies the weighted majority voting rule based on the votes obtained by
    each base classifier and their
    respective weights.

    Parameters
    ----------
    votes : array of shape (n_samples, n_classifiers),
        The votes obtained by each classifier for each sample.

    weights : array of shape (n_samples, n_classifiers)
        Weights associated to each base classifier for each sample

    labels_set : (Default=None) set with the possible classes in the problem.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    pass


def get_weighted_votes(votes, weights, labels_set=None):
    pass


def sum_votes_per_class(predictions, n_classes):
    """Sum the number of votes for each class. Accepts masked arrays as input.

    Parameters
    ----------
    predictions : array of shape (n_samples, n_classifiers),
        The votes obtained by each classifier for each sample. Can be a masked
        array.

    n_classes : int
        Number of classes.

    Returns
    -------
    summed_votes : array of shape (n_samples, n_classes)
        Summation of votes for each class
    """
    pass


def _get_ensemble_probabilities(classifier_ensemble, X,
                                estimator_features=None):
    """Get the probabilities estimate for each base classifier in the ensemble

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    estimator_features : array of shape (n_classifiers, n_selected_features)
        Indices containing the features used by each classifier.

    Returns
    -------
    list_proba : array of shape (n_samples, n_classifiers, n_classes)
        Probabilities predicted by each base classifier in the ensemble for all
        samples in X.
    """
    pass


def predict_proba_ensemble(classifier_ensemble, X, estimator_features=None):
    """Estimates the posterior probabilities of the give ensemble for each
    sample in X.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    estimator_features : array of shape (n_classifiers, n_selected_features)
        Indices containing the features used by each classifier.

    Returns
    -------
    predicted_proba : array of shape (n_samples, n_classes)
        Posterior probabilities estimates for each samples in X.
    """
    pass


def aggregate_proba_ensemble_weighted(ensemble_proba, weights):
    pass


def average_combiner(classifier_ensemble, X):
    """Ensemble combination using the Average rule.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    pass


def product_combiner(classifier_ensemble, X):
    """Ensemble combination using the Product rule.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    predicted_label : array of shape = [n_classifiers, n_samples, n_classes]
        Probabilities predicted by each base classifier in the ensemble for all
        samples in X.
    """
    pass


def maximum_combiner(classifier_ensemble, X):
    """Ensemble combination using the Maximum rule.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    pass


def minimum_combiner(classifier_ensemble, X):
    """Ensemble combination using the Minimum rule.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    pass


def median_combiner(classifier_ensemble, X):
    """Ensemble combination using the Median rule.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    pass


def average_rule(predictions):
    """Apply the average fusion rule to the predicted vector of class supports
    (predictions).

    Parameters
    ----------
    predictions : np array of shape (n_samples, n_classifiers, n_classes)
        Vector of class supports predicted by each base classifier for sample

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    pass


def product_rule(predictions):
    """Apply the product fusion rule to the predicted vector of class supports
    (predictions).

    Parameters
    ----------
    predictions : array of shape (n_samples, n_classifiers, n_classes)
        Vector of class supports predicted by each base classifier for sample

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    pass


def median_rule(predictions):
    """Apply the product fusion rule to the predicted vector of class supports
    (predictions).

    Parameters
    ----------
    predictions : np array of shape (n_samples, n_classifiers, n_classes)
        Vector of class supports predicted by each base classifier for sample

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    pass


def maximum_rule(predictions):
    """Apply the product fusion rule to the predicted vector of class supports
    (predictions).

    Parameters
    ----------
    predictions : np array of shape (n_samples, n_classifiers, n_classes)
        Vector of class supports predicted by each base classifier for sample

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    pass


def minimum_rule(predictions):
    """Apply the product fusion rule to the predicted vector of class supports
    (predictions).

    Parameters
    ----------
    predictions : np array of shape (n_samples, n_classifiers, n_classes)
        Vector of class supports predicted by each base classifier for sample

    Returns
    -------
    list_proba : array of shape = [n_classifiers, n_samples, n_classes]
        Probabilities predicted by each base classifier in the ensemble for all
        samples in X.
    """
    pass


def _check_predictions(predictions):
    """Check if the predictions array has the correct size.

    Raises a value error if the array do not contain exactly 3 dimensions:
    [n_samples, n_classifiers, n_classes]

    """
    pass
