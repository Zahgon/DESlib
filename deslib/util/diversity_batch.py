import sys

import numpy as np

"""
This file contains the implementation of key diversity measures found in the
ensemble literature:

- Double Fault
- Negative Double fault
- Q-statistics
- Ratio of errors
- Agreement/Disagreement
- Classifier Correlation

The implementation are made according to the specifications from the book
"Combining Pattern Classifiers" based on Oracle outputs, i.e., taking into
account if the pair of classifiers made the correct/incorrect prediction:

N00 : represents samples that both classifiers made a wrong prediction

N10 : represents samples  that only classifier 2 predicts the wrong label.

N10 : represents samples  that only classifier 1 predicts the wrong label.

N11 :  represents samples  that both classifiers predicts the correct label.

References
----------
Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms.
John Wiley & Sons, 2004.

Shipp, Catherine A., and Ludmila I. Kuncheva. "Relationships between
combination methods and measures of diversity in combining classifiers."
Information fusion 3.2 (2002): 135-148.

Giacinto, Giorgio, and Fabio Roli. "Design of effective neural network
ensembles for image classification purposes."
Image and Vision Computing 19.9 (2001): 699-707.

Aksela, Matti. "Comparison of classifier selection methods for improving
committee performance."
Multiple Classifier Systems (2003): 159-159.
"""


def _process_predictions(y: np.array, y_pred1: np.array,
                         y_pred2: np.array) -> np.array:
    """Pre-process the predictions of a pair of base classifiers for the
    computation of the diversity measures

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    N00 : Array of shape (n_samples,)
        Percentage of samples that both classifiers predict the wrong label

    N10 : Array of shape (n_samples,)
        Percentage of samples that only classifier 2 predicts the wrong label

    N01 : Array of shape (n_samples,)
        Percentage of samples that only classifier 1 predicts the wrong label

    N11 : Array of shape (n_samples,)
        Percentage of samples that both classifiers predict the correct label
    """
    pass


def double_fault(y: np.array, y_pred1: np.array,
                 y_pred2: np.array) -> np.array:
    """Calculates the double fault (df) measure. This measure represents the
    probability that both classifiers makes the wrong prediction. A lower value
    of df means the base classifiers are less likely to make the same error.
    This measure must be minimized to increase diversity.

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    df : The double fault measure between two classifiers

    References
    ----------
    Giacinto, Giorgio, and Fabio Roli. "Design of effective neural network
    ensembles for image classification purposes."
    Image and Vision Computing 19.9 (2001): 699-707.
    """
    pass


def negative_double_fault(
        y: np.array, y_pred1: np.array, y_pred2: np.array
) -> np.array:
    """The negative of the double fault measure. This measure should be
    maximized for a higher diversity.

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    df : The negative double fault measure between two classifiers

    References
    ----------
    Giacinto, Giorgio, and Fabio Roli. "Design of effective neural network
    ensembles for image classification purposes."
    Image and Vision Computing 19.9 (2001): 699-707.
    """
    pass


def Q_statistic(y: np.array, y_pred1: np.array, y_pred2: np.array) -> np.array:
    """Calculates the Q-statistics diversity measure between a pair of
    classifiers. The Q value is in a range [-1, 1]. Classifiers that tend to
    classify the same object correctly will have positive values of Q, and
    Q = 0 for two independent classifiers.

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    Q : The q-statistic measure between two classifiers
    """
    pass


def ratio_errors(y: np.array, y_pred1: np.array,
                 y_pred2: np.array) -> np.array:
    """Calculates Ratio of errors diversity measure between a pair of
    classifiers. A higher value means that the base classifiers are less likely
    to make the same errors. The ratio must be maximized for a higher diversity

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    ratio : The q-statistic measure between two classifiers

    References
    ----------
    Aksela, Matti. "Comparison of classifier selection methods for improving
    committee performance."
    Multiple Classifier Systems (2003): 159-159.
    """
    pass


def disagreement_measure(y: np.array, y_pred1: np.array,
                         y_pred2: np.array) -> np.array:
    """Calculates the disagreement measure between a pair of classifiers. This
    measure is calculated by the frequency that only one classifier makes the
    correct prediction.

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    disagreement : The frequency at which both classifiers disagrees
    """
    pass


def agreement_measure(y: np.array, y_pred1: np.array,
                      y_pred2: np.array) -> np.array:
    """Calculates the agreement measure between a pair of classifiers. This
    measure is calculated by the frequency that both classifiers either
    obtained the correct or incorrect prediction for any given sample

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    agreement : The frequency at which both classifiers agrees
    """
    pass


def correlation_coefficient(
        y: np.array, y_pred1: np.array, y_pred2: np.array
) -> np.array:
    """Calculates the correlation  between two classifiers using oracle
    outputs. Coefficient is a value in a range [-1, 1].

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    rho : The correlation coefficient measured between two classifiers
    """
    pass


def compute_pairwise_diversity(
        targets: np.array, prediction_matrix: np.array,
        diversity_func: np.array
) -> np.array:
    """Computes the pairwise diversity matrix.

    Parameters
    ----------
    targets : array of shape (n_samples):
       Class labels of each sample in X.

    prediction_matrix : array of shape (n_samples, n_classifiers):
       Predicted class labels for each classifier in the pool

    diversity_func : Function
       Function used to estimate the pairwise diversity

    Returns
    -------
    diversity : array of shape = [n_classifiers]
       The average pairwise diversity matrix calculated for the pool of
       classifiers

    """
    pass
