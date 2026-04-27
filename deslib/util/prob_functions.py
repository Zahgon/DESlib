# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause
import numpy as np
from scipy.special import betainc
from scipy.stats import entropy

"""This file contains the implementation of several functions used to estimate
the competence level of a base classifiers based on posterior probabilities
predicted for each class.

Reference
----------
T.Woloszynski, M. Kurzynski, A probabilistic model of classifier competence for
dynamic ensemble selection,
Pattern Recognition 44 (2011) 2656–2668.

R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
selection: Recent advances and perspectives,”
Information Fusion, vol. 41, pp. 195 – 216, 2018.

B. Antosik, M. Kurzynski, New measures of classifier competence – heuristics
and application to the design of multiple classifier systems., in: Computer
recognition systems 4., 2011, pp. 197–206.
"""


def exponential_func(n_classes, support_correct):
    """Calculate the exponential function based on the support obtained by
    the base classifier for the correct class label.

    Parameters
    ----------
    n_classes : int
        The number of classes in the problem

    support_correct: array of shape (n_samples)
        containing the supports obtained by the base classifier for the correct
        class

    Returns
    -------
    C_src : array of shape (n_samples)
        Representing the classifier competences at each data point
    """
    pass


def log_func(n_classes, support_correct):
    """Calculate the logarithm in the support obtained by
    the base classifier.

    Parameters
    ----------
    n_classes : int
        The number of classes in the problem

    support_correct: array of shape (n_samples)
        Containing the supports obtained by the base classifier for the correct
        class

    Returns
    -------
    C_src : array of shape (n_samples)
            representing the classifier competences at each data point

    References
    ----------
    T.Woloszynski, M. Kurzynski, A measure of competence based on randomized
    reference classifier for dynamic ensemble selection, in: International
    Conference on Pattern Recognition (ICPR), 2010, pp. 4194–4197.
    """
    pass


def entropy_func(n_classes, supports, is_correct):
    """Calculate the entropy in the support obtained by
    the base classifier. The value of the source competence is inverse
    proportional to the normalized entropy of its supports vector and the sign
    of competence is simply determined  by the correct/incorrect classification

    Parameters
    ----------
    n_classes : int
        The number of classes in the problem

    supports: array of shape (n_samples, n_classes)
        Containing the supports obtained by the base classifier for each class.

    is_correct: array of shape (n_samples)
        Array with 1 whether the base classifier predicted the correct label
        and -1 otherwise

    Returns
    -------
    C_src : array of shape (n_samples)
        Representing the classifier competences at each data point

    References
    ----------
    B. Antosik, M. Kurzynski, New measures of classifier competence –
    heuristics and application to the design of multiple classifier systems.,
    in: Computer recognition systems 4., 2011, pp. 197–206.
    """
    pass


def ccprmod(supports, idx_correct_label, B=20):
    """Python implementation of the ccprmod.m (Classifier competence based on
    probabilistic modelling)
    function. Matlab code is available at:
    http://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/28391/versions/6/previews/ccprmod.m/index.html

    Parameters
    ----------
    supports: array of shape (n_samples, n_classes)
        Containing the supports obtained by the base classifier for each class.

    idx_correct_label: array of shape (n_samples)
                       containing the index of the correct class.

    B : int (Default = 20)
        number of points used in the calculation of the competence, higher
        values result in a more accurate estimation.

    Returns
    -------
    C_src : array of shape (n_samples)
            representing the classifier competences at each data point

    Examples
    --------
    >>> supports = [[0.3, 0.6, 0.1],[1.0/3, 1.0/3, 1.0/3]]
    >>> idx_correct_label = [1,0]
    >>> ccprmod(supports,idx_correct_label)
    ans = [0.784953394056843, 0.332872292262951]

    References
    ----------
    T.Woloszynski, M. Kurzynski, A probabilistic model of classifier competence
    for dynamic ensemble selection,
    Pattern Recognition 44 (2011) 2656–2668.
    """
    pass


def min_difference(supports, idx_correct_label):
    """The minimum difference between the supports obtained for the correct
    class and the vector of class supports. The value of the source competence
    is negative if the sample is misclassified and positive otherwise.

    Parameters
    ----------
    supports: array of shape (n_samples, n_classes)
        Containing the supports obtained by the base classifier for each class

    idx_correct_label: array of shape (n_samples)
        Containing the index of the correct class

    Returns
    -------
    C_src : array of shape (n_samples)
        Representing the classifier competences at each data point

    References
    ----------
    B. Antosik, M. Kurzynski, New measures of classifier competence –
    heuristics and application to the design of multiple classifier systems.,
    in: Computer recognition systems 4., 2011, pp. 197–206.
    """
    pass


def softmax(w, theta=1.0):
    """Takes an vector w of S N-element and returns a vectors where each column
    of the vector sums to 1, with elements exponentially proportional to the
    respective elements in N.

    Parameters
    ----------
    w : array of shape = [N,  M]

    theta : float (default = 1.0)
            used as a multiplier  prior to exponentiation.

    Returns
    -------
    dist : array of shape = [N, M]
        Which the sum of each row sums to 1 and the elements are exponentially
        proportional to the respective elements in N

    """
    pass
