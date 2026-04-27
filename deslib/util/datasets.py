# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.utils.validation import check_random_state

"""
This file contains routines to generate 2D classification datasets
that can be used to test the performance of different machine learning
algorithms.

Datasets:

- P2 Dataset
- Circle and Square
- Banana
- Banana 2
- XOR

"""


def make_P2(size_classes, random_state=None):
    """Generate the P2 Dataset:

    The P2 is a two-class problem, presented by Valentini[1], in which each
    class is defined in multiple decision regions delimited by polynomial
    and trigonometric functions (E1, E2, E3 and E4):

    .. math:: \\begin{eqnarray}
        \\label{eq:problem1}
        E1(x) = sin(x) + 5 \\\\
        \\label{eq:problem2}
        E2(x) = (x - 2)^{2} + 1 \\\\
        \\label{eq:problem3}
        E3(x) = -0.1 \\cdot x^{2} + 0.6sin(4x) + 8 \\\\
        \\label{eq:problem4}
        E4(x) = \\frac{(x - 10)^{2}}{2} + 7.902
        \\end{eqnarray}

    Parameters
    ----------
    size_classes : list with the number of samples for each class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    returns
    -------
    X : array of shape = [size_classes, 2]
        The generated data points.

    y : array of shape = [size_classes]
        Class labels associated with each class.

    References
    ----------
    G. Valentini, An experimental bias-variance analysis of svm ensembles
    based on resampling techniques, IEEE Transactions on Systems, Man,
    and Cybernetics, Part B 35 (2005) 1252–1271.

    """
    pass


def make_circle_square(size_classes, random_state=None):
    """Generate the circle square dataset.

    Parameters
    ----------
    size_classes : list with the number of samples for each class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    returns
    -------
    X : array of shape = [size_classes, 2]
        The generated data points.

    y : array of shape = [size_classes]
        Class labels associated with each class.

    References
    ----------
    P. Henniges, E. Granger, R. Sabourin, Factors of overtraining
    with fuzzy artmap neural networks, International Joint Conference
    on Neural Networks (2005) 1075–1080.

    """
    pass


def make_banana(size_classes, na=0.1, random_state=None):
    """Generate the Banana dataset.

    Parameters
    ----------
    size_classes : list with the number of samples for each class.

    na : float (Default = 0.2),
        Noise amplitude. It must be < 1.0

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape = [size_classes, 2]
        The generated data points.

    y : array of shape = [size_classes]
        Class labels associated with each class.

    References
    ----------
    Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms.
    John Wiley & Sons, 2004.

    """
    pass


def make_banana2(size_classes, sigma=1, random_state=None):
    """Generate the Banana dataset similar to the Matlab PRTools toolbox.

    Parameters
    ----------
    size_classes : list with the number of samples for each class.

    sigma : float (Default = 1),
        variance of the normal distribution

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape = [size_classes, 2]
        The generated data points.

    y : array of shape = [size_classes]
        Class labels associated with each class.

    References
    ----------
    R.P.W. Duin, P. Juszczak, D.de Ridder, P. Paclik, E. Pekalska, D.M.Tax,
    Prtools, a matlab toolbox for
    pattern recognition, 2004. URL 〈http://www.prtools.org〉.

    """
    pass


def make_xor(n_samples, random_state=None):
    """Generate the exclusive-or (XOR) dataset.

    Parameters
    ----------
    n_samples : int
                Number of generated data points.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape = [size_classes, 2]
        The generated data points.

    y : array of shape = [size_classes]
        Class labels associated with each class.

    """
    pass
