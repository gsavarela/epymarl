import math

import numpy as np
import matplotlib.pyplot as plt


def z_table(confidence):
    """Hand-coded Z-Table

    Parameters
    ----------
    confidence: float
        The confidence level for the z-value.

    Returns
    -------
        The z-value for the confidence level given.
    """
    return {
        0.99: 2.576,
        0.95: 1.96,
        0.90: 1.645
    }[confidence]


def confidence_interval(mean, n, confidence):
    """Computes the confidence interval of a sample.

    Parameters
    ----------
    mean: float
        The mean of the sample
    n: int
        The size of the sample
    confidence: float
        The confidence level for the z-value.

    Returns
    -------
        The confidence interval.
    """
    return z_table(confidence) * (mean / math.sqrt(n))


def standard_error(std_dev, n, confidence):
    """Computes the standard error of a sample.

    Parameters
    ----------
    std_dev: float
        The standard deviation of the sample
    n: int
        The size of the sample
    confidence: float
        The confidence level for the z-value.

    Returns
    -------
        The standard error.
    """
    return z_table(confidence) * (std_dev / math.sqrt(n))
