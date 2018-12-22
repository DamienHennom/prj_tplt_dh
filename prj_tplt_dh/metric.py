"""
Metric for ML
"""

import numpy as np


def gini(y_valid, y_pred):
    """Calculate gini coefficient."""
    assert y_valid.shape == y_pred.shape
    n_samples = y_valid.shape[0]

    # Sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_valid, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # Get Lorenz curves
    l_true = np.cumsum(true_order) / np.sum(true_order)
    l_pred = np.cumsum(pred_order) / np.sum(pred_order)
    l_ones = np.linspace(1 / n_samples, 1, n_samples)

    # Get Gini coefficients (area between curves)
    g_true = np.sum(l_ones - l_true)
    g_pred = np.sum(l_ones - l_pred)

    # Normalize to true Gini coefficient
    return g_pred / g_true


def gini_norm(y_valid, y_pred):
    """Calculate normalised gini coefficient."""
    return gini(y_valid, y_pred) / gini(y_valid, y_valid)
