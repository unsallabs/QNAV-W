"""
Allan Deviation Utilities
============================

The industry-standard method for characterizing IMU noise is Allan
variance analysis (IEEE Std 952-2020). This module computes the
(non-overlapping) Allan deviation curve from a sensor output time series,
and is used to validate/visualize the classical vs. quantum sensor noise
stack-up (see tests/ and notebooks/).

    AVAR(tau) = 1/(2*(K-1)) * sum_{i=1}^{K-1} (ybar_{i+1} - ybar_i)^2

    ybar_i : the average measurement value in cluster i (over tau seconds,
             m samples)
    K      : number of clusters = floor(N/m)
"""

import numpy as np


def allan_deviation(data: np.ndarray, dt: float, taus: np.ndarray) -> np.ndarray:
    """Computes the non-overlapping Allan deviation of a time series for the given taus."""
    n = len(data)
    result = np.zeros(len(taus))

    for idx, tau in enumerate(taus):
        m = max(int(round(tau / dt)), 1)
        k = n // m
        if k < 2:
            result[idx] = np.nan
            continue
        clusters = data[: k * m].reshape(k, m).mean(axis=1)
        diffs = np.diff(clusters)
        avar = np.sum(diffs ** 2) / (2.0 * (k - 1))
        result[idx] = np.sqrt(avar)

    return result
