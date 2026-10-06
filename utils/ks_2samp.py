# Functions `mecdf` and `ks_2samp` cloned from:
# https://github.com/o-laurent/multivariate-ks-test (written by Olivier Laurent)
#
# Here we habve modification in ks_2samp that enables to get KS_statistic, KS_pvalue, KS_critical_val
# not only testing hypothesis result (True/False)

from typing import Any, Tuple

import numpy as np


def boundary_for_pval(value):
    return np.max([0, np.min([value, 1])])


def p_value_from_crit_value_one_sample(t_s, n, d):
    return boundary_for_pval(2 * d  * np.exp(-2 * n * (t_s**2)))


def p_value_from_crit_value_two_sample(t_s, n, d):
    # fix: const. Thanks to M. Marszewska
    return boundary_for_pval(2 * d * (n + 1) * np.exp(-n / 2 * (t_s**2)))


def mecdf(x_val: np.ndarray, t: np.ndarray) -> float:
    """Computes the multivariate empirical cdf of x_val at t.

    Args:
        x_val: A numpy array of shape (num_samples_x, dim) representing the sample.
        t: A numpy array of shape (num_samples_t, dim) representing the point at which to evaluate
            the cdf.

    Returns:
        The multivariate empirical cdf of x_val at t.
    """
    lower = (x_val <= t) * 1.0
    return np.mean(np.prod(lower, axis=1))


def ks_2samp(
    x_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float,
    asymptotic: bool = False,
    verbose: bool = False,
) -> Tuple[float, Any, Any, Any]:
    """Performs a multivariate two-sample extension of the Kolmogorov-Smirnov test.

    Args:
        x_val: A numpy array of shape (num_samples_x, dim) representing the first sample.
        y_val: A numpy array of shape (num_samples_y, dim) representing the second sample.
        alpha: The significance level.
        asymptotic: Whether to use the asymptotic approximation or not.
        verbose: Whether to print the test statistic and the critical value.

    Returns:
        A boolean indicating whether the null hypothesis is rejected.
    """
    num_samples_x, dim = x_val.shape
    num_samples_y, num_feats_y = y_val.shape

    if dim != num_feats_y:
        raise ValueError("The two samples do not have the same number of features.")

    z = np.zeros((num_samples_x, dim, dim))
    for h in range(dim):
        ind = np.argsort(x_val[:, h])[::-1]
        temp = np.take(x_val, ind, axis=0)
        z[:, :, h] = temp
        for i in range(dim):
            for j in range(num_samples_x - 1, -1, -1):
                if j == num_samples_x - 1:
                    runmax = temp[num_samples_x - 1, i]
                else:
                    runmax = max(runmax, temp[j, i])
                z[j, i, h] = runmax

    diff = np.zeros((num_samples_x, dim))
    for h in range(dim):
        for i in range(num_samples_x):
            val = np.abs(mecdf(x_val, z[i, :, h]) - mecdf(y_val, z[i, :, h])) * (
                round(num_samples_x * mecdf(x_val, z[i, :, h])) == num_samples_x - i
            )
            diff[i, h] = val
            if h == 0:
                diff[i, h] = max(
                    diff[i, h],
                    np.abs(mecdf(x_val, x_val[i, :]) - mecdf(y_val, x_val[i, :])),
                )
    KS = np.max(diff)

    if asymptotic:
        KS_critical_val = np.sqrt(-np.log(alpha / (4 * dim)) * (0.5 / num_samples_x)) + np.sqrt(
            (-1) * np.log(alpha / (4 * dim)) * (0.5 / num_samples_y)
        )
    else:
        KS_critical_val = np.sqrt(-np.log(alpha / (2 * (num_samples_x + 1) * dim)) * (0.5 / num_samples_x)) + np.sqrt(
            (-1) * np.log(alpha / (2 * (num_samples_y + 1) * dim)) * (0.5 / num_samples_y)
        )

    if verbose:
        print("test statistic: ", KS)
        print("test statistic critical value: ", KS_critical_val)

    return float(KS), KS_critical_val, KS_critical_val < KS, p_value_from_crit_value_two_sample(t_s=KS, n=num_samples_x, d=dim)
