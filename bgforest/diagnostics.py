"""
MCMC Convergence and Diagnostic Metrics
=======================================
Implements Gelman-Rubin R-hat, Effective Sample Size (ESS),
and trace plot statistics for evaluating MCMC chain convergence.
"""

from typing import List, Union

import numpy as np


def compute_gelman_rubin_rhat(chains: List[List[float]]) -> float:
    """
    Compute Gelman-Rubin potential scale reduction factor (R-hat) across multiple chains.

    R-hat < 1.05 indicates adequate convergence across chains.

    Parameters
    ----------
    chains : List[List[float]]
        List of MCMC trace outputs from independent chains.

    Returns
    -------
    rhat : float
        Potential scale reduction factor.
    """
    m = len(chains)
    if m < 2:
        raise ValueError("Gelman-Rubin diagnostic requires at least 2 independent MCMC chains.")

    n = min(len(chain) for chain in chains)
    if n <= 1:
        return 1.0

    # Truncate chains to equal length n
    chain_matrix = np.array([chain[:n] for chain in chains], dtype=np.float64)

    # Chain means and variances
    chain_means = np.mean(chain_matrix, axis=1)
    chain_vars = np.var(chain_matrix, axis=1, ddof=1)

    # Between-chain variance B and Within-chain variance W
    grand_mean = np.mean(chain_means)
    B = (n / (m - 1.0)) * np.sum((chain_means - grand_mean) ** 2)
    W = np.mean(chain_vars)

    if W == 0:
        return 1.0

    # Estimated marginal posterior variance
    var_plus = ((n - 1.0) / n) * W + (1.0 / n) * B
    rhat = float(np.sqrt(max(1.0, var_plus / W)))

    return rhat


def compute_effective_sample_size(trace: Union[List[float], np.ndarray]) -> float:
    """
    Estimate Effective Sample Size (ESS) for an MCMC trace using autocorrelation.

    Parameters
    ----------
    trace : List[float] or np.ndarray
        Single chain scalar trace output (e.g. log-posterior).

    Returns
    -------
    ess : float
        Estimated number of independent posterior samples.
    """
    trace_arr = np.asarray(trace, dtype=np.float64)
    N = len(trace_arr)
    if N <= 1:
        return float(N)

    # Mean center
    centered = trace_arr - np.mean(trace_arr)
    var = np.var(trace_arr)
    if var == 0:
        return float(N)

    # Compute autocorrelation via FFT
    n = 1 << (2 * N - 1).bit_length()
    fft = np.fft.fft(centered, n=n)
    autocorr = np.fft.ifft(fft * np.conj(fft)).real[:N]
    autocorr /= var * N

    # Sum autocorrelations until first negative lag pair
    sum_autocorr = 0.0
    for i in range(1, N - 1, 2):
        pair_sum = autocorr[i] + autocorr[i + 1]
        if pair_sum < 0:
            break
        sum_autocorr += pair_sum

    act = 1.0 + 2.0 * sum_autocorr
    ess = N / act
    return float(max(1.0, ess))
