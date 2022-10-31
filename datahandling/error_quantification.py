from typing import List
from interp import natspline1d, eval_all_spline_1d
import numpy as np
import pandas as pd


def compute_eps_pandas(Xles, Xdns, h=15e-3):
    """
    Helper function that takes pandas dataframes and
    sends the data to compute_eps
    """
    les_mesh = Xles.index.to_numpy().astype(np.float64)
    dns_mesh = Xdns.index.to_numpy().astype(np.float64)
    Xles, Xdns = Xles.to_numpy().astype(np.float64), \
        Xdns.to_numpy().astype(np.float64)
    return compute_eps(Xles, Xdns, les_mesh, dns_mesh, h)


def compute_eps(Xles, Xdns, les_mesh, dns_mesh, h=15e-3):
    """
    Computes the error between two given statistics fields where each variable
    was averaged over time and each x, z plane in the channel

    Parameters:
    -----------
    Xles: np.array
        Array of statistics of LES
    Xdns: np.array
        Array of statistics of DNS

    Returns:
    --------
    epsilon: Union[float, double]
        Quantification of error between the LES and DNS quantity
    """
    delta = h
    nles, ndns = Xles.shape[0], Xdns.shape[0]
    half_nles = nles//2
    spline = natspline1d(dns_mesh, Xdns, ndns)
    Xdns_interp = eval_all_spline_1d(
        les_mesh,
        dns_mesh,
        ndns,
        Xdns,
        spline
    )
    half_length_les, half_length_dns = Xles[:half_nles], \
        Xdns_interp[:half_nles]

    half_length_les_rev, half_length_dns_rev = Xles[half_nles:], \
        Xdns_interp[half_nles:]

    # log(y_{i+1} / y_i)
    log_mesh = np.log(les_mesh[1:]/les_mesh[:-1])[:half_nles]
    # log(\frac{2\delta - y_{i+1}}{2-y_i})
    log_height = np.log(
        (2*delta - les_mesh[1:]) / (2*delta - les_mesh[:-1])
    )[:half_nles]

    sum1 = np.sum(
        log_mesh * np.abs((half_length_les - half_length_dns)
                          * half_length_les)
    ) / \
        np.sum(log_mesh * half_length_dns**2)

    sum2 = np.sum(
        log_height * np.abs(
            (half_length_les_rev - half_length_dns_rev) * half_length_les_rev
        )
    ) / \
        np.sum(log_height * half_length_dns_rev ** 2)

    return sum1 + sum2


def assess_mean_error_across_quantities(
    quantities: List[str],
    LES: pd.DataFrame,
    DNS: pd.DataFrame,
    h=15e-3
):
    """
    Assesses the mean error across all quantities given in LES and DNS
    Parameters:
    -----------
    quantities: List[str]
        String of quantities we want to assess the error of
    LES: pd.DataFrame
        Pandas DataFrame of averaged LES statistics across the XZ plane
    LES: pd.DataFrame
        Pandas DataFrame of averaged DNS statistics across the XZ plane

    Returns:
    --------
    epsilon: Tuple[float, List[float]]
        Quantification of error between the LES and DNS quantity and their
        respective errors across all desired quantities
    """
    qt_LES = [LES[q] for q in quantities]
    qt_DNS = [DNS[q] for q in quantities]
    eps = []
    for les, dns in zip(qt_LES, qt_DNS):
        epsilon_qt = compute_eps_pandas(les, dns, h)
        eps.append(epsilon_qt)
    return np.mean(eps), eps
