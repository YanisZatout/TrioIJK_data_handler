from interp import natspline1d, eval_all_spline_1d
import numpy as np


def compute_eps(Xles, Xdns, h=15e-1):
    """
    Computes the error between two given statistics fields where each variable
    was averaged over time and each x, z plane in the channel

    Parameters:
    -----------
    Xles: np.ndarray
        Array of statistics of LES
    Xdns: np.ndarray
        Array of statistics of DNS

    Returns:
    --------
    epsilon: Union[float, double]
        Quantification of error between the LES and DNS quantity
    """
    delta = h
    nles, ndns = Xles.shape[0], Xdns.shape[0]
    half_nles = nles//2
    les_mesh, dns_mesh = loader_les.space, loader_dns.space
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
