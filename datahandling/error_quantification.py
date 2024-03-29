from typing import List
from scipy.interpolate import CubicSpline
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
    nles = Xles.shape[0]
    half_nles = nles//2
    spline = CubicSpline(dns_mesh, Xdns)
    Xdns_interp = spline(les_mesh)
    half_length_les, half_length_dns = Xles[:
                                            half_nles], Xdns_interp[:half_nles]

    half_length_les_rev, half_length_dns_rev = Xles[half_nles:], \
        Xdns_interp[half_nles:]

    # log(y_{i+1} / y_i)
    log_mesh = np.log(les_mesh[1:]/les_mesh[:-1])[:half_nles]
    # log(\frac{2\delta - y_{i+1}}{2-y_i})
    log_height = np.log(
        (2*delta - les_mesh[1:]) / (2*delta - les_mesh[:-1])
    )[-half_nles:]

    sum1 = np.sum(
        log_mesh * np.abs((half_length_les - half_length_dns)
                          * half_length_les)
    ) / np.sum(log_mesh * half_length_dns**2)

    sum2 = np.sum(
        log_height * np.abs(
            (half_length_les_rev - half_length_dns_rev) * half_length_les_rev
        )
    ) / np.sum(log_height * half_length_dns_rev ** 2)

    return sum1 + sum2

def compute_error_bars(df_les, df_dns, Cp=1155, *, delta: float = 0.0029846):
    h = delta
    from .quantities_of_interest import compute_rms_quantities
    rms_les, rms_dns = compute_rms_quantities(df_les, df_dns, Cp)
    urms_dns      = rms_dns["urms"]
    vrms_dns      = rms_dns["vrms"]
    wrms_dns      = rms_dns["wrms"]
    uv_dns        = rms_dns["uv"]
    u_theta_dns   = rms_dns["u_theta"]
    v_theta_dns   = rms_dns["v_theta"]
    theta_rms_dns = rms_dns["theta_rms"]

    urms_les      = rms_les["urms"]      
    vrms_les      = rms_les["vrms"]      
    wrms_les      = rms_les["wrms"]      
    uv_les        = rms_les["uv"]        
    u_theta_les   = rms_les["u_theta"]   
    v_theta_les   = rms_les["v_theta"]   
    theta_rms_les = rms_les["theta_rms"] 
    eps_u    = [compute_eps(df["U"].values.astype("float64"),          df_dns["U"].values.astype("float64"),          df.index.values.astype("float64"), df_dns.index.values.astype("float64"), h=delta)*100 for df in df_les]
    eps_v    = [compute_eps(df["W"].values.astype("float64"),          df_dns["W"].values.astype("float64"),          df.index.values.astype("float64"), df_dns.index.values.astype("float64"), h=delta)*100 for df in df_les]
    eps_t    = [compute_eps(df["T"].values.astype("float64"),          df_dns["T"].values.astype("float64"),          df.index.values.astype("float64"), df_dns.index.values.astype("float64"), h=delta)*100 for df in df_les]
    eps_phi  = [compute_eps(df["LAMBDADTDZ"].values.astype("float64"), df_dns["LAMBDADTDZ"].values.astype("float64"), df.index.values.astype("float64"), df_dns.index.values.astype("float64"), h=delta)*100 for df in df_les]
    eps_urms = [compute_eps(np.sqrt(np.abs(urms     .values.astype("float64"))), np.sqrt(np.abs(urms_dns     .values.astype("float64"))), df.index.values.astype("float64"), df_dns.index.values.astype("float64"), h=delta)*100 for df, urms      in zip(df_les, urms_les     )]
    eps_vrms = [compute_eps(np.sqrt(np.abs(vrms     .values.astype("float64"))), np.sqrt(np.abs(vrms_dns     .values.astype("float64"))), df.index.values.astype("float64"), df_dns.index.values.astype("float64"), h=delta)*100 for df, vrms      in zip(df_les, vrms_les     )]
    eps_wrms = [compute_eps(np.sqrt(np.abs(wrms     .values.astype("float64"))), np.sqrt(np.abs(wrms_dns     .values.astype("float64"))), df.index.values.astype("float64"), df_dns.index.values.astype("float64"), h=delta)*100 for df, wrms      in zip(df_les, wrms_les     )]
    eps_uv   = [compute_eps(np.sqrt(np.abs(uv       .values.astype("float64"))), np.sqrt(np.abs(uv_dns       .values.astype("float64"))), df.index.values.astype("float64"), df_dns.index.values.astype("float64"), h=delta)*100 for df, uv        in zip(df_les, uv_les       )]
    eps_ut   = [compute_eps(np.sqrt(np.abs(u_theta  .values.astype("float64"))), np.sqrt(np.abs(u_theta_dns  .values.astype("float64"))), df.index.values.astype("float64"), df_dns.index.values.astype("float64"), h=delta)*100 for df, u_theta   in zip(df_les, u_theta_les  )]
    eps_vt   = [compute_eps(np.sqrt(np.abs(v_theta  .values.astype("float64"))), np.sqrt(np.abs(v_theta_dns  .values.astype("float64"))), df.index.values.astype("float64"), df_dns.index.values.astype("float64"), h=delta)*100 for df, v_theta   in zip(df_les, v_theta_les  )]
    eps_trms = [compute_eps(np.sqrt(np.abs(theta_rms.values.astype("float64"))), np.sqrt(np.abs(theta_rms_dns.values.astype("float64"))), df.index.values.astype("float64"), df_dns.index.values.astype("float64"), h=delta)*100 for df, theta_rms in zip(df_les, theta_rms_les)]

    eps_mean = {
       "u": eps_u,
       "v": eps_v,
       "t": eps_t,
       "phi": eps_phi
    }
    eps_rms = {
        "urms":  eps_urms,
        "vrms":  eps_vrms,
        "wrms":  eps_wrms,
        "uv":    eps_uv  ,
        "ut":    eps_ut  ,
        "vt":    eps_vt  ,
        "trms":  eps_trms,
    }
    return eps_mean, eps_rms






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


def error_profile_quantification_order_1(x, y):
    error = np.abs(x - y)/x.max()
    return np.mean(error, axis=(0, 1))


def error_profile_quantification_general(error):
    return np.mean(error, axis=(0, 1))


def compute_rms(x):
    return np.mean(x**2, axis=(0, 1)) - np.mean(x, axis=(0, 1))**2


def compute_rms_diff(x, y):
    xrms = compute_rms(x)
    yrms = compute_rms(y)
    return np.abs((xrms-yrms)/xrms.max())


def compute_gradient_error(x, y, mesh):
    gradx = np.gradient(x, *mesh, edge_order=2)
    grady = np.gradient(y, *mesh, edge_order=2)
    return [error_profile_quantification_order_1(gx, gy)
            for gx, gy in zip(gradx, grady)]


def compute_all_errors(x, y, mesh):
    error_profile = error_profile_quantification_order_1(x, y)
    rms_profile = compute_rms(y)
    gradient_profile = compute_gradient_error(x, y, mesh)
    grady = np.gradient(y, *mesh, edge_order=2)
    rms_gradient_profile = [compute_rms(gradi) for gradi in grady]
    return error_profile, rms_profile, gradient_profile, rms_gradient_profile


def friction_quantities_les(df_les: List[pd.DataFrame], delta):
    """
    Computes the adim quantities of interest from a list of pandas dataframes
    df_les: List[pd.DataFrame]
        List of dataframes of statistics after steady state is achieved
    delta: float
        Canal half height
    Returns:
    u_tau:
        Friction velocity
    re_tau:
        Friction Reynolds number
    t_tau:
        Friction temperature
    y_plus:
        Adim wall normal distance
    """
    utau_cold_les = [
        np.sqrt(df["NU"].iloc[0] * df["U"].iloc[0] /
                df["coordonnee_K"].iloc[0]) for df in df_les
    ]
    utau_hot_les = [
        np.sqrt(df["NU"].iloc[-1] * df["U"].iloc[-1] /
                (2*delta - df["coordonnee_K"].iloc[-1])) for df in df_les
    ]

    re_cold_les = [
        h/df["NU"].iloc[0] * utau for df,
        utau in zip(df_les, utau_cold_les)
    ]
    re_hot_les = [
        h/df["NU"].iloc[-1] * utau for df,
        utau in zip(df_les, utau_hot_les)
    ]

    ttau_cold_les = [
        df["LAMBDADTDZ"].iloc[0]/(df["RHO"].iloc[0]*Cp*utau) for df, utau in zip(df_les, utau_cold_les)
    ]
    ttau_hot_les = [
        df["LAMBDADTDZ"].iloc[-1]/(df["RHO"].iloc[-1]*Cp*utau) for df, utau in zip(df_les, utau_hot_les)
    ]
    # u_center_les
    y_plus_cold = [
        (df.index*utau/df["NU"]).iloc[:len(df["coordonnee_K"])//2]
        for df, utau in zip(df_les, utau_cold_les)
    ]

    y_plus_hot = [
        ((2*delta - df.index)*utau / df["NU"]
         ).iloc[-len(df["coordonnee_K"])//2:]
        for df, utau in zip(df_les, utau_cold_les)
    ]
    return utau_cold_les, utau_hot_les, re_cold_les, re_hot_les, ttau_cold_les, ttau_hot_les, y_plus_cold, y_plus_hot


def friction_quantities_dns(df_les: List[pd.DataFrame], delta):
    """
    Computes the adim quantities of interest from a list of pandas dataframes
    df_les: List[pd.DataFrame]
        List of dataframes of statistics after steady state is achieved
    delta: float
        Canal half height
    Returns:
    u_tau:
        Friction velocity
    re_tau:
        Friction Reynolds number
    t_tau:
        Friction temperature
    y_plus:
        Adim wall normal distance
    """
    utau_cold_les = [
        np.sqrt(df["NU"].iloc[0] * df["U"].iloc[0] /
                df["coordonnee_K"].iloc[0]) for df in df_les
    ]
    utau_hot_les = [
        np.sqrt(df["NU"].iloc[-1] * df["U"].iloc[-1] /
                (2*delta - df["coordonnee_K"].iloc[-1])) for df in df_les
    ]

    re_cold_les = [
        h/df["NU"].iloc[0] * utau for df,
        utau in zip(df_les, utau_cold_les)
    ]
    re_hot_les = [
        h/df["NU"].iloc[-1] * utau for df,
        utau in zip(df_les, utau_hot_les)
    ]

    ttau_cold_les = [
        df["LAMBDADTDZ"].iloc[0]/(df["RHO"].iloc[0]*Cp*utau) for df, utau in zip(df_les, utau_cold_les)
    ]
    ttau_hot_les = [
        df["LAMBDADTDZ"].iloc[-1]/(df["RHO"].iloc[-1]*Cp*utau) for df, utau in zip(df_les, utau_hot_les)
    ]
    # u_center_les
    y_plus_cold = [
        (df.index*utau/df["NU"]).iloc[:len(df.index)//2]
        for df, utau in zip(df_les, utau_cold_les)
    ]

    y_plus_hot = [
        ((2*delta - df.index)*utau / df["NU"]).iloc[-len(df.index)//2:]
        for df, utau, re in zip(df_les, utau_cold_les, re_hot_les)
    ]
    return utau_cold_les[0], utau_hot_les[0], re_cold_les[0], re_hot_les[0], ttau_cold_les[0], ttau_hot_les[0], y_plus_cold[0], y_plus_hot[0]


def adim_les_quantities(df: pd.DataFrame, delta):
    utau_cold_les, \
        utau_hot_les, \
        re_cold_les, \
        re_hot_les, \
        ttau_cold_les, \
        ttau_hot_les, \
        y_plus_cold, \
        y_plus_hot = friction_quantities_les(df, delta)
    half_len = [len(dataframe)//2 for dataframe in df]
    uplus_cold = [dataframe["U"]/utau for dataframe,
                  utau in zip(df, utau_cold_les)]
    uplus_hot = [dataframe["U"]/utau for dataframe in zip(df, utau_hot_les)]
