""" from .oscillation import Ref, RefData """

from typing import Dict, Type
import numpy as np
from scipy.integrate import cumulative_trapezoid


Ref = Type["RefData"]


def utau_semilocal(ref: Ref) -> Dict[str, float]:
    df = ref.df
    u = df.U.values
    mu = df.MU.values
    rho = df.RHO.values
    h = ref.h
    y = ref.y
    middle = ref.middle

    tau_cold = mu[0] * u[0] / y[0]
    tau_hot = mu[-1] * u[-1] / (2 * h - y[-1])

    utau_cold = np.sqrt(tau_cold / rho)[:middle]
    utau_hot = np.sqrt(tau_hot / rho)[::-1][:middle]
    return {"cold": utau_cold, "hot": utau_hot}


def trettel_larson(ref):
    df = ref.df
    y = ref.y
    h = ref.h
    middle = ref.middle

    u = df.U.values
    mu = df.MU.values
    rho = df.RHO.values

    rho_w_c = rho[0]
    tau_w_c = mu[0] * u[0] / y[0]
    utau_c = np.sqrt(tau_w_c / rho_w_c)

    y_c = y[:middle]
    u_c = u[:middle]
    rho_c = rho[:middle]
    mu_c = mu[:middle]

    drho_dy_c = np.gradient(rho_c, y_c, edge_order=2)
    dmu_dy_c = np.gradient(mu_c, y_c, edge_order=2)

    kernel_c = np.sqrt(rho_c / rho_w_c) * (
        1 + 0.5 * (y_c / rho_c) * drho_dy_c - (y_c / mu_c) * dmu_dy_c
    )

    utl_c = cumulative_trapezoid(kernel_c, x=u_c, initial=0) / utau_c

    rho_w_h = rho[-1]
    dist_h = 2 * h - y
    tau_w_h = mu[-1] * u[-1] / dist_h[-1]
    utau_h = np.sqrt(tau_w_h / rho_w_h)

    u_h = u[::-1][:middle]
    rho_h = rho[::-1][:middle]
    mu_h = mu[::-1][:middle]
    dist_h_local = dist_h[::-1][:middle]

    drho_dy_h = np.gradient(rho_h, dist_h_local, edge_order=2)
    dmu_dy_h = np.gradient(mu_h, dist_h_local, edge_order=2)

    kernel_h = np.sqrt(rho_h / rho_w_h) * (
        1 + 0.5 * (dist_h_local / rho_h) * drho_dy_h - (dist_h_local / mu_h) * dmu_dy_h
    )

    utl_h = cumulative_trapezoid(kernel_h, x=u_h, initial=0) / utau_h

    return {"cold": utl_c, "hot": utl_h}


def ttau_semilocal(ref: Ref) -> Dict[str, float]:
    df = ref.df
    rho = df.RHO.values
    middle = ref.middle
    utau = utau_semilocal(ref)
    ldtdz = df.LAMBDADTDZ.values
    cp = ref.Cp

    cold = np.abs(ldtdz[0]) / (rho[:middle] * cp * utau["cold"])
    hot = np.abs(ldtdz[-1]) / (rho[::-1][:middle] * cp * utau["hot"])

    return {"hot": hot, "cold": cold}


def y_semilocal(ref: Ref) -> Dict[str, float]:
    df = ref.df
    mu = df.MU.values
    rho = df.RHO.values
    nu = mu / rho
    h = ref.h
    y = ref.y
    middle = ref.middle

    utau = utau_semilocal(ref)

    # print(utau["cold"]*nu[:middle]*y[:middle])
    y_cold = y[:middle] / nu[:middle] * utau["cold"]
    y_hot = (2 * h - y)[::-1][:middle] / nu[::-1][:middle] * utau["hot"]

    return {"hot": y_hot, "cold": y_cold}


def cf(ref: Ref) -> Dict[str, float]:
    _bulk = bulk(ref)
    df = ref.df
    h = ref.h
    u = ref.df["U"].values
    y = ref.y
    mu = df.MU.values
    rhobulk = _bulk["rho"]
    ubulk = _bulk["u"]

    cold = 2 * mu[0] * (u[0] / y[0]) / (rhobulk * ubulk**2)
    hot = 2 * mu[-1] * (u[-1] / (2 * h - y[-1])) / (rhobulk * ubulk**2)

    return {"hot": hot, "cold": cold}


def nu(ref: Ref) -> Dict[str, float]:
    _bulk = bulk(ref)
    df = ref.df
    h = ref.h
    tw = ref.tw
    tbulk = _bulk["t"]

    lbdadtdz = df.LAMBDADTDZ.values
    lbda = df.LAMBDA.values

    cold = 4 * h * lbdadtdz[0] / (lbda[0] * np.abs(tbulk - tw["cold"]))
    hot = 4 * h * lbdadtdz[-1] / (lbda[-1] * np.abs(tbulk - tw["hot"]))

    return {"hot": hot, "cold": cold}


def retau_semilocal(ref: Ref) -> Dict[str, float]:
    df = ref.df
    mu = df.MU.values
    rho = df.RHO.values
    nu = mu / rho
    h = ref.h
    middle = ref.middle

    utau = utau_semilocal(ref)

    retau_hot = utau["hot"][0] * h / nu[-1]
    retau_cold = utau["cold"][0] * h / nu[0]

    hot = (
        np.sqrt(rho[::-1][:middle] / rho[-1]) * (mu[-1] / mu[::-1][:middle]) * retau_hot
    )
    cold = np.sqrt(rho[:middle] / rho[0]) * (mu[0] / mu[:middle]) * retau_cold

    return {"hot": hot, "cold": cold}


def bulk(ref: Ref) -> Dict[str, float]:
    df = ref.df
    h = ref.h
    y = ref.y
    gamma = ref.gamma
    r_air = ref.r_air
    rho = df.RHO.values
    rhobulk = np.trapz(rho, y) / (2 * h)

    u = ref.df["U"].values
    um = np.trapz(u, y) / (2 * h)
    t = ref.df["T"].values
    tm = np.trapz(t, y) / (2 * h)
    """ t = ref.df["RHOUT_MOY"] / ref.df["URHO"] """
    rhou = ref.df["URHO"]
    rhout = ref.df["RHOUT_MOY"].values

    mu = df.MU.values
    mubulk = np.trapz(mu, y) / (2 * h)

    ubulk = np.trapz(rhou, y) / np.trapz(rho, y)
    tbulk = np.trapz(rhout, y) / np.trapz(rhou, y)
    tbulk = np.trapz(rhout, y) / np.trapz(rhou, y)
    rebulk = ubulk * ref.h * rhobulk / mubulk
    mach = ubulk / np.sqrt(gamma * r_air * tbulk)
    return {
        "rho": rhobulk,
        "u": ubulk,
        "t": tbulk,
        "re": rebulk,
        "mu": mubulk,
        "um": um,
        "tm": tm,
        "mach": mach,
    }


def gradu(ref):
    h = ref.h
    y = ref.y
    u = ref.df["U"].values
    return np.gradient([0, *u, 0], [0, *y, 2 * h], edge_order=2)[1:-1]


def gradu_side(ref):
    gu = gradu(ref)
    m = ref.middle
    return {"hot": gu[::-1][:m], "cold": gu[:m]}
