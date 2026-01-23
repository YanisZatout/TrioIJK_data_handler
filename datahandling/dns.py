""" from .oscillation import Ref, RefData """
from typing import Dict, Type
import numpy as np


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


def ttau_semilocal(ref: Ref) -> Dict[str, float]:
    df = ref.df
    u = df.U.values
    mu = df.MU.values
    rho = df.RHO.values
    h = ref.h
    y = ref.y
    middle = ref.middle
    utau = utau_semilocal(ref)
    ldtdz = df.LAMBDADTDZ.values
    cp = ref.Cp

    cold = ldtdz[:middle] / (rho[:middle] * cp * utau["cold"])
    hot = ldtdz[::-1][:middle] / (rho[::-1][:middle] * cp * utau["hot"])

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
    hot = 2 * mu[-1] * (u[-1] / (2*h - y[-1])) / (rhobulk * ubulk**2)

    return {"hot": hot, "cold": cold}


def nu(ref: Ref) -> Dict[str, float]:
    _bulk = bulk(ref)
    df = ref.df
    u = ref.df["U"].values
    y = ref.y
    h = ref.h
    tw = ref.tw
    tbulk = _bulk["t"]

    ldtdz = df.LAMBDADTDZ.values
    l = df.LAMBDA.values

    cold = 4 * h * ldtdz[0] / (l[0] * np.abs(tbulk - tw["cold"]))
    hot = 4 * h * ldtdz[-1] / (l[-1] * np.abs(tbulk - tw["hot"]))

    return {"hot": hot, "cold": cold}


def retau_semilocal(ref: Ref) -> Dict[str, float]:
    df = ref.df
    u = df.U.values
    mu = df.MU.values
    rho = df.RHO.values
    nu = mu / rho
    h = ref.h
    y = ref.y
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
    return {
        "rho": rhobulk,
        "u": ubulk,
        "t": tbulk,
        "re": rebulk,
        "mu": mubulk,
        "um": um,
        "tm": tm,
    }
