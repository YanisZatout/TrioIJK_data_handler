from typing import Tuple, Dict, Type, Union
import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator

from datahandling.dns import ttau_semilocal
from .dataloader import DataLoaderPandas
from .dns import utau_semilocal, y_semilocal, retau_semilocal, bulk, cf, nu
import pandas as pd
from numpy import typing as npt

Ref = Type["RefData"]


def ref_values(ref: pd.DataFrame, /, Cp, h, Tw) -> Tuple[Dict]:
    if not isinstance(Tw, dict):
        raise ValueError(f"Tw must be a dict. A {type(Tw)} was given")
    hot = -1
    cold = 0

    utau_temp = np.sqrt(
        ref["MU"] / ref["RHO"] * np.abs(np.gradient(ref["U"], ref.index, edge_order=2))
    ).values
    utau = {"hot": utau_temp[-1], "cold": utau_temp[0]}

    y = ref.index.values
    middle = len(y) // 2

    y = ref.index.values
    y_plus_hot = (2 * h - y[middle:])[::-1] * utau["hot"] / ref["NU"].iloc[hot]
    y_plus_cold = y[:middle] * utau["cold"] / ref["NU"].iloc[cold]

    yplus = {"hot": y_plus_hot, "cold": y_plus_cold}

    retau_temp = (utau_temp * h * ref["RHO"] / ref["MU"]).values

    rho_bulk = rhobulk = np.trapz(ref["RHO"], ref.index) / (2 * h)
    rho_m = rho_bulk
    mu_bulk = mubulk = np.trapz(ref["MU"], ref.index) / (2 * h)
    u_bulk = ubulk = np.trapz(ref["URHO"], ref.index) / np.trapz(ref["RHO"], ref.index)
    u_m = np.trapz(ref["U"], ref.index) / (2 * h)
    t_bulk = np.trapz(ref["RHOUT_MOY"], ref.index) / np.trapz(ref["URHO"], ref.index)
    re_bulk = ubulk * h * rhobulk / mubulk

    Tw_hot = Tw["hot"]
    Tw_cold = Tw["cold"]

    theta_hot_temp = (ref["T"].values[::-1][:middle] - Tw_hot) / (t_bulk - Tw_hot)
    theta_cold_temp = (ref["T"].values[:middle] - Tw_cold) / (t_bulk - Tw_cold)
    theta = {"hot": theta_hot_temp, "cold": theta_cold_temp}

    thetatau_temp = (ref["LAMBDADTDZ"] / (ref["RHO"] * Cp * utau_temp)).values
    phi_tau = {"hot": thetatau_temp[-1], "cold": thetatau_temp[0]}

    retau = {"hot": retau_temp[-1], "cold": retau_temp[0]}
    thetatau = {"hot": thetatau_temp[-1], "cold": thetatau_temp[0]}

    nusselt_hot_temp = np.abs(
        4
        * h
        * ref["LAMBDADTDZ"].iloc[-1]
        / (ref["LAMBDA"].iloc[-1] * (t_bulk - Tw_hot))
    )
    nusselt_cold_temp = np.abs(
        4 * h * ref["LAMBDADTDZ"].iloc[0] / (ref["LAMBDA"].iloc[0] * (t_bulk - Tw_cold))
    )

    nusselt = {"hot": nusselt_hot_temp, "cold": nusselt_cold_temp}

    Cf_hot_temp = (
        2
        * ref["MU"].values[hot]
        / (rho_m * u_m**2)
        * np.gradient(
            ref["U"].values[::-1][:4], ref.index.values[::-1][:4], edge_order=2
        )[0]
    )
    Cf_cold_temp = (
        2
        * ref["MU"].values[cold]
        / (rho_m * u_m**2)
        * np.gradient(ref["U"].values[:4], ref.index.values[:4], edge_order=2)[0]
    )

    Cf = {"hot": Cf_hot_temp, "cold": Cf_cold_temp}

    return (
        h,
        utau,
        thetatau,
        retau,
        rho_bulk,
        u_bulk,
        t_bulk,
        re_bulk,
        theta,
        utau,
        retau,
        nusselt,
        Cf,
        y,
        yplus,
        ref,
        mu_bulk,
        phi_tau,
    )


class RefData(object):
    def __init__(
        self,
        path: str,
        /,
        Cp: float = 0,
        h: float = 0,
        Tw: Dict[str, float] = dict(list([])),
    ) -> None:
        assert Cp != 0, "Cp must be provided as a floating point value"
        assert h != 0, "h must be provided as a floating point value"
        assert Tw != dict(
            list([])
        ), "You must provide the boundary temperatures as a dict of floating point values"
        self.path = path
        self.Cp = self.cp = Cp
        self.h = h
        self.Tw = self.tw = Tw
        self.gamma = 1.4
        self.r_air = self.Cp * (1 - 1 / self.gamma)

    def load(self, path=None, Cp=None):
        if not path:
            path = self.path
        if not Cp:
            Cp = self.Cp
        ref_df = DataLoaderPandas(path, "statistiques").load_last()
        self.df = ref_df
        self.y = y = self.df.index.values
        self.middle = len(y) // 2
        self.ref_values()
        self.sheer = {
            "utau": self.utau,
            "retau": self.retau,
            "Cf": self.Cf,
            "ttau": self.ttau,
        }
        self.wall = {"T": self.Tw}

    def __repr__(self):
        return (
            f"bulk quantities {self.bulk}\n"
            f"sheer quantities {self.sheer}\n"
            f"wall quantities {self.wall}\n"
        )

    def ref_values(self) -> None:
        self.utau_star = utau_semilocal(self)
        self.utau = {"hot": self.utau_star["hot"][0], "cold": self.utau_star["cold"][0]}
        self.y_star = y_semilocal(self)
        self.retau_star = retau_semilocal(self)
        self.retau = {
            "hot": self.retau_star["hot"][0],
            "cold": self.retau_star["cold"][0],
        }
        self.ttau_star = ttau_semilocal(self)
        self.ttau = {"hot": self.ttau_star["hot"][0], "cold": self.ttau_star["cold"][0]}
        self.Cf = self.cf = cf(self)
        self.Nu = self.nu = nu(self)
        self.bulk = bulk(self)
        df = self.df
        r_air = self.r_air
        self.p0 = self.P0 = (df["T"] * df["RHO"] * r_air).iloc[-1]
        return


def osc_post_treat(df, ref) -> Tuple[Union[npt.ArrayLike, pd.DataFrame]]:
    bulk = ref.bulk
    Tw = ref.Tw
    Tw_hot = Tw["hot"]
    Tw_cold = Tw["cold"]
    times = [d.index.get_level_values(0).unique() for d in df]
    t_plus = tplus = [t * (ref.retau["hot"] * ref.utau["hot"]) / ref.h for t in times]
    theta_hot = [((d["T"] - ref.Tw["hot"]) / (bulk["t"] - ref.Tw["hot"])) for d in df]

    theta_cold = [
        ((d["T"] - ref.Tw["cold"]) / (bulk["t"] - ref.Tw["cold"])) for d in df
    ]
    theta = {"hot": theta_hot, "cold": theta_cold}
    y = [dataframe.index.get_level_values(1).unique() for dataframe in df]
    h = ref.h
    tb = [
        dataframe["RHOUT_MOY"].groupby("time").apply(lambda x: np.trapezoid(x, yy))
        / dataframe["URHO"].groupby("time").apply(lambda x: np.trapezoid(x, yy))
        for dataframe, yy in zip(df, y)
    ]
    rb = [
        dataframe["URHO"].groupby("time").apply(lambda x: np.trapezoid(x, yy))
        / dataframe["U"].groupby("time").apply(lambda x: np.trapezoid(x, yy))
        for dataframe, yy in zip(df, y)
    ]
    ub = [
        dataframe["U"].groupby("time").apply(lambda x: np.trapezoid(x, yy)) / (2 * h)
        for dataframe, yy in zip(df, y)
    ]
    Nu_hot = [
        np.abs(
            4
            * h
            * dataframe["LAMBDADTDZ"].groupby("time").apply(lambda x: x.iloc[-1])
            / (
                dataframe["LAMBDA"].groupby("time").apply(lambda x: x.iloc[-1])
                * (tbulk - Tw_hot)
            )
        )
        for dataframe, tbulk in zip(df, tb)
    ]
    Nu_cold = [
        np.abs(
            4
            * h
            * dataframe["LAMBDADTDZ"].groupby("time").apply(lambda x: x.iloc[0])
            / (
                dataframe["LAMBDA"].groupby("time").apply(lambda x: x.iloc[0])
                * (tbulk - Tw_cold)
            )
        )
        for dataframe, tbulk in zip(df, tb)
    ]

    Nu_hot_norm = [nu / nu.iloc[0] for nu in Nu_hot]
    Nu_cold_norm = [nu / nu.iloc[0] for nu in Nu_cold]

    Nu = {"hot": Nu_hot, "cold": Nu_cold}
    Nu_norm = {"hot": Nu_hot_norm, "cold": Nu_cold_norm}

    tplus = t_plus = [t - t[0] for t in tplus]

    Cf_act_hot = [
        2
        * d["MU"].groupby("time").apply(lambda x: x.iloc[-1])
        * d["U"]
        .groupby("time")
        .apply(
            lambda x: np.gradient(x.values[::-1][:4], ref.y[::-1][:4], edge_order=2)[0]
        )
        / (rho_bulk * (u_bulk**2))
        for d, rho_bulk, u_bulk in zip(df, rb, ub)
    ]
    Cf_act_cold = [
        2
        * d["MU"].groupby("time").apply(lambda x: x.iloc[0])
        * d["U"]
        .groupby("time")
        .apply(lambda x: np.gradient(x.values[:4], ref.y[:4], edge_order=2)[0])
        / (rho_bulk * (u_bulk**2))
        for d, rho_bulk, u_bulk in zip(df, rb, ub)
    ]

    Cf_act_norm_hot = [cf / cf.iloc[0] for cf in Cf_act_hot]
    Cf_act_norm_cold = [cf / cf.iloc[0] for cf in Cf_act_cold]
    Cf_act = {"hot": Cf_act_hot, "cold": Cf_act_cold}
    Cf_act_norm = {"hot": Cf_act_norm_hot, "cold": Cf_act_norm_cold}

    return theta, Nu, Nu_norm, Cf_act, Cf_act_norm, tplus


from typing import Union


class Osc(object):
    def __init__(self, df, theta, Nu, Cf, tplus):
        self.df = df
        self.theta = theta
        self.Nu = self.nusselt = Nu
        self.Cf = self.cf = Cf
        self.t = self.tplus = tplus

    def __len__(self):
        return len(self.t)

    def compute_analogy_factor(self):
        analogy_factor_hot = [a / b for a, b in zip(self.Nu["hot"], self.Cf["hot"])]
        analogy_factor_cold = [a / b for a, b in zip(self.Nu["cold"], self.Cf["cold"])]
        self.analogy_factor = {"hot": analogy_factor_hot, "cold": analogy_factor_cold}
        return self.analogy_factor


def phase_average_boundary(
    qty, tplus, freq: float, *, ncycles=float(5), sampling_rate=64
) -> npt.ArrayLike:
    """
    Returns phase average of given quantity
    """
    t = tplus
    cycles = t / freq  # Each cycle is easy to retreive
    last_n_cycles = cycles >= cycles[-1] - ncycles  # We take the last n cycles

    relevent = cycles[last_n_cycles]
    relevent -= relevent[0]  # Make time start at 0
    linear_time = np.linspace(
        0, relevent[-1], int(ncycles * sampling_rate)
    )  # We sample linearly
    cs = CubicSpline(relevent, qty.iloc[last_n_cycles])  # Compute cubic splines
    interpolated = cs(linear_time)  # Interpolate
    # then reshape in shape (ncycles, -1),
    # such that we only have to average along one axis
    average = interpolated.reshape(int(ncycles), -1).T.mean(1)

    return average


def phase_average_field(data, tplus, y, freq, *, ncycles=float(5), sampling_rate=64):
    cycles = tplus / freq
    last_n_cycles = cycles >= cycles[-1] - ncycles  # We take the last n cycles
    relevent = cycles[last_n_cycles]
    relevent -= relevent[0]  # Make time start at 0
    linear_time = np.linspace(
        0, relevent[-1], int(ncycles * sampling_rate)
    )  # We sample linearly

    nk = y.shape[0]
    time, height = np.meshgrid(linear_time, y, indexing="ij", sparse=True)
    interpolator = RegularGridInterpolator((cycles, y), data.values.reshape(-1, len(y)))
    interpolated = interpolator((time, height))
    # nt, ncycles, nk
    return interpolated, interpolated.reshape(-1, int(ncycles), nk, order="F").mean(1)


def plot_uv_ut_fig(
    ref,
    uv,
    vt,
    raw_uv,
    raw_vt,
    unactuated_uv,
    unactuated_vt,
    *,
    figsize=(28, 12),
    sharex=True,
    sharey=False,
    Re=180,
    title=r"$p_{eriod}^+=500, V^+=30$",
):
    fig, ax = plt.subplots(2, 2, figsize=(28, 12), sharex=True)  # , sharey=True)
    lines = []

    ####### HOT
    (line,) = ax[0, 0].semilogx(
        ref.yplus["hot"],
        uv["hot"][idx_min_af],
        color="lightcoral",
        label=r"$\cdot ''$ hot",
    )
    lines.append(line)
    ax[0, 1].semilogx(ref.yplus["hot"], uv["hot"][idx_max_af], color="lightcoral")
    ax[1, 0].semilogx(ref.yplus["hot"], vt["hot"][idx_min_af], color="lightcoral")
    ax[1, 1].semilogx(ref.yplus["hot"], vt["hot"][idx_max_af], color="lightcoral")

    ####### COLD
    (line,) = ax[0][0].semilogx(
        ref.yplus["cold"],
        uv["cold"][idx_min_af],
        color="lightblue",
        label=r"$\cdot ''$ cold",
    )
    lines.append(line)
    ax[0][1].semilogx(ref.yplus["cold"], uv["cold"][idx_max_af], color="lightblue")
    ax[1][0].semilogx(ref.yplus["cold"], vt["cold"][idx_min_af], color="lightblue")
    ax[1][1].semilogx(ref.yplus["cold"], vt["cold"][idx_max_af], color="lightblue")

    ####### AVERAGE OSCILLATION
    (line,) = ax[0][0].semilogx(
        ref.yplus["hot"],
        raw_uv["hot"].mean(0),
        "--",
        color="red",
        label=r"$\overline{\cdot}$ hot",
    )
    lines.append(line)
    ax[0][1].semilogx(ref.yplus["hot"], raw_uv["hot"].mean(0), "--", color="red")
    ax[1][0].semilogx(ref.yplus["hot"], raw_vt["hot"].mean(0), "--", color="red")
    ax[1][1].semilogx(ref.yplus["hot"], raw_vt["hot"].mean(0), "--", color="red")

    (line,) = ax[0][0].semilogx(
        ref.yplus["cold"],
        raw_uv["cold"].mean(0),
        "--",
        color="blue",
        label=r"$\overline{\cdot}$ cold",
    )
    lines.append(line)
    ax[0][1].semilogx(ref.yplus["cold"], raw_uv["cold"].mean(0), "--", color="blue")
    ax[1][0].semilogx(ref.yplus["cold"], raw_vt["cold"].mean(0), "--", color="blue")
    ax[1][1].semilogx(ref.yplus["cold"], raw_vt["cold"].mean(0), "--", color="blue")

    ####### NO OSCILLATION

    (line,) = ax[0][0].semilogx(
        ref.yplus["hot"],
        unactuated_uv["hot"],
        ".",
        color="darkred",
        markersize=2,
        label="unactuated hot",
    )
    lines.append(line)
    ax[0][1].semilogx(
        ref.yplus["hot"], unactuated_uv["hot"], ".", color="darkred", markersize=2
    )
    ax[1][0].semilogx(
        ref.yplus["hot"], unactuated_vt["hot"], ".", color="darkred", markersize=2
    )
    ax[1][1].semilogx(
        ref.yplus["hot"], unactuated_vt["hot"], ".", color="darkred", markersize=2
    )

    (line,) = ax[0][0].semilogx(
        ref.yplus["cold"],
        unactuated_uv["cold"],
        ".",
        color="darkblue",
        markersize=2,
        label="unactuated cold",
    )
    lines.append(line)
    ax[0][1].semilogx(
        ref.yplus["cold"], unactuated_uv["cold"], ".", color="darkblue", markersize=2
    )
    ax[1][0].semilogx(
        ref.yplus["cold"], unactuated_vt["cold"], ".", color="darkblue", markersize=2
    )
    ax[1][1].semilogx(
        ref.yplus["cold"], unactuated_vt["cold"], ".", color="darkblue", markersize=2
    )

    ####### LEGEND
    ax[1][0].set_xlabel(r"$y^+$")
    ax[1][1].set_xlabel(r"$y^+$")

    ax[0][0].set_ylabel(r"$\widetilde{u'' v''}^+(\frac{y^+}{Re_\tau} - 1)$")
    ax[1][0].set_ylabel(r"$\widetilde{\Theta'' v''}^+ (\frac{y^+}{Re_\tau} - 1)$")

    ax[0][0].set_title(r"$C_{f, \ max}$", y=1.1)
    ax[0][1].set_title(r"$C_{f, \ min}$", y=1.1)

    for a in ax.ravel():
        a.grid(which="both")

    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc="center", bbox_to_anchor=(0.99, 0.5))

    fig.suptitle(title)

    fig.text(0.1, 1.01, rf"$Re_\tau={Re}$")
    plt.subplots_adjust(hspace=0.05, wspace=0.1)
