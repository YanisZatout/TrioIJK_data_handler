from typing import Tuple, Dict, Type, Union
import numpy as np
from .dataloader import DataLoaderPandas
import pandas as pd

Ref = Type["RefData"]

def ref_values(path: str, Cp=1005, h=0.029846 / 2) -> Tuple[Dict]:
    hot = -1
    cold = 0

    ref = DataLoaderPandas(path, "statistiques").load_last()
    # h = ref.index.values[-1]/2
    Tw_hot = Twhot = ref["T"].iloc[hot]
    Tw_cold = Twcold = ref["T"].iloc[cold]
    Tw = {"hot": Tw_hot, "cold": Tw_cold}

    utau_temp = np.sqrt(
        ref["NU"] * np.abs(np.gradient(ref["U"], ref.index, edge_order=2))
    ).values
    retau_temp = (utau_temp * h / ref["NU"]).values

    rho_bulk = rhobulk = np.trapz(ref["URHO"], ref.index) / np.trapz(
        ref["U"], ref.index
    )
    mu_bulk = mubulk = np.trapz(ref["MU"], ref.index) / (2 * h)
    u_bulk = ubulk = np.trapz(ref["URHO"], ref.index) / np.trapz(ref["RHO"], ref.index)
    t_bulk = tbulk = np.trapz(ref["RHOUT_MOY"], ref.index) / np.trapz(
        ref["URHO"], ref.index
    )
    re_bulk = rebulk = ubulk * h * rhobulk / mubulk

    theta_hot_temp = (ref["T"] - Tw_hot) / (t_bulk - Tw_hot)
    theta_cold_temp = (ref["T"] - Tw_cold) / (t_bulk - Tw_cold)
    theta = {"hot": theta_hot_temp, "cold": theta_cold_temp}

    thetatau_temp = (
        ref["LAMBDADTDZ"] / (ref["RHO"] * Cp * utau_temp)
    ).values  # {"hot":, "cold":}

    utau = {"hot": utau_temp[-1], "cold": utau_temp[0]}
    retau = {"hot": retau_temp[-1], "cold": retau_temp[0]}
    thetatau = {"hot": thetatau_temp[-1], "cold": thetatau_temp[0]}

    nusselt_hot_temp = 2 * np.gradient(theta_hot_temp, ref.index, edge_order=2)
    nusselt_cold_temp = 2 * np.gradient(theta_cold_temp, ref.index, edge_order=2)

    nusselt = {"hot": nusselt_hot_temp[hot], "cold": nusselt_cold_temp[cold]}

    Cf_hot_temp = (
        2 / (rebulk * ubulk**2) * np.gradient(ref["U"], ref.index, edge_order=2)[hot]
    )
    Cf_cold_temp = (
        2 / (rebulk * ubulk**2) * np.gradient(ref["U"], ref.index, edge_order=2)[cold]
    )

    Cf = {"hot": Cf_hot_temp, "cold": Cf_cold_temp}

    y = ref.index.values
    middle = len(y) // 2

    msh = y = ref.index.values
    y_plus_hot = y[:middle] * utau["hot"] / ref["NU"].iloc[hot]
    y_plus_cold = (2 * h - y[middle:][::-1]) * utau["cold"] / ref["NU"].iloc[cold]

    y_plus = yplus = {"hot": y_plus_hot, "cold": y_plus_cold}

    return (
        h,
        Tw,
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
    )


class RefData(object):
    def __init__(self, path: str, Cp=1005) -> None:
        self.path = path
        self.Cp = self.cp = Cp

    def load(self, path=None, Cp=None):
        if not path:
            path = self.path
        if not Cp:
            Cp = self.Cp
        (
            self.h,
            self.Tw,
            self.utau,
            self.thetatau,
            self.retau,
            self.rho_bulk,
            self.u_bulk,
            self.t_bulk,
            self.re_bulk,
            self.theta,
            self.utau,
            self.retau,
            self.nusselt,
            self.Cf,
            self.y,
            self.yplus,
            self.df,
        ) = ref_values(path, self.Cp)
        self.ubulk = self.u_bulk
        self.rebulk = self.re_bulk

        self.wall = {"T": self.Tw}
        self.sheer = self.tau = {
            "utau": self.utau,
            "retau": self.retau,
            "Cf": self.Cf,
            "thetatau": self.thetatau,
        }
        self.msh = {"y": self.y, "h": self.h}
        self.T = self.thermal = {"Nu": self.nusselt, "nusselt": self.nusselt}
        self.bulk = {
            "rho_bulk": self.rho_bulk,
            "u_bulk": self.u_bulk,
            "t_bulk": self.t_bulk,
            "re_bulk": self.re_bulk,
        }
        self.middle = len(self.y) // 2
        self.Nusselt = self.Nu = self.nusselt
        self.ny = self.y.shape[0]


def osc_post_treat(
    df: pd.DataFrame, ref: Ref, *, ignore_idx: int = 0
) -> Tuple[Union[np.typing.ArrayLike, pd.DataFrame]]:
    theta_hot = [((d["T"] - ref.Tw["hot"]) / (ref.t_bulk - ref.Tw["hot"])) for d in df]

    theta_cold = [
        ((d["T"] - ref.Tw["cold"]) / (ref.t_bulk - ref.Tw["cold"])) for d in df
    ]
    theta = {"hot": theta_hot, "cold": theta_cold}

    Nu_hot = [
        2
        * t.groupby("time").apply(
            lambda x: np.gradient(x[-4:], ref.yplus[-4:], edge_order=2)[-1]
        )
        for t in theta_hot
    ]
    Nu_cold = [
        2
        * t.groupby("time").apply(
            lambda x: np.gradient(x[:4], ref.yplus[:4], edge_order=2)[0]
        )
        for t in theta_cold
    ]

    Nu_hot_norm = [nu / ref.nusselt["hot"] for nu in Nu_hot]
    Nu_cold_norm = [nu / ref.nusselt["cold"] for nu in Nu_cold]

    Nu = {"hot": Nu_hot, "cold": Nu_cold}
    Nu_norm = {"hot": Nu_hot_norm, "cold": Nu_cold_norm}

    times = [d.index.get_level_values(0).unique() for d in df]
    t_plus = tplus = [t * (ref.retau["hot"] * ref.utau["hot"]) / ref.h for t in times]
    # mintimes = np.array([t[0] for t in tplus])
    # mintime = np.min(mintimes)
    tplus = t_plus = [t - t[0] for t in tplus]

    Cf_act_hot = [
        2
        / (ref.ubulk * ref.ubulk * ref.rebulk)
        * d["U"]
        .groupby("time")
        .apply(lambda x: np.gradient(x[-4:], ref.yplus[-4:], edge_order=2)[-1])
        for d in df
    ]
    Cf_act_cold = [
        2
        / (ref.ubulk * ref.ubulk * ref.rebulk)
        * d["U"]
        .groupby("time")
        .apply(lambda x: np.gradient(x[:4], ref.yplus[:4], edge_order=2)[0])
        for d in df
    ]

    Cf_act_norm_hot = [cf / ref.Cf["hot"] for cf in Cf_act_hot]
    Cf_act_norm_cold = [cf / ref.Cf["cold"] for cf in Cf_act_cold]
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
) -> np.typing.ArrayLike:
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
