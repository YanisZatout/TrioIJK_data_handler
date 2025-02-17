from typing import Dict
import numpy as np
import pandas as pd
from .oscillation import ref_values


class LesData(object):
    def __init__(
        self,
        ref_df: pd.DataFrame,
        /,
        Cp: float = 0,
        h: float = 0,
        Tw: Dict[str, float] = {"hot": None, "cold": None},
    ) -> None:
        assert Cp != 0, "Cp must be provided as a floating point value"
        assert h != 0, "h must be provided as a floating point value"
        assert (
            Tw["hot"] is not None and Tw["cold"] is not None
        ), "You must provide boundary temperatures"
        self.ref_df = ref_df
        self.Cp = self.cp = Cp
        self.h = h
        self.Tw = self.tw = Tw

    def load(self, ref_df=None, Cp=None):
        if not Cp:
            Cp = self.Cp
        (
            _,
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
            self.mu_bulk,
            self.phi_tau,
        ) = ref_values(self.ref_df, self.Cp, h=self.h, Tw=self.Tw)
        self.ubulk = self.u_bulk
        self.rebulk = self.re_bulk

        self.wall = {"T": self.Tw}
        self.sheer = self.tau = {
            "utau": self.utau,
            "retau": self.retau,
            "Cf": self.Cf,
            "thetatau": self.thetatau,
            "phitau": self.phi_tau,
        }
        self.msh = {"y": self.y, "h": self.h}
        self.T = self.thermal = {"Nu": self.nusselt, "nusselt": self.nusselt}
        self.bulk = {
            "rho_bulk": self.rho_bulk,
            "u_bulk": self.u_bulk,
            "t_bulk": self.t_bulk,
            "re_bulk": self.re_bulk,
            "mu_bulk": self.mu_bulk,
        }
        self.middle = len(self.y) // 2
        self.Nusselt = self.Nu = self.nusselt
        self.ny = self.y.shape[0]

    def __repr__(self):
        return (
            f"bulk quantities {self.bulk}\n"
            f"sheer quantities {self.sheer}\n"
            f"wall quantities {self.wall}\n"
        )


def adim_y_face(y_face, ref):
    utau = ref.utau
    nu = ref.df["NU"]
    middle = ref.middle
    y = y_face
    h = ref.h
    hot = -1
    cold = 0
    y_plus_hot = (2 * h - y[middle:])[::-1] * utau["hot"] / nu.iloc[hot]
    y_plus_cold = y[:middle] * utau["cold"] / ref["NU"].iloc[cold]
    return {"hot": y_plus_hot, "cold": y_plus_cold}


def adim_mean_les(df, ref, mod, mesh):
    r"""
    Adimentionalize mean quantities for LES such that:
    \langle T \rangle^+  = \langle \frac{T - T_\omega}{T_\tau} \rangle
    \langle U \rangle^+  = \langle \frac{U}{u_\tau} \rangle
    \langle V \rangle^+  = \langle \frac{V}{u_\tau} \rangle
    \langle Nu \rangle^+ = \langle 2\frac{\partial \langle T \rangle ^+}{\partial y^+} \rangle
    \langle Cf \rangle^+ = \langle \tau_\omega \rangle

    This last one is subject to change
    """
    df = df[mod][mesh]
    ref = ref[mod][mesh]

    out = dict()
    out["T"] = {
        "hot": np.abs((df["T"] - ref.Tw["hot"]) / ref.thetatau["hot"]).values[::-1][
            : ref.middle
        ],
        "cold": np.abs((df["T"] - ref.Tw["cold"]) / ref.thetatau["cold"]).values[
            : ref.middle
        ],
    }
    out["U"] = {
        "hot": (df["U"] / ref.utau["hot"]).values[::-1][: ref.middle],
        "cold": (df["U"] / ref.utau["cold"]).values[: ref.middle],
    }
    out["V"] = {
        "hot": (df["W"] / ref.utau["hot"]).values[::-1][: ref.middle],
        "cold": (df["W"] / ref.utau["cold"]).values[: ref.middle],
    }

    out["Nu"] = {
        "hot": 2 * np.gradient(out["T"]["hot"], ref.yplus["hot"], edge_order=2),
        "cold": 2 * np.gradient(out["T"]["cold"], ref.yplus["cold"], edge_order=2),
    }

    out["Cf"] = {
        "hot": np.abs(
            df["MU"] * np.gradient(df["U"], df["coordonnee_K"], edge_order=2)
        ).values[::-1][: ref.middle],
        "cold": np.abs(
            df["MU"] * np.gradient(df["U"], df["coordonnee_K"], edge_order=2)
        ).values[: ref.middle],
    }

    return out


def adim_mean_dns(ref):
    r"""
    Adimentionalize mean quantities for DNS such that:
    \langle T \rangle^+  = \langle \frac{T - T_\omega}{T_\tau} \rangle
    \langle U \rangle^+  = \langle \frac{U}{u_\tau} \rangle
    \langle V \rangle^+  = \langle \frac{V}{u_\tau} \rangle
    \langle Nu \rangle^+ = \langle 2\frac{\partial \langle T \rangle ^+}{\partial y^+} \rangle
    \langle Cf \rangle^+ = \langle \tau_\omega \rangle

    This last one is subject to change
    """
    df = ref.df

    out = dict()
    out["T"] = {
        "hot": np.abs((df["T"] - ref.Tw["hot"]) / ref.thetatau["hot"]).values[::-1][
            : ref.middle
        ],
        "cold": np.abs((df["T"] - ref.Tw["cold"]) / ref.thetatau["cold"]).values[
            : ref.middle
        ],
    }
    out["U"] = {
        "hot": (df["U"] / ref.utau["hot"]).values[::-1][: ref.middle],
        "cold": (df["U"] / ref.utau["cold"]).values[: ref.middle],
    }
    out["V"] = {
        "hot": (df["W"] / ref.utau["hot"]).values[::-1][: ref.middle],
        "cold": (df["W"] / ref.utau["cold"]).values[: ref.middle],
    }

    out["Nu"] = {
        "hot": 2 * np.gradient(out["T"]["hot"], ref.yplus["hot"], edge_order=2),
        "cold": 2 * np.gradient(out["T"]["cold"], ref.yplus["cold"], edge_order=2),
    }

    out["Cf"] = {
        "hot": np.abs(
            df["MU"] * np.gradient(df["U"], df["coordonnee_K"], edge_order=2)
        ).values[::-1][: ref.middle],
        "cold": np.abs(
            df["MU"] * np.gradient(df["U"], df["coordonnee_K"], edge_order=2)
        ).values[: ref.middle],
    }

    return out


def adim_rms_les(df, ref, mod, mesh, Cp):
    r"""
    Adimentionalize mean quantities for LES such that:
    \langle U'^2 \rangle^+  = \frac{\langle U^2 \rangle - \langle U \rangle^2}{u_\tau^2}
    \langle V'^2 \rangle^+  = \frac{\langle V^2 \rangle - \langle V \rangle^2}{u_\tau^2}
    \langle W'^2 \rangle^+  = \frac{\langle V^2 \rangle - \langle V \rangle^2}{u_\tau^2}
    \langle U'T' \rangle^+  = \langle 2\frac{\partial \langle T \rangle ^+}{\partial y^+} \rangle
    \langle V'T' \rangle^+  = \langle \tau_\omega \rangle
    \langle T'T' \rangle^+  = \langle \tau_\omega \rangle

    This last one is subject to change
    """
    df = df[mod][mesh]
    ref = ref[mod][mesh]

    out = dict()
    out["urms"] = (
        df["UU"]
        - df["U"] ** 2
        - 1
        / 3
        * (df["UU"] - df["U"] ** 2 + df["VV"] - df["V"] ** 2 + df["WW"] - df["W"] ** 2)
    )
    out["urms"] += -2 * (
        df["NUTURB_XX_DUDX"]
        - 1 / 3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])
    )
    out["urms"] += (
        df["STRUCTURAL_UU"] / df["RHO"]
        - 1
        / 3
        * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])
        / df["RHO"]
    )

    out["vrms"] = (
        df["WW"]
        - df["W"] ** 2
        - 1
        / 3
        * (df["UU"] - df["U"] ** 2 + df["VV"] - df["V"] ** 2 + df["WW"] - df["W"] ** 2)
    )
    out["vrms"] += (
        +df["STRUCTURAL_WW"] / df["RHO"]
        - 1
        / 3
        * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])
        / df["RHO"]
    )
    out["vrms"] += -2 * (
        df["NUTURB_ZZ_DWDZ"]
        - 1 / 3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])
    )

    out["wrms"] = (
        df["VV"]
        - df["V"] ** 2
        - 1
        / 3
        * (df["UU"] - df["U"] ** 2 + df["VV"] - df["V"] ** 2 + df["WW"] - df["W"] ** 2)
    )
    out["wrms"] += (
        +df["STRUCTURAL_VV"] / df["RHO"]
        - 1
        / 3
        * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])
        / df["RHO"]
    )
    out["wrms"] += -2 * (
        df["NUTURB_YY_DVDY"]
        - 1 / 3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])
    )

    out["u_theta"] = df["UT"] - df["U"] * df["T"]
    out["u_theta"] += -2 * df["KAPPATURB_X_DSCALARDX"] + df["STRUCTURAL_USCALAR"] / (
        df["RHO"] * Cp
    )

    out["v_theta"] = df["WT"] - df["W"] * df["T"]
    out["v_theta"] += -2 * df["KAPPATURB_Z_DSCALARDZ"] + df["STRUCTURAL_WSCALAR"] / (
        df["RHO"] * Cp
    )

    out["theta_rms"] = df["T2"] - df["T"] ** 2

    out2 = dict()
    for key in out.keys():
        out2[key] = {
            "hot": out[key].values[::-1][: ref.middle],
            "cold": out[key].values[: ref.middle],
        }

    for side in ["hot", "cold"]:
        out2["urms"][side] /= ref.utau[side] * ref.utau[side]
        out2["vrms"][side] /= ref.utau[side] * ref.utau[side]
        out2["wrms"][side] /= ref.utau[side] * ref.utau[side]
        out2["u_theta"][side] /= ref.utau[side] * ref.thetatau[side]
        out2["v_theta"][side] /= ref.utau[side] * ref.thetatau[side]
        out2["theta_rms"][side] /= ref.thetatau[side] * ref.thetatau[side]

    return out2


def adim_closure_les(df, ref, mod, mesh, Cp):
    r""" """
    df = df[mod][mesh]
    ref = ref[mod][mesh]

    out = dict()
    out["urms_func"] = -2 * (
        df["NUTURB_XX_DUDX"]
        - 1 / 3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])
    )
    out["urms_struct"] = (
        df["STRUCTURAL_UU"] / df["RHO"]
        - 1
        / 3
        * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])
        / df["RHO"]
    )

    out["vrms_func"] = (
        +df["STRUCTURAL_WW"] / df["RHO"]
        - 1
        / 3
        * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])
        / df["RHO"]
    )
    out["vrms_struct"] = -2 * (
        df["NUTURB_ZZ_DWDZ"]
        - 1 / 3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])
    )

    out["wrms_func"] = (
        +df["STRUCTURAL_VV"] / df["RHO"]
        - 1
        / 3
        * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])
        / df["RHO"]
    )
    out["wrms_struct"] = -2 * (
        df["NUTURB_YY_DVDY"]
        - 1 / 3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])
    )

    out["u_theta_func"] = -2 * df["KAPPATURB_X_DSCALARDX"]
    out["u_theta_struct"] = +df["STRUCTURAL_USCALAR"] / (df["RHO"] * Cp)

    out["v_theta_func"] = -2 * df["KAPPATURB_Z_DSCALARDZ"]
    out["v_theta_struct"] = +df["STRUCTURAL_WSCALAR"] / (df["RHO"] * Cp)

    out["theta_rms_func"] = pd.DataFrame(np.array([0.0]))
    out["theta_rms_struct"] = pd.DataFrame(np.array([0.0]))

    out2 = dict()
    for key in out.keys():
        out2[key] = {
            "hot": out[key].values[::-1][: ref.middle],
            "cold": out[key].values[: ref.middle],
        }

    for side in ["hot", "cold"]:
        out2["urms_func"][side] /= ref.utau[side] * ref.utau[side]
        out2["vrms_func"][side] /= ref.utau[side] * ref.utau[side]
        out2["wrms_func"][side] /= ref.utau[side] * ref.utau[side]
        out2["u_theta_func"][side] /= ref.utau[side] * ref.thetatau[side]
        out2["v_theta_func"][side] /= ref.utau[side] * ref.thetatau[side]
        out2["theta_rms_func"][side] /= ref.thetatau[side] * ref.thetatau[side]
        out2["theta_rms_func"][side] = out2["theta_rms_func"][side][0]

        out2["urms_struct"][side] /= ref.utau[side] * ref.utau[side]
        out2["vrms_struct"][side] /= ref.utau[side] * ref.utau[side]
        out2["wrms_struct"][side] /= ref.utau[side] * ref.utau[side]
        out2["u_theta_struct"][side] /= ref.utau[side] * ref.thetatau[side]
        out2["v_theta_struct"][side] /= ref.utau[side] * ref.thetatau[side]
        out2["theta_rms_struct"][side] /= ref.thetatau[side] * ref.thetatau[side]
        out2["theta_rms_struct"][side] = out2["theta_rms_struct"][side][0]

    return out2


def adim_rms_dns(ref):
    r""" """
    df = ref.df

    out = dict()
    out["urms"] = (
        df["UU"]
        - df["U"] ** 2
        - 1
        / 3
        * (df["UU"] - df["U"] ** 2 + df["VV"] - df["V"] ** 2 + df["WW"] - df["W"] ** 2)
    )

    out["vrms"] = (
        df["WW"]
        - df["W"] ** 2
        - 1
        / 3
        * (df["UU"] - df["U"] ** 2 + df["VV"] - df["V"] ** 2 + df["WW"] - df["W"] ** 2)
    )

    out["wrms"] = (
        df["VV"]
        - df["V"] ** 2
        - 1
        / 3
        * (df["UU"] - df["U"] ** 2 + df["VV"] - df["V"] ** 2 + df["WW"] - df["W"] ** 2)
    )

    out["u_theta"] = df["UT"] - df["U"] * df["T"]

    out["v_theta"] = df["WT"] - df["W"] * df["T"]

    out["theta_rms"] = df["T2"] - df["T"] ** 2

    out2 = dict()
    for key in out.keys():
        out2[key] = {
            "hot": out[key].values[::-1][: ref.middle],
            "cold": out[key].values[: ref.middle],
        }
    for side in ["hot", "cold"]:
        out2["urms"][side] /= ref.utau[side] * ref.utau[side]
        out2["vrms"][side] /= ref.utau[side] * ref.utau[side]
        out2["wrms"][side] /= ref.utau[side] * ref.utau[side]
        out2["u_theta"][side] /= ref.utau[side] * ref.thetatau[side]
        out2["v_theta"][side] /= ref.utau[side] * ref.thetatau[side]
        out2["theta_rms"][side] /= ref.thetatau[side] * ref.thetatau[side]

    return out2


def pad(x, size):
    sizex, sizey, sizez = size
    x = np.pad(x, ((sizex // 2, sizex // 2), (0, 0), (0, 0)), "wrap")
    x = np.pad(
        x,
        (
            (0, 0),
            (
                sizey // 2,
                sizey // 2,
            ),
            (0, 0),
        ),
        "wrap",
    )
    x = np.pad(x, ((0, 0), (0, 0), (sizez // 2, sizez // 2)), "edge")
    return x


def weighted_convolution(x, coord_face, size):
    """
    Weighted convolution for filtering DNS time steps
    """
    import scipy.signal as si

    sx, sy, sz = size
    out = np.zeros_like(x)

    padded = pad(x, size)
    filterx = si.get_window("boxcar", sx).reshape(-1, 1, 1)
    filterx /= filterx.sum()
    filtery = si.get_window("boxcar", sy).reshape(1, -1, 1)
    filtery /= filtery.sum()

    padded = si.convolve(padded, filterx, "valid")
    padded = si.convolve(padded, filtery, "valid")
    cell_size = np.diff(coord_face[-1])
    cell_size = np.pad(cell_size, (sz // 2, sz // 2), "edge")
    for k in range(x.shape[-1]):
        kernel = cell_size[k : k + sz] / cell_size[k : k + sz].sum()
        out[..., k : k + 1] = si.convolve(
            padded[..., k : k + sz], kernel[None, None], "valid"
        )
    return out


def adim_y_face(y_face, ref):
    utau = ref.utau
    nu = ref.df["NU"]
    y = y_face
    h = ref.h
    hot = -1
    cold = 0
    
    y_plus_hot = (2 * h - y)[::-1] * utau["hot"] / nu.iloc[hot]
    y_plus_cold = y * utau["cold"] / nu.iloc[cold]
    
    return {"hot": y_plus_hot, "cold": y_plus_cold}


def compute_eps_quantity_side(
    quantity, les, dns, ref_les, ref_dns, model, mesh, Cp, side
):
    from scipy.interpolate import CubicSpline
    try:
        les = adim_rms_les(les, ref_les, model, mesh, Cp)[quantity][side]
    except:
        les = adim_mean_les(les, ref_les, model, mesh)[quantity][side]
    dns = dns[quantity][side]
    ref_les = ref_les[model][mesh]
    
    yplus_dns = ref_dns.yplus[side]
    yplus_les = ref_les.yplus[side]
    
    spline = CubicSpline(yplus_dns, dns)
    outspline = spline(yplus_les)
    y_face = ref_les.y
    yplus_les_face = adim_y_face(y_face, ref_les)[side]
    logy_les = np.log(yplus_les_face[1:]/yplus_les_face[:-1])[:ref_les.middle]
    
    diff = (les - outspline)
    out = logy_les * np.abs(diff * les)
    
    denom = logy_les * outspline**2
    
    out = out/denom
    return out.sum()


def compute_eps(quantity, les, dns, ref_les, ref_dns, model, mesh, Cp):
    return compute_eps_quantity_side(
            quantity,
            les,
            dns,
            ref_les,
            ref_dns,
            model,
            mesh,
            Cp,
            "hot"
        ) + compute_eps_quantity_side(
                quantity,
                les,
                dns,
                ref_les,
                ref_dns,
                model,
                mesh,
                Cp,
                "cold"
        )

