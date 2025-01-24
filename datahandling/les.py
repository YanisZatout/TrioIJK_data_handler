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
        Tw: Dict[str, float] = {"hot": 293, "cold": 586},
    ) -> None:
        assert Cp != 0, "Cp must be provided as a floating point value"
        assert h != 0, "h must be provided as a floating point value"
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
        "hot":   np.abs((df["T"] - ref.Tw["hot" ])/ref.thetatau["hot" ]).values[::-1][:ref.middle],
        "cold":  np.abs((df["T"] - ref.Tw["cold"])/ref.thetatau["cold"]).values      [:ref.middle]
    }
    out["U"] = {
        "hot":   (df["U"]/ref.utau["hot" ]).values[::-1][:ref.middle],
        "cold":  (df["U"]/ref.utau["cold"]).values      [:ref.middle]
    }
    out["V"] = {
        "hot":   (df["W"]/ref.utau["hot" ]).values[::-1][:ref.middle],
        "cold":  (df["W"]/ref.utau["cold"]).values      [:ref.middle]
    }
    
    out["Nu"] = {
        "hot":   2*np.gradient(out["T"]["hot" ], ref.yplus["hot" ], edge_order=2),
        "cold":  2*np.gradient(out["T"]["cold"], ref.yplus["cold"], edge_order=2)
    }
    
    out["Cf"] = {
        "hot":   np.abs(df["MU"] * np.gradient(df["U"], df["coordonnee_K"], edge_order=2)).values[::-1][:ref.middle],
        "cold":  np.abs(df["MU"] * np.gradient(df["U"], df["coordonnee_K"], edge_order=2)).values      [:ref.middle]
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
        "hot":   np.abs((df["T"] - ref.Tw["hot" ])/ref.thetatau["hot" ]).values[::-1][:ref.middle],
        "cold":  np.abs((df["T"] - ref.Tw["cold"])/ref.thetatau["cold"]).values      [:ref.middle]
    }
    out["U"] = {
        "hot":   (df["U"]/ref.utau["hot" ]).values[::-1][:ref.middle],
        "cold":  (df["U"]/ref.utau["cold"]).values      [:ref.middle]
    }
    out["V"] = {
        "hot":   (df["W"]/ref.utau["hot" ]).values[::-1][:ref.middle],
        "cold":  (df["W"]/ref.utau["cold"]).values      [:ref.middle]
    }
    
    out["Nu"] = {
        "hot":   2*np.gradient(out["T"]["hot" ], ref.yplus["hot" ], edge_order=2),
        "cold":  2*np.gradient(out["T"]["cold"], ref.yplus["cold"], edge_order=2)
    }
    
    out["Cf"] = {
        "hot":   np.abs(df["MU"] * np.gradient(df["U"], df["coordonnee_K"], edge_order=2)).values[::-1][:ref.middle],
        "cold":  np.abs(df["MU"] * np.gradient(df["U"], df["coordonnee_K"], edge_order=2)).values      [:ref.middle]
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
    
    out   = dict()
    out["urms"]  = df["UU"] - df["U"]**2 - 1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
    out["urms"] += - 2 * (df["NUTURB_XX_DUDX"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))
    out["urms"] += df["STRUCTURAL_UU"]/df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]
    
    
    out["vrms"]  = df["WW"] - df["W"]**2 - 1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
    out["vrms"] += + df["STRUCTURAL_WW"] /df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]
    out["vrms"] += - 2 * ( df["NUTURB_ZZ_DWDZ"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))
    
    
    out["wrms"]  = df["VV"] - df["V"]**2 -  1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
    out["wrms"] += + df["STRUCTURAL_VV"]/df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]
    out["wrms"] += - 2 *(df["NUTURB_YY_DVDY"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))

    
    out["u_theta"]  = df["UT"] - df["U"] * df["T"]
    out["u_theta"] +=  - 2 * df["KAPPATURB_X_DSCALARDX"] + df["STRUCTURAL_USCALAR"]/(df["RHO"]*Cp)
    
    out["v_theta"]  = df["WT"] - df["W"] * df["T"]
    out["v_theta"] +=  - 2 * df["KAPPATURB_Z_DSCALARDZ"] + df["STRUCTURAL_WSCALAR"]/(df["RHO"]*Cp)

    out["theta_rms"] = df["T2"] - df["T"]**2

    out2 = dict()
    for key in out.keys():
        out2[key] = {"hot": out[key].values[::-1][:ref.middle], "cold": out[key].values      [:ref.middle]}

    for side in ["hot", "cold"]:
        out2["urms"]     [side] /= (ref.utau[side] * ref.utau[side])
        out2["vrms"]     [side] /= (ref.utau[side] * ref.utau[side])
        out2["wrms"]     [side] /= (ref.utau[side] * ref.utau[side])
        out2["u_theta"]  [side] /= (ref.utau[side] * ref.thetatau[side])
        out2["v_theta"]  [side] /= (ref.utau[side] * ref.thetatau[side])
        out2["theta_rms"][side] /= (ref.thetatau[side] * ref.thetatau[side])

    return out2


def adim_closure_les(df, ref, mod, mesh, Cp):
    r"""
    """
    df = df[mod][mesh]
    ref = ref[mod][mesh]
    
    out   = dict()
    out["urms"] = - 2 * (df["NUTURB_XX_DUDX"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))
    out["urms"] += df["STRUCTURAL_UU"]/df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]
    
    
    out["vrms"] = + df["STRUCTURAL_WW"] /df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]
    out["vrms"] += - 2 * ( df["NUTURB_ZZ_DWDZ"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))
    
    
    out["wrms"] = + df["STRUCTURAL_VV"]/df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]
    out["wrms"] += - 2 *(df["NUTURB_YY_DVDY"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))

    
    out["u_theta"] =  - 2 * df["KAPPATURB_X_DSCALARDX"] + df["STRUCTURAL_USCALAR"]/(df["RHO"]*Cp)
    
    out["v_theta"] =  - 2 * df["KAPPATURB_Z_DSCALARDZ"] + df["STRUCTURAL_WSCALAR"]/(df["RHO"]*Cp)

    out["theta_rms"] = np.array([0])

    out2 = dict()
    for key in out.keys():
        out2[key] = {"hot": out[key].values[::-1][:ref.middle], "cold": out[key].values      [:ref.middle]}

    for side in ["hot", "cold"]:
        out2["urms"]     [side] /= (ref.utau[side] * ref.utau[side])
        out2["vrms"]     [side] /= (ref.utau[side] * ref.utau[side])
        out2["wrms"]     [side] /= (ref.utau[side] * ref.utau[side])
        out2["u_theta"]  [side] /= (ref.utau[side] * ref.thetatau[side])
        out2["v_theta"]  [side] /= (ref.utau[side] * ref.thetatau[side])
        out2["theta_rms"][side] /= (ref.thetatau[side] * ref.thetatau[side])

    return out2

def adim_rms_dns(ref):
    r"""
    """
    df = ref.df
    
    out = dict()
    out["urms"]  = df["UU"] - df["U"]**2 - 1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
    
    out["vrms"]  = df["WW"] - df["W"]**2 - 1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
    
    out["wrms"]  = df["VV"] - df["V"]**2 -  1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
    
    out["u_theta"]  = df["UT"] - df["U"] * df["T"]
    
    out["v_theta"]  = df["WT"] - df["W"] * df["T"]

    out["theta_rms"] = df["T2"] - df["T"]**2
    
    out2 = dict()
    for key in out.keys():
        out2[key] = {"hot": out[key].values[::-1][:ref.middle], "cold": out[key].values      [:ref.middle]}
    for side in ["hot", "cold"]:
        out2["urms"]     [side] /= (ref.utau[side] * ref.utau[side])
        out2["vrms"]     [side] /= (ref.utau[side] * ref.utau[side])
        out2["wrms"]     [side] /= (ref.utau[side] * ref.utau[side])
        out2["u_theta"]  [side] /= (ref.utau[side] * ref.thetatau[side])
        out2["v_theta"]  [side] /= (ref.utau[side] * ref.thetatau[side])
        out2["theta_rms"][side] /= (ref.thetatau[side] * ref.thetatau[side])

    return out2
