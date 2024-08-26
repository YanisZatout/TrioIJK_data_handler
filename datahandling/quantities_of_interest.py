import numpy as np
from typing import Dict, List, Tuple, Union
import pandas as pd

from datahandling.oscillation import RefData


def adim_qqty(
    df_les: list[pd.DataFrame],
    df_dns: pd.DataFrame,
    Cp=1155,
    *,
    delta: float = 0.0029846,
) -> Union[tuple[dict, dict], None]:
    assert df_dns is not None, "df_dns was found to be a None, careful !"
    assert df_les is not None, "df_les was found to be a None, careful !"
    if df_dns is not None and df_les != [] and len(df_les) >= 1:
        half_les = [len(df) // 2 for df in df_les]
        utau_cold_les = [
            np.sqrt(df["MU"] / df["RHO"] * df["U"] / df.index[0]).iloc[0]
            for df in df_les
        ]
        re_cold_les = [
            (df["RHO"] * delta * utau / df["MU"]).iloc[0]
            for df, utau in zip(df_les, utau_cold_les)
        ]
        ttau_cold_les = [
            (df["LAMBDADTDZ"] / (df["RHO"] * Cp * utau)).iloc[0]
            for df, utau in zip(df_les, utau_cold_les)
        ]
        y_plus_cold_les = [
            ((df.index.to_numpy() / delta) * re)[:half]
            for df, re, half in zip(df_les, re_cold_les, half_les)
        ]

        utau_hot_les = [
            np.sqrt(df["MU"] / df["RHO"] * df["U"] / (2 * delta - df.index[-1])).iloc[
                -1
            ]
            for df in df_les
        ]
        re_hot_les = [
            (delta * utau / df["NU"]).iloc[-1] for df, utau in zip(df_les, utau_hot_les)
        ]
        ttau_hot_les = [
            (df["LAMBDADTDZ"] / (df["RHO"] * Cp * utau)).iloc[-1]
            for df, utau in zip(df_les, utau_hot_les)
        ]
        y_plus_hot_les = [
            ((2 * delta - df.index.to_numpy()) / delta * re)[-half:]
            for df, re, half in zip(df_les, re_hot_les, half_les)
        ]

        les = {
            "utau_cold": utau_cold_les,
            "re_cold": re_cold_les,
            "ttau_cold": ttau_cold_les,
            "y_plus_cold": y_plus_cold_les,
            "utau_hot": utau_hot_les,
            "re_hot": re_hot_les,
            "ttau_hot": ttau_hot_les,
            "y_plus_hot": y_plus_hot_les,
        }

        half = len(df_dns.index) // 2

        utau_cold_dns = np.sqrt(
            df_dns["MU"] / df_dns["RHO"] * df_dns["U"] / df_dns.index[0]
        ).iloc[0]
        re_cold_dns = (df_dns["RHO"] * delta * utau_cold_dns / df_dns["MU"]).iloc[0]
        ttau_cold_dns = (
            df_dns["LAMBDADTDZ"] / (df_dns["RHO"] * Cp * utau_cold_dns)
        ).iloc[0]
        y_plus_cold_dns = ((df_dns.index.to_numpy() / delta) * re_cold_dns)[:half]

        utau_hot_dns = np.sqrt(
            df_dns["MU"] / df_dns["RHO"] * df_dns["U"] / (2 * delta - df_dns.index[-1])
        ).iloc[-1]
        re_hot_dns = (delta * utau_hot_dns / df_dns["NU"]).iloc[-1]
        ttau_hot_dns = (
            df_dns["LAMBDADTDZ"] / (df_dns["RHO"] * Cp * utau_hot_dns)
        ).iloc[-1]
        y_plus_hot_dns = ((2 * delta - df_dns.index.to_numpy()) / delta * re_hot_dns)[
            -half:
        ]
        dns = {
            "utau_cold": utau_cold_dns,
            "re_cold": re_cold_dns,
            "ttau_cold": ttau_cold_dns,
            "y_plus_cold": y_plus_cold_dns,
            "utau_hot": utau_hot_dns,
            "re_hot": re_hot_dns,
            "ttau_hot": ttau_hot_dns,
            "y_plus_hot": y_plus_hot_dns,
        }
        return les, dns

def compute_rms_quantities(df_les: List[pd.DataFrame], df_dns: pd.DataFrame, Cp=1155) -> Tuple[Dict]:
    """
    Computes RMS quantities of interest:
    \langle u^{'2} \rangle ^{dev}
    \langle v^{'2} \rangle ^{dev}
    \langle w^{'2} \rangle ^{dev}

    \langle u' v^{'} \rangle

    \langle u' T^{'} \rangle
    \langle v' T^{'} \rangle
    \langle T' T^{'} \rangle
    """
    assert df_dns is not None, "df_dns was found to be a None, careful !"
    assert df_les is not None, "df_les was found to be a None, careful !"
    if df_dns is not None and df_les != [] and len(df_les) >= 1:
        # Urms
        urms_les = [
            df["UU"] - df["U"]**2 - 1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
            + df["STRUCTURAL_UU"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])
            - 2 * df["NUTURB_XX_DUDX"] + 2/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])
            for df in df_les
        ]
        urms_dns = df_dns["UU"] - df_dns["U"]**2 - 1/3 * (df_dns["UU"] - df_dns["U"]**2 + df_dns["VV"] - df_dns["V"]**2 + df_dns["WW"] - df_dns["W"]**2)

        # Tau_{x, x}
        urms_struct = [df["STRUCTURAL_UU"] for df in df_les]
        urms_struct_dev = [df["STRUCTURAL_UU"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"]) for df in df_les]
        urms_fonc = [- 2 * df["NUTURB_XX_DUDX"]  for df in df_les]
        urms_fonc_dev  = [- 2 * df["NUTURB_XX_DUDX"] + 2/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]) for df in df_les]

        # Vrms
        vrms_les = [
            df["WW"] - df["W"]**2 - 1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
            + df["STRUCTURAL_WW"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])
            - 2 * df["NUTURB_ZZ_DWDZ"] + 2/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])
            for df in df_les
        ]
        vrms_dns = ( df_dns["WW"] - df_dns["W"]**2 - 1/3 * (df_dns["UU"] - df_dns["U"]**2 + df_dns["VV"] - df_dns["V"]**2 + df_dns["WW"] - df_dns["W"]**2))

        # Tau_{y, y}
        vrms_struct = [df["STRUCTURAL_WW"] for df in df_les]
        vrms_struct_dev = [df["STRUCTURAL_WW"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"]) for df in df_les]
        vrms_fonc = [- 2 * df["NUTURB_ZZ_DWDZ"] for df in df_les]
        vrms_fonc_dev  = [- 2 * df["NUTURB_ZZ_DWDZ"] + 2/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]) for df in df_les]

        # Wrms
        wrms_les = [
            df["VV"] - df["V"]**2 -  1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
            + df["STRUCTURAL_VV"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])
            - 2 * df["NUTURB_YY_DVDY"] + 2/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])
            for df in df_les
        ]
        wrms_dns = (df_dns["VV"] - df_dns["V"]**2 - 1/3 * (df_dns["UU"] - df_dns["U"]**2 + df_dns["VV"] - df_dns["V"]**2 + df_dns["WW"] - df_dns["W"]**2))

        # Tau_{z, z}
        wrms_struct = [df["STRUCTURAL_VV"]  for df in df_les]
        wrms_struct_dev = [df["STRUCTURAL_VV"] - 2/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"]) for df in df_les]
        wrms_fonc = [- 2 * df["NUTURB_YY_DVDY"]  for df in df_les]
        wrms_fonc_dev  = [- 2 * df["NUTURB_YY_DVDY"] + 2/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]) for df in df_les]

        # UV
        uv_les = [(df["UW"] - df["U"] * df["W"] + df["STRUCTURAL_UW"] - (df["NUTURB_XZ_DUDZ"] + df["NUTURB_XZ_DWDX"])) for df in df_les]
        uv_dns = df_dns["UW"] - df_dns["U"]*df_dns["W"]
        uv_struct =[ df["STRUCTURAL_UW"] for df in df_les]
        uv_fonc   = [- (df["NUTURB_XZ_DUDZ"] + df["NUTURB_XZ_DWDX"]) for df in df_les]

        # UT
        u_theta_les = [(df["UT"] - df["U"] * df["T"] - 2* df["KAPPATURB_X_DSCALARDX"]  + df["STRUCTURAL_USCALAR"]) for df in df_les]
        u_theta_dns = (df_dns["UT"] - df_dns["U"] * df_dns["T"])
        u_theta_struct = [df["STRUCTURAL_USCALAR"] for df in df_les]
        u_theta_fonc   =[ - 2* df["KAPPATURB_X_DSCALARDX"]  for df in df_les]

        # VT
        v_theta_les = [(df["WT"] - df["W"] * df["T"] - 2 * df["KAPPATURB_Z_DSCALARDZ"] + df["STRUCTURAL_WSCALAR"]) for df in df_les]
        v_theta_dns = df_dns["WT"] - df_dns["W"] * df_dns["T"]
        v_theta_struct = [df["STRUCTURAL_WSCALAR"] for df in df_les]
        v_theta_fonc   = [- 2 * df["KAPPATURB_Z_DSCALARDZ"] for df in df_les]

        # TT
        theta_rms_les = [((df["T2"] - df["T"] * df["T"])) for df in df_les]
        theta_rms_dns = ((df_dns["T2"] - df_dns["T"]*df_dns["T"]))

        # put everything in a tidy container
        les = {"urms": urms_les, "vrms": vrms_les, "wrms": wrms_les, "uv": uv_les, "u_theta": u_theta_les, "v_theta": v_theta_les, "theta_rms": theta_rms_les}
        dns = {"urms": urms_dns, "vrms": vrms_dns, "wrms": wrms_dns, "uv": uv_dns, "u_theta": u_theta_dns, "v_theta": v_theta_dns, "theta_rms": theta_rms_dns}
        struct = {"urms_dev": urms_struct_dev, "urms":urms_struct, "vrms": vrms_struct,"vrms_dev": vrms_struct_dev, "wrms":wrms_struct,
                  "wrms_dev":wrms_struct_dev , "uv": uv_struct, "u_theta": u_theta_struct, "v_theta": v_theta_struct}
        fonc = {"urms_dev": urms_fonc_dev, "urms":urms_fonc, "vrms": vrms_fonc,"vrms_dev": vrms_fonc_dev, "wrms":wrms_fonc,
                  "wrms_dev":wrms_fonc_dev , "uv": uv_fonc, "u_theta": u_theta_fonc, "v_theta": v_theta_fonc}

        return les, dns, struct, fonc


def compute_rms_quantities_compressible(df_les: List[pd.DataFrame], df_dns: pd.DataFrame, Cp=1155) -> Tuple[Dict]:
    """
    Computes RMS quantities of interest for compressible structural models:
    \langle u^{'2} \rangle ^{dev}
    \langle v^{'2} \rangle ^{dev}
    \langle w^{'2} \rangle ^{dev}

    \langle u' v^{'} \rangle

    \langle u' T^{'} \rangle
    \langle v' T^{'} \rangle
    \langle T' T^{'} \rangle
    """
    urms_les = [
        df["UU"] - df["U"]**2 - 1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
        + df["STRUCTURAL_UU"]/df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]
        - 2 * (df["NUTURB_XX_DUDX"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))
        for df in df_les
    ]
    urms_dns = df_dns["UU"] - df_dns["U"]**2 - 1/3 * (df_dns["UU"] - df_dns["U"]**2 + df_dns["VV"] - df_dns["V"]**2 + df_dns["WW"] - df_dns["W"]**2)

    # Tau_{x, x}
    urms_struct = [df["STRUCTURAL_UU"]/df["RHO"] for df in df_les]
    urms_struct_dev = [df["STRUCTURAL_UU"]/df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"] for df in df_les]
    urms_fonc = [- 2 * df["NUTURB_XX_DUDX"] for df in df_les]
    urms_fonc_dev  = [- 2 * (df["NUTURB_XX_DUDX"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])) for df in df_les]

    vrms_les = [
        df["WW"] - df["W"]**2 - 1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
        + df["STRUCTURAL_WW"] /df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]
        - 2 * ( df["NUTURB_ZZ_DWDZ"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))
        for df in df_les
    ]
    vrms_dns = ( df_dns["WW"] - df_dns["W"]**2 - 1/3 * (df_dns["UU"] - df_dns["U"]**2 + df_dns["VV"] - df_dns["V"]**2 + df_dns["WW"] - df_dns["W"]**2))

    # Tau_{y, y}
    vrms_struct = [df["STRUCTURAL_WW"] /df["RHO"] for df in df_les]
    vrms_struct_dev = [df["STRUCTURAL_WW"] /df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"] for df in df_les]
    vrms_fonc = [- 2 * ( df["NUTURB_ZZ_DWDZ"]) for df in df_les]
    vrms_fonc_dev  = [- 2 * ( df["NUTURB_ZZ_DWDZ"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])) for df in df_les]

    wrms_les = [
        df["VV"] - df["V"]**2 -  1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
        + df["STRUCTURAL_VV"]/df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]
        - 2 *(df["NUTURB_YY_DVDY"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))
        for df in df_les
    ]
    wrms_dns = (df_dns["VV"] - df_dns["V"]**2  - 1/3 * (df_dns["UU"] - df_dns["U"]**2 + df_dns["VV"] - df_dns["V"]**2 + df_dns["WW"] - df_dns["W"]**2))

    # Tau_{z, z}
    wrms_struct = [df["STRUCTURAL_VV"]/df["RHO"] for df in df_les]
    wrms_struct_dev = [df["STRUCTURAL_VV"]/df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]  for df in df_les]
    wrms_fonc = [- 2 *df["NUTURB_YY_DVDY"] for df in df_les]
    wrms_fonc_dev  = [ - 2 *(df["NUTURB_YY_DVDY"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))  for df in df_les]

    uv_les = [(df["UW"]-df["U"]*df["W"] + df["STRUCTURAL_UW"]/df["RHO"] - (df["NUTURB_XZ_DUDZ"]+df["NUTURB_XZ_DWDX"])) for df in df_les]
    uv_dns = df_dns["UW"] - df_dns["U"] * df_dns["W"]
    uv_struct = [df["STRUCTURAL_UW"] / df["RHO"] for df in df_les]
    uv_fonc   = [- (df["NUTURB_XZ_DUDZ"] + df["NUTURB_XZ_DWDX"]) for df in df_les]

    u_theta_les = [(df["UT"] - df["U"] * df["T"] - 2* df["KAPPATURB_X_DSCALARDX"] + df["STRUCTURAL_USCALAR"]/df["RHO"]) for df in df_les]
    u_theta_dns = (df_dns["UT"]-df_dns["U"]*df_dns["T"])

    u_theta_struct = [df["STRUCTURAL_USCALAR"]/df["RHO"] for df in df_les]
    u_theta_fonc   =[- 2* df["KAPPATURB_X_DSCALARDX"]   for df in df_les]

    v_theta_les = [(df["WT"] - df["W"] * df["T"] - 2*df["KAPPATURB_Z_DSCALARDZ"] + df["STRUCTURAL_WSCALAR"]/df["RHO"]) for df in df_les]
    v_theta_dns = df_dns["WT"]-df_dns["W"]*df_dns["T"]
    v_theta_struct = [df["STRUCTURAL_WSCALAR"]/df["RHO"]  for df in df_les]
    v_theta_fonc   =[- 2 * df["KAPPATURB_Z_DSCALARDZ"]  for df in df_les]

    theta_rms_les = [((df["T2"]-df["T"]*df["T"])) for df in df_les]
    theta_rms_dns = ((df_dns["T2"]-df_dns["T"]*df_dns["T"]))

    les = {"urms": urms_les, "vrms": vrms_les, "wrms": wrms_les, "uv": uv_les, "u_theta": u_theta_les, "v_theta": v_theta_les, "theta_rms": theta_rms_les}
    dns = {"urms": urms_dns, "vrms": vrms_dns, "wrms": wrms_dns, "uv": uv_dns, "u_theta": u_theta_dns, "v_theta": v_theta_dns, "theta_rms": theta_rms_dns}
    struct = {"urms_dev": urms_struct_dev, "urms":urms_struct, "vrms": vrms_struct,"vrms_dev": vrms_struct_dev, "wrms":wrms_struct,
              "wrms_dev":wrms_struct_dev , "uv": uv_struct, "u_theta": u_theta_struct, "v_theta": v_theta_struct}
    fonc = {"urms_dev": urms_fonc_dev, "urms":urms_fonc, "vrms": vrms_fonc,"vrms_dev": vrms_fonc_dev, "wrms":wrms_fonc,
              "wrms_dev":wrms_fonc_dev , "uv": uv_fonc, "u_theta": u_theta_fonc, "v_theta": v_theta_fonc}

    return les, dns, struct, fonc


def rms(df_les: List[pd.DataFrame], df_dns: pd.DataFrame, Cp=1155, tau_compressible=False, pi_compressible=False) -> Tuple[Dict]:
    """
    Computes RMS quantities of interest for compressible structural models:
    \langle u^{'2} \rangle ^{dev}
    \langle v^{'2} \rangle ^{dev}
    \langle w^{'2} \rangle ^{dev}

    \langle u' v^{'} \rangle

    \langle u' T^{'} \rangle
    \langle v' T^{'} \rangle
    \langle T' T^{'} \rangle
    """
    if tau_compressible:
        urms_les = [
            df["UU"] - df["U"]**2 - 1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
            + df["STRUCTURAL_UU"]/df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]
            - 2 * (df["NUTURB_XX_DUDX"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))
            for df in df_les
        ]

        vrms_les = [
            df["WW"] - df["W"]**2 - 1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
            + df["STRUCTURAL_WW"] /df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]
            - 2 * ( df["NUTURB_ZZ_DWDZ"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))
            for df in df_les
        ]

        wrms_les = [
            df["VV"] - df["V"]**2 -  1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
            + df["STRUCTURAL_VV"]/df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]
            - 2 *(df["NUTURB_YY_DVDY"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))
            for df in df_les
        ]
        uv_les = [(df["UW"]-df["U"]*df["W"] + df["STRUCTURAL_UW"]/df["RHO"] - (df["NUTURB_XZ_DUDZ"]+df["NUTURB_XZ_DWDX"])) for df in df_les]
    else:
        urms_les = [
            df["UU"] - df["U"]**2 - 1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
            + df["STRUCTURAL_UU"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])
            - 2 * (df["NUTURB_XX_DUDX"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))
            for df in df_les
        ]

        vrms_les = [
            df["WW"] - df["W"]**2 - 1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
            + df["STRUCTURAL_WW"]  - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])
            - 2 * ( df["NUTURB_ZZ_DWDZ"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))
            for df in df_les
        ]

        wrms_les = [
            df["VV"] - df["V"]**2 -  1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
            + df["STRUCTURAL_VV"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])
            - 2 *(df["NUTURB_YY_DVDY"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))
            for df in df_les
        ]
        uv_les = [(df["UW"]-df["U"]*df["W"] + df["STRUCTURAL_UW"] - (df["NUTURB_XZ_DUDZ"]+df["NUTURB_XZ_DWDX"])) for df in df_les]
    if pi_compressible:
        u_theta_les = [(df["UT"] - df["U"] * df["T"] - 2* df["KAPPATURB_X_DSCALARDX"] + df["STRUCTURAL_USCALAR"]/df["RHO"]) for df in df_les]
        v_theta_les = [(df["WT"] - df["W"] * df["T"] - 2*df["KAPPATURB_Z_DSCALARDZ"] + df["STRUCTURAL_WSCALAR"]/df["RHO"]) for df in df_les]
    else:
        u_theta_les = [(df["UT"] - df["U"] * df["T"] - 2* df["KAPPATURB_X_DSCALARDX"] + df["STRUCTURAL_USCALAR"]) for df in df_les]
        v_theta_les = [(df["WT"] - df["W"] * df["T"] - 2*df["KAPPATURB_Z_DSCALARDZ"] + df["STRUCTURAL_WSCALAR"]) for df in df_les]

    urms_dns = df_dns["UU"] - df_dns["U"]**2 - 1/3 * (df_dns["UU"] - df_dns["U"]**2 + df_dns["VV"] - df_dns["V"]**2 + df_dns["WW"] - df_dns["W"]**2)
    vrms_dns = ( df_dns["WW"] - df_dns["W"]**2 - 1/3 * (df_dns["UU"] - df_dns["U"]**2 + df_dns["VV"] - df_dns["V"]**2 + df_dns["WW"] - df_dns["W"]**2))
    wrms_dns = (df_dns["VV"] - df_dns["V"]**2  - 1/3 * (df_dns["UU"] - df_dns["U"]**2 + df_dns["VV"] - df_dns["V"]**2 + df_dns["WW"] - df_dns["W"]**2))
    uv_dns = df_dns["UW"] - df_dns["U"] * df_dns["W"]
    u_theta_dns = (df_dns["UT"]-df_dns["U"]*df_dns["T"])
    v_theta_dns = df_dns["WT"]-df_dns["W"]*df_dns["T"]

    theta_rms_les = [((df["T2"]-df["T"]*df["T"])) for df in df_les]

    theta_rms_dns = ((df_dns["T2"]-df_dns["T"]*df_dns["T"]))

    les = {"urms": urms_les, "vrms": vrms_les, "wrms": wrms_les, "uv": uv_les, "u_theta": u_theta_les, "v_theta": v_theta_les, "theta_rms": theta_rms_les}
    dns = {"urms": urms_dns, "vrms": vrms_dns, "wrms": wrms_dns, "uv": uv_dns, "u_theta": u_theta_dns, "v_theta": v_theta_dns, "theta_rms": theta_rms_dns}

    return les, dns



def closure_terms(df_les: List[pd.DataFrame], Cp=1155, tau_compressible=False, pi_compressible=False) -> Tuple[Dict]:
    """
    Gets the closure terms for velocity velocity correlations
    and velocity temperature correlations
    """
    # Tau_{x, x}
    if tau_compressible:
        urms_struct = [df["STRUCTURAL_UU"]/df["RHO"] for df in df_les]
        urms_struct_dev = [df["STRUCTURAL_UU"]/df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"] for df in df_les]

        vrms_struct = [df["STRUCTURAL_WW"] /df["RHO"] for df in df_les]
        vrms_struct_dev = [df["STRUCTURAL_WW"] /df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"] for df in df_les]

        wrms_struct = [df["STRUCTURAL_VV"]/df["RHO"] for df in df_les]
        wrms_struct_dev = [df["STRUCTURAL_VV"]/df["RHO"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]  for df in df_les]

        uv_struct = [df["STRUCTURAL_UW"] / df["RHO"] for df in df_les]
    else: 
        urms_struct = [df["STRUCTURAL_UU"] for df in df_les]
        urms_struct_dev = [df["STRUCTURAL_UU"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"]) for df in df_les]

        vrms_struct = [df["STRUCTURAL_WW"]  for df in df_les]
        vrms_struct_dev = [df["STRUCTURAL_WW"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"]) for df in df_les]

        wrms_struct = [df["STRUCTURAL_VV"] for df in df_les]
        wrms_struct_dev = [df["STRUCTURAL_VV"] - 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])  for df in df_les]

        uv_struct = [df["STRUCTURAL_UW"] for df in df_les]


    if pi_compressible:
        u_theta_struct = [df["STRUCTURAL_USCALAR"]/df["RHO"] for df in df_les]
        v_theta_struct = [df["STRUCTURAL_WSCALAR"]/df["RHO"]  for df in df_les]
    else:
        u_theta_struct = [df["STRUCTURAL_USCALAR"] for df in df_les]
        v_theta_struct = [df["STRUCTURAL_WSCALAR"]  for df in df_les]

    urms_fonc = [- 2 * df["NUTURB_XX_DUDX"] for df in df_les]
    urms_fonc_dev  = [- 2 * (df["NUTURB_XX_DUDX"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])) for df in df_les]

    # Tau_{y, y}
    vrms_fonc = [- 2 * ( df["NUTURB_ZZ_DWDZ"]) for df in df_les]
    vrms_fonc_dev  = [- 2 * ( df["NUTURB_ZZ_DWDZ"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])) for df in df_les]

    # Tau_{z, z}
    wrms_fonc = [- 2 *df["NUTURB_YY_DVDY"] for df in df_les]
    wrms_fonc_dev  = [ - 2 *(df["NUTURB_YY_DVDY"] - 1/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"]))  for df in df_les]

    uv_fonc   = [- (df["NUTURB_XZ_DUDZ"] + df["NUTURB_XZ_DWDX"]) for df in df_les]
    u_theta_fonc   =[- 2* df["KAPPATURB_X_DSCALARDX"]   for df in df_les]
    v_theta_fonc   =[- 2 * df["KAPPATURB_Z_DSCALARDZ"]  for df in df_les]

    struct = {"urms_dev": urms_struct_dev, "urms":urms_struct, "vrms": vrms_struct,"vrms_dev": vrms_struct_dev, "wrms":wrms_struct,
              "wrms_dev":wrms_struct_dev , "uv": uv_struct, "u_theta": u_theta_struct, "v_theta": v_theta_struct}
    fonc = {"urms_dev": urms_fonc_dev, "urms":urms_fonc, "vrms": vrms_fonc,"vrms_dev": vrms_fonc_dev, "wrms":wrms_fonc,
              "wrms_dev":wrms_fonc_dev , "uv": uv_fonc, "u_theta": u_theta_fonc, "v_theta": v_theta_fonc}

    return struct, fonc


def adim_closure_terms(ref: RefData, tau: Dict[str, list[pd.DataFrame]], pi: Dict[str, list[pd.DataFrame]])->Dict[str, list[pd.DataFrame]]:
    middles  = [df.shape[0] //2 for df in tau["urms"]]
    tau_hot  = dict()
    tau_cold = dict()

    pi_hot   = dict()
    pi_cold  = dict()
    
    for key in ["urms_dev", "urms", "vrms_dev", "vrms", "wrms_dev", "wrms", "uv"]:
        tau_hot[key] = [t.values[middle:][::-1]/(ref.utau["hot"]**2) for t, middle in zip(tau[key], middles)]
        tau_cold[key] = [t.values[:middle]/(ref.utau["cold"]**2) for t, middle in zip(tau[key], middles)]

        pi_hot[key] = [t.values[middle:][::-1]/(ref.utau["hot"]**2) for t, middle in zip(pi[key], middles)]
        pi_cold[key] = [t.values[:middle]/(ref.utau["cold"]**2) for t, middle in zip(pi[key], middles)]

    for key in ["u_theta", "v_theta"]:
        tau_hot[key] = [t.values[middle:][::-1]/(ref.utau["hot"] * ref.sheer["thetatau"]["hot"]) for t, middle in zip(tau[key], middles)]
        tau_cold[key] = [t.values[:middle]/(ref.utau["cold"] * ref.sheer["thetatau"]["cold"]) for t, middle in zip(tau[key], middles)]

        pi_hot[key] = [t.values[middle:][::-1]/(ref.utau["hot"] * ref.sheer["thetatau"]["hot"]) for t, middle in zip(pi[key], middles)]
        pi_cold[key] = [t.values[:middle]/(ref.utau["cold"] * ref.sheer["thetatau"]["cold"]) for t, middle in zip(pi[key], middles)]

    tau_all = {"hot": tau_hot, "cold": tau_cold}
    pi_all  = {"hot": pi_hot, "cold": pi_cold}
    return tau_all, pi_all

def adim_second_order_stats(ref: RefData, les: Dict[str, list[pd.DataFrame]], dns: Dict[str, list[pd.DataFrame]]) -> Dict[str, list[pd.DataFrame]]:
    middles = [df.shape[0] //2 for df in les["urms"]]
    les_all_hot = dict()
    les_all_cold = dict()

    dns_all_hot = dict()
    dns_all_cold = dict()
    for key in ["urms", "vrms", "wrms", "uv"]:
        les_all_hot[key] = [t.values[middle:][::-1]/(ref.utau["hot"]**2) for t, middle in zip(les[key], middles)]
        les_all_cold[key] = [t.values[:middle]/(ref.utau["cold"]**2) for t, middle in zip(les[key], middles)]

    for key in ["u_theta", "v_theta"]:
        les_all_hot[key] = [t.values[middle:][::-1]/(ref.utau["hot"] * ref.thetatau["hot"]) for t, middle in zip(les[key], middles)]
        les_all_cold[key] = [t.values[:middle]/(ref.utau["cold"] * ref.thetatau["cold"]) for t, middle in zip(les[key], middles)]
    
    les_all_hot["theta_rms"] = [t.values[middle:][::-1]/(ref.thetatau["hot"]**2) for t, middle in zip(les[key], middles)]
    les_all_cold["theta_rms"] = [t.values[:middle]/(ref.thetatau["cold"]**2) for t, middle in zip(les[key], middles)]

    middle_dns = dns["urms"].shape[0]//2
    for key in ["urms", "vrms", "wrms", "uv"]:
        dns_all_hot[key]  = dns[key].values[middle_dns:][::-1]/(ref.utau["hot"]**2)
        dns_all_cold[key] = dns[key].values[:middle_dns]/(ref.utau["cold"]**2)

    for key in ["u_theta", "v_theta"]:
        dns_all_hot[key]  = dns[key].values[middle_dns:][::-1]/(ref.utau["hot"] * ref.thetatau["hot"])
        dns_all_cold[key] = dns[key].values[:middle_dns]/(ref.utau["cold"] * ref.thetatau["cold"])
    
    dns_all_hot["theta_rms"]  = dns[key].values[middle_dns:][::-1]/(ref.thetatau["hot"]**2)
    dns_all_cold["theta_rms"] = dns[key].values[:middle_dns]/(ref.thetatau["cold"]**2)

    les_all = {"hot":les_all_hot, "cold":les_all_cold}
    dns_all = {"hot":dns_all_hot, "cold":dns_all_cold}

    return les_all, dns_all



def mean_over_n_times(df: pd.DataFrame, n: int = 0):
    assert n > 0, "You need to specify a positive number of timesteps to do the mean over"
    times = df.index.get_level_values(0)
    time = times.unique(0)
    return df.loc[times >= time[-n]].groupby(level=1).mean()


def reynolds_bulk_each_time(df: pd.DataFrame, h: float):
    """
    Returns the bulk Reynolds number for each time step
    """
    return df["U"].groupby("time").apply(lambda x: x.mean()) \
        * h \
        * df["RHO"].groupby("time").apply(lambda x: x.mean()) \
        / df["NU"].groupby("time").apply(lambda x: x.mean())


def reynolds_bulk(df: pd.DataFrame, h: float):
    """
    Returns the bulk Reynolds number averaged over all time steps
    """
    return reynolds_bulk_each_time(df, h).mean()


def u_tau(df: pd.DataFrame, y: np.typing.ArrayLike):
    utau = np.sqrt(df["NU"] * np.abs(np.gradient(df["U"], y, edge_order=2)))
    return utau.iloc[0], utau.iloc[-1]


def re_tau(df: pd.DataFrame, y: np.array, h: float):
    utau = u_tau(df, y)
    return utau[0] * h / df["NU"].iloc[0], utau[-1] * h / df["NU"].iloc[-1]



def rms_les(x: List[pd.DataFrame], Cp: Union[float, None]=None)->Dict[str, List[pd.DataFrame]]:
    """
    Computes the RMS of the LES without adim, just plain RMS for
    urms, vrms, wrms, uv, u_theta, v_theta, theta_rms
    """
    if Cp is None:
        from warnings import warn
        warn("Implicitly using the function rms_les without setting Cp can be dangerous")
        Cp = 1155
    out = dict()
    out["urms"]      = [xx["UU"] - xx["U"]*xx["U"] - 1/3*(xx["UU"] - xx["U"]**2 + xx["VV"] - xx["V"]**2 + xx["WW"] - xx["W"]**2) + (xx["STRUCTURAL_UU"] - 1/3 * (xx["STRUCTURAL_UU"] + xx["STRUCTURAL_VV"] + xx["STRUCTURAL_WW"])) / xx["RHO"] - 2 * (xx["NUTURB_XX_DUDX"] - 1/3 * (xx["NUTURB_XX_DUDX"] + xx["NUTURB_YY_DVDY"] + xx["NUTURB_ZZ_DWDZ"])) / (xx["RHO"]) for xx in x]
    out["vrms"]      = [xx["WW"] - xx["W"]*xx["W"] - 1/3*(xx["UU"] - xx["U"]**2 + xx["VV"] - xx["V"]**2 + xx["WW"] - xx["W"]**2) + (xx["STRUCTURAL_WW"] - 1/3 * (xx["STRUCTURAL_UU"] + xx["STRUCTURAL_VV"] + xx["STRUCTURAL_WW"])) / xx["RHO"] - 2 * (xx["NUTURB_ZZ_DWDZ"] - 1/3 * (xx["NUTURB_XX_DUDX"] + xx["NUTURB_YY_DVDY"] + xx["NUTURB_ZZ_DWDZ"])) / (xx["RHO"]) for xx in x]
    out["wrms"]      = [xx["VV"] - xx["V"]*xx["V"] - 1/3*(xx["UU"] - xx["U"]**2 + xx["VV"] - xx["V"]**2 + xx["WW"] - xx["W"]**2) + (xx["STRUCTURAL_WW"] - 1/3 * (xx["STRUCTURAL_UU"] + xx["STRUCTURAL_VV"] + xx["STRUCTURAL_WW"])) / xx["RHO"] - 2 * (xx["NUTURB_YY_DVDY"] - 1/3 * (xx["NUTURB_XX_DUDX"] + xx["NUTURB_YY_DVDY"] + xx["NUTURB_ZZ_DWDZ"])) / (xx["RHO"]) for xx in x]
    out["uv"]        = [xx["UW"] - xx["U"]*xx["W"] + xx["STRUCTURAL_UW"]/xx["RHO"] - (xx["NUTURB_XZ_DUDZ"] + xx["NUTURB_XZ_DWDX"])/xx["RHO"] for xx in x]
    #### we multiply by rho cp = rho (r/Pth) (Pth CP/r):
    ####  * r/Pth: because we use grad T instead of grad 1/rho
    ####  * Pth Cp/r: because div_lambda_grad_T will be multiplied by r/Cp/Pth
    ####              but this should not
    ####  * rho: required in the favre formulation
    out["u_theta"]   = [xx["UT"] - xx["U"]*xx["T"] + (xx["STRUCTURAL_USCALAR"] - 2 * xx["KAPPATURB_X_DSCALARDX"])/(xx["RHO"] * Cp) for xx in x]
    out["v_theta"]   = [xx["WT"] - xx["W"]*xx["T"] + (xx["STRUCTURAL_WSCALAR"] - 2 * xx["KAPPATURB_Z_DSCALARDZ"])/(xx["RHO"] * Cp) for xx in x]
    out["theta_rms"] = [xx["T2"] - xx["T"]*xx["T"] for xx in x]
    
    return out

def mean_les(x, ref, mesh) -> Dict[str, List[pd.DataFrame]]:
    out = dict()
    half = len(mesh["hot"])
    out["U"]   = [xx["U"] for xx in x] 
    out["V"]   = [xx["W"] for xx in x] 
    out["T"]   = {"hot" : [(xx["T"]-ref.Tw["hot" ]).values[half:] for xx in x],
                  "cold": [(xx["T"]-ref.Tw["cold"]).values[:half] for xx in x]}
    out["NU"] = {
        "hot": [np.gradient(xx[::-1], mesh["hot" ], edge_order=2) for xx in out["T"]["hot" ]],
        "cold":[np.gradient(xx,       mesh["cold"], edge_order=2) for xx in out["T"]["cold"]]
    }
    out["Cf"]  = [xx["MU"] * np.gradient(xx["U"], xx["coordonnee_K"], edge_order=2) for xx in x]
    return out

def rms_dns(xx):
    """
    Computes the RMS of the DNS without adim, just plain RMS for
    urms, vrms, wrms, uv, u_theta, v_theta, theta_rms
    """
    out = dict()
    out["urms"]      = xx["UU"] - xx["U"]*xx["U"] - 1/3*(xx["UU"] - xx["U"]**2 + xx["VV"] - xx["V"]**2 + xx["WW"] - xx["W"]**2)
    out["vrms"]      = xx["WW"] - xx["W"]*xx["W"] - 1/3*(xx["UU"] - xx["U"]**2 + xx["VV"] - xx["V"]**2 + xx["WW"] - xx["W"]**2)
    out["wrms"]      = xx["VV"] - xx["V"]*xx["V"] - 1/3*(xx["UU"] - xx["U"]**2 + xx["VV"] - xx["V"]**2 + xx["WW"] - xx["W"]**2)
    out["uv"]        = xx["UW"] - xx["U"]*xx["W"]
    out["u_theta"]   = xx["UT"] - xx["U"]*xx["T"]
    out["v_theta"]   = xx["WT"] - xx["W"]*xx["T"]
    out["theta_rms"] = xx["T2"] - xx["T"]*xx["T"]
    
    return out


def mean_dns(x: pd.DataFrame, ref) -> Dict[str, pd.DataFrame]:
    xx = x
    out = dict()
    half = len(ref.df.U)//2
    out["U"]   = x["U"]
    out["V"]   = x["W"]
    out["T"]   = {"hot" : (xx["T"]-ref.Tw["hot" ]).values[::-1][:half],
                  "cold": (xx["T"]-ref.Tw["cold"]).values[:half]}
    out["NU"] = {
        "hot" : np.gradient(out["T"]["hot" ][:half], ref.yplus["hot" ], edge_order=2),
        "cold": np.gradient(out["T"]["cold"][:half]      , ref.yplus["cold"], edge_order=2)
    }
    out["Cf"]  = x["MU"] * np.gradient(x["U"], x["coordonnee_K"], edge_order=2)
    return out

def closure(xx, Cp: Union[float, None]=None):
    """
    Gets the closure, with knowledge of the compressibility
    of a model
    """
    if Cp is None:
        from warnings import warn
        warn("Implicitly using the function rms_les without setting Cp can be dangerous")
        Cp = 1155
    fonc              = dict()
    struct            = dict()

    rho = xx["RHO"]
    
    struct["urms"]        = (xx["STRUCTURAL_UU"] - 1/3 * (xx["STRUCTURAL_UU"] + xx["STRUCTURAL_VV"] + xx["STRUCTURAL_WW"])) / rho
    struct["vrms"]        = (xx["STRUCTURAL_WW"] - 1/3 * (xx["STRUCTURAL_UU"] + xx["STRUCTURAL_VV"] + xx["STRUCTURAL_WW"])) / rho
    struct["wrms"]        = (xx["STRUCTURAL_WW"] - 1/3 * (xx["STRUCTURAL_UU"] + xx["STRUCTURAL_VV"] + xx["STRUCTURAL_WW"])) / rho
    struct["uv"]          = (xx["STRUCTURAL_UW"]) / rho
    struct["u_theta"]     = (xx["STRUCTURAL_USCALAR"])/(rho*Cp)
    struct["v_theta"]     = (xx["STRUCTURAL_WSCALAR"])/(rho*Cp)
    
    fonc["urms"]      = - 2 * (xx["NUTURB_XX_DUDX"] - 1/3 * (xx["NUTURB_XX_DUDX"] + xx["NUTURB_YY_DVDY"] + xx["NUTURB_ZZ_DWDZ"])) / rho
    fonc["vrms"]      = - 2 * (xx["NUTURB_ZZ_DWDZ"] - 1/3 * (xx["NUTURB_XX_DUDX"] + xx["NUTURB_YY_DVDY"] + xx["NUTURB_ZZ_DWDZ"])) / rho
    fonc["wrms"]      = - 2 * (xx["NUTURB_YY_DVDY"] - 1/3 * (xx["NUTURB_XX_DUDX"] + xx["NUTURB_YY_DVDY"] + xx["NUTURB_ZZ_DWDZ"])) / rho
    fonc["uv"]        = - (xx["NUTURB_XZ_DUDZ"] + xx["NUTURB_XZ_DWDX"]) / rho
    fonc["u_theta"]   = - 2 * xx["KAPPATURB_X_DSCALARDX"] / (rho * Cp)
    fonc["v_theta"]   = - 2 * xx["KAPPATURB_Z_DSCALARDZ"] / (rho * Cp)
    return fonc, struct

def normalize_quantity(qtty, ref):
    """
    Get the normalizing quantities in the form of a dictionary
    """
    if "urms" in qtty or "vrms" in qtty or "wrms" in qtty or "uv" in qtty:
        return {"hot": ref.utau["hot"]**2, "cold": ref.utau["cold"]**2}
    if "u_theta" in qtty.lower() or "v_theta" in qtty.lower():
        return {"hot": ref.utau["hot"] * ref.thetatau["hot"], "cold": ref.utau["cold"] * ref.thetatau["cold"]}
    if "theta_rms" in qtty.lower():
        return {"hot": ref.thetatau["hot"]**2, "cold": ref.thetatau["cold"]**2}
    if "u".lower() in qtty.lower() or "v".lower() in qtty.lower() or "w".lower() in qtty.lower():
        return {"hot": ref.utau["hot"], "cold": ref.utau["cold"]}
    if "nu" in qtty.lower() or "lambdadtdz" in qtty.lower():
        return {"hot": 1, "cold": 1}
    if "cf".lower() in qtty.lower():
        return {
            "hot": ref.rho_bulk * (ref.u_bulk ** 2)/2,
            "cold":ref.rho_bulk * (ref.u_bulk ** 2)/2
        }
    if "T".lower() in qtty.lower():
        return {"hot": ref.thetatau["hot"], "cold": ref.thetatau["cold"]}
