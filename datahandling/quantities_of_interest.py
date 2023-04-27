from typing import List
import pandas as pd

def compute_rms_quantities(df_les: List[pd.DataFrame], df_dns: pd.DataFrame):
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
    urms_les = [
        df["UU"] - df["U"]**2 - df["STRUCTURAL_UU"]/df["RHO"]
        - 2 * df["NUTURB_XX_DUDX"]
        + 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"] 
        + 2/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])
        for df in df_les
    ]
    urms_dns = df_dns["UU"] - df_dns["U"]**2 \
        - df_dns["STRUCTURAL_UU"]/df_dns["RHO"] \
        - 2 * df_dns["NUTURB_XX_DUDX"] \
        + 1/3 * (df_dns["STRUCTURAL_UU"] + df_dns["STRUCTURAL_VV"] + df_dns["STRUCTURAL_WW"])/df_dns["RHO"] \
        + 2/3 * (df_dns["NUTURB_XX_DUDX"] + df_dns["NUTURB_YY_DVDY"] + df_dns["NUTURB_ZZ_DWDZ"])
    vrms_les = [
            df["WW"] - df["W"]**2 
            - 1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
            - df["STRUCTURAL_WW"]/df["RHO"]
            - 2 * df["NUTURB_ZZ_DWDZ"]
            + 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]
            + 2/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])
        for df in df_les
    ]
    vrms_dns = (
            df_dns["WW"] - df_dns["W"]**2  \
            - 1/3 * (df_dns["UU"] - df_dns["U"]**2 + df_dns["VV"] - df_dns["V"]**2 + df_dns["WW"] - df_dns["W"]**2) \
            - df_dns["STRUCTURAL_WW"]/df_dns["RHO"] \
            - 2 * df_dns["NUTURB_ZZ_DWDZ"] \
            + 1/3 * (df_dns["STRUCTURAL_UU"] + df_dns["STRUCTURAL_VV"] + df_dns["STRUCTURAL_WW"])/df_dns["RHO"] \
            + 2/3 * (df_dns["NUTURB_XX_DUDX"] + df_dns["NUTURB_YY_DVDY"] + df_dns["NUTURB_ZZ_DWDZ"]))
    wrms_les = [
            df["V"] - df["V"]**2 
            - 1/3 * (df["UU"] - df["U"]**2 + df["VV"] - df["V"]**2 + df["WW"] - df["W"]**2)
            - df["STRUCTURAL_VV"]/df["RHO"]
            - 2 * df["NUTURB_YY_DVDY"]
            + 1/3 * (df["STRUCTURAL_UU"] + df["STRUCTURAL_VV"] + df["STRUCTURAL_WW"])/df["RHO"]
            + 2/3 * (df["NUTURB_XX_DUDX"] + df["NUTURB_YY_DVDY"] + df["NUTURB_ZZ_DWDZ"])
        for df in df_les
    ]
    wrms_dns =    (
            df_dns["V"] - df_dns["V"]**2  \
            - 1/3 * (df_dns["UU"] - df_dns["U"]**2 + df_dns["VV"] - df_dns["V"]**2 + df_dns["WW"] - df_dns["W"]**2) \
            - df_dns["STRUCTURAL_VV"]/df_dns["RHO"] \
            - 2 * df_dns["NUTURB_YY_DVDY"] \
            + 1/3 * (df_dns["STRUCTURAL_UU"] + df_dns["STRUCTURAL_VV"] + df_dns["STRUCTURAL_WW"])/df_dns["RHO"] \
            + 2/3 * (df_dns["NUTURB_XX_DUDX"] + df_dns["NUTURB_YY_DVDY"] + df_dns["NUTURB_ZZ_DWDZ"]))

    uv_les = [(df["UW"]-df["U"]*df["W"]-df["STRUCTURAL_UW"]/df["RHO"]-(df["NUTURB_XZ_DUDZ"]+df["NUTURB_XZ_DWDX"])) for df in df_les]
    uv_dns = df_dns["UW"]-df_dns["U"]*df_dns["W"]-df_dns["STRUCTURAL_UW"]/df_dns["RHO"]-(df_dns["NUTURB_XZ_DUDZ"]+df_dns["NUTURB_XZ_DWDX"])

    u_theta_les = [   (df["UT"]-    df["U"]*    df["T"]-    df["KAPPATURB_X_DSCALARDX"]*     df["RHO"]*Cp-    df["STRUCTURAL_USCALAR"]/    df["T"]/    df["RHO"]) for df in df_les]
    u_theta_dns = (df_dns["UT"]-df_dns["U"]*df_dns["T"]-df_dns["KAPPATURB_X_DSCALARDX"]* df_dns["RHO"]*Cp-df_dns["STRUCTURAL_USCALAR"]/df_dns["T"]/df_dns["RHO"])

    v_theta_les = [(   df["WT"]-    df["W"]*    df["T"]-    df["KAPPATURB_Z_DSCALARDZ"]*     df["RHO"]*Cp-    df["STRUCTURAL_WSCALAR"]/    df["RHO"]/    df["T"]) for df in df_les]
    v_theta_dns =  df_dns["WT"]-df_dns["W"]*df_dns["T"]-df_dns["KAPPATURB_Z_DSCALARDZ"]* df_dns["RHO"]*Cp-df_dns["STRUCTURAL_WSCALAR"]/df_dns["RHO"]/df_dns["T"]


    theta_rms_les = [(sqrt(df["T2"]-df["T"]*df["T"])) for df in df_les]
    theta_rms_dns = (sqrt(df_dns["T2"]-df_dns["T"]*df_dns["T"]))
    
    return (urms_les, urms_dns, vrms_les, vrms_dns, wrms_les, wrms_dns, uv_les, uv_dns, u_theta_les, u_theta_dns, v_theta_les, v_theta_dns, theta_rms_les, theta_rms_dns)
