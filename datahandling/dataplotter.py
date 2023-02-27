from typing import Tuple, Union

import re
import pandas as pd
from datahandling.dataloader import DataLoader, DataLoaderPandas
import numpy as np
import matplotlib.pyplot as plt


def matplotlib_latex_params(size=20) -> None:
    plt.rc(
        'text', usetex=True
    )
    plt.rc(
        'font',
        **
        {
            "family": "sans-serif",
            "sans-serif": "Helvetica",
            "size": size
        })
    plt.rcParams.update(
        {
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )


class DataPlotter(DataLoader):
    def __init__(self,
                 directory: str = "rep22",
                 type_stat: str = "statistiques",
                 columns: Union[str, int] = None,
                 separator: str = "\s",
                 type_file: str = ".txt"):
        super().__init__(directory, type_stat, columns, separator, type_file)
        self.load_data()

    def get_column_label(self, column_number: Union[int, np.integer]) -> str:
        if isinstance(column_number, (int, np.integer)):
            return self.column_handler_num2key(column_number)
        raise TypeError(
            f"column_number must be a string, you entered a {type(column_number)}")

    def plot_space(self, time_index, variables):
        for var in variables:
            fig = plt.figure(figsize=(16, 9))
            plt.plot(self.space, self.data[time_index, :, self.key2num(var)])
            plt.xlabel("K direction (m)")
            plt.ylabel(f"{self.num2key(var)}")
            plt.title(f"Plot of {var} as a function of space")
        plt.show()

    def plot_time(self, space_index, variables):
        for var in variables:
            plt.figure(figsize=(16, 9))
            plt.plot(self.time, self.data[:, space_index, self.key2num(var)])
            plt.xlabel("Time (s)")
            plt.ylabel(f"{self.num2key(var)}")
            plt.title(f"Plot of {var} as a function of space")
        plt.show()


def compute_Rij_middle_canal(loader: DataLoaderPandas, df: pd.DataFrame):
    middle_canal = len(loader.space)//2
    newdf = pd.DataFrame()
    from itertools import combinations_with_replacement
    for comb in combinations_with_replacement("UVW", 2):
        column = f"{comb[0]}'{comb[1]}'"

        newdf[column] = (df[f"{comb[0]}{comb[1]}"] - df[comb[0]] * df[comb[1]]
                         ).groupby("time").apply(lambda x: x.iloc[middle_canal])

    return newdf


def compute_uu_middle_canal(loader: DataLoaderPandas, df: pd.DataFrame):
    middle_canal = len(loader.space)//2
    newdf = pd.DataFrame()
    newdf["U"] = df["U"].groupby("time").apply(lambda x: x.iloc[middle_canal])
    newdf["U'U'"] = (df["UU"] - df["U"] **
                     2).groupby("time").apply(lambda x: x.iloc[middle_canal])
    return newdf


def compute_T_middle_canal(loader: DataLoaderPandas, df: pd.DataFrame):
    middle_canal = len(loader.space)//2
    newdf = pd.DataFrame()
    newdf["T"] = df["T"].groupby("time").apply(lambda x: x.iloc[middle_canal])
    newdf["T'T'"] = (df["T2"] - df["T"] **
                     2).groupby("time").apply(lambda x: x.iloc[middle_canal])
    return newdf


def verify_convergence(rep: str = "./"):
    import pandas as pd
    matplotlib_latex_params()
    loader = DataLoaderPandas(rep, type_stat="statistiques")
    df: pd.DataFrame = loader.load_data()
    uu = compute_uu_middle_canal(loader, df)
    T = compute_T_middle_canal(loader, df)

    # all = pd.concat([uu, T], axis=1)
    fig, axs = plt.subplots(1, len(uu.columns), figsize=(16, 9))
    for c, ax in zip(uu.columns, axs):
        uu[c].plot(ylabel=c, ax=ax)
    plt.grid(True)
    fig.suptitle(r"$U$ and $\langle U' U' \rangle$ in middle of canal")

    fig, axs = plt.subplots(1, len(T.columns), figsize=(16, 9))
    for c, ax in zip(T.columns, axs):
        T[c].plot(ylabel=c, ax=ax)
    plt.grid(True)
    fig.suptitle(r"$T$ and $\langle T' T' \rangle$ in middle of canal")

    plt.show()


def verif_convergence_irene_les(rep: str = "./"):
    import pandas as pd
    matplotlib_latex_params()
    loader = DataLoaderPandas(rep, type_stat="statistiques")
    df: pd.DataFrame = loader.load_data()
    uu = compute_uu_middle_canal(loader, df)
    T = compute_T_middle_canal(loader, df)
    import os
    home_directory = os.path.expanduser('~')
    # all = pd.concat([uu, T], axis=1)
    fig, axs = plt.subplots(1, len(uu.columns), figsize=(16, 9))
    for c, ax in zip(uu.columns, axs):
        uu[c].plot(ylabel=c, ax=ax)
    plt.grid(True)
    type_sim, _, models, mesh = os.getcwd().split("/")[-4:]
    turb_visc_diff, disc_qdm, disc_rho = models.split("_")
    discretisation_qdm, discretisation_rho = disc_qdm, disc_rho
    turb_visc, turb_diff = turb_visc_diff.split("-")
    turb_visc_models, turb_diff_models = turb_visc.split(
        "+"), turb_diff.split("+")

    turb_visc_constants, turb_diff_constants = re.findall(
        r"\d+", turb_visc), re.findall(r"\d+", turb_diff)

    turb_visc_constants_corrected = ["0."+x[1:]
                                     for x in turb_visc_constants if x.startswith("0")]
    turb_diff_constants_corrected = ["0."+x[1:]
                                     for x in turb_diff_constants if x.startswith("0")]

    tau_string = ""
    for tau_constant, tau_bad_constant, tau_model in zip(turb_visc_constants_corrected, turb_visc_constants, turb_visc_models):
        tau_model = tau_model.split(tau_bad_constant)[0]
        tau_string += f"{tau_constant}" + r"\tau^{" + f"{tau_model}" + r"}"
        if len(turb_visc_constants_corrected) > 1:
            tau_string += "+"

    pi_string = ""
    for pi_constant, pi_bad_constant, pi_model in zip(turb_diff_constants_corrected, turb_diff_constants, turb_diff_models):
        pi_model = pi_model.split(pi_bad_constant)[0]
        pi_string += f"{pi_constant}" + r"\pi^{" + f"{pi_model}" + r"}"
        if len(turb_visc_constants_corrected) > 1:
            pi_string += "+"
    fig.suptitle(
        r"$\langle U \rangle$, $\langle U' U' \rangle$ middle canal for  $\tau_{ij}=" +
        tau_string + r"$, $\pi_j=" + pi_string + "$"
        f"qdm: {discretisation_qdm}, rho: {discretisation_rho}, mesh: {mesh}")
    os.makedirs(os.path.join(home_directory,
                             "simulation_advancement"), exist_ok=True)
    plt.savefig(os.path.join(home_directory,
                "simulation_advancement", f"{models}_{mesh}_velocity.pdf"), dpi=150)

    fig, axs = plt.subplots(1, len(T.columns), figsize=(16, 9))
    for c, ax in zip(T.columns, axs):
        T[c].plot(ylabel=c, ax=ax)
    plt.grid(True)
    fig.suptitle(
        r"$\langle T \rangle$ and $\langle T' T' \rangle$ $\tau_{ij}=" +
        tau_string + r"$, $\pi_j=" + pi_string + "$"

        f"qdm: {discretisation_qdm}, rho: {discretisation_rho}, mesh: {mesh}")
    plt.savefig(
        os.path.join(home_directory,
                     "simulation_advancement", f"{models}_{mesh}_temperature.pdf"), dpi=150

    )

    # plt.show()
