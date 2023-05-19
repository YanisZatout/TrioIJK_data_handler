from typing import Union

import re
import pandas as pd
from datahandling.dataloader import DataLoader, DataLoaderPandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


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


def verif_convergence_irene_les(rep: str = "./", savepath=None):
    import pandas as pd
    # matplotlib_latex_params()
    loader = DataLoaderPandas(rep, type_stat="statistiques")
    df: pd.DataFrame = loader.load_data()
    uu = compute_uu_middle_canal(loader, df)
    T = compute_T_middle_canal(loader, df)
    import os
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

    turb_visc_constants_corrected = ["0."+x[1:] for x in turb_visc_constants if x.startswith("0")]
    turb_diff_constants_corrected = ["0."+x[1:] for x in turb_diff_constants if x.startswith("0")]

    tau_string = ""
    for tau_constant, tau_bad_constant, tau_model in zip(turb_visc_constants_corrected, turb_visc_constants, turb_visc_models):
        tau_model = tau_model.split(tau_bad_constant)[0]
        tau_string += f"{tau_constant}" + "tau^" + f"{tau_model}"
        if len(turb_visc_constants_corrected) > 1:
            tau_string += "+"

    pi_string = ""
    for pi_constant, pi_bad_constant, pi_model in zip(turb_diff_constants_corrected, turb_diff_constants, turb_diff_models):
        pi_model = pi_model.split(pi_bad_constant)[0]
        pi_string += f"{pi_constant}" + "\pi^" + f"{pi_model}"
        if len(turb_visc_constants_corrected) > 1:
            pi_string += "+"
    fig.suptitle(
        "U, U' U'  middle canal for  tau_{ij}=" +
        tau_string + ", \pi_j=" + pi_string + f" qdm: {discretisation_qdm}, rho: {discretisation_rho}, mesh: {mesh}")
    plt.savefig(os.path.join(savepath,
                "simulation_advancement", f"{models}_{mesh}_{discretisation_qdm}_{discretisation_rho}_velocity.pdf"), dpi=150)

    fig, axs = plt.subplots(1, len(T.columns), figsize=(16, 9))
    for c, ax in zip(T.columns, axs):
        T[c].plot(ylabel=c, ax=ax)
    plt.grid(True)
    fig.suptitle(
        "T and T' T' tau_{ij}=" +
        tau_string + ", pi_j=" + pi_string + f" qdm: {discretisation_qdm}, rho: {discretisation_rho}, mesh: {mesh}")

    plt.savefig(
        os.path.join(savepath,
                     "simulation_advancement", f"{models}_{mesh}_{discretisation_qdm}_{discretisation_rho}_temperature.pdf"), dpi=150

    )

class PlotParams(object):
    def __init__(self, models_str, info_func=None, lines=["", "--", "-.", ":"], markers=None, hot_cold="cold", color_scaling_coeff=2):
        if not info_func:
            def info_func(model):
                mesh = model.split(" ")[0]
                model_tau = model.split()[1].split("-")[0]
                model_pi = model.split()[1].split("-")[1].split("_")[0]
                disc_qdm = model.split("_")[1]
                disc_mass = model.split("_")[2]
                return dict(mesh=mesh, models=model_tau+"-"+model_pi, disc=disc_qdm+"-"+disc_mass)
        self.info_func = info_func
        self.lines = lines
        self.n_models = len(models_str)

        models_str = [self.info_func(m) for m in models_str]

        # Unreadable part
        # At each unique different mesh, model and discretisation, we give
        # an integer value

        self.meshes = dict(zip(np.unique(np.array([m["mesh"] for m in models_str])).tolist(
        ), np.arange(len(np.unique(np.array([m["mesh"] for m in models_str])).tolist()))))
        self.models = dict(zip(np.unique(np.array([m["models"] for m in models_str])).tolist(
        ), np.arange(len(np.unique(np.array([m["models"] for m in models_str])).tolist()))))
        self.disc = dict(zip(np.unique(np.array([m["disc"] for m in models_str])).tolist(
        ), np.arange(len(np.unique(np.array([m["disc"] for m in models_str])).tolist()))))

        from matplotlib.lines import Line2D
        if not markers:
            self.markers = [m for m, func in Line2D.markers.items(
            ) if func != 'nothing' and m not in Line2D.filled_markers]
        else:
            self.markers = markers

        # Number of meshes
        self.n_lines = len(self.meshes)*color_scaling_coeff
        self.min_index_value = self.n_lines//3
        c = np.arange(self.n_lines)[::-1]
        # If the simulation is in an anisothermal setting
        self.c_max = c_max = np.percentile(c, 100)
        self.c_min = c_min = np.percentile(c, 0)
        if hot_cold == "cold":
            norm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)
            self.cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
            self.cmap.set_array([])
        else:
            norm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)
            self.cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds)
            self.cmap.set_array([])

    def __getitem__(self, model):
        a = self.info_func(model)
        mesh = self.c_max - self.meshes[a["mesh"]]
        model = self.models[a["models"]]
        disc = self.disc[a["disc"]]
        return dict(
            c=self.cmap.to_rgba(mesh),
            linestyle=self.lines[model],
            marker=self.markers[disc]
        )
