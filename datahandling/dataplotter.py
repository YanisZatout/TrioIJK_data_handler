from typing import Union, Type
import re
import pandas as pd
from datahandling.dataloader import DataLoader, DataLoaderPandas
from .quantities_of_interest import closure_terms, adim_closure_terms, rms, adim_second_order_stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from natsort import natsorted
import os


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
    df_24h: pd.DataFrame = loader.load_last_24h()
    uu = compute_uu_middle_canal(loader, df)
    T = compute_T_middle_canal(loader, df)

    uu24h = compute_uu_middle_canal(loader, df_24h)
    T24h = compute_T_middle_canal(loader, df_24h)
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
                 f"{models}_{mesh}_{discretisation_qdm}_{discretisation_rho}_velocity.pdf"), dpi=150)

    fig, axs = plt.subplots(1, len(T.columns), figsize=(16, 9))
    for c, ax in zip(T.columns, axs):
        T[c].plot(ylabel=c, ax=ax)
    plt.grid(True)
    fig.suptitle(
        "T and T' T' tau_{ij}=" +
        tau_string + ", pi_j=" + pi_string + f" qdm: {discretisation_qdm}, rho: {discretisation_rho}, mesh: {mesh}")

    plt.savefig(os.path.join(savepath, f"{models}_{mesh}_{discretisation_qdm}_{discretisation_rho}_temperature.pdf"), dpi=150)

    ### DEFINE 24h part
    # Convergence check
    converged = False
    upct = np.abs((uu24h.iloc[:, 0].max() - uu24h.iloc[:, 0].min())/uu24h.iloc[:, 0].min()) * 100
    uuptc = np.abs((uu24h.iloc[:, 1].max() - uu24h.iloc[:, 1].min())/uu24h.iloc[:, 1].min()) * 100
    Tpct = np.abs((T24h.iloc[:, 0].max() - T24h.iloc[:, 0].min())/T24h.iloc[:, 0].min()) * 100
    TTpct = np.abs((T24h.iloc[:, 1].max() - T24h.iloc[:, 1].min())/T24h.iloc[:, 1].min()) * 100
    if upct < 1 and uuptc < 1 and Tpct < 1 and TTpct < 1:
        converged = True
    # End of convergence check

    fig, axs = plt.subplots(1, len(uu.columns), figsize=(16, 9))
    for c, ax in zip(uu.columns, axs):
        uu24h[c].plot(ylabel=c, ax=ax)
        plt.grid(True)

    fig.suptitle(fr"$\langle U \rangle$, $\langle U' U' \rangle$ last 24h, Convergence: {converged}")
    plt.savefig(os.path.join(savepath, f"centerline_streamwise_velocity_24h.pdf"), dpi=150)

    fig, axs = plt.subplots(1, len(T.columns), figsize=(16, 9))
    for c, ax in zip(T.columns, axs):
        T24h[c].plot(ylabel=c, ax=ax)
    plt.grid(True)
    fig.suptitle(fr"$\langle T \rangle$, $\langle T' T' \rangle$ last24h, Convergence: {converged}")

    plt.savefig(os.path.join(savepath, f"centerline_streamwise_velocity_temperature_24h.pdf"), dpi=150)
    ### END 24h part
    return uu24h, T24h

def verif_convergence_irene_dns(rep: str = "./", savepath=None):
    import pandas as pd
    # matplotlib_latex_params()
    loader = DataLoaderPandas(rep, type_stat="statistiques")
    df: pd.DataFrame = loader.load_data()
    df_24h: pd.DataFrame = loader.load_last_24h()
    uu = compute_uu_middle_canal(loader, df)
    T =  compute_T_middle_canal(loader, df)

    uu24h = compute_uu_middle_canal(loader, df_24h)
    T24h  = compute_T_middle_canal (loader, df_24h)
    import os

    fig, axs = plt.subplots(1, len(uu.columns), figsize=(16, 9))
    for c, ax in zip(uu.columns, axs):
        uu[c].plot(ylabel=c, ax=ax)
        plt.grid(True)

    fig.suptitle(r"$\langle U \rangle$, $\langle U' U' \rangle$")
    plt.savefig(os.path.join(savepath, f"centerline_streamwise_velocity.pdf"), dpi=150)

    fig, axs = plt.subplots(1, len(T.columns), figsize=(16, 9))
    for c, ax in zip(T.columns, axs):
        T[c].plot(ylabel=c, ax=ax)
    plt.grid(True)
    fig.suptitle(r"$\langle T \rangle$, $\langle T' T' \rangle$")

    plt.savefig(os.path.join(savepath, f"centerline_streamwise_temperature.pdf"), dpi=150)


    fig, axs = plt.subplots(1, len(uu.columns), figsize=(16, 9))
    for c, ax in zip(uu.columns, axs):
        uu24h[c].plot(ylabel=c, ax=ax)
        plt.grid(True)

    fig.suptitle(r"$\langle U \rangle$, $\langle U' U' \rangle$ last 24h")
    plt.savefig(os.path.join(savepath, f"centerline_streamwise_velocity_24h.pdf"), dpi=150)

    fig, axs = plt.subplots(1, len(T.columns), figsize=(16, 9))
    for c, ax in zip(T.columns, axs):
        T24h[c].plot(ylabel=c, ax=ax)
    plt.grid(True)
    fig.suptitle(r"$\langle T \rangle$, $\langle T' T' \rangle$ last24h")

    plt.savefig(os.path.join(savepath, f"centerline_streamwise_velocity_temperature_24h.pdf"), dpi=150)
    return None

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

def make_legend_mean(
    ax: Union[plt.axes, np.typing.ArrayLike],
    elements: list[dict]= {
        "linestyle": {"DNS":'-', "LES":'--'}, 
        "marker": {"DNS":'', "LES":''}, 
        "colors": {"hot":"lightcoral", "cold":"lightblue"}
    },
    quantities: list = None
) -> None:
    linestyle_les = elements["linestyle"]["LES"]
    linestyle_dns = elements["linestyle"]["DNS"]
    
    marker_les = elements["marker"]["LES"]
    marker_dns = elements["marker"]["DNS"]
    
    color_cold = elements["color"]["cold"]
    color_hot  = elements["color"]["hot"]
    
    
    for idx_quantity, quantity in enumerate(quantities):
        dummy_lines = [
                plt.Line2D([0], [0], color='black', marker=marker_les, linestyle=linestyle_les, label="LES"),
                plt.Line2D([0], [0], color='black', marker=marker_dns, linestyle=linestyle_dns, label="DNS")
        ]
        dummy_lines = [*dummy_lines,
            plt.Line2D([0], [0], color=color_cold, linestyle='-', label="cold"),
            plt.Line2D([0], [0], color=color_hot, linestyle='-', label="hot")
        ]
        ax[idx_quantity][-1].legend(handles=dummy_lines, loc='center left', bbox_to_anchor=(1, 0.5))


def make_legend_rms(
    ax: Union[plt.axes, np.typing.ArrayLike],
    elements: list[dict]= {
        "linestyle": {"DNS":'-', "LES":'--', "struct":'none', "fonc":'none'}, 
        "marker": {"DNS":'', "LES":'', "struct":'.', "fonc":'*'}, 
        "colors": {"hot":"lightcoral", "cold":"lightblue"}
    },
    quantities: list = None,
    is_fonc=True,
    is_struct=True
) -> None:
    linestyle_les = elements["linestyle"]["LES"]
    linestyle_dns = elements["linestyle"]["DNS"]
    linestyle_struct = elements["linestyle"]["struct"]
    linestyle_fonc = elements["linestyle"]["fonc"]
    
    marker_les = elements["marker"]["LES"]
    marker_dns = elements["marker"]["DNS"]
    marker_struct = elements["marker"]["struct"]
    marker_fonc = elements["marker"]["fonc"]
    
    color_cold = elements["color"]["cold"]
    color_hot  = elements["color"]["hot"]
    
    
    for idx_quantity, quantity in enumerate(quantities):
        dummy_lines = [
                plt.Line2D([0], [0], color='black', marker=marker_les, linestyle=linestyle_les, label="LES"),
                plt.Line2D([0], [0], color='black', marker=marker_dns, linestyle=linestyle_dns, label="DNS")
        ]
        if not "theta_rms" in quantity:
            if is_struct:
                dummy_lines = [*dummy_lines,
                    plt.Line2D([0], [0], color='black', marker=marker_struct, linestyle=linestyle_struct, label="struct"),
                ]
            if is_fonc:
                dummy_lines=[*dummy_lines,
                    plt.Line2D([0], [0], color='black', marker=marker_fonc, linestyle=linestyle_fonc,   label="fonct")
                ]
        dummy_lines = [*dummy_lines,
            plt.Line2D([0], [0], color=color_cold, linestyle='-', label="cold"),
            plt.Line2D([0], [0], color=color_hot, linestyle='-', label="hot")
        ]
        ax[idx_quantity][-1].legend(handles=dummy_lines, loc='center left', bbox_to_anchor=(1, 0.5))


class SimuMartin(object):
    def __init__(self, /, h, Cp: float = float(1155), dir_les: str=None, dir_dns: str=None)->None:
        self.h = self.delta = h
        if dir_les is None:
            self.dir_les = "."
        else:
            self.dir_les = dir_les
        if dir_dns is None:
            self.dir_dns = "."
        else:
            self.dir_dns = dir_dns
        self.Cp = Cp
    def load(self, dir_les: str = None, dir_dns: str = None)->None:
        if dir_les is None:
            dir_les = self.dir_les
        if dir_dns is None:
            dir_dns = self.dir_dns
        directory = dir_les
        afiles = natsorted(os.listdir(os.path.join(directory, "AAA")))
        bfiles = natsorted(os.listdir(os.path.join(directory, "BAB")))
        cfiles = natsorted(os.listdir(os.path.join(directory, "CAC")))
        
        a_inter_b = list(set(afiles).intersection(bfiles))
        self.o = list(set(a_inter_b).intersection(cfiles))

        self.models = np.array(self.o)
        self.df = {"A":[], "B":[], "C":[]}
        
        for file in self.o:
            path = os.path.join(directory, "AAA", file)
            self.df["A"].append(
                DataLoaderPandas(path, "statistiques").load_last()
            )
            # print(len(self.models))
        for file in self.o:
            path = os.path.join(directory, "BAB", file)
            self.df["B"].append(
                DataLoaderPandas(path, "statistiques").load_last()
            )
        for file in self.o:
            path = os.path.join(directory, "CAC", file)
            self.df["C"].append(
                DataLoaderPandas(path, "statistiques").load_last()
            )
        self.df_dns = DataLoaderPandas(directory=dir_dns, type_stat="statistiques").load_last()
    def rms(self) -> None:
        from .quantities_of_interest import compute_rms_quantities
        self.arms, self.rms_dns = compute_rms_quantities(self.dfa, self.df_dns, Cp=self.Cp)
        
    def discriminate_compressibility(self)->None:
        self.tau_comp = []
        self.pi_comp  = []
        
        compressible_regex = "(((A|G|B|S))c[0-9]|_comp)"
        for amod in self.models:
            aword = amod.split("-")[0]
            matches = re.findall(compressible_regex, aword)
            if len(matches) >0:
                self.tau_comp.append(True)
            else:
                self.tau_comp.append(False)
            aword = amod.split("-")[-1] if len(amod.split("-")) > 1 else ""
            matches = re.findall(compressible_regex, aword)
            if len(matches) > 0:
                self.pi_comp.append(True)
            else:
                self.pi_comp.append(False)
        self.tau_comp = np.array(self.tau_comp)
        self.pi_comp  = np.array(self.pi_comp )
    def compute_yplus(self, ref: Type["RefData"])->None:
        self.amiddle = len(self.df["A"][0].coordonnee_K)//2
        self.bmiddle = len(self.df["B"][0].coordonnee_K)//2
        self.cmiddle = len(self.df["C"][0].coordonnee_K)//2
        ayplus_cold = self.df["A"][0].coordonnee_K.values[:self.amiddle] * ref.utau["cold"] * ref.df["RHO"].iloc[0] / ref.df["MU"].iloc[0]
        byplus_cold = self.df["B"][0].coordonnee_K.values[:self.bmiddle] * ref.utau["cold"] * ref.df["RHO"].iloc[0] / ref.df["MU"].iloc[0]
        cyplus_cold = self.df["C"][0].coordonnee_K.values[:self.cmiddle] * ref.utau["cold"] * ref.df["RHO"].iloc[0] / ref.df["MU"].iloc[0]
        
        ayplus_hot = (2*self.h - self.df["A"][0].coordonnee_K.values[self.amiddle:][::-1]) * ref.utau["hot"] * ref.df["RHO"].iloc[-1] / ref.df["MU"].iloc[-1]
        byplus_hot = (2*self.h - self.df["B"][0].coordonnee_K.values[self.bmiddle:][::-1]) * ref.utau["hot"] * ref.df["RHO"].iloc[-1] / ref.df["MU"].iloc[-1]
        cyplus_hot = (2*self.h - self.df["C"][0].coordonnee_K.values[self.cmiddle:][::-1]) * ref.utau["hot"] * ref.df["RHO"].iloc[-1] / ref.df["MU"].iloc[-1]
        
        self.y     = {"A": self.df["A"][0].coordonnee_K.values, "B": self.df["B"][0].coordonnee_K.values, "C":self.df["C"][0].coordonnee_K.values}
        self.yplus = {"hot": {"A": ayplus_hot, "B": byplus_hot, "C": cyplus_hot}, "cold": {"A": ayplus_cold, "B": byplus_cold, "C": cyplus_cold}}
    def compute_tau_pi(self, ref: Type["RefData"])->None:
        Cp = self.Cp
        self.pi = {"A":[], "B":[], "C":[]}
        self.tau = {"A":[], "B":[], "C":[]}
        self.rms_les = {"A":[], "B":[], "C":[]}
        self.rms_les_dim = {"A":[], "B":[], "C":[]}
        self.ordered_models = []
        self.is_empty = {"A": np.zeros((2, 2), dtype=bool), "B": np.zeros((2, 2), dtype=bool), "C": np.zeros((2, 2), dtype=bool)}
        
        for idx_tau, comp_tau in enumerate([True, False]):
            for idx_pi, comp_pi in enumerate([True, False]):
                dfa = np.array(self.df["A"].copy())
                dfb = np.array(self.df["B"].copy())
                dfc = np.array(self.df["C"].copy())
                cond = (self.tau_comp == comp_tau) & (self.pi_comp == comp_pi)
                dfa = dfa[cond]
                dfb = dfb[cond]
                dfc = dfc[cond]
                self.ordered_models.append(self.models[cond])
                
                dfa = [pd.DataFrame(df, columns=self.df["A"][0].columns) for df in dfa]
                dfb = [pd.DataFrame(df, columns=self.df["A"][0].columns) for df in dfb]
                dfc = [pd.DataFrame(df, columns=self.df["A"][0].columns) for df in dfc]
                
                a_tau, a_pi = closure_terms(dfa, Cp, comp_tau, comp_pi)
                b_tau, b_pi = closure_terms(dfb, Cp, comp_tau, comp_pi)
                c_tau, c_pi = closure_terms(dfc, Cp, comp_tau, comp_pi)
                
                a_tau, a_pi = adim_closure_terms(ref, a_tau, a_pi)
                b_tau, b_pi = adim_closure_terms(ref, b_tau, b_pi)
                c_tau, c_pi = adim_closure_terms(ref, c_tau, c_pi)
                
                self.tau["A"].append(a_tau)
                self.tau["B"].append(b_tau)
                self.tau["C"].append(c_tau)
                
                self.pi ["A"].append(a_pi)
                self.pi ["B"].append(b_pi)
                self.pi ["C"].append(c_pi)
                
                rms_les_temp_a, self.rms_dns_temp = rms(dfa, ref.df, Cp, comp_tau, comp_pi)
                rms_les_temp_b, _       = rms(dfb, ref.df, Cp, comp_tau, comp_pi)
                rms_les_temp_c, _       = rms(dfc, ref.df, Cp, comp_tau, comp_pi)
                
                self.rms_les_dim["A"].append(rms_les_temp_a)
                self.rms_les_dim["B"].append(rms_les_temp_b)
                self.rms_les_dim["C"].append(rms_les_temp_c)
                
                # debug purposes
                rms_dns = self.rms_dns_temp
                self.rms_les_a_temp = rms_dns
                # return rms_les_temp_a, rms_dns
                rms_les_temp_a, rms_dns_ = adim_second_order_stats(ref, rms_les_temp_a, rms_dns)
                rms_les_temp_b, rms_dns_ = adim_second_order_stats(ref, rms_les_temp_b, rms_dns)
                rms_les_temp_c, rms_dns  = adim_second_order_stats(ref, rms_les_temp_c, rms_dns)
                
                self.rms_les["A"].append(rms_les_temp_a)
                self.rms_les["B"].append(rms_les_temp_b)
                self.rms_les["C"].append(rms_les_temp_c)
                self.rms_dns = rms_dns
                
                
                self.is_empty["A"][idx_tau, idx_pi] = not(len(self.models[cond]) > 0) # are we empty here?
                # A 1 hot urms 0
