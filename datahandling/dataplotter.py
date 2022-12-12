from typing import Tuple, Union

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
    newdf["U'U'"] = (df["UU"] - df["U"] **
                     2).groupby("time").apply(lambda x: x.iloc[middle_canal])
    newdf["U"] = df["U"].groupby("time").apply(lambda x: x.iloc[middle_canal])
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
