from typing import Tuple, Union
from datahandling.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt


def matplotlib_latex_params() -> None:
    plt.rc(
        'text', usetex=True
    )
    plt.rc(
        'font',
        **
        {
            "family": "sans-serif",
            "sans-serif": "Helvetica",
            "size": 18
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
