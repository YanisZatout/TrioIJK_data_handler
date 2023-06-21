from typing import List, Tuple, Union
import re
import numpy as np
import os
from numpy import typing as npt
import pandas as pd
from scipy.integrate import simpson
import re
from glob import glob


class DataLoader:
    """
    DataLoader class, loads data and lets you do stuff with it
    """

    def __init__(self,
                 directory: str = "rep22",
                 type_stat: str = "statistiques",
                 columns: Union[str, int, None] = None,
                 separator: str = "\s",
                 type_file: str = ".txt"
                 ):
        self.directory: str = directory
        self.type_stat: str = type_stat
        self.columns: Union[str, int, None] = columns
        self.separator: str = separator
        self.type_file: str = type_file

    def read_header(self, filename: str) -> None:
        """
        Reads the header of a .txt file in a TrioIJK simulation
        Parameters:
        filename: str
            Name of file, or file path
        ----------
        Returns
        header: np.array
            Name of columns of given .txt file
        """
        lines = []
        with open(filename, 'r') as input:
            for line in input:
                if '#' not in line:
                    break
                lines = [*lines, line]
        if self.type_stat == "moyenne_spatiale":
            lines = lines[1:]
        elif self.type_stat == "statistiques":
            lines = lines[2:]
        else:
            raise ValueError("Type of statistics "
                             f"{self.type_stat} not recognized")

        for i in range(len(lines)):
            lines[i] = re.sub(r'# colonne [0-9]+ : ', '', lines[i])
            lines[i] = lines[i].replace("\n", "")
        self.header = np.array(lines)
        # key2num_dict: for letter key, we have an int
        # Is handy to recover index of variable type.
        # num_2_key is essentially the same in reverse
        self.key2num_dict = {h: i for i, h in enumerate(self.header)}
        self.num2key_dict = {i: h for i, h in enumerate(self.header)}

    def parse_stats_directory(self,
                              directory: Union[str, None] = None
                              ) -> Tuple[List[str], npt.ArrayLike]:
        """
        Gives names of files needed, as well as time steps for respective stats
        ----------
        Parameters:
        directory: str
            Which directory to search in
        type_stat: str
            Type of statistics to search for, for example "moyenne_spatiale"
            for instance
        type_file: str
            Extention of file, example: ".txt"
        ----------
        Returns:
        file_path: str
            The path of files
        time: np.array
            Numpy array of steps of time
        """
        if directory is None:
            directory = self.directory
        time = []
        file_path = []
        from os.path import join
        file_path = glob(join(self.directory, self.type_stat+"*.txt"))

        for fp in file_path:
            fp = fp.replace(os.path.join(self.directory.replace(
                "*", ""), f"{self.type_stat}_"), "")
            fp = fp.replace(".txt", "")
            time = [*time, float(fp)]
        return file_path, np.array(time)

    def load_data(self) -> None:
        """
        loads data into data variable of class
        Parameters:
        ----------
        None
        ----------
        Returs:
        None
        """
        data = []
        self.file_path, self.time = self.parse_stats_directory()
        self.read_header(self.file_path[0])

        if self.columns is not None:
            self.columns_index = self.column_handler_key2num(self.columns)

        cols = None
        if self.columns:
            cols = self.columns_index
        for file in self.file_path:
            data = [*data, np.loadtxt(file, usecols=cols)]
        data = np.array(data)
        self.data = data
        self.shape = self.data.shape
        self.space = self.data[0, :, 0]

    def load_last(self) -> None:
        """
        Loads data into class data variable for the last time
        step for the given statistics
        Parameters:
        ----------
        None
        ----------
        Returs:
        None
        """

        data = []
        self.file_path, self.time = self.parse_stats_directory()
        last_time = f"{self.type_stat}_{self.time[-1]}"
        self.read_header(self.file_path[0])

        if self.columns is not None:
            self.columns_index = self.column_handler_key2num(self.columns)

        cols = None
        if self.columns:
            cols = self.columns_index
        file = self.file_path[-1]
        data = np.loadtxt(file, usecols=cols)
        data = np.array(data)
        self.data = data
        self.shape = self.data.shape
        self.space = self.data[:, 0]

    def load_last_df(self) -> None:
        """
        Loads data into class data variable for the last time
        step for the given statistics and puts it in a pandas dataframe
        Parameters:
        ----------
        None
        ----------
        Returs:
        None
        """
        data = []
        self.file_path, self.time = self.parse_stats_directory()
        last_time = f"{self.type_stat}_{self.time[-1]}"
        self.read_header(self.file_path[0])

        if self.columns is not None:
            self.columns_index = self.column_handler_key2num(self.columns)

        cols = None
        if self.columns:
            cols = self.columns_index
        file = self.file_path[-1]
        data = np.loadtxt(file, usecols=cols)
        data = np.array(data)
        self.data = data
        self.shape = self.data.shape
        self.space = self.data[:, 0]
        self.data = pd.DataFrame(
            data=self.data,
            index=self.space,
            columns=self.header,
            dtype=np.float32
        )
        # Hacky way to force it to use pandas in this case
        self.__getitem__ = self.data.__getitem__
        self.loc = self.data.loc
        self.iloc = self.data.iloc

    def key2num(self, variable: Union[str, int, np.integer]) -> int:
        """
        key2num function, handles types for a list of strings or integers for
        only \textbf{ONE} variable
        Parameters:
        ----------
        variable: Union[str, int, np.integer]
            Variables of interest, whether it be an int or a name, like "T" for
            temperature
        ----------
        Returs:
        None
        index: int
            Index of variable of interest
        """
        if isinstance(variable, str):
            return self.key2num_dict[variable]
        return variable

    def num2key(self, variable: Union[str, int, np.integer]) -> str:
        """
        num2key function, handles types for a list of strings or integers for
        only \textbf{ONE} variable
        Parameters:
        ----------
        variable: Union[int, np.integer]
            Variables of interest, wheter it be an int or a string
        ----------
        Returs:
        None
        index: int
            Index of variable of interest
        """
        if isinstance(variable, (int, np.integer)):
            return self.num2key_dict[variable]
        return variable

    def column_handler_key2num(
        self,
            variable: Union[str, list,
                            int, npt.ArrayLike]
    ) -> Tuple[int]:
        """
        column handler, handles which columns are to be saved. If you only study the
        temperature for instance, there's no need for loading other variables. This function
        returns the number of the column of interest
        Parameters:
        ----------
        variable: Union[str, list, int, np.ndarray]
            Variables of interest. Can be one or multiple of them.
        ----------
        index: Tuple[int]
            The index(es) if the variables of interest inside the file for them to be loaded properly
        """
        if isinstance(variable, list):
            return tuple(self.key2num(var) for var in variable)

        if isinstance(variable, str):
            return (self.key2num(variable),)
        return (variable)

    def column_handler_num2key(self, variable: Union[str, list, int, npt.ArrayLike]) -> Tuple:
        """
        column handler, handles which columns are to be saved. If you only study the
        temperature for instance, there's no need for loading other variables. This function
        returns the number of the column of interest
        Parameters:
        ----------
        variable: Union[str, list, int, np.ndarray]
            Variables of interest. Can be one or multiple of them.
        ----------
        Returs:
        index: Tuple[int]
            The index(es) if the variables of interest inside the file for them to be loaded properly
        """
        if isinstance(variable, list):
            return tuple(self.num2key(var) for var in variable)

        if isinstance(variable, str):
            return (self.num2key(variable),)
        return (variable)

    def __getitem__(
        self,
        column: Union[str, list, int, npt.ArrayLike]
    ) -> Union[npt.ArrayLike]:
        """
        Gets the element of interest whether it be a string, a list, or a numpy array.
        Parameters:
        ----------
        column: Union[str, list, int, np.ndarray]
            Columns of interest, whether it be one, or multiple, with a name or a number
        ----------
        Returs:
        dataofinterest: Union[np.ndarray, np.float64]
            The data you're interested in, an array or a floating point value
        """
        # TODO: Handle slices

        if isinstance(column, str):
            column = self.column_handler_key2num(column)
            return self.data[..., column].squeeze()
        return self.data[column]


class DataLoaderPandas:
    """
    Class that reads a specific directory to get statistics related to
    a TrioIJK simulations
    """

    def __init__(
        self,
            directory: str = "./",
            type_stat: Union[str, None] = None,
            columns: Union[str, int, None] = None,
            separator: str = "\s",
            type_file: str = ".txt"
    ):
        """
        The initialisation function for the class. For a specific directory
        parses the statistics text files. Statistics can be mean over space,
        or mean over space and time
        """
        self.directory = directory
        assert type_stat is not None, "You need to provide a type of statistics, beeing either 'statistiques' or 'moyenne_spatiale'"
        self.type_stat = type_stat
        self.columns = columns
        self.separator = separator
        self.type_file = type_file

    def read_header(self, filename: str) -> None:
        """
        Reads the header of a .txt file in a TrioIJK simulation
        Parameters:
        filename: str
            Name of file, or file path
        ----------
        Returns
        header: np.array
            Name of columns of given .txt file
        """
        lines = []
        with open(filename, 'r') as input:
            for line in input:
                if '#' not in line:
                    break
                lines = [*lines, line]
        if self.type_stat == "moyenne_spatiale":
            lines = lines[1:]
        elif self.type_stat == "statistiques":
            lines = lines[2:]
        else:
            raise ValueError("Type of statistics "
                             f"{self.type_stat} not recognized")

        for i in range(len(lines)):
            lines[i] = re.sub(r'# colonne [0-9]+ : ', '', lines[i])
            lines[i] = lines[i].replace("\n", "")
        self.header = np.array(lines)
        # key2num_dict: for letter key, we have an int
        # Is handy to recover index of variable type.
        # num_2_key is essentially the same in reverse
        self.key2num_dict = {h: i for i, h in enumerate(self.header)}
        self.num2key_dict = {i: h for i, h in enumerate(self.header)}

    def parse_stats_directory(
        self,
            directory: Union[str, None] = None
    ) -> Tuple[List[str], npt.ArrayLike]:
        """
        Gives names of files needed, as well as time steps for respective stats
        ----------
        Parameters:
        directory: str
            Which directory to search in
        type_stat: str
            Type of statistics to search for, for example "moyenne_spatiale"
            for instance
        type_file: str
            Extention of file, example: ".txt"
        ----------
        Returns:
        file_path: str
            The path of files
        time: np.array
            Numpy array of steps of time
        """
        if directory is None:
            directory = self.directory
        time = []
        file_path = []
        from glob import glob
        file_path = sorted(
            glob(os.path.join(self.directory, f"{self.type_stat}_*")),
            key=lambda x: float(
                    x.split(self.type_stat+"_") [1].split(".txt")[0]
                )
        )
        pattern = f'[^/]+/{self.type_stat}_(\\d+\\.\\d+)\\.txt'
        time = [float(match) for fp in file_path for match in re.findall(pattern, fp)] 
        return file_path, np.array(time)

    def load_last(self) -> pd.DataFrame:
        """
        Loads data into class data variable for the last time
        step for the given statistics
        Parameters:
        ----------
        None
        ----------
        Returs:
        None
        """

        data = []
        self.file_path, self.time = self.parse_stats_directory()
        last_time = f"{self.type_stat}_{self.time[-1]}"
        self.read_header(self.file_path[0])

        file = self.file_path[-1]
        data = np.loadtxt(file)
        data = np.array(data)
        self.shape = data.shape
        self.space = data[:, 0]
        return pd.DataFrame(
            data=data,
            index=self.space,
            columns=self.header,
            dtype=np.float32,
        )

    def load_data(self, *, verbose=False) -> Union[List, pd.DataFrame]:
        """
        loads data into data variable of class
        Parameters:
        ----------
        None
        ----------
        Returs:
        None
        """
        data = []
        self.file_path, self.time = self.parse_stats_directory()
        self.read_header(self.file_path[0])

        if self.columns is not None:
            self.columns_index = self.column_handler_key2num(self.columns)

        cols = None
        if self.columns:
            cols = self.columns_index
        for file, time in zip(self.file_path, self.time):
            if verbose:
                print(file, time)
            dd = np.array(np.loadtxt(file, usecols=cols))
            data = [*data, dd]
        self.data = np.array(data)
        data = np.concatenate(data)
        self.space = np.loadtxt(self.file_path[0], usecols=0)
        indexes = pd.MultiIndex.from_product(
            [self.time, self.space],
            names=("time", "k")
        )
        columns = self.header
        data = pd.DataFrame(data, columns=columns, index=indexes)
        return data

    def column_handler_key2num(
        self,
            variable: Union[str, list,
                            int, npt.ArrayLike]
    ) -> Tuple[int]:
        """
        column handler, handles which columns are to be saved. If you only study the
        temperature for instance, there's no need for loading other variables. This function
        returns the number of the column of interest
        Parameters:
        ----------
        variable: Union[str, list, int, np.ndarray]
            Variables of interest. Can be one or multiple of them.
        ----------
        index: Tuple[int]
            The index(es) if the variables of interest inside the file for them to be loaded properly
        """
        if isinstance(variable, list):
            return tuple(self.key2num(var) for var in variable)

        if isinstance(variable, str):
            return (self.key2num(variable),)
        return (variable)

    def integrate_statistics(self) -> None:
        self.statistics = [(1/self.time[-1] - self.time[0]) * simpson(
            self.data[..., num], x=self.time, axis=0
        )
            for num in range(self.data.shape[-1])]

    @staticmethod
    def mean_over_n_times(df: pd.DataFrame, n: int = 0):
        assert n > 0, "You need to specify a positive number of timesteps to do the mean over"
        return df.groupby("k").apply(lambda x: x.iloc[-n]).groupby(level=0).mean()
