from typing import List, Tuple, Union
import re
import numpy as np
import os
from numpy import typing as npt
import pandas as pd
from scipy.integrate import simpson
from glob import glob


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
        type_file: str = ".txt",
    ):
        """
        The initialisation function for the class. For a specific directory
        parses the statistics text files. Statistics can be mean over space,
        or mean over space and time
        """
        self.directory = directory
        assert (
            type_stat is not None
        ), "You need to provide a type of statistics, beeing either 'statistiques' or 'moyenne_spatiale'"
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
        with open(filename, "r") as input:
            for line in input:
                if "#" not in line:
                    break
                lines = [*lines, line]
        if self.type_stat == "moyenne_spatiale":
            lines = lines[1:]
        elif self.type_stat == "statistiques":
            lines = lines[2:]
        else:
            raise ValueError("Type of statistics " f"{self.type_stat} not recognized")

        for i in range(len(lines)):
            lines[i] = re.sub(r"# colonne [0-9]+ : ", "", lines[i])
            lines[i] = lines[i].replace("\n", "")
        self.header = np.array(lines)
        # key2num_dict: for letter key, we have an int
        # Is handy to recover index of variable type.
        # num_2_key is essentially the same in reverse
        self.key2num_dict = {h: i for i, h in enumerate(self.header)}
        self.num2key_dict = {i: h for i, h in enumerate(self.header)}

    @staticmethod
    def creation_date(path_to_file: str):
        """
        Try to get the date that a file was created, falling back to when it
        was last modified if that isn't possible.
        See http://stackoverflow.com/a/39501288/1709587 for explanation.
        """

        import platform

        if platform.system() == "Windows":
            return os.path.getctime(path_to_file)
        else:
            stat = os.stat(path_to_file)
            try:
                return stat.st_birthtime
            except AttributeError:
                # We're probably on Linux. No easy way to get creation dates here,
                # so we'll settle for when its content was last modified.
                return stat.st_mtime

    def parse_stats_directory(
        self, directory: Union[str, None] = None, last_24h: bool = False
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
        if isinstance(directory, (list, tuple)) and len(directory) > 1:
            # In the case the user provides multiple directories to explore:
            from glob import iglob

            def multi_glob(patterns):
                for pattern in patterns:
                    yield from iglob(pattern)

            directory_and_pattern = [
                os.path.join(d, f"{self.type_stat}_*") for d in directory
            ]
            list_of_files = list(multi_glob(directory_and_pattern))
        else:
            # In the case the user gave one directory
            from glob import glob

            list_of_files = glob(os.path.join(self.directory, f"{self.type_stat}_*"))

        time = []
        file_path = []
        # Match stat files
        file_path = sorted(
            list_of_files,
            key=lambda x: float(x.split(self.type_stat + "_")[1].split(".txt")[0]),
        )
        # If we only want the last 24 hours files
        import time

        if last_24h:
            last_24h_files = []
            """ import datetime as dt """
            """ today = dt.datetime.now().date() """
            for file in file_path:
                if abs(self.creation_date(file) - time.time()) <= 24 * 60 * 60:
                    last_24h_files.append(file)
            file_path = last_24h_files

        # Remove duplicates
        unique_files = set()
        duplicates = []
        final_files = []
        for file_paths in file_path:
            file_name = os.path.basename(file_paths)
            floating_point_number = file_name.split(self.type_stat + "_")[1].split(
                ".txt"
            )[0]
            if floating_point_number in unique_files:
                duplicates.append(file_paths)
            else:
                unique_files.add(floating_point_number)
                final_files.append(
                    f"{os.path.dirname(file_paths)}/{self.type_stat}_{floating_point_number}.txt"
                )

        file_path = final_files

        # pattern = f"[^/]+/{self.type_stat}_(\\d+\\.\\d+)\\.txt"
        # time = [float(match) for fp in file_path for match in re.findall(pattern, fp)]
        time = np.array(
            [
                float(x.split(self.type_stat + "_")[1].split(".txt")[0])
                for x in file_path
            ]
        )
        return file_path, time

    def load_first(self) -> pd.DataFrame:
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
        last_time = f"{self.type_stat}_{self.time[0]}"
        self.read_header(self.file_path[0])

        file = self.file_path[0]
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

    def load_last(self, path=None) -> pd.DataFrame:
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
        # TODO: Change this mess as to make it clearer.
        # For now it loads a file path is given
        # Or a broad directory path is given
        # But it's still a mess
        pattern = re.compile(f".*/{self.type_stat}_[0-9]\.[0-9]+.txt")
        specific_file = bool(pattern.match(str(path)))
        if not specific_file:
            data = []
            self.file_path, self.time = self.parse_stats_directory()
            last_time = f"{self.type_stat}_{self.time[-1]}"
            self.read_header(self.file_path[0])
            file = self.file_path[-1]
        else:
            # Directly give filepath
            file = path
            self.read_header(file)

        data = np.loadtxt(file, dtype=np.float64)
        data = np.array(data)
        self.shape = data.shape
        self.space = data[:, 0]
        return pd.DataFrame(
            data=data,
            index=self.space,
            columns=self.header,
            dtype=np.float64,
        )

    def load_data(
        self, *, verbose=False, parallel=False, pbar=False
    ) -> Union[List, pd.DataFrame, None]:
        """
        loads data into data variable of class
        Parameters:
        ----------
        None
        ----------
        Returs:
        None
        """
        if verbose and pbar:
            import warnings

            warnings.warn(
                "Using verbose and pbar flags together shows ugly output when used"
            )

        self.file_path, self.time = self.parse_stats_directory()
        self.read_header(self.file_path[0])

        if self.columns is not None:
            self.columns_index = self.column_handler_key2num(self.columns)

        cols = None
        if self.columns:
            cols = self.columns_index
        if pbar:
            from tqdm import tqdm

            progress_bar = tqdm(
                zip(self.file_path, self.time), total=len(self.file_path)
            )
        else:
            progress_bar = zip(self.file_path, self.time)
        if parallel:
            try:
                import dask.array as da
                from dask import delayed, compute
            except ImportError:
                return None
            data = []
            for file, time in progress_bar:
                if verbose:
                    print(file, time)
                dd = delayed(np.loadtxt)(file, usecols=cols)
                data.append(dd)
            tuple_of_data = compute(*data)
            data = self.data = np.concatenate(tuple_of_data)
            self.space = np.loadtxt(self.file_path[0], usecols=0, dtype=np.float64)
            indexes = pd.MultiIndex.from_product(
                [self.time, self.space], names=("time", "k")
            )
            columns = self.header
            data = pd.DataFrame(data, columns=columns, index=indexes)
            return data
        else:
            data = []
            for file, time in progress_bar:
                if verbose:
                    print(file, time)
                dd = np.array(np.loadtxt(file, usecols=cols))
                data = [*data, dd]
            self.data = np.array(data)
            data = np.concatenate(data)
            self.space = np.loadtxt(self.file_path[0], usecols=0)
            indexes = pd.MultiIndex.from_product(
                [self.time, self.space], names=("time", "k")
            )
            columns = self.header
            data = pd.DataFrame(data, columns=columns, index=indexes, dtype=np.float64)
            return data

    def load_last_24h(self, *, verbose=False) -> Union[List, pd.DataFrame]:
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
        self.file_path, self.time = self.parse_stats_directory(last_24h=True)
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
            [self.time, self.space], names=("time", "k")
        )
        columns = self.header
        data = pd.DataFrame(data, columns=columns, index=indexes)
        return data

    def column_handler_key2num(
        self, variable: Union[str, list, int, npt.ArrayLike]
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
        return variable

    def integrate_statistics(self) -> None:
        self.statistics = [
            (1 / self.time[-1] - self.time[0])
            * simpson(self.data[..., num], x=self.time, axis=0)
            for num in range(self.data.shape[-1])
        ]

    @staticmethod
    def mean_over_n_times(df: pd.DataFrame, n: int = 0):
        assert (
            n > 0
        ), "You need to specify a positive number of timesteps to do the mean over"
        return df.groupby("k").apply(lambda x: x.iloc[-n]).groupby(level=0).mean()
