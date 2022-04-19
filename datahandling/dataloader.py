#TODO: add key2num from numbers to key
from typing import Tuple, Union
from dataclasses import dataclass
import re
import numpy as np
import sys, os


@dataclass
class DataLoader:
    """
    DataLoader class, loads data and lets you do stuff with it
    """
    directory: str = "rep22"
    type_stat: str = "statistiques"
    columns: Union[str, int] = None

    separator: str = "\s"
    type_file: str = ".txt"

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
        lines=[]
        with open(filename, 'r') as input:
            for line in input:
                if not '#' in line:
                    break
                lines = [*lines, line]
        lines = lines[1:]

        for i in range(len(lines)):
            lines[i] = re.sub(r'# colonne [0-9]+ : ', '', lines[i])
            lines[i] = lines[i].replace("\n", "")
        self.header = np.array(lines)
        # key2num_dict: for letter key, we have an int
        # Is handy to recover index of variable type.
        # num_2_key is essentially the same in reverse
        self.key2num_dict = {h: i for i, h in enumerate(self.header)}
        self.num2key_dict = {i: h for i, h in enumerate(self.header)}

    def parse_stats_directory(self, directory: str = None) -> Tuple[str,np.ndarray]:
        """
        Gives names of files needed, as well as time steps for respective stats
        ----------
        Parameters:
        directory: str 
            Which directory to search in
        type_stat: str 
            Type of statistics to search for, for example "moyenne_spatiale" for instance
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
        for filename in os.listdir(self.directory):
            f = os.path.join(self.directory, filename)
            if os.path.isfile(f) and self.type_stat in filename:
                file_path = [*file_path, f]
        file_path.sort()
        for fp in file_path:
            fp = fp.replace(os.path.join(self.directory, f"{self.type_stat}_"), "")
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
            self.columns_index = self.column_handler(self.columns)

        cols = None
        if self.columns:
            cols = self.columns_index
        for file in self.file_path:
            data = [*data, np.loadtxt(file, usecols=cols)]
        data = np.array(data)
        self.data = data
        self.shape = self.data.shape

    def key2num(self, variable: Union[str, int, np.integer]) -> int:
        """
        key2num function, handles types for a list of strings or integers for only \textbf{ONE} variable
        Parameters:
        ----------
        variable: Union[str, int, np.integer]
            Variables of interest, whether it be an int or a name, like "T" for temperature
        ----------
        Returs:
        None
        index: int
            Index of variable of interest
        """
        if isinstance(variable, str):
            return self.key2num_dict[variable]
        return variable
    
    def num2key(self, variable: Union[str, int, np.integer]) -> int:
        """
        key2num function, handles types for a list of strings or integers for only \textbf{ONE} variable
        Parameters:
        ----------
        variable: Union[str, int, np.integer]
            Variables of interest, whether it be an int or a name, like "T" for temperature
        ----------
        Returs:
        None
        index: int
            Index of variable of interest
        """
        if isinstance(variable, str):
            return self.num2key_dict[variable]
        return variable

    def column_handler(self, variable: Union[str, list, int, np.ndarray])-> Tuple[int]:
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
            return tuple(self.key2num(var) for var in variable)

        if isinstance(variable, str): 
            return (self.key2num(variable),)
        return (variable)

    def __getitem__(self, column: Union[str, list, int, np.ndarray]) -> Union[np.ndarray, np.float64]:
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
        if isinstance(column, str):
            column = self.column_handler(column)
            return self.data[column]
        return self.data[column]