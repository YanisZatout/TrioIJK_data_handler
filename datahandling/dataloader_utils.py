import numpy as np
from typing import Union
from pathlib import Path


def read_p_thermo(filepath: Union[Path, str]):
    with open(filepath, "r") as file:
        for line in file:
            if "p_thermo_init" in line:
                return np.float64(line.split()[1])
