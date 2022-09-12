from typing import List, Tuple
import numpy as np
import re
from torch.utils.data import Dataset


def parse_n_coordinates(filename: str = "dns_180/dns_lata_1.sauv.lata") \
        -> Tuple[List[int], List[int]]:
    """
    Gets the number of points for the vector/scalar values. Reads from a file
    Parameters
    ----------
    filename: str
        Name of the file where information about the simulation is stored
    Returns
    ----------
    n_vertex: List[int, int, int]
        Number of vertices in the x, y, and z directions
    n: List[int, int, int]
        Number of scalar points in the x, y, and z directions
    """
    with open(filename, "r") as f:
        data = f.read().replace('\n', '')
    n_vertex = re.findall(r"\b\d{3,4}\b", data)
    n_vertex = [int(nv) for nv in n_vertex]
    n = [i - 1 for i in n_vertex]
    return n_vertex, n


class DNSHandler(object):
    def __init__(self, filename: str = "dns_180/dns_lata_0.sauv.lata"):
        pass

