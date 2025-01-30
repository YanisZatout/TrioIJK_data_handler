from typing import Union, Type
import re
import pandas as pd
from datahandling.dataloader import DataLoaderPandas
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


