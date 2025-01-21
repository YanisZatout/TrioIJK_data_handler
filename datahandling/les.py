from typing import Dict
import numpy as np
import pandas as pd
from .oscillation import ref_values


class LesData(object):
    def __init__(
        self,
        ref_df: pd.DataFrame,
        /,
        Cp: float = 0,
        h: float = 0,
        Tw: Dict[str, float] = {"hot": 293, "cold": 586},
    ) -> None:
        assert Cp != 0, "Cp must be provided as a floating point value"
        assert h != 0, "h must be provided as a floating point value"
        self.ref_df = ref_df
        self.Cp = self.cp = Cp
        self.h = h
        self.Tw = self.tw = Tw

    def load(self, ref_df=None, Cp=None):
        if not Cp:
            Cp = self.Cp
        (
            _,
            self.utau,
            self.thetatau,
            self.retau,
            self.rho_bulk,
            self.u_bulk,
            self.t_bulk,
            self.re_bulk,
            self.theta,
            self.utau,
            self.retau,
            self.nusselt,
            self.Cf,
            self.y,
            self.yplus,
            self.df,
            self.mu_bulk,
            self.phi_tau,
        ) = ref_values(self.ref_df, self.Cp, h=self.h, Tw=self.Tw)
        self.ubulk = self.u_bulk
        self.rebulk = self.re_bulk

        self.wall = {"T": self.Tw}
        self.sheer = self.tau = {
            "utau": self.utau,
            "retau": self.retau,
            "Cf": self.Cf,
            "thetatau": self.thetatau,
            "phitau": self.phi_tau,
        }
        self.msh = {"y": self.y, "h": self.h}
        self.T = self.thermal = {"Nu": self.nusselt, "nusselt": self.nusselt}
        self.bulk = {
            "rho_bulk": self.rho_bulk,
            "u_bulk": self.u_bulk,
            "t_bulk": self.t_bulk,
            "re_bulk": self.re_bulk,
            "mu_bulk": self.mu_bulk,
        }
        self.middle = len(self.y) // 2
        self.Nusselt = self.Nu = self.nusselt
        self.ny = self.y.shape[0]

    def __repr__(self):
        return (
            f"bulk quantities {self.bulk}\n"
            f"sheer quantities {self.sheer}\n"
            f"wall quantities {self.wall}\n"
        )
