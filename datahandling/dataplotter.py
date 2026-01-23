from typing import Union, Type
import re
import pandas as pd
from datahandling.dataloader import DataLoaderPandas
from .quantities_of_interest import (
    closure_terms,
    adim_closure_terms,
    rms,
    adim_second_order_stats,
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from natsort import natsorted
import os
from matplotlib.text import Annotation
from matplotlib.transforms import Affine2D


def matplotlib_latex_params(size=20) -> None:
    plt.rc("text", usetex=True)
    plt.rc("font", **{"family": "sans-serif", "sans-serif": "Helvetica", "size": size})
    plt.rcParams.update(
        {
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )


class PlotParams(object):
    def __init__(
        self,
        models_str,
        info_func=None,
        lines=["", "--", "-.", ":"],
        markers=None,
        hot_cold="cold",
        color_scaling_coeff=2,
    ):
        if not info_func:

            def info_func(model):
                mesh = model.split(" ")[0]
                model_tau = model.split()[1].split("-")[0]
                model_pi = model.split()[1].split("-")[1].split("_")[0]
                disc_qdm = model.split("_")[1]
                disc_mass = model.split("_")[2]
                return dict(
                    mesh=mesh,
                    models=model_tau + "-" + model_pi,
                    disc=disc_qdm + "-" + disc_mass,
                )

        self.info_func = info_func
        self.lines = lines
        self.n_models = len(models_str)

        models_str = [self.info_func(m) for m in models_str]

        # Unreadable part
        # At each unique different mesh, model and discretisation, we give
        # an integer value

        self.meshes = dict(
            zip(
                np.unique(np.array([m["mesh"] for m in models_str])).tolist(),
                np.arange(
                    len(np.unique(np.array([m["mesh"] for m in models_str])).tolist())
                ),
            )
        )
        self.models = dict(
            zip(
                np.unique(np.array([m["models"] for m in models_str])).tolist(),
                np.arange(
                    len(np.unique(np.array([m["models"] for m in models_str])).tolist())
                ),
            )
        )
        self.disc = dict(
            zip(
                np.unique(np.array([m["disc"] for m in models_str])).tolist(),
                np.arange(
                    len(np.unique(np.array([m["disc"] for m in models_str])).tolist())
                ),
            )
        )

        from matplotlib.lines import Line2D

        if not markers:
            self.markers = [
                m
                for m, func in Line2D.markers.items()
                if func != "nothing" and m not in Line2D.filled_markers
            ]
        else:
            self.markers = markers

        # Number of meshes
        self.n_lines = len(self.meshes) * color_scaling_coeff
        self.min_index_value = self.n_lines // 3
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
            marker=self.markers[disc],
        )


def make_legend_mean(
    ax: Union[plt.axes, np.typing.ArrayLike],
    elements: list[dict] = {
        "linestyle": {"DNS": "-", "LES": "--"},
        "marker": {"DNS": "", "LES": ""},
        "colors": {"hot": "lightcoral", "cold": "lightblue"},
    },
    quantities: list = None,
) -> None:
    linestyle_les = elements["linestyle"]["LES"]
    linestyle_dns = elements["linestyle"]["DNS"]

    marker_les = elements["marker"]["LES"]
    marker_dns = elements["marker"]["DNS"]

    color_cold = elements["color"]["cold"]
    color_hot = elements["color"]["hot"]

    for idx_quantity, quantity in enumerate(quantities):
        dummy_lines = [
            plt.Line2D(
                [0],
                [0],
                color="black",
                marker=marker_les,
                linestyle=linestyle_les,
                label="LES",
            ),
            plt.Line2D(
                [0],
                [0],
                color="black",
                marker=marker_dns,
                linestyle=linestyle_dns,
                label="DNS",
            ),
        ]
        dummy_lines = [
            *dummy_lines,
            plt.Line2D([0], [0], color=color_cold, linestyle="-", label="cold"),
            plt.Line2D([0], [0], color=color_hot, linestyle="-", label="hot"),
        ]
        ax[idx_quantity][-1].legend(
            handles=dummy_lines, loc="center left", bbox_to_anchor=(1, 0.5)
        )


def make_legend_rms(
    ax: Union[plt.axes, np.typing.ArrayLike],
    elements: list[dict] = {
        "linestyle": {"DNS": "-", "LES": "--", "struct": "none", "fonc": "none"},
        "marker": {"DNS": "", "LES": "", "struct": ".", "fonc": "*"},
        "colors": {"hot": "lightcoral", "cold": "lightblue"},
    },
    quantities: list = None,
    is_fonc=True,
    is_struct=True,
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
    color_hot = elements["color"]["hot"]

    for idx_quantity, quantity in enumerate(quantities):
        dummy_lines = [
            plt.Line2D(
                [0],
                [0],
                color="black",
                marker=marker_les,
                linestyle=linestyle_les,
                label="LES",
            ),
            plt.Line2D(
                [0],
                [0],
                color="black",
                marker=marker_dns,
                linestyle=linestyle_dns,
                label="DNS",
            ),
        ]
        if not "theta_rms" in quantity:
            if is_struct:
                dummy_lines = [
                    *dummy_lines,
                    plt.Line2D(
                        [0],
                        [0],
                        color="black",
                        marker=marker_struct,
                        linestyle=linestyle_struct,
                        label="struct",
                    ),
                ]
            if is_fonc:
                dummy_lines = [
                    *dummy_lines,
                    plt.Line2D(
                        [0],
                        [0],
                        color="black",
                        marker=marker_fonc,
                        linestyle=linestyle_fonc,
                        label="fonct",
                    ),
                ]
        dummy_lines = [
            *dummy_lines,
            plt.Line2D([0], [0], color=color_cold, linestyle="-", label="cold"),
            plt.Line2D([0], [0], color=color_hot, linestyle="-", label="hot"),
        ]
        ax[idx_quantity][-1].legend(
            handles=dummy_lines, loc="center left", bbox_to_anchor=(1, 0.5)
        )


class SimuMartin(object):
    def __init__(
        self, /, h, Cp: float = float(1155), dir_les: str = None, dir_dns: str = None
    ) -> None:
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

    def load(self, dir_les: str = None, dir_dns: str = None) -> None:
        if dir_les is None:
            dir_les = self.dir_les
        if dir_dns is None:
            dir_dns = self.dir_dns
        directory = dir_les
        self.afiles = afiles = natsorted(os.listdir(os.path.join(directory, "AAA")))
        self.bfiles = bfiles = natsorted(os.listdir(os.path.join(directory, "BAB")))
        self.cfiles = cfiles = natsorted(os.listdir(os.path.join(directory, "CAC")))

        a_inter_b = list(set(afiles).intersection(bfiles))
        self.o = list(set(a_inter_b).intersection(cfiles))

        self.models = np.array(self.o)
        self.df = {"A": [], "B": [], "C": []}

        for file in self.o:
            path = os.path.join(directory, "AAA", file)
            self.df["A"].append(DataLoaderPandas(path, "statistiques").load_last())
            # print(len(self.models))
        for file in self.o:
            path = os.path.join(directory, "BAB", file)
            self.df["B"].append(DataLoaderPandas(path, "statistiques").load_last())
        for file in self.o:
            path = os.path.join(directory, "CAC", file)
            self.df["C"].append(DataLoaderPandas(path, "statistiques").load_last())
        self.df_dns = DataLoaderPandas(
            directory=dir_dns, type_stat="statistiques"
        ).load_last()

    def rms(self) -> None:
        from .quantities_of_interest import compute_rms_quantities

        self.arms, self.rms_dns = compute_rms_quantities(
            self.dfa, self.df_dns, Cp=self.Cp
        )

    def discriminate_compressibility(self) -> None:
        self.tau_comp = []
        self.pi_comp = []

        compressible_regex = "(((A|G|B|S))c[0-9]|_comp)"
        for amod in self.models:
            aword = amod.split("-")[0]
            matches = re.findall(compressible_regex, aword)
            if len(matches) > 0:
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
        self.pi_comp = np.array(self.pi_comp)

    def compute_yplus(self, ref: Type["RefData"]) -> None:
        self.amiddle = len(self.df["A"][0].coordonnee_K) // 2
        self.bmiddle = len(self.df["B"][0].coordonnee_K) // 2
        self.cmiddle = len(self.df["C"][0].coordonnee_K) // 2
        ayplus_cold = (
            self.df["A"][0].coordonnee_K.values[: self.amiddle]
            * ref.utau["cold"]
            * ref.df["RHO"].iloc[0]
            / ref.df["MU"].iloc[0]
        )
        byplus_cold = (
            self.df["B"][0].coordonnee_K.values[: self.bmiddle]
            * ref.utau["cold"]
            * ref.df["RHO"].iloc[0]
            / ref.df["MU"].iloc[0]
        )
        cyplus_cold = (
            self.df["C"][0].coordonnee_K.values[: self.cmiddle]
            * ref.utau["cold"]
            * ref.df["RHO"].iloc[0]
            / ref.df["MU"].iloc[0]
        )

        ayplus_hot = (
            (2 * self.h - self.df["A"][0].coordonnee_K.values[self.amiddle :][::-1])
            * ref.utau["hot"]
            * ref.df["RHO"].iloc[-1]
            / ref.df["MU"].iloc[-1]
        )
        byplus_hot = (
            (2 * self.h - self.df["B"][0].coordonnee_K.values[self.bmiddle :][::-1])
            * ref.utau["hot"]
            * ref.df["RHO"].iloc[-1]
            / ref.df["MU"].iloc[-1]
        )
        cyplus_hot = (
            (2 * self.h - self.df["C"][0].coordonnee_K.values[self.cmiddle :][::-1])
            * ref.utau["hot"]
            * ref.df["RHO"].iloc[-1]
            / ref.df["MU"].iloc[-1]
        )

        self.y = {
            "A": self.df["A"][0].coordonnee_K.values,
            "B": self.df["B"][0].coordonnee_K.values,
            "C": self.df["C"][0].coordonnee_K.values,
        }
        self.yplus = {
            "hot": {"A": ayplus_hot, "B": byplus_hot, "C": cyplus_hot},
            "cold": {"A": ayplus_cold, "B": byplus_cold, "C": cyplus_cold},
        }

    def compute_tau_pi(self, ref: Type["RefData"]) -> None:
        Cp = self.Cp
        self.pi = {"A": [], "B": [], "C": []}
        self.tau = {"A": [], "B": [], "C": []}
        self.rms_les = {"A": [], "B": [], "C": []}
        self.rms_les_dim = {"A": [], "B": [], "C": []}
        self.ordered_models = []
        self.is_empty = {
            "A": np.zeros((2, 2), dtype=bool),
            "B": np.zeros((2, 2), dtype=bool),
            "C": np.zeros((2, 2), dtype=bool),
        }

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

                self.pi["A"].append(a_pi)
                self.pi["B"].append(b_pi)
                self.pi["C"].append(c_pi)

                rms_les_temp_a, self.rms_dns_temp = rms(
                    dfa, ref.df, Cp, comp_tau, comp_pi
                )
                rms_les_temp_b, _ = rms(dfb, ref.df, Cp, comp_tau, comp_pi)
                rms_les_temp_c, _ = rms(dfc, ref.df, Cp, comp_tau, comp_pi)

                self.rms_les_dim["A"].append(rms_les_temp_a)
                self.rms_les_dim["B"].append(rms_les_temp_b)
                self.rms_les_dim["C"].append(rms_les_temp_c)

                # debug purposes
                rms_dns = self.rms_dns_temp
                self.rms_les_a_temp = rms_dns
                # return rms_les_temp_a, rms_dns
                rms_les_temp_a, rms_dns_ = adim_second_order_stats(
                    ref, rms_les_temp_a, rms_dns
                )
                rms_les_temp_b, rms_dns_ = adim_second_order_stats(
                    ref, rms_les_temp_b, rms_dns
                )
                rms_les_temp_c, rms_dns = adim_second_order_stats(
                    ref, rms_les_temp_c, rms_dns
                )

                self.rms_les["A"].append(rms_les_temp_a)
                self.rms_les["B"].append(rms_les_temp_b)
                self.rms_les["C"].append(rms_les_temp_c)
                self.rms_dns = rms_dns

                self.is_empty["A"][idx_tau, idx_pi] = not (
                    len(self.models[cond]) > 0
                )  # are we empty here?
                # A 1 hot urms 0


class LineAnnotation(Annotation):
    """A sloped annotation to *line* at position *x* with *text*
    Optionally an arrow pointing from the text to the graph at *x* can be drawn.
    Usage
    -----
    fig, ax = subplots()
    x = linspace(0, 2*pi)
    line, = ax.plot(x, sin(x))
    ax.add_artist(LineAnnotation("text", line, 1.5))
    """

    def __init__(
        self, text, line, x, xytext=(0, 5), textcoords="offset points", **kwargs
    ):
        """Annotate the point at *x* of the graph *line* with text *text*.

        By default, the text is displayed with the same rotation as the slope of the
        graph at a relative position *xytext* above it (perpendicularly above).

        An arrow pointing from the text to the annotated point *xy* can
        be added by defining *arrowprops*.

        Parameters
        ----------
        text : str
            The text of the annotation.
        line : Line2D
            Matplotlib line object to annotate
        x : float
            The point *x* to annotate. y is calculated from the points on the line.
        xytext : (float, float), default: (0, 5)
            The position *(x, y)* relative to the point *x* on the *line* to place the
            text at. The coordinate system is determined by *textcoords*.
        **kwargs
            Additional keyword arguments are passed on to `Annotation`.

        See also
        --------
        `Annotation`
        `line_annotate`
        """
        assert textcoords.startswith(
            "offset "
        ), "*textcoords* must be 'offset points' or 'offset pixels'"

        self.line = line
        self.xytext = xytext

        # Determine points of line immediately to the left and right of x
        xs, ys = line.get_data()

        def neighbours(x, xs, ys, try_invert=True):
            (inds,) = np.where((xs <= x)[:-1] & (xs > x)[1:])
            if len(inds) == 0:
                assert try_invert, "line must cross x"
                return neighbours(x, xs[::-1], ys[::-1], try_invert=False)

            i = inds[0]
            return np.asarray([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])])

        self.neighbours = n1, n2 = neighbours(x, xs, ys)

        # Calculate y by interpolating neighbouring points
        y = n1[1] + ((x - n1[0]) * (n2[1] - n1[1]) / (n2[0] - n1[0]))

        kwargs = {
            "horizontalalignment": "center",
            "rotation_mode": "anchor",
            **kwargs,
        }
        super().__init__(text, (x, y), xytext=xytext, textcoords=textcoords, **kwargs)

    def get_rotation(self):
        """Determines angle of the slope of the neighbours in display coordinate system"""
        transData = self.line.get_transform()
        dx, dy = np.diff(transData.transform(self.neighbours), axis=0).squeeze()
        return np.rad2deg(np.arctan2(dy, dx))

    def update_positions(self, renderer):
        """Updates relative position of annotation text
        Note
        ----
        Called during annotation `draw` call
        """
        xytext = Affine2D().rotate_deg(self.get_rotation()).transform(self.xytext)
        self.set_position(xytext)
        super().update_positions(renderer)


def line_annotate(text, line, x, *args, **kwargs):
    """Add a sloped annotation to *line* at position *x* with *text*

    Optionally an arrow pointing from the text to the graph at *x* can be drawn.

    Usage
    -----
    x = linspace(0, 2*pi)
    line, = ax.plot(x, sin(x))
    line_annotate("sin(x)", line, 1.5)

    See also
    --------
    `LineAnnotation`
    `plt.annotate`
    """
    ax = line.axes
    a = LineAnnotation(text, line, x, *args, **kwargs)
    if "clip_on" in kwargs:
        a.set_clip_path(ax.patch)
    ax.add_artist(a)
    return a
