from typing import Union, List
import numpy as np
import numpy.typing as npt
import pyvista as pv
import torch
from matplotlib import pyplot as plt


def flatten_fortran_w_clone(t: torch.Tensor) -> torch.Tensor:
    nt = t.clone()
    nt = nt.set_(nt.storage(), nt.storage_offset(), nt.size(),
                 tuple(reversed(nt.stride())))
    return nt.flatten()


class Plotter(object):
    """
    Plotting helper
    Only takes one array made of scalars
    """

    def __init__(self,
                 values: Union[npt.ArrayLike, torch.Tensor],
                 coord: List[npt.ArrayLike],
                 ) -> None:
        """
            Initializes the grid of the object we want to look at
            Parameters:
            ----------
            values: Union[npt.ArrayLike]
                Quantity of interest. Has to be a scalar
            coord: List[np.ndarray, np.ndarray, np.ndarray]
                Coordinates of points in the x, y and z directions
            Returns:
            ----------
            grid: pv.UniformGrid
                The grid of values of interest
            """
        grid = pv.UniformGrid()
        grid.dimensions = np.array(values.shape) + 1
        # Origin point coordinates is the first point of the coordinates
        grid.origin = tuple(coord_i_direction[0]
                            for coord_i_direction in coord)

        # Maximum spacing in each direction
        grid.spacing = tuple(np.max(np.diff(coord_i_direction))
                             for coord_i_direction in coord)
        if isinstance(values, torch.Tensor):
            grid.cell_data["values"] = flatten_fortran_w_clone(values)
        else:
            grid.cell_data["values"] = values.flatten(order="F")
        self.grid = grid

    def plot_pavement(self, cmap: plt.cm = None) -> None:
        """
        Plots a 3D pavement of the quantity of interest
        Parameters:
        ----------
        cmap: str
            Colormap you want to use
        Returns:
        ----------
        None
        """
        self.grid.plot(cmap=cmap, show_edges=False)

    def plot_slices(self, slice_index: List[int] = None, cmap: plt.cm = None) -> None:
        """
        Plots the 3 intersecting planes
        Parameters:
        ----------
        slice_index: List[int, int, int]
            The index of the slices you want in the x, y and z directions
        Returns:
        ----------
        """
        x, y, z = None, None, None
        if slice_index:
            x, y, z = slice_index
        self.grid.slice_orthogonal(x=x, y=y, z=z).plot(
            cmap=cmap, show_edges=False)


class IndexTracker:
    def __init__(self, ax, data, direction="Z"):
        self.ax = ax
        self.direction = direction.lower()

        self.X = data
        self.shape = data.shape

        if direction == "x":
            self.slices, cols, height = data.shape
            self.ind = self.slices // 2
            self.im = ax.imshow(self.X[self.ind, :, :])
            self.dim = 0
        elif direction == "y":
            rows, self.slices, height = data.shape
            self.ind = self.slices // 2
            self.im = ax.imshow(self.X[:, self.ind, :])
            self.dim = 1
        else:
            rows, cols, self.slices = data.shape
            self.ind = self.slices // 2
            self.im = ax.imshow(self.X[:, :, self.ind])
            self.dim = 2
        self.update()

    def on_scroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        if self.ind >= self.shape[self.dim]:
            self.ind = 0
        elif self.ind < 0:
            self.ind = self.shape[self.dim]
        self.update()

    def update(self):
        if self.direction == "x":
            self.im.set_data(self.X[self.ind, :, :])
        if self.direction == "y":
            self.im.set_data(self.X[:, self.ind, :])
        else:
            self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_title(f'slice {self.ind}/{self.shape[self.dim]}')
        # plt.legend(['slice %s' % self.ind])
        self.im.axes.figure.canvas.draw()


def look_slices(data):
    fig, ax = plt.subplots(1, 1)

    tracker = IndexTracker(ax, data)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()
