from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray


class TrellisPlotter:
    def __init__(self) -> None:
        self.cmap: Colormap | None = None
        self.colorbar_label_rotation: float | None = None
        self.grouped_fun: NDArray[np.float64] | None = None
        self.fun: NDArray[np.float64] | None = None
        self.data: NDArray[np.float64] | list[NDArray[np.float64]] | None = None
        self.data_sets: list[NDArray[np.float64]] = []
        self.intervals: NDArray[np.intp] = np.array([], dtype=np.intp)
        # options
        self.figsize: tuple[float, float] | None = None
        self.constrained_layout: bool = False
        self.marker: str = "o"
        self.markersize: float | None = None
        self.markeralpha: float | None = None
        self.n_xticks: int = 3
        self.n_yticks: int = 3
        self.xspace: float = 0.3
        self.yspace: float = 0.3
        self.xlabel: str = ""
        self.ylabel: str = ""
        self.xticks: NDArray[np.float64] | None = None
        self.yticks: NDArray[np.float64] | None = None
        self.xticklabels: list[str] | None = None
        self.yticklabels: list[str] | None = None
        self.oaxis_size: float = 0.20
        self.oaxis_n_xticks: int = 3
        self.oaxis_n_yticks: int = 3
        self.oaxis_xticks: NDArray[np.float64] | None = None
        self.oaxis_yticks: NDArray[np.float64] | None = None
        self.oaxis_xticklabels: list[str] | None = None
        self.oaxis_yticklabels: list[str] | None = None
        self.oaxis_bar_transparency: float = 0.6
        self.oaxis_xlabel: str = ""
        self.oaxis_ylabel: str = ""
        self.n_colorbar_ticks: int = 3
        self.label_fontsize: float | None = None
        # computed
        self.bounds: NDArray[np.float64] = np.empty((0, 2))
        self.bins: list[NDArray[np.float64]] = []
        self.group_bins: NDArray[np.float64] = np.empty((0,))
        self.n_groups: int = 0
        self.grouped_data: NDArray[np.float64] = np.empty((0,))
        # private
        self._multiple_data_sets: bool = False

    def initialize(self) -> None:
        if isinstance(self.data, list):
            self._multiple_data_sets = True
        else:
            self._multiple_data_sets = False

        # check data's validity
        if self._multiple_data_sets:
            data_list: list[NDArray[np.float64]] = self.data  # type: ignore[assignment]
            if not np.all([isinstance(datum, np.ndarray) for datum in data_list]):
                raise SyntaxError("All data sets must be a numpy array.")
            if not np.all([datum.ndim == 2 for datum in data_list]):
                raise SyntaxError("All data sets must be a 2D-array.")
            if not np.all([
                datum.shape[1] == data_list[0].shape[1]
                for datum in data_list
            ]):
                raise SyntaxError(
                    "Dimensions of points in the different data sets are inconsistent"
                )
        else:
            self.data = np.asarray(self.data)
            if not isinstance(self.data, np.ndarray):
                raise SyntaxError("Data must be a numpy array.")
            if self.data.ndim != 2:
                raise SyntaxError("Data must be a 2D-array.")

        # check if all data sets have the same dimension

        # check interval's validity
        self.intervals = np.asarray(self.intervals)
        if not isinstance(self.intervals, np.ndarray):
            raise SyntaxError("Intervals must be a numpy array.")
        if self.intervals.ndim != 1:
            raise SyntaxError("Intervals must be a 1D-array.")

        # check if interval agrees with given data
        if self._multiple_data_sets:
            data_list = self.data  # type: ignore[assignment]
            if self.intervals.shape[0] != (data_list[0].shape[1] - 2):
                raise SyntaxError("Dimensions in given interval and data does not agree.")
        else:
            data_arr: NDArray[np.float64] = self.data  # type: ignore[assignment]
            if self.intervals.shape[0] != (data_arr.shape[1] - 2):
                raise SyntaxError("Dimensions in given interval and data does not agree.")

        self.n_groups = int(np.prod(self.intervals))

        if self._multiple_data_sets:
            self.data_sets = self.data  # type: ignore[assignment]

        if self.fun is not None:
            self.fun = np.asarray(self.fun)
            if not isinstance(self.fun, np.ndarray):
                raise SyntaxError("Function values must be a numpy array.")
            if self.fun.ndim != 1:
                raise SyntaxError(f"Function values must be 1D array")
            data_arr = self.data  # type: ignore[assignment]
            if self.fun.size != data_arr.shape[0]:
                raise SyntaxError(f"Length of function values and given data points "
                                  f"are inconsistent.")

        return None

    def scatter(self) -> Figure:
        self.initialize()

        fig: Figure
        if not self._multiple_data_sets:
            self.classify_data()

            intervals = self.intervals
            ni, nj = int(intervals[0]), int(intervals[1])
            bounds = self.bounds
            group_bins = self.group_bins
            grouped_data = self.grouped_data

            width_ratios = np.ones(nj + 1)
            width_ratios[-1] = self.oaxis_size
            height_ratios = np.ones(ni + 1)
            height_ratios[0] = self.oaxis_size

            fig, axes = plt.subplots(
                nrows=ni + 1,
                ncols=nj + 1,
                gridspec_kw={
                    "wspace": self.xspace,
                    "hspace": self.yspace,
                    "width_ratios": width_ratios,
                    "height_ratios": height_ratios,
                },
                figsize=self.figsize,
                constrained_layout=self.constrained_layout
            )

            for pos, axis in np.ndenumerate(axes):
                r, c = pos
                axis.tick_params(
                    axis="both",
                    which="major",
                    labelsize=self.label_fontsize,
                )
                if r == 0 and c == nj:
                    fig.delaxes(axis)
                # horizontal outer axis
                elif r == 0 and c != nj:

                    # handle limits
                    axis.set_xlim([bounds[3, 0], bounds[3, 1]])
                    axis.set_ylim([0, 1])
                    # handle ticks
                    axis.set_yticks([])
                    axis.xaxis.tick_top()
                    if c % 2 == 0:
                        self.oaxis_xticks = np.linspace(
                            bounds[3, 0],
                            bounds[3, 1],
                            self.oaxis_n_xticks
                        )
                        axis.set_xticks(self.oaxis_xticks)
                        if self.oaxis_xticklabels is None:
                            self.oaxis_xticklabels = [
                                f"{tick:.2f}" for tick in self.oaxis_xticks
                            ]
                        axis.xaxis.set_ticklabels(self.oaxis_xticklabels)
                    else:
                        axis.set_xticks([])
                    # draw bar
                    axis.fill_between(
                        x=[
                            group_bins[0, c, 1, 0],
                            group_bins[0, c, 1, 1],
                        ],
                        y1=[1, 1],
                        y2=[0, 0],
                        facecolor="gray",
                        alpha=1 - self.oaxis_bar_transparency
                    )
                    # add label
                    axis.annotate(
                        text=self.oaxis_xlabel,
                        xy=(np.mean(bounds[3, :]), 0.5),
                        ha="center",
                        va="center",
                        size=self.label_fontsize,
                    )
                # vertical outer axis
                elif r != 0 and c == nj:
                    # draw vertical outer axes
                    axis.set_xlim([0, 1])
                    axis.set_ylim([bounds[2, 0], bounds[2, 1]])
                    # handle ticks
                    axis.set_xticks([])
                    axis.yaxis.tick_right()
                    if r % 2 == 0:
                        if self.oaxis_yticks is None:
                            self.oaxis_yticks = np.linspace(
                                bounds[2, 0], bounds[2, 1], self.oaxis_n_yticks
                            )
                        axis.set_yticks(self.oaxis_yticks)
                        if self.oaxis_yticklabels is None:
                            self.oaxis_yticklabels = [f"{tick:.2f}"
                                                      for tick in self.oaxis_yticks]
                        axis.yaxis.set_ticklabels(self.oaxis_yticklabels)
                    else:
                        axis.set_yticks([])
                    # draw bar
                    axis.fill_between(
                        x=[0, 1],
                        y1=[
                            group_bins[r-1, 0, 0, 1],
                            group_bins[r-1, 0, 0, 1],
                        ],
                        y2=[
                            group_bins[r-1, 0, 0, 0],
                            group_bins[r-1, 0, 0, 0],
                        ],
                        facecolor="gray",
                        alpha=1 - self.oaxis_bar_transparency
                    )
                    # add label
                    axis.annotate(
                        text=self.oaxis_ylabel,
                        xy=(0.50, np.mean(bounds[2, :])),
                        verticalalignment="center",
                        horizontalalignment="center",
                        size=self.label_fontsize,
                        rotation=270,
                    )
                # scatter
                elif r != 0 and c != nj:
                    axis.scatter(
                        grouped_data[r-1, c, :, 0],
                        grouped_data[r-1, c, :, 1],
                        marker=self.marker,
                        s=self.markersize,
                        alpha=self.markeralpha,
                    )
                    axis.set_xlim([
                        bounds[0, 0] - 0.10 * (bounds[0, 1] - bounds[0, 0]),
                        bounds[0, 1] + 0.10 * (bounds[0, 1] - bounds[0, 0]),
                    ])
                    axis.set_ylim([
                        bounds[1, 0] - 0.10 * (bounds[1, 1] - bounds[1, 0]),
                        bounds[1, 1] + 0.10 * (bounds[1, 1] - bounds[1, 0]),
                    ])
                    if c % 2 == 0 and r == ni:
                        if self.xticks is None:
                            self.xticks = np.linspace(
                                bounds[0, 0], bounds[0, 1], self.n_xticks
                            )
                        axis.set_xticks(self.xticks)
                        if self.xticklabels is None:
                            self.xticklabels = [f"{ticks:.2f}" for ticks in self.xticks]
                        axis.xaxis.set_ticklabels(self.xticklabels)
                    else:
                        axis.set_xticks([])

                    if r % 2 == 0 and c == 0:
                        if self.yticks is None:
                            self.yticks = np.linspace(
                                bounds[1, 0], bounds[1, 1], self.n_yticks
                            )
                        axis.set_yticks(self.yticks)
                        if self.yticklabels is None:
                            self.yticklabels = [f"{ticks:.2f}" for ticks in self.yticks]
                        axis.yaxis.set_ticklabels(self.yticklabels)
                    else:
                        axis.set_yticks([])

                    if c % 2 == 1 and r == ni:
                        axis.set_xlabel(self.xlabel, fontsize=self.label_fontsize)
                    if r % 2 == 1 and c == 0:
                        axis.set_ylabel(self.ylabel, fontsize=self.label_fontsize)
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()  # type: ignore[union-attr]

        else:
            for d_set in self.data_sets:
                self.data = d_set
                fig = self.scatter()

        return fig

    def contour(self, fun: NDArray[np.float64] | None = None,
                levels: int | None = None, scatter_data: bool = False) -> Figure:
        if fun is not None:
            self.fun = fun

        self.initialize()

        fig: Figure
        if not self._multiple_data_sets:
            self.classify_data()

            intervals = self.intervals
            ni, nj = int(intervals[0]), int(intervals[1])
            bounds = self.bounds
            group_bins = self.group_bins
            grouped_data = self.grouped_data
            grouped_fun = self.grouped_fun

            width_ratios = np.ones(nj + 1)
            width_ratios[-1] = self.oaxis_size
            height_ratios = np.ones(ni + 1)
            height_ratios[0] = self.oaxis_size

            fig, axes = plt.subplots(
                nrows=ni + 1,
                ncols=nj + 1,
                gridspec_kw={
                    "wspace": self.xspace,
                    "hspace": self.yspace,
                    "width_ratios": width_ratios,
                    "height_ratios": height_ratios,
                },
                figsize=self.figsize,
                constrained_layout=self.constrained_layout,
                # sharex="col",
                # sharey="row",
            )
            fig.subplots_adjust(
                top=0.95,
                bottom=0.05,
                left=0.05,
                right=0.95,
                hspace=0.2,
                wspace=0.2
            )

            for pos, axis in np.ndenumerate(axes):
                axis.tick_params(
                    axis="both",
                    which="major",
                    labelsize=self.label_fontsize,
                )
                c_axes = []
                r, c = pos
                if r == 0 and c == nj:
                    fig.delaxes(axis)
                # horizontal outer axis
                elif r == 0 and c != nj:
                    # handle limits
                    axis.set_xlim([bounds[3, 0], bounds[3, 1]])
                    axis.set_ylim([0, 1])
                    # handle ticks
                    axis.set_yticks([])
                    axis.xaxis.tick_top()
                    if c % 2 == 0:
                        self.oaxis_xticks = np.linspace(
                            bounds[3, 0],
                            bounds[3, 1],
                            self.oaxis_n_xticks
                        )
                        axis.set_xticks(self.oaxis_xticks)
                        if self.oaxis_xticklabels is None:
                            self.oaxis_xticklabels = [
                                f"{tick:.2f}" for tick in self.oaxis_xticks
                            ]
                        axis.xaxis.set_ticklabels(self.oaxis_xticklabels)
                    else:
                        axis.set_xticks([])
                    # draw bar
                    axis.fill_between(
                        x=[
                            group_bins[0, c, 1, 0],
                            group_bins[0, c, 1, 1],
                        ],
                        y1=[1, 1],
                        y2=[0, 0],
                        facecolor="gray",
                        alpha=1 - self.oaxis_bar_transparency
                    )
                    # add label
                    axis.annotate(
                        text=self.oaxis_xlabel,
                        xy=(np.mean(bounds[3, :]), 0.5),
                        ha="center",
                        va="center",
                        fontsize=self.label_fontsize,
                    )
                # vertical outer axis
                elif r != 0 and c == nj:
                    # draw vertical outer axes
                    axis.set_xlim([0, 1])
                    axis.set_ylim([bounds[2, 0], bounds[2, 1]])
                    # handle ticks
                    axis.set_xticks([])
                    axis.yaxis.tick_right()
                    if r % 2 == 0:
                        if self.oaxis_yticks is None:
                            self.oaxis_yticks = np.linspace(
                                bounds[2, 0], bounds[2, 1], self.oaxis_n_yticks
                            )
                        axis.set_yticks(self.oaxis_yticks)
                        if self.oaxis_yticklabels is None:
                            self.oaxis_yticklabels = [f"{tick:.2f}"
                                                      for tick in self.oaxis_yticks]
                        axis.yaxis.set_ticklabels(self.oaxis_yticklabels)
                    else:
                        axis.set_yticks([])
                    # draw bar
                    axis.fill_between(
                        x=[0, 1],
                        y1=[
                            group_bins[r - 1, 0, 0, 1],
                            group_bins[r - 1, 0, 0, 1],
                        ],
                        y2=[
                            group_bins[r - 1, 0, 0, 0],
                            group_bins[r - 1, 0, 0, 0],
                        ],
                        facecolor="gray",
                        alpha=1 - self.oaxis_bar_transparency
                    )
                    # add label
                    axis.annotate(
                        text=self.oaxis_ylabel,
                        xy=(0.50, np.mean(bounds[2, :])),
                        verticalalignment="center",
                        horizontalalignment="center",
                        fontsize=self.label_fontsize,
                        rotation=270,
                    )
                # contour
                elif r != 0 and c != nj:
                    c_axes.append(axis)
                    assert grouped_fun is not None
                    contourf = axis.tricontourf(
                        grouped_data[r - 1, c, :, 0][~np.isnan(grouped_data[r-1, c, :, 0])],
                        grouped_data[r - 1, c, :, 1][~np.isnan(grouped_data[r-1, c, :, 1])],
                        grouped_fun[r - 1, c, :][~np.isnan(grouped_fun[r - 1, c, :])],
                        levels=levels,
                        cmap=self.cmap,
                    )
                    if scatter_data:
                        axis.scatter(
                            grouped_data[r - 1, c, :, 0],
                            grouped_data[r - 1, c, :, 1],
                            alpha=self.markeralpha,
                            marker="o",
                            c="white",
                            s=self.markersize,
                        )
                    axis.set_xlim([
                        bounds[0, 0] - 0.10 * (
                                    bounds[0, 1] - bounds[0, 0]),
                        bounds[0, 1] + 0.10 * (
                                    bounds[0, 1] - bounds[0, 0]),
                    ])
                    axis.set_ylim([
                        bounds[1, 0] - 0.10 * (
                                    bounds[1, 1] - bounds[1, 0]),
                        bounds[1, 1] + 0.10 * (
                                    bounds[1, 1] - bounds[1, 0]),
                    ])

                    if c % 2 == 0 and r == ni:
                        if self.xticks is None:
                            self.xticks = np.linspace(
                                bounds[0, 0], bounds[0, 1], self.n_xticks
                            )
                        axis.set_xticks(self.xticks)
                        if self.xticklabels is None:
                            self.xticklabels = [f"{ticks:.2f}" for ticks in self.xticks]
                        axis.xaxis.set_ticklabels(self.xticklabels)
                    else:
                        axis.set_xticks([])

                    if r % 2 == 0 and c == 0:
                        if self.yticks is None:
                            self.yticks = np.linspace(
                                bounds[1, 0], bounds[1, 1], self.n_yticks
                            )
                        axis.set_yticks(self.yticks)
                        if self.yticklabels is None:
                            self.yticklabels = [f"{ticks:.2f}" for ticks in self.yticks]
                        axis.yaxis.set_ticklabels(self.yticklabels)
                    else:
                        axis.set_yticks([])

                    if c % 2 == 1 and r == ni:
                        axis.set_xlabel(self.xlabel, fontsize=self.label_fontsize)
                    if r % 2 == 1 and c == 0:
                        axis.set_ylabel(self.ylabel, fontsize=self.label_fontsize)
                    colorbar_ticks = np.linspace(
                            np.nanmin(grouped_fun[r-1, c, :]),
                            np.nanmax(grouped_fun[r-1, c, :]),
                            self.n_colorbar_ticks,
                        )
                    colorbar = fig.colorbar(
                        contourf,
                        ax=axis,
                        shrink=1.0,
                        orientation="vertical",
                        pad=0.05,
                        fraction=0.15,
                        ticks=colorbar_ticks,
                    )
                    colorbar.ax.tick_params(
                        labelsize=self.label_fontsize,
                        labelrotation=self.colorbar_label_rotation,
                    )

            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()  # type: ignore[union-attr]

            plt.show()
        else:
            for d_set in self.data_sets:
                self.data = d_set
                fig = self.scatter()

        return fig

    def get_bounds(self) -> NDArray[np.float64]:
        assert self.data is not None
        self.bounds = np.array(
            [np.nanmin(self.data, axis=0), np.nanmax(self.data, axis=0)]
        ).T
        return self.bounds

    def get_bins(self) -> NDArray[np.float64]:
        intervals = self.intervals
        self.bins = []
        for d, bound in enumerate(self.bounds):
            if d > 1:
                self.bins.append(np.linspace(bound[0], bound[1], int(intervals[d-2])+1))
        ni, nj = int(intervals[0]), int(intervals[1])
        self.group_bins = np.empty(shape=(ni, nj, 2, 2))
        for r in range(ni):
            for c in range(nj):
                self.group_bins[r, c, :, :] = np.array([
                    [self.bins[0][r], self.bins[0][r+1]],
                    [self.bins[1][c], self.bins[1][c+1]]
                ])
        self.group_bins = np.flip(self.group_bins, axis=0)
        return self.group_bins

    def classify_data(self) -> NDArray[np.float64]:
        self.get_bounds()
        self.get_bins()
        assert self.data is not None
        data: NDArray[np.float64] = self.data  # type: ignore[assignment]
        intervals = self.intervals
        ni, nj = int(intervals[0]), int(intervals[1])
        self.grouped_data = np.full(
            (ni, nj, data.shape[0], data.shape[1]),
            fill_value=np.nan,
        )
        if self.fun is not None:
            self.grouped_fun = np.full(
                (ni, nj, data.shape[0]),
                fill_value=np.nan,
            )
        group_bins = self.group_bins
        for r in range(ni):
            for c in range(nj):
                for p, datum in enumerate(data):
                    check1 = datum[2] >= group_bins[r, c, 0, 0]
                    check2 = datum[2] <= group_bins[r, c, 0, 1]
                    check3 = datum[3] >= group_bins[r, c, 1, 0]
                    check4 = datum[3] <= group_bins[r, c, 1, 1]
                    if np.all([check1, check2, check3, check4]):
                        self.grouped_data[r, c, p, :] = datum
                        if self.fun is not None:
                            assert self.grouped_fun is not None
                            self.grouped_fun[r, c, p] = self.fun[p]
        return self.grouped_data

    def add_data(self, data: NDArray[np.float64]) -> None:
        if self.data is None:
            self.data = data
        else:
            self.data = [self.data, data]  # type: ignore[list-item]


if __name__ == "__main__":
    # def fun(x):
    #     return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2

    def fun(x):  # type: ignore[no-untyped-def]
        return x[0] + x[1] + x[2]**2 + x[3]**3

    # def fun(x):
    #     return x[0] + x[1] + x[2] + x[3]

    # def fun(x):
    #     return x[0] ** 3 + x[1] ** 3 + x[2] ** 3 + x[3] ** 3

    # def fun(x):
    #     return np.sin(x[0]) + np.sin(x[1]) + np.sin(x[2]) + np.sin(x[3])

    # def fun(x):
    #     return x[0] ** 4 + x[1] ** 4 + x[2] ** 4 + x[3] ** 4

    plotter1 = TrellisPlotter()
    reso = 5j
    multiplier = 2
    x1, x2, x3, x4 = np.mgrid[
                     -1:1:reso*multiplier,  # type: ignore[misc]
                     -1:1:reso*multiplier,  # type: ignore[misc]
                     -1:1:5j,  # type: ignore[misc]
                     -1:1:7j  # type: ignore[misc]
                     ]
    plotter1.data = np.array([x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten()]).T
    plotter1.fun = fun(plotter1.data.T)  # type: ignore[union-attr]

    plotter1.intervals = np.array([5, 7])

    plotter1.label_fontsize = 6
    plotter1.xlabel = "x1"
    plotter1.ylabel = "x2"
    plotter1.oaxis_xlabel = "x4"
    plotter1.oaxis_ylabel = "x3"
    plotter1.markeralpha = 0.10
    plotter1.markersize = 5
    plotter1.n_colorbar_ticks = 4
    plotter1.cmap = plt.get_cmap("inferno")
    plotter1.contour(levels=5, scatter_data=False)
