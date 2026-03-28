from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import matplotlib
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.widgets import CheckButtons, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
from scipy.stats import chi2

if TYPE_CHECKING:
    # Attributes defined in Designer.__init__ that these methods access
    _biobjective_values: Any
    _criterion_value: Any
    _current_criterion: Any
    _cvar_problem: bool
    _discrete_design: bool
    _dynamic_controls: bool
    _dynamic_system: bool
    _opt_sampling_times: bool
    _pseudo_bayesian: bool
    _scr_sens: Any
    _sensitivity_is_normalized: bool
    _specified_n_spt: bool
    _status: str
    _verbose: int

    bayesian_pe_samples: Any
    beta: Any
    candidate_names: Any
    data: Any
    efforts: NDArray[np.float64]
    fim: NDArray[np.float64]
    max_n_opt_spt: int
    measurable_responses: Any
    model_parameter_names: Any
    model_parameter_unit_names: Any
    model_parameters: Any
    n_c: int
    n_exp: int
    n_m_r: int
    n_mp: int
    n_opt_c: int
    n_scr: int
    n_spt: int
    n_tic: int
    non_trimmed_apportionments: Any
    oed_result: Any
    optimal_candidates: Any
    phi: Any
    pvars: Any
    response: Any
    response_names: Any
    response_unit_names: Any
    run_no: int
    s: Any
    sampling_times_candidates: Any
    scr_responses: Any
    sensitivities: Any
    ti_controls_candidates: Any
    ti_controls_names: Any
    time_unit_name: Any
    tv_controls_candidates: Any
    v: Any


class DesignerVisualization:

    def plot_bayesian_inference_samples(self, bounds=None, title=None, contours=True, density=False, reso=201j, plot_fim_confidence=True, figsize=None, write=True, dpi=160):
        fig = corner.corner(
            self.bayesian_pe_samples,
            truths=self.model_parameters,
            plot_contours=contours,
            plot_density=density,
            range=bounds,
            levels=1.0 - np.exp(-0.5 * np.arange(1.0, 2.1, 1.0) ** 2),
            show_titles=True,
            quantiles=[0.025, 0.16, 0.50, 0.84, 0.975],
        )

        if plot_fim_confidence:
            bpe_mean = np.mean(self.bayesian_pe_samples, axis=0)
            _old_efforts = np.copy(self.efforts)
            _old_fim = np.copy(self.fim)
            print(f"Old FIM: \n {_old_fim}")
            unnorm_fim = self.eval_fim(self.non_trimmed_apportionments, store_predictions=False)
            print(f"Unnorm FIM: \n {unnorm_fim}")
            axes = fig.get_axes()
            axes = np.array(axes).reshape((self.n_mp, self.n_mp))
            for p1 in range(self.n_mp):
                for p2 in range(self.n_mp):
                    if p1 > p2:
                        temp_unnorm_fim = unnorm_fim[(p2, p1), ][:, (p2, p1)]
                        temp_bpe_mean = bpe_mean[(p2, p1), ]
                        print(f"The mean for p1: {p1}, p2: {p2} \n {temp_bpe_mean}")
                        print(f"The FIM for p1: {p1}, p2: {p2} \n {temp_unnorm_fim}")
                        x_lim = axes[p1, p2].get_xlim()
                        y_lim = axes[p1, p2].get_ylim()
                        x_grid, y_grid = np.mgrid[x_lim[0]:x_lim[1]:reso, y_lim[0]:y_lim[1]:reso]
                        x_grid, y_grid = x_grid.flatten(), y_grid.flatten()
                        xy_grid = np.array([x_grid, y_grid]).T
                        fim_pdf = []
                        for xy in xy_grid:
                            fim_pdf.append((xy - temp_bpe_mean) @ temp_unnorm_fim @ (xy - temp_bpe_mean).T)
                        fim_pdf = np.array(fim_pdf)
                        sigma_levels = chi2.ppf([chi2.cdf(1.0, 1), chi2.cdf(2.0**2, 1)], df=1)
                        if self._verbose >= 3:
                            axes[p1, p2].set_title(f"p1: {p1}, p2: {p2}")
                        axes[p1, p2].tricontour(
                            xy_grid[:, 0],
                            xy_grid[:, 1],
                            fim_pdf,
                            levels=sigma_levels,
                            colors=["tab:red"],
                            linestyles="dashed",
                            linewidths=1.5,
                        )
                        axes[p1, p2].scatter(
                            temp_bpe_mean[0],
                            temp_bpe_mean[1],
                            marker="H",
                            color="tab:red",
                            alpha=0.5,
                            s=100,
                        )
            self.fim = _old_fim
            self.efforts = _old_efforts
        fig.tight_layout()
        fig.suptitle(title)
        if write:
            fn = f"corner_bayes_pe_{self.n_exp}_exp"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fp, dpi=dpi)
        return fig


    def plot_criterion_cdf(self, write=False, iteration=None, dpi=360, figsize=(4.5, 3.5), annotate=False, minor_ticks=False, legend=False, grid=False):
        if not self._pseudo_bayesian or not self._cvar_problem:
            raise SyntaxError(
                "Plotting cumulative distribution function only valid for pseudo-"
                "bayesian and cvar problems."
            )

        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(111)
        if self._cvar_problem:
            x = np.sort(self.phi.value)
            mean = self.phi.value.mean()
            x = np.insert(x, 0, x[0])
            y = np.linspace(0, 1, x.size)
            axes.plot(x, y, "o--", alpha=0.3, c="#1f77b4")
            axes.plot(x, y, drawstyle="steps-post", c="#1f77b4")
            axes.axvline(
                x=self.v.value,
                ymin=0,
                ymax=1,
                c="tab:red",
                label=f"VaR {self.beta}",
            )
            axes.axvline(
                x=(self.v - 1 / (self.n_scr * (1 - self.beta)) * cp.sum(self.s)).value,
                ymin=0,
                ymax=1,
                c="tab:green",
                label=f"CVaR {self.beta}",
            )
            axes.axvline(
                x=mean,
                ymin=0,
                ymax=1,
                c="tab:blue",
                label=f"Mean",
            )
            axes.set_xlabel(f"{self._current_criterion}")
            axes.set_ylim(0, 1)
            axes.set_ylabel("Cumulative Probability")

            if legend:
                axes.legend()

            if minor_ticks:
                axes.xaxis.set_minor_locator(AutoMinorLocator(5))
                axes.yaxis.set_minor_locator(AutoMinorLocator(5))

            if grid:
                axes.grid(visible=False, which="both")

            if annotate:
                axes.axhline(
                    y=1-self.beta,
                    ls="--",
                    c="tab:red",
                )
                axes.annotate(
                    rf"$(1 - \beta) = {1 - self.beta:.2f}$",
                    xy=(0.20, 1 - self.beta),
                    xytext=(0.50, 1 - self.beta + 0.25),
                    xycoords="axes fraction",
                    arrowprops={
                        "width": 5,
                        "shrink": 0.05,
                        "facecolor": "tab:red",
                        "edgecolor": "k",
                    },
                )

                axes.annotate(
                    "VaR",
                    xy=(self.v.value, 0.80),
                    xytext=(self.v.value + 0.2 * np.abs(self.v.value), 0.80),
                    arrowprops={
                        "width": 5,
                        "shrink": 0.05,
                        "facecolor": "tab:red",
                        "edgecolor": "k",
                    },
                )
                cvar = (self.v - 1 / (self.n_scr * (1 - self.beta)) * cp.sum(self.s)).value
                axes.annotate(
                    "CVaR",
                    xy=(cvar, 0.50),
                    xytext=(cvar + 0.2 * np.abs(cvar), 0.50),
                    arrowprops={
                        "width": 5,
                        "shrink": 0.05,
                        "facecolor": "tab:green",
                        "edgecolor": "k",
                    },
                )
                axes.annotate(
                    "Mean",
                    xy=(mean, 0.10),
                    xytext=(mean + 0.2 * np.abs(mean), 0.10),
                    arrowprops={
                        "width": 5,
                        "shrink": 0.05,
                        "facecolor": "tab:blue",
                        "edgecolor": "k",
                    },
                )
            fig.tight_layout()
        else:
            raise NotImplementedError(
                "Plotting cumulative distribution function not implemented for pseudo-"
                "bayesian problems."
            )

        if write:
            fn = f"cdf_{self.beta*100}_beta_{self.n_scr}_scr"
            fp = self._generate_result_path(fn, "png", iteration=iteration)
            fig.savefig(fname=fp, dpi=dpi)

        return fig


    def plot_criterion_pdf(self, n_bins=20, write=False, iteration=None, dpi=360):
        if not self._pseudo_bayesian or not self._cvar_problem:
            raise SyntaxError(
                "Plotting probability density function only valid for pseudo-"
                "bayesian and cvar problems."
            )

        fig = plt.figure()
        axes = fig.add_subplot(111)
        if self._cvar_problem:
            x = self.phi.value
            axes.hist(x, bins=n_bins)
            axes.axvline(
                self.v.value,
                0,
                1,
                c="tab:red",
                label=f"VaR {self.beta}",
            )
            axes.axvline(
                self._criterion_value,
                0,
                1,
                c="tab:green",
                label=f"CVaR {self.beta}",
            )
            axes.set_xlabel(f"{self._current_criterion}")
            axes.set_ylabel("Frequency")
            axes.legend()
            fig.tight_layout()
        else:
            raise NotImplementedError(
                "Plotting probability density function not implemented for pseudo-"
                "bayesian problems."
            )

        if write:
            fn = f"pdf_{self.beta*100}_beta_{self.n_scr}_scr"
            fp = self._generate_result_path(fn, "png", iteration=iteration)
            fig.savefig(fname=fp, dpi=dpi)

        return fig


    def plot_optimal_efforts(self, width=None, write=False, dpi=720,
                             force_3d=False, tol=1e-4, heatmap=False, figsize=None):
        if self.optimal_candidates is None:
            self.get_optimal_candidates()
        if self.n_opt_c == 0:
            print("Empty candidates, skipping plotting of optimal efforts.")
            return
        if heatmap:
            if not self._dynamic_system:
                print(
                    f"Warning: heatmaps are not suitable for non-dynamic experimental "
                    f"results. Reverting to bar charts."
                )
                fig = self._plot_current_efforts_2d(width=width, write=write, dpi=dpi,
                                                    tol=tol, figsize=figsize)
                return fig
            return self._efforts_heatmap(figsize=figsize, write=write)
        if (self._opt_sampling_times or force_3d) and self._dynamic_system:
            fig = self._plot_current_efforts_3d(tol=tol, width=width, write=write,
                                                dpi=dpi, figsize=figsize)
            return fig
        else:
            if force_3d:
                print(
                    "Warning: force 3d only works for dynamic systems, plotting "
                    "current design in 2D."
                )
            fig = self._plot_current_efforts_2d(width=width, write=write, dpi=dpi,
                                                tol=tol, figsize=figsize)
        return fig


    def _heatmap(self, data, row_labels, col_labels, ax=None,
                 cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        ax.tick_params(top=False, bottom=True,
                       labeltop=False, labelbottom=True)

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax.set_title(f"{self._current_criterion} Efforts")
        ax.set_xlabel(f"Sampling Times (min)")

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar


    def _annotate_heatmap(self, im, data=None, valfmt="{x:.2f}",
                          textcolors=("black", "white"),
                          threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts


    def _efforts_heatmap(self, figsize=None, write=False, dpi=360):
        if figsize is None:
            fig = plt.figure(figsize=(3 + 1.0 * self.max_n_opt_spt, 2 + 0.40 * self.n_opt_c))
        else:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        c_id = [f"Candidate {opt_c[0]+1}" for opt_c in self.optimal_candidates]
        spt_id = [opt_c[3] for opt_c in self.optimal_candidates]
        spt_id = np.unique(np.array(list(itertools.zip_longest(*spt_id, fillvalue=spt_id[0][0]))).T)

        eff = np.zeros((len(c_id), spt_id.shape[0]))
        for c, opt_c in enumerate(self.optimal_candidates):
            for opt_spt, opt_eff in zip(opt_c[3], opt_c[4]):
                spt_index = np.where(spt_id == opt_spt)[0][0]
                eff[c, spt_index] = opt_eff

        im, cbar = self._heatmap(eff * 100, c_id, spt_id, ax=ax, cmap="YlGn")
        texts = self._annotate_heatmap(im, valfmt="{x:.2f}%")

        fig.tight_layout()
        if write:
            fn = f'efforts_heatmap_{self._current_criterion}'
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)

        return fig


    def plot_optimal_controls(self, alpha=0.3, markersize=3, non_opt_candidates=False,
                              n_ticks=3, visualize_efforts=True, tol=1e-4,
                              intervals=None, title=False, write=False, dpi=720):
        if self._dynamic_system:
            print(
                "[Warning]: Plot optimal controls is not implemented for dynamic "
                "system, use print_optimal_candidates, or plot_optimal_sensitivities "
                "for visualization."
            )
            return
        if self.optimal_candidates is None:
            self.get_optimal_candidates()
        if self.n_opt_c == 0:
            print(
                f"[Warning]: empty optimal candidates, skipping plotting of optimal "
                f"controls."
            )
            return
        if self._dynamic_controls:
            raise NotImplementedError(
                "Plot controls not implemented for dynamic controls"
            )
        if self.n_tic > 4:
            raise NotImplementedError(
                "Plot controls not implemented for systems with more than 4 ti_controls"
            )
        if self.n_tic == 1:
            fig, axes = plt.subplots(1, 1)
            if title:
                axes.set_title(self._current_criterion)
            if visualize_efforts:
                opt_idx = np.where(self.efforts >= tol)
                delta = self.ti_controls_candidates[:, 0].max() - self.ti_controls_candidates[:, 0].min()
                axes.bar(
                    self.ti_controls_candidates[:, 0],
                    self.efforts[:, 0],
                    width=0.01 * delta,
                )
                axes.set_ylim([0, 1])
                axes.set_xlabel("Control 1")
                axes.set_ylabel("Efforts")
        elif self.n_tic == 2:
            fig, axes = plt.subplots(1, 1)
            if title:
                axes.set_title(self._current_criterion)
            if non_opt_candidates:
                axes.scatter(
                    self.ti_controls_candidates[:, 0],
                    self.ti_controls_candidates[:, 1],
                    alpha=alpha,
                    marker="o",
                    s=18*markersize,
                )
            if visualize_efforts:
                opt_idx = np.where(self.efforts >= tol)
                axes.scatter(
                    self.ti_controls_candidates[opt_idx[0], 0].T,
                    self.ti_controls_candidates[opt_idx[0], 1].T,
                    facecolor="none",
                    edgecolor="red",
                    marker="o",
                    s=self.efforts[opt_idx]*500*markersize,
                )
            if self.ti_controls_names is None:
                axes.set_xlabel("Time-invariant Control 1")
                axes.set_ylabel("Time-invariant Control 2")
            else:
                axes.set_xlabel(self.ti_controls_names[0])
                axes.set_ylabel(self.ti_controls_names[1])
            axes.set_xticks(
                np.linspace(
                    self.ti_controls_candidates[:, 0].min(),
                    self.ti_controls_candidates[:, 0].max(),
                    n_ticks,
                )
            )
            axes.set_yticks(
                np.linspace(
                    self.ti_controls_candidates[:, 1].min(),
                    self.ti_controls_candidates[:, 1].max(),
                    n_ticks,
                )
            )
            fig.tight_layout()
        elif self.n_tic == 3:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection="3d")
            if non_opt_candidates:
                axes.scatter(
                    self.ti_controls_candidates[:, 0],
                    self.ti_controls_candidates[:, 1],
                    self.ti_controls_candidates[:, 2],
                    alpha=alpha,
                    marker="o",
                    s=18*markersize,
                )
            opt_idx = np.where(self.efforts >= tol)[0]
            axes.scatter(
                self.ti_controls_candidates[opt_idx, 0],
                self.ti_controls_candidates[opt_idx, 1],
                self.ti_controls_candidates[opt_idx, 2],
                facecolor="r",
                edgecolor="r",
                s=self.efforts[opt_idx] * 500 * markersize,
            )
            if self.ti_controls_names is not None:
                axes.set_xlabel(f"{self.ti_controls_names[0]}")
                axes.set_ylabel(f"{self.ti_controls_names[1]}")
                axes.set_zlabel(f"{self.ti_controls_names[2]}")
            axes.grid(False)
            fig.tight_layout()
        elif self.n_tic == 4:
            trellis_plotter = TrellisPlotter()
            trellis_plotter.data = self.ti_controls_candidates
            trellis_plotter.markersize = self.efforts * 500
            if intervals is None:
                intervals = np.array([5, 5])
            trellis_plotter.intervals = intervals
            fig = trellis_plotter.scatter()

        if write:
            fn = f"optimal_controls_{self.oed_result['optimality_criterion']}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)

        return fig


    def plot_parity(self):
        if self.response is None:
            raise RuntimeError(
                "Cannot generate parity plot when response is empty. "
                "Run simulate_all_candidates and store responses."
            )
        if self.data is None:
            raise RuntimeError(
                "Cannot generate parity plot when data is empty. "
                "Please specify data to the designer."
            )
        fig = plt.figure()
        n_rows = np.ceil(np.sqrt(self.n_m_r)).astype(int)
        gridspec = plt.GridSpec(nrows=n_rows, ncols=n_rows)
        for r in range(self.n_m_r):
            row = r % n_rows
            col = np.floor_divide(r, n_rows)
            axes = fig.add_subplot(gridspec[row, col])
            axes.scatter(
                [dat for dat in self.data[:, :, r]],
                [res for res in self.response[:, :, self.measurable_responses[r]]],
                marker="1",
            )
            axes.plot(
                [-1e10, 1e10],
                [-1e10, 1e10],
                linestyle="-",
                marker="None",
                c="gray",
                alpha=0.3,
            )
            data_lim: list[Any] = [
                np.nanmin(self.data[:, :, r]),
                np.nanmax(self.data[:, :, r]),
            ]
            res_lim: list[Any] = [
                np.nanmin(self.response[:, :, self.measurable_responses[r]]),
                np.nanmax(self.response[:, :, self.measurable_responses[r]]),
            ]
            lim: list[Any] = [
                np.min([data_lim[0], res_lim[0]]),
                np.max([data_lim[1], res_lim[1]]),
            ]
            lim = lim + np.array([
                -0.1 * (lim[1] - lim[0]),
                 0.1 * (lim[1] - lim[0]),
            ])
            axes.set_xlim(lim)
            axes.set_ylim(lim)
            if self.response_names is not None:
                axes.set_title(f"{self.response_names[r]}")
            else:
                axes.set_title(f"Response {r}")
            axes.set_xlabel(f"Data")
            axes.set_ylabel(f"Prediction")
        plt.get_current_fig_manager().window.showMaximized()
        plt.gcf().tight_layout()
        return fig


    def plot_predictions(self, plot_data=False, figsize=None, label_candidates=True):
        if not self._dynamic_system:
            raise NotImplementedError(
                f"Plot predictions not supported for non-dynamic systems."
            )
        if figsize is None:
            figsize = (15, 8)
        if self.response is None:
            self.simulate_candidates()
        figs = []
        for res in range(self.n_m_r):
            fig = plt.figure(figsize=figsize)
            n_rows = np.ceil(np.sqrt(self.n_c)).astype(int)
            n_cols = n_rows
            gridspec = plt.GridSpec(
                nrows=n_rows,
                ncols=n_cols,
            )
            res_lim: list[Any] = [
                np.nanmin(self.response[:, :, self.measurable_responses[res]]),
                np.nanmax(self.response[:, :, self.measurable_responses[res]]),
            ]
            if plot_data:
                data_lim: list[Any] = [
                    np.nanmin(self.data[:, :, res]),
                    np.nanmax(self.data[:, :, res]),
                ]
            else:
                data_lim = res_lim
            lim: list[Any] = [
                np.min([data_lim[0], res_lim[0]]),
                np.max([data_lim[1], res_lim[1]])
            ]
            lim = lim + np.array([
                - 0.1 * (lim[1] - lim[0]),
                + 0.1 * (lim[1] - lim[0]),
            ])
            for row in range(n_rows):
                for col in range(n_cols):
                    cand = n_cols * row + col
                    if cand < self.n_c:
                        axes = fig.add_subplot(gridspec[row, col])
                        axes.plot(
                            self.sampling_times_candidates[cand, :],
                            self.response[n_cols*row + col, :, self.measurable_responses[res]],
                            linestyle="-",
                            marker="1",
                            label="Prediction"
                        )
                        if plot_data:
                            axes.plot(
                                self.sampling_times_candidates[cand, :],
                                self.data[n_cols * row + col, :, res],
                                linestyle="none",
                                marker="v",
                                fillstyle="none",
                                label="Data"
                            )
                        axes.set_ylim(lim)
                        if self.time_unit_name is not None:
                            axes.set_xlabel(f"Time ({self.time_unit_name})")
                        else:
                            axes.set_xlabel('Time')
                        ylabel = self.response_names[res]
                        if self.response_unit_names is not None:
                            ylabel += f" ({self.response_unit_names[res]})"
                        axes.set_ylabel(ylabel)
                        if cand + 1 == self.n_c:
                            axes.legend(prop={"size": 6})
                        if label_candidates:
                            axes.set_title(f"{self.candidate_names[cand]}")
            if self.response_names is not None:
                fig.suptitle(f"Response: {self.response_names[res]}")
            fig.tight_layout()
            figs.append(fig)
        return figs


    def plot_sensitivities(self, absolute=False, legend=None, figsize=None):
        # n_c, n_s_times, n_res, n_theta = self.sensitivity.shape
        if self.sensitivities is None:
            self.eval_sensitivities()
        if figsize is None:
            figsize = (self.n_mp * 4.0, 1.0 + 2.5 * self.n_m_r)
        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=self.n_m_r,
            ncols=self.n_mp,
            sharex=True,
        )
        if legend is None:
            if self.n_c < 6:
                legend = True
        if self._sensitivity_is_normalized:
            norm_status = 'Normalized '
        else:
            norm_status = 'Unnormalized '
        if absolute:
            abs_status = 'Absolute '
        else:
            abs_status = 'Directional '

        fig.suptitle('%s%sSensitivity Plots' % (norm_status, abs_status))
        for row in range(self.n_m_r):
            for col in range(self.n_mp):
                for c, exp_candidate in enumerate(
                        zip(self.ti_controls_candidates, self.tv_controls_candidates,
                            self.sampling_times_candidates)):
                    sens = self.sensitivities[
                           c,
                           :,
                           self.measurable_responses[row],
                           col,
                           ]
                    axes[row, col].plot(
                        exp_candidate[2],
                        sens,
                        "-o",
                        label=f"Candidate {c + 1}"
                    )
                    axes[row, col].ticklabel_format(
                        axis="y",
                        style="sci",
                        scilimits=(0, 0),
                    )
                    if self.time_unit_name is not None:
                        axes.set_xlabel(f"Sampling Times ({self.time_unit_name})")
                    else:
                        axes.set_xlabel('Sampling Times')
                    ylabel = self.response_names[self.measurable_responses[row]]
                    ylabel += "/"
                    ylabel += self.model_parameter_names[col]
                    if self.response_unit_names is not None:
                        if self.model_parameter_unit_names is not None:
                            ylabel += f" ({self.response_unit_names[row]}/{self.model_parameter_unit_names[col]})"
                    axes.set_ylabel(ylabel)
                if legend and self.n_c <= 10:
                    axes[-1, -1].legend()
        fig.tight_layout()
        return fig


    def plot_optimal_predictions(self, legend=None, figsize=None, markersize=10,
                                 fontsize=10, legend_size=8, colour_map="jet",
                                 write=False, dpi=720):
        if not self._dynamic_system:
            raise SyntaxError("Prediction plots are only for dynamic systems.")

        if self._status != 'ready':
            raise SyntaxError(
                'Initialize the designer first.'
            )

        if self._pseudo_bayesian:
            if self.scr_responses is None:
                raise SyntaxError(
                    'Cannot plot prediction vs data when scr_response is empty, please '
                    'run a semi-bayes experimental design, and store predictions.'
                )
            mean_res = np.average(self.scr_responses, axis=0)
            std_res = np.std(self.scr_responses, axis=0)
        else:
            if self.response is None:
                self.simulate_candidates(store_predictions=True)

        if self.optimal_candidates is None:
            self.get_optimal_candidates()
        if self.n_opt_c == 0:
            print(
                f"[Warning]: empty optimal candidates, skipping plotting of optimal "
                f"predictions."
            )
            return
        if legend is None:
            if self.n_opt_c < 6:
                legend = True
        if figsize is None:
            figsize = (4.0, 1.0 + 2.5 * self.n_m_r)

        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=self.n_m_r,
            ncols=1,
            sharex=True,
        )
        if self.n_m_r == 1:
            axes = [axes]
        """ defining fig's subplot axes limits """
        x_axis_lim: list[Any] = [
            np.min(self.sampling_times_candidates[
                       ~np.isnan(self.sampling_times_candidates)]),
            np.max(self.sampling_times_candidates[
                       ~np.isnan(self.sampling_times_candidates)])
        ]
        for res in range(self.n_m_r):
            if self._pseudo_bayesian:
                res_max: Any = np.nanmax(mean_res[:, :, res] + std_res[:, :, res])
                res_min: Any = np.nanmin(mean_res[:, :, res] - std_res[:, :, res])
            else:
                res_max = np.nanmax(self.response[:, :, res])
                res_min = np.nanmin(self.response[:, :, res])
            y_axis_lim = [res_min, res_max]
            if self._pseudo_bayesian:
                plot_response = mean_res
            else:
                plot_response = self.response
            ax = axes[res]
            cmap = cm.get_cmap(colour_map, len(self.optimal_candidates))
            colors = itertools.cycle([
                cmap(_) for _ in np.linspace(0, 1, len(self.optimal_candidates))
            ])
            for c, cand in enumerate(self.optimal_candidates):
                color = next(colors)
                ax.plot(
                    self.sampling_times_candidates[cand[0]],
                    plot_response[
                        cand[0],
                        :,
                        self.measurable_responses[res]
                    ],
                    linestyle="--",
                    label=f"Candidate {cand[0] + 1:d}",
                    zorder=0,
                    c=color,
                )
                if self._pseudo_bayesian:
                    ax.fill_between(
                        self.sampling_times_candidates[cand[0]],
                        plot_response[
                            cand[0],
                            :,
                            self.measurable_responses[res]
                        ]
                        +
                        std_res[
                            cand[0],
                            :,
                            self.measurable_responses[res]
                        ],
                        mean_res[
                            cand[0],
                            :,
                            self.measurable_responses[res]
                        ]
                        -
                        std_res[
                            cand[0],
                            :,
                            self.measurable_responses[res]
                        ],
                        alpha=0.1,
                        facecolor=color,
                        zorder=1
                    )
                if not self._specified_n_spt:
                    ax.scatter(
                        cand[3],
                        plot_response[
                            cand[0],
                            cand[5],
                            self.measurable_responses[res]
                        ],
                        marker="o",
                        s=markersize * 50 * np.array(cand[4]),
                        zorder=2,
                        # c=np.array([color]),
                        color=color,
                        facecolors="none",
                    )
                else:
                    markers = itertools.cycle(["o", "s", "h", "P"])
                    for i, (eff, spt, spt_idx) in enumerate(zip(cand[4], cand[3], cand[5])):
                        marker = next(markers)
                        ax.scatter(
                            spt,
                            plot_response[
                                cand[0],
                                spt_idx,
                                self.measurable_responses[res]
                            ],
                            marker=marker,
                            s=markersize * 50 * np.array(eff),
                            color=color,
                            label=f"Variant {i + 1}",
                            facecolors="none",
                        )
                ax.set_xlim(
                    x_axis_lim[0] - 0.1 * (x_axis_lim[1] - x_axis_lim[0]),
                    x_axis_lim[1] + 0.1 * (x_axis_lim[1] - x_axis_lim[0])
                )
                ax.set_ylim(
                    y_axis_lim[0] - 0.1 * (y_axis_lim[1] - y_axis_lim[0]),
                    y_axis_lim[1] + 0.1 * (y_axis_lim[1] - y_axis_lim[0])
                )
                ax.tick_params(axis="both", which="major", labelsize=fontsize)
                ax.yaxis.get_offset_text().set_fontsize(fontsize)
                if self.response_names is None:
                    ylabel = f"Response {res+1}"
                else:
                    ylabel = f"{self.response_names[res]}"
                if self.response_unit_names is None:
                    pass
                else:
                    ylabel += f" ({self.response_unit_names[res]})"
                ax.set_ylabel(ylabel)
        if self.time_unit_name is not None:
            axes[-1].set_xlabel(f"Time ({self.time_unit_name})")
        else:
            axes[-1].set_xlabel('Time')
        if legend and len(self.optimal_candidates) > 1:
            axes[-1].legend(prop={"size": legend_size})

        fig.tight_layout()

        if write:
            fn = f"response_plot_{self.oed_result['optimality_criterion']}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)

        return fig


    def plot_optimal_sensitivities(self, figsize=None, markersize=10, colour_map="jet",
                                   write=False, dpi=720, interactive=False):
        if interactive:
            self._plot_optimal_sensitivities_interactive(
                figsize=figsize,
                markersize=markersize,
                colour_map=colour_map,
            )
        else:
            self._plot_optimal_sensitivities(
                figsize=figsize,
                markersize=markersize,
                colour_map=colour_map,
                write=write,
                dpi=dpi,
            )


    def plot_pareto_frontier(self, write=False, dpi=720):
        if not self._cvar_problem:
            raise SyntaxError(
                "Pareto Frontier can only be plotted after solution of a CVaR problem."
            )

        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.scatter(
            self._biobjective_values[:, 0],
            self._biobjective_values[:, 1],
        )
        axes.set_xlabel("Mean Criterion Value")
        axes.set_ylabel(f"CVaR of Bottom {100 * (1 - self.beta):.2f}%")

        fig.tight_layout()

        if write:
            fn = f"optimal_controls_{self.oed_result['optimality_criterion']}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)


    def plot_prediction_variance(self, reso=None, bounds=None, alpha=0.5):
        """
        Plots the prediction variance of the optimal experiment design. To be run after
        an optimal design is computed. Only supports time-invariant, static systems with
        less than or equal to two inputs and outputs.
        """
        if self._dynamic_system:
            print(
                "[WARNING]: dynamic systems are not supported for "
                "plot_prediction_variance. Skipping command."
            )
            return
        if self.n_tic > 2:
            print(
                f"[WARNING]: plot_prediction_variance supports less than or equal to"
                f" two time-invariant controls. The designer detects {self.n_tic} number"
                f" of tics. Skipping command."
            )
            return
        if self.n_m_r > 2:
            print(
                f"[WARNING]: plot_prediction_variance supports less than or equal to"
                f" two measured responses. The designer detects {self.n_m_r} number"
                f" of measured responses. Skipping command."
            )
            return
        if reso:
            pass
        else:
            reso = 11j
        fig1 = plt.figure(figsize=(12, 5))
        axes1 = fig1.add_subplot(121)

        axes1.scatter(
            self.ti_controls_candidates[:, 0],
            self.ti_controls_candidates[:, 1],
            alpha=alpha,
        )
        axes1.scatter(
            self.ti_controls_candidates[:, 0],
            self.ti_controls_candidates[:, 1],
            s=self.efforts * 400,
        )

        if self.pvars is None:
            self.eval_pim_for_v_opt(self.efforts)

        axes2 = fig1.add_subplot(122)
        y1, y2 = np.mgrid[bounds[0][0]:bounds[0][1]:reso, bounds[1][0]:bounds[1][1]:reso]
        y1 = y1.flatten()
        y2 = y2.flatten()
        y_list = np.array([y1, y2]).transpose()

        print("Please select initial control to initialize plot.")
        global x, pvar
        x = np.array(fig1.ginput(1))[0]
        print("Chosen:")
        print(x)
        current_control = axes1.scatter(x[0], x[1], marker='x', s=50)
        contour_levels = [
            chi2.ppf(q=0.6827, df=2),
            chi2.ppf(q=0.9545, df=2),
            chi2.ppf(q=0.9973, df=2),
        ]
        contour1 = axes2.tricontour(
            y_list[:, 0],
            y_list[:, 1],
            predict_var,  # type: ignore[name-defined]
            levels=contour_levels,
        )
        c_labels = [r'$68.27\%$',
                    r'$95.45\%$',
                    r'$99.73\%$']
        c_fmt = {}
        for level, label in zip(contour1.levels, c_labels):
            c_fmt[level] = label
        axes2.clabel(contour1, inline=1, fontsize=20, fmt=c_fmt)
        axes2.set_title(r"$x_1 = $ %.2f, $x_2 = $%.2f" % (x[0], x[1]))
        axes2.set_ylabel(r"$y_2$")
        axes2.set_xlabel(r"$y_1$")

        plt.draw()

        def recentre(event):
            if event.button == 1 and event.inaxes == axes2:
                bounds = np.array([axes2.get_xlim(), axes2.get_ylim()])
                ranges = np.array(
                    [bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]]) / 2
                bounds = np.array([[event.xdata - ranges[0], event.xdata + ranges[0]],
                                   [event.ydata - ranges[1], event.ydata + ranges[1]]])
                y1, y2 = np.mgrid[bounds[0][0]:bounds[0][1]:reso,
                         bounds[1][0]:bounds[1][1]:reso]
                y1 = y1.flatten()
                y2 = y2.flatten()
                y_list = np.array([y1, y2]).transpose()

                predict_var = np.array([])
                for y in y_list:
                    predict_var = np.append(predict_var,
                                            y.dot(np.linalg.inv(pvar)).dot(y.transpose()))

                axes2.clear()
                contour1 = axes2.tricontour(y_list[:, 0], y_list[:, 1], predict_var,
                                            levels=contour_levels)
                axes2.clabel(contour1, inline=1, fontsize=10, fmt=c_fmt)
                axes2.set_title(r"$x_1 = $ %.2f, $x_2 = $%.2f" % (x[0], x[1]))
                axes2.set_ylabel(r"$y_2$")
                axes2.set_xlabel(r"$y_1$")

                plt.draw()

        def change_x(event):
            if event.inaxes == axes1:
                bounds = np.array([axes2.get_xlim(), axes2.get_ylim()])
                y1, y2 = np.mgrid[bounds[0][0]:bounds[0][1]:reso,
                         bounds[1][0]:bounds[1][1]:reso]
                y1 = y1.flatten()
                y2 = y2.flatten()
                y_list = np.array([y1, y2]).transpose()

                global x, pvar
                x = np.array([event.xdata, event.ydata])
                pvar = self.eval_pvar(x)  # type: ignore[attr-defined]
                predict_var = np.array([])
                for y in y_list:
                    predict_var = np.append(predict_var,
                                            y.dot(np.linalg.inv(pvar)).dot(y.transpose()))

                current_control.set_offsets([x[0], x[1]])

                axes2.clear()
                contour1 = axes2.tricontour(y_list[:, 0], y_list[:, 1], predict_var,
                                            levels=contour_levels)
                axes2.clabel(contour1, inline=1, fontsize=10, fmt=c_fmt)
                axes2.set_title(r"$x_1 = $ %.2f, $x_2 = $%.2f" % (x[0], x[1]))
                axes2.set_ylabel(r"$y_2$")
                axes2.set_xlabel(r"$y_1$")

                plt.draw()

        def zoom(event):
            sensitivity = 0.2
            if event.inaxes == axes2:
                bounds = np.array([axes2.get_xlim(), axes2.get_ylim()])
                ranges = np.array(
                    [bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]]) / 2
                if keyboard.is_pressed("shift"):  # type: ignore[name-defined]
                    bounds = bounds + sensitivity * np.array(
                        [[-event.step * ranges[0], event.step * ranges[0]], [0, 0]])
                elif keyboard.is_pressed("ctrl"):  # type: ignore[name-defined]
                    bounds = bounds + sensitivity * np.array(
                        [[0, 0], [-event.step * ranges[1], event.step * ranges[1]]])
                else:
                    bounds = bounds + sensitivity * np.array(
                        [[-event.step * ranges[0], event.step * ranges[0]],
                         [-event.step * ranges[1], event.step * ranges[1]]])
                y1, y2 = np.mgrid[bounds[0][0]:bounds[0][1]:reso,
                         bounds[1][0]:bounds[1][1]:reso]
                y1 = y1.flatten()
                y2 = y2.flatten()
                y_list = np.array([y1, y2]).transpose()

                predict_var = np.array([])
                for y in y_list:
                    predict_var = np.append(predict_var,
                                            y.dot(np.linalg.inv(pvar)).dot(y.transpose()))

                axes2.clear()
                contour1 = axes2.tricontour(y_list[:, 0], y_list[:, 1], predict_var,
                                            levels=contour_levels)
                axes2.clabel(contour1, inline=1, fontsize=10, fmt=c_fmt)
                axes2.set_title(r"$x_1 = $ %.2f, $x_2 = $%.2f" % (x[0], x[1]))
                axes2.set_ylabel(r"$y_2$")
                axes2.set_xlabel(r"$y_1$")

                plt.draw()

        fig1.canvas.mpl_connect('button_press_event', recentre)
        fig1.canvas.mpl_connect('button_press_event', change_x)
        fig1.canvas.mpl_connect("scroll_event", zoom)

        plt.show()

    """ evaluators """


    def _plot_optimal_sensitivities(self, absolute=False, legend=None,
                                   markersize=10, colour_map="jet",
                                   write=False, dpi=720, figsize=None):
        if not self._dynamic_system:
            raise SyntaxError("Sensitivity plots are only for dynamic systems.")

        if self.optimal_candidates is None:
            self.get_optimal_candidates()
        if self.n_opt_c == 0:
            print(
                f"[Warning]: empty optimal candidates, skipping plotting of optimal "
                f"predictions."
            )
            return
        if legend is None:
            if self.n_opt_c < 6:
                legend = True
        if figsize is None:
            figsize = (self.n_mp * 4.0, 1.0 + 2.5 * self.n_m_r)

        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=self.n_m_r,
            ncols=self.n_mp,
            sharex=True,
        )
        if self.n_m_r == 1 and self.n_mp == 1:
            axes = np.array([[axes]])
        elif self.n_m_r == 1:
            axes = np.array([axes])
        elif self.n_mp == 1:
            axes = np.array([axes]).T

        if self._pseudo_bayesian:
            mean_sens = np.nanmean(self._scr_sens, axis=0)
            std_sens = np.nanstd(self._scr_sens, axis=0)

        for row in range(self.n_m_r):
            for col in range(self.n_mp):
                cmap = cm.get_cmap(colour_map, len(self.optimal_candidates))
                colors = itertools.cycle(
                    cmap(_) for _ in np.linspace(0, 1, len(self.optimal_candidates))
                )
                for c, cand in enumerate(self.optimal_candidates):
                    opt_spt = self.sampling_times_candidates[cand[0]]
                    if self._pseudo_bayesian:
                        sens = mean_sens[
                                   cand[0],
                                   :,
                                   self.measurable_responses[row],
                                   col
                               ]
                        std = std_sens[
                                  cand[0],
                                  :,
                                  self.measurable_responses[row],
                                  col
                              ]
                    else:
                        sens = self.sensitivities[
                                   cand[0],
                                   :,
                                   self.measurable_responses[row],
                                   col
                               ]
                    color = next(colors)
                    if absolute:
                        sens = np.abs(sens)
                    ax = axes[row, col]
                    ax.plot(
                        opt_spt,
                        sens,
                        linestyle="--",
                        label=f"Candidate {cand[0] + 1:d}",
                        color=color
                    )
                    if not self._specified_n_spt:
                        if self._opt_sampling_times:
                            plot_sens = sens[cand[5]]
                        else:
                            plot_sens = sens[tuple(cand[5])]
                        ax.scatter(
                            cand[3],
                            plot_sens,
                            marker="o",
                            s=markersize * 50 * np.array(cand[4]),
                            color=color,
                            facecolors="none",
                        )
                    else:
                        markers = itertools.cycle(["o", "s", "h", "P"])
                        for i, (eff, spt, spt_idx) in enumerate(zip(cand[4], cand[3], cand[5])):
                            marker = next(markers)
                            ax.scatter(
                                spt,
                                sens[spt_idx],
                                marker=marker,
                                s=markersize * 50 * np.array(eff),
                                color=color,
                                label=f"Variant {i+1}",
                                facecolors="none",
                            )
                    if self._pseudo_bayesian:
                        ax.fill_between(
                            opt_spt,
                            sens + std,
                            sens - std,
                            facecolor=color,
                            alpha=0.1,
                        )
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

                    if row == self.n_m_r - 1:
                        if self.time_unit_name is not None:
                            ax.set_xlabel(f"Time ({self.time_unit_name})")
                        else:
                            ax.set_xlabel('Time')
                    if self.response_names is None or self.model_parameter_names is None:
                        pass
                    else:
                        ylabel = r"$\partial$"
                        ylabel += self.response_names[self.measurable_responses[row]]
                        ylabel += r"/$\partial$"
                        ylabel += self.model_parameter_names[col]
                        if self.response_unit_names is None or self.model_parameter_unit_names is None:
                            pass
                        else:
                            ylabel += f" [({self.response_unit_names[row]})/({self.model_parameter_unit_names[col]})]"
                        ax.set_ylabel(ylabel)
                        # ax.set_ylabel(
                        #     f"$\\partial {self.response_names[self.measurable_responses[row]]}"
                        #     f"/"
                        #     f"\\partial {self.model_parameter_names[col]}$"
                        # )
        if legend and len(self.optimal_candidates) > 1:
            axes[-1, -1].legend()

        fig.tight_layout()

        if write:
            fn = f"sensitivity_plot_{self.oed_result['optimality_criterion']}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)
            self.run_no = 1

        return fig


    def _plot_optimal_sensitivities_interactive(self, figsize=None, markersize=10,
                                                colour_map="jet"):
        if not self._dynamic_system:
            raise SyntaxError("Sensitivity plots are only for dynamic systems.")

        if self.sensitivities is None:
            self.eval_sensitivities()
        if figsize is None:
            figsize = (18, 7)
        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=2,
            ncols=3,
            gridspec_kw={
                "width_ratios": [2, 1, 1],
                "height_ratios": [2, 1],
            }
        )

        for axis_list in axes[:, 1:]:
            for ax in axis_list:
                ax.remove()

        gs = axes[0, 0].get_gridspec()
        res_rad_ax = fig.add_subplot(gs[:, 1])
        mp_rad_ax = fig.add_subplot(gs[:, 2])

        if self.time_unit_name is not None:
            axes[0, 0].set_xlabel(f"Time ({self.time_unit_name})")
        else:
            axes[0, 0].set_xlabel('Time')

        lines = []
        fill_lines = []
        cmap = plt.get_cmap(colour_map)
        colors = itertools.cycle(
            cmap(_)
            for _ in np.linspace(0, 1, len(self.optimal_candidates))
        )

        if self._pseudo_bayesian:
            mean_sens = np.nanmean(
                self._scr_sens,
                axis=0,
            )
            std_sens = np.nanstd(
                self._scr_sens,
                axis=0,
            )

        for opt_c in self.optimal_candidates:
            color = next(colors)
            label = f"Candidate {opt_c[0]+1}"
            if self._pseudo_bayesian:
                line, = axes[0, 0].plot(
                    self.sampling_times_candidates[opt_c[0]],
                    mean_sens[opt_c[0], :, 0, 0],
                    visible=True,
                    label=label,
                    marker="o",
                    markersize=markersize,
                    color=color,
                )
                fill_line = axes[0, 0].fill_between(
                    self.sampling_times_candidates[opt_c[0]],
                    mean_sens[opt_c[0], :, 0, 0] + std_sens[opt_c[0], :, 0, 0],
                    mean_sens[opt_c[0], :, 0, 0] - std_sens[opt_c[0], :, 0, 0],
                    facecolor=color,
                    alpha=0.1,
                    visible=True,
                )
            else:
                line, = axes[0, 0].plot(
                    self.sampling_times_candidates[opt_c[0]],
                    self.sensitivities[opt_c[0], :, 0, 0],
                    visible=True,
                    label=label,
                    marker="o",
                    markersize=markersize,
                    color=color,
                )
            lines.append(line)
            if self._pseudo_bayesian:
                fill_lines.append(fill_line)
            axes[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        labels = [str(line.get_label()) for line in lines]
        visibilities = [line.get_visible() for line in lines]
        cand_check = CheckButtons(
            axes[1, 0],
            labels=labels,
            actives=visibilities,
        )

        def _cand_check(label):
            index = labels.index(label)
            lines[index].set_visible(not lines[index].get_visible())
            if self._pseudo_bayesian:
                fill_lines[index].set_visible(not fill_lines[index].get_visible())
            plt.draw()

        cand_check.on_clicked(_cand_check)

        res_dict = {
            f"{res_name}": i
            for i, res_name in enumerate(self.response_names)
        }
        mp_dict = {
            f"{mp_name}": j
            for j, mp_name in enumerate(self.model_parameter_names)
        }

        res_rad = RadioButtons(
            res_rad_ax,
            labels=[
                f"{res_name}"
                for res_name in self.response_names
            ],
        )

        def _res_rad(label):
            res_idx = res_dict[label]
            mp_idx = mp_dict[mp_rad.value_selected]
            for i, (opt_c, line) in enumerate(zip(self.optimal_candidates, lines)):
                color = next(colors)
                if self._pseudo_bayesian:
                    sens_data = mean_sens[opt_c[0], :, res_idx, mp_idx]
                    fill_lines[i].remove()
                    fill_lines[i] = axes[0, 0].fill_between(
                        self.sampling_times_candidates[opt_c[0]],
                        sens_data + std_sens[opt_c[0], :, res_idx, mp_idx],
                        sens_data - std_sens[opt_c[0], :, res_idx, mp_idx],
                        facecolor=color,
                        alpha=0.1,
                    )
                else:
                    sens_data = self.sensitivities[opt_c[0], :, res_idx, mp_idx]
                line.set_ydata(sens_data)
            axes[0, 0].relim()
            axes[0, 0].autoscale()
            plt.draw()
        res_rad.on_clicked(_res_rad)

        mp_rad = RadioButtons(
            mp_rad_ax,
            labels=[
                f"{mp_name}"
                for mp_name in self.model_parameter_names
            ],
        )

        def _mp_rad(label):
            res_idx = res_dict[res_rad.value_selected]
            mp_idx = mp_dict[label]
            for i, (opt_c, line) in enumerate(zip(self.optimal_candidates, lines)):
                color = next(colors)
                if self._pseudo_bayesian:
                    sens_data = mean_sens[opt_c[0], :, res_idx, mp_idx]
                    fill_lines[i].remove()
                    fill_lines[i] = axes[0, 0].fill_between(
                        self.sampling_times_candidates[opt_c[0]],
                        sens_data + std_sens[opt_c[0], :, res_idx, mp_idx],
                        sens_data - std_sens[opt_c[0], :, res_idx, mp_idx],
                        facecolor=color,
                        alpha=0.1,
                    )
                else:
                    sens_data = self.sensitivities[opt_c[0], :, res_idx, mp_idx]
                line.set_ydata(sens_data)
            axes[0, 0].relim()
            axes[0, 0].autoscale()
            plt.draw()
        mp_rad.on_clicked(_mp_rad)

        fig.tight_layout()
        plt.show()
        return fig


    def _plot_current_efforts_2d(self, tol=1e-4, width=None, write=False, dpi=720,
                                 figsize=None):
        self.get_optimal_candidates(tol=tol)

        if self._verbose >= 2:
            print("Plotting current continuous design.")

        if width is None:
            width = 0.7

        if self.efforts.ndim == 2:
            p_plot = np.array([np.sum(opt_cand[4]) for opt_cand in self.optimal_candidates])
        else:
            p_plot = np.array([opt_cand[4][0] for opt_cand in self.optimal_candidates])

        x = np.array([opt_cand[0]+1 for opt_cand in self.optimal_candidates]).astype(str)
        if figsize is None:
            fig = plt.figure(figsize=(15, 7))
        else:
            fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(111)

        axes.bar(x, p_plot, width=width)

        axes.set_xticks(x)
        axes.set_xlabel("Candidate Number")

        axes.set_ylabel("Optimal Experimental Effort")
        if not self._discrete_design:
            axes.set_ylim([0, 1])
            axes.set_yticks(np.linspace(0, 1, 11))
        else:
            axes.set_ylim([0, self.efforts.max()])
            axes.set_yticks(
                np.linspace(0, self.efforts.max(), self.efforts.max().astype(int))
            )

        if write:
            fn = f"efforts_{self._current_criterion}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)

        fig.tight_layout()
        return fig


    def _plot_current_efforts_3d(self, width=None, write=False, dpi=720, tol=1e-4,
                                 figsize=None):
        self.get_optimal_candidates(tol=tol)

        if self._specified_n_spt:
            print(f"Warning, plot_optimal_efforts not implemented for specified n_spt.")
            return

        if self._verbose >= 2:
            print("Plotting current continuous design.")

        if width is None:
            width = 0.7

        p = self.efforts.reshape([self.n_c, self.n_spt])

        sampling_time_scale: Any = np.nanmin(np.diff(self.sampling_times_candidates, axis=1))

        if figsize is None:
            fig = plt.figure(figsize=(12, 8))
        else:
            fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(111, projection='3d')
        opt_cand = np.unique(np.where(p > tol)[0], axis=0)
        for c, spt in enumerate(self.sampling_times_candidates[opt_cand]):
            x = np.array([c] * self.n_spt) - width / 2
            z = np.zeros(self.n_spt)

            dx = width
            dy = width * sampling_time_scale * width
            dz = p[opt_cand[c], :]

            x = x[~np.isnan(spt)]
            y = spt[~np.isnan(spt)]
            z = z[~np.isnan(spt)]
            dz = dz[~np.isnan(spt)]

            axes.bar3d(
                x=x,
                y=y,
                z=z,
                dx=dx,
                dy=dy,
                dz=dz
            )

        axes.grid(False)
        axes.set_xlabel('Candidate')
        xticks = opt_cand + 1
        axes.set_xticks(
            [c for c, _ in enumerate(self.sampling_times_candidates[opt_cand])])
        axes.set_xticklabels(labels=xticks)

        if self.time_unit_name is not None:
            axes.set_ylabel(f"Sampling Times ({self.time_unit_name})")
        else:
            axes.set_ylabel('Sampling Times')

        axes.set_zlabel('Experimental Effort')
        axes.set_zlim([0, 1])
        axes.set_zticks(np.linspace(0, 1, 6))

        fig.tight_layout()

        if write:
            fn = f'efforts_{self.oed_result["optimality_criterion"]}'
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)
        return fig


