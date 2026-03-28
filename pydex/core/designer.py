from datetime import datetime
from inspect import signature
from os import getcwd, path, makedirs
from pickle import dump, load
from string import Template
from time import time
import itertools
import __main__ as main
import dill
import sys
import corner
from collections.abc import Callable
from typing import Any
from numpy.typing import NDArray

import emcee as mc
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.widgets import RadioButtons, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import minimize, least_squares
from scipy.stats import chi2
from pydex.utils.trellis_plotter import TrellisPlotter
from pydex.core.bnb.tree import Tree
from pydex.core.bnb.node import Node
from pydex.core.oa.manager import OAManager
from pydex.core.oa.primal import OAPrimalProblem
from pydex.core.oa.master import OAMasterProblem
from pydex.core.logger import Logger
from pydex.core._mixin_io import DesignerIO
from pydex.core._mixin_criteria import DesignerCriteria
from pydex.core._mixin_visualization import DesignerVisualization
from pydex.core._mixin_apportionment import DesignerApportionment
from pydex.core._mixin_estimability import DesignerEstimability
from pydex.core._mixin_estimation import DesignerEstimation
from pydex.core._mixin_sensitivity import DesignerSensitivity
from pydex.core._mixin_candidates import DesignerCandidates
from pydex.core._mixin_init import DesignerInit
import matplotlib
import cvxpy as cp
import numdifftools as nd
import numpy as np



class Designer(DesignerIO, DesignerCriteria, DesignerVisualization, DesignerApportionment, DesignerEstimability, DesignerEstimation, DesignerSensitivity, DesignerCandidates, DesignerInit):
    """
    An experiment designer with capabilities to do parameter estimation, parameter
    estimability study, and computes both continuous and exact experimental designs.

    Interfaces to optimization solvers via scipy, and cvxpy. Supports virtually any Python
    functions as long as one can specify the model within the required general syntax.
    Special support for ODE models solved via Pyomo.DAE: allow model, and simulator to
    be passed to the designer to prevent re-build of model and simulator each time a
    the model is simulated, optimizing computational time.

    Designer comes equipped with convenient built-in visualization capabilities, using
    matplotlib.
    """
    def __init__(self):
        """
        Pydex' main class to instantiate an experimental designer. The designer
        serves as the main user-interface to use Pydex to solve experimental design
        problems.

        All details on the experimental design problem is passed to the designer, which
        Pydex then compiles into an optimization problem passed to the optimization
        package it supports to be solved by a numerical optimizer.

        The designer comes with various built-in plotting capabilities through
        matplotlib's plotting features.
        """
        self.__version__: str = "0.0.9"

        """ In Silico Experiments """
        self._bayes_pe_time: float | None = None
        self.insilico_data: Any = None
        self.bayesian_pe_samples: NDArray[np.float64] | None = None

        """ Experimental """
        self._alt_cvar: bool | None = None
        self.error_cov: NDArray[np.float64] | None = None
        self.error_fim: NDArray[np.float64] | None = None

        """ goal_oriented_ds"""
        self.n_c_go: int = 0
        self.n_spt_go: int = 0
        self.n_tic_go: int = 0
        self.n_r_go: int = 0
        self._candidates_swapped: bool = False

        self.go_simulate: Callable[..., NDArray[np.float64]] | None = None
        self.go_tic: NDArray[np.float64] | None = None
        self.go_tvc: Any = None
        self.go_spt: NDArray[np.float64] | None = None
        self.go_sensitivities: Any = None
        self.go_sample_sensitivities_done: bool = False
        self.go_error_cov: NDArray[np.float64] | None = None
        self._step_nom: NDArray[np.float64] | None = None

        """ CVaR-exclusive """
        self.n_cvar_scr: int = 0
        self.cvar_optimal_candidates: list[NDArray[np.float64]] = []
        self.cvar_solution_times: list[Any] = []
        self._biobjective_values: Any = None
        self._constrained_cvar: bool | None = None
        self.beta: Any = None
        self._cvar_problem: bool | None = None

        """ pseudo-Bayesian exclusive """
        self.pb_atomic_fims: Any = None
        self._scr_sens: Any = None
        self.scr_responses: Any = None
        self._current_scr: int = 0
        self._pseudo_bayesian_type: Any = None
        self.scr_fims: Any = None
        self.scr_criterion_val: NDArray[np.float64] | None = None
        self._current_scr_mp: Any = None

        """ Logging """
        # options
        self.sens_report_freq: int = 10
        self._memory_threshold: int | None = None  # threshold for large problems in bytes, default: 1 GB
        # store designer status and its verbal level after initialization
        self._status: str = 'empty'
        self._verbose: int = 0
        self._sensitivity_analysis_done: bool = False

        """ The current optimal experimental design """
        self.opt_eff: NDArray[np.float64] | None = None
        self.opt_tic: NDArray[np.float64] | None = None
        self.n_opt_c: int = 0
        self.mp_covar: NDArray[np.float64] | None = None

        # exclusive to discrete designs
        self.spt_binary: NDArray[np.float64] | None = None

        # exclusive to dynamic systems
        self.opt_tvc: Any = None
        self.opt_spt: NDArray[np.float64] | None = None
        self.opt_spt_combs: NDArray[np.float64] | None = None
        self.spt_candidates_combs: Any = None

        # experimental
        self.cost: Callable[..., float] | None = None
        self.cand_cost: Callable[..., float] | None = None
        self.spt_cost: Callable[..., float] | None = None
        self._norm_sens_by_params: bool = True

        """" Type of Problem """
        self._invariant_controls: bool | None = None
        self._specified_n_spt: bool | None = None
        self._discrete_design: bool | None = None
        self._pseudo_bayesian: bool = False
        self._large_memory_requirement: bool = False
        self._current_criterion: Any = None
        self._efforts_transformed: bool = False
        self._unconstrained_form: bool = False
        self.normalized_sensitivity: Any = None
        self._dynamic_controls: bool = False
        self._dynamic_system: bool = False

        """ Attributes to determine if re-computation of atomics is necessary """
        self._candidates_changed: bool = False
        self._model_parameters_changed: bool = False
        self._compute_atomics: bool = False
        self._compute_sensitivities: bool = False

        """ Core user-defined Variables """
        self._tvcc: Any = None  # NDArray or list of dicts for time-varying controls
        self._ticc: Any = None
        self._sptc: Any = None
        self._model_parameters: Any = None
        self._simulate_signature: int = 0

        # optional user inputs
        self.measurable_responses: Any = None  # subset of measurable states

        """ Labelling """
        self.candidate_names: Any = None  # plotting names
        self.measurable_responses_names: list[str] | None = None
        self.ti_controls_names: list[str] | None = None
        self.tv_controls_names: list[str] | None = None
        self.model_parameters_names: list[str] | None = None
        self.model_parameter_unit_names: list[str] | None = None
        self.response_unit_names: list[str] | None = None
        self.time_unit_name: str | None = None
        self.model_parameter_names: Any = None
        self.response_names: Any = None
        self.use_finite_difference: bool = True
        self.do_sensitivity_analysis: bool = False

        """ Core designer outputs """
        self.response: Any = None
        self.sensitivities: Any = None
        self.optimal_candidates: Any = None
        self.atomic_fims: Any = None
        self.apportionments: Any = None
        self.non_trimmed_apportionments: NDArray[np.intp] | None = None
        self.n_exp: int = 0
        self.epsilon: float | None = None

        # exclusive to prediction-oriented criteria
        self.pvars: Any = None

        """ problem dimension sizes """
        self.n_c: int = 0
        self.n_c_tic: int = 0
        self.n_c_tvc: int = 0
        self.n_c_spt: int = 0
        self.n_tic: int = 0
        self.n_spt: int = 0
        self.n_r: int = 0
        self.n_mp: int = 0
        self.n_e: int = 0
        self.n_m_r: int = 0
        self.n_scr: int = 0
        self.n_spt_comb: int = 0
        self._n_spt_spec: int = 0
        self.max_n_opt_spt: int = 0
        self.n_factor_sups: int = 0

        """ parameter estimation """
        self.data: Any = None  # stored data, 3D array, same shape as response.
        # Whenever data is missing, use np.nan to fill the array.
        self.residuals: NDArray[np.float64] | None = None  # stored residuals, 3D array, same shape as data.
        # Will skip entries whenever data is empty.

        """ performance-related """
        self.feval_simulation: int = 0
        self.feval_sensitivity: int = 0
        self._fim_eval_time: float | None = None
        # temporary for current design
        self._sensitivity_analysis_time: float = 0
        self._optimization_time: float = 0

        """ parameter estimability """
        self.estimable_columns: NDArray[np.intp] | None = None
        self.responses_scales: Any = None
        self.estimability: NDArray[np.float64] | None = None
        self.estimable_model_parameters: Any = []

        """ continuous oed-related quantities """
        # sensitivities
        self.efforts: Any = None
        self.F: NDArray[np.float64] | None = None  # overall regressor matrix
        self.fim: Any = None  # information matrix for current design
        self.p_var: NDArray[np.float64] | None = None  # prediction covariance matrix

        """ saving, loading attributes """
        # current oed result
        self.run_no: int = 1
        self.oed_result: Any = None
        self.result_dir_daily: Any = None
        self.result_dir: Any = None

        """ plotting attributes """
        self.grid: Any = None  # storing grid when create_grid method is used

        """ [Private]: current candidate within eval_sensitivities() """
        self._current_tic: NDArray[np.float64] | None = None
        self._current_tvc: Any = None
        self._current_spt: NDArray[np.float64] | None = None
        self._current_res: Any = None

        """ User-specified Behaviour """
        # problem types
        self._sensitivity_is_normalized: bool | None = None
        self._opt_sampling_times: bool = False
        self._var_n_sampling_time: bool | None = None
        # numerical options
        self._regularize_fim: bool | None = None
        self._num_steps: int = 5
        self._eps: float = 1e-5
        self._trim_fim: bool = False
        self._fd_jac: bool = True
        self._store_responses_rtol: float = 1e-5
        self._store_responses_atol: float = 1e-8

        # store chosen package to interface with the optimizer, and the chosen optimizer
        self._optimization_package: str | None = None
        self._optimizer: str | None = None

        # store current criterion value
        self._criterion_value: Any = None
        self.rounded_criterion_value: Any = None

        """ user saving options """
        self._save_sensitivities: bool = False
        self._save_txt: bool = False
        self._save_txt_nc: int = 0
        self._save_txt_fmt: str = '% 7.3e'
        self._save_atomics: bool = False

        """ discrete design options """
        self._discrete_design_solver = None
        self._MIP_solver = None

    def solve_cvar_problem(self, criterion, beta, n_spt=None, n_exp=None,
                           optimize_sampling_times=False, package="cvxpy",
                           optimizer=None, opt_options=None, e0=None, write=False,
                           save_sensitivities=False, fd_jac=True,
                           unconstrained_form=False, trim_fim=False,
                           pseudo_bayesian_type=None, regularize_fim=False,
                           reso=5, plot=False, n_bins=20, tol=1e-4, dpi=360,
                           **kwargs):
        self._current_criterion = criterion.__name__

        if "cvar" not in self._current_criterion:
            raise SyntaxError(
                "Please pass in a valid cvar criterion e.g., cvar_d_opt_criterion."
            )

        # computing number of parameter scenarios that will be considered in CVaR
        self.beta = beta
        self.n_cvar_scr = (1 - self.beta) * self.n_scr
        if self.n_cvar_scr < 1:
            print(
                "[WARNING]: "
                "given n_scr * beta given is smaller than 1, this yields a maximin "
                "design. Please provide a larger number of n_scr if a CVaR design "
                "was desired."
            )
            self.n_cvar_scr = np.ceil(self.n_cvar_scr).astype(int)
        else:
            self.n_cvar_scr = np.floor(self.n_cvar_scr).astype(int)

        # check if given reso is less than 3
        if reso < 3:
            print(
                f"The input reso is given as {reso}; the minimum value of reso is 3. "
                "Continuing with reso = 3."
            )
            reso = 3

        # initializing result lists
        self.cvar_optimal_candidates = []
        self.cvar_solution_times = []
        self._biobjective_values = np.empty((reso, 2))
        if plot:
            figs = []

            def add_fig(cdf, pdf):
                figs.append([cdf, pdf])

        """ Iteration 1: Maximal (Type 1) Mean Design """
        if self._verbose >= 1:
            print(f" CVaR Problem ".center(100, "*"))
            print(f"")
            print(f"[Iteration 1/{reso}]".center(100, "="))
            print(f"Computing the maximal mean design, obtaining the mean UB and CVaR LB"
                  f" in the Pareto Frontier.")
            print(f"")
        self.design_experiment(
            criterion,
            n_spt=n_spt,
            n_exp=n_exp,
            optimize_sampling_times=optimize_sampling_times,
            package=package,
            optimizer=optimizer,
            opt_options=opt_options,
            e0=e0,
            write=False,
            save_sensitivities=save_sensitivities,
            fd_jac=fd_jac,
            unconstrained_form=unconstrained_form,
            trim_fim=trim_fim,
            pseudo_bayesian_type=pseudo_bayesian_type,
            regularize_fim=regularize_fim,
            beta=0.00,
            **kwargs,
        )
        self.beta = beta
        self.get_optimal_candidates()
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol)
        iter_1_efforts = np.copy(self.efforts) / np.sum(self.efforts)
        mean_ub = self._criterion_value
        iter_1_phi = np.copy(self.phi.value)
        if self._verbose >= 1:
            print("")
            print("Computing CVaR of Iteration 1's Solution")

        # computing CVaR of Maximal Type 1 Mean Design
        self.design_experiment(
            criterion,
            n_spt=n_spt,
            n_exp=n_exp,
            optimize_sampling_times=optimize_sampling_times,
            package=package,
            optimizer=optimizer,
            opt_options=opt_options,
            e0=e0,
            write=False,
            save_sensitivities=False,
            fd_jac=fd_jac,
            unconstrained_form=unconstrained_form,
            trim_fim=trim_fim,
            pseudo_bayesian_type=pseudo_bayesian_type,
            regularize_fim=regularize_fim,
            beta=self.beta,
            fix_effort=iter_1_efforts,
            **kwargs,
        )
        cvar_lb = self._criterion_value
        if self._verbose >= 2:
            print(
                    f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds."
                )

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[0, :] = np.array([mean_ub, cvar_lb])
        if self._verbose >= 1:
            print(f"CVaR LB: {cvar_lb}")
            print(f"Mean UB: {mean_ub}")
            print(f"[Iteration 1/{reso} Completed]".center(100, "="))
            print(f"")
        if plot:
            self.phi.value = iter_1_phi
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=1),
                self.plot_criterion_pdf(write=False, iteration=1),
            )

        """ Iteration 2: Maximal CVaR_beta Design """
        if self._verbose >= 1:
            print(f"[Iteration 2/{reso}]".center(100, "="))
            print(f"Computing the maximal CVaR design, obtaining the CVaR UB, and mean "
                  f"LB in the Pareto Frontier.")
            print(f"")
        self.design_experiment(
            criterion,
            n_spt=n_spt,
            n_exp=n_exp,
            optimize_sampling_times=optimize_sampling_times,
            package=package,
            optimizer=optimizer,
            opt_options=opt_options,
            e0=e0,
            write=False,
            save_sensitivities=False,
            fd_jac=fd_jac,
            unconstrained_form=unconstrained_form,
            trim_fim=trim_fim,
            pseudo_bayesian_type=pseudo_bayesian_type,
            regularize_fim=regularize_fim,
            beta=self.beta,
            **kwargs,
        )
        iter2_s = np.copy(self.s.value)
        self.get_optimal_candidates()
        iter_2_efforts = np.copy(self.efforts) / np.sum(self.efforts)
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol)
        iter2_var = self.v.value
        cvar_ub = self._criterion_value

        if self._verbose >= 1:
            print("")
            print("Computing Mean of Iteration 2's Solution")

        self.design_experiment(
            criterion,
            n_spt=n_spt,
            n_exp=n_exp,
            optimize_sampling_times=optimize_sampling_times,
            package=package,
            optimizer=optimizer,
            opt_options=opt_options,
            e0=e0,
            write=False,
            save_sensitivities=False,
            fd_jac=fd_jac,
            unconstrained_form=unconstrained_form,
            trim_fim=trim_fim,
            pseudo_bayesian_type=pseudo_bayesian_type,
            regularize_fim=regularize_fim,
            beta=0.00,
            fix_effort=iter_2_efforts,
            **kwargs,
        )
        self.beta = beta
        mean_lb = self._criterion_value
        if self._verbose >= 2:
            print(
                    f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds."
                )

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[1, :] = np.array([mean_lb, cvar_ub])
        if self._verbose >= 1:
            print(f"CVaR UB: {cvar_ub}")
            print(f"MEAN LB: {mean_lb}")
            print(f"[Iteration 2/{reso} Completed]".center(100, "="))
            print(f"")
        if plot:
            self.v.value = iter2_var
            self.s.value = iter2_s
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=2),
                self.plot_criterion_pdf(write=False, iteration=2),
            )

        """ Iterations 3+: Intermediate Points """
        mean_values = np.linspace(mean_lb, mean_ub, reso)
        mean_values = mean_values[:-1]
        mean_values = mean_values[1:]

        for i, mean in enumerate(mean_values):
            if self._verbose >= 1:
                print(f"[Iteration {i + 3}/{reso}]".center(100, "="))
            self.design_experiment(
                criterion,
                n_spt=n_spt,
                n_exp=n_exp,
                optimize_sampling_times=optimize_sampling_times,
                package=package,
                optimizer=optimizer,
                opt_options=opt_options,
                e0=e0,
                write=False,
                save_sensitivities=False,
                fd_jac=fd_jac,
                unconstrained_form=unconstrained_form,
                trim_fim=trim_fim,
                pseudo_bayesian_type=pseudo_bayesian_type,
                regularize_fim=regularize_fim,
                beta=self.beta,
                min_expected_value=mean,
                **kwargs,
            )
            self.get_optimal_candidates()
            self.cvar_optimal_candidates.append(self.optimal_candidates)
            self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
            self._biobjective_values[i + 2, :] = np.array([mean, self._criterion_value])

            if plot:
                add_fig(
                    self.plot_criterion_cdf(write=False, iteration=i+3),
                    self.plot_criterion_pdf(write=False, iteration=i+3),
                )
            if self._verbose >= 1:
                self.print_optimal_candidates(tol=tol)
                print(f"CVaR: {self._criterion_value}")
                print(f"MEAN: {cp.sum(self.phi).value / self.n_scr}")
                print(f"[Iteration {i + 3}/{reso} Completed]".center(100, "="))
                print(f"")

        # use the same axes.xlim for all plotted cdfs and pdfs
        if plot:
            xlims = []
            for i, fig in enumerate(figs):
                cdf, pdf = fig[0], fig[1]
                xlims.append(cdf.axes[0].get_xlim())
            xlims = np.asarray(xlims)
            for i, fig in enumerate(figs):
                cdf, pdf = fig[0], fig[1]
                cdf.axes[0].set_xlim(xlims[:, 0].min(), xlims[:, 1].max())
                pdf.axes[0].set_xlim(xlims[:, 0].min(), xlims[:, 1].max())
                cdf.tight_layout()
                pdf.tight_layout()
                if write:
                    fn_cdf = f"iter_{i + 1}_cdf_{self.beta}_beta_{self.n_scr}_scr"
                    fp_cdf = self._generate_result_path(fn_cdf, "png")
                    fn_pdf = f"iter_{i + 1}_pdf_{self.beta}_beta_{self.n_scr}_scr"
                    fp_pdf = self._generate_result_path(fn_pdf, "png")
                    cdf.savefig(fp_cdf, dpi=dpi)
                    pdf.savefig(fp_pdf, dpi=dpi)

    def _formulate_cvar_problem(self, criterion, beta, p_cons, min_expected_value=None):
        self.v = cp.Variable()
        self.s = cp.Variable((self.n_scr,), nonneg=True)
        self.phi = cp.Variable((self.n_scr,))
        self.phi_mean = cp.Variable()

        self.eval_fim(self.efforts)

        for scr, (s_q, phi_q, mp, fim) in enumerate(zip(self.s, self.phi, self.model_parameters, self.scr_fims)):
            p_cons += [phi_q <= -criterion(fim)]
            p_cons += [s_q >= self.v - phi_q]
        if min_expected_value is not None:
            self._constrained_cvar = True
            p_cons += [cp.sum(self.phi) / self.n_scr >= min_expected_value]
        else:
            self._constrained_cvar = False
        obj = cp.Maximize(self.v - 1 / (self.n_scr * (1 - beta)) * cp.sum(self.s))
        return obj

    def solve_cvar_problem_alt(self, criterion, beta, n_spt=None, n_exp=None,
                           optimize_sampling_times=False, package="cvxpy",
                           optimizer=None, opt_options=None, e0=None, write=True,
                           save_sensitivities=False, fd_jac=True,
                           unconstrained_form=False, trim_fim=False,
                           pseudo_bayesian_type=None, regularize_fim=False,
                           reso=5, plot=False, n_bins=20, tol=1e-4, **kwargs):
        self._current_criterion = criterion.__name__

        if "cvar" not in self._current_criterion:
            raise SyntaxError(
                "Please pass in a valid cvar criterion e.g., cvar_d_opt_criterion."
            )

        # computing number of parameter scenarios that will be considered in CVaR
        self.n_cvar_scr = (1 - beta) * self.n_scr
        if self.n_cvar_scr < 1:
            print(
                "[WARNING]: "
                "given n_scr * beta given is smaller than 1, this yields a maximin "
                "design. Please provide a larger number of n_scr if a CVaR design "
                "was desired."
            )
            self.n_cvar_scr = np.ceil(self.n_cvar_scr).astype(int)
        else:
            self.n_cvar_scr = np.floor(self.n_cvar_scr).astype(int)

        # check if given reso is less than 3
        if reso < 3:
            print(
                f"The input reso is given as {reso}; the minimum value of reso is 3. "
                "Continuing with reso = 3."
            )
            reso = 3

        # initializing result lists
        self.cvar_optimal_candidates = []
        self.cvar_solution_times = []
        self._biobjective_values = np.empty((reso, 2))
        if plot:
            figs = []

            def add_fig(cdf, pdf):
                figs.append([cdf, pdf])

        self._alt_cvar = True
        """ Iteration 1: Maximal (Type 1) Mean Design """
        if self._verbose >= 1:
            print(f" CVaR Problem ".center(100, "*"))
            print(f"")
            print(f"[Iteration 1/{reso}]".center(100, "="))
            print(f"Computing the maximal mean design, obtaining the mean UB and CVaR LB"
                  f" in the Pareto Frontier.")
            print(f"")
        self.design_experiment(
            criterion,
            n_spt=n_spt,
            n_exp=n_exp,
            optimize_sampling_times=optimize_sampling_times,
            package=package,
            optimizer=optimizer,
            opt_options=opt_options,
            e0=e0,
            write=False,
            save_sensitivities=False,
            fd_jac=fd_jac,
            unconstrained_form=unconstrained_form,
            trim_fim=trim_fim,
            pseudo_bayesian_type=pseudo_bayesian_type,
            regularize_fim=regularize_fim,
            min_expected_value=-1000,
        )
        self.get_optimal_candidates()
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol, write=False)
        iter_1_efforts = np.copy(self.efforts)
        mean_ub = self._criterion_value
        iter_1_phi = np.copy(self.phi.value)
        if self._verbose >= 1:
            print("")
            print("Computing CVaR of Iteration 1's Solution")
        cvar_lb = (self.v - 1 / (self.n_scr * (1 - beta)) * cp.sum(self.s)).value
        if self._verbose >= 2:
            print(
                    f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds."
                )

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[0, :] = np.array([mean_ub, cvar_lb])
        if self._verbose >= 1:
            print(f"CVaR LB: {cvar_lb}")
            print(f"Mean UB: {mean_ub}")
            print(f"[Iteration 1/{reso} Completed]".center(100, "="))
            print(f"")
        if plot:
            self.phi.value = iter_1_phi
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=1),
                self.plot_criterion_pdf(write=False, iteration=1),
            )

        """ Iteration 2: Maximal CVaR_beta Design """
        if self._verbose >= 1:
            print(f"[Iteration 2/{reso}]".center(100, "="))
            print(f"Computing the maximal CVaR design, obtaining the CVaR UB, and mean "
                  f"LB in the Pareto Frontier.")
            print(f"")
        self.design_experiment(
            criterion,
            n_spt=n_spt,
            n_exp=n_exp,
            optimize_sampling_times=optimize_sampling_times,
            package=package,
            optimizer=optimizer,
            opt_options=opt_options,
            e0=e0,
            write=False,
            save_sensitivities=False,
            fd_jac=fd_jac,
            unconstrained_form=unconstrained_form,
            trim_fim=trim_fim,
            pseudo_bayesian_type=pseudo_bayesian_type,
            regularize_fim=regularize_fim,
            beta=self.beta if self.beta is not None else 0.90,
        )
        self.get_optimal_candidates()
        iter_2_efforts = np.copy(self.efforts)
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol, write=False)
        iter2_var = self.v.value
        cvar_ub = self._criterion_value

        if self._verbose >= 1:
            print("")
            print("Computing Mean of Iteration 2's Solution")

        self.design_experiment(
            criterion,
            n_spt=n_spt,
            n_exp=n_exp,
            optimize_sampling_times=optimize_sampling_times,
            package=package,
            optimizer=optimizer,
            opt_options=opt_options,
            e0=e0,
            write=False,
            save_sensitivities=False,
            fd_jac=fd_jac,
            unconstrained_form=unconstrained_form,
            trim_fim=trim_fim,
            pseudo_bayesian_type=pseudo_bayesian_type,
            regularize_fim=regularize_fim,
            beta=0.00,
            fix_effort=iter_2_efforts,
        )
        mean_lb = self._criterion_value
        if self._verbose >= 2:
            print(
                    f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds."
                )

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[1, :] = np.array([mean_lb, cvar_ub])
        if self._verbose >= 1:
            print(f"CVaR UB: {cvar_ub}")
            print(f"MEAN LB: {mean_lb}")
            print(f"[Iteration 2/{reso} Completed]".center(100, "="))
            print(f"")
        if plot:
            self.v.value = iter2_var
            self._criterion_value = cvar_ub
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=2),
                self.plot_criterion_pdf(write=False, iteration=2),
            )

        """ Iterations 3+: Intermediate Points """
        mean_values = np.linspace(mean_lb, mean_ub, reso)
        mean_values = mean_values[:-1]
        mean_values = mean_values[1:]

        for i, mean in enumerate(mean_values):
            print(f"[Iteration {i + 3}/{reso}]".center(100, "="))
            self.design_experiment(
                criterion,
                n_spt=n_spt,
                n_exp=n_exp,
                optimize_sampling_times=optimize_sampling_times,
                package=package,
                optimizer=optimizer,
                opt_options=opt_options,
                e0=e0,
                write=False,
                save_sensitivities=False,
                fd_jac=fd_jac,
                unconstrained_form=unconstrained_form,
                trim_fim=trim_fim,
                pseudo_bayesian_type=pseudo_bayesian_type,
                regularize_fim=regularize_fim,
                beta=beta,
                min_expected_value=mean,
            )
            self.get_optimal_candidates()
            self.cvar_optimal_candidates.append(self.optimal_candidates)
            self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
            self._biobjective_values[i + 2, :] = np.array([mean, self._criterion_value])

            if plot:
                add_fig(
                    self.plot_criterion_cdf(write=False, iteration=i+3),
                    self.plot_criterion_pdf(write=False, iteration=i+3),
                )
            if self._verbose >= 1:
                self.print_optimal_candidates(tol=tol, write=False)
                print(f"CVaR: {self._criterion_value}")
                print(f"MEAN: {cp.sum(self.phi).value / self.n_scr}")
                print(f"[Iteration {i + 3}/{reso} Completed]".center(100, "="))
                print(f"")

        # use the same axes.xlim for all plotted cdfs and pdfs
        if plot:
            xlims = []
            for i, fig in enumerate(figs):
                cdf, pdf = fig[0], fig[1]
                xlims.append(cdf.axes[0].get_xlim())
            xlims = np.asarray(xlims)
            for i, fig in enumerate(figs):
                cdf, pdf = fig[0], fig[1]
                cdf.axes[0].set_xlim(xlims[:, 0].min(), xlims[:, 1].max())
                pdf.axes[0].set_xlim(xlims[:, 0].min(), xlims[:, 1].max())

    def _formulate_cvar_problem_alt(self, criterion, beta, p_cons, min_cvar_value=None):
        self.v = cp.Variable()
        self.s = cp.Variable((self.n_scr,), nonneg=True)
        self.phi = cp.Variable((self.n_scr,))
        self.phi_mean = cp.Variable()

        self.eval_fim(self.efforts)

        for scr, (s_q, phi_q, mp, fim) in enumerate(zip(self.s, self.phi, self.model_parameters, self.scr_fims)):
            p_cons += [phi_q <= -criterion(fim)]
            p_cons += [s_q >= self.v - phi_q]
        if min_cvar_value is not None:
            self._constrained_cvar = True
            p_cons += [self.v - 1 / (self.n_scr * (1 - beta)) * cp.sum(self.s) >= min_cvar_value]
        else:
            self._constrained_cvar = False
        obj = cp.Maximize(cp.sum(self.phi) / self.n_scr)
        return obj

    def design_experiment(self, criterion: Callable[..., Any],
                          n_spt: int | None = None, n_exp: int | None = None,
                          optimize_sampling_times: bool = False, package: str = "cvxpy",
                          optimizer: str | None = None,
                          opt_options: dict[str, Any] | None = None,
                          e0: NDArray[np.float64] | None = None, write: bool = False,
                          save_sensitivities: bool = False, fd_jac: bool = True,
                          unconstrained_form: bool = False, trim_fim: bool = False,
                          pseudo_bayesian_type: str | None = None,
                          regularize_fim: bool = False, beta: float = 0.90,
                          min_expected_value: float | None = None,
                          fix_effort: NDArray[np.float64] | None = None,
                          save_atomics: bool = False,
                          discrete_design_solver: str | None = None,
                          assess_potential_gain: bool = False,
                          atol: float | None = None,
                          rtol: float = 1e-3,
                          draw_progress: bool = True,
                          singular_tol: float | None = None,
                          max_iters: float = 1e5,
                          MIP_solver: str | None = None,
                          **kwargs: Any) -> Any:
        # storing user choices
        self._regularize_fim = regularize_fim
        self._optimization_package = package
        self._optimizer = optimizer
        self._opt_sampling_times = optimize_sampling_times
        self._save_sensitivities = save_sensitivities
        self._current_criterion = criterion.__name__
        self._fd_jac = fd_jac
        self._unconstrained_form = unconstrained_form
        self._trim_fim = trim_fim
        self._save_atomics = save_atomics
        self._discrete_design_solver = discrete_design_solver

        """ checking if CVaR problem """
        if "cvar" in self._current_criterion:
            self._cvar_problem = True
            self.beta = beta
        else:
            self._cvar_problem = False

        """ resetting optimal candidates """
        self.optimal_candidates = None

        """ setting verbal behaviour """
        if self._verbose >= 2:
            opt_verbose = True
        else:
            opt_verbose = False

        """ handling problems with defined n_spt """
        if n_spt is not None:
            if not self._dynamic_system:
                raise SyntaxError(
                    f"n_spt specified for a non-dynamic system."
                )
            if not self._opt_sampling_times:
                print(
                    f"[Warning]: n_spt specified, but "
                    f"optimize_sampling_times = False. "
                    f"Overriding, and setting optimize_sampling_times = True."
                )
            self._opt_sampling_times = True
            self._n_spt_spec = n_spt
            if not isinstance(n_spt, int):
                raise SyntaxError(
                    f"Supplied n_spt is a {type(n_exp)}, "
                    f"but \"n_spt\" must be an integer."
                )
            self._specified_n_spt = True
            self.spt_candidates_combs = []
            for spt in self.sampling_times_candidates:
                spt_idx = np.arange(0, len(spt))
                self.spt_candidates_combs.append(
                    list(itertools.combinations(spt_idx, n_spt))
                )
            self.spt_candidates_combs = np.asarray(
                self.spt_candidates_combs
            )
            _, self.n_spt_comb, _ = self.spt_candidates_combs.shape
        else:
            self._specified_n_spt = False
            self._n_spt_spec = 1

        """ determining if discrete design problem """
        if n_exp is not None:
            self._discrete_design = True
            if not isinstance(n_exp, int):
                raise SyntaxError(
                    f"Supplied n_exp is a {type(n_exp)}, "
                    f"but \"n_exp\" must be an integer."
                )
        else:
            self._discrete_design = False

        """ setting default semi-bayes behaviour """
        if self._pseudo_bayesian:
            if pseudo_bayesian_type is None:
                self._pseudo_bayesian_type = 0
            else:
                valid_types = [
                    0, 1,
                    "avg_inf", "avg_crit",
                    "average_information", "average_criterion"
                ]
                if pseudo_bayesian_type in valid_types:
                    self._pseudo_bayesian_type = pseudo_bayesian_type
                else:
                    raise SyntaxError(
                        "Unrecognized pseudo_bayesian criterion type. Valid types: '0' "
                        "for average information, '1' for average criterion."
                    )

        """ force fd_jac for large problems """
        if self._large_memory_requirement and not self._fd_jac:
            print("Warning: analytic Jacobian is specified on a large problem."
                  "Overwriting and continuing with finite differences.")
            self._fd_jac = True

        """ setting default optimizers, and its options """
        if self._optimization_package == "scipy":
            if optimizer is None:
                self._optimizer = "SLSQP"
            if opt_options is None:
                opt_options = {"disp": opt_verbose}
        if self._optimization_package == "cvxpy":
            if optimizer is None:
                self._optimizer = None  # let cvxpy auto-select from installed solvers

        """ deal with unconstrained form """
        if self._optimization_package == "scipy":
            if self._optimizer not in ["COBYLA", "SLSQP", "trust-constr"]:
                if self._verbose >= 2:
                    print(f"Note: {self._optimization_package}'s optimizer "
                          f"{self._optimizer} requires unconstrained form.")
                self._unconstrained_form = True
        if self._optimization_package == "cvxpy":
            if self._unconstrained_form:
                self._unconstrained_form = False
                print("Warning: unconstrained form is not supported by cvxpy; "
                      "continuing normally with constrained form.")

        """ Settings for discrete designs """
        if self._discrete_design:
            if discrete_design_solver is None:
                self._discrete_design_solver = "OA"
                self._MIP_solver = "GUROBI"

        """ main codes """
        if self._verbose >= 1:
            print(" Computing Optimal Experiment Design ".center(100, "#"))
        if self._verbose >= 2:
            print(f"{'Started on':<40}: {datetime.now()}")
            print(f"{'Criterion':<40}: {self._current_criterion}")
            print(f"{'Pseudo-bayesian':<40}: {self._pseudo_bayesian}")
            if self._pseudo_bayesian:
                print(f"{'Pseudo-bayesian Criterion Type':<40}: {self._pseudo_bayesian_type}")
            print(f"{'Dynamic':<40}: {self._dynamic_system}")
            print(f"{'Time-invariant Controls':<40}: {self._invariant_controls}")
            print(f"{'Time-varying Controls':<40}: {self._dynamic_controls}")
            print(f"{'Number of Candidates':<40}: {self.n_c}")
            if self._dynamic_system:
                print(f"{'Number of Sampling Time Choices':<40}: {self.n_spt}")
                print(f"{'Sampling Times Optimized':<40}: {self._opt_sampling_times}")
            if self._pseudo_bayesian:
                print(f"{'Number of Scenarios':<40}: {self.n_scr}")
        """ 
        set initial guess for optimal experimental efforts, if none given, equal 
        efforts for all candidates 
        """
        if e0 is None:
            if self._specified_n_spt:
                e0 = np.ones((self.n_c, self.n_spt_comb)) / (self.n_c * self.n_spt_comb)
            else:
                e0 = np.ones((self.n_c, self.n_spt)) / (self.n_c * self.n_spt)
        else:
            msg = 'Initial guess for effort must be a 2D numpy array.'
            if not isinstance(e0, np.ndarray):
                raise SyntaxError(msg)
            elif e0.ndim != 2:
                raise SyntaxError(msg)
            elif e0.shape[0] != self.n_c:
                raise SyntaxError(
                    f"Error: inconsistent number of candidates provided;"
                    f"number of candidates in e0: {e0.shape[0]},"
                    f"number of candidates from initialization: {self.n_c}."
                )
            if self._specified_n_spt:
                if e0.shape[1] != self.n_spt_comb:
                    raise SyntaxError(
                        f"Error: second dimension of e0 must be {self.n_spt_comb} "
                        f"long, corresponding to n_spt_combs; given is {e0.shape[1]}."
                    )
            else:
                if e0.shape[1] != self.n_spt:
                    raise SyntaxError(
                        f"Error: inconsistent number of sampling times provided;"
                        f"number of sampling times in e0: {e0.shape[1]},"
                        f"number of candidates from initialization: {self.n_spt}."
                    )

        # declare and solve optimization problem
        self._sensitivity_analysis_time = 0
        start = time()
        # solvers
        if self._optimization_package == "scipy":
            if fix_effort is not None:
                raise NotImplementedError(
                    "Fixing effort is not supported for scipy solvers yet."
                )
            if self._discrete_design:
                raise NotImplementedError(
                    "Scipy cannot be used to compute discrete designs."
                )
            if self._unconstrained_form:
                opt_result = minimize(
                    fun=criterion,
                    x0=e0,
                    method=optimizer,
                    options=opt_options,
                    jac=not self._fd_jac,
                )
            else:
                e_bound = [[(0, 1) for _ in eff0] for eff0 in e0]
                if self._specified_n_spt:
                    e_bound = np.asarray(e_bound).reshape((self.n_c * self.n_spt_comb, 2))
                else:
                    e_bound = np.asarray(e_bound).reshape((self.n_c * self.n_spt, 2))
                constraint = [
                    {"type": "eq", "fun": lambda e: sum(e) - 1.0},
                ]
                if self._dynamic_system and not self._opt_sampling_times:
                    raise NotImplementedError(
                        "Scipy solvers only supports optimize_sampling_times=True, "
                        "please use cvxpy solvers for optimize_sampling_times=False."
                    )
                opt_result = minimize(
                    fun=criterion,
                    x0=e0.flatten(),
                    method=optimizer,
                    options=opt_options,
                    constraints=constraint,
                    bounds=e_bound,
                    jac=not self._fd_jac,
                    **kwargs,
                )
            if self._specified_n_spt:
                self.efforts = opt_result.x.reshape((self.n_c, self.n_spt_comb))
            else:
                self.efforts = opt_result.x.reshape((self.n_c, self.n_spt))
            self._efforts_transformed = False
            self._transform_efforts()
            opt_fun = opt_result.fun
        elif self._optimization_package == "cvxpy":
            # optimization variable and initial value
            if self._specified_n_spt:
                self.efforts = cp.Variable((self.n_c, self.n_spt_comb), nonneg=True)
            else:
                self.efforts = cp.Variable((self.n_c, self.n_spt), nonneg=True)
            self.efforts.value = e0
            # constraints and objective
            if self._discrete_design and self._discrete_design_solver == "B&B":
                p_cons = [cp.sum(self.efforts) == n_exp]
            else:
                p_cons = [cp.sum(self.efforts) <= 1]
                if not self._opt_sampling_times:
                    p_cons += [eff == eff[0] for c, eff in enumerate(self.efforts)]
            # cvxpy problem
            if self._cvar_problem:
                if self._alt_cvar:
                    obj = self._formulate_cvar_problem_alt(criterion, beta, p_cons, min_cvar_value=min_expected_value)
                else:
                    obj = self._formulate_cvar_problem(criterion, beta, p_cons, min_expected_value=min_expected_value)
            else:
                obj = cp.Maximize(-criterion(self.efforts))
            if fix_effort is not None:
                p_cons += [self.efforts == fix_effort / fix_effort.sum()]
            problem = cp.Problem(obj, p_cons)
            # solution
            if self._discrete_design and self._discrete_design_solver == "B&B":
                root = Node(self.efforts, problem)
                tree = Tree(root)
                tree._verbose = self._verbose
                opt_node = tree.solve()
                assert opt_node is not None
                self.efforts = opt_node.int_var_val
                opt_fun = opt_node.ub
            if self._discrete_design and self._discrete_design_solver == "OA":
                opt_fun = problem.solve(
                    verbose=opt_verbose,
                    solver=self._optimizer,
                    **kwargs
                )
                self.efforts, ced_efforts = self.efforts.value, self.efforts.value * n_exp
                ced_obj = self.compute_criterion_value(criterion, ced_efforts)
                self._criterion_value = opt_fun
                self.apportion(n_exp)
                # old_verbosity = np.copy(self._verbose)
                # self._verbose = 0
                # self._verbose = old_verbosity
                self.optimal_candidates = None  # clean optimal candidates after apportionment
                apportioned_obj = criterion(self.non_trimmed_apportionments).value
                oasolver = OAManager(criterion, self.eval_fim)
                oasolver.sensitivities = self.sensitivities
                oasolver.atomics = self.atomic_fims
                opt_fun = -oasolver.solve(
                    n_exp=n_exp,
                    y0=self.non_trimmed_apportionments,
                    ced_efforts=ced_efforts,
                    ced_obj=ced_obj,
                    apportioned_effort=self.non_trimmed_apportionments,
                    apportioned_obj_val=apportioned_obj,
                    rtol=rtol,
                    atol=atol,
                    assess_potential_oa_gain=assess_potential_gain,
                    draw_progress=draw_progress,
                    max_iters=max_iters,
                    singular_tol=singular_tol,
                    MIP_solver=MIP_solver,
                )
                self.efforts = oasolver.final_effort
            else:
                opt_fun = problem.solve(
                    verbose=opt_verbose,
                    solver=self._optimizer,
                    **kwargs
                )
                self.efforts = self.efforts.value
        else:
            raise SyntaxError("Unrecognized package; try \"scipy\" or \"cvxpy\".")

        finish = time()

        """ report status and performance """
        self._optimization_time = finish - start - self._sensitivity_analysis_time
        if self._verbose >= 2:
            print(
                f"[Optimization Complete in {self._optimization_time:.2f} s]".center(100, "-")
            )
        if self._verbose >= 1:
            print(
                f"Complete: \n"
                f" ~ sensitivity analysis took {self._sensitivity_analysis_time:.2f} "
                f"CPU seconds.\n"
                f" ~ optimization with {self._optimizer!s} via "
                f"{self._optimization_package} took "
                f"{self._optimization_time:.2f} CPU seconds."
            )
            print("".center(100, "#"))

        """ storing and writing result """
        self._criterion_value = opt_fun
        self.oed_result = {
            "solution_time": finish - start,
            "optimization_time": self._optimization_time,
            "sensitivity_analysis_time": self._sensitivity_analysis_time,
            "optimality_criterion": criterion.__name__,
            "ti_controls_candidates": self.ti_controls_candidates,
            "tv_controls_candidates": self.tv_controls_candidates,
            "model_parameters": self.model_parameters,
            "sampling_times_candidates": self.sampling_times_candidates,
            "optimal_efforts": self.efforts,
            "criterion_value": self._criterion_value,
            "optimization_package": self._optimization_package,
            "optimizer": self._optimizer,
            "pseudo_bayesian": self._pseudo_bayesian,
            "pseudo_bayesian_type": self._pseudo_bayesian_type,
            "optimize_sampling_times": self._opt_sampling_times,
            "regularized": self._regularize_fim,
            "n_spt_spec": self._n_spt_spec,
        }
        if write:
            self.write_oed_result()

        return self.oed_result


