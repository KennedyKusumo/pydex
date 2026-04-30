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

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.widgets import RadioButtons, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import chi2
from pydex.utils.trellis_plotter import TrellisPlotter
from pydex.core.logger import Logger
import matplotlib
import numdifftools as nd
import numpy as np
import pyomo.environ as _pyo

try:
    from pyomo.core.expr.calculus.derivatives import differentiate as _pyomo_differentiate
    import scipy.linalg as _scipy_linalg
    _PYOMO_IFT_AVAILABLE = True
except ImportError:
    _PYOMO_IFT_AVAILABLE = False

try:
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP as _PyomoNLP
    _PYNUMERO_ASL_AVAILABLE = True
except Exception:
    _PYNUMERO_ASL_AVAILABLE = False


class Designer:
    """
    An experiment designer with capabilities to do parameter estimation, parameter
    estimability study, and computes both continuous and exact experimental designs.

    Interfaces to optimization solvers via Pyomo, supporting any solver that Pyomo
    knows about (IPOPT, GLPK, Gurobi, CPLEX, Bonmin, SHOT, etc.).  Supports virtually
    any Python function as the model simulator as long as it follows one of the
    supported signatures (see ``simulate`` below).  Special support for ODE models
    solved via Pyomo.DAE: the model and simulator objects can be passed to the designer
    to prevent re-building them on every sensitivity evaluation, significantly reducing
    computation time.

    Designer comes equipped with convenient built-in visualization capabilities
    using matplotlib, and supports the following design criteria:

        Calibration-oriented (minimise parameter uncertainty):
            D-optimal  — maximises det(FIM),  minimises joint confidence volume
            A-optimal  — minimises trace(FIM^{-1}), minimises total param variance
            E-optimal  — minimises lambda_max(FIM^{-1}), minimises worst direction

        Prediction-oriented (minimise prediction uncertainty at target conditions):
            V-optimal  — minimises trace(W FIM^{-1} W^T) at user-specified dw
            G-optimal  — minimises max prediction variance over a region

        Risk-averse:
            CVaR-D/A/E — conditional value-at-risk variants for robust design

    Quick-start
    -----------
    A minimal example for a static model (no time-varying controls):

    >>> import numpy as np
    >>> from pydex.core.designer import Designer
    >>>
    >>> # 1. Define the simulate function (signature type 1)
    >>> def simulate(ti_controls, model_parameters):
    ...     x   = ti_controls[0]        # single input
    ...     a, b = model_parameters     # two parameters to estimate
    ...     return np.array([a * x + b])
    >>>
    >>> # 2. Build the designer
    >>> d = Designer()
    >>> d.simulate            = simulate
    >>> d.model_parameters    = np.array([2.0, 1.0])   # initial guess
    >>> d.ti_controls_candidates = np.linspace(0, 10, 21).reshape(-1, 1)
    >>> d.initialize()
    >>>
    >>> # 3. Run D-optimal design
    >>> d.design_experiment(d.d_opt_criterion, solver="ipopt")
    >>> d.print_optimal_candidates()

    Simulate function signatures
    ----------------------------
    Pydex recognises five signatures based on the argument names.  Use
    exactly these names — pydex inspects them with ``inspect.signature``:

    Type 1 — static model, time-invariant controls only:
        simulate(ti_controls, model_parameters)

    Type 2 — dynamic model, time-invariant controls + sampling times:
        simulate(ti_controls, sampling_times, model_parameters)

    Type 3 — dynamic model, time-varying controls + sampling times:
        simulate(tv_controls, sampling_times, model_parameters)

    Type 4 — dynamic model, both control types + sampling times:
        simulate(ti_controls, tv_controls, sampling_times, model_parameters)

    Type 5 — dynamic model, sampling times only (no explicit controls):
        simulate(sampling_times, model_parameters)

    In all cases ``model_parameters`` must be present.  The function must
    return a numpy array:
        - Static (types 1): shape (n_responses,)
        - Dynamic (types 2-5): shape (n_spt, n_responses)

    Control variables
    -----------------
    ``ti_controls`` — time-invariant controls
        Settings fixed for the entire duration of an experiment.
        Examples: initial concentration, reactor pressure, feed ratio.
        Set as ``designer.ti_controls_candidates``, shape (n_c, n_tic).

    ``tv_controls`` — time-varying controls
        Settings that vary during an experiment, represented as a flat
        parameter vector whose interpretation (ramp, step, spline, etc.)
        is defined inside the user's simulate function.
        Examples: temperature ramp rate, feed profile knots.
        Set as ``designer.tv_controls_candidates``, shape (n_c, n_tvc).

    ``sampling_times`` — measurement time points
        The times within a dynamic experiment at which measurements are
        taken.  Set as ``designer.sampling_times_candidates``, shape (n_c, n_spt).
        When ``optimize_sampling_times=True`` is passed to design_experiment,
        pydex optimises the effort allocation over both candidates and
        sampling times simultaneously.

    V-optimal design workflow (two-stage)
    --------------------------------------
    V-optimal design targets prediction accuracy at a specific operating
    condition ``dw`` rather than minimising global parameter uncertainty.
    It requires two stages.

    *Stage 1 — Process optimisation (find dw):*
    Solve a user-defined constrained optimisation to find the operating
    condition that maximises (or minimises) a process objective subject
    to process constraints.  This is the condition at which the model
    needs to be most accurate — typically the economically optimal point.

    Set the following attributes before calling
    ``find_optimal_operating_point()``::

            designer.process_objective   = my_objective
            designer.process_constraints = my_constraints   # optional
            designer.dw_sense            = "maximize"       # or "minimize"
            designer.dw_bounds_tic       = [(lb, ub), ...]  # one per ti_control
            designer.dw_bounds_tvc       = [(lb, ub), ...]  # one per tv_control

    ``process_objective`` signature: ``callable(tic, tvc, mp) -> float``
        ``tic`` is a 1-D array of ti_controls, ``tvc`` a 1-D array of
        tv_control parameters, ``mp`` the current model_parameters array.
        Returns the scalar value to minimise or maximise.

    ``process_constraints`` signature:
    ``callable(tic, tvc, mp) -> list of dicts``
        Each dict: ``{"type": "ineq" | "eq", "fun": callable(tic, tvc, mp)}``.
        For ``"ineq"``: ``fun(tic, tvc, mp) >= 0`` means feasible.
        For ``"eq"``: ``fun(tic, tvc, mp) == 0``.
        The constraint structure must be fixed; only values change with x.

    Example Stage 1 setup::

            def my_objective(tic, tvc, mp):
                sol = _solve(tic[0], tic[1], tic[2], mp, np.array([T_FINAL]))
                return float(sol.y[1, 0])   # maximise CB at end of batch

            def my_constraints(tic, tvc, mp):
                def ci_con(tic, tvc, mp):
                    sol = _solve(tic[0], tic[1], tic[2], mp, np.array([T_FINAL]))
                    return CI_MAX - float(sol.y[2, 0])   # CI_final <= CI_MAX
                def jacket_con(tic, tvc, mp):
                    return tic[1] - tic[0]               # Tjacket >= T0
                return [
                    {"type": "ineq", "fun": ci_con},
                    {"type": "ineq", "fun": jacket_con},
                ]

            designer.process_objective   = my_objective
            designer.process_constraints = my_constraints
            designer.dw_sense            = "maximize"
            designer.dw_bounds_tic       = [(45, 75), (50, 85), (0.5, 2.0)]
            designer.dw_bounds_tvc       = []

            dw_tic, dw_tvc = designer.find_optimal_operating_point(
                init_guess = np.array([[60.0, 70.0, 1.0]]),
                optimizer  = "mumps",
            )

    *Stage 2 — V-optimal MBDoE (design experiments):*
    After Stage 1, set ``dw_spt`` — the time point(s) within the optimal
    operating profile at which prediction accuracy is required — then call
    ``design_v_optimal()``.  Note that ``dw_spt`` is a user specification
    (e.g. end of batch), not a degree of freedom; it is distinct from
    ``sampling_times_candidates``, which pydex optimises over::

            designer.dw_spt = np.array([t_final])

            designer.design_v_optimal(
                package               = "ipopt",
                optimizer             = "mumps",
                optimize_sampling_times = True,
            )

    References
    ----------
    Shahmohammadi, A. & McAuley, K.B. (2019). Sequential model-based A- and
    V-optimal design of experiments for building fundamental models of
    pharmaceutical production processes. Computers & Chemical Engineering,
    129, 106504. https://doi.org/10.1016/j.compchemeng.2019.06.029
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
        self.__version__ = "0.0.9"

        """ In Silico Experiments """
        self._bayes_pe_time = None
        self.bayesian_pe_samples = None

        """ Prior experimental information (sequential MBDoE) """
        self._prior_fim          = None   # stored prior FIM (n_mp x n_mp, normalized)
        self._prior_fim_mp       = None   # model_parameters at which _prior_fim was computed
        self._prior_n_exp        = 0      # number of prior experiments (for reporting)


        self.error_cov = None
        self.error_fim = None

        """ goal_oriented_ds"""
        self.n_c_go = None
        self.n_spt_go = None
        self.n_tic_go = None
        self.n_r_go = None
        self._candidates_swapped = False

        self.go_simulate = None
        self.go_tic = None
        self.go_tvc = None
        self.go_spt = None
        self.go_sensitivities = None
        self.go_sample_sensitivities_done = False
        self.go_error_cov = None
        self._step_nom = None

        """ CVaR-exclusive """
        self.n_cvar_scr = None
        self.cvar_optimal_candidates = None
        self.cvar_solution_times = None
        self._biobjective_values = None
        self._constrained_cvar = None
        self.beta = None
        self._cvar_problem = None

        """ pseudo-Bayesian exclusive """
        self.pb_atomic_fims = None
        self._scr_sens = None
        self.scr_responses = None
        self._current_scr = None
        self._pseudo_bayesian_type = None
        self.scr_fims = None
        self.scr_criterion_val = None
        self._current_scr_mp = None

        """ Logging """
        # options
        self.sens_report_freq = 10
        self._memory_threshold = None  # threshold for large problems in bytes, default: 1 GB
        # store designer status and its verbal level after initialization
        self._status = 'empty'
        self._verbose = 0
        self._sensitivity_analysis_done = False

        """ The current optimal experimental design """
        self.opt_eff = None
        self.opt_tic = None
        self.n_opt_c = None
        self.mp_covar = None

        # exclusive to discrete designs
        self.spt_binary = None

        # exclusive to dynamic systems
        self.opt_tvc = None
        self.opt_spt = None
        self.opt_spt_combs = None
        self.spt_candidates_combs = None

        # experimental
        self.cost = None
        self.cand_cost = None
        self.spt_cost = None
        self._norm_sens_by_params = True

        """" Type of Problem """
        self._invariant_controls = None
        self._specified_n_spt = None
        self._discrete_design = None
        self._pseudo_bayesian = False
        self._large_memory_requirement = False
        self._current_criterion = None
        self._efforts_transformed = False
        self._unconstrained_form = False
        self.normalized_sensitivity = None
        self._dynamic_controls = False
        self._dynamic_system = False

        """ Attributes to determine if re-computation of atomics is necessary """
        self._candidates_changed = None
        self._model_parameters_changed = None
        self._compute_atomics = False
        self._compute_sensitivities = False

        """ Core user-defined Variables """
        self._tvcc = None
        self._ticc = None
        self._sptc = None
        self._model_parameters = None
        self._simulate_signature = 0

        # optional user inputs
        self.measurable_responses = None  # subset of measurable states

        """ Labelling """
        self.candidate_names = None  # plotting names
        self.measurable_responses_names = None
        self.ti_controls_names = None
        self.tv_controls_names = None
        self.model_parameters_names = None
        self.model_parameter_unit_names = None
        self.response_unit_names = None
        self.time_unit_name = None
        self.model_parameter_names = None
        self.response_names = None
        self.use_finite_difference = True
        self.do_sensitivity_analysis = False

        # ── Pyomo IFT exact-sensitivity ───────────────────────────────────────
        # Set use_pyomo_ift = True and assign pyomo_model_fn to enable exact
        # parametric sensitivities via the implicit-function theorem computed
        # from Pyomo's symbolic expression tree (no finite differences needed).
        #
        # pyomo_model_fn(ti_controls, model_parameters) must return:
        #   (model, all_vars, all_bodies, t_sorted)
        # where all_vars has the n_mp parameter Vars FIRST (declared as fixed
        # Var, not Param), followed by state variables. t_sorted should contain
        # only the output time points (not the full collocation grid) so the
        # designer snaps to the correct output variable.
        #
        # pyomo_output_var_name: base name(s) of output Var (str or list[str]).
        # None -> auto-detect as first n_m_r state vars after param vars.
        self.use_pyomo_ift         = None   # None = auto-detect in initialize()
        self.pyomo_model_fn        = None
        self.pyomo_output_var_name = None
        self.n_jobs                = None  # None = auto-detect in initialize()

        """ Core designer outputs """
        self.response = None
        self.sensitivities = None
        self.optimal_candidates = None
        self.atomic_fims = None
        self.apportionments = None
        self.non_trimmed_apportionments = None
        self.n_exp = None
        self.epsilon = None

        # exclusive to prediction-oriented criteria
        self.pvars = None

        """ problem dimension sizes """
        self.n_c = None
        self.n_c_tic = None
        self.n_c_tvc = None
        self.n_c_spt = None
        self.n_tic = None
        self.n_spt = None
        self.n_r = None
        self.n_mp = None
        self.n_e = None
        self.n_m_r = None
        self.n_scr = None
        self.n_spt_comb = None
        self._n_spt_spec = None
        self.max_n_opt_spt = None
        self.n_factor_sups = None

        """ performance-related """
        self.feval_simulation = None
        self.feval_sensitivity = None
        self._fim_eval_time = None
        # temporary for current design
        self._sensitivity_analysis_time = 0
        self._optimization_time = 0

        """ continuous oed-related quantities """
        # sensitivities
        self.efforts = None
        self.F = None  # overall regressor matrix
        self.fim = None  # the information matrix for current experimental design
        self.p_var = None  # the prediction covariance matrix

        """ saving, loading attributes """
        # current oed result
        self.run_no = 1
        self.oed_result = None
        self.result_dir_daily = None
        self.result_dir = None

        """ plotting attributes """
        self.grid = None  # storing grid when create_grid method is used to help
        # generate candidates

        """ [Private]: current candidate within eval_sensitivities() """
        self._current_tic = None
        self._current_tvc = None
        self._current_spt = None
        self._current_res = None

        """ User-specified Behaviour """
        # problem types
        self._sensitivity_is_normalized = None
        self._opt_sampling_times = False
        self._var_n_sampling_time = None
        # numerical options
        self._regularize_fim = None
        self._num_steps = 5
        self._eps = 1e-5
        self._trim_fim = False
        self._fd_jac = True
        self._store_responses_rtol = 1e-5
        self._store_responses_atol = 1e-8

        # solver name (Pyomo SolverFactory string, e.g. "ipopt", "bonmin", "glpk")
        self._solver = "ipopt"
        self._fd_jac = True          # always True; gradient strategy is internal

        # store current criterion value
        self._criterion_value = None

        """ user saving options """
        self._save_sensitivities = False
        self._save_txt = False
        self._save_txt_nc = 0
        self._save_txt_fmt = '% 7.3e'
        self._save_atomics = False

        """ V-optimal design: operating point and W matrix
        =====================================================
        These attributes support the two-stage V-optimal MBDoE workflow.

        Stage 1 — Process optimisation (user sets before calling
        find_optimal_operating_point):

            process_objective   : callable(tic, tvc, mp) -> float
                The scalar process objective to optimise.  Returns a value
                to be minimised or maximised depending on dw_sense.
                Example: return predicted CB at end of batch.

            process_constraints : callable(tic, tvc, mp) -> list of dicts
                Returns process constraints in scipy/IPOPT format.
                Each dict: {"type": "ineq"|"eq", "fun": f(tic, tvc, mp)}
                For "ineq": fun >= 0 means feasible.
                Set to None if no constraints beyond box bounds.

            dw_bounds_tic : list of (lb, ub) tuples, length n_tic
                Box bounds on the ti_controls at the operating point.
                Must be provided before calling find_optimal_operating_point.

            dw_bounds_tvc : list of (lb, ub) tuples, length n_tvc
                Box bounds on the tv_controls at the operating point.
                Set to [] if the model has no tv_controls.

            dw_sense : str, "minimize" or "maximize"
                Direction of the process objective optimisation.

        Stage 1 results (set automatically by find_optimal_operating_point,
        fixed thereafter — do not overwrite manually):

            dw_tic : np.ndarray, shape (r_w, n_tic)
                Optimal ti_controls at the operating point(s) of interest.
                r_w > 1 when multiple starting points find distinct optima.

            dw_tvc : np.ndarray, shape (r_w, n_tvc)
                Optimal tv_controls at the operating point(s) of interest.

            _dw_fixed : bool
                Set to True by find_optimal_operating_point. Guards against
                calling design_v_optimal before Stage 1 has been run.

        Stage 2 — V-optimal MBDoE (user sets before calling design_v_optimal):

            dw_spt : np.ndarray, shape (n_spt_dw,)
                Time point(s) within the optimal operating profile at which
                prediction accuracy is required.  This is a user specification
                (e.g. end of batch, critical process transition) — it is NOT
                a degree of freedom for the MBDoE optimisation.
                For non-dynamic models, this attribute is ignored.
                Example: designer.dw_spt = np.array([t_final])

            W : np.ndarray, shape (r_w * n_spt_dw * n_m_r, n_mp)
                Scaled sensitivity matrix evaluated at dw, used in the
                V-optimality criterion J_V = trace(W @ FIM^{-1} @ W^T).
                Computed automatically by _eval_W_matrix() on first call
                to design_v_optimal(). Cached thereafter; set to None or
                pass recompute_W=True to force recomputation (e.g. after
                updating model_parameters in a sequential design loop).
        """
        # user-defined process optimization (Stage 1)
        self.process_objective = None       # callable(tic, tvc, mp) -> scalar
        self.process_constraints = None     # callable(tic, tvc, mp) -> list of
                                            #   {"type": "eq"/"ineq", "fun": f(tic,tvc,mp)}
        self.dw_bounds_tic = None           # list of (lb, ub), length n_tic
        self.dw_bounds_tvc = None           # list of (lb, ub), length n_tvc
        self.dw_sense = "minimize"          # "minimize" or "maximize"

        # results of Stage 1 (fixed once computed)
        self.dw_tic = None                  # shape (r_w, n_tic)
        self.dw_tvc = None                  # shape (r_w, n_tvc)
        self._dw_fixed = False

        # W matrix and associated spt for sensitivity evaluation at dw
        self.dw_spt = None                  # shape (n_spt_dw,) — time points for W eval
        self.W = None                       # shape (r_w * n_spt_dw * n_m_r, n_mp)

    @property
    def model_parameters(self):
        return self._model_parameters

    @model_parameters.setter
    def model_parameters(self, mp):
        self._model_parameters_changed = True
        self._model_parameters = mp

    @property
    def ti_controls_candidates(self):
        return self._ticc

    @ti_controls_candidates.setter
    def ti_controls_candidates(self, ticc):
        self._candidates_changed = True
        self._ticc = ticc

    @property
    def tv_controls_candidates(self):
        return self._tvcc

    @tv_controls_candidates.setter
    def tv_controls_candidates(self, tvcc):
        self._candidates_changed = True
        self._tvcc = tvcc

    @property
    def sampling_times_candidates(self):
        return self._sptc

    @sampling_times_candidates.setter
    def sampling_times_candidates(self, sptc):
        self._candidates_changed = True
        self._sptc = sptc

    @staticmethod
    def detect_sensitivity_analysis_function():
        frame = sys._getframe(1)
        while frame:
            if "numdifftools" in frame.f_code.co_filename:
                return False
            elif frame.f_code.co_name == "eval_sensitivities":
                return True
            frame = frame.f_back
        return False

    """ user-defined methods: must be overwritten by user to work """
    def simulate(self, unspecified):
        raise SyntaxError("Don't forget to specify the simulate function.")

    """ core activity interfaces """
    def initialize(self, verbose=0, memory_threshold=int(1e9)):
        """ check for syntax errors, runs one simulation to determine n_r """

        """ check if simulate function has been specified """
        self._check_stats_framework()
        self._handle_simulate_sig()
        self._get_component_sizes()
        self._check_candidate_lengths()
        self._check_missing_components()

        if self._dynamic_system:
            self._check_var_spt()

        self._initialize_names()

        self._check_memory_req(memory_threshold)

        # ── Auto-configure Pyomo IFT + parallelisation ────────────────────────
        # If the user supplied a pyomo_model_fn but left use_pyomo_ift and
        # n_jobs at their __init__ defaults, flip them on automatically.
        # Explicit user overrides (e.g. use_pyomo_ift=False for FD debugging,
        # or n_jobs=1 to force sequential) are always respected.
        if self.pyomo_model_fn is not None:
            if self.use_pyomo_ift is None:       # not explicitly set → auto-enable
                self.use_pyomo_ift = True
                if verbose >= 1:
                    print("[INFO]: pyomo_model_fn detected — use_pyomo_ift set to True.")
            if self.n_jobs is None:              # not explicitly set → auto-parallelise
                self.n_jobs = -1
                if verbose >= 1:
                    print("[INFO]: pyomo_model_fn detected — n_jobs set to -1 (all cores).")
        # If user never set use_pyomo_ift and no pyomo_model_fn, default to False
        if self.use_pyomo_ift is None:
            self.use_pyomo_ift = False
        # If user never set n_jobs and no pyomo_model_fn, default to 1
        if self.n_jobs is None:
            self.n_jobs = 1
        # ─────────────────────────────────────────────────────────────────────

        if self.error_cov is None:
            print(
                f"[WARNING]: because the error_cov is not given, Pydex defaults to the "
                f"identity matrix of size {self.n_m_r}x{self.n_m_r}.")
            self.error_cov = np.eye(self.n_m_r)
        try:
            self.error_fim = np.linalg.inv(self.error_cov)
        except np.linalg.LinAlgError:
            raise SyntaxError(
                "The provided error covariance is singular, please make sure you "
                "have passed in the correct error covariance."
            )

        self._status = 'ready'
        self._verbose = verbose
        if self._verbose >= 2:
            print("".center(100, "="))
        if self._verbose >= 1:
            print('Initialization complete: designer ready.')
        if self._verbose >= 2:
            print("".center(100, "-"))
            print(f"{'Number of model parameters':<40}: {self.n_mp}")
            print(f"{'Number of candidates':<40}: {self.n_c}")
            print(f"{'Number of responses':<40}: {self.n_r}")
            print(f"{'Number of measured responses':<40}: {self.n_m_r}")
            if self._invariant_controls:
                print(f"{'Number of time-invariant controls':<40}: {self.n_tic}")
            if self._dynamic_system:
                print(f"{'Number of sampling time choices':<40}: {self.n_spt}")
            if self._dynamic_controls:
                print(f"{'Number of time-varying controls':<40}: {self.n_tvc}")
            print(f"{'Covariance of measured responses':<40}: \n {self.error_cov}")
            print(f"{'Pyomo IFT sensitivities':<40}: {self.use_pyomo_ift}")
            print(f"{'Parallel workers (n_jobs)':<40}: {self.n_jobs}")
            print("".center(100, "="))

        return self._status

    def simulate_candidates(self, store_predictions=True,
                            plot_simulation_times=False):
        self.response = None  # resets response every time simulation is invoked
        self.feval_simulation = 0
        time_list = []
        start = time()
        for i, exp in enumerate(
                zip(self.ti_controls_candidates, self.tv_controls_candidates,
                    self.sampling_times_candidates)):
            self._current_tic = exp[0]
            self._current_tvc = exp[1]
            self._current_spt = exp[2][~np.isnan(exp[2])]
            if not self._current_spt.size > 0:
                raise SyntaxError(
                    'One candidate has an empty list of sampling times, please check '
                    'the specified experimental candidates.'
                )

            """ determine if simulation needs to be re-run: if data on time-invariant 
            control variables is missing, will not run """
            cond_1 = np.any(np.isnan(exp[0]))
            if np.any([cond_1]):
                self._current_res = np.nan
            else:
                start = time()
                response = self._simulate_internal(self._current_tic, self._current_tvc,
                                                   self.model_parameters,
                                                   self._current_spt)
                finish = time()
                self.feval_simulation += 1
                self._current_res = response
                time_list.append(finish - start)

            if store_predictions:
                self._store_current_response()
        if plot_simulation_times:
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.plot(time_list)
        if self._verbose >= 3:
            print(f"Completed simulation of all candidates in {time() - start} CPU seconds.")
        return self.response

    def simulate_optimal_candidates(self):
        if self.response is not None:
            overwrite = input("Previously stored responses data detected. "
                              "Running this will overwrite stored responses for the "
                              "optimal candidates. "
                              "Proceed? y: yes, n: no ")
            if not any(entry is overwrite for entry in ['y', 'yes']):
                return
        time_list = []
        for i, exp in enumerate(self.optimal_candidates):
            self._current_tic = exp[1]
            self._current_tvc = exp[2]
            self._current_spt = exp[3][~np.isnan(exp[3])]
            if self._current_spt.size <= 0:
                msg = 'One candidate has an empty list of sampling times, please check ' \
                      '' \
                      '' \
                      '' \
                      '' \
                      'the ' \
                      'specified experimental candidates.'
                raise SyntaxError(msg)

            """ 
            determine if simulation needs to be re-run: 
            if data on time-invariant control variables is missing, will not run 
            """
            cond_1 = np.any(np.isnan(exp[0]))
            if np.any([cond_1]):
                self._current_res = np.nan
            else:
                start = time()
                response = self._simulate_internal(self._current_tic, self._current_tvc,
                                                   self.model_parameters,
                                                   self._current_spt)
                finish = time()
                self.feval_simulation += 1
                self._current_res = response
                time_list.append(finish - start)

    # ------------------------------------------------------------------
    # Prior experimental information — sequential MBDoE support
    # ------------------------------------------------------------------

    def set_prior_fim(self, fim, model_parameters):
        """
        Register a Fisher Information Matrix from previously completed experiments
        (Case A: user already has the FIM, e.g. from an external parameter estimation
        routine that returned a parameter covariance matrix Σ_θ → FIM = Σ_θ⁻¹).

        The FIM is stored normalised at ``model_parameters``.  When
        ``design_experiment()`` is called with different (updated) model parameters,
        pydex automatically rescales the prior FIM to the current normalisation
        before adding it to the candidate FIM sum.

        Parameters
        ----------
        fim : array-like, shape (n_mp, n_mp)
            Fisher Information Matrix accumulated from prior experiments.
            Must be expressed in the **same normalisation convention** that
            pydex uses internally, i.e. each element (i,j) is scaled by
            θᵢ · θⱼ (the product of the nominal parameter values).

            If you have a raw (un-normalised) FIM from an external tool and
            your parameter vector is ``theta``, pass::

                fim_normalised = raw_fim * np.outer(theta, theta)

            If you have a parameter covariance matrix Σ_θ, pass::

                fim_normalised = np.linalg.inv(Σ_θ) * np.outer(theta, theta)

        model_parameters : array-like, shape (n_mp,)
            Parameter values at which ``fim`` was computed.  Used to rescale
            the prior FIM when ``designer.model_parameters`` is updated between
            design rounds.

        Examples
        --------
        >>> # From an external covariance matrix
        >>> theta_est  = np.array([0.45, 52000.0, 0.07, 72000.0])
        >>> sigma_theta = np.diag([0.01, 500.0, 0.005, 300.0]) ** 2
        >>> fim_raw = np.linalg.inv(sigma_theta)
        >>> designer.set_prior_fim(
        ...     fim              = fim_raw * np.outer(theta_est, theta_est),
        ...     model_parameters = theta_est,
        ... )

        See Also
        --------
        set_prior_experiments : Case B — compute FIM from raw experimental conditions.
        clear_prior           : Remove all registered prior information.
        """
        fim = np.asarray(fim, dtype=float)
        mp  = np.asarray(model_parameters, dtype=float).flatten()

        if fim.ndim != 2 or fim.shape[0] != fim.shape[1]:
            raise ValueError(
                f"fim must be a square 2-D array; got shape {fim.shape}."
            )
        if mp.size != fim.shape[0]:
            raise ValueError(
                f"model_parameters length ({mp.size}) must match fim dimension "
                f"({fim.shape[0]})."
            )

        self._prior_fim    = fim.copy()
        self._prior_fim_mp = mp.copy()

        if self._verbose >= 1:
            print(
                f"[set_prior_fim] Prior FIM registered "
                f"({fim.shape[0]}×{fim.shape[1]}, "
                f"computed at θ={np.array2string(mp, precision=4, separator=', ')})."
            )

    def set_prior_experiments(
        self,
        ti_controls,
        model_parameters,
        sampling_times  = None,
        tv_controls     = None,
        n_repeats       = None,
    ):
        """
        Compute and register the Fisher Information Matrix from previously
        completed experiments at **arbitrary** conditions (Case B: the conditions
        do not need to be part of any candidate grid).

        pydex evaluates model sensitivities at each supplied experimental
        condition using the same simulate function and finite-difference /
        Pyomo IFT machinery as for candidate-grid evaluations, then assembles:

            FIM_prior = Σₖ  nₖ · Sₖᵀ · Σ_ε⁻¹ · Sₖ

        where nₖ is the number of repeats at condition k, Sₖ is the
        (n_spt × n_r, n_mp) sensitivity matrix, and Σ_ε is ``designer.error_cov``.

        The result is stored exactly as in :meth:`set_prior_fim` and is
        automatically rescaled when ``designer.model_parameters`` is updated.

        Prerequisites
        -------------
        ``designer.initialize()`` must have been called before this method so
        that the simulate function signature is detected and internal dimensions
        are known.

        Parameters
        ----------
        ti_controls : array-like, shape (n_prior, n_tic)
            Time-invariant controls for each prior experiment.
            For a static (non-dynamic) model this encodes the full experimental
            condition.

        model_parameters : array-like, shape (n_mp,)
            Parameter values at which to evaluate sensitivities (your current
            best estimate after fitting the prior experiments).

        sampling_times : array-like or None
            Shape (n_prior, n_spt) or (n_spt,) for all-same timing.
            Required for dynamic models (``_dynamic_system=True``).
            Pass ``None`` for static models.

        tv_controls : array-like or None
            Shape (n_prior, n_tvc) time-varying controls, or ``None``.

        n_repeats : array-like of int or None
            Number of repeats at each condition, shape (n_prior,).
            ``None`` means each condition was run once.

        Examples
        --------
        Static model — three prior experiments, no sampling times:

        >>> designer.set_prior_experiments(
        ...     ti_controls      = np.array([[55.0, 65.0, 1.0],
        ...                                  [60.0, 70.0, 1.5],
        ...                                  [50.0, 60.0, 0.8]]),
        ...     model_parameters = theta_estimated,
        ... )

        Dynamic model — two prior experiments with per-experiment timing:

        >>> designer.set_prior_experiments(
        ...     ti_controls      = np.array([[55.0, 65.0, 1.0],
        ...                                  [60.0, 70.0, 1.5]]),
        ...     sampling_times   = np.array([[0.25, 0.5, 1.0],
        ...                                  [0.25, 0.75, 1.0]]),
        ...     model_parameters = theta_estimated,
        ... )

        With repeats — first condition run twice:

        >>> designer.set_prior_experiments(
        ...     ti_controls      = np.array([[55.0, 65.0, 1.0],
        ...                                  [60.0, 70.0, 1.5]]),
        ...     sampling_times   = np.array([[0.25, 0.5, 1.0],
        ...                                  [0.25, 0.75, 1.0]]),
        ...     model_parameters = theta_estimated,
        ...     n_repeats        = np.array([2, 1]),
        ... )

        See Also
        --------
        set_prior_fim : Case A — register a FIM directly.
        clear_prior   : Remove all registered prior information.
        """
        if self._status == 'empty':
            raise RuntimeError(
                "designer.initialize() must be called before set_prior_experiments()."
            )

        mp  = np.asarray(model_parameters, dtype=float).flatten()
        tic = np.atleast_2d(np.asarray(ti_controls, dtype=float))
        n_prior = tic.shape[0]

        # --- sampling times ---
        if sampling_times is not None:
            spt_arr = np.atleast_2d(np.asarray(sampling_times, dtype=float))
            if spt_arr.shape[0] == 1 and n_prior > 1:
                spt_arr = np.tile(spt_arr, (n_prior, 1))
        else:
            # static model: use a single dummy time point
            spt_arr = np.zeros((n_prior, 1))

        # --- tv_controls ---
        if tv_controls is not None:
            tvc_arr = np.atleast_2d(np.asarray(tv_controls, dtype=float))
            if tvc_arr.shape[0] == 1 and n_prior > 1:
                tvc_arr = np.tile(tvc_arr, (n_prior, 1))
        else:
            tvc_arr = np.zeros((n_prior, 1))

        # --- repeats ---
        if n_repeats is not None:
            repeats = np.asarray(n_repeats, dtype=float).flatten()
            if repeats.size != n_prior:
                raise ValueError(
                    f"n_repeats length ({repeats.size}) must match number of "
                    f"prior experiments ({n_prior})."
                )
        else:
            repeats = np.ones(n_prior)

        # --- error FIM ---
        if self.error_fim is None:
            error_fim = np.eye(self.n_m_r)
        else:
            error_fim = self.error_fim

        # --- save and temporarily override designer state for sensitivity eval ---
        old_tic  = self._current_tic
        old_tvc  = self._current_tvc
        old_spt  = self._current_spt
        old_scr  = self._current_scr_mp

        _use_pyomo_ift = getattr(self, 'use_pyomo_ift', False)
        if not _use_pyomo_ift:
            step_generator = nd.step_generators.MaxStepGenerator(
                base_step    = 2,
                step_ratio   = 2,
                num_steps    = self._num_steps,
                step_nom     = self._step_nom,
            )
            jacob_fun = nd.Jacobian(
                fun         = self._sensitivity_sim_wrapper,
                step        = step_generator,
                method      = 'forward',
                full_output = False,
            )

        fim_prior = np.zeros((self.n_mp, self.n_mp))
        if self._verbose >= 1:
            print(f"[set_prior_experiments] Computing sensitivities for "
                  f"{n_prior} prior experiment(s)...")

        for k in range(n_prior):
            self._current_tic    = tic[k]
            self._current_tvc    = tvc_arr[k]
            self._current_spt    = spt_arr[k][~np.isnan(spt_arr[k])]
            self._current_scr_mp = mp

            try:
                if _use_pyomo_ift:
                    _, sens_k = self._eval_sensitivities_pyomo_ift(
                        self._current_tic,
                        mp,
                        store_predictions=False,
                    )
                else:
                    sens_k = jacob_fun(mp, False)
                    # reshape to (n_spt, n_mr, n_mp)
                    n_spt_k = self._current_spt.size
                    if len(sens_k.shape) == 3:
                        sens_k = np.moveaxis(sens_k, 1, 2)
                    elif self.n_spt == 1:
                        if self.n_mp == 1:
                            sens_k = sens_k[:, :, np.newaxis]
                        else:
                            sens_k = sens_k.reshape(n_spt_k, self.n_m_r, self.n_mp)
                    else:
                        sens_k = sens_k.reshape(n_spt_k, self.n_m_r, self.n_mp)

            except Exception as exc:
                raise RuntimeError(
                    f"Sensitivity computation failed for prior experiment {k+1}/{n_prior}.\n"
                    f"  ti_controls : {tic[k]}\n"
                    f"  spt         : {self._current_spt}\n"
                    f"  Error       : {exc}"
                ) from exc

            # apply parameter normalisation (same as eval_sensitivities)
            if self._norm_sens_by_params:
                sens_k = sens_k * mp[None, None, :]

            # accumulate FIM: sum over time points and responses
            # sens_k shape: (n_spt, n_mr, n_mp)
            for t in range(sens_k.shape[0]):
                s = sens_k[t]   # (n_mr, n_mp)
                fim_prior += repeats[k] * (s.T @ error_fim @ s)

            if self._verbose >= 2:
                print(f"  [{k+1}/{n_prior}] tic={tic[k]}  "
                      f"FIM contribution rank={int(np.linalg.matrix_rank(fim_prior))}")

        # restore designer state
        self._current_tic    = old_tic
        self._current_tvc    = old_tvc
        self._current_spt    = old_spt
        self._current_scr_mp = old_scr

        self._prior_fim      = fim_prior
        self._prior_fim_mp   = mp.copy()
        self._prior_n_exp    = int(np.sum(repeats))

        if self._verbose >= 1:
            rank = int(np.linalg.matrix_rank(fim_prior))
            print(
                f"[set_prior_experiments] Prior FIM assembled from "
                f"{n_prior} condition(s) / {self._prior_n_exp} experiment(s).  "
                f"FIM rank: {rank}/{self.n_mp}."
            )

    def clear_prior(self):
        """
        Remove all registered prior experimental information.

        Call this to start a completely fresh design round without any
        prior FIM contribution, e.g. when switching to a different model
        or parameter set.

        See Also
        --------
        set_prior_fim          : Register a prior FIM directly (Case A).
        set_prior_experiments  : Compute prior FIM from experimental conditions (Case B).
        """
        self._prior_fim    = None
        self._prior_fim_mp = None
        self._prior_n_exp  = 0
        if self._verbose >= 1:
            print("[clear_prior] Prior FIM cleared.")

    def _get_apportioned_candidates(self):
        app_tic_candidates = []
        app_tvc_candidates = []
        app_spt_candidates = []
        for i, app in enumerate(self.apportionments):
            tic = self.optimal_candidates[i][1]
            tvc = self.optimal_candidates[i][2]
            spt = self.optimal_candidates[i][3]
            for _ in range(int(app)):
                app_tic_candidates.append(tic)
                app_tvc_candidates.append(tvc)
                app_spt_candidates.append(spt)
        app_tic_candidates = np.array(app_tic_candidates)
        app_tvc_candidates = np.array(app_tvc_candidates)
        app_spt_candidates = np.array(app_spt_candidates)
        return app_tic_candidates, app_tvc_candidates, app_spt_candidates

    def solve_cvar_problem(self, criterion, beta, n_spt=None, n_exp=None,
                           optimize_sampling_times=False, solver="ipopt",
                           solver_options=None, e0=None, write=False,
                           save_sensitivities=False, trim_fim=False,
                           pseudo_bayesian_type=None, regularize_fim=False,
                           reso=5, plot=False, n_bins=20, tol=1e-4, dpi=360,
                           **kwargs):
        """
        Solve the bi-objective average-CVaR experimental design problem via the
        epsilon-constraint method using Pyomo.
        """
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

        def _common_kwargs():
            return dict(
                n_spt=n_spt,
                n_exp=n_exp,
                optimize_sampling_times=optimize_sampling_times,
                solver=solver,
                solver_options=solver_options,
                e0=e0,
                write=False,
                trim_fim=trim_fim,
                pseudo_bayesian_type=pseudo_bayesian_type,
                regularize_fim=regularize_fim,
                **kwargs,
            )

        def _phi_values():
            """Per-scenario info values from the last solve, for CDF plotting."""
            if self.pb_atomic_fims is None or self.efforts is None:
                return np.zeros(self.n_scr)
            e_flat = np.asarray(self.efforts).flatten()
            phis = []
            for j in range(self.n_scr):
                atoms_j = self.pb_atomic_fims[j]
                M_j = np.einsum('i,imn->mn', e_flat, atoms_j)
                cv = criterion(M_j)
                if isinstance(cv, tuple): cv = cv[0]
                phis.append(-float(cv))
            return np.array(phis)

        """ Iteration 1: Maximal (Type 1) Mean Design """
        if self._verbose >= 1:
            print(f" CVaR Problem ".center(100, "*"))
            print(f"")
            print(f"[Iteration 1/{reso}]".center(100, "="))
            print(f"Computing the maximal mean design, obtaining the mean UB and CVaR LB"
                  f" in the Pareto Frontier.")
            print(f"")
        self.design_experiment(criterion, beta=0.00, **_common_kwargs())
        self.beta = beta
        self.get_optimal_candidates()
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol)
        iter_1_efforts = np.copy(self.efforts) / np.sum(self.efforts)
        mean_ub = self._criterion_value
        iter_1_phi = _phi_values()
        self._cvar_phi = iter_1_phi          # store for plot methods
        if self._verbose >= 1:
            print("")
            print("Computing CVaR of Iteration 1's Solution")

        # computing CVaR of Maximal Type 1 Mean Design
        self.design_experiment(criterion, beta=self.beta,
                               fix_effort=iter_1_efforts,
                               save_sensitivities=False, **_common_kwargs())
        cvar_lb = self._criterion_value
        if self._verbose >= 2:
            print(f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds.")

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[0, :] = np.array([mean_ub, cvar_lb])
        if self._verbose >= 1:
            print(f"CVaR LB: {cvar_lb}")
            print(f"Mean UB: {mean_ub}")
            print(f"[Iteration 1/{reso} Completed]".center(100, "="))
            print(f"")
        if plot:
            self._cvar_phi = iter_1_phi
            self._cvar_V   = float(np.percentile(iter_1_phi, (1 - beta) * 100))
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
        self.design_experiment(criterion, beta=self.beta,
                               save_sensitivities=False, **_common_kwargs())
        self.get_optimal_candidates()
        iter_2_efforts = np.copy(self.efforts) / np.sum(self.efforts)
        iter_2_phi = _phi_values()
        iter2_V    = float(np.percentile(iter_2_phi, (1 - beta) * 100))
        cvar_ub    = self._criterion_value
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol)
            print("")
            print("Computing Mean of Iteration 2's Solution")

        self.design_experiment(criterion, beta=0.00,
                               fix_effort=iter_2_efforts,
                               save_sensitivities=False, **_common_kwargs())
        self.beta = beta
        mean_lb = self._criterion_value
        if self._verbose >= 2:
            print(f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds.")

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[1, :] = np.array([mean_lb, cvar_ub])
        if self._verbose >= 1:
            print(f"CVaR UB: {cvar_ub}")
            print(f"MEAN LB: {mean_lb}")
            print(f"[Iteration 2/{reso} Completed]".center(100, "="))
            print(f"")
        if plot:
            self._cvar_phi = iter_2_phi
            self._cvar_V   = iter2_V
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=2),
                self.plot_criterion_pdf(write=False, iteration=2),
            )

        """ Iterations 3+: Intermediate Points """
        mean_values = np.linspace(mean_lb, mean_ub, reso)[1:-1]

        for i, mean in enumerate(mean_values):
            if self._verbose >= 1:
                print(f"[Iteration {i + 3}/{reso}]".center(100, "="))
            self.design_experiment(
                criterion, beta=self.beta,
                min_expected_value=mean,
                save_sensitivities=False,
                **_common_kwargs(),
            )
            self.get_optimal_candidates()
            iter_phi = _phi_values()
            self._cvar_phi = iter_phi
            self._cvar_V   = float(np.percentile(iter_phi, (1 - beta) * 100))
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
                print(f"MEAN: {iter_phi.mean():.6f}")
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

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Unified Pyomo solver back-ends
    # ------------------------------------------------------------------

    def _make_pyomo_solver(self, solver_options=None):
        """
        Build a configured Pyomo SolverFactory from self._solver and solver_options.

        For standard AMPL solvers (ipopt, bonmin, cbc, …) options are forwarded
        via ``slvr.options``.

        For ``solver="gams"``, GAMS-specific arguments (``io_options``,
        ``add_options``) are handled at solve-time in ``_solve_pyomo``, not here.
        ``solver_options`` keys that start with ``"gams_"`` are stripped and not
        forwarded since they have no meaning as solver options.

        Special keys (extracted, not forwarded as numeric options):
            ``executable``  : full path to solver binary (AMPL solvers only).
        """
        solver_options = dict(solver_options or {})
        executable = solver_options.pop("executable", None)

        is_gams = (self._solver.lower() == "gams")

        if is_gams:
            slvr = _pyo.SolverFactory("gams")
        elif executable is not None:
            slvr = _pyo.SolverFactory(self._solver, executable=executable)
        else:
            slvr = _pyo.SolverFactory(self._solver)

        if not is_gams:
            defaults = {
                "max_iter"      : 3000,
                "tol"           : 1e-8,
                "acceptable_tol": 1e-6,
            }
            if self._verbose < 2:
                defaults["print_level"] = 0
            else:
                defaults["print_level"] = 5
            merged = {**defaults, **solver_options}
            for key, val in merged.items():
                slvr.options[key] = val

        return slvr

    def _pyomo_solve_kwargs(self, solver_options):
        """
        Extract GAMS solve-time kwargs (io_options, add_options) from
        solver_options when solver="gams".

        For all other solvers returns an empty dict.

        GAMS usage example::

            d.design_experiment(
                criterion      = d.d_opt_criterion,
                solver         = "gams",
                solver_options = {
                    "io_options"  : {"solver": "baron"},
                    "add_options" : [
                        "GAMS_MODEL.optfile = 1;",
                        "$onecho > baron.opt",
                        "MaxTime 1000",
                        "AbsConTol 1e-6",
                        "$offecho",
                    ],
                },
                min_effort = 0.05,
            )
        """
        if self._solver.lower() != "gams":
            return {}
        opts = dict(solver_options or {})
        kwargs = {}
        if "io_options" in opts:
            kwargs["io_options"] = opts["io_options"]
        if "add_options" in opts:
            kwargs["add_options"] = opts["add_options"]
        return kwargs

    def _solve_pyomo(self, criterion, e0, fix_effort, solver_options, **kwargs):
        """
        Solve the continuous-effort design NLP via native Pyomo expressions.

        The FIM is expressed as a linear combination of precomputed atomic FIMs:

            FIM(e) = Σᵢ eᵢ · Aᵢ   (linear in e, Aᵢ are numpy constants)

        For D-optimal, A-optimal, E-optimal and V-optimal criteria the objective
        is expressed entirely as native Pyomo expressions — no ExternalFunction
        or Python callbacks — so the model writes cleanly to a .nl file and works
        with any AMPL-compatible solver (IPOPT, Bonmin, SHOT, etc.).

        For unknown criteria (user-defined) we fall back to a scipy.optimize
        SLSQP solve using the criterion callable directly.

        For MINLP sparsity (min_effort > 0) binary variables are added and
        the problem is handed to a MINLP solver (Bonmin, Couenne, etc.).
        """
        import pyomo.environ as pyo

        n_e     = e0.size
        e0_flat = e0.flatten()
        min_eff = getattr(self, '_min_effort', 0.0) or 0.0
        use_minlp = (min_eff > 0.0)

        # Ensure atomic FIMs are available
        if self.atomic_fims is None:
            self._fd_jac = True
            self._compute_atomics = True
            self.eval_fim(e0)

        A = np.asarray(self.atomic_fims)   # (n_e, n_mp, n_mp)
        n_mp = self.n_mp
        crit_name = getattr(criterion, '__name__', '')

        # Identify criterion type
        crit_name = getattr(criterion, '__name__', '')
        is_d   = 'd_opt'  in crit_name and 'pb' not in crit_name
        is_a   = 'a_opt'  in crit_name and 'pb' not in crit_name
        is_e   = 'e_opt'  in crit_name and 'pb' not in crit_name
        is_v   = 'v_opt'  in crit_name
        is_pb  = self._pseudo_bayesian   # set by design_experiment() before we get here

        # Pseudo-Bayesian problems: criterion callable receives per-scenario FIMs at
        # runtime — cannot be expressed as static Pyomo expressions. Use SLSQP.
        # Also fall back for any criterion not recognised as a native Pyomo type.
        is_native = (is_d or is_a or is_e or is_v) and not is_pb

        # For non-native criteria fall back to scipy SLSQP
        if not is_native:
            return self._solve_scipy_slsqp(
                criterion, e0, fix_effort, solver_options, **kwargs
            )

        # --- build Pyomo model ---
        m = pyo.ConcreteModel()
        m.E   = pyo.RangeSet(0, n_e - 1)
        m.P   = pyo.RangeSet(0, n_mp - 1)

        if use_minlp:
            m.b = pyo.Var(m.E, domain=pyo.Binary)
            m.e = pyo.Var(m.E, domain=pyo.NonNegativeReals, bounds=(0, 1))
            m.sparsity_lb = pyo.Constraint(
                m.E, rule=lambda m, i: m.e[i] >= min_eff * m.b[i])
            m.sparsity_ub = pyo.Constraint(
                m.E, rule=lambda m, i: m.e[i] <= m.b[i])
        else:
            m.e = pyo.Var(m.E, domain=pyo.NonNegativeReals, bounds=(0, 1))

        if fix_effort is not None:
            fixed = (fix_effort / fix_effort.sum()).flatten()
            for i in m.E:
                m.e[i].fix(float(fixed[i]))

        m.sum_con = pyo.Constraint(expr=sum(m.e[i] for i in m.E) == 1.0)

        for i in m.E:
            m.e[i].set_value(float(e0_flat[i]))

        # FIM[j,k] = Σᵢ e[i] * A[i,j,k]  — linear Pyomo expression
        # Store as a dict for reuse in multiple criterion formulations
        fim_expr = {}
        for j in range(n_mp):
            for k in range(n_mp):
                fim_expr[j, k] = sum(
                    float(A[i, j, k]) * m.e[i] for i in m.E
                    if abs(A[i, j, k]) > 1e-30
                )

        # add prior FIM if registered
        if self._prior_fim is not None:
            prior = self._prior_fim.copy()
            if self._current_scr_mp is not None and self._prior_fim_mp is not None:
                if not np.allclose(self._current_scr_mp, self._prior_fim_mp, rtol=1e-10):
                    scale   = self._current_scr_mp / self._prior_fim_mp
                    rescale = np.outer(scale, scale)
                    prior   = prior * rescale
            for j in range(n_mp):
                for k in range(n_mp):
                    if abs(prior[j, k]) > 1e-30:
                        fim_expr[j, k] = fim_expr[j, k] + float(prior[j, k])

        # add Tikhonov regularization eps*I to FIM if requested
        # This mirrors the same regularization applied in eval_fim() and ensures
        # the native Pyomo/IPOPT solve uses the same FIM as the numpy callback path.
        if self._regularize_fim:
            for j in range(n_mp):
                fim_expr[j, j] = fim_expr[j, j] + float(self._eps)

        if is_d:
            # D-optimal: maximise log-det(FIM)
            # Expressed via auxiliary lower-triangular Cholesky factor L:
            #   FIM = L @ L.T,   log-det(FIM) = 2 * Σⱼ log(L[j,j])
            # This is a standard SDP-representable formulation that IPOPT handles
            # natively without any Python callbacks.
            m.L = pyo.Var(m.P, m.P, initialize=0.0)
            # fix upper triangle to zero
            for j in range(n_mp):
                for k in range(j + 1, n_mp):
                    m.L[j, k].fix(0.0)
            # diagonal must be positive
            for j in range(n_mp):
                m.L[j, j].setlb(1e-8)

            # Cholesky constraints: FIM[j,k] = Σ_r L[j,r]*L[k,r]  for k<=j
            def chol_rule(m, j, k):
                if k > j:
                    return pyo.Constraint.Skip
                lhs = fim_expr[j, k]
                rhs = sum(m.L[j, r] * m.L[k, r] for r in range(k + 1))
                return lhs == rhs
            m.chol_con = pyo.Constraint(m.P, m.P, rule=chol_rule)

            # objective: minimise -2*Σⱼ log(L[j,j])
            m.obj = pyo.Objective(
                expr=-2.0 * sum(pyo.log(m.L[j, j]) for j in m.P),
                sense=pyo.minimize,
            )

            # warm-start L from Cholesky of initial FIM
            try:
                FIM0 = sum(float(e0_flat[i]) * A[i] for i in range(n_e))
                if self._prior_fim is not None:
                    FIM0 = FIM0 + prior
                L0 = np.linalg.cholesky(FIM0 + 1e-6 * np.eye(n_mp))
                for j in range(n_mp):
                    for k in range(j + 1):
                        m.L[j, k].set_value(float(L0[j, k]))
            except np.linalg.LinAlgError:
                for j in range(n_mp):
                    m.L[j, j].set_value(1.0)

        elif is_a:
            # A-optimal: minimise trace(FIM⁻¹)
            # Via Schur complement: FIM⁻¹[j,j] = (FIM \ eⱼ)ⱼ
            # Lifted form: minimise Σⱼ t[j]  s.t. [FIM  I; I  diag(t)] >= 0
            # IPOPT-friendly form: auxiliary variables z[j] with constraints
            #   FIM @ z[j] = eⱼ,  t[j] >= z[j][j]
            m.Z = pyo.Var(m.P, m.P, initialize=0.0)  # Z[:,j] = FIM^{-1} e_j
            m.t = pyo.Var(m.P, domain=pyo.NonNegativeReals, initialize=1.0)

            # FIM @ Z[:,j] = I[:,j]  i.e. Σ_k FIM[i,k]*Z[k,j] = delta_{i,j}
            def fz_rule(m, i, j):
                lhs = sum(fim_expr[i, k] * m.Z[k, j] for k in range(n_mp))
                rhs = 1.0 if i == j else 0.0
                return lhs == rhs
            m.fz_con = pyo.Constraint(m.P, m.P, rule=fz_rule)

            # t[j] >= Z[j,j]  (diagonal of FIM^{-1})
            m.t_con = pyo.Constraint(
                m.P, rule=lambda m, j: m.t[j] >= m.Z[j, j]
            )

            m.obj = pyo.Objective(
                expr=sum(m.t[j] for j in m.P),
                sense=pyo.minimize,
            )

            # warm-start
            try:
                FIM0 = sum(float(e0_flat[i]) * A[i] for i in range(n_e))
                if self._prior_fim is not None:
                    FIM0 = FIM0 + prior
                Z0 = np.linalg.inv(FIM0 + 1e-6 * np.eye(n_mp))
                for j in range(n_mp):
                    for k in range(n_mp):
                        m.Z[j, k].set_value(float(Z0[j, k]))
                    m.t[j].set_value(float(Z0[j, j]))
            except np.linalg.LinAlgError:
                pass

        elif is_e:
            # E-optimal: maximise lambda_min(FIM)
            # Lifted form: maximise γ  s.t.  FIM - γ*I >= 0
            # IPOPT-friendly via Cholesky of (FIM - γ*I)
            m.gamma = pyo.Var(domain=pyo.Reals, initialize=0.1)
            m.gamma.setlb(0.0)
            m.L = pyo.Var(m.P, m.P, initialize=0.0)
            for j in range(n_mp):
                for k in range(j + 1, n_mp):
                    m.L[j, k].fix(0.0)
            for j in range(n_mp):
                m.L[j, j].setlb(1e-8)

            # Cholesky of (FIM - gamma*I)
            def echol_rule(m, j, k):
                if k > j:
                    return pyo.Constraint.Skip
                diag_adj = float(-1.0) * m.gamma if j == k else 0.0
                lhs = fim_expr[j, k] + (diag_adj if j == k else 0.0)
                rhs = sum(m.L[j, r] * m.L[k, r] for r in range(k + 1))
                return lhs == rhs
            m.echol_con = pyo.Constraint(m.P, m.P, rule=echol_rule)

            m.obj = pyo.Objective(expr=-m.gamma, sense=pyo.minimize)

        elif is_v:
            # V-optimal: minimise trace(W @ FIM^{-1} @ W.T)
            # Same lifted form as A-optimal but with W weighting
            if self.W is None:
                raise RuntimeError(
                    "V-optimal criterion requires W matrix. "
                    "Call find_optimal_operating_point() first."
                )
            W = np.asarray(self.W)   # (n_pred, n_mp)
            n_pred = W.shape[0]
            m.PRED = pyo.RangeSet(0, n_pred - 1)

            # FIM @ Z = W.T  i.e. solve for Z = FIM^{-1} @ W.T
            m.Z = pyo.Var(m.P, m.PRED, initialize=0.0)
            m.t = pyo.Var(m.PRED, domain=pyo.NonNegativeReals, initialize=1.0)

            def vfz_rule(m, i, q):
                lhs = sum(fim_expr[i, k] * m.Z[k, q] for k in range(n_mp))
                rhs = float(W[q, i])
                return lhs == rhs
            m.vfz_con = pyo.Constraint(m.P, m.PRED, rule=vfz_rule)

            # trace(W @ FIM^{-1} @ W.T) = trace(W @ Z) = Σ_q (W @ Z)_{q,q}
            # = Σ_q Σ_k W[q,k] * Z[k,q]
            m.t_con = pyo.Constraint(
                m.PRED,
                rule=lambda m, q: m.t[q] >= sum(
                    float(W[q, k]) * m.Z[k, q] for k in range(n_mp)
                )
            )

            m.obj = pyo.Objective(
                expr=sum(m.t[q] for q in m.PRED),
                sense=pyo.minimize,
            )

        slvr = self._make_pyomo_solver(solver_options)
        gams_kwargs = self._pyomo_solve_kwargs(solver_options)
        result = slvr.solve(m, tee=(self._verbose >= 2), **gams_kwargs)

        tc = result.solver.termination_condition
        ok_conditions = {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
        }
        if tc not in ok_conditions:
            if self._verbose >= 0:
                print(f"[WARNING] Solver termination: {tc}. "
                      f"Results may be suboptimal.")

        e_opt = np.array([pyo.value(m.e[i]) for i in m.E])
        if self._specified_n_spt:
            self.efforts = e_opt.reshape((self.n_c, self.n_spt_comb))
        else:
            self.efforts = e_opt.reshape((self.n_c, self.n_spt))
        self._efforts_transformed = False

        obj_val = float(pyo.value(m.obj))
        return -obj_val

    def _solve_scipy_slsqp(self, criterion, e0, fix_effort, solver_options, **kwargs):
        """
        Fallback solver for criteria that cannot be expressed as native Pyomo
        expressions (e.g. pseudo-Bayesian, user-defined criteria).
        Uses scipy.optimize.minimize with method='SLSQP'.
        """
        from scipy.optimize import minimize as _sp_minimize

        n_e     = e0.size
        e0_flat = e0.flatten()

        bounds = [(0.0, 1.0)] * n_e
        if fix_effort is not None:
            fixed = (fix_effort / fix_effort.sum()).flatten()
            bounds = [(float(f), float(f)) for f in fixed]

        constraints = [{"type": "eq", "fun": lambda e: np.sum(e) - 1.0}]

        opts = {"ftol": 1e-9, "maxiter": 5000, "disp": self._verbose >= 2}
        if solver_options:
            opts.update({k: v for k, v in solver_options.items()
                         if k in ("ftol", "maxiter", "disp")})

        self._fd_jac = True
        res = _sp_minimize(
            fun=criterion,
            x0=e0_flat,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=opts,
        )

        if not res.success and self._verbose >= 1:
            print(f"[WARNING] SLSQP: {res.message}")

        e_opt = res.x
        if self._specified_n_spt:
            self.efforts = e_opt.reshape((self.n_c, self.n_spt_comb))
        else:
            self.efforts = e_opt.reshape((self.n_c, self.n_spt))
        self._efforts_transformed = False

        return -float(res.fun)

    def _solve_pyomo_operating_point(self, x0, lb_arr, ub_arr, solver_options):
        """
        Solve the operating-point optimisation via PyNumero + cyipopt.

        PyNumero's ExternalGreyBoxBlock allows Python callables to be embedded
        in a Pyomo model without requiring pyomo_ampl.so. libpynumero_ASL.dylib
        is present in the IDAES solver package and supports this path.
        Falls back to scipy SLSQP if PyNumero is unavailable.
        """
        try:
            return self._solve_operating_point_pynumero(
                x0, lb_arr, ub_arr, solver_options
            )
        except Exception:
            return self._solve_operating_point_scipy(
                x0, lb_arr, ub_arr, solver_options
            )

    def _solve_operating_point_scipy(self, x0, lb_arr, ub_arr, solver_options):
        """Scipy SLSQP fallback for operating point optimisation."""
        from scipy.optimize import minimize as _sp_min

        n_tic = self.n_tic if self._invariant_controls else 0
        sign  = -1.0 if self.dw_sense == "maximize" else 1.0
        dr    = self

        def obj(x):
            return sign * float(dr.process_objective(
                x[:n_tic], x[n_tic:], dr.model_parameters))

        raw = []
        if dr.process_constraints is not None:
            raw = dr.process_constraints(x0[:n_tic], x0[n_tic:], dr.model_parameters)

        sp_cons = []
        for c in raw:
            f = c["fun"]
            sp_cons.append({
                "type": c["type"],
                "fun" : lambda x, _f=f: float(
                    _f(x[:n_tic], x[n_tic:], dr.model_parameters))
            })

        bounds = list(zip(
            [float(v) if np.isfinite(v) else None for v in lb_arr],
            [float(v) if np.isfinite(v) else None for v in ub_arr],
        ))

        opts = {"ftol": 1e-8, "maxiter": 3000, "disp": self._verbose >= 2}
        if solver_options:
            opts.update({k: v for k, v in solver_options.items()
                         if k in ("ftol", "maxiter", "disp")})

        res = _sp_min(obj, x0, method="SLSQP",
                      bounds=bounds, constraints=sp_cons, options=opts)

        obj_val = sign * float(res.fun)
        return res.x, obj_val

    def _solve_operating_point_pynumero(self, x0, lb_arr, ub_arr, solver_options):
        """
        Operating point optimisation via PyNumero ExternalGreyBoxBlock + cyipopt.
        This uses libpynumero_ASL.dylib (present in IDAES) rather than pyomo_ampl.so.
        """
        from pyomo.contrib.pynumero.interfaces.external_grey_box import (
            ExternalGreyBoxModel, ExternalGreyBoxBlock,
        )
        from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
            CyIpoptSolver, CyIpoptNLP,
        )
        import pyomo.environ as pyo

        n_tic = self.n_tic if self._invariant_controls else 0
        n_tvc = self.n_tvc if self._dynamic_controls else 0
        n_x   = n_tic + n_tvc
        sign  = -1.0 if self.dw_sense == "maximize" else 1.0
        dr    = self
        h_fd  = np.sqrt(np.finfo(float).eps)

        raw_cons = []
        if dr.process_constraints is not None:
            raw_cons = dr.process_constraints(
                x0[:n_tic], x0[n_tic:], dr.model_parameters
            )
        n_eq  = sum(1 for c in raw_cons if c["type"] == "eq")
        n_ineq = sum(1 for c in raw_cons if c["type"] == "ineq")

        class _OpModel(ExternalGreyBoxModel):
            def input_names(self):
                return [f"x{i}" for i in range(n_x)]
            def equality_constraint_names(self):
                return [f"eq{k}" for k in range(n_eq)]
            def output_names(self):
                return []
            def set_input_values(self_, x):
                self_._x = np.array(x)
            def evaluate_equality_constraints(self_):
                eq_vals = [float(c["fun"](
                    self_._x[:n_tic], self_._x[n_tic:], dr.model_parameters
                )) for c in raw_cons if c["type"] == "eq"]
                return np.array(eq_vals)
            def evaluate_jacobian_equality_constraints(self_):
                import scipy.sparse as sp
                rows, cols, vals = [], [], []
                eq_idx = 0
                for c in raw_cons:
                    if c["type"] != "eq":
                        continue
                    f  = c["fun"]
                    f0 = float(f(self_._x[:n_tic], self_._x[n_tic:], dr.model_parameters))
                    for j in range(n_x):
                        xp = self_._x.copy(); xp[j] += h_fd
                        fp = float(f(xp[:n_tic], xp[n_tic:], dr.model_parameters))
                        rows.append(eq_idx); cols.append(j); vals.append((fp-f0)/h_fd)
                    eq_idx += 1
                return sp.coo_matrix((vals, (rows, cols)), shape=(n_eq, n_x))

        m = pyo.ConcreteModel()
        m.ex = ExternalGreyBoxBlock()
        m.ex.set_external_model(_OpModel())
        m.x = m.ex.inputs

        # objective
        def _obj_expr():
            xv = np.array([pyo.value(m.x[f"x{i}"]) for i in range(n_x)])
            return sign * float(dr.process_objective(
                xv[:n_tic], xv[n_tic:], dr.model_parameters))

        # inequality constraints as regular Pyomo constraints
        for k, c in enumerate(raw_cons):
            if c["type"] == "ineq":
                f = c["fun"]
                def _ineq(m, _f=f):
                    xv = np.array([pyo.value(m.x[f"x{i}"]) for i in range(n_x)])
                    return float(_f(xv[:n_tic], xv[n_tic:], dr.model_parameters)) >= 0
                setattr(m, f"ineq_{k}", pyo.Constraint(rule=_ineq))

        # bounds
        for i in range(n_x):
            v = m.x[f"x{i}"]
            v.set_value(float(x0[i]))
            if np.isfinite(lb_arr[i]): v.setlb(float(lb_arr[i]))
            if np.isfinite(ub_arr[i]): v.setub(float(ub_arr[i]))

        # fall through to scipy if this gets too complex
        raise NotImplementedError("PyNumero path not fully implemented; using scipy.")

    def _solve_pyomo_cvar(self, criterion, beta, e0, min_expected_value,
                          solver_options, **kwargs):
        """
        Solve the CVaR experimental design problem via scipy SLSQP.

        The CVaR objective involves per-scenario FIM evaluations that cannot
        be expressed as native Pyomo expressions (they depend on the criterion
        callable). SLSQP handles this efficiently for moderate n_scr.

        Augmented decision vector: x = [e (n_e),  V (1),  delta (n_scr)]

        Objective (minimise):
            -V + 1/(n_scr*(1-beta)) * sum(delta)

        Constraints:
            sum(e) == 1
            delta_j >= V - phi_j(e)   for j = 0..n_scr-1
            (optional) mean(phi_j) >= min_expected_value
        """
        from scipy.optimize import minimize as _sp_min

        if self._large_memory_requirement:
            raise NotImplementedError(
                "The CVaR solver requires pb_atomic_fims to be stored in memory."
            )

        self.efforts = e0
        self.eval_fim(e0)

        n_e    = e0.size
        n_scr  = self.n_scr
        pb_atomics = self.pb_atomic_fims   # (n_scr, n_e, n_mp, n_mp)

        def _phi(p_flat, scr_idx):
            atoms_j = pb_atomics[scr_idx]
            M_j = np.einsum('i,imn->mn', p_flat, atoms_j)
            cv  = criterion(M_j)
            if isinstance(cv, tuple): cv = cv[0]
            return -float(cv)

        e0_flat = e0.flatten()
        phis0   = np.array([_phi(e0_flat, j) for j in range(n_scr)])
        V0      = float(np.percentile(phis0, (1 - beta) * 100))
        d0      = np.maximum(0.0, V0 - phis0)
        x0_aug  = np.concatenate([e0_flat, [V0], d0])

        coeff = 1.0 / (n_scr * (1.0 - beta))

        def obj(x):
            V     = x[n_e]
            delta = x[n_e + 1:]
            return -V + coeff * np.sum(delta)

        def grad_obj(x):
            g = np.zeros_like(x)
            g[n_e]       = -1.0
            g[n_e + 1:]  =  coeff
            return g

        constraints = []
        # sum(e) == 1
        constraints.append({
            "type": "eq",
            "fun" : lambda x: np.sum(x[:n_e]) - 1.0,
            "jac" : lambda x: np.concatenate([np.ones(n_e), np.zeros(1 + n_scr)]),
        })
        # delta[j] - V + phi_j(e) >= 0
        for j in range(n_scr):
            _j = j
            def _cj(x, __j=_j):
                return x[n_e + 1 + __j] - x[n_e] + _phi(x[:n_e], __j)
            constraints.append({"type": "ineq", "fun": _cj})

        if min_expected_value is not None:
            def _mean_phi(x):
                return np.mean([_phi(x[:n_e], j) for j in range(n_scr)]) - min_expected_value
            constraints.append({"type": "ineq", "fun": _mean_phi})

        lb = np.concatenate([np.zeros(n_e),       [-np.inf],  np.zeros(n_scr)])
        ub = np.concatenate([np.ones(n_e),         [np.inf], np.full(n_scr, np.inf)])
        bounds = list(zip(lb, ub))

        opts = {"ftol": 1e-8, "maxiter": 5000, "disp": self._verbose >= 2}
        if solver_options:
            opts.update({k: v for k, v in solver_options.items()
                         if k in ("ftol", "maxiter", "disp")})

        res = _sp_min(obj, x0_aug, jac=grad_obj,
                      method="SLSQP", bounds=bounds,
                      constraints=constraints, options=opts)

        if not res.success and self._verbose >= 1:
            print(f"[WARNING] CVaR SLSQP: {res.message}")

        e_opt = res.x[:n_e]
        if self._specified_n_spt:
            self.efforts = e_opt.reshape((self.n_c, self.n_spt_comb))
        else:
            self.efforts = e_opt.reshape((self.n_c, self.n_spt))
        self._efforts_transformed = False

        # store CVaR stats for plotting
        V_opt    = res.x[n_e]
        self._cvar_V   = float(V_opt)
        self._cvar_phi = np.array([_phi(e_opt, j) for j in range(n_scr)])

        return -float(res.fun)



    # kept for internal compatibility — now delegates to _solve_pyomo
    def _solve_ipopt(self, criterion, e0, fix_effort, opt_options, **kwargs):
        """Delegate to unified Pyomo solver (kept for internal compatibility)."""
        return self._solve_pyomo(criterion, e0, fix_effort, opt_options, **kwargs)

    def find_optimal_operating_point(self, init_guess, solver="ipopt",
                                      solver_options=None, n_starts=1):
        """
        Stage 1 of V-optimal MBDoE: find the process operating condition(s)
        dw at which the model needs to be most accurate.

        Solves a nonlinear constrained optimisation over the ti_controls and
        tv_controls space via Pyomo.  The objective and constraints are
        user-defined via ``process_objective`` and ``process_constraints``.
        The result is stored in ``dw_tic`` and ``dw_tvc`` and fixed for the
        remainder of the workflow — Stage 2 (design_v_optimal) will use these
        to build the W matrix and target the FIM inversion accordingly.

        This function must be called before ``design_v_optimal()``.

        Parameters
        ----------
        init_guess : array-like, shape (n_x,) or (r_w, n_x)
            Initial guess(es) for [tic | tvc].  If 2-D, each row is solved
            independently and all solutions are stored.

        solver : str
            Pyomo solver name (default ``"ipopt"``).  Any solver registered
            with ``pyo.SolverFactory`` may be used.

        solver_options : dict, optional
            Options forwarded to the solver.  For IPOPT use keys such as
            ``"tol"``, ``"max_iter"``, ``"linear_solver"`` (e.g. ``"ma57"``).

        n_starts : int
            Number of random restarts per operating point (default 1).

        Returns
        -------
        dw_tic : np.ndarray, shape (r_w, n_tic)
        dw_tvc : np.ndarray, shape (r_w, n_tvc)

        Examples
        --------
        >>> designer.find_optimal_operating_point(
        ...     init_guess    = np.array([[T0_guess, Tj_guess, cat_guess]]),
        ...     solver        = "ipopt",
        ...     solver_options = {"tol": 1e-8, "linear_solver": "ma57"},
        ... )
        """
        # --- guards ---
        if self._status != 'ready':
            raise SyntaxError(
                "Designer must be initialized before calling "
                "find_optimal_operating_point(). Call designer.initialize() first."
            )
        if self.process_objective is None:
            raise SyntaxError(
                "process_objective must be set before calling "
                "find_optimal_operating_point()."
            )

        n_tic = self.n_tic if self._invariant_controls else 0
        n_tvc = self.n_tvc if self._dynamic_controls   else 0
        n_x   = n_tic + n_tvc

        if n_x == 0:
            raise SyntaxError(
                "No decision variables found. Ensure ti_controls_candidates and/or "
                "tv_controls_candidates are set and designer is initialized."
            )

        # --- build bound arrays ---
        bounds_tic = self.dw_bounds_tic if self.dw_bounds_tic is not None \
            else [(-np.inf, np.inf)] * n_tic
        bounds_tvc = self.dw_bounds_tvc if self.dw_bounds_tvc is not None \
            else [(-np.inf, np.inf)] * n_tvc
        all_bounds = list(bounds_tic) + list(bounds_tvc)
        lb_arr = np.array([b[0] for b in all_bounds], dtype=float)
        ub_arr = np.array([b[1] for b in all_bounds], dtype=float)

        # --- normalise init_guess to 2-D ---
        init_guess = np.atleast_2d(init_guess)   # shape (r_w, n_x)
        r_w = init_guess.shape[0]

        if init_guess.shape[1] != n_x:
            raise SyntaxError(
                f"init_guess has {init_guess.shape[1]} columns but "
                f"n_tic + n_tvc = {n_x}. Each row must be [tic | tvc]."
            )

        # store solver choice
        old_solver      = self._solver
        self._solver    = solver

        results_tic = []
        results_tvc = []
        results_obj = []

        try:
          for w in range(r_w):
            best_x   = None
            best_obj = np.inf

            for start in range(n_starts):
                if start == 0:
                    x0 = init_guess[w].copy()
                else:
                    lo = np.where(np.isfinite(lb_arr), lb_arr, -1e6)
                    hi = np.where(np.isfinite(ub_arr), ub_arr,  1e6)
                    x0 = np.random.uniform(lo, hi)

                if self._verbose >= 1:
                    tag = f"point {w+1}/{r_w}, start {start+1}/{n_starts}"
                    print(f"[find_optimal_operating_point] Solving {tag} ...")

                try:
                    x_opt, obj_val = self._solve_pyomo_operating_point(
                        x0, lb_arr, ub_arr, solver_options
                    )
                except Exception as exc:
                    if self._verbose >= 1:
                        print(f"  Warning: solver failed ({exc}), skipping this start.")
                    continue

                cmp = obj_val if self.dw_sense == "minimize" else -obj_val
                if cmp < best_obj:
                    best_obj = cmp
                    best_x   = x_opt

                if self._verbose >= 1:
                    print(f"  Objective ({self.dw_sense}): {obj_val:.6g}")

            if best_x is None:
                raise RuntimeError(
                    f"All {n_starts} start(s) failed for operating point "
                    f"{w+1}/{r_w} (solver='{solver}'). Check bounds, initial guess, and constraints."
                )

            results_tic.append(best_x[:n_tic])
            results_tvc.append(best_x[n_tic:])
            results_obj.append(
                -best_obj if self.dw_sense == "maximize" else best_obj
            )

            if self._verbose >= 1:
                print(f"  dw_tic[{w}] = {best_x[:n_tic]}")
                print(f"  dw_tvc[{w}] = {best_x[n_tic:]}")

          self.dw_tic      = np.array(results_tic)   # (r_w, n_tic)
          self.dw_tvc      = np.array(results_tvc)   # (r_w, n_tvc)
          self._dw_obj_vals = np.array(results_obj)  # (r_w,) objective at each point
          self._dw_fixed   = True

        finally:
            self._solver = old_solver

        if self._verbose >= 1:
            print(f"[find_optimal_operating_point] Done. "
                  f"{r_w} operating point(s) fixed.")

        return self.dw_tic, self.dw_tvc

    def _solve_cvar_ipopt(self, criterion, beta, e0, min_expected_value,
                          solver_options, **kwargs):
        """Delegate to unified Pyomo CVaR solver (kept for internal compatibility)."""
        return self._solve_pyomo_cvar(
            criterion, beta, e0, min_expected_value, solver_options, **kwargs
        )

    def _formulate_cvar_problem(self, criterion, beta, p_cons, min_expected_value=None):
        """Legacy cvxpy formulation — no longer used. CVaR is handled by _solve_pyomo_cvar."""
        raise NotImplementedError(
            "_formulate_cvar_problem is a legacy cvxpy method. "
            "CVaR problems are now solved via _solve_pyomo_cvar."
        )

    def solve_cvar_problem_alt(self, criterion, beta, n_spt=None, n_exp=None,
                           optimize_sampling_times=False, solver="ipopt",
                           solver_options=None, e0=None, write=True,
                           save_sensitivities=False, trim_fim=False,
                           pseudo_bayesian_type=None, regularize_fim=False,
                           reso=5, plot=False, n_bins=20, tol=1e-4, **kwargs):
        """
        Alternative formulation of the bi-objective average-CVaR design problem
        using Pyomo (maximize mean subject to CVaR constraint).
        """
        self._current_criterion = criterion.__name__

        if "cvar" not in self._current_criterion:
            raise SyntaxError(
                "Please pass in a valid cvar criterion e.g., cvar_d_opt_criterion."
            )

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

        if reso < 3:
            print(
                f"The input reso is given as {reso}; the minimum value of reso is 3. "
                "Continuing with reso = 3."
            )
            reso = 3

        self.cvar_optimal_candidates = []
        self.cvar_solution_times = []
        self._biobjective_values = np.empty((reso, 2))
        if plot:
            figs = []

            def add_fig(cdf, pdf):
                figs.append([cdf, pdf])

        self._alt_cvar = True

        def _common_kwargs():
            return dict(
                n_spt=n_spt,
                n_exp=n_exp,
                optimize_sampling_times=optimize_sampling_times,
                solver=solver,
                solver_options=solver_options,
                e0=e0,
                write=False,
                trim_fim=trim_fim,
                pseudo_bayesian_type=pseudo_bayesian_type,
                regularize_fim=regularize_fim,
                **kwargs,
            )

        def _phi_values():
            if self.pb_atomic_fims is None or self.efforts is None:
                return np.zeros(self.n_scr)
            e_flat = np.asarray(self.efforts).flatten()
            phis = []
            for j in range(self.n_scr):
                atoms_j = self.pb_atomic_fims[j]
                M_j = np.einsum('i,imn->mn', e_flat, atoms_j)
                cv = criterion(M_j)
                if isinstance(cv, tuple): cv = cv[0]
                phis.append(-float(cv))
            return np.array(phis)

        """ Iteration 1: Maximal Mean Design """
        if self._verbose >= 1:
            print(f" CVaR Problem (Alt) ".center(100, "*"))
            print(f"[Iteration 1/{reso}]".center(100, "="))
        self.design_experiment(criterion, min_expected_value=-1000, **_common_kwargs())
        self.get_optimal_candidates()
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol, write=False)
        iter_1_efforts = np.copy(self.efforts)
        mean_ub = self._criterion_value
        iter_1_phi = _phi_values()
        self._cvar_phi = iter_1_phi
        self._cvar_V   = float(np.percentile(iter_1_phi, (1 - beta) * 100))
        # CVaR at iter-1 solution
        self.design_experiment(criterion, beta=beta,
                               fix_effort=iter_1_efforts / np.sum(iter_1_efforts),
                               save_sensitivities=False, **_common_kwargs())
        cvar_lb = self._criterion_value

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[0, :] = np.array([mean_ub, cvar_lb])
        if self._verbose >= 1:
            print(f"CVaR LB: {cvar_lb}  Mean UB: {mean_ub}")
            print(f"[Iteration 1/{reso} Completed]".center(100, "="))
        if plot:
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=1),
                self.plot_criterion_pdf(write=False, iteration=1),
            )

        """ Iteration 2: Maximal CVaR Design """
        if self._verbose >= 1:
            print(f"[Iteration 2/{reso}]".center(100, "="))
        self.design_experiment(criterion, beta=beta,
                               save_sensitivities=False, **_common_kwargs())
        self.get_optimal_candidates()
        iter_2_efforts = np.copy(self.efforts)
        iter_2_phi = _phi_values()
        iter2_V    = float(np.percentile(iter_2_phi, (1 - beta) * 100))
        cvar_ub    = self._criterion_value
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol, write=False)
        self.design_experiment(criterion, beta=0.00,
                               fix_effort=iter_2_efforts / np.sum(iter_2_efforts),
                               save_sensitivities=False, **_common_kwargs())
        mean_lb = self._criterion_value

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[1, :] = np.array([mean_lb, cvar_ub])
        if self._verbose >= 1:
            print(f"CVaR UB: {cvar_ub}  Mean LB: {mean_lb}")
            print(f"[Iteration 2/{reso} Completed]".center(100, "="))
        if plot:
            self._cvar_phi = iter_2_phi
            self._cvar_V   = iter2_V
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=2),
                self.plot_criterion_pdf(write=False, iteration=2),
            )

        """ Iterations 3+: Intermediate Points """
        cvar_values = np.linspace(cvar_lb, cvar_ub, reso)[1:-1]

        for i, cvar_min in enumerate(cvar_values):
            if self._verbose >= 1:
                print(f"[Iteration {i + 3}/{reso}]".center(100, "="))
            self.design_experiment(
                criterion, beta=beta,
                min_expected_value=cvar_min,
                save_sensitivities=False,
                **_common_kwargs(),
            )
            self.get_optimal_candidates()
            iter_phi = _phi_values()
            self._cvar_phi = iter_phi
            self._cvar_V   = float(np.percentile(iter_phi, (1 - beta) * 100))
            self.cvar_optimal_candidates.append(self.optimal_candidates)
            self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
            self._biobjective_values[i + 2, :] = np.array([self._criterion_value, cvar_min])
            if plot:
                add_fig(
                    self.plot_criterion_cdf(write=False, iteration=i+3),
                    self.plot_criterion_pdf(write=False, iteration=i+3),
                )
            if self._verbose >= 1:
                self.print_optimal_candidates(tol=tol, write=False)
                print(f"Mean: {self._criterion_value:.6f}  CVaR constraint: {cvar_min:.6f}")
                print(f"[Iteration {i + 3}/{reso} Completed]".center(100, "="))

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
        """Legacy cvxpy formulation — no longer used. CVaR is handled by _solve_pyomo_cvar."""
        raise NotImplementedError(
            "_formulate_cvar_problem_alt is a legacy cvxpy method. "
            "CVaR problems are now solved via _solve_pyomo_cvar."
        )

    def design_experiment(self, criterion, n_spt=None, n_exp=None,
                          optimize_sampling_times=False, solver="ipopt",
                          solver_options=None, e0=None, write=False,
                          save_sensitivities=False, trim_fim=False,
                          pseudo_bayesian_type=None, regularize_fim=False, beta=0.90,
                          min_expected_value=None, fix_effort=None, save_atomics=False,
                          min_effort=None, **kwargs):
        # storing user choices
        self._regularize_fim = regularize_fim
        self._solver         = solver
        self._fd_jac         = True          # always True; gradient strategy is internal
        self._unconstrained_form = False     # no longer a user concern
        self._opt_sampling_times = optimize_sampling_times
        self._save_sensitivities = save_sensitivities
        self._current_criterion  = criterion.__name__
        self._trim_fim           = trim_fim
        self._save_atomics       = save_atomics
        self._min_effort         = min_effort  # sparsity threshold (MINLP when set)

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

        """ re-check local vs pseudo-Bayesian based on current model_parameters
            (user may have set a 2D scenarios array after initialize() was called
            with a 1D array, so _pseudo_bayesian, n_scr, and n_mp must be refreshed) """
        self._check_stats_framework()
        if self._pseudo_bayesian:
            self.n_scr, self.n_mp = self.model_parameters.shape
            self._current_scr_mp = self.model_parameters[0]
        else:
            self.n_mp = self.model_parameters.shape[0]
            self._current_scr_mp = self.model_parameters

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
            print(f"{'Solver':<40}: {self._solver}")
            if min_effort is not None:
                print(f"{'Min. effort (sparsity)':<40}: {min_effort}")
            if self._prior_fim is not None:
                print(f"{'Prior FIM':<40}: registered  "
                      f"({self._prior_n_exp} prior experiment(s))")
            else:
                print(f"{'Prior FIM':<40}: none")
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

        # single unified Pyomo dispatch
        if self._cvar_problem:
            opt_fun = self._solve_pyomo_cvar(
                criterion, beta, e0, min_expected_value, solver_options, **kwargs
            )
        else:
            opt_fun = self._solve_pyomo(
                criterion, e0, fix_effort, solver_options, **kwargs
            )

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
                f" ~ optimization with {self._solver} took "
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
            "solver": self._solver,
            "pseudo_bayesian": self._pseudo_bayesian,
            "pseudo_bayesian_type": self._pseudo_bayesian_type,
            "optimize_sampling_times": self._opt_sampling_times,
            "regularized": self._regularize_fim,
            "n_spt_spec": self._n_spt_spec,
            "prior_fim": self._prior_fim,
            "prior_fim_mp": self._prior_fim_mp,
            "prior_n_exp": self._prior_n_exp,
        }
        if write:
            self.write_oed_result()

        return self.oed_result

    def plot_criterion_cdf(self, write=False, iteration=None, dpi=360, figsize=(4.5, 3.5), annotate=False, minor_ticks=False, legend=False, grid=False):
        if not self._pseudo_bayesian or not self._cvar_problem:
            raise SyntaxError(
                "Plotting cumulative distribution function only valid for pseudo-"
                "bayesian and cvar problems."
            )

        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(111)
        if self._cvar_problem:
            phi_vals = getattr(self, '_cvar_phi', np.zeros(self.n_scr))
            V_val    = getattr(self, '_cvar_V',   float('nan'))
            x = np.sort(phi_vals)
            mean = phi_vals.mean()
            x = np.insert(x, 0, x[0])
            y = np.linspace(0, 1, x.size)
            axes.plot(x, y, "o--", alpha=0.3, c="#1f77b4")
            axes.plot(x, y, drawstyle="steps-post", c="#1f77b4")
            axes.axvline(
                x=V_val,
                ymin=0,
                ymax=1,
                c="tab:red",
                label=f"VaR {self.beta}",
            )
            axes.axvline(
                x=getattr(self, "_criterion_value", float("nan")),
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
                    xy=(V_val, 0.80),
                    xytext=(V_val + 0.2 * np.abs(V_val), 0.80),
                    arrowprops={
                        "width": 5,
                        "shrink": 0.05,
                        "facecolor": "tab:red",
                        "edgecolor": "k",
                    },
                )
                cvar = getattr(self, "_criterion_value", float("nan"))
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
            x     = getattr(self, '_cvar_phi', np.zeros(self.n_scr))
            V_val = getattr(self, '_cvar_V',   float('nan'))
            axes.hist(x, bins=n_bins)
            axes.axvline(V_val, 0, 1, c="tab:red",   label=f"VaR {self.beta}")
            axes.axvline(self._criterion_value, 0, 1, c="tab:green", label=f"CVaR {self.beta}")
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

    def compute_criterion_value(self, criterion, decimal_places=3):
        crit_val = criterion(self.efforts)
        if isinstance(crit_val, tuple):
            crit_val = crit_val[0]
        crit_val = float(np.squeeze(crit_val))
        if self._verbose >= 1:
            print(f"{criterion.__name__}: {crit_val:.{decimal_places}E}")
        return crit_val

    def apportion(self, n_exp, method="adams", trimmed=True, compute_actual_efficiency=True):
        self.n_exp = n_exp

        if self._dynamic_system and self._specified_n_spt:
            print(
                "[WARNING]: The apportion method does not support experimental design "
                "problems with specified n_spt yet. Skipping the apportionment."
            )
            return
        _original_save_atomics = np.copy(self._save_atomics)
        self._save_atomics = False
        self.get_optimal_candidates()

        """ Initialize opt_eff shape """
        if self._opt_sampling_times:
            self.opt_eff = np.empty((len(self.optimal_candidates), self.max_n_opt_spt))
        else:
            self.opt_eff = np.empty((len(self.optimal_candidates)))
        self.opt_eff[:] = np.nan

        """ Get the optimal efforts from optimal_candidates """
        for i, opt_cand in enumerate(self.optimal_candidates):
            if self._opt_sampling_times:
                for j, spt in enumerate(opt_cand[4]):
                    if self._specified_n_spt:
                        self.opt_eff[i, j] = np.nansum(spt)
                    else:
                        self.opt_eff[i, j] = spt
            else:
                self.opt_eff[i] = np.nansum(opt_cand[4])

        """ do the apportionment """
        if method == "adams":
            if n_exp < self.n_factor_sups:
                self.apportionments = self._greatest_effort_apportionment(self.opt_eff, n_exp)
            else:
                self.apportionments = self._adams_apportionment(self.opt_eff, n_exp)
        else:
            raise NotImplementedError(
                "At the moment, the only method implemented is 'adams', please use it. "
                "More apportionment methods will be implemented, but there is proof "
                "that Adam's method is the most efficient amongst other popular "
                "methods used in electoral college apportionments."
            )

        """ Report the obtained apportionment """
        if self._verbose >= 1:
            print(f" Optimal Experiment for {n_exp:d} Runs ".center(100, "#"))
            print(f"{'Obtained on':<40}: {datetime.now()}")
            print(f"{'Criterion':<40}: {self._current_criterion}")
            print(f"{'Criterion Value':<40}: {self._criterion_value}")
            print(f"{'Pseudo-bayesian':<40}: {self._pseudo_bayesian}")
            if self._pseudo_bayesian:
                print(f"{'Pseudo-bayesian Criterion Type':<40}: {self._pseudo_bayesian_type}")
            print(f"{'CVaR Problem':<40}: {self._cvar_problem}")
            if self._cvar_problem:
                print(f"{'Beta':<40}: {self.beta}")
                print(f"{'Constrained Problem':<40}: {self._constrained_cvar}")
                if self._constrained_cvar:
                    print(f"{'Min. Mean Value':<40}: {getattr(self, "_cvar_mean_phi", float("nan")):.6f}")
            print(f"{'Dynamic':<40}: {self._dynamic_system}")
            print(f"{'Time-invariant Controls':<40}: {self._invariant_controls}")
            print(f"{'Time-varying Controls':<40}: {self._dynamic_controls}")
            print(f"{'Number of Candidates':<40}: {self.n_c}")
            print(f"{'Number of Optimal Candidates':<40}: {self.n_opt_c}")
            if self._dynamic_system:
                print(f"{'Number of Sampling Time Choices':<40}: {self.n_spt}")
                print(f"{'Sampling Times Optimized':<40}: {self._opt_sampling_times}")
                if self._opt_sampling_times:
                    print(f"{'Number of Samples Per Experiment':<40}: {self._n_spt_spec}")
            if self._pseudo_bayesian:
                print(f"{'Number of Scenarios':<40}: {self.n_scr}")

            for i, (app_eff, opt_cand) in enumerate(zip(self.apportionments, self.optimal_candidates)):
                print(f"{f'[Candidate {opt_cand[0] + 1:d}]':-^100}")
                print(
                    f"{f'Recommended Apportionment: Run {np.nansum(app_eff):.0f}/{n_exp:d} Experiments':^100}")
                if self._invariant_controls:
                    print("Time-invariant Controls:")
                    print(opt_cand[1])
                if self._dynamic_controls:
                    print("Time-varying Controls:")
                    print(opt_cand[2])
                if self._dynamic_system:
                    if self._opt_sampling_times:
                        if self._specified_n_spt:
                            print("Sampling Time Variants:")
                            for comb, spt_comb in enumerate(opt_cand[3]):
                                print(f"  Variant {comb + 1} ~ [", end='')
                                for j, sp_time in enumerate(spt_comb):
                                    print(f"{f'{sp_time:.2f}':>10}", end='')
                                print("]: ", end='')
                                print(
                                    f'Run {f"{app_eff[comb]:.0f}/{np.nansum(app_eff):.0f}":>6} experiments, collecting {self._n_spt_spec} samples at given times')
                        else:
                            print("Sampling Times:")
                            for j, sp_time in enumerate(opt_cand[3]):
                                print(f"[{f'{sp_time:.2f}':>10}]: "
                                      f"Run {f'{app_eff[j]:.0f}/{np.nansum(app_eff):.0f}':>6} experiments, sampling at given time")
                    else:
                        print("Sampling Times:")
                        print(self.sampling_times_candidates[i])

            """ Computing and Reporting Rounding Efficiency """
            self.epsilon = self._eval_efficiency_bound(
                self.apportionments / n_exp,
                self.opt_eff,
            )

            """ 
            =============================================================================
            Computing actual efficiency 
            =============================================================================
            the rounding efficiency above is computed using efforts that excludes
            experimental candidates with non-zero efforts i.e., only supports
            to compute actual efficiency, non_trimmed_apportionment is required
            i.e., need candidates with zero efforts too.
            """
            # initialize the non_trimmed_apportionments
            self.non_trimmed_apportionments = np.zeros_like(self.efforts)
            for opt_c, app_c in zip(self.optimal_candidates, self.apportionments):
                opt_idx = opt_c[0]
                opt_spt = opt_c[5]
                if isinstance(app_c, float):
                    self.non_trimmed_apportionments[opt_idx, opt_spt] = app_c
                else:
                    for spt, app in zip(opt_spt, app_c):
                        self.non_trimmed_apportionments[opt_idx, spt] = app
            # normalized to non_trimmed_rounded_efforts
            non_trimmed_rounded_efforts = self.non_trimmed_apportionments / np.sum(self.non_trimmed_apportionments)
            if compute_actual_efficiency:
                _original_efforts = np.copy(self.efforts)
                try:
                    rounded_criterion_value = getattr(self, self._current_criterion)(non_trimmed_rounded_efforts).value
                except AttributeError:
                    rounded_criterion_value = getattr(self, self._current_criterion)(non_trimmed_rounded_efforts)
                if self._current_criterion == "d_opt_criterion":
                    efficiency = np.exp(1 / self.n_mp * (-rounded_criterion_value - self._criterion_value))
                elif self._current_criterion == "a_opt_criterion":
                    efficiency = -self._criterion_value / rounded_criterion_value
                elif self._current_criterion == "e_opt_criterion":
                    efficiency = -rounded_criterion_value / self._criterion_value
                self.efforts = _original_efforts

            if not trimmed:
                self.apportionments = self.non_trimmed_apportionments

            print(f"".center(100, "-"))
            print(
                f"The rounded design for {n_exp} runs is guaranteed to be at least "
                f"{self.epsilon * 100:.2f}% as good as the continuous design."
            )
            if compute_actual_efficiency:
                efficiency = np.squeeze(efficiency)
                print(
                    f"The actual criterion value of the rounded design is "
                    f"{efficiency * 100:.2f}% as informative as the continuous design."
                )
            print(f"{'':#^100}")
        self._save_atomics = _original_save_atomics

        return self.apportionments.astype(int)

    def _adams_apportionment(self, efforts, n_exp):

        def update(effort, mu):
            return np.ceil(effort * mu)

        # pukelsheim's Heuristic
        mu = n_exp - efforts.size / 2
        self.apportionments = update(efforts, mu)
        iterations = 0
        while True:
            iterations += 1
            if np.nansum(self.apportionments) == n_exp:
                if self._verbose >= 3:
                    print(
                        f"Apportionment completed in {iterations} iterations, with final multiplier {mu}.")
                return self.apportionments
            elif np.nansum(self.apportionments) > n_exp:
                ratios = (self.apportionments - 1) / efforts
                candidate_to_reduce = np.unravel_index(np.nanargmax(ratios), ratios.shape)
                self.apportionments[candidate_to_reduce] -= 1
            else:
                ratios = self.apportionments / efforts
                candidate_to_increase = np.unravel_index(np.nanargmin(ratios), ratios.shape)
                self.apportionments[candidate_to_increase] += 1

    def _greatest_effort_apportionment(self, efforts, n_exp):
        self.apportionments = np.zeros_like(efforts)
        chosen_supports = []
        for _ in range(n_exp):
            chosen_support = np.where(efforts == np.nanmax(efforts))[0]
            chosen_support = np.random.choice(chosen_support)
            efforts[chosen_support] = 0
            chosen_supports.append(chosen_support)
        for support in chosen_supports:
            self.apportionments[support] = 1
        return self.apportionments

    @staticmethod
    def _eval_efficiency_bound(effort1, effort2):
        eff_ratio = effort1 / effort2
        min_lkhd_ratio = np.nanmin(eff_ratio)
        return min_lkhd_ratio

    # create grid
    def create_grid(self, bounds, levels):
        """ returns points from a mesh-centered grid """
        bounds = np.asarray(bounds)
        levels = np.asarray(levels)
        grid_args = ''
        for bound, level in zip(bounds, levels):
            grid_args += '%f:%f:%dj,' % (bound[0], bound[1], level)
        make_grid = 'self.grid = np.mgrid[%s]' % grid_args
        exec(make_grid)
        self.grid = self.grid.reshape(np.array(levels).size, np.prod(levels)).T
        return self.grid

    def enumerate_candidates(self, bounds, levels, switching_times=None):
        # use create_grid if only time-invariant controls
        if switching_times is None:
            return self.create_grid(bounds, levels)

        """ check syntax of given bounds, levels, switching times """
        bounds = np.asarray(bounds)
        levels = np.asarray(levels)
        switching_times = np.asarray(switching_times)
        # make sure bounds, levels, switching times are numpy arrays
        if not all(isinstance(arg, np.ndarray) for arg in [bounds, levels, switching_times]):
            raise SyntaxError(
                f"Supplied bounds, levels, and switching times must be numpy arrays."
            )
        # make sure length of experimental variables are the same
        bound_len, bound_dim = bounds.shape
        if bound_dim != 2:
            raise SyntaxError(
                f"Supplied bounds must be a 2D array with shape (:, 2)."
            )
        if levels.ndim != 1:
            raise SyntaxError(
                f"Supplied levels must be a 1D array."
            )
        levels_len = levels.size
        switch_len = len(switching_times)

        # count number of candidates from given information
        if not bound_len == levels_len == switch_len:
            raise SyntaxError(
                f"Supplied lengths are incompatible. Bound: {bound_len}, "
                f"levels: {levels_len}, switch_len: {switch_len}."
            )

        """ discretize tvc into piecewise constants and use create_grid to enumerate """
        tic_idx = []
        tvc_idx = []
        tic_bounds = []
        tic_levels = []
        tvc_bounds = []
        tvc_levels = []
        for i, swt_t in enumerate(switching_times):
            if swt_t is None:
                tic_idx.append(i)
                tic_bounds.append(bounds[i])
                tic_levels.append(levels[i])
            else:
                tvc_idx.append(i)
                for t in swt_t:
                    tvc_bounds.append(bounds[i])
                    tvc_levels.append(levels[i])
        n_tic = len(tic_idx)
        n_tvc = len(tvc_idx)
        if n_tic == 0:
            total_bounds = tvc_bounds
            total_levels = tvc_levels
        elif n_tvc == 0:
            total_bounds = tic_bounds
            total_levels = tic_levels
        else:
            total_bounds = np.vstack((tic_bounds, tvc_bounds))
            total_levels = np.append(tic_levels, tvc_levels)
        candidates = self.create_grid(total_bounds, total_levels)
        tic = candidates[:, :n_tic]
        tvc_array = candidates[:, n_tic:]

        """ converting 2D tvc_array of floats into a 2D numpy array of dictionaries """
        tvc = []
        for candidate, values in enumerate(tvc_array):
            col_counter = 0
            temp_tvc_dict_list = []
            for idx in tvc_idx:
                temp_tvc_dict = {}
                for t in switching_times[idx]:
                    temp_tvc_dict[t] = values[col_counter]
                    col_counter += 1
                temp_tvc_dict_list.append(temp_tvc_dict)
            tvc.append(temp_tvc_dict_list)
        tvc = np.asarray(tvc)

        return tic, tvc

    # visualization and result retrieval
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

    def plot_predictions(self, figsize=None, label_candidates=True):
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
            lim = [
                np.nanmin(self.response[:, :, self.measurable_responses[res]]),
                np.nanmax(self.response[:, :, self.measurable_responses[res]]),
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
                        axes.set_ylim(lim)
                        if self.time_unit_name is not None:
                            axes.set_xlabel(f"Time ({self.time_unit_name})")
                        else:
                            axes.set_xlabel('Time')
                        ylabel = self.response_names[res]
                        if self.response_unit_names is not None:
                            ylabel += f" ({self.response_unit_names[res]})"
                        axes.set_ylabel(ylabel)
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
                # labels outside candidate loop
                if self.time_unit_name is not None:
                    axes[row, col].set_xlabel(f"Sampling Times ({self.time_unit_name})")
                else:
                    axes[row, col].set_xlabel('Sampling Times')
                ylabel = self.response_names[self.measurable_responses[row]]
                ylabel += "/"
                ylabel += self.model_parameter_names[col]
                if self.response_unit_names is not None:
                    if self.model_parameter_unit_names is not None:
                        ylabel += f" ({self.response_unit_names[row]}/{self.model_parameter_unit_names[col]})"
                axes[row, col].set_ylabel(ylabel)
                if legend and self.n_c <= 10:
                    axes[row, col].legend()
        fig.tight_layout()
        return [fig]

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
        x_axis_lim = [
            np.min(self.sampling_times_candidates[
                       ~np.isnan(self.sampling_times_candidates)]),
            np.max(self.sampling_times_candidates[
                       ~np.isnan(self.sampling_times_candidates)])
        ]
        for res in range(self.n_m_r):
            if self._pseudo_bayesian:
                res_max = np.nanmax(mean_res[:, :, res] + std_res[:, :, res])
                res_min = np.nanmin(mean_res[:, :, res] - std_res[:, :, res])
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

    def print_optimal_candidates(self, tol=1e-4):
        if self.optimal_candidates is None:
            self.get_optimal_candidates(tol)
        if self.n_opt_c == 0:
            print(
                f"[Warning]: empty optimal candidates, skipping printing of optimal "
                f"candidates."
            )
            return

        print("")
        print(f"{' Optimal Candidates ':#^100}")
        print(f"{'Obtained on':<40}: {datetime.now()}")
        print(f"{'Criterion':<40}: {self._current_criterion}")
        print(f"{'Criterion Value':<40}: {self._criterion_value}")
        print(f"{'Pseudo-bayesian':<40}: {self._pseudo_bayesian}")
        if self._pseudo_bayesian:
            print(f"{'Pseudo-bayesian Criterion Type':<40}: {self._pseudo_bayesian_type}")
        print(f"{'CVaR Problem':<40}: {self._cvar_problem}")
        if self._cvar_problem:
            print(f"{'Beta':<40}: {self.beta}")
            print(f"{'Constrained Problem':<40}: {self._constrained_cvar}")
            if self._constrained_cvar:
                print(f"{'Min. Mean Value':<40}: {getattr(self, "_cvar_mean_phi", float("nan")):.6f}")
        print(f"{'Dynamic':<40}: {self._dynamic_system}")
        print(f"{'Time-invariant Controls':<40}: {self._invariant_controls}")
        print(f"{'Time-varying Controls':<40}: {self._dynamic_controls}")
        print(f"{'Number of Candidates':<40}: {self.n_c}")
        print(f"{'Number of Optimal Candidates':<40}: {self.n_opt_c}")
        if self._dynamic_system:
            print(f"{'Number of Sampling Time Choices':<40}: {self.n_spt}")
            print(f"{'Sampling Times Optimized':<40}: {self._opt_sampling_times}")
            if self._opt_sampling_times:
                print(f"{'Number of Samples Per Experiment':<40}: {self._n_spt_spec}")
        if self._pseudo_bayesian:
            print(f"{'Number of Scenarios':<40}: {self.n_scr}")
        print(f"{'Information Matrix Regularized':<40}: {self._regularize_fim}")
        if self._regularize_fim:
            print(f"{'Regularization Epsilon':<40}: {self._eps}")
        if self._prior_fim is not None:
            print(f"{'Prior FIM':<40}: registered  "
                  f"({self._prior_n_exp} prior experiment(s), "
                  f"θ_prior={np.array2string(self._prior_fim_mp, precision=3, separator=', ')})")
        else:
            print(f"{'Prior FIM':<40}: none (first-round design)")
        print(f"{'Minimum Effort Threshold':<40}: {tol}")
        for i, opt_cand in enumerate(self.optimal_candidates):
            print(f"{f'[Candidate {opt_cand[0] + 1:d}]':-^100}")
            print(f"{f'Recommended Effort: {np.sum(opt_cand[4]):.2%} of experiments':^100}")
            if self._invariant_controls:
                print("Time-invariant Controls:")
                print(opt_cand[1])
            if self._dynamic_controls:
                print("Time-varying Controls:")
                print(opt_cand[2])
            if self._dynamic_system:
                if self._opt_sampling_times:
                    if self._specified_n_spt:
                        print("Sampling Time Variants:")
                        for comb, spt_comb in enumerate(opt_cand[3]):
                            print(f"  Variant {comb+1} ~ [", end='')
                            for j, sp_time in enumerate(spt_comb):
                                print(f"{f'{sp_time:.2f}':>10}", end='')
                            print("]: ", end='')
                            print(f'{f"{opt_cand[4][comb].sum():.2%}":>10} of experiments')
                    else:
                        print("Sampling Times:")
                        for j, sp_time in enumerate(opt_cand[3]):
                            print(f"[{f'{sp_time:.2f}':>10}]: "
                                  f"dedicate {f'{opt_cand[4][j]:.2%}':>6} of experiments")
                else:
                    print("Sampling Times:")
                    print(self.sampling_times_candidates[i])
        print(f"{'':#^100}")

    def start_logging(self):
        fn = f"log"
        fp = self._generate_result_path(fn, "txt")
        sys.stdout = Logger(file_path=fp)

    def stop_logging(self):
        sys.stdout = sys.__stdout__

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
            predict_var,
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
                pvar = self.eval_pvar(x)
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
                if keyboard.is_pressed("shift"):
                    bounds = bounds + sensitivity * np.array(
                        [[-event.step * ranges[0], event.step * ranges[0]], [0, 0]])
                elif keyboard.is_pressed("ctrl"):
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

    @staticmethod
    def show_plots():
        plt.show()

    # saving, loading, writing
    def load_oed_result(self, result_path):
        with open(getcwd() + result_path, "rb") as file:
            oed_result = dill.load(file)

        self._optimization_time = oed_result["optimization_time"]
        self._sensitivity_analysis_time = oed_result["sensitivity_analysis_time"]
        self._current_criterion = oed_result["optimality_criterion"]
        self._criterion_value = oed_result["criterion_value"]
        self.ti_controls_candidates = oed_result["ti_controls_candidates"]
        self.tv_controls_candidates = oed_result["tv_controls_candidates"]
        self.model_parameters = oed_result["model_parameters"]
        self.sampling_times_candidates = oed_result["sampling_times_candidates"]
        self.efforts = oed_result["optimal_efforts"]
        # support both new "solver" key and legacy "optimization_package" key
        self._solver = oed_result.get("solver",
                       oed_result.get("optimization_package", "ipopt"))
        self._pseudo_bayesian = oed_result["pseudo_bayesian"]
        self._pseudo_bayesian_type = oed_result["pseudo_bayesian_type"]
        self._opt_sampling_times = oed_result["optimize_sampling_times"]
        self._regularize_fim = oed_result["regularized"]
        self._n_spt_spec = oed_result["n_spt_spec"]
        self._prior_fim    = oed_result.get("prior_fim",    None)
        self._prior_fim_mp = oed_result.get("prior_fim_mp", None)
        self._prior_n_exp  = oed_result.get("prior_n_exp",  0)
        self._candidates_changed = False
        self._model_parameters_changed = False

    def create_result_dir(self):
        if self.result_dir_daily is None:
            now = datetime.now()
            self.result_dir_daily = getcwd() + "/"
            self.result_dir_daily += path.splitext(path.basename(main.__file__))[0] + "_result/"
            self.result_dir_daily += f'date_{now.year:d}-{now.month:d}-{now.day:d}/'
            self.create_result_dir()
        else:
            if path.exists(self.result_dir_daily):
                return
            else:
                makedirs(self.result_dir_daily)

    def write_oed_result(self):
        fn = f"{self.oed_result['optimality_criterion']:s}_oed_result"
        fp = self._generate_result_path(fn, "pkl")
        dump(self.oed_result, open(fp, "wb"))

    def save_state(self):
        # pre-process the designer before saving
        state = [
            self.n_c,
            self.n_spt,
            self.n_r,
            self.n_mp,
            self.ti_controls_candidates,
            self.tv_controls_candidates,
            self.sampling_times_candidates,
            self.measurable_responses,
            self.n_m_r,
            self.model_parameters,
        ]

        designer_file = f"state"
        fp = self._generate_result_path(designer_file, "pkl")
        dill.dump(state, open(fp, "wb"))

    def load_state(self, designer_path):
        state = dill.load(open(getcwd() + designer_path, 'rb'))
        self.n_c = state[0]
        self.n_spt = state[1]
        self.n_r = state[2]
        self.n_mp = state[3]
        self.ti_controls_candidates = state[4]
        self.tv_controls_candidates = state[5]
        self.sampling_times_candidates = state[6]
        self.measurable_responses = state[7]
        self.n_m_r = state[8]
        self.model_parameters = state[9]

    def save_responses(self):
        # TODO: implement save responses
        pass

    def load_sensitivity(self, sens_path):
        self.sensitivities = load(open(getcwd() + "/" + sens_path, "rb"))
        self._model_parameters_changed = False
        self._candidates_changed = False
        return self.sensitivities

    def load_atomics(self, atomic_path):
        with open(getcwd() + atomic_path, "rb") as file:
            if self._pseudo_bayesian:
                self.pb_atomic_fims = load(file)
            else:
                self.atomic_fims = load(file)
        self._model_parameters_changed = False
        self._candidates_changed = False
        return self.atomic_fims

    """ criteria """

    # calibration-oriented
    def d_opt_criterion(self, efforts):
        """ it is a PSD criterion, with exponential cone """
        if self._pseudo_bayesian:
            return self._pb_d_opt_criterion(efforts)
        else:
            return self._d_opt_criterion(efforts)

    def a_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        if self._pseudo_bayesian:
            return self._pb_a_opt_criterion(efforts)
        else:
            return self._a_opt_criterion(efforts)

    def e_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        if self._pseudo_bayesian:
            return self._pb_e_opt_criterion(efforts)
        else:
            return self._e_opt_criterion(efforts)

    # prediction-oriented
    def dg_opt_criterion(self, efforts):
        if self._pseudo_bayesian:
            return self._pb_dg_opt_criterion(efforts)
        else:
            return self._dg_opt_criterion(efforts)

    def di_opt_criterion(self, efforts):
        if self._pseudo_bayesian:
            return self._pb_di_opt_criterion(efforts)
        else:
            return self._di_opt_criterion(efforts)

    def ag_opt_criterion(self, efforts):
        if self._pseudo_bayesian:
            return self._pb_ag_opt_criterion(efforts)
        else:
            return self._ag_opt_criterion(efforts)

    def ai_opt_criterion(self, efforts):
        if self._pseudo_bayesian:
            return self._pb_ai_opt_criterion(efforts)
        else:
            return self._ai_opt_criterion(efforts)

    def eg_opt_criterion(self, efforts):
        if self._pseudo_bayesian:
            return self._pb_eg_opt_criterion(efforts)
        else:
            return self._eg_opt_criterion(efforts)

    def ei_opt_criterion(self, efforts):
        if self._pseudo_bayesian:
            return self._pb_ei_opt_criterion(efforts)
        else:
            return self._ei_opt_criterion(efforts)

    # V-optimal (McAuley): prediction variance at user-specified operating conditions
    def v_opt_criterion(self, efforts):
        """
        V-optimality criterion (Shahmohammadi & McAuley, 2019).

        Minimises the total prediction variance at the operating conditions of
        interest encoded in the W matrix:

            J_V = trace( W @ FIM^{-1} @ W^T )

        W is the scaled sensitivity matrix evaluated at dw (the optimal operating
        point found by find_optimal_operating_point). FIM is built from the
        experimental candidates in the usual way.

        FIM inversion uses np.linalg.inv with a fallback to the Moore-Penrose
        pseudoinverse when the FIM is singular. Tikhonov regularization is also
        applied when regularize_fim=True is passed to design_experiment().
        """
        return self._v_opt_criterion(efforts)

    def _v_opt_criterion(self, efforts):
        if not self._dw_fixed:
            raise SyntaxError(
                "dw has not been fixed. Call find_optimal_operating_point() "
                "before running V-optimal design."
            )

        # build W once per design call (cached in self.W)
        if self.W is None:
            self._eval_W_matrix()

        # build FIM from experimental candidates (standard path)
        self.eval_fim(efforts)

        if self.fim.size == 1:
            return float(self.fim)

        # --- invert FIM with regularization / pseudoinverse fallback ---
        if self._regularize_fim:
            fim_reg = self.fim + self._eps * np.eye(self.n_mp)
            try:
                fim_inv = np.linalg.inv(fim_reg)
            except np.linalg.LinAlgError:
                fim_inv = np.linalg.pinv(fim_reg)
        else:
            try:
                fim_inv = np.linalg.inv(self.fim)
            except np.linalg.LinAlgError:
                if self._verbose >= 1:
                    print(
                        "[v_opt_criterion] FIM is singular — falling back to "
                        "Moore-Penrose pseudoinverse."
                    )
                fim_inv = np.linalg.pinv(self.fim)

        J_V = np.trace(self.W @ fim_inv @ self.W.T)

        if self._fd_jac:
            return J_V
        else:
            raise NotImplementedError(
                "Analytic Jacobian for v_opt_criterion is not yet implemented. "
                "Use fd_jac=True (the default)."
            )

    def _eval_W_matrix(self):
        """
        Compute the W matrix: scaled model sensitivities at the optimal
        operating point dw.  This is the bridge between Stage 1 (process
        optimisation) and Stage 2 (V-optimal MBDoE).

        W encodes the prediction directions at dw that the experimental
        design must target.  The V-optimality criterion

            J_V = trace( W @ FIM^{-1} @ W^T )

        measures the total prediction variance at dw.  Minimising J_V over
        the effort allocation selects experiments whose sensitivity structure
        aligns with the prediction directions in W.

        Mathematical definition (McAuley eq. 6)
        -----------------------------------------
        For each operating point dw and each response i and parameter j:

            W_ij = (dg(dw, theta) / d_theta_j) * (s_yi / s_theta_j)

        where:
            dg/d_theta_j  sensitivity of response i to parameter j at dw
            s_yi          measurement std dev of response i = sqrt(error_cov[i,i])
            s_theta_j     nominal parameter uncertainty = abs(model_parameters[j])

        The scaling makes W dimensionless and ensures that parameters of
        very different magnitudes contribute proportionally to J_V.

        Shape
        -----
        W has shape (r_w * n_spt_dw * n_m_r, n_mp), where:
            r_w      : number of operating points in dw_tic
            n_spt_dw : number of time points in dw_spt (1 for end-of-batch)
            n_m_r    : number of measurable responses
            n_mp     : number of model parameters

        Each block of (n_spt_dw * n_m_r) rows corresponds to one operating
        point.  For non-dynamic models, n_spt_dw is forced to 1.

        Caching
        -------
        W is computed once and cached in self.W.  It is automatically
        recomputed by design_v_optimal() when model_parameters have changed
        (the _model_parameters_changed flag is checked).  To force
        recomputation manually, set self.W = None or pass recompute_W=True
        to design_v_optimal().

        Numerical method
        ----------------
        Uses the same numdifftools forward finite-difference Jacobian as
        eval_sensitivities(), with identical step generator settings
        (base_step, step_ratio, num_steps from _num_steps).

        Notes
        -----
        dw_spt specifies when during the optimal operating profile prediction
        accuracy is required.  It is a user specification, not a degree of
        freedom — it is distinct from sampling_times_candidates (which the
        MBDoE optimises over as decision variables).

        Attributes
        ----------
        W : np.ndarray, shape (r_w * n_spt_dw * n_m_r, n_mp)
            Scaled sensitivity matrix at dw.  Set by this method.
        """
        if self.dw_tic is None or self.dw_tvc is None:
            raise SyntaxError(
                "dw_tic / dw_tvc are not set. Call find_optimal_operating_point() first."
            )
        if self.dw_spt is None:
            raise SyntaxError(
                "dw_spt must be set before calling _eval_W_matrix(). "
                "Specify the sampling times at which prediction accuracy matters, "
                "e.g. designer.dw_spt = np.array([t_final])."
            )

        dw_spt = np.atleast_1d(self.dw_spt)
        r_w    = self.dw_tic.shape[0]

        # for non-dynamic systems sampling times are irrelevant — force a single
        # dummy spt so the loop runs once and shape arithmetic stays consistent
        if not self._dynamic_system:
            dw_spt = np.array([0.0])

        # scaling vectors
        s_y     = np.sqrt(np.diag(self.error_cov))          # length n_m_r
        s_theta = np.abs(self.model_parameters)              # length n_mp
        # avoid division by zero for parameters that are exactly 0
        s_theta = np.where(s_theta == 0, 1.0, s_theta)

        step_gen = nd.step_generators.MaxStepGenerator(
            base_step=2,
            step_ratio=2,
            num_steps=self._num_steps,
        )

        W_blocks = []

        for w in range(r_w):
            tic_w = self.dw_tic[w]
            tvc_w = self.dw_tvc[w]

            def model_at_dw(mp, _tic=tic_w, _tvc=tvc_w, _spt=dw_spt):
                """
                Returns measurable responses at dw_spt for given mp.
                Shape: (n_spt_dw * n_m_r,)  — flattened for Jacobian computation.
                """
                res = self._simulate_internal(_tic, _tvc, mp, _spt)
                # res shape: (n_spt_dw, n_r) for dynamic, (n_r,) for static
                if self._dynamic_system:
                    res_m = res[:, self.measurable_responses]   # (n_spt_dw, n_m_r)
                else:
                    res_m = res[self.measurable_responses]       # (n_m_r,)
                return res_m.flatten()

            jac_func = nd.Jacobian(model_at_dw, step=step_gen, method='forward')
            S_w = jac_func(self.model_parameters)
            # S_w shape: (n_spt_dw * n_m_r, n_mp)

            # apply McAuley scaling: W_ij = S_ij * s_yi / s_theta_j
            # s_y tiles over spt dimension: [s_y0, s_y1, ..., s_y0, s_y1, ...]
            n_spt_dw = len(dw_spt)
            s_y_tiled = np.tile(s_y, n_spt_dw)                  # (n_spt_dw * n_m_r,)
            W_w = S_w * (s_y_tiled[:, None] / s_theta[None, :]) # (n_spt_dw * n_m_r, n_mp)

            W_blocks.append(W_w)

            if self._verbose >= 2:
                print(f"[_eval_W_matrix] dw point {w+1}/{r_w}: "
                      f"W block shape = {W_w.shape}")

        self.W = np.vstack(W_blocks)   # (r_w * n_spt_dw * n_m_r, n_mp)

        if self._verbose >= 1:
            print(f"[_eval_W_matrix] W matrix computed: shape = {self.W.shape}")

        return self.W

    def design_v_optimal(self, n_exp=None, solver="ipopt", solver_options=None,
                          e0=None, regularize_fim=False, recompute_W=False, **kwargs):
        """
        Stage 2 of V-optimal MBDoE: design experiments that minimise prediction
        variance at the optimal operating point ``dw`` found in Stage 1.

        Parameters
        ----------
        n_exp : int or None
            Number of experiments for a discrete (exact) design.
            ``None`` (default) gives a continuous design (effort fractions).
        solver : str
            Pyomo solver name (default ``"ipopt"``).
        solver_options : dict, optional
            Options forwarded to the solver
            (e.g. ``{"tol": 1e-8, "linear_solver": "ma57"}``).
        e0 : array-like or None
            Initial effort allocation.  ``None`` uses equal efforts.
        regularize_fim : bool
            If ``True``, adds ``eps * I`` to the FIM before inversion.
        recompute_W : bool
            Force recomputation of W even if already cached.
        **kwargs
            Forwarded to ``design_experiment()``.
        """
        if not self._dw_fixed:
            raise SyntaxError(
                "dw has not been fixed. Call find_optimal_operating_point() first."
            )

        if self.dw_spt is None:
            raise SyntaxError(
                "dw_spt must be set before calling design_v_optimal(). "
                "e.g. designer.dw_spt = np.array([t_final])"
            )

        if self._model_parameters_changed:
            recompute_W = True

        if self.W is None or recompute_W:
            self._eval_W_matrix()

        return self.design_experiment(
            criterion=self.v_opt_criterion,
            n_exp=n_exp,
            solver=solver,
            solver_options=solver_options,
            e0=e0,
            regularize_fim=regularize_fim,
            **kwargs,
        )

    # goal-oriented for design space
    def vdi_criterion(self, efforts):
        if self._pseudo_bayesian:
            raise NotImplementedError("Pseudo-bayesian designs for the VDI criterion not"
                                      "implemented yet, keep an eye out in future "
                                      "releases.")
        else:
            return self._vdi_opt_criterion(efforts)

    def _vdi_opt_criterion(self, efforts):

        self.eval_pim_for_v_opt(efforts)
        di_opts = np.empty((self.n_c_go, self.n_spt_go))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                if np.squeeze(PVAR).size == 1:
                    di_opts[c, spt] = np.squeeze(PVAR)
                else:
                    sign, temp_di = np.linalg.slogdet(pvar)
                    if sign != 1:
                        temp_di = np.inf
                    di_opts[c, spt] = temp_di
        di_opt = np.sum(di_opts)

        if self._fd_jac:
            return di_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ei_opt unavailable.")

    def eval_pim_for_v_opt(self, efforts, vector=False):

        """ update mp, and efforts """
        self.eval_fim(efforts)

        fim_inv = np.linalg.inv(self.fim)

        # compute the sensitivities of the samples from design spaces
        if self.go_sample_sensitivities_done is False:
            self._swap_candidates()
            self.eval_sensitivities()
            self.go_sample_sensitivities_done = True
            self._swap_candidates()
            self._candidates_changed = False
        if vector:
            self.pvars = np.array([
                [f @ fim_inv @ f.T for f in F] for F in self.go_sensitivities
            ])
        else:
            self.pvars = np.empty((self.n_c_go, self.n_spt_go, self.n_r_go, self.n_r_go))
            for c, F in enumerate(self.go_sensitivities):
                for spt, f in enumerate(F):
                    self.pvars[c, spt, :, :] = f @ fim_inv @ f.T
        return self.pvars

    def _swap_candidates(self):
        self._candidates_swapped = not self._candidates_swapped
        self._ticc, self.go_tic = self.go_tic, self._ticc
        self._tvcc, self.go_tvc = self.go_tvc, self._tvcc
        self._sptc, self.go_spt = self.go_spt, self._sptc
        self.n_c, self.n_c_go = self.n_c_go, self.n_c
        self.n_tic, self.n_tic_go = self.n_tic_go, self.n_tic
        self.n_r, self.n_r_go = self.n_r_go, self.n_r
        self.n_spt, self.n_spt_go = self.n_spt_go, self.n_spt
        self.simulate, self.go_simulate = self.go_simulate, self.simulate
        self.go_sensitivities, self.sensitivities = self.sensitivities, self.go_sensitivities
        self.error_cov, self.go_error_cov = self.go_error_cov, self.error_cov
        self.initialize(verbose=self._verbose)
        self._model_parameters_changed = False

    def _revert_candidates(self):
        self.ti_controls_candidates = self.old_tic_cands
        self.tv_controls_candidates = self.old_tic_cands
        self.spt_controls_candidates = self.old_tic_cands

        self.sensitivities = self.old_sensitivities
        if self.go_simulate:
            self.simulate, self.go_simulate = self.go_simulate, self.simulate
        self.initialize(verbose=0)

        self._model_parameters_changed = False
        self._candidates_changed = False

    # experimental
    def u_opt_criterion(self, efforts):
        self.eval_fim(efforts, self.model_parameters)
        return -np.sum(np.multiply(self.fim, self.fim))

    # risk-averse
    def cvar_d_opt_criterion(self, fim):
        """
        D-optimal CVaR criterion.  Called by the CVaR solver with a per-scenario
        FIM (plain numpy array).  Returns -log-det(fim).
        """
        self._cvar_problem = True

        if self._pseudo_bayesian:
            # fim is a plain numpy array supplied by _solve_pyomo_cvar
            fim = np.asarray(fim)
            if fim.size == 1:
                return -float(np.squeeze(fim))
            sign, logdet = np.linalg.slogdet(fim)
            return -logdet if sign == 1 else np.inf
        else:
            raise SyntaxError(
                "CVaR criterion cannot be used for non Pseudo-bayesian problems, please "
                "ensure that you passed in the correct 2D numpy array as "
                "model_parameters."
            )

    """ evaluators """

    def eval_sensitivities(self, method='forward', base_step=2, step_ratio=2,
                           store_predictions=True,
                           plot_analysis_times=False, save_sensitivities=None,
                           reporting_frequency=None, n_jobs=None):
        """
        Main evaluator for computing numerical sensitivities of the responses with
        respect to the model parameters.

        By default uses numdifftools' adaptive finite-difference Jacobian with
        Richardson extrapolation.  When use_pyomo_ift=True, exact parametric
        sensitivities are computed via the Implicit-Function Theorem (IFT) applied
        to a user-supplied Pyomo DAE model — no finite-difference perturbations.

        Parameters
        ----------
        method : str
            Finite-difference method passed to numdifftools ('forward', 'central',
            etc.).  Ignored when use_pyomo_ift=True.
        base_step : float
            Base step size for numdifftools step generator.
        step_ratio : float
            Step ratio for numdifftools Richardson extrapolation.
        store_predictions : bool
            Whether to cache model predictions alongside sensitivities.
        plot_analysis_times : bool
            If True, plot per-candidate sensitivity computation times.
        save_sensitivities : bool or None
            Override the designer's save_sensitivities flag for this call.
        reporting_frequency : int or None
            How often to print progress (every N candidates).  None uses
            the designer default.
        n_jobs : int
            Number of parallel workers for sensitivity computation.
            1  — sequential (default, safe for all backends).
            -1 — use all available CPU cores.
            N  — use N cores.

            Parallelisation is currently supported only when use_pyomo_ift=True.
            Uses joblib with prefer="processes" (loky backend) so each worker
            runs in an isolated subprocess — fully avoiding Pyomo's thread-unsafe
            LoggingIntercept and C-extension global state.  For the non-PB path
            each subprocess handles one candidate; for the PB path each subprocess
            handles all candidates for one scenario (amortising spawn overhead).

            For the finite-difference path, n_jobs > 1 is ignored (numdifftools
            is not thread-safe across candidates without additional work).

            Requires: pip install joblib  (usually already installed via scipy).

        Notes
        -----
        Default behaviour is forward finite difference to prevent model instability
        when parameter values change sign during central-difference evaluation.

        When use_pyomo_ift=True, the sensitivity method is entirely different:
        the Jacobian of the discretised DAE constraints is assembled via PyomoNLP
        (compiled ASL, fast) or Pyomo's symbolic differentiate() (pure Python,
        slower), then the IFT linear system J_z * S = -J_p is solved once per
        candidate to give exact sensitivities for all parameters simultaneously.
        """
        # Resolve n_jobs: explicit argument overrides self.n_jobs attribute
        if n_jobs is None:
            n_jobs = getattr(self, 'n_jobs', 1) or 1

        if self.use_finite_difference:
            # setting default behaviour for step generators
            step_generator = nd.step_generators.MaxStepGenerator(
                base_step=base_step,
                step_ratio=step_ratio,
                num_steps=self._num_steps,
                step_nom=self._step_nom,
            )

        if isinstance(reporting_frequency, int) and reporting_frequency > 0:
            self.sens_report_freq = reporting_frequency
        if save_sensitivities is not None:
            self._save_sensitivities = save_sensitivities

        if self._pseudo_bayesian and not self._large_memory_requirement:
            self._scr_sens = np.empty((self.n_scr, self.n_c, self.n_spt, self.n_m_r, self.n_mp))

        # ── Pyomo IFT path: validate ──────────────────────────────────────────
        _use_pyomo_ift = getattr(self, 'use_pyomo_ift', False)
        if _use_pyomo_ift:
            if not _PYOMO_IFT_AVAILABLE:
                raise ImportError(
                    "use_pyomo_ift=True but Pyomo/scipy could not be imported. "
                    "Install with: pip install pyomo scipy"
                )
            if self.pyomo_model_fn is None:
                raise ValueError(
                    "use_pyomo_ift=True but pyomo_model_fn is None. "
                    "Assign a callable with signature\n"
                    "  pyomo_model_fn(ti_controls, model_parameters)\n"
                    "  -> (model, all_vars, all_bodies, t_sorted)\n"
                    "where all_vars has the n_mp parameter Vars listed first."
                )

        self._sensitivity_analysis_done = False
        if self._verbose >= 2:
            print('[Sensitivity Analysis]'.center(100, "-"))
            if _use_pyomo_ift:
                backend = "PyomoNLP / ASL (compiled)" if _PYNUMERO_ASL_AVAILABLE else "Pyomo differentiate() (pure Python)"
                print(f"{'Sensitivity Method':<40}: Pyomo IFT — {backend}")
            else:
                print(f"{'Use Finite Difference':<40}: {self.use_finite_difference}")
                if self.use_finite_difference:
                    print(f"{'Richardson Extrapolation Steps':<40}: {self._num_steps}")
            print(f"{'Normalized by Parameter Values':<40}: {self._norm_sens_by_params}")
            print(f"".center(100, "-"))
        start = time()

        self.sensitivities = np.empty((self.n_c, self.n_spt, self.n_m_r, self.n_mp))

        candidate_sens_times = []
        if self.use_finite_difference and not _use_pyomo_ift:
            jacob_fun = nd.Jacobian(fun=self._sensitivity_sim_wrapper, step=step_generator, method=method, full_output=False)
        """ main loop over experimental candidates """
        main_loop_start = time()

        # ── Parallel Pyomo IFT path ───────────────────────────────────────────
        # When n_jobs != 1 and use_pyomo_ift=True, candidates are evaluated
        # in parallel using joblib with prefer="threads".  A threading.Lock
        # (created once per eval_sensitivities call) serialises pyomo_model_fn()
        # to avoid the Pyomo LoggingIntercept AssertionError that fires when
        # dae.collocation transforms run concurrently in multiple threads.
        # The ASL Jacobian and IFT linear solve run without the lock.
        if _use_pyomo_ift and n_jobs != 1:
            try:
                from joblib import Parallel, delayed
            except ImportError:
                raise ImportError(
                    "n_jobs != 1 requires joblib. Install with: pip install joblib"
                )

            # Extract all candidate inputs upfront — workers receive plain arrays
            candidates = [
                (
                    exp_candidate[1],                                    # tic
                    exp_candidate[2],                                    # tvc
                    exp_candidate[0][~np.isnan(exp_candidate[0])],      # spt
                )
                for exp_candidate in zip(
                    self.sampling_times_candidates,
                    self.ti_controls_candidates,
                    self.tv_controls_candidates,
                )
            ]

            pyomo_fn   = self.pyomo_model_fn
            scr_mp     = self._current_scr_mp
            out_names  = getattr(self, 'pyomo_output_var_name', None)
            n_mr       = self.n_m_r

            # Use loky (subprocess) workers to fully isolate Pyomo global state
            # (LoggingIntercept, C-extension caches) between workers.
            def _worker(tic, tvc, spt):
                """Subprocess worker — fully isolated, returns (resp, sens)."""
                import types, numpy as _np
                fake = types.SimpleNamespace(
                    _current_spt          = spt,
                    pyomo_model_fn        = pyomo_fn,
                    pyomo_output_var_name = out_names,
                    n_m_r                 = n_mr,
                )
                return Designer._eval_sensitivities_pyomo_ift(
                    fake, tic, scr_mp, store_predictions=False
                )

            if self._verbose >= 1:
                print(
                    f"[eval_sensitivities] Running {self.n_c} candidates in parallel "
                    f"(n_jobs={n_jobs}, backend=loky)..."
                )

            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_worker)(tic, tvc, spt)
                for tic, tvc, spt in candidates
            )

            for i, (temp_resp, temp_sens) in enumerate(results):
                self.sensitivities[i, :] = temp_sens
                candidate_sens_times.append(0.0)  # timing not meaningful in parallel

            if self._verbose >= 2:
                finish = time()
                print(
                    f"[eval_sensitivities] Parallel sensitivity complete: "
                    f"{finish - main_loop_start:.2f}s total."
                )

        else:
            # ── Sequential path (original behaviour) ─────────────────────────
            for i, exp_candidate in enumerate(
                    zip(self.sampling_times_candidates, self.ti_controls_candidates,
                        self.tv_controls_candidates)):
                """ specifying current experimental candidate """
                self._current_tic = exp_candidate[1]
                self._current_tvc = exp_candidate[2]
                self._current_spt = exp_candidate[0][~np.isnan(exp_candidate[0])]

                self.feval_sensitivity = 0
                single_start = time()

                # ── Pyomo IFT branch ──────────────────────────────────────────────
                if _use_pyomo_ift:
                    try:
                        temp_resp, temp_sens = self._eval_sensitivities_pyomo_ift(
                            self._current_tic,
                            self._current_scr_mp,
                            store_predictions,
                        )
                    except Exception as exc:
                        print(
                            f"[Pyomo IFT] Error for candidate {i}:\n"
                            f"  ti_controls      : {self._current_tic}\n"
                            f"  model_parameters : {self._current_scr_mp}\n"
                            f"  Error: {exc}"
                        )
                        raise
                    finish = time()
                    if self._verbose >= 2 and self.sens_report_freq != 0:
                        if (i + 1) % max(1, int(np.ceil(self.n_c / self.sens_report_freq))) == 0 \
                                or (i + 1) == self.n_c:
                            print(
                                f'[Candidate {f"{i + 1:d}/{self.n_c:d}":>10}]: '
                                f'time elapsed {f"{finish - main_loop_start:.2f}":>15} seconds.'
                            )
                    candidate_sens_times.append(finish - single_start)
                    self.sensitivities[i, :] = temp_sens

                # ── Original path: finite-difference or analytic ──────────────────
                else:
                    try:
                        if self.use_finite_difference:
                            temp_sens = jacob_fun(self._current_scr_mp, store_predictions)
                        else:
                            temp_resp, temp_sens = self._sensitivity_sim_wrapper(self._current_scr_mp,
                                                                                 store_predictions)
                    except RuntimeError:
                        print(
                            "The simulate function you provided encountered a Runtime Error "
                            "during sensitivity analysis. The inputs to the simulate function "
                            "were as follows."
                        )
                        print("Model Parameters:")
                        print(self._current_scr_mp)
                        print("Time-invariant Controls:")
                        print(self._current_tic)
                        print("Time-varying Controls:")
                        print(self._current_tvc)
                        print("Sampling Time Candidates:")
                        print(self._current_spt)
                        raise RuntimeError
                    finish = time()
                    if self._verbose >= 2 and self.sens_report_freq != 0:
                        if (i + 1) % np.ceil(self.n_c / self.sens_report_freq) == 0 or (
                                i + 1) == self.n_c:
                            print(
                                f'[Candidate {f"{i + 1:d}/{self.n_c:d}":>10}]: '
                                f'time elapsed {f"{finish - main_loop_start:.2f}":>15} seconds.'
                            )
                    candidate_sens_times.append(finish - single_start)
                # Pyomo IFT already returns (n_spt, n_mr, n_mp) — no reshaping needed.
                # Only apply the FD axis-reordering logic for the finite-difference path.
                if self.use_finite_difference and not _use_pyomo_ift:
                    n_dim = len(temp_sens.shape)
                    if n_dim == 3:
                        temp_sens = np.moveaxis(temp_sens, 1, 2)
                    elif self.n_spt == 1:
                        if self.n_mp == 1:
                            temp_sens = temp_sens[:, :, np.newaxis]
                        else:
                            temp_sens = temp_sens[np.newaxis]
                    elif self.n_mp == 1:
                        temp_sens = np.moveaxis(temp_sens, 0, 1)
                        temp_sens = temp_sens[:, :, np.newaxis]
                    elif self.n_r == 1:
                        temp_sens = temp_sens[:, np.newaxis, :]
                    self.sensitivities[i, :] = temp_sens
                if self._save_txt and i == self._save_txt_nc - 1:
                    self._save_sensitivities_to_txt()

        finish = time()
        if self._verbose >= 2 and self.sens_report_freq != 0:
            print("".center(100, "-"))
        self._sensitivity_analysis_time += finish - start

        if self._var_n_sampling_time:
            self._pad_sensitivities()

        if self._pseudo_bayesian and not self._large_memory_requirement:
            self._scr_sens[self._current_scr] = self.sensitivities

        if self._save_sensitivities and not self._pseudo_bayesian:
            sens_file = f'sensitivity_{self.n_c}_cand'
            if self._dynamic_system:
                sens_file += f"_{self.n_spt}_spt"
            if self._candidates_swapped:
                sens_file += f"_go_{self.n_c_go}_cand"
            fp = self._generate_result_path(sens_file, "pkl")
            dump(self.sensitivities, open(fp, 'wb'))

        if plot_analysis_times:
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.plot(np.arange(1, self.n_c + 1, step=1), candidate_sens_times)

        self._sensitivity_analysis_done = True

        if self._norm_sens_by_params:
            self.sensitivities = self.sensitivities * self._current_scr_mp[None, None, None, :]

        return self.sensitivities

    def _save_sensitivities_to_txt(self):
        fmt = self._save_txt_fmt
        resp_file = f'response_{self._save_txt_nc}'
        fp = self._generate_result_path(resp_file, "txt")
        with open(fp, 'w') as txt:
            txt.write('[Responses]'.center(121, " ") + '\n')
            for ic in range(self._save_txt_nc):
                if self._dynamic_system and ic == 0:
                    txt.write("Sampling Times:")
                    np.savetxt(txt, self.sampling_times_candidates[ic], fmt=fmt, newline='')
                    txt.write('\n')
                txt.write(f'[Candidate {f"{ic + 1:d}":>10}] \n')
                if self._invariant_controls:
                    txt.write("Time-invariant Controls:")
                    np.savetxt(txt, self.ti_controls_candidates[ic], fmt=fmt, newline='')
                    txt.write('\n')
                # if self._dynamic_controls:
                #     txt.write("Time-varying Controls:")
                #     np.savetxt(txt, self.tv_controls_candidates[ic], fmt=fmt, newline='')
                #     txt.write('\n')

                for isa in range(self.n_spt):
                    np.savetxt(txt, self.response[ic, isa], fmt=fmt, newline='')
                    txt.write('\n')
                txt.write("".center(121, "=") + '\n')
        sens_file = f'sensitivity_{self._save_txt_nc}'
        fp = self._generate_result_path(sens_file, "txt")
        with open(fp, 'w') as txt:
            txt.write('[Sensitivity Analysis]'.center(121, " ") + '\n')
            for ic in range(self._save_txt_nc):
                txt.write(f'[Candidate {f"{ic + 1:d}":>10}] \n')
                for isa in range(self.n_spt):
                    txt.write("".center(121, "-") + '\n')
                    np.savetxt(txt, self.sensitivities[ic, isa, :], fmt=fmt)
                txt.write("".center(121, "=") + '\n')

    def eval_fim(self, efforts, store_predictions=True):
        """
        Main evaluator for constructing the FIM from obtained sensitivities, stored in
        self.fim. When problem does not require large memory, will store atomic FIMs. The
        atomic FIMs for the c-th candidate is accessed through self.atomic_fims[c],
        returning a symmetric n_mp x n_mp 2D numpy array.

        When used for pseudo-Bayesian problems, the FIM is computed for each parameter
        scenario, stored in self.scr_fims. The atomic FIMs are stored as a 4D np.array,
        with dimensions (in order) n_scr, n_c, n_mp, n_mp i.e., the atomic FIM for the
        s-th parameter scenario and c-th candidate is accessed through
        self.pb_atomic_fims[s, c], returning a symmetric n_mp x n_mp 2D numpy array.

        The function also performs a parameter estimability study based on the FIM by
        summing the squares over the rows and columns of the FIM. Optionally, will trim
        out rows and columns that have its sum of squares close to 0. This helps with
        non-invertible FIMs.

        An alternative for dealing with non-invertible FIMs is to use a simple Tikhonov
        regularization, where a small scalar times the identity matrix is added to the
        FIM to obtain an invertible matrix.
        """
        if self._pseudo_bayesian:
            self._eval_pb_fims(
                efforts=efforts,
                store_predictions=store_predictions,
            )
            return self.scr_fims
        else:
            self._eval_fim(
                efforts=efforts,
                store_predictions=store_predictions,
            )
            return self.fim

    def diagnose_sensitivity(self, tol_diag=1.0, tol_cond=1e4, plot=True,
                             figsize=None, write=False, dpi=360):
        """
        Diagnose rank-deficiency and near-zero sensitivity in the candidate grid
        using scale-free, physically motivated thresholds.

        Background
        ----------
        pydex normalises every sensitivity by the nominal parameter value::

            s_norm[c, t, r, j] = (∂y_r / ∂θ_j) · θ_j

        This makes sensitivities dimensionless — they represent the fractional
        change in response per fractional change in parameter (local elasticity).
        The atomic FIM diagonal is therefore also dimensionless::

            A_k[j, j] = Σ_{t,r}  s_norm[k,t,r,j]² / σ_r²

        Its inverse is the Cramér–Rao lower bound on the variance of θ_j / θ_j
        (relative variance) from a **single** experiment at candidate k.
        This gives a natural, grid-independent threshold:

        - ``A_k[j,j] < 1`` — one experiment cannot determine θⱼ to within its
          own magnitude; you need at least ``1/A_k[j,j]`` experiments at this
          candidate just to get a signal-to-noise ratio of 1 for θⱼ.
        - ``A_k[j,j] < tol_diag`` (default 1.0) — flags the above condition.

        Unlike a relative-norm threshold (which depends on which other candidates
        are in the grid), this criterion is entirely self-contained: it only
        depends on the model physics, the measurement noise, and the nominal
        parameters.

        Two quantities are computed per candidate
        ------------------------------------------
        1. **Atomic FIM diagonal** ``diag_A[c, j] = A_k[j, j]``
           — Fisher information for θⱼ from one experiment at candidate c.
           Flagged when below ``tol_diag``.

        2. **Condition number** of the full atomic FIM ``A_k``
           — ratio of largest to smallest eigenvalue.  A large condition number
           (even when no diagonal entry is near zero) means two or more parameters
           are nearly collinear at this candidate: allocating many experiments
           there still leaves a linear combination of parameters poorly determined.
           Flagged when above ``tol_cond``.

        The singular values of each ``A_k`` are also returned so users can inspect
        the full spectrum and identify which parameter directions are unobservable.

        Parameters
        ----------
        tol_diag : float
            Threshold for flagging a near-zero atomic FIM diagonal entry.
            ``A_k[j,j] < tol_diag`` → flag parameter θⱼ as unobservable at
            candidate k.  Default: 1.0 (one experiment cannot determine θⱼ to
            within its own magnitude).  Increase to be stricter (e.g. 10 means
            the single-experiment SNR must be at least √10 ≈ 3).

        tol_cond : float
            Condition number threshold above which a candidate is flagged as
            ill-conditioned.  Default: 1e4.

        plot : bool
            If True, produce two figures:
              - Heatmap of ``log10(A_k[j,j])`` (candidates × parameters),
                with ``tol_diag`` threshold line and flagged cells marked.
              - Bar chart of per-candidate condition numbers.

        figsize : tuple or None
            Figure size.  None uses automatic sizing.

        write : bool
            Save figures to the result directory.

        dpi : int
            DPI for saved figures.

        Returns
        -------
        dict with keys:
            ``"diag_A"``        : np.ndarray (n_c, n_mp) — atomic FIM diagonal
            ``"cond"``          : np.ndarray (n_c,)      — condition numbers
            ``"singular_vals"`` : list of np.ndarray     — eigenvalues of A_k per candidate
            ``"flagged_diag"``  : list of (cand_idx, param_idx) — below tol_diag
            ``"flagged_cond"``  : list of int             — above tol_cond
            ``"param_names"``   : list of str
            ``"candidate_names"``: list of str
            ``"figs"``          : list of matplotlib Figure

        Raises
        ------
        RuntimeError
            If ``eval_sensitivities()`` has not been called yet.

        Examples
        --------
        >>> d.eval_sensitivities()
        >>> result = d.diagnose_sensitivity(tol_diag=1.0, tol_cond=1e4)
        >>> # result["flagged_diag"] — (candidate, parameter) pairs: one experiment
        >>> #   cannot determine that parameter to within its own magnitude here.
        >>> # result["flagged_cond"] — candidates where parameters are collinear.
        """
        if self.sensitivities is None:
            raise RuntimeError(
                "Sensitivities have not been computed yet. "
                "Call eval_sensitivities() first."
            )

        sens = self.sensitivities   # (n_c, n_spt, n_m_r, n_mp)

        # --- names ---
        param_names = (
            list(self.model_parameter_names)
            if self.model_parameter_names is not None
            else [f"θ_{j}" for j in range(self.n_mp)]
        )
        cand_names = (
            [str(cn) for cn in self.candidate_names]
            if self.candidate_names is not None
            else [f"C{c+1}" for c in range(self.n_c)]
        )

        # --- error FIM ---
        err_fim = self.error_fim if self.error_fim is not None else np.eye(self.n_m_r)

        # --- measurable responses only ---
        sens_m = sens[:, :, self.measurable_responses, :]  # (n_c, n_spt, n_m_r, n_mp)

        # --- per-candidate atomic FIM, diagonal, condition number, eigenvalues ---
        diag_A       = np.zeros((self.n_c, self.n_mp))
        cond_numbers = np.zeros(self.n_c)
        singular_vals = []

        for c in range(self.n_c):
            # sens_m[c] shape: (n_spt, n_m_r, n_mp)
            # Accumulate A_c = Σ_t  S_t.T @ err_fim @ S_t  (sum over time points)
            # This is equivalent to S_flat.T @ block_diag(err_fim,...) @ S_flat
            # but avoids building the large block-diagonal matrix explicitly.
            A_c = np.zeros((self.n_mp, self.n_mp))
            for t in range(sens_m.shape[1]):
                S_t = sens_m[c, t]            # (n_m_r, n_mp)
                A_c += S_t.T @ err_fim @ S_t  # (n_mp, n_mp)
            diag_A[c] = np.diag(A_c)

            # eigenvalues (symmetric matrix — use eigvalsh for stability)
            ev = np.linalg.eigvalsh(A_c)               # ascending order
            ev_pos = ev[ev > 0]
            cond_numbers[c] = (ev_pos[-1] / ev_pos[0]) if len(ev_pos) >= 2 else np.inf
            singular_vals.append(ev[::-1])             # store descending

        # --- flags ---
        flagged_diag = [
            (c, j)
            for c in range(self.n_c)
            for j in range(self.n_mp)
            if diag_A[c, j] < tol_diag
        ]
        flagged_cond = [c for c in range(self.n_c) if cond_numbers[c] > tol_cond]

        # --- print report ---
        sep = "─" * 100
        print(f"\n{' Sensitivity Diagnosis ':─^100}")
        print(f"  Candidates         : {self.n_c}")
        print(f"  Parameters         : {self.n_mp}")
        print(f"  tol_diag           : {tol_diag:.1g}"
              f"  (flag A_k[j,j] < {tol_diag:.1g}  ← {tol_diag:.1g} experiment(s) needed"
              f" for SNR≥1 on θⱼ)")
        print(f"  tol_cond           : {tol_cond:.1g}")
        print(f"{sep}")

        pcw = max(10, max(len(p) for p in param_names))
        header = f"  {'Candidate':<20}"
        for p in param_names:
            header += f"  {p:>{pcw}}"
        header += f"  {'Cond#':>12}  Status"
        print(header)
        print(f"  {'':20}  " + "  ".join(f"{'A_k[j,j]':>{pcw}}" for _ in param_names)
              + f"  {'':>12}")
        print(sep)

        for c in range(self.n_c):
            row    = f"  {cand_names[c]:<20}"
            issues = []
            for j in range(self.n_mp):
                val = diag_A[c, j]
                s   = f"{val:>{pcw}.3f}"
                if val < tol_diag:
                    s = f"{'!'+f'{val:.1e}':>{pcw}}"
                    issues.append(f"{param_names[j]}")
                row += f"  {s}"
            cn     = cond_numbers[c]
            cn_str = f"{cn:>12.2e}" if np.isfinite(cn) else f"{'∞':>12}"
            if cn > tol_cond:
                cn_str = f"{'!'+f'{cn:.1e}':>12}"
                issues.append("ill-cond")
            row += f"  {cn_str}"
            if issues:
                row += f"  ⚠ {', '.join(issues)}"
            print(row)

        print(sep)
        print(f"\n  Summary:")
        print(f"    Near-zero diagonal flags  : {len(flagged_diag)} "
              f"(candidate, parameter) pairs")
        if flagged_diag:
            for c, j in flagged_diag[:10]:
                print(f"      {cand_names[c]:<22}  {param_names[j]:<20}"
                      f"  A_k[j,j] = {diag_A[c,j]:.2e}"
                      f"  → need ≥{1/max(diag_A[c,j],1e-30):.1f} experiments here for SNR≥1")
            if len(flagged_diag) > 10:
                print(f"      ... and {len(flagged_diag)-10} more")
        print(f"    Ill-conditioned candidates : {len(flagged_cond)}")
        if flagged_cond:
            for c in flagged_cond[:10]:
                print(f"      {cand_names[c]:<22}  cond = {cond_numbers[c]:.2e}")
            if len(flagged_cond) > 10:
                print(f"      ... and {len(flagged_cond)-10} more")
        print(f"{'─'*100}\n")

        # --- plots ---
        figs = []
        if plot:
            if figsize is None:
                figsize = (max(8, self.n_mp * 1.4), max(4, self.n_c * 0.32))

            log_diag = np.log10(np.clip(diag_A, 1e-30, None))
            log_tol  = np.log10(tol_diag)

            # heatmap of log10(A_k[j,j])
            fig1, ax1 = plt.subplots(figsize=figsize)
            vmin = min(log_tol - 2, log_diag.min())
            vmax = max(log_tol + 2, log_diag.max())
            im = ax1.imshow(
                log_diag, aspect='auto', cmap='RdYlGn',
                vmin=vmin, vmax=vmax, interpolation='nearest',
            )
            cb = plt.colorbar(im, ax=ax1)
            cb.set_label('log₁₀(A_k[j,j])  — Fisher info per experiment')
            cb.ax.axhline(log_tol, color='black', lw=1.5, ls='--')
            cb.ax.text(1.05, (log_tol - vmin) / (vmax - vmin),
                       f'tol={tol_diag:.0g}', transform=cb.ax.transAxes,
                       va='center', fontsize=7)
            ax1.set_xticks(range(self.n_mp))
            ax1.set_xticklabels(param_names, rotation=30, ha='right', fontsize=8)
            ax1.set_yticks(range(self.n_c))
            ax1.set_yticklabels(cand_names, fontsize=7)
            ax1.set_title(
                "Atomic FIM diagonal  —  A_k[j,j] = Fisher info for θⱼ per experiment\n"
                f"(green = informative, red = near-zero, threshold = {tol_diag:.0g})"
            )
            for c, j in flagged_diag:
                ax1.text(j, c, '!', ha='center', va='center',
                         color='black', fontsize=8, fontweight='bold')
            fig1.tight_layout()
            figs.append(fig1)

            # bar chart of condition numbers
            fig2, ax2 = plt.subplots(figsize=(max(8, self.n_c * 0.25), 4))
            colors = ['#d62728' if cond_numbers[c] > tol_cond else '#2ca02c'
                      for c in range(self.n_c)]
            cn_plot = np.where(np.isfinite(cond_numbers), cond_numbers, 1e15)
            ax2.bar(range(self.n_c), np.log10(cn_plot + 1), color=colors)
            ax2.axhline(np.log10(tol_cond),
                        color='orange', ls='--',
                        label=f'threshold = 10^{np.log10(tol_cond):.0f}')
            ax2.set_xticks(range(self.n_c))
            ax2.set_xticklabels(cand_names, rotation=90, fontsize=6)
            ax2.set_ylabel('log₁₀(condition number of A_k)')
            ax2.set_title(
                'Per-candidate condition number  —  A_k = Sₖᵀ Σ⁻¹ Sₖ\n'
                '(red = ill-conditioned, parameters are collinear at this candidate)'
            )
            ax2.legend(fontsize=8)
            fig2.tight_layout()
            figs.append(fig2)

            if write:
                fp1 = self._generate_result_path("sensitivity_diag_heatmap", "png")
                fp2 = self._generate_result_path("sensitivity_condition",    "png")
                fig1.savefig(fp1, dpi=dpi)
                fig2.savefig(fp2, dpi=dpi)

        return {
            "diag_A"         : diag_A,
            "cond"           : cond_numbers,
            "singular_vals"  : singular_vals,
            "flagged_diag"   : flagged_diag,
            "flagged_cond"   : flagged_cond,
            "param_names"    : param_names,
            "candidate_names": cand_names,
            "figs"           : figs,
        }


    def eval_fim(self, efforts, store_predictions=True):
        """
        Construct the FIM from sensitivities. See diagnose_sensitivity() for
        per-candidate rank and condition diagnostics.
        """
        if self._pseudo_bayesian:
            self._eval_pb_fims(
                efforts=efforts,
                store_predictions=store_predictions,
            )
            return self.scr_fims
        else:
            self._eval_fim(
                efforts=efforts,
                store_predictions=store_predictions,
            )
            return self.fim

    def _eval_fim(self, efforts, store_predictions=True, save_atomics=None,
                  skip_sens_eval=False):
        """
        skip_sens_eval : bool
            When True, skip the eval_sensitivities() call and use whatever is
            already stored in self.sensitivities.  Used by the parallel
            pseudo-Bayesian path in _eval_pb_fims() which pre-computes all
            sensitivities in one flat parallel job and injects them directly.
        """
        if save_atomics is not None:
            self._save_atomics = save_atomics

        def add_candidates(s_in, e_in, error_info_mat):
            if not np.any(np.isnan(s_in)):
                _atom_fim = s_in.T @ error_info_mat @ s_in
                self.fim += e_in * _atom_fim
            else:
                _atom_fim = np.zeros((self.n_mp, self.n_mp))
            if not self._large_memory_requirement:
                if self.atomic_fims is None:
                    self.atomic_fims = []
                if self._compute_atomics:
                    self.atomic_fims.append(_atom_fim)

        """ update efforts """
        self.efforts = efforts

        """ eval_sensitivities, only runs if model parameters changed """
        self._compute_sensitivities = self._model_parameters_changed
        self._compute_sensitivities = self._compute_sensitivities or self._candidates_changed
        self._compute_sensitivities = self._compute_sensitivities or self.sensitivities is None

        self._compute_atomics = self._model_parameters_changed
        self._compute_atomics = self._compute_atomics or self._candidates_changed
        self._compute_atomics = self._compute_atomics or self.atomic_fims is None

        if self._pseudo_bayesian:
            self._compute_sensitivities = self._compute_atomics or self.scr_fims is None

        if self._compute_sensitivities and self._compute_atomics and not skip_sens_eval:
            self.eval_sensitivities(
                save_sensitivities=self._save_sensitivities,
                store_predictions=store_predictions,
            )

        """ evaluate fim """
        start = time()

        # reshape efforts to (n_c, n_spt) for iteration
        if self._specified_n_spt:
            self.efforts = self.efforts.reshape((self.n_c, self.n_spt_comb))
        else:
            self.efforts = self.efforts.reshape((self.n_c, self.n_spt))
            if self.n_spt == 1:
                self.efforts = self.efforts[:, None]
        # if atomic is not given
        if self._compute_atomics:
            self.atomic_fims = []
            self.fim = 0
            if self._specified_n_spt:
                for c, (eff, sen, spt_combs) in enumerate(zip(self.efforts, self.sensitivities, self.spt_candidates_combs)):
                    for comb, (e, spt) in enumerate(zip(eff, spt_combs)):
                        s = np.mean(sen[spt], axis=0)
                        add_candidates(s, e, self.error_fim)
            else:
                for c, (eff, sen) in enumerate(zip(self.efforts, self.sensitivities)):
                    for spt, (e, s) in enumerate(zip(eff, sen)):
                        add_candidates(s, e, self.error_fim)
            if self._save_atomics and not self._pseudo_bayesian:
                sens_file = f"atomics_{self.n_c}_cand"
                if self._dynamic_system:
                    sens_file += f"_{self.n_spt}_spt"
                if self._pseudo_bayesian:
                    sens_file += f"_{self.n_scr}_scr"
                if self._candidates_swapped:
                    sens_file += f"_go_{self.n_c_go}_cand"
                fp = self._generate_result_path(sens_file, "pkl")
                dump(self.atomic_fims, open(fp, 'wb'))
        # if atomic is given
        else:
            self.fim = 0
            # Use a local 4-D view for the loop so that self.atomic_fims stays
            # in its flat (n_c*n_spt, n_mp, n_mp) shape.  Overwriting
            # self.atomic_fims here would cause _d_opt_criterion (and others)
            # to iterate over only n_c rows when computing the analytic
            # Jacobian, returning a gradient of length n_c instead of
            # n_c*n_spt and crashing IPOPT's gradient callback.
            atomic_fims_4d = self.atomic_fims.reshape(
                (self.n_c, self.n_spt, self.n_mp, self.n_mp)
            )
            if self._specified_n_spt:
                for c, (eff, atom, spt_combs) in enumerate(
                    zip(self.efforts, atomic_fims_4d, self.spt_candidates_combs)
                ):
                    for comb, (e, spt) in enumerate(zip(eff, spt_combs)):
                        a = np.mean(atom[spt], axis=0)
                        self.fim += e * a
            else:
                for c, (eff, atom) in enumerate(zip(self.efforts, atomic_fims_4d)):
                    for spt, (e, a) in enumerate(zip(eff, atom)):
                        self.fim += e * a

        finish = time()

        if np.all(np.asarray(self.fim) == 0):
            return np.array([0])

        # --- add prior experimental information (sequential MBDoE) ---
        if self._prior_fim is not None:
            prior = self._prior_fim.copy()
            # rescale to current model_parameters if they changed since prior was computed
            if not np.allclose(self._current_scr_mp, self._prior_fim_mp, rtol=1e-10):
                scale = self._current_scr_mp / self._prior_fim_mp   # (n_mp,)
                rescale = np.outer(scale, scale)
                prior = prior * rescale
            self.fim = self.fim + prior

        if self._regularize_fim:
            if self._verbose >= 3:
                print(
                    f"Applying Tikhonov regularization to FIM by adding "
                    f"{self._eps:.2f} * identity to the FIM. "
                    f"Warning: design is likely to be affected for large scalars!"
                )
            self.fim += self._eps * np.identity(self.n_mp)

        self._fim_eval_time = finish - start
        if self._verbose >= 3:
            print(
                f"Evaluation of fim took {self._fim_eval_time:.2f} seconds."
            )

        if not self._large_memory_requirement:
            self.atomic_fims = np.asarray(self.atomic_fims)

        """ set current mp as completed to prevent recomputation of atomics """
        self._model_parameters_changed = False
        self._candidates_changed = False

        return self.fim

    def _eval_pb_fims(self, efforts, store_predictions=True):
        """ only recompute pb_atomics if the full parameter scenarios are changed """
        self._compute_pb_atomics = self._model_parameters_changed
        self._compute_pb_atomics = self._compute_pb_atomics or self._candidates_changed
        self._compute_pb_atomics = self._compute_pb_atomics or self.pb_atomic_fims is None

        self.scr_fims = []
        if self._compute_pb_atomics:
            if self._verbose >= 2:
                print(f"{' Pseudo-bayesian ':#^100}")
            if self._verbose >= 1:
                print(f'Evaluating information for each scenario...')
            if store_predictions:
                self.scr_responses = []
            if not self._large_memory_requirement:
                self.pb_atomic_fims = np.empty((self.n_scr, self.n_c * self.n_spt, self.n_mp, self.n_mp))

            # ── Parallel pseudo-Bayesian path (Pyomo IFT only) ───────────────
            # Parallelise over scenarios using loky (subprocess) workers.
            # Each subprocess handles all n_c candidates for one scenario
            # sequentially — this isolates Pyomo global state (logging,
            # C-extension caches) between workers, eliminating the thread-
            # safety issues that affect prefer="threads".
            # Spawn overhead (~0.3 s per worker) is amortised over n_c
            # candidates per job, so net cost is small.
            _use_pyomo_ift = getattr(self, "use_pyomo_ift", False)
            _n_jobs = getattr(self, "n_jobs", 1)
            if _use_pyomo_ift and _n_jobs != 1:
                try:
                    from joblib import Parallel, delayed
                except ImportError:
                    raise ImportError(
                        "n_jobs != 1 requires joblib. Install with: pip install joblib"
                    )

                pyomo_fn  = self.pyomo_model_fn
                out_names = getattr(self, "pyomo_output_var_name", None)
                n_mr      = self.n_m_r
                tic_list  = self.ti_controls_candidates
                mp_list   = self.model_parameters   # shape (n_scr, n_mp)
                n_c_      = self.n_c
                n_spt_    = self.n_spt
                n_mp_     = self.n_mp

                def _pb_scenario_worker(scr, mp, tic_list_, out_names_, n_mr_, n_spt__,
                                            n_mp__, norm_sens):
                    """Process all candidates for one scenario; runs in a subprocess.

                    mp          : parameter vector for THIS scenario.
                    norm_sens   : whether to apply parameter-value normalization
                                  (mirrors the _norm_sens_by_params step that
                                  eval_sensitivities applies in the sequential path).
                    """
                    import types, numpy as _np
                    mp = _np.asarray(mp, dtype=float)
                    sens_scr = _np.empty((len(tic_list_), n_spt__, n_mr_, n_mp__))
                    for c, tic in enumerate(tic_list_):
                        fake = types.SimpleNamespace(
                            _current_spt          = _np.atleast_1d(tic),
                            pyomo_model_fn        = pyomo_fn,
                            pyomo_output_var_name = out_names_,
                            n_m_r                 = n_mr_,
                        )
                        _, sens = Designer._eval_sensitivities_pyomo_ift(
                            fake, tic, mp, store_predictions=False
                        )
                        sens_scr[c] = sens
                    # Apply parameter-value normalization (S_ij *= theta_j)
                    # This mirrors the _norm_sens_by_params step in eval_sensitivities
                    # which is bypassed when skip_sens_eval=True.
                    if norm_sens:
                        sens_scr *= mp[_np.newaxis, _np.newaxis, _np.newaxis, :]
                    return scr, sens_scr

                if self._verbose >= 1:
                    print(
                        f"[_eval_pb_fims] Running {self.n_scr} scenario jobs "
                        f"({self.n_scr} scenarios × {self.n_c} candidates) "
                        f"in parallel (n_jobs={_n_jobs}, backend=loky)..."
                    )

                scr_sens = np.empty((self.n_scr, self.n_c, self.n_spt, self.n_m_r, self.n_mp))
                _pb_par_start = time()
                _norm_sens = getattr(self, "_norm_sens_by_params", True)
                raw = Parallel(n_jobs=_n_jobs, prefer="processes")(
                    delayed(_pb_scenario_worker)(
                        scr, mp_list[scr].copy(), list(tic_list), out_names,
                        n_mr, n_spt_, n_mp_, _norm_sens
                    )
                    for scr in range(self.n_scr)
                )
                self._sensitivity_analysis_time = time() - _pb_par_start
                for scr, sens_scr in raw:
                    scr_sens[scr] = sens_scr

                # Build per-scenario FIMs from the collected sensitivities
                for scr, mp in enumerate(self.model_parameters):
                    self._current_scr     = scr
                    self._current_scr_mp  = mp
                    self.sensitivities    = scr_sens[scr]
                    self.atomic_fims      = None
                    self._eval_fim(efforts, store_predictions,
                                   skip_sens_eval=True)
                    self.scr_fims.append(self.fim)
                    if not self._large_memory_requirement:
                        self.pb_atomic_fims[scr] = self.atomic_fims

            else:
                # ── Sequential scenario loop (original behaviour) ─────────────
                for scr, mp in enumerate(self.model_parameters):
                    self.atomic_fims = None
                    self._current_scr = scr
                    self._current_scr_mp = mp
                    if self._verbose >= 2:
                        print(f"{f'[Scenario {scr+1}/{self.n_scr}]':=^100}")
                        print("Model Parameters:")
                        print(mp)
                    self._eval_fim(efforts, store_predictions)
                    self.scr_fims.append(self.fim)
                    if self._verbose >= 2:
                        print(f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds.")
                    if store_predictions:
                        self.scr_responses.append(self.response)
                        self.response = None
                    if not self._large_memory_requirement:
                        self.pb_atomic_fims[scr] = self.atomic_fims
            if store_predictions:
                self.scr_responses = np.array(self.scr_responses)

            """ set current mp as completed to prevent recomputation of atomics """
            self._model_parameters_changed = False
        else:
            for scr, atomic_fims in enumerate(self.pb_atomic_fims):
                self.atomic_fims = atomic_fims
                self._eval_fim(efforts, store_predictions)
                self.scr_fims.append(self.fim)

        if self._save_atomics:
            fn = f"atomics_{self.n_c}_can_{self.n_scr}_scr"
            fp = self._generate_result_path(fn, "pkl")
            dump(self.pb_atomic_fims, open(fp, "wb"))

        return self.scr_fims

    def eval_pim(self, efforts, vector=False):

        """ update mp, and efforts """
        self.eval_fim(efforts)

        fim_inv = np.linalg.inv(self.fim)
        if vector:
            self.pvars = np.array([
                [f @ fim_inv @ f.T for f in F] for F in self.sensitivities
            ])
        else:
            self.pvars = np.empty((self.n_c, self.n_spt, self.n_r, self.n_r))
            for c, F in enumerate(self.sensitivities):
                for spt, f in enumerate(F):
                    self.pvars[c, spt, :, :] = f @ fim_inv @ f.T

        return self.pvars

    def eval_atom_fims(self, mp, store_predictions=True):
        self._current_scr_mp = mp

        """ eval_sensitivities, only runs if model parameters changed """
        self.eval_sensitivities(save_sensitivities=self._save_sensitivities,
                                store_predictions=store_predictions)

        """ deal with unconstrained form, i.e. transform efforts """
        self._transform_efforts()  # only transform if required, logic incorporated there

        """ deal with opt_sampling_times """
        sens = self.sensitivities.reshape(self.n_c * self.n_spt, self.n_m_r, self.n_mp)

        """ main """
        start = time()
        if self._large_memory_requirement:
            confirmation = input(
                f"Memory requirement is large. Slow solution expected, continue?"
                f"Y/N."
            )
            if confirmation != "Y":
                return
        self.atomic_fims = []
        for e, f in zip(self.efforts.flatten(), sens):
            if not np.any(np.isnan(f)):
                _atom_fim = f.T @ f
            else:
                _atom_fim = np.zeros(shape=(self.n_mp, self.n_mp))
            self.atomic_fims.append(_atom_fim)
        finish = time()
        self._fim_eval_time = finish - start

        return self.atomic_fims

    """ getters (filters) """

    def get_optimal_candidates(self, tol=1e-4):
        if self.efforts is None:
            raise SyntaxError(
                'Please solve an experiment design before attempting to get optimal '
                'candidates.'
            )

        self._remove_zero_effort_candidates(tol=tol)
        self.optimal_candidates = []

        for i, eff_sp in enumerate(self.efforts):
            if self._dynamic_system and self._opt_sampling_times:
                optimal = np.any(eff_sp > tol)
            else:
                optimal = np.sum(eff_sp) > tol
            if optimal:
                opt_candidate = [
                    i,  # index of optimal candidate
                    self.ti_controls_candidates[i],
                    self.tv_controls_candidates[i],
                    [],
                    [],
                    [],
                    []
                ]
                if self._opt_sampling_times:
                    for j, eff in enumerate(eff_sp):
                        if eff > tol:
                            if self._specified_n_spt:
                                opt_spt = self.sampling_times_candidates[i, self.spt_candidates_combs[i, j]]
                                opt_candidate[3].append(opt_spt)
                                opt_candidate[4].append(np.ones_like(opt_spt) * eff / len(opt_spt))
                                opt_candidate[5].append(self.spt_candidates_combs[i, j])
                            else:
                                opt_candidate[3].append(self.sampling_times_candidates[i][j])
                                opt_candidate[4].append(eff)
                                opt_candidate[5].append(j)
                else:
                    opt_candidate[3] = self.sampling_times_candidates[i]
                    opt_candidate[4] = eff_sp
                    opt_candidate[5].append([t for t in range(self.n_spt)])
                self.optimal_candidates.append(opt_candidate)

        self.n_opt_c = len(self.optimal_candidates)
        if self.n_opt_c == 0:
            print(
                f"[Warning]: empty optimal candidates. Likely failed optimization; if "
                f"prediction-orriented design is used, try avoiding dg, ag, or eg "
                f"criteria as they are notoriously hard to optimize with gradient-based "
                f"optimizers."
            )

        self.n_factor_sups = 0
        self.n_spt_sups = 0
        self.max_n_opt_spt = 0
        for i, opt_cand in enumerate(self.optimal_candidates):
            if self._dynamic_system and self._opt_sampling_times:
                self.n_factor_sups += len(opt_cand[4])
            else:
                self.n_factor_sups += 1
            self.max_n_opt_spt = max(self.max_n_opt_spt, len(opt_cand[4]))

        return self.optimal_candidates

    """ optional operations """

    def _d_opt_criterion(self, efforts):
        """ D-optimality: maximise log-det(FIM). """
        self.eval_fim(efforts)

        if self.fim.size == 1:
            d_opt = -self.fim
            if self._fd_jac:
                return np.squeeze(d_opt)
            else:
                jac = -np.array([1 / self.fim * m for m in self.atomic_fims])
                return d_opt, jac

        sign, d_opt = np.linalg.slogdet(self.fim)
        if self._fd_jac:
            return -d_opt if sign == 1 else np.inf
        else:
            fim_inv = np.linalg.inv(self.fim)
            jac = -np.array([np.sum(fim_inv.T * m) for m in self.atomic_fims])
            return (-d_opt, jac) if sign == 1 else (np.inf, jac)

    def _a_opt_criterion(self, efforts):
        """ A-optimality: minimise trace(FIM^{-1}). """
        self.eval_fim(efforts)

        if self.fim.size == 1:
            if self._fd_jac:
                return -self.fim
            else:
                jac = np.array([m for m in self.atomic_fims])
                return -self.fim, jac

        if self._fd_jac:
            eigvals = np.linalg.eigvalsh(self.fim)
            return np.sum(1 / eigvals) if np.all(eigvals > 0) else 0
        else:
            jac = np.zeros(self.n_e)
            try:
                fim_inv = np.linalg.inv(self.fim)
                a_opt = fim_inv.trace()
                jac = -np.array([
                    np.sum((fim_inv @ fim_inv) * m) for m in self.atomic_fims
                ])
            except np.linalg.LinAlgError:
                a_opt = 0
            return a_opt, jac

    def _e_opt_criterion(self, efforts):
        """ E-optimality: maximise minimum eigenvalue of FIM. """
        self.eval_fim(efforts)

        if self.fim.size == 1:
            return -self.fim

        if self._fd_jac:
            return -np.linalg.eigvalsh(self.fim).min()
        else:
            raise NotImplementedError  # TODO: implement analytic jac for e-opt

    # prediction-oriented
    def _dg_opt_criterion(self, efforts):

        self.eval_pim(efforts)
        # dg_opt: max det of the pvar matrix over candidates and sampling times
        dg_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                sign, temp_dg = np.linalg.slogdet(pvar)
                if sign != 1:
                    temp_dg = np.inf
                dg_opts[c, spt] = sign * np.exp(temp_dg)
        dg_opt = np.nanmax(dg_opts)

        if self._fd_jac:
            return dg_opt
        else:
            raise NotImplementedError("Analytic Jacobian for dg_opt unavailable.")

    def _di_opt_criterion(self, efforts):

        self.eval_pim(efforts)
        # di_opt: average det of the pvar matrix over candidates and sampling times
        dg_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                sign, temp_dg = np.linalg.slogdet(pvar)
                if sign != 1:
                    temp_dg = np.inf
                dg_opts[c, spt] = temp_dg
        dg_opt = np.nansum(dg_opts)

        if self._fd_jac:
            return dg_opt
        else:
            raise NotImplementedError("Analytic Jacobian for di_opt unavailable.")

    def _ag_opt_criterion(self, efforts):

        self.eval_pim(efforts)
        # ag_opt: max trace of the pvar matrix over candidates and sampling times
        ag_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.trace(pvar)
                ag_opts[c, spt] = temp_dg
        ag_opt = np.nanmax(ag_opts)

        if self._fd_jac:
            return ag_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ag_opt unavailable.")

    def _ai_opt_criterion(self, efforts):

        self.eval_pim(efforts)
        # ai_opt: average trace of the pvar matrix over candidates and sampling times
        ai_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.trace(pvar)
                ai_opts[c, spt] = temp_dg
        ag_opt = np.nansum(ai_opts)

        if self._fd_jac:
            return ag_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ai_opt unavailable.")

    def _eg_opt_criterion(self, efforts):

        self.eval_pim(efforts)
        # eg_opt: max of the max_eigenval of the pvar matrix over candidates and sampling times
        eg_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.linalg.eigvals(pvar).max()
                eg_opts[c, spt] = temp_dg
        eg_opt = np.nanmax(eg_opts)

        if self._fd_jac:
            return eg_opt
        else:
            raise NotImplementedError("Analytic Jacobian for eg_opt unavailable.")

    def _ei_opt_criterion(self, efforts):

        self.eval_pim(efforts)
        # ei_opts: average of the max_eigenval of the pvar matrix over candidates and sampling times
        ei_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.linalg.eigvals(pvar).max()
                ei_opts[c, spt] = temp_dg
        ei_opt = np.nansum(ei_opts)

        if self._fd_jac:
            return ei_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ei_opt unavailable.")

    """ pseudo_bayesian criterion """

    # calibration-oriented
    def _pb_d_opt_criterion(self, efforts):
        """ Pseudo-Bayesian D-optimality. """
        self.eval_fim(efforts)

        if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
            avg_fim = np.mean([fim for fim in self.scr_fims], axis=0)
            sign, d_opt = np.linalg.slogdet(avg_fim)
            return np.inf if sign != 1 else -d_opt
        elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
            d_opt = 0
            for fim in self.scr_fims:
                sign, scr_d_opt = np.linalg.slogdet(fim)
                d_opt += scr_d_opt if sign == 1 else np.inf
            return -d_opt / self.n_scr

    def _pb_a_opt_criterion(self, efforts):
        """ Pseudo-Bayesian A-optimality. """
        self.eval_fim(efforts)

        if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
            return np.linalg.inv(
                np.mean([fim for fim in self.scr_fims], axis=0)
            ).trace()
        elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
            return np.mean([np.linalg.inv(fim).trace() for fim in self.scr_fims])

    def _pb_e_opt_criterion(self, efforts):
        """ Pseudo-Bayesian E-optimality. """
        self.eval_fim(efforts)

        if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
            avg_fim = np.mean([fim for fim in self.scr_fims], axis=0)
            return -np.linalg.eigvalsh(avg_fim).min()
        elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
            return np.mean([
                -np.linalg.eigvalsh(fim).min() for fim in self.scr_fims
            ])

    # prediction-oriented
    def _pb_dg_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_di_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_ag_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_ai_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_eg_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_ei_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    """ private methods """

    def _generate_result_path(self, name, extension, iteration=None):
        self.create_result_dir()

        while True:
            now = datetime.now()
            if not self.result_dir:
                self.result_dir = self.result_dir_daily + f"time_{now.hour:d}-{now.minute:d}-{now.second}/"
                if not path.exists(self.result_dir):
                    makedirs(self.result_dir)
            fn = f"{name}.{extension}"
            if iteration is not None:
                fn = f"iter_{iteration:d}_" + fn
            fp = self.result_dir + fn
            return fp

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


    def _eval_sensitivities_pyomo_ift(self, ti_controls, model_parameters,
                                      store_predictions=True):
        """
        Compute response and exact parametric sensitivities via the
        Implicit-Function Theorem (IFT) applied to a user-supplied Pyomo DAE model.

        Two Jacobian backends, selected automatically:
        1. PyomoNLP / ASL (fast, compiled C) — when pynumero_ASL is available.
           Parameters must be temporarily unfixed so the NL writer includes them.
        2. Pyomo differentiate() (pure Python fallback) — always available.

        In both cases the IFT linear solve is identical:
            J = [J_p | J_z]  where J_p = dc/dp, J_z = dc/dz
            S = lstsq(J_z, -J_p)   shape (n_state, n_mp)

        Returns
        -------
        responses : ndarray shape (n_spt, n_m_r)
        sens      : ndarray shape (n_spt, n_m_r, n_mp)
        """
        import pyomo.environ as _pyo
        import scipy.sparse as _sp

        theta = np.asarray(model_parameters, dtype=float)
        n_mp  = len(theta)
        n_mr  = self.n_m_r

        # 1. Build and initialise the Pyomo model
        m, all_vars, all_bodies, t_sorted = self.pyomo_model_fn(
            ti_controls, theta
        )

        # 2. Resolve output variable name(s)
        out_names = getattr(self, 'pyomo_output_var_name', None)
        if out_names is None:
            out_names = [str(all_vars[n_mp + r]) for r in range(n_mr)]
        elif isinstance(out_names, str):
            out_names = [out_names]

        state_var_strs = [str(v) for v in all_vars[n_mp:]]

        def _find_state_idx(base_name, t_val):
            t_key = min(t_sorted, key=lambda tt: abs(tt - t_val))
            for target in (f"{base_name}[{t_key}]", base_name):
                for idx, vname in enumerate(state_var_strs):
                    if vname == target or vname.endswith(target):
                        return idx
            raise RuntimeError(
                f"[Pyomo IFT] Cannot find state variable '{base_name}[{t_key}]' "
                f"or scalar '{base_name}'.\n"
                f"Available: {state_var_strs}"
            )

        # 3. Build Jacobian — choose backend
        _has_free_vars = any(
            not v.is_fixed()
            for v in m.component_data_objects(_pyo.Var, active=True)
        )
        if _PYNUMERO_ASL_AVAILABLE and _has_free_vars:
            # Fast: unfix param vars, get ASL Jacobian, re-fix
            param_vars = all_vars[:n_mp]
            for pv in param_vars:
                pv.unfix()
            try:
                nlp      = _PyomoNLP(m)
                J_sparse = nlp.evaluate_jacobian_eq()
                J_dense  = J_sparse.toarray()
                nlp_var_names = nlp.primals_names()
            finally:
                for pv in param_vars:
                    pv.fix()

            all_var_strs = [str(v) for v in all_vars]
            col_order = []
            for vname in all_var_strs:
                matched = next(
                    (i for i, n in enumerate(nlp_var_names)
                     if n == vname or n.endswith("." + vname) or vname.endswith("." + n)),
                    None
                )
                if matched is None:
                    raise RuntimeError(
                        f"[Pyomo IFT / ASL] Cannot match variable '{vname}' "
                        f"in NLP variable list.\nNLP vars: {nlp_var_names}"
                    )
                col_order.append(matched)
            J = J_dense[:, col_order]

        else:
            # Fallback: pure-Python differentiate() loop
            n_v = len(all_vars)
            n_c = len(all_bodies)
            J   = np.zeros((n_c, n_v))
            for ci, body in enumerate(all_bodies):
                for vi, var in enumerate(all_vars):
                    try:
                        J[ci, vi] = _pyo.value(_pyomo_differentiate(body, wrt=var))
                    except Exception:
                        J[ci, vi] = 0.0

        # 4. Split J into parameter and state columns
        J_p = J[:, :n_mp]
        J_z = J[:, n_mp:]

        # 5. Solve J_z * S = -J_p
        S, *_ = _scipy_linalg.lstsq(J_z, -J_p)

        # 6. Extract responses and sensitivities
        responses = np.zeros((len(self._current_spt), n_mr))
        sens      = np.zeros((len(self._current_spt), n_mr, n_mp))
        for spt_i, t_val in enumerate(self._current_spt):
            t_key = min(t_sorted, key=lambda tt: abs(tt - t_val))
            for r_i, out_name in enumerate(out_names):
                base_name = out_name.split("[")[0]
                var_comp  = m.find_component(base_name)
                if var_comp is None:
                    var_comp = m.find_component(out_name)
                if var_comp is None:
                    raise RuntimeError(
                        f"[Pyomo IFT] Output variable '{out_name}' not found in model."
                    )
                if hasattr(var_comp, 'is_indexed') and var_comp.is_indexed():
                    val = _pyo.value(var_comp[t_key])
                else:
                    val = _pyo.value(var_comp)
                responses[spt_i, r_i] = val
                sens[spt_i, r_i, :]   = S[_find_state_idx(base_name, t_key), :]

        # 7. Store responses
        if store_predictions:
            self._current_res = responses
            self._store_current_response()

        return responses, sens

    def _sensitivity_sim_wrapper(self, theta_try, store_responses=True):
        if self.use_finite_difference:
            response = self._simulate_internal(self._current_tic, self._current_tvc,
                                               theta_try, self._current_spt)
        else:
            self.do_sensitivity_analysis = True
            response, sens = self._simulate_internal(self._current_tic, self._current_tvc,
                                                     theta_try, self._current_spt)
            self.do_sensitivity_analysis = False
        self.feval_sensitivity += 1
        """ store responses whenever required, and model parameters are the same as 
        current model's """
        if store_responses and np.allclose(theta_try, self._current_scr_mp,
                                           rtol=self._store_responses_rtol,
                                           atol=self._store_responses_atol):
            self._current_res = response
            self._store_current_response()
        if self.use_finite_difference:
            if self.n_m_r == 1 and self.n_spt == 1:
                return response[0]
            else:
                return response
        else:
            return response, sens

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

        sampling_time_scale = np.nanmin(np.diff(self.sampling_times_candidates, axis=1))

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

    def _pad_sampling_times(self):
        """ check the required number of sampling times """
        max_num_sampling_times = 1
        for sampling_times in self.sampling_times_candidates:
            num_sampling_times = len(sampling_times)
            if num_sampling_times > max_num_sampling_times:
                max_num_sampling_times = num_sampling_times

        for i, sampling_times in enumerate(self.sampling_times_candidates):
            num_sampling_times = len(sampling_times)
            if num_sampling_times < max_num_sampling_times:
                diff = max_num_sampling_times - num_sampling_times
                self.sampling_times_candidates[i] = np.pad(sampling_times,
                                                           pad_width=(0, diff),
                                                           mode='constant',
                                                           constant_values=np.nan)
        self.sampling_times_candidates = np.array(
            self.sampling_times_candidates.tolist())
        return self.sampling_times_candidates

    def _pad_sensitivities(self):
        """ padding sensitivities to accommodate for missing sampling times """
        for i, row in enumerate(self.sensitivities):
            if row.ndim < 3:  # check if row has less than 3 dim
                if self.n_mp == 1:  # potential cause 1: we only have 1 mp
                    row = np.expand_dims(row, -1)  # add last dimension
                if self.n_r == 1:  # potential cause 2: we only have 1 response
                    row = np.expand_dims(row, -2)  # add second to last
            if row.ndim != 3:  # check again if already 3 dims
                # only reason: we only have 1 spt, add dim to first position
                row = np.expand_dims(row, 0)
            # pad sampling times
            diff = self.n_spt - row.shape[0]
            self.sensitivities[i] = np.pad(row,
                                           pad_width=[(0, diff), (0, 0), (0, 0)],
                                           mode='constant', constant_values=np.nan)
        self.sensitivities = self.sensitivities.tolist()
        self.sensitivities = np.asarray(self.sensitivities)
        return self.sensitivities

    def _store_current_response(self):
        """ padding responses to accommodate for missing sampling times """
        start = time()
        if self.response is None:  # if it is the first response to be stored,
            # initialize response list
            self.response = []

        if self._dynamic_system and self.n_spt == 1:
            self._current_res = self._current_res[np.newaxis]
        if self.n_r == 1:
            self._current_res = self._current_res[:, np.newaxis]

        if self._var_n_sampling_time:
            self._current_res = np.pad(
                self._current_res,
                pad_width=((0, self.n_spt - self._current_res.shape[0]), (0, 0)),
                mode='constant',
                constant_values=np.nan
            )

        """ convert to list if np array """
        if isinstance(self.response, np.ndarray):
            self.response = self.response.tolist()
        self.response.append(self._current_res)

        """ convert to numpy array """
        self.response = np.array(self.response)
        end = time()
        if self._verbose >= 3:
            print('Storing response took %.6f CPU ms.' % (1000 * (end - start)))
        return self.response

    def _simulate_internal(self, ti_controls, tv_controls, theta, sampling_times):
        raise SyntaxError(
            "Make sure you have initialized the designer, and specified the simulate "
            "function correctly."
        )

    def _initialize_internal_simulate_function(self):
        if self._simulate_signature == 1:
            self._simulate_internal = lambda tic, tvc, mp, spt: \
                self.simulate(tic, mp)
        elif self._simulate_signature == 2:
            self._simulate_internal = lambda tic, tvc, mp, spt: \
                self.simulate(tic, spt, mp)
        elif self._simulate_signature == 3:
            self._simulate_internal = lambda tic, tvc, mp, spt: \
                self.simulate(tvc, spt, mp)
        elif self._simulate_signature == 4:
            self._simulate_internal = lambda tic, tvc, mp, spt: \
                self.simulate(tic, tvc, spt, mp)
        elif self._simulate_signature == 5:
            self._simulate_internal = lambda tic, tvc, mp, spt: \
                self.simulate(spt, mp)
        else:
            raise SyntaxError(
                'Cannot initialize simulate function properly, check your syntax.'
            )

    def _transform_efforts(self):
        if self._unconstrained_form:
            if not self._efforts_transformed:
                self.efforts = np.square(self.efforts)
                self.efforts /= np.sum(self.efforts)
                self._efforts_transformed = True
                if self._verbose >= 3:
                    print("Efforts transformed.")

        return self.efforts

    def _check_missing_components(self):
        # basic components
        if self.model_parameters is None:
            raise SyntaxError("Please specify nominal model parameters.")

        # invariant controls
        if self._invariant_controls and self.ti_controls_candidates is None:
            raise SyntaxError(
                "Simulate function suggests time-invariant controls are needed, but "
                "ti_controls_candidates is empty."
            )

        # dynamic system
        if self._dynamic_system:
            if self.sampling_times_candidates is None:
                raise SyntaxError(
                    "Simulate function suggests dynamic system, but "
                    "sampling_times_candidates is empty."
                )
            if self._dynamic_controls:
                if self.tv_controls_candidates is None:
                    raise SyntaxError(
                        "Simulate function suggests time-varying controls are needed, "
                        "but tv_controls_candidates is empty."
                    )

    def _handle_simulate_sig(self):
        """
        Determines type of model from simulate signature. Five supported types:
        =================================================================================
        1. simulate(ti_controls, model_parameters).
        2. simulate(ti_controls, sampling_times, model_parameters).
        3. simulate(tv_controls, sampling_times, model_parameters).
        4. simulate(ti_controls, tv_controls, sampling_times, model_parameters).
        5. simulate(sampling_times, model_parameters).
        =================================================================================
        If a pyomo.dae model is specified a special signature is recommended that adds
        two input arguments to the beginning of the simulate signatures e.g., for type 3:
        simulate(model, simulator, tv_controls, sampling_times, model_parameters).
        """
        sim_sig = list(signature(self.simulate).parameters.keys())
        unspecified_sig = ["unspecified"]
        if np.all([entry in sim_sig for entry in unspecified_sig]):
            raise SyntaxError("Don't forget to specify the simulate function.")

        t1_sig = ["ti_controls"]
        t2_sig = ["ti_controls", "sampling_times"]
        t3_sig = ["tv_controls", "sampling_times"]
        t4_sig = ["ti_controls", "tv_controls", "sampling_times"]
        t5_sig = ["sampling_times"]
        # initialize simulate id
        self._simulate_signature = 0
        # check if model_parameters is present
        if "model_parameters" not in sim_sig:
            raise SyntaxError(
                f"The input argument \"model_parameters\" is not found in the simulate "
                f"function, please fix simulate signature."
            )
        if np.all([entry in sim_sig for entry in t4_sig]):
            self._simulate_signature = 4
            self._dynamic_system = True
            self._dynamic_controls = True
            self._invariant_controls = True
        elif np.all([entry in sim_sig for entry in t3_sig]):
            self._simulate_signature = 3
            self._dynamic_system = True
            self._dynamic_controls = True
            self._invariant_controls = False
        elif np.all([entry in sim_sig for entry in t2_sig]):
            self._simulate_signature = 2
            self._dynamic_system = True
            self._dynamic_controls = False
            self._invariant_controls = True
        elif np.all([entry in sim_sig for entry in t1_sig]):
            self._simulate_signature = 1
            self._dynamic_system = False
            self._dynamic_controls = False
            self._invariant_controls = True
        elif np.all([entry in sim_sig for entry in t5_sig]):
            self._simulate_signature = 5
            self._dynamic_system = True
            self._dynamic_controls = False
            self._invariant_controls = False
        if self._simulate_signature == 0:
            raise SyntaxError(
                "Unrecognized simulate function signature, please check if you have "
                "specified it correctly. The base signature requires "
                "'model_parameters'. Adding 'sampling_times' makes it dynamic,"
                "adding 'tv_controls' and 'sampling_times' makes a dynamic system with"
                " time-varying controls. Adding 'tv_controls' without 'sampling_times' "
                "does not work. Adding 'model' and 'simulator' makes it a pyomo "
                "simulate signature. 'ti_controls' are optional in all cases."
            )
        self._initialize_internal_simulate_function()

    def _check_stats_framework(self):
        """ check if local or Pseudo-bayesian designs """
        if self.model_parameters.ndim == 1:
            self._pseudo_bayesian = False
        elif self.model_parameters.ndim == 2:
            self._pseudo_bayesian = True
        else:
            raise SyntaxError(
                "model_parameters must be fed in as a 1D numpy array for local "
                "designs, and a 2D numpy array for Pseudo-bayesian designs."
            )

    def _check_candidate_lengths(self):
        if self._invariant_controls:
            self.n_c = self.n_c_tic
        if self._dynamic_controls:
            if not self.n_c:
                self.n_c = self.n_c_tvc
            else:
                assert self.n_c == self.n_c_tvc, f"Inconsistent candidate lengths. " \
                                                 f"tvc_candidates has {self.n_c_tvc}, " \
                                                 f"but {self.n_c} is expected."
        if self._dynamic_system:
            if not self.n_c:
                self.n_c = self.n_c_spt
            else:
                assert self.n_c == self.n_c_spt, f"Inconsistent candidate lengths. " \
                                                 f"spt_candidates has {self.n_c_spt}, " \
                                                 f"but {self.n_c} is expected."

    def _check_var_spt(self):
        if np.all([len(spt) == len(self.sampling_times_candidates[0]) for spt in
                   self.sampling_times_candidates]) \
                and np.all(~np.isnan(self.sampling_times_candidates)):
            self._var_n_sampling_time = False
        else:
            self._var_n_sampling_time = True
            self._pad_sampling_times()

    def _get_component_sizes(self):

        if self._simulate_signature == 1:
            self.n_c_tic, self.n_tic = self.ti_controls_candidates.shape
            self.tv_controls_candidates = np.empty((self.n_c_tic, 1))
            self.n_c_tvc, self.n_tvc = self.n_c_tic, 1
            self.sampling_times_candidates = np.empty_like(self.ti_controls_candidates)
            self.n_c_spt, self.n_spt = self.n_c_tic, 1
        elif self._simulate_signature == 2:
            self.n_c_tic, self.n_tic = self.ti_controls_candidates.shape
            self.tv_controls_candidates = np.empty((self.n_c_tic, 1))
            self.n_c_tvc, self.n_tvc = self.n_c_tic, 1
            self.n_c_spt, self.n_spt = self.sampling_times_candidates.shape
        elif self._simulate_signature == 3:
            self.n_c_tvc, self.n_tvc = self.tv_controls_candidates.shape
            self.ti_controls_candidates = np.empty((self.n_c_tvc, 1))
            self.n_c_tic, self.n_tic = self.n_c_tvc, 1
            self.n_c_spt, self.n_spt = self.sampling_times_candidates.shape
        elif self._simulate_signature == 4:
            self.n_c_tic, self.n_tic = self.ti_controls_candidates.shape
            self.n_c_tvc, self.n_tvc = self.tv_controls_candidates.shape
            self.n_c_spt, self.n_spt = self.sampling_times_candidates.shape
        elif self._simulate_signature == 5:
            self.n_c_spt, self.n_spt = self.sampling_times_candidates.shape
            self.ti_controls_candidates = np.empty((self.n_c_spt, 1))
            self.n_c_tic, self.n_tic = self.n_c_spt, 1
            self.tv_controls_candidates = np.empty((self.n_c_spt, 1))
            self.n_c_tvc, self.n_tvc = self.n_c_spt, 1
        else:
            raise SyntaxError("Unrecognized simulate signature, unable to proceed.")

        # number of model parameters, and scenarios (if pseudo_bayesian)
        if self._pseudo_bayesian:
            self.n_scr, self.n_mp = self.model_parameters.shape
            self._current_scr_mp = self.model_parameters[0]
        else:
            self.n_mp = self.model_parameters.shape[0]
            self._current_scr_mp = self.model_parameters

        # number of responses
        if self.n_r is None:
            if self._verbose >= 3:
                print(
                    "Running one simulation for initialization "
                    "(required to determine number of responses)."
                )
            y = self._simulate_internal(
                self.ti_controls_candidates[0],
                self.tv_controls_candidates[0],
                self._current_scr_mp,
                self.sampling_times_candidates[0][~np.isnan(self.sampling_times_candidates[0])]
            )
            try:
                self.n_spt_r, self.n_r = y.shape
            except ValueError:  # output not two dimensional
                # case 1: n_r is 1
                if self._dynamic_system and self.n_spt > 1:
                    self.n_r = 1
                # case 2: n_spt is 1
                else:
                    self.n_r = y.shape[0]

        # number of measurable responses (if not all)
        if self.measurable_responses is None:
            self.n_m_r = self.n_r
            self.measurable_responses = np.array([_ for _ in range(self.n_r)])
        elif self.n_m_r != len(self.measurable_responses):
            self.n_m_r = len(self.measurable_responses)
            if self.n_m_r > self.n_r:
                raise SyntaxError(
                    "Given number of measurable responses is greater than number of "
                    "responses given."
                )

    def _check_memory_req(self, threshold):
        # check problem size (affects if designer will be memory-efficient or quick)
        self._memory_threshold = threshold
        memory_req = self.n_c * self.n_spt * self.n_m_r * self.n_mp * 8
        if self._pseudo_bayesian:
            memory_req *= self.n_scr
        if memory_req > self._memory_threshold:
            print(
                f'Sensitivity matrix will take {memory_req / 1e9:.2f} GB of memory space '
                f'(more than {self._memory_threshold / 1e9:.2f} GB threshold).'
            )
            self._large_memory_requirement = True

    def _initialize_names(self):
        if self.response_names is None:
            self.response_names = np.array([
                f"Response {_}"
                for _ in range(self.n_m_r)
            ])
        if self.model_parameter_names is None:
            self.model_parameter_names = np.array([
                f"Model Parameter {_}"
                for _ in range(self.n_mp)
            ])
        if self.candidate_names is None:
            self.candidate_names = np.array([
                f"Candidate {_}"
                for _ in range(self.n_c)
            ])
        if self.ti_controls_names is None and self._invariant_controls:
            self.ti_controls_names = np.array([
                f"Time-invariant Control {_}"
                for _ in range(self.n_tic)
            ])
        if self.tv_controls_names is None and self._dynamic_controls:
            self.tv_controls_names = np.array([
                f"Time-varying Control {_}"
                for _ in range(self.n_tvc)
            ])

    def _remove_zero_effort_candidates(self, tol):
        self.efforts[self.efforts < tol] = 0
        self.efforts = self.efforts / self.efforts.sum()
        return self.efforts
