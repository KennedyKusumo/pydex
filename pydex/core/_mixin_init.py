from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime
from inspect import signature
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from pydex.core.simulate import SimulatorBase, _LegacySimulatorAdapter

if TYPE_CHECKING:
    _verbose: int
    _status: str
    _simulate_signature: int
    _dynamic_system: bool
    _dynamic_controls: bool
    _invariant_controls: bool
    _pseudo_bayesian: bool
    _large_memory_requirement: bool
    _var_n_sampling_time: bool
    _model_parameters_changed: bool
    _candidates_changed: bool
    _memory_threshold: int
    _opt_sampling_times: bool
    _specified_n_spt: bool
    _regularize_fim: bool
    _cvar_problem: bool
    _constrained_cvar: bool
    _eps: float
    _n_spt_spec: int
    _current_criterion: str
    _criterion_value: Any
    _pseudo_bayesian_type: str | int | None
    _model_parameters: NDArray[np.float64]
    _ticc: NDArray[np.float64]
    _tvcc: NDArray[np.float64]
    _sptc: NDArray[np.float64]
    _current_scr_mp: NDArray[np.float64]

    n_c: int
    n_c_tic: int
    n_c_tvc: int
    n_c_spt: int
    n_spt: int
    n_spt_r: int
    n_mp: int
    n_r: int
    n_m_r: int
    n_tic: int
    n_tvc: int
    n_scr: int
    n_exp: int
    n_opt_c: int
    beta: float

    model_parameters: NDArray[np.float64]
    ti_controls_candidates: NDArray[np.float64]
    tv_controls_candidates: NDArray[np.float64]
    sampling_times_candidates: NDArray[np.float64]
    measurable_responses: NDArray[np.intp] | None
    error_cov: NDArray[np.float64] | None
    error_fim: NDArray[np.float64]
    optimal_candidates: list[Any] | None
    grid: NDArray[np.float64]
    phi: Any
    response_names: NDArray | None
    model_parameter_names: NDArray | None
    candidate_names: NDArray | None
    ti_controls_names: NDArray | None
    tv_controls_names: NDArray | None


class DesignerInit:

    @property
    def model_parameters(self) -> Any:
        return self._model_parameters

    @model_parameters.setter
    def model_parameters(self, mp: Any) -> None:
        self._model_parameters_changed = True
        self._model_parameters = mp

    @property
    def ti_controls_candidates(self) -> Any:
        return self._ticc

    @ti_controls_candidates.setter
    def ti_controls_candidates(self, ticc: Any) -> None:
        self._candidates_changed = True
        self._ticc = ticc

    @property
    def tv_controls_candidates(self) -> Any:
        return self._tvcc

    @tv_controls_candidates.setter
    def tv_controls_candidates(self, tvcc: Any) -> None:
        self._candidates_changed = True
        self._tvcc = tvcc

    @property
    def sampling_times_candidates(self) -> Any:
        return self._sptc

    @sampling_times_candidates.setter
    def sampling_times_candidates(self, sptc: Any) -> None:
        self._candidates_changed = True
        self._sptc = sptc

    @staticmethod
    def detect_sensitivity_analysis_function() -> bool:
        frame: Any = sys._getframe(1)
        while frame:
            if "numdifftools" in frame.f_code.co_filename:
                return False
            elif frame.f_code.co_name == "eval_sensitivities":
                return True
            frame = frame.f_back
        return False

    """ core activity interfaces """
    def initialize(self, verbose: int = 0, memory_threshold: int = int(1e9)) -> str:
        """ check for syntax errors, runs one simulation to determine n_r """

        """ check if simulate function has been specified """
        self._data_type_check()
        self._check_stats_framework()
        self._configure_simulator()
        self._get_component_sizes()
        self._check_candidate_lengths()
        self._check_missing_components()

        if self._dynamic_system:
            self._check_var_spt()

        self._initialize_names()

        self._check_memory_req(memory_threshold)

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
            print("".center(100, "="))

        return self._status

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
        total_levels: Any
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
    def print_optimal_candidates(self, tol: float = 1e-4, write: bool = False) -> None:
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
        print(f"{'Criterion (higher is better)':<40}: {self._current_criterion}")
        print(f"{f'Criterion Value (efforts sum to {self.n_exp})':<40}: {self._criterion_value}")
        print(f"{'Pseudo-bayesian':<40}: {self._pseudo_bayesian}")
        if self._pseudo_bayesian:
            print(f"{'Pseudo-bayesian Criterion Type':<40}: {self._pseudo_bayesian_type}")
        print(f"{'CVaR Problem':<40}: {self._cvar_problem}")
        if self._cvar_problem:
            print(f"{'Beta':<40}: {self.beta}")
            print(f"{'Constrained Problem':<40}: {self._constrained_cvar}")
            if self._constrained_cvar:
                print(f"{'Min. Mean Value':<40}: {cp.sum(self.phi).value / self.n_scr:.6f}")
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

    def _data_type_check(self):
        self.model_parameters = np.asarray(self.model_parameters)
        self.ti_controls_candidates = np.asarray(self.ti_controls_candidates)
        self.tv_controls_candidates = np.asarray(self.tv_controls_candidates)
        self.sampling_times_candidates = np.asarray(self.sampling_times_candidates)
        if not isinstance(self.model_parameters, (list, np.ndarray)):
            raise SyntaxError('model_parameters must be supplied as a numpy array.')
        if self._invariant_controls:
            if not isinstance(self.ti_controls_candidates, np.ndarray):
                raise SyntaxError(
                    'ti_controls_candidates must be supplied as a numpy array.'
                )
        if self._dynamic_system:
            if not isinstance(self.sampling_times_candidates, np.ndarray):
                raise SyntaxError("sampling_times_candidates must be supplied as a "
                                  "numpy array.")
            if self._dynamic_controls:
                if not isinstance(self.tv_controls_candidates, np.ndarray):
                    raise SyntaxError("tv_controls_candidates must be supplied as a "
                                      "numpy array.")

    def _configure_simulator(self):
        """
        Configures the internal simulator, setting system-type flags.

        If a SimulatorBase instance is attached, reads flags directly from it.
        If a plain callable is attached (legacy), sniffs its argument names,
        wraps it in _LegacySimulatorAdapter (emits DeprecationWarning), and
        sets _simulate_signature for _get_component_sizes compatibility.
        """
        sim = self._simulator

        if sim is None:
            raise SyntaxError(
                "No simulator specified. Assign one via designer.simulator = <SimulatorBase> "
                "or the legacy designer.simulate = <callable>."
            )

        if isinstance(sim, SimulatorBase):
            # new-style: read flags directly, no sniffing needed
            self._dynamic_system = sim.is_dynamic
            self._dynamic_controls = sim.has_tv_controls
            self._invariant_controls = sim.has_ti_controls
            # derive _simulate_signature for _get_component_sizes
            if not sim.is_dynamic and sim.has_ti_controls:
                self._simulate_signature = 1
            elif sim.is_dynamic and sim.has_ti_controls and not sim.has_tv_controls:
                self._simulate_signature = 2
            elif sim.is_dynamic and sim.has_tv_controls and not sim.has_ti_controls:
                self._simulate_signature = 3
            elif sim.is_dynamic and sim.has_ti_controls and sim.has_tv_controls:
                self._simulate_signature = 4
            else:
                self._simulate_signature = 5
            return

        # legacy path: plain callable — sniff signature and wrap
        sim_sig = list(signature(sim).parameters.keys())
        t1_sig = ["ti_controls"]
        t2_sig = ["ti_controls", "sampling_times"]
        t3_sig = ["tv_controls", "sampling_times"]
        t4_sig = ["ti_controls", "tv_controls", "sampling_times"]
        t5_sig = ["sampling_times"]

        self._simulate_signature = 0
        if "model_parameters" not in sim_sig:
            raise SyntaxError(
                'The input argument "model_parameters" is not found in the simulate '
                'function, please fix simulate signature.'
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
                "'model_parameters'. Adding 'sampling_times' makes it dynamic, "
                "adding 'tv_controls' and 'sampling_times' makes a dynamic system with "
                "time-varying controls. Adding 'tv_controls' without 'sampling_times' "
                "does not work. 'ti_controls' are optional in all cases."
            )
        self._simulator = _LegacySimulatorAdapter(sim, self._simulate_signature)

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
        if self.n_r == 0:
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
