from __future__ import annotations

from pickle import dump
from time import time
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numdifftools as nd
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

if TYPE_CHECKING:
    _verbose: int
    _dynamic_system: bool
    _pseudo_bayesian: bool
    _large_memory_requirement: bool
    _save_sensitivities: bool
    _save_atomics: bool
    _save_txt: bool
    _save_txt_nc: int
    _save_txt_fmt: str
    _sensitivity_analysis_done: bool
    _sensitivity_analysis_time: float
    _fim_eval_time: float
    _model_parameters_changed: bool
    _candidates_changed: bool
    _compute_sensitivities: bool
    _compute_atomics: bool
    _compute_pb_atomics: bool
    _norm_sens_by_params: bool
    _unconstrained_form: bool
    _efforts_transformed: bool
    _invariant_controls: bool
    _var_n_sampling_time: bool
    _specified_n_spt: bool
    _candidates_swapped: bool
    _regularize_fim: bool
    _optimization_package: str
    _simulate_signature: int
    _eps: float
    _num_steps: int
    _step_nom: Any
    _store_responses_rtol: float
    _store_responses_atol: float
    _current_tic: Any
    _current_tvc: Any
    _current_spt: NDArray[np.float64]
    _current_scr_mp: NDArray[np.float64]
    _current_scr: int
    _current_res: Any
    _scr_sens: NDArray[np.float64]

    use_finite_difference: bool
    do_sensitivity_analysis: bool
    sens_report_freq: int
    feval_simulation: int
    feval_sensitivity: int
    n_c: int
    n_spt: int
    n_m_r: int
    n_mp: int
    n_r: int
    n_scr: int
    n_c_go: int
    n_spt_comb: int

    model_parameters: NDArray[np.float64]
    response: NDArray[np.float64] | None
    sensitivities: NDArray[np.float64] | None
    efforts: NDArray[np.float64]
    atomic_fims: NDArray[np.float64] | None
    pb_atomic_fims: NDArray[np.float64] | None
    scr_fims: list[Any]
    scr_responses: NDArray[np.float64]
    fim: Any
    pvars: NDArray[np.float64]
    error_fim: NDArray[np.float64]
    ti_controls_candidates: NDArray[np.float64]
    tv_controls_candidates: NDArray[np.float64]
    sampling_times_candidates: NDArray[np.float64]
    spt_candidates_combs: NDArray[np.float64]
    optimal_candidates: list[Any]


class DesignerSensitivity:

    def simulate_candidates(self, store_predictions: bool = True,
                            plot_simulation_times: bool = False) -> Any:
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

    def simulate_optimal_candidates(self) -> None:
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

    def eval_sensitivities(self, method: str = 'forward', base_step: float | int | None = 2,
                           step_ratio: float | int | None = 2, store_predictions: bool = True,
                           plot_analysis_times: bool = False,
                           save_sensitivities: bool | None = None,
                           reporting_frequency: int | None = None) -> NDArray[np.float64] | None:
        """
        Main evaluator for computing numerical sensitivities of the responses with
        respect to the model parameters. Simply provides an interface to numdifftool's
        Jacobian method.

        Numdifftool uses adaptive finite difference to approximate the sensitivities,
        coupled with Richard extrapolation for improved accuracy. Although less accurate,
        default behaviour is to use forward finite difference. This is to prevent model
        instability in common situations where during sensitivity evaluation (e.g.
        with central) model parameter values that are passed to the model changes sign
        and causes the model to fail to run.
        """
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

        self._sensitivity_analysis_done = False
        if self._verbose >= 2:
            print('[Sensitivity Analysis]'.center(100, "-"))
            print(f"{'Use Finite Difference':<40}: {self.use_finite_difference}")
            if self.use_finite_difference:
                print(f"{'Richardson Extrapolation Steps':<40}: {self._num_steps}")
            print(f"{'Normalized by Parameter Values':<40}: {self._norm_sens_by_params}")
            print(f"".center(100, "-"))
        start = time()

        self.sensitivities = np.empty((self.n_c, self.n_spt, self.n_m_r, self.n_mp))

        candidate_sens_times = []
        if self.use_finite_difference:
            jacob_fun = nd.Jacobian(fun=self._sensitivity_sim_wrapper, step=step_generator, method=method, full_output=False)
        """ main loop over experimental candidates """
        main_loop_start = time()
        for i, exp_candidate in enumerate(
                zip(self.sampling_times_candidates, self.ti_controls_candidates,
                    self.tv_controls_candidates)):
            """ specifying current experimental candidate """
            self._current_tic = exp_candidate[1]
            self._current_tvc = exp_candidate[2]
            self._current_spt = exp_candidate[0][~np.isnan(exp_candidate[0])]

            self.feval_sensitivity = 0
            single_start = time()
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
            """
            bunch of lines to make sure the Jacobian method returns the
            sensitivity (temp_sens) with dims: n_sp, n_res, n_theta
            -------------------------------------------------------------------------
            8 possible cases
            -------------------------------------------------------------------------
            case_1: complete                        n_sp    n_theta     n_res
            case_2: n_sp = 1                        n_res   n_theta
            case_3: n_theta = 1                     n_res   n_sp
            case_4: n_res = 1                       n_sp    n_theta
            case_5: n_sp & n_theta = 1              1       n_res
            case_6: n_sp & n_res = 1                1       n_theta
            case_7: n_theta & n_res = 1             1       n_sp
            case_8: n_sp, n_res, n_theta = 1        1       1
            -------------------------------------------------------------------------
            """
            if self.use_finite_difference:
                n_dim = len(temp_sens.shape)
                if n_dim == 3:  # covers case 1
                    temp_sens = np.moveaxis(temp_sens, 1, 2)  # switch n_theta and n_res
                elif self.n_spt == 1:
                    if self.n_mp == 1:  # covers case 5: add a new axis in the last dim
                        temp_sens = temp_sens[:, :, np.newaxis]
                    else:  # covers case 2, 6, and 8: add a new axis in
                        # the first dim
                        temp_sens = temp_sens[np.newaxis]
                elif self.n_mp == 1:  # covers case 3 and 7
                    temp_sens = np.moveaxis(temp_sens, 0,
                                            1)  # move n_sp to the first dim as needed
                    temp_sens = temp_sens[:, :,
                                np.newaxis]  # create a new axis as the last dim for
                    # n_theta
                elif self.n_r == 1:  # covers case 4
                    temp_sens = temp_sens[:, np.newaxis,
                                :]  # create axis in the middle for n_res

            self.sensitivities[i, :] = temp_sens
            if self._save_txt and i == self._save_txt_nc-1:
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

    def eval_fim(self, efforts: NDArray[np.float64] | Any,
                 store_predictions: bool = True,
                 mp: NDArray[np.float64] | Any | None = None) -> NDArray[np.float64]:
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

    def _eval_fim(self, efforts, store_predictions=True, save_atomics=None):
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

        if self._compute_sensitivities and self._compute_atomics:
            self.eval_sensitivities(
                save_sensitivities=self._save_sensitivities,
                store_predictions=store_predictions,
            )

        """ deal with unconstrained form, i.e. transform efforts """
        if self._unconstrained_form:
            self._efforts_transformed = False
        self._transform_efforts()  # only transform if required, logic incorporated there

        """ evaluate fim """
        start = time()

        if self._optimization_package == "scipy":
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
            self.atomic_fims = np.array(self.atomic_fims)
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
            self.atomic_fims = self.atomic_fims.reshape((self.n_c, self.n_spt, self.n_mp, self.n_mp))
            if self._specified_n_spt:
                for c, (eff, atom, spt_combs) in enumerate(zip(self.efforts, self.atomic_fims, self.spt_candidates_combs)):
                    for comb, (e, spt) in enumerate(zip(eff, spt_combs)):
                        a = np.mean(atom[spt], axis=0)
                        self.fim += e * a
            else:
                for c, (eff, atom) in enumerate(zip(self.efforts, self.atomic_fims)):
                    for spt, (e, a) in enumerate(zip(eff, atom)):
                        self.fim += e * a

        finish = time()

        try:
            if isinstance(self.fim, cp.Expression):
                if np.all(self.fim.value == 0):
                    return np.array([0])
            else:
                if np.all(self.fim == 0):
                    return np.array([0])
        except AttributeError:
            if isinstance(self.fim, cp.expressions.expression.Expression):
                if np.all(self.fim.value == 0):
                    return np.array([0])
            else:
                if np.all(self.fim == 0):
                    return np.array([0])

        self.evaluate_estimability_index()

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

    def eval_pim(self, efforts: NDArray[np.float64] | Any, vector: bool = False) -> NDArray[np.float64]:
        if self._optimization_package == "cvxpy":
            raise NotImplementedError

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
        return self._simulator(ti_controls, tv_controls, theta, sampling_times)

    def _transform_efforts(self):
        if self._unconstrained_form:
            if not self._efforts_transformed:
                self.efforts = np.square(self.efforts)
                self.efforts /= np.sum(self.efforts)
                self._efforts_transformed = True
                if self._verbose >= 3:
                    print("Efforts transformed.")

        return self.efforts
