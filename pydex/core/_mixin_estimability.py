from __future__ import annotations

from pickle import dump
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    _save_sensitivities: bool
    _compute_sensitivities: bool
    _model_parameters_changed: bool
    _candidates_changed: bool
    _sensitivity_is_normalized: bool
    _optimization_package: str
    _trim_fim: bool
    _verbose: int

    n_c: int
    n_spt: int
    n_m_r: int
    n_mp: int
    n_r: int

    sensitivities: NDArray[np.float64] | None
    normalized_sensitivity: NDArray[np.float64]
    responses_scales: NDArray[np.float64] | None
    response: NDArray[np.float64] | None
    measurable_responses: list[int]
    model_parameters: NDArray[np.float64]
    efforts: NDArray[np.float64]
    fim: Any

    estimable_columns: NDArray[np.intp]
    estimable_model_parameters: NDArray[np.intp]
    estimability: Any


class DesignerEstimability:

    def estimability_study(self, base_step: float | None = None,
                           step_ratio: float | None = None,
                           num_steps: int | None = None,
                           estimable_tolerance: float = 0.04, write: bool = False,
                           save_sensitivities: bool = False,
                           normalize: bool = False) -> NDArray[np.intp]:
        self._save_sensitivities = save_sensitivities
        self._compute_sensitivities = self._model_parameters_changed
        self._compute_sensitivities = self._compute_sensitivities or self._candidates_changed
        self._compute_sensitivities = self._compute_sensitivities or self.sensitivities is None

        if self._compute_sensitivities:
            self.eval_sensitivities(
                base_step=base_step,
                step_ratio=step_ratio,
            )
        if normalize:
            self.normalize_sensitivities()
        else:
            self.normalized_sensitivity = self.sensitivities / self.responses_scales[None, None, :, None]

        z = self.normalized_sensitivity[:, :, self.measurable_responses, :].reshape(
            self.n_spt * self.n_m_r * self.n_c, self.n_mp)

        z_col_mag = np.nansum(np.power(z, 2), axis=0)
        next_estim_param = np.argmax(z_col_mag)
        self.estimable_columns = np.array([next_estim_param])
        self.estimability = [z_col_mag[next_estim_param]]
        finished = False
        while not finished:
            x_l = z[:, self.estimable_columns]
            z_theta = np.linalg.inv(x_l.T.dot(x_l)).dot(x_l.T).dot(z)
            z_hat = x_l.dot(z_theta)
            r = z - z_hat
            r_col_mag = np.nansum(np.power(r, 2), axis=0)
            next_estim_param = np.argmax(r_col_mag)
            if r_col_mag[next_estim_param] <= estimable_tolerance:
                if write:
                    fn = f"estimability_{self.n_c}_cand"
                    fp = self._generate_result_path(fn, "pkl")
                    dump(self.estimable_columns, open(fp, 'wb'))
                print(
                    f'Identified estimable parameters are: '
                    f'{np.array2string(self.estimable_columns,separator=", ")}'
                )
                with np.printoptions(precision=1):
                    print(
                        f"Degree of Estimability: "
                        f"{np.array2string(np.asarray(self.estimability),separator=', ')}"
                    )
                return self.estimable_columns
            self.estimability.append(r_col_mag[next_estim_param])
            self.estimable_columns = np.append(self.estimable_columns, next_estim_param)
        return self.estimable_columns  # unreachable: loop exits via return inside

    def estimability_study_fim(self, save_sensitivities=False):
        self._save_sensitivities = save_sensitivities
        self.efforts = np.ones((self.n_c, self.n_spt))
        self.eval_fim(self.efforts)
        print(f"Estimable parameters: {self.estimable_model_parameters}")
        with np.printoptions(precision=1):
            print(f"Degree of Estimability: {self.estimability}")
        return self.estimable_model_parameters, self.estimability

    def evaluate_estimability_index(self):
        self.estimable_model_parameters = np.array([])
        self.estimability = np.array([])
        if self._optimization_package == 'cvxpy':
            try:
                fim_value = self.fim.value
            except AttributeError:
                fim_value = self.fim
        else:
            fim_value = self.fim
        try:
            if isinstance(self.fim, cp.Expression):
                if np.all(self.fim.value == 0):
                    return
            else:
                if np.all(self.fim == 0):
                    return
        except AttributeError:
            if isinstance(self.fim, cp.expressions.expression.Expression):
                if np.all(self.fim.value == 0):
                    return
            else:
                if np.all(self.fim == 0):
                    return

        for i, row in enumerate(fim_value):
            if not np.allclose(row, 0.0):
                self.estimable_model_parameters = np.append(
                    self.estimable_model_parameters, i)
                self.estimability = np.append(self.estimability,
                                              np.sqrt(np.inner(row, row)))
        self.estimable_model_parameters = self.estimable_model_parameters.astype(int)

        if self._trim_fim:
            if len(self.estimable_model_parameters) == 0:
                self.fim = np.array([0])
            else:
                self.fim = self.fim[
                    np.ix_(self.estimable_model_parameters, self.estimable_model_parameters)
                ]

    def normalize_sensitivities(self, overwrite_unnormalized=False):
        assert not np.allclose(self.model_parameters,
                               0), 'At least one nominal model parameter value is ' \
                                   'equal to 0, cannot normalize sensitivities. ' \
                                   'Consider re-estimating your parameters or ' \
                                   're-parameterize your model.'

        # normalize parameter values
        self.normalized_sensitivity = np.multiply(self.sensitivities,
                                                  self.model_parameters[None, None, None,
                                                  :])
        if self.responses_scales is None:
            if self._verbose >= 0:
                print(
                    'Scale for responses not given, using raw prediction values to '
                    'normalize sensitivities; '
                    'likely to fail (e.g. if responses are near 0). Recommend: provide '
                    'designer with scale '
                    'info through: "designer.responses_scale = <your_scale_array>."')
            if self.response is None:
                self.simulate_candidates(store_predictions=True)
            # normalize response values
            self.normalized_sensitivity = np.divide(
                self.normalized_sensitivity,
                self.response[:, :, :, None],
            )
        else:
            assert isinstance(self.responses_scales,
                              np.ndarray), "Please specify responses_scales as a 1D " \
                                           "numpy array."
            assert self.responses_scales.size == self.n_r, 'Length of responses scales ' \
                                                           'those which are measurable ' \
                                                           'and not).)'
            self.normalized_sensitivity = np.divide(self.normalized_sensitivity,
                                                    self.responses_scales[None, None, :,
                                                    None])
        if overwrite_unnormalized:
            self.sensitivities = self.normalized_sensitivity
            self._sensitivity_is_normalized = True
            return self.sensitivities
        return self.normalized_sensitivity
