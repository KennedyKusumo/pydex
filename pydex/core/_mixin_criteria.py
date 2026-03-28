from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    _pseudo_bayesian: bool
    _pseudo_bayesian_type: str | int | None
    _optimization_package: str
    _fd_jac: bool
    _cvar_problem: bool
    _verbose: int
    _model_parameters_changed: bool
    _candidates_changed: bool
    _candidates_swapped: bool

    n_c: int
    n_c_go: int
    n_spt: int
    n_spt_go: int
    n_tic: int
    n_tic_go: int
    n_r: int
    n_r_go: int
    n_e: int
    n_scr: int

    fim: NDArray[np.float64]
    atomic_fims: Any
    scr_fims: list[Any]
    pvars: Any
    phi: Any
    efforts: NDArray[np.float64]
    sensitivities: Any
    go_sensitivities: Any
    go_sample_sensitivities_done: bool
    _go_simulator: Any
    _simulator: Any
    error_cov: Any
    go_error_cov: Any

    _ticc: Any
    go_tic: Any
    _tvcc: Any
    go_tvc: Any
    _sptc: Any
    go_spt: Any

    old_tic_cands: Any
    old_sensitivities: Any


class DesignerCriteria:

    def compute_criterion_value(self, criterion: Callable[..., Any], decimal_places: int = 3) -> Any:
        crit_val = criterion(self.efforts)
        try:
            crit_val = crit_val.value
        except AttributeError:
            pass
        if self._verbose >= 2:
            print(f"{criterion.__name__}: {crit_val:.{decimal_places}E}")
        return crit_val

    # -------------------------------------------------------------------------
    # Public criteria
    # -------------------------------------------------------------------------

    # calibration-oriented
    def d_opt_criterion(self, efforts: NDArray[np.float64] | Any) -> Any:
        """ it is a PSD criterion, with exponential cone """
        if self._pseudo_bayesian:
            return self._pb_d_opt_criterion(efforts)
        else:
            return self._d_opt_criterion(efforts)

    def a_opt_criterion(self, efforts: NDArray[np.float64] | Any) -> Any:
        """ it is a PSD criterion """
        if self._pseudo_bayesian:
            return self._pb_a_opt_criterion(efforts)
        else:
            return self._a_opt_criterion(efforts)

    def e_opt_criterion(self, efforts: NDArray[np.float64] | Any) -> Any:
        """ it is a PSD criterion """
        if self._pseudo_bayesian:
            return self._pb_e_opt_criterion(efforts)
        else:
            return self._e_opt_criterion(efforts)

    # prediction-oriented
    def dg_opt_criterion(self, efforts: NDArray[np.float64] | Any) -> Any:
        if self._pseudo_bayesian:
            return self._pb_dg_opt_criterion(efforts)
        else:
            return self._dg_opt_criterion(efforts)

    def di_opt_criterion(self, efforts: NDArray[np.float64] | Any) -> Any:
        if self._pseudo_bayesian:
            return self._pb_di_opt_criterion(efforts)
        else:
            return self._di_opt_criterion(efforts)

    def ag_opt_criterion(self, efforts: NDArray[np.float64] | Any) -> Any:
        if self._pseudo_bayesian:
            return self._pb_ag_opt_criterion(efforts)
        else:
            return self._ag_opt_criterion(efforts)

    def ai_opt_criterion(self, efforts: NDArray[np.float64] | Any) -> Any:
        if self._pseudo_bayesian:
            return self._pb_ai_opt_criterion(efforts)
        else:
            return self._ai_opt_criterion(efforts)

    def eg_opt_criterion(self, efforts: NDArray[np.float64] | Any) -> Any:
        if self._pseudo_bayesian:
            return self._pb_eg_opt_criterion(efforts)
        else:
            return self._eg_opt_criterion(efforts)

    def ei_opt_criterion(self, efforts: NDArray[np.float64] | Any) -> Any:
        if self._pseudo_bayesian:
            return self._pb_ei_opt_criterion(efforts)
        else:
            return self._ei_opt_criterion(efforts)

    # goal-oriented for design space
    def vdi_criterion(self, efforts: NDArray[np.float64] | Any) -> Any:
        if self._pseudo_bayesian:
            raise NotImplementedError("Pseudo-bayesian designs for the VDI criterion not"
                                      "implemented yet, keep an eye out in future "
                                      "releases.")
        else:
            return self._vdi_opt_criterion(efforts)

    def _vdi_opt_criterion(self, efforts):
        if self._optimization_package == "cvxpy":
            raise NotImplementedError("CVXPY unavailable for vdi_opt.")

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
        di_opt: Any = np.sum(di_opts)

        if self._fd_jac:
            return di_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ei_opt unavailable.")

    def eval_pim_for_v_opt(self, efforts, vector=False):
        if self._optimization_package == "cvxpy":
            raise NotImplementedError

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
        self._simulator, self._go_simulator = self._go_simulator, self._simulator
        self.go_sensitivities, self.sensitivities = self.sensitivities, self.go_sensitivities
        self.error_cov, self.go_error_cov = self.go_error_cov, self.error_cov
        self.initialize(verbose=self._verbose)
        self._model_parameters_changed = False

    def _revert_candidates(self):
        self.ti_controls_candidates = self.old_tic_cands  # type: ignore[attr-defined]
        self.tv_controls_candidates = self.old_tic_cands  # type: ignore[attr-defined]
        self.spt_controls_candidates = self.old_tic_cands  # type: ignore[attr-defined]

        self.sensitivities = self.old_sensitivities  # type: ignore[attr-defined]
        if self._go_simulator:
            self._simulator, self._go_simulator = self._go_simulator, self._simulator
        self.initialize(verbose=0)

        self._model_parameters_changed = False
        self._candidates_changed = False

    # experimental
    def u_opt_criterion(self, efforts):
        self.eval_fim(efforts, self.model_parameters)
        return -np.sum(np.multiply(self.fim, self.fim))

    # risk-averse
    def cvar_d_opt_criterion(self, fim):
        self._cvar_problem = True

        if self._pseudo_bayesian:
            # old behaviour
            if False:
                self._eval_fim(efforts, mp)  # type: ignore[name-defined]
                self._model_parameters_changed = True
                if self.fim.size == 1:
                    return -self.fim
                else:
                    return -cp.log_det(self.fim)
                # new behaviour
            if True:
                if self.fim.size == 1:
                    return -fim
                else:
                    return -cp.log_det(fim)
        else:
            raise SyntaxError(
                "CVaR criterion cannot be used for non Pseudo-bayesian problems, please "
                "ensure that you passed in the correct 2D numpy array as "
                "model_parameters."
            )

    # -------------------------------------------------------------------------
    # Local (private) criteria
    # -------------------------------------------------------------------------

    # calibration-oriented
    def _d_opt_criterion(self, efforts):
        """ it is a PSD criterion, with exponential cone """
        self.eval_fim(efforts)

        if self.fim.size == 1:
            if self._optimization_package == "scipy":
                d_opt = -self.fim
                if self._fd_jac:
                    return np.squeeze(d_opt)
                else:
                    jac = -np.array([
                        1 / self.fim * m
                        for m in self.atomic_fims
                    ])
                    return d_opt, jac
            elif self._optimization_package == 'cvxpy':
                return -self.fim

        if self._optimization_package == "scipy":
            sign, d_opt = np.linalg.slogdet(self.fim)
            if self._fd_jac:
                if sign == 1:
                    return -d_opt
                else:
                    return np.inf
            else:
                fim_inv = np.linalg.inv(self.fim)
                jac = -np.array([
                    np.sum(fim_inv.T * m)
                    for m in self.atomic_fims
                ])
                if sign == 1:
                    return -d_opt, jac
                else:
                    return np.inf, jac

        elif self._optimization_package == 'cvxpy':
            return -cp.log_det(self.fim)

    def _a_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        self.eval_fim(efforts)

        if self.fim.size == 1:
            if self._optimization_package == "scipy":
                if self._fd_jac:
                    return -self.fim
                else:
                    jac = np.array([
                        m for m in self.atomic_fims
                    ])
                    return -self.fim, jac
            elif self._optimization_package == "cvxpy":
                return -self.fim

        if self._optimization_package == "scipy":
            if self._fd_jac:
                eigvals = np.linalg.eigvalsh(self.fim)
                if np.all(eigvals > 0):
                    a_opt: Any = np.sum(1 / eigvals)
                else:
                    a_opt = 0
                return a_opt
            else:
                jac = np.zeros(self.n_e)
                try:
                    fim_inv = np.linalg.inv(self.fim)
                    a_opt = fim_inv.trace()
                    if not self._fd_jac:
                        jac = -np.array([
                            np.sum((fim_inv @ fim_inv) * m) for m in self.atomic_fims
                        ])
                except np.linalg.LinAlgError:
                    a_opt = 0
                return a_opt, jac

        elif self._optimization_package == 'cvxpy':
            try:
                return cp.matrix_frac(np.identity(self.fim.shape[0]), self.fim)
            except ValueError:
                return cp.matrix_frac(np.identity(self.fim.shape[0]).tolist(), self.fim.tolist())

    def _e_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        self.eval_fim(efforts)

        if self.fim.size == 1:
            return -self.fim

        if self._optimization_package == "scipy":
            if self._fd_jac:
                return -np.linalg.eigvalsh(self.fim).min()
            else:
                raise NotImplementedError  # TODO: implement analytic jac for e-opt
        elif self._optimization_package == 'cvxpy':
            return -cp.lambda_min(self.fim)

    # prediction-oriented
    def _dg_opt_criterion(self, efforts):
        if self._optimization_package == "cvxpy":
            raise NotImplementedError("CVXPY unavailable for dg_opt.")

        self.eval_pim(efforts)
        # dg_opt: max det of the pvar matrix over candidates and sampling times
        dg_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                sign, temp_dg = np.linalg.slogdet(pvar)
                if sign != 1:
                    temp_dg = np.inf
                dg_opts[c, spt] = sign * np.exp(temp_dg)
        dg_opt: Any = np.nanmax(dg_opts)

        if self._fd_jac:
            return dg_opt
        else:
            raise NotImplementedError("Analytic Jacobian for dg_opt unavailable.")

    def _di_opt_criterion(self, efforts):
        if self._optimization_package == "cvxpy":
            raise NotImplementedError("CVXPY unavailable for di_opt.")

        self.eval_pim(efforts)
        # di_opt: average det of the pvar matrix over candidates and sampling times
        dg_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                sign, temp_dg = np.linalg.slogdet(pvar)
                if sign != 1:
                    temp_dg = np.inf
                dg_opts[c, spt] = temp_dg
        dg_opt: Any = np.nansum(dg_opts)

        if self._fd_jac:
            return dg_opt
        else:
            raise NotImplementedError("Analytic Jacobian for di_opt unavailable.")

    def _ag_opt_criterion(self, efforts):
        if self._optimization_package == "cvxpy":
            raise NotImplementedError("CVXPY unavailable for ag_opt.")

        self.eval_pim(efforts)
        # ag_opt: max trace of the pvar matrix over candidates and sampling times
        ag_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.trace(pvar)
                ag_opts[c, spt] = temp_dg
        ag_opt: Any = np.nanmax(ag_opts)

        if self._fd_jac:
            return ag_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ag_opt unavailable.")

    def _ai_opt_criterion(self, efforts):
        if self._optimization_package == "cvxpy":
            raise NotImplementedError("CVXPY unavailable for ai_opt.")

        self.eval_pim(efforts)
        # ai_opt: average trace of the pvar matrix over candidates and sampling times
        ai_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.trace(pvar)
                ai_opts[c, spt] = temp_dg
        ag_opt: Any = np.nansum(ai_opts)

        if self._fd_jac:
            return ag_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ai_opt unavailable.")

    def _eg_opt_criterion(self, efforts):
        if self._optimization_package == "cvxpy":
            raise NotImplementedError("CVXPY unavailable for eg_opt.")

        self.eval_pim(efforts)
        # eg_opt: max of the max_eigenval of the pvar matrix over candidates and sampling times
        eg_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.linalg.eigvals(pvar).max()
                eg_opts[c, spt] = temp_dg
        eg_opt: Any = np.nanmax(eg_opts)

        if self._fd_jac:
            return eg_opt
        else:
            raise NotImplementedError("Analytic Jacobian for eg_opt unavailable.")

    def _ei_opt_criterion(self, efforts):
        if self._optimization_package == "cvxpy":
            raise NotImplementedError("CVXPY unavailable for ei_opt.")

        self.eval_pim(efforts)
        # ei_opts: average of the max_eigenval of the pvar matrix over candidates and sampling times
        ei_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.linalg.eigvals(pvar).max()
                ei_opts[c, spt] = temp_dg
        ei_opt: Any = np.nansum(ei_opts)

        if self._fd_jac:
            return ei_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ei_opt unavailable.")

    # -------------------------------------------------------------------------
    # Pseudo-Bayesian criteria
    # -------------------------------------------------------------------------

    # calibration-oriented
    def _pb_d_opt_criterion(self, efforts):
        """ it is a PSD criterion, with exponential cone """
        self.eval_fim(efforts)

        if self._optimization_package == "scipy":
            if self._fd_jac:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = np.mean([fim for fim in self.scr_fims], axis=0)
                    sign, d_opt = np.linalg.slogdet(avg_fim)
                    if sign != 1:
                        return np.inf
                    else:
                        return -d_opt
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    d_opt = 0
                    for fim in self.scr_fims:
                        sign, scr_d_opt = np.linalg.slogdet(fim)
                        if sign != 1:
                            scr_d_opt = np.inf
                        d_opt += scr_d_opt
                    return -d_opt / self.n_scr
            else:
                raise NotImplementedError(
                    "Analytical Jacobian unimplemented for Pseudo-bayesian D-optimal."
                )

        elif self._optimization_package == 'cvxpy':
            if np.any([fim.shape == (1, 1) for fim in self.scr_fims]):
                return cp.sum([-fim for fim in self.scr_fims]) / self.n_scr
            else:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = cp.sum([fim for fim in self.scr_fims], axis=0) / self.n_scr
                    return -cp.log_det(avg_fim)
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    return cp.sum([
                        -cp.log_det(fim) for fim in self.scr_fims
                    ],
                    axis=0,
                    ) / self.n_scr

    def _pb_a_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        self.eval_fim(efforts)

        if self._optimization_package == "scipy":
            if self._fd_jac:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    a_opt = np.linalg.inv(
                        np.mean([fim for fim in self.scr_fims], axis=0)
                    ).trace()
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    np.mean([
                        np.linalg.inv(fim).trace()
                        for fim in self.scr_fims
                    ])
            else:
                raise NotImplementedError(
                    "Analytical Jacobian unimplemented for Pseudo-bayesian D-optimal."
                )

        elif self._optimization_package == 'cvxpy':
            if np.any([fim.shape == (1, 1) for fim in self.scr_fims]):
                return cp.sum([-fim for fim in self.scr_fims]) / self.n_scr
            else:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = cp.sum([fim for fim in self.scr_fims], axis=0) / self.n_scr
                    return cp.matrix_frac(np.identity(avg_fim.shape[0]), avg_fim)
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    return cp.sum([
                        cp.matrix_frac(np.identity(fim.shape[0]), fim)
                        for fim in self.scr_fims
                    ]) / self.n_scr

    def _pb_e_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        self.eval_fim(efforts)

        if self._optimization_package == "scipy":
            if self._fd_jac:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = np.sum([fim for fim in self.scr_fims], axis=0) / self.n_scr
                    return -np.linalg.eigvalsh(avg_fim).min()
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    return np.sum([
                        -np.linalg.eigvalsh(fim).min()
                        for fim in self.scr_fims
                    ]) / self.n_scr
            else:
                raise NotImplementedError(
                    "Analytical Jacobian unimplemented for Pseudo-bayesian D-optimal."
                )

        elif self._optimization_package == 'cvxpy':
            if np.any([fim.shape == (1, 1) for fim in self.scr_fims]):
                return cp.sum([-fim for fim in self.scr_fims])
            else:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = cp.sum([fim for fim in self.scr_fims], axis=0) / self.n_scr
                    return -cp.lambda_min(avg_fim)
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    return cp.sum([
                        -cp.lambda_min(fim)
                        for fim in self.scr_fims
                    ]) / self.n_scr

    # prediction-oriented (not yet implemented for pseudo-Bayesian)
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
