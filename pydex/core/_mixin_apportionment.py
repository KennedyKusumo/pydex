from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    _dynamic_system: bool
    _specified_n_spt: bool
    _save_atomics: bool
    _opt_sampling_times: bool
    _pseudo_bayesian: bool
    _pseudo_bayesian_type: str | int | None
    _cvar_problem: bool
    _constrained_cvar: bool
    _invariant_controls: bool
    _dynamic_controls: bool
    _current_criterion: str
    _criterion_value: Any
    _verbose: int
    _n_spt_spec: int

    n_exp: int
    n_c: int
    n_opt_c: int
    n_mp: int
    n_spt: int
    n_scr: int
    n_factor_sups: int
    max_n_opt_spt: int

    beta: float
    epsilon: float
    rounded_criterion_value: Any

    efforts: NDArray[np.float64]
    apportionments: NDArray[np.float64] | None
    non_trimmed_apportionments: NDArray[np.float64]
    opt_eff: NDArray[np.float64]
    optimal_candidates: list[Any] | None
    sampling_times_candidates: NDArray[np.float64]

    phi: Any
    s: Any
    v: Any


class DesignerApportionment:

    def apportion(self, n_exp: int, method: str = "adams", trimmed: bool = True,
                  compute_actual_efficiency: bool = True) -> NDArray[np.intp] | None:
        self.n_exp = n_exp

        if self._dynamic_system and self._specified_n_spt:
            print(
                "[WARNING]: The apportion method does not support experimental design "
                "problems with specified n_spt yet. Skipping the apportionment."
            )
            return None
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
        non_trimmed_rounded_efforts = self.non_trimmed_apportionments
        self._criterion_value = -getattr(self, self._current_criterion)(self.efforts * n_exp).value
        if compute_actual_efficiency:
            _original_efforts = np.copy(self.efforts)
            try:
                self.rounded_criterion_value = getattr(self, self._current_criterion)(non_trimmed_rounded_efforts).value
            except AttributeError:
                self.rounded_criterion_value = getattr(self, self._current_criterion)(non_trimmed_rounded_efforts)
            if self._current_criterion == "d_opt_criterion":
                efficiency = np.exp(1 / self.n_mp * (-self.rounded_criterion_value - self._criterion_value))
            elif self._current_criterion == "a_opt_criterion":
                efficiency = -self._criterion_value / self.rounded_criterion_value
            elif self._current_criterion == "e_opt_criterion":
                efficiency = -self.rounded_criterion_value / self._criterion_value
            self.efforts = _original_efforts
            efficiency = np.squeeze(efficiency)
            print(f"{'':#^100}")
        if not trimmed:
            self.apportionments = self.non_trimmed_apportionments
        self._save_atomics = _original_save_atomics

        """ Report the obtained apportionment """
        if self._verbose >= 1:
            print(f" Optimal Experiment for {n_exp:d} Runs ".center(100, "#"))
            print(f"{'Obtained on':<40}: {datetime.now()}")
            print(f"{'Criterion (higher is better)':<40}: {self._current_criterion}")
            print(f"{f'Criterion Value with sum(efforts) = {self.n_exp:d}':<40}: {self._criterion_value}")
            print(f"{f'Rounded Criterion with sum(efforts) = {self.n_exp:d}':<40}: {-self.rounded_criterion_value}")
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
            print(f"".center(100, "-"))
            print(
                f"The actual criterion value of the rounded design is "
                f"{efficiency * 100:.2f}% as informative as the continuous design."
            )
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

        return self.apportionments.astype(int) if self.apportionments is not None else None

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
        min_lkhd_ratio: Any = np.nanmin(eff_ratio)
        return min_lkhd_ratio
