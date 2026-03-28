from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    _dynamic_system: bool
    _opt_sampling_times: bool
    _specified_n_spt: bool

    n_spt: int
    n_opt_c: int
    n_factor_sups: int
    n_spt_sups: int
    max_n_opt_spt: int

    efforts: NDArray[np.float64] | None
    optimal_candidates: list[Any] | None
    ti_controls_candidates: NDArray[np.float64]
    tv_controls_candidates: NDArray[np.float64]
    sampling_times_candidates: NDArray[np.float64]
    spt_candidates_combs: NDArray[np.float64]


class DesignerCandidates:

    def get_optimal_candidates(self, tol: float = 1e-4) -> Any:
        if self.efforts is None:
            raise SyntaxError(
                'Please solve an experiment design before attempting to get optimal '
                'candidates.'
            )

        self._remove_zero_effort_candidates(tol=tol)
        self.optimal_candidates = []

        for i, eff_sp in enumerate(self.efforts):
            optimal: Any
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
            try:
                self.max_n_opt_spt = max(self.max_n_opt_spt, len(opt_cand[4]))
            # catch error when opt_cand[4] is a float
            except TypeError:
                self.max_n_opt_spt = max(self.max_n_opt_spt, 1)

        return self.optimal_candidates

    def _remove_zero_effort_candidates(self, tol):
        self.efforts[self.efforts < tol] = 0
        self.efforts = self.efforts / self.efforts.sum()
        return self.efforts
