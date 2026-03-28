from __future__ import annotations

import sys
import __main__ as main
import dill
from datetime import datetime
from os import getcwd, path, makedirs
from pickle import dump, load
from typing import TYPE_CHECKING, Any

from pydex.core.logger import Logger
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    # Attributes that DesignerIO reads/writes, defined in Designer.__init__
    result_dir_daily: Any
    result_dir: Any
    oed_result: Any
    run_no: int
    n_c: int
    n_spt: int
    n_r: int
    n_mp: int
    n_m_r: int
    measurable_responses: Any
    ti_controls_candidates: Any
    tv_controls_candidates: Any
    sampling_times_candidates: Any
    model_parameters: Any
    efforts: Any
    sensitivities: Any
    atomic_fims: Any
    pb_atomic_fims: Any
    _pseudo_bayesian: bool
    _optimization_time: float
    _sensitivity_analysis_time: float
    _current_criterion: Any
    _criterion_value: Any
    _optimization_package: Any
    _optimizer: Any
    _pseudo_bayesian_type: Any
    _opt_sampling_times: bool
    _regularize_fim: Any
    _n_spt_spec: int
    _candidates_changed: bool
    _model_parameters_changed: bool


class DesignerIO:

    def start_logging(self) -> None:
        fn = f"log"
        fp = self._generate_result_path(fn, "txt")
        sys.stdout = Logger(file_path=fp)

    def stop_logging(self) -> None:
        sys.stdout = sys.__stdout__

    @staticmethod
    def show_plots() -> None:
        plt.show()

    # saving, loading, writing
    def load_oed_result(self, result_path: str) -> None:
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
        self._optimization_package = oed_result["optimization_package"]
        self._optimizer = oed_result["optimizer"]
        self._pseudo_bayesian = oed_result["pseudo_bayesian"]
        self._pseudo_bayesian_type = oed_result["pseudo_bayesian_type"]
        self._opt_sampling_times = oed_result["optimize_sampling_times"]
        self._regularize_fim = oed_result["regularized"]
        self._n_spt_spec = oed_result["n_spt_spec"]
        self._candidates_changed = False
        self._model_parameters_changed = False

    def create_result_dir(self) -> None:
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

    def write_oed_result(self) -> None:
        fn = f"{self.oed_result['optimality_criterion']:s}_oed_result"
        fp = self._generate_result_path(fn, "pkl")
        dump(self.oed_result, open(fp, "wb"))

    def save_state(self) -> None:
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

    def load_state(self, designer_path: str) -> None:
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

    def save_responses(self) -> None:
        # TODO: implement save responses
        pass

    def load_sensitivity(self, sens_path: str) -> Any:
        self.sensitivities = load(open(getcwd() + "/" + sens_path, "rb"))
        self._model_parameters_changed = False
        self._candidates_changed = False
        return self.sensitivities

    def load_atomics(self, atomic_path: str) -> Any:
        with open(getcwd() + atomic_path, "rb") as file:
            if self._pseudo_bayesian:
                self.pb_atomic_fims = load(file)
            else:
                self.atomic_fims = load(file)
        self._model_parameters_changed = False
        self._candidates_changed = False
        return self.atomic_fims

    def _generate_result_path(self, name: str, extension: str, iteration: int | None = None) -> str:
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
