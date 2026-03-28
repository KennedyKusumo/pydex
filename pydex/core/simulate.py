from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


class SimulatorBase(ABC):
    """
    Abstract base for all pydex simulator types.

    Subclasses declare which controls and time structure the model expects by
    overriding ``simulate()`` with the appropriate signature. Designer reads the
    three class-level flags directly at ``initialize()`` time instead of
    sniffing argument names.

    The canonical internal call signature used by Designer is always::

        simulator(ti_controls, tv_controls, model_parameters, sampling_times)

    Each named subclass implements ``__call__`` to map this to the user-facing
    ``simulate()`` signature.
    """

    #: True if the model receives sampling times (dynamic/ODE system).
    is_dynamic: bool = False
    #: True if the model receives time-varying controls.
    has_tv_controls: bool = False
    #: True if the model receives time-invariant controls.
    has_ti_controls: bool = True

    @abstractmethod
    def __call__(
        self,
        ti_controls: NDArray[np.float64],
        tv_controls: NDArray[np.float64],
        model_parameters: NDArray[np.float64],
        sampling_times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Internal canonical signature used by all Designer internals."""
        ...


class StaticSimulator(SimulatorBase):
    """
    Simulator for static systems with time-invariant controls.

    User signature::

        def simulate(self, ti_controls, model_parameters): ...
    """

    is_dynamic = False
    has_tv_controls = False
    has_ti_controls = True

    @abstractmethod
    def simulate(
        self,
        ti_controls: NDArray[np.float64],
        model_parameters: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    def __call__(
        self,
        ti_controls: NDArray[np.float64],
        tv_controls: NDArray[np.float64],
        model_parameters: NDArray[np.float64],
        sampling_times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.simulate(ti_controls, model_parameters)


class DynamicTISimulator(SimulatorBase):
    """
    Simulator for dynamic systems with time-invariant controls.

    User signature::

        def simulate(self, ti_controls, sampling_times, model_parameters): ...
    """

    is_dynamic = True
    has_tv_controls = False
    has_ti_controls = True

    @abstractmethod
    def simulate(
        self,
        ti_controls: NDArray[np.float64],
        sampling_times: NDArray[np.float64],
        model_parameters: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    def __call__(
        self,
        ti_controls: NDArray[np.float64],
        tv_controls: NDArray[np.float64],
        model_parameters: NDArray[np.float64],
        sampling_times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.simulate(ti_controls, sampling_times, model_parameters)


class DynamicTVSimulator(SimulatorBase):
    """
    Simulator for dynamic systems with time-varying controls only.

    User signature::

        def simulate(self, tv_controls, sampling_times, model_parameters): ...
    """

    is_dynamic = True
    has_tv_controls = True
    has_ti_controls = False

    @abstractmethod
    def simulate(
        self,
        tv_controls: NDArray[np.float64],
        sampling_times: NDArray[np.float64],
        model_parameters: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    def __call__(
        self,
        ti_controls: NDArray[np.float64],
        tv_controls: NDArray[np.float64],
        model_parameters: NDArray[np.float64],
        sampling_times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.simulate(tv_controls, sampling_times, model_parameters)


class DynamicFullSimulator(SimulatorBase):
    """
    Simulator for dynamic systems with both time-invariant and time-varying controls.

    User signature::

        def simulate(self, ti_controls, tv_controls, sampling_times, model_parameters): ...
    """

    is_dynamic = True
    has_tv_controls = True
    has_ti_controls = True

    @abstractmethod
    def simulate(
        self,
        ti_controls: NDArray[np.float64],
        tv_controls: NDArray[np.float64],
        sampling_times: NDArray[np.float64],
        model_parameters: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    def __call__(
        self,
        ti_controls: NDArray[np.float64],
        tv_controls: NDArray[np.float64],
        model_parameters: NDArray[np.float64],
        sampling_times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.simulate(ti_controls, tv_controls, sampling_times, model_parameters)


class DynamicUncontrolledSimulator(SimulatorBase):
    """
    Simulator for dynamic systems with no explicit controls.

    User signature::

        def simulate(self, sampling_times, model_parameters): ...
    """

    is_dynamic = True
    has_tv_controls = False
    has_ti_controls = False

    @abstractmethod
    def simulate(
        self,
        sampling_times: NDArray[np.float64],
        model_parameters: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    def __call__(
        self,
        ti_controls: NDArray[np.float64],
        tv_controls: NDArray[np.float64],
        model_parameters: NDArray[np.float64],
        sampling_times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.simulate(sampling_times, model_parameters)


class _LegacySimulatorAdapter(SimulatorBase):
    """
    Wraps a plain callable using the legacy argument-name-sniffing convention.

    Created automatically inside Designer._configure_simulator() when a bare
    function is assigned to Designer.simulator. Emits a DeprecationWarning at
    construction time (i.e. at initialize()) so it appears once, not per call.

    Not part of the public API.
    """

    def __init__(self, fn: Callable[..., Any], sig_id: int) -> None:
        import warnings
        warnings.warn(
            f"Assigning a plain function to Designer.simulator is deprecated and will be "
            f"removed in a future release. Wrap your function in the appropriate simulator "
            f"class (StaticSimulator, DynamicTISimulator, DynamicTVSimulator, "
            f"DynamicFullSimulator, or DynamicUncontrolledSimulator). "
            f"See MIGRATION.md for before/after examples.",
            DeprecationWarning,
            stacklevel=4,
        )
        self._fn = fn
        self._sig_id = sig_id
        self.is_dynamic = sig_id in (2, 3, 4, 5)
        self.has_tv_controls = sig_id in (3, 4)
        self.has_ti_controls = sig_id in (1, 2, 4)

    def __call__(
        self,
        ti_controls: NDArray[np.float64],
        tv_controls: NDArray[np.float64],
        model_parameters: NDArray[np.float64],
        sampling_times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self._sig_id == 1:
            return self._fn(ti_controls, model_parameters)
        elif self._sig_id == 2:
            return self._fn(ti_controls, sampling_times, model_parameters)
        elif self._sig_id == 3:
            return self._fn(tv_controls, sampling_times, model_parameters)
        elif self._sig_id == 4:
            return self._fn(ti_controls, tv_controls, sampling_times, model_parameters)
        else:  # sig_id == 5
            return self._fn(sampling_times, model_parameters)
