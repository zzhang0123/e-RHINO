"""System temperature and receiver chain — PLACEHOLDERS.

Real physics to come (generic single-dish receiver model):
- ``SystemTemperatureOperator``: receiver temperature, ground spill-over,
  atmospheric contribution (instrument-specific refinements such as
  noise-wave parameters come later as concrete configurations).
- ``ReceiverOperator``: frequency-dependent bandpass, and eventually
  reflection/impedance-mismatch effects along the signal chain.
"""

from typing import ClassVar

import jax

from erhino.core.errors import StateValidationError
from erhino.core.operator import AbstractOperator
from erhino.core.state import State


class SystemTemperatureOperator(AbstractOperator):
    """Add a system-temperature offset [K] to ``state.data`` (placeholder).

    Attributes:
        t_sys: system temperature — differentiable scalar or ``(n_freq,)`` array.
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    t_sys: jax.Array

    def __call__(self, state: State) -> State:
        # Scalar broadcasts everywhere; (n_freq,) broadcasts along the last axis.
        return state.with_data(state.data + self.t_sys)


class ReceiverOperator(AbstractOperator):
    """Apply a frequency-dependent bandpass to ``state.data`` (placeholder).

    Attributes:
        bandpass: ``(n_freq,)`` dimensionless bandpass — differentiable.
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    bandpass: jax.Array

    def __call__(self, state: State) -> State:
        if self.bandpass.shape[-1] != state.data.shape[-1]:
            raise StateValidationError(
                f"bandpass has {self.bandpass.shape[-1]} channels but data has "
                f"{state.data.shape[-1]}."
            )
        return state.with_data(state.data * self.bandpass[None, :])
