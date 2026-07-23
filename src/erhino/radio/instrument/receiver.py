"""System temperature and receiver chain — PLACEHOLDERS.

Real physics to come (generic single-dish receiver model):
- ``SystemTemperatureOperator``: *sky-side* additive temperatures —
  atmosphere, ground spill-over — which enter *before* the antenna/receiver
  reflection terms and therefore see the ``(1-|Gamma|^2)`` loss. The receiver
  temperature itself does NOT belong here: it enters after the reflection,
  as the noise-wave ``T_0`` (see
  :class:`~erhino.radio.instrument.noise_wave.NoiseWaveOperator`) and the
  post-gain thermal noise ``T_n``.
- ``ReceiverOperator``: frequency-dependent bandpass, and eventually
  reflection/impedance-mismatch effects along the signal chain.
"""

from typing import ClassVar

import jax

from erhino.core.errors import StateValidationError
from erhino.core.operator import AbstractOperator
from erhino.core.state import State


class SystemTemperatureOperator(AbstractOperator):
    """Add a sky-side system-temperature offset [K] to ``state.data`` (placeholder).

    Scope: contributions that arrive *with* the sky signal (atmosphere, ground
    spill) — applied before reflection/noise-wave terms in the chain.

    Attributes:
        t_sys: sky-side temperature — differentiable scalar or ``(n_freq,)`` array.
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
