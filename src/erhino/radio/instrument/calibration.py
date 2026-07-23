"""CWCalibrationOperator — PLACEHOLDER continuous-wave calibration tone.

Elements: "Calibration signals, which are switched in and out of the signal
path on some pre-defined cycle. Each calibration source has its own signal
shape, particularly in frequency, as well as typical power levels, stability,
and additional reflection and noise contributions."

RHINO's central design choice (paper, Sect. 4): a continuous-wave source
injects a known, narrow-band, large-amplitude signal so the overall gain
level is monitored *continuously*, without Dicke switching.

ORDERING CONSTRAINT: the tone is combined with the antenna signal *before*
the receiver chain — paper Eq. 6: ``P_rec = g(nu,t) (T_ant + T_nw + T_cw)
+ T_n``. This operator must therefore sit BEFORE the bandpass and gain
operators in a pipeline: the tone tracks g(t) drift only if it passes
through the gain (Eqs. 13-16: delta P_cw ~ g(nu_cw, t)).

Real physics to come: the tone's spectral shape, drift/stability, its own
reflection and noise contributions, and the switched reference loads used
for noise-wave calibration (GCR draft). The placeholder injects a constant
amplitude into the channel nearest the tone frequency.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from erhino.core.errors import StateValidationError
from erhino.core.operator import AbstractOperator
from erhino.core.state import State


class CWCalibrationOperator(AbstractOperator):
    """Inject a narrow-band CW tone into the nearest frequency channel.

    Attributes:
        amplitude: tone amplitude [K-equivalent] — differentiable scalar.
        tone_freq: tone frequency [Hz] (static configuration).
    """

    requires: ClassVar[tuple[str, ...]] = ("data", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    amplitude: jax.Array
    tone_freq: float = eqx.field(static=True)

    def __check_init__(self):
        if self.tone_freq <= 0:
            raise StateValidationError(f"tone_freq must be > 0, got {self.tone_freq}.")

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.freq is None:
            raise StateValidationError("CWCalibrationOperator requires state.coords.freq.")
        channel = jnp.argmin(jnp.abs(state.coords.freq - self.tone_freq))
        return state.with_data(state.data.at[:, channel].add(self.amplitude))
