"""Generic single-dish radio telescope operators (placeholder physics for now).

The forward chain of a single-dish instrument emerges as one Pipeline::

    Sky -> Beam -> SystemTemperature -> Receiver -> Gain -> Noise -> ADC -> Backend

Every operator is a trivial-but-runnable placeholder that establishes the
contract. The real physics will be ported from limTOD (single-dish TOD
simulation, itself to be rewritten in JAX + Equinox) and the related family —
see DESIGN.md for the roadmap. Instrument-specific parameters (e.g. RHINO's)
enter later as concrete operator configurations, never as framework
assumptions.
"""

from erhino.radio.adc import ADCOperator
from erhino.radio.backend import BackendOperator
from erhino.radio.beam import BeamOperator
from erhino.radio.gain import GainOperator
from erhino.radio.noise import NoiseOperator
from erhino.radio.receiver import ReceiverOperator, SystemTemperatureOperator
from erhino.radio.sky import SkyOperator

__all__ = [
    "ADCOperator",
    "BackendOperator",
    "BeamOperator",
    "GainOperator",
    "NoiseOperator",
    "ReceiverOperator",
    "SkyOperator",
    "SystemTemperatureOperator",
]
