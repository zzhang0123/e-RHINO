"""Radio-telescope digital-twin operators (placeholder physics for now).

The forward chain of a single-dish instrument emerges as one Pipeline::

    Sky -> Beam -> SystemTemperature -> Receiver -> Gain -> Noise -> ADC -> Backend

Every operator documents the real physics it will eventually hold; the current
bodies are trivial-but-runnable placeholders that establish the contracts.
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
