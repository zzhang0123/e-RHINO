"""Instrumental response chain (elements taxonomy: "Instrumental").

Typical ordering in a forward model (RHINO paper Eq. 6:
``P_rec = g (T_ant + T_nw + T_cw) + T_n`` — the CW tone joins *before*
bandpass and gain so it tracks gain drift)::

    Beam -> SystemTemperature(sky-side) -> NoiseWave -> CWCalibration
         -> Receiver(bandpass) -> Gain -> Noise -> EMI -> ADC
"""

from erhino.radio.instrument.adc import ADCOperator
from erhino.radio.instrument.beam import BeamOperator
from erhino.radio.instrument.calibration import ApplyCalibrationOperator, CWCalibrationOperator
from erhino.radio.instrument.emi import EMIOperator
from erhino.radio.instrument.gain import GainOperator
from erhino.radio.instrument.noise import NoiseOperator
from erhino.radio.instrument.noise_wave import NoiseWaveOperator
from erhino.radio.instrument.receiver import ReceiverOperator, SystemTemperatureOperator

__all__ = [
    "ADCOperator",
    "ApplyCalibrationOperator",
    "BeamOperator",
    "CWCalibrationOperator",
    "EMIOperator",
    "GainOperator",
    "NoiseOperator",
    "NoiseWaveOperator",
    "ReceiverOperator",
    "SystemTemperatureOperator",
]
