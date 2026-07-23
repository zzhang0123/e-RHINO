"""Generic single-dish radio telescope operators (placeholder physics for now).

Organized by the element taxonomy (see ``DESIGN.md``):

- ``erhino.radio.sky`` — astrophysical: 21 cm global signal, foregrounds,
  point sources (+ the simplest uniform ``SkyOperator``).
- ``erhino.radio.environment`` — ionosphere, ground pickup, RFI.
- ``erhino.radio.instrument`` — beam, system temperature, noise-wave /
  reflection terms, bandpass, gain, CW calibration tone, thermal noise,
  self-generated EMI, digitisation.
- ``erhino.radio.backend`` — flagging, averaging.

A forward model composes them with the two core combinators::

    astro = Pipeline(SumOperator(signal, foregrounds, point_sources), ionosphere)
    t_ant = SumOperator(astro, ground, rfi)
    twin  = Pipeline(t_ant, beam, tsys, noise_wave, cw_tone, bandpass, gain,
                     noise, emi, adc, flagging, averaging)

Every operator is a trivial-but-runnable placeholder that establishes the
contract. The real physics will be ported from limTOD (single-dish TOD
simulation, itself to be rewritten in JAX + Equinox) and the related family —
see DESIGN.md for the roadmap. Instrument-specific parameters (e.g. RHINO's)
enter later as concrete operator configurations, never as framework
assumptions.
"""

from erhino.radio.backend import BackendOperator, FlaggingOperator
from erhino.radio.environment import (
    GroundPickupOperator,
    IonosphereOperator,
    RFIOperator,
)
from erhino.radio.instrument import (
    ADCOperator,
    BeamOperator,
    CWCalibrationOperator,
    EMIOperator,
    GainOperator,
    NoiseOperator,
    NoiseWaveOperator,
    ReceiverOperator,
    SystemTemperatureOperator,
)
from erhino.radio.sky import (
    ForegroundOperator,
    GlobalSignalOperator,
    PointSourceOperator,
    SkyOperator,
)

__all__ = [
    "ADCOperator",
    "BackendOperator",
    "BeamOperator",
    "CWCalibrationOperator",
    "EMIOperator",
    "FlaggingOperator",
    "ForegroundOperator",
    "GainOperator",
    "GlobalSignalOperator",
    "GroundPickupOperator",
    "IonosphereOperator",
    "NoiseOperator",
    "NoiseWaveOperator",
    "PointSourceOperator",
    "RFIOperator",
    "ReceiverOperator",
    "SkyOperator",
    "SystemTemperatureOperator",
]
