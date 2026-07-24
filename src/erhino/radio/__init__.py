"""Generic single-dish radio telescope operators (placeholder physics for now).

Organized by the element taxonomy (see ``DESIGN.md``):

- ``erhino.radio.sky`` — astrophysical: 21 cm global signal, foregrounds,
  point sources (+ the simplest uniform ``SkyOperator``).
- ``erhino.radio.environment`` — ionosphere, ground pickup, RFI.
- ``erhino.radio.instrument`` — beam, system temperature, noise-wave /
  reflection terms, bandpass, gain, CW calibration tone, thermal noise,
  self-generated EMI, digitisation.
- ``erhino.radio.backend`` — flagging, averaging.

A forward model composes them with the two core combinators, following the
canonical signal-path graph (``erhino.radio.graph``; RFI enters as a
pre-beam field, ground pickup as a post-beam effective temperature)::

    astro = Pipeline(SumOperator(signal, foregrounds, point_sources), ionosphere)
    field = Pipeline(SumOperator(astro, rfi_field), beam)
    t_ant = SumOperator(field, ground_pickup)
    twin  = Pipeline(t_ant, atmosphere, noise_wave, cw_tone, bandpass, gain,
                     noise, emi, adc, flagging, averaging)

or, equivalently, by just providing the operators::

    twin = assemble(signal, foregrounds, point_sources, ionosphere, rfi_field,
                    beam, ground_pickup, atmosphere, noise_wave, cw_tone,
                    bandpass, gain, noise, emi, adc, flagging, averaging)

Every operator is a trivial-but-runnable placeholder that establishes the
contract. The real physics will be ported from limTOD (single-dish TOD
simulation, itself to be rewritten in JAX + Equinox) and the related family —
see DESIGN.md for the roadmap. Instrument-specific parameters (e.g. RHINO's)
enter later as concrete operator configurations, never as framework
assumptions.
"""

from erhino.radio.backend import BackendOperator, FlaggingOperator, MomentRFIFlaggingOperator
from erhino.radio.environment import (
    GroundPickupOperator,
    IonosphereOperator,
    RFIOperator,
)
from erhino.radio.filters import (
    AbstractLinearFilter,
    FourierBandFilter,
    SiderealFilter,
    SkySpaceFilter,
)
from erhino.radio.instrument import (
    ADCOperator,
    ApplyCalibrationOperator,
    BeamOperator,
    CalLoadOperator,
    CWCalibrationOperator,
    EMIOperator,
    GainOperator,
    NoiseOperator,
    NoiseWaveOperator,
    ReceiverOperator,
    SystemTemperatureOperator,
)
from erhino.radio.sky import (
    AbstractSkyModel,
    AbstractSkyProjector,
    ForegroundOperator,
    GlobalSignalOperator,
    LimTODProjector,
    MatrixProjector,
    MModeProjector,
    PointSourceOperator,
    PowerLawSkyModel,
    SkyOperator,
    SkySourceOperator,
    UniformSkyModel,
)

__all__ = [
    "ADCOperator",
    "AbstractLinearFilter",
    "AbstractSkyModel",
    "AbstractSkyProjector",
    "ApplyCalibrationOperator",
    "BackendOperator",
    "BeamOperator",
    "CWCalibrationOperator",
    "CalLoadOperator",
    "EMIOperator",
    "FlaggingOperator",
    "ForegroundOperator",
    "FourierBandFilter",
    "GainOperator",
    "GlobalSignalOperator",
    "GroundPickupOperator",
    "IonosphereOperator",
    "LimTODProjector",
    "MModeProjector",
    "MatrixProjector",
    "MomentRFIFlaggingOperator",
    "NoiseOperator",
    "NoiseWaveOperator",
    "PointSourceOperator",
    "PowerLawSkyModel",
    "RFIOperator",
    "ReceiverOperator",
    "SiderealFilter",
    "SkyOperator",
    "SkySourceOperator",
    "SkySpaceFilter",
    "SystemTemperatureOperator",
    "UniformSkyModel",
]

from erhino.radio.graph import RADIO_GRAPH, assemble  # noqa: E402  (needs operators above)

__all__ += ["RADIO_GRAPH", "assemble"]

from erhino.radio.graph import _validate_registrations as _v  # noqa: E402

_v()
del _v
