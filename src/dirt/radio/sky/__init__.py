"""Astrophysical sky components (elements taxonomy: "Astrophysical").

Compose additively with :class:`~dirt.core.combinators.SumOperator`::

    sky = SumOperator(
        GlobalSignalOperator(...), ForegroundOperator(...), PointSourceOperator(...),
        names=("signal", "foregrounds", "point_sources"),
    )

``SkyOperator`` (uniform brightness) remains as the simplest single-component
placeholder for quick tests and demos.
"""

from dirt.radio.sky.foregrounds import ForegroundOperator
from dirt.radio.sky.global_signal import GlobalSignalOperator
from dirt.radio.sky.model import AbstractSkyModel, PowerLawSkyModel, UniformSkyModel
from dirt.radio.sky.native import NativeLimTODProjector
from dirt.radio.sky.point_sources import PointSourceOperator
from dirt.radio.sky.projection import (
    AbstractSkyProjector,
    LimTODProjector,
    MatrixProjector,
    MModeProjector,
)
from dirt.radio.sky.source import SkySourceOperator
from dirt.radio.sky.uniform import SkyOperator

__all__ = [
    "AbstractSkyModel",
    "AbstractSkyProjector",
    "ForegroundOperator",
    "GlobalSignalOperator",
    "LimTODProjector",
    "MModeProjector",
    "MatrixProjector",
    "NativeLimTODProjector",
    "PointSourceOperator",
    "PowerLawSkyModel",
    "SkyOperator",
    "SkySourceOperator",
    "UniformSkyModel",
]
