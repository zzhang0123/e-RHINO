"""Astrophysical sky components (elements taxonomy: "Astrophysical").

Compose additively with :class:`~erhino.core.combinators.SumOperator`::

    sky = SumOperator(
        GlobalSignalOperator(...), ForegroundOperator(...), PointSourceOperator(...),
        names=("signal", "foregrounds", "point_sources"),
    )

``SkyOperator`` (uniform brightness) remains as the simplest single-component
placeholder for quick tests and demos.
"""

from erhino.radio.sky.foregrounds import ForegroundOperator
from erhino.radio.sky.global_signal import GlobalSignalOperator
from erhino.radio.sky.model import AbstractSkyModel, PowerLawSkyModel, UniformSkyModel
from erhino.radio.sky.native import NativeLimTODProjector
from erhino.radio.sky.point_sources import PointSourceOperator
from erhino.radio.sky.projection import (
    AbstractSkyProjector,
    LimTODProjector,
    MatrixProjector,
    MModeProjector,
)
from erhino.radio.sky.source import SkySourceOperator
from erhino.radio.sky.uniform import SkyOperator

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
