"""Environmental contributions (elements taxonomy: "Environmental").

Ionosphere distorts the astrophysical signal (apply after the sky sum);
ground pickup and RFI are additive terrestrial contributions (branches of the
antenna-temperature :class:`~erhino.core.combinators.SumOperator`).
"""

from erhino.radio.environment.ground import GroundPickupOperator
from erhino.radio.environment.ionosphere import IonosphereOperator
from erhino.radio.environment.rfi import RFIOperator

__all__ = [
    "GroundPickupOperator",
    "IonosphereOperator",
    "RFIOperator",
]
