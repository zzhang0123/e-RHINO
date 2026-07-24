"""Environmental contributions (elements taxonomy: "Environmental").

Ionosphere distorts the astrophysical signal (apply after the sky sum);
ground pickup and RFI are additive terrestrial contributions (branches of the
antenna-temperature :class:`~dirt.core.combinators.SumOperator`).
"""

from dirt.radio.environment.ground import GroundPickupOperator
from dirt.radio.environment.ionosphere import IonosphereOperator
from dirt.radio.environment.rfi import RFIOperator

__all__ = [
    "GroundPickupOperator",
    "IonosphereOperator",
    "RFIOperator",
]
