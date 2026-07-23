"""Backend / data processing (elements taxonomy: flagging, averaging, ...).

Processing steps live in the same Operator formalism as the physics — so the
biases they can introduce ("if the models are slightly wrong") are part of
the differentiable forward model too.
"""

from erhino.radio.backend.averaging import BackendOperator
from erhino.radio.backend.flagging import FlaggingOperator, MomentRFIFlaggingOperator

__all__ = [
    "BackendOperator",
    "FlaggingOperator",
    "MomentRFIFlaggingOperator",
]
