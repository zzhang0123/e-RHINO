"""FlaggingOperator — PLACEHOLDER RFI flagging.

Element: "We then apply various flagging, averaging, and calibration steps
that can correct for some of these contributions, but can also introduce
their own additional issues, e.g. if the models are slightly wrong/biased."

Real physics to come: MomentRFI-based flagging (as in the noise-wave GCR
draft, where flags inform the noise covariance). Flagging is a *data
processing* operator living in the same pipeline formalism — which is exactly
how "processing steps introduce their own issues" becomes modellable. The
placeholder thresholds the data and stores a boolean mask in ``state.aux``
(the traced side-channel), leaving the data itself untouched.
"""

from typing import ClassVar

import equinox as eqx

from erhino.core.operator import AbstractOperator
from erhino.core.state import State


class FlaggingOperator(AbstractOperator):
    """Store a threshold-based flag mask in ``state.aux["flags"]`` (placeholder).

    ``True`` marks a flagged (bad) sample. Data is not modified; downstream
    operators (averaging, likelihoods) decide how to use the mask.

    Attributes:
        threshold: flag samples with ``data > threshold`` (static
            configuration; thresholding is not differentiable anyway).
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("aux.flags",)

    threshold: float = eqx.field(static=True)

    def __call__(self, state: State) -> State:
        flags = state.data > self.threshold
        return state.replace(aux={**state.aux, "flags": flags})
