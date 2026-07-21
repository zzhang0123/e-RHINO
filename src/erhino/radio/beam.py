"""BeamOperator — PLACEHOLDER beam response.

Real physics to come (ported from TIBEC / limTOD): full-Stokes beam matrices
derived from CST full-wave EM simulations of the RHINO horn, rotated into
equatorial coordinates (ZYZ Euler convention) and convolved with the sky in
harmonic space. The horn design makes the beam nearly independent of soil
moisture (<1% correction variation) — a claim the real operator should let us
verify by parametrizing residual beam/ground coupling.
"""

from typing import ClassVar

import jax

from erhino.core.operator import AbstractOperator
from erhino.core.state import State


class BeamOperator(AbstractOperator):
    """Scale ``state.data`` by an effective beam solid-angle factor (placeholder).

    Attributes:
        solid_angle: dimensionless beam coupling factor — differentiable scalar.
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    solid_angle: jax.Array

    def __call__(self, state: State) -> State:
        return state.with_data(state.data * self.solid_angle)
