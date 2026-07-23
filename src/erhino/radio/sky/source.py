"""SkySourceOperator: the modular sky slot of the forward model.

Composes the two halves of the sky abstraction::

    SkySourceOperator(
        sky_model=PowerLawSkyModel(...),   # what the sky is  (differentiable params)
        projector=MatrixProjector(A),      # how it is seen   (swappable engine)
    )

Either half swaps independently — e.g. replace the projector with
``eqx.tree_at(lambda p: p["t_ant"]["sky"].projector, twin, MModeProjector(B))``
without touching the sky parameters, or infer sky parameters through any
engine via ``erhino.inference.build_forward_fn``.
"""

from typing import ClassVar

from erhino.core.errors import StateValidationError
from erhino.core.operator import AbstractOperator
from erhino.core.state import State
from erhino.radio.sky.model import AbstractSkyModel
from erhino.radio.sky.projection import AbstractSkyProjector


class SkySourceOperator(AbstractOperator):
    """Produce the sky's antenna-temperature contribution: projector(sky_model).

    Attributes:
        sky_model: what the sky is — :class:`~erhino.radio.sky.model.AbstractSkyModel`.
        projector: how it is seen — :class:`~erhino.radio.sky.projection.AbstractSkyProjector`.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    sky_model: AbstractSkyModel
    projector: AbstractSkyProjector

    def __check_init__(self):
        if not isinstance(self.sky_model, AbstractSkyModel):
            raise StateValidationError(
                f"sky_model must be an AbstractSkyModel, got {type(self.sky_model).__name__}."
            )
        if not isinstance(self.projector, AbstractSkyProjector):
            raise StateValidationError(
                f"projector must be an AbstractSkyProjector, got {type(self.projector).__name__}."
            )

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.time is None or state.coords.freq is None:
            raise StateValidationError(
                "SkySourceOperator requires state.coords with time and freq axes."
            )
        sky = self.sky_model(state.coords.freq)
        return state.with_data(self.projector.forward(sky, state.coords))
