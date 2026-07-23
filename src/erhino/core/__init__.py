"""Domain-agnostic core abstractions: State, Operator, Pipeline.

This subpackage must never import from ``erhino.radio`` or ``erhino.inference``,
so it can later be extracted as a standalone framework package.
"""

from erhino.core.combinators import SumOperator
from erhino.core.coordinates import Coordinates
from erhino.core.environment import Environment
from erhino.core.errors import (
    ErhinoError,
    MissingKeyError,
    PipelineError,
    StateValidationError,
)
from erhino.core.frozen import FrozenMapping
from erhino.core.operator import AbstractOperator, LambdaOperator
from erhino.core.pipeline import Pipeline
from erhino.core.state import State

__all__ = [
    "AbstractOperator",
    "Coordinates",
    "Environment",
    "ErhinoError",
    "FrozenMapping",
    "LambdaOperator",
    "MissingKeyError",
    "Pipeline",
    "PipelineError",
    "State",
    "StateValidationError",
    "SumOperator",
]
