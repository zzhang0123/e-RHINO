"""Domain-agnostic core abstractions: State, Operator, Pipeline.

This subpackage must never import from ``dirt.radio`` or ``dirt.inference``,
so it can later be extracted as a standalone framework package.
"""

from dirt.core.combinators import SelectOperator, SumOperator
from dirt.core.coordinates import Coordinates
from dirt.core.environment import Environment
from dirt.core.errors import (
    DirtError,
    MissingKeyError,
    PipelineError,
    StateValidationError,
)
from dirt.core.frozen import FrozenMapping
from dirt.core.graph import (
    Assembly,
    AssemblyError,
    At,
    NodeSpec,
    SignalGraph,
    assemble,
    get_graph,
    register_graph,
)
from dirt.core.operator import AbstractOperator, LambdaOperator, SnapshotOperator
from dirt.core.pipeline import Pipeline
from dirt.core.state import State

__all__ = [
    "AbstractOperator",
    "Assembly",
    "AssemblyError",
    "At",
    "NodeSpec",
    "SignalGraph",
    "assemble",
    "get_graph",
    "register_graph",
    "Coordinates",
    "Environment",
    "DirtError",
    "FrozenMapping",
    "LambdaOperator",
    "MissingKeyError",
    "Pipeline",
    "SnapshotOperator",
    "PipelineError",
    "State",
    "StateValidationError",
    "SelectOperator",
    "SumOperator",
]
