"""erhino: a differentiable scientific pipeline framework built on JAX + Equinox.

Core principle: **everything is an Operator acting on a State.**

- ``erhino.core`` — domain-agnostic State / Operator / Pipeline abstractions.
- ``erhino.radio`` — generic single-dish radio telescope operators
  (placeholder physics; RHINO is the eventual target instrument).
- ``erhino.inference`` — likelihood / calibration layer, separate from forward models.
"""

from erhino.core import (
    AbstractOperator,
    Coordinates,
    Environment,
    ErhinoError,
    FrozenMapping,
    LambdaOperator,
    MissingKeyError,
    Pipeline,
    PipelineError,
    SnapshotOperator,
    State,
    StateValidationError,
    SumOperator,
)

__version__ = "0.1.0"

__all__ = [
    "AbstractOperator",
    "Coordinates",
    "Environment",
    "ErhinoError",
    "FrozenMapping",
    "LambdaOperator",
    "MissingKeyError",
    "Pipeline",
    "SnapshotOperator",
    "PipelineError",
    "State",
    "StateValidationError",
    "SumOperator",
    "__version__",
]
