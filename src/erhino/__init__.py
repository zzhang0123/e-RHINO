"""erhino: a differentiable scientific pipeline framework built on JAX + Equinox.

Core principle: **everything is an Operator acting on a State.**

- ``erhino.core`` — domain-agnostic State / Operator / Pipeline abstractions.
- ``erhino.radio`` — radio-telescope digital-twin operators (RHINO first).
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
    State,
    StateValidationError,
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
    "PipelineError",
    "State",
    "StateValidationError",
    "__version__",
]
