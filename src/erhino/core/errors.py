"""Exception hierarchy for erhino.

All framework errors derive from :class:`ErhinoError` so users can catch the
whole family with one ``except`` clause. Subclasses additionally derive from
the closest builtin (``ValueError`` / ``RuntimeError``) so generic handlers
keep working.
"""


class ErhinoError(Exception):
    """Base class for all erhino errors."""


class StateValidationError(ErhinoError, ValueError):
    """A State (or one of its containers) was constructed with invalid contents.

    Raised only for *structural* problems (wrong ndim, wrong dtype, bad key
    types) — never for traced array *values*, so validation stays jit-safe.
    """


class MissingKeyError(ErhinoError, RuntimeError):
    """An operator needed randomness but ``State.key`` is ``None``.

    Fix: construct the state with ``key=jax.random.key(seed)``.
    """


class PipelineError(ErhinoError, ValueError):
    """A Pipeline was misconfigured (empty, bad stage type, name collision...)."""
