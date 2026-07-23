"""Combinators: parallel composition of operators.

:class:`~erhino.core.pipeline.Pipeline` composes operators *sequentially*;
:class:`SumOperator` composes them *in parallel*, summing their data
contributions. Together they express the typical structure of a physical
forward model, e.g. an antenna temperature assembled from independent
components::

    astro = Pipeline(
        SumOperator(global_signal, foregrounds, point_sources),
        ionosphere,                       # distorts the astrophysical sum
    )
    t_ant = SumOperator(astro, ground_pickup)

Both combinators are themselves operators, so they nest freely.
"""

from collections.abc import Iterator, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from erhino.core.errors import PipelineError
from erhino.core.operator import AbstractOperator
from erhino.core.pipeline import resolve_names, validate_operators
from erhino.core.state import State


def _tree_add(total: Any, contribution: Any, branch_name: str) -> Any:
    """Leafwise sum of two data pytrees with loud structural/shape validation.

    Python ``+`` would concatenate tuples and fail on dicts; silent NumPy
    broadcasting would smear shape-mismatched contributions. Both are trace-time
    (static-shape) checks, so this stays jit-safe.
    """

    total_structure = jax.tree.structure(total)
    if total_structure != jax.tree.structure(contribution):
        raise PipelineError(
            f"SumOperator branch {branch_name!r} produced a data pytree whose structure "
            f"differs from the previous branches: {jax.tree.structure(contribution)} "
            f"vs accumulated {total_structure}."
        )

    def add(a: jax.Array, b: jax.Array) -> jax.Array:
        if jnp.shape(a) != jnp.shape(b):
            raise PipelineError(
                f"SumOperator branch {branch_name!r} produced a contribution of shape "
                f"{jnp.shape(b)}, which does not match the accumulated shape {jnp.shape(a)}. "
                "Branches must produce identical shapes on the shared coordinate grid "
                "(shape-changing operators like BackendOperator cannot be Sum branches)."
            )
        return a + b

    return jax.tree.map(add, total, contribution)


class SumOperator(AbstractOperator):
    """Run source-type branches on the same input state and sum their data.

    Semantics:
        * Every branch receives the *input* context (coords, env, meta) with
          ``data`` stripped to ``None`` — SumOperator is a *source* combinator
          whose branches each produce their own contribution on the shared
          coordinate grid. A branch that tries to read input data fails
          loudly instead of silently depending on caller state.
        * Branches must not change coords/env/meta; the output state carries
          the input context with ``data = sum(branch outputs)``.
        * PRNG: each branch gets its own subkey split off the main chain, so
          stochastic branches draw independent randomness and a single seed
          reproduces the whole sum. The output state carries the advanced key.

    Example::

        sky = SumOperator(
            GlobalSignalOperator(...), ForegroundOperator(...),
            names=("signal", "foregrounds"),
        )
    """

    branches: tuple[AbstractOperator, ...]
    names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *branches: AbstractOperator,
        names: Sequence[str] | None = None,
    ):
        self.branches = validate_operators(branches, "SumOperator")
        self.names = resolve_names(branches, names)

    def __call__(self, state: State) -> State:
        total = None
        for name, branch in zip(self.names, self.branches, strict=True):
            if state.key is not None:
                branch_key, state = state.next_key()
                branch_in = state.replace(key=branch_key, data=None)
            else:
                branch_in = state.replace(data=None)
            contribution = branch(branch_in).data
            if contribution is None:
                raise PipelineError(
                    f"SumOperator branch {name!r} produced no data (state.data is None); "
                    "every branch must be a source-type operator."
                )
            total = contribution if total is None else _tree_add(total, contribution, name)
        return state.with_data(total)

    def __getitem__(self, index: int | str) -> AbstractOperator:
        if isinstance(index, str):
            try:
                index = self.names.index(index)
            except ValueError:
                raise KeyError(
                    f"No branch named {index!r}; available: {self.names}"
                ) from None
        return self.branches[index]

    def __len__(self) -> int:
        return len(self.branches)

    def __iter__(self) -> Iterator[AbstractOperator]:
        return iter(self.branches)

    def replace_branch(self, index: int | str, operator: AbstractOperator) -> "SumOperator":
        """Return a new SumOperator with one branch swapped; names preserved.

        Parity with :meth:`~erhino.core.pipeline.Pipeline.replace_stage`.
        """
        if not isinstance(operator, AbstractOperator):
            raise PipelineError(
                f"Replacement must be an AbstractOperator, got {type(operator).__name__}."
            )
        if isinstance(index, str):
            try:
                index = self.names.index(index)
            except ValueError:
                raise KeyError(
                    f"No branch named {index!r}; available: {self.names}"
                ) from None
        new_branches = list(self.branches)
        new_branches[index] = operator
        return SumOperator(*new_branches, names=self.names)
