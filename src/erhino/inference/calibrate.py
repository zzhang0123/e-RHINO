"""Calibration: infer pipeline parameters from observed data.

Deliberately OUTSIDE the forward model — a calibrator consumes the
``forward(params)`` function built by :func:`~erhino.inference.forward.build_forward_fn`
and never reaches into operators. :class:`GradientCalibrator` is a minimal
working demonstration (fixed-step gradient descent, pure JAX); real work will
use optax optimizers, NumPyro posteriors, or Gibbs schemes — all through the
same seam.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax

from erhino.core.errors import StateValidationError
from erhino.inference.likelihood import mean_squared_error


class GradientCalibrator(eqx.Module):
    """Fixed-step gradient descent on a forward model (minimal demonstrator).

    Attributes:
        learning_rate: step size (static configuration).
        n_steps: number of gradient steps (static configuration).
    """

    learning_rate: float = eqx.field(static=True, default=1e-2)
    n_steps: int = eqx.field(static=True, default=100)

    def __check_init__(self):
        if self.learning_rate <= 0:
            raise StateValidationError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if not isinstance(self.n_steps, int) or self.n_steps < 1:
            raise StateValidationError(f"n_steps must be a positive int, got {self.n_steps!r}.")

    def fit(
        self,
        forward: Callable[[Any], jax.Array],
        params0: Any,
        observed: jax.Array,
        loss_fn: Callable[[jax.Array, jax.Array], jax.Array] = mean_squared_error,
    ) -> tuple[Any, jax.Array]:
        """Minimize ``loss_fn(forward(params), observed)`` from ``params0``.

        Returns:
            ``(params_fit, losses)``: the fitted parameter pytree and the
            per-step loss history, shape ``(n_steps,)``.
        """

        def loss(params: Any) -> jax.Array:
            return loss_fn(forward(params), observed)

        def step(params: Any, _: None) -> tuple[Any, jax.Array]:
            value, grads = jax.value_and_grad(loss)(params)
            params = jax.tree.map(lambda p, g: p - self.learning_rate * g, params, grads)
            return params, value

        params_fit, losses = jax.lax.scan(step, params0, None, length=self.n_steps)
        return params_fit, losses


def to_numpyro_model(pipeline: Any, state_template: Any, **kwargs: Any):
    """Bridge a Pipeline to a NumPyro probabilistic model. NOT YET IMPLEMENTED.

    Planned: sample pipeline parameters from priors, run the forward model,
    condition on observed data — reusing the same partition/combine seam as
    :func:`~erhino.inference.forward.build_forward_fn`.

    Requires the optional dependency: ``pip install erhino[numpyro]``.
    """
    try:
        import numpyro  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "to_numpyro_model requires numpyro: pip install erhino[numpyro]"
        ) from exc
    raise NotImplementedError(
        "The NumPyro bridge is a roadmap item; see DESIGN.md. "
        "Meanwhile, build_forward_fn gives you f(params) to wrap manually."
    )
