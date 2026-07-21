"""Likelihoods: score a forward-model prediction against observed data.

A likelihood is any callable ``(prediction, observed) -> scalar log-prob``.
The Protocol below documents the contract; :class:`GaussianLikelihood` is the
minimal concrete instance. Real instrument likelihoods (radiometer-equation
noise, 1/f covariance, Toeplitz solvers ported from hydra-tod/comat) will
implement the same contract.
"""

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp


@runtime_checkable
class Likelihood(Protocol):
    """Contract: ``logp = likelihood(prediction, observed)`` (scalar)."""

    def __call__(self, prediction: jax.Array, observed: jax.Array) -> jax.Array: ...


class GaussianLikelihood(eqx.Module):
    """Independent Gaussian likelihood with fixed noise level.

    Attributes:
        noise_std: noise standard deviation — scalar or broadcastable to the
            data shape; a differentiable leaf (so it can itself be inferred).
    """

    noise_std: jax.Array

    def __call__(self, prediction: jax.Array, observed: jax.Array) -> jax.Array:
        resid = (observed - prediction) / self.noise_std
        return -0.5 * jnp.sum(resid**2 + jnp.log(2.0 * jnp.pi * self.noise_std**2))


def mean_squared_error(prediction: jax.Array, observed: jax.Array) -> jax.Array:
    """Plain MSE — the default loss for quick gradient calibration."""
    return jnp.mean((prediction - observed) ** 2)
