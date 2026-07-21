"""NoiseOperator — PLACEHOLDER radiometric noise.

Real physics to come (radiometer equation, hydra-tod noise model): the noise
is *multiplicative* — sigma proportional to the total power itself,
``sigma = T_total / sqrt(dt * dnu)`` — plus correlated 1/f fluctuations
sharing the gain's flicker spectrum. This placeholder adds white Gaussian
noise with a fixed sigma, demonstrating the PRNG-consumption contract.
"""

from typing import ClassVar

import jax

from erhino.core.operator import AbstractOperator
from erhino.core.state import State


class NoiseOperator(AbstractOperator):
    """Add white Gaussian noise to ``state.data`` (placeholder).

    Consumes randomness through the State PRNG protocol: the returned state
    carries an *advanced* key, so repeated application gives fresh draws while
    a single seed reproduces the whole pipeline.

    Attributes:
        sigma: noise standard deviation [K] — differentiable scalar.
    """

    requires: ClassVar[tuple[str, ...]] = ("data", "key")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    sigma: jax.Array

    def __call__(self, state: State) -> State:
        subkey, state = state.next_key()
        noise = self.sigma * jax.random.normal(subkey, jax.numpy.shape(state.data))
        return state.with_data(state.data + noise)
