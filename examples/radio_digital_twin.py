"""Radio telescope digital twin — end-to-end demo.

Runs the full placeholder instrument pipeline (sky -> ... -> backend) under
jit, then demonstrates the inference seam: gradient calibration of the gain
against "observed" data, without touching the forward model's internals.

Run:  uv run python examples/radio_digital_twin.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from erhino import Coordinates, Environment, Pipeline, State
from erhino.inference import GradientCalibrator, build_forward_fn
from erhino.radio import (
    ADCOperator,
    BackendOperator,
    BeamOperator,
    GainOperator,
    NoiseOperator,
    ReceiverOperator,
    SkyOperator,
    SystemTemperatureOperator,
)

N_TIME, N_FREQ = 128, 32

# ----------------------------------------------------------------- forward --
state = State(
    coords=Coordinates(
        time=jnp.linspace(0.0, 60.0, N_TIME),           # seconds
        freq=jnp.linspace(60e6, 85e6, N_FREQ),          # Hz (demo values)
    ),
    env=Environment(temperature=jnp.array(280.0)),      # rides along for diagnostics
    key=jax.random.key(0),
    meta={"telescope": "RHINO", "obs_id": "demo-001"},
)

pipeline = Pipeline(
    SkyOperator(amplitude=jnp.array(1.0e3)),
    BeamOperator(solid_angle=jnp.array(0.8)),
    SystemTemperatureOperator(t_sys=jnp.array(150.0)),
    ReceiverOperator(bandpass=jnp.ones(N_FREQ)),
    GainOperator(gain=jnp.array(1.1)),                  # the "true" gain to recover
    NoiseOperator(sigma=jnp.array(0.5)),
    ADCOperator(scale=jnp.array(1.0), n_bits=14),
    BackendOperator(n_chunk=4),
    names=("sky", "beam", "tsys", "receiver", "gain", "noise", "adc", "backend"),
)

observation = eqx.filter_jit(pipeline)(state)
print(f"simulated waterfall: {observation.data.shape}  "
      f"(meta preserved: telescope={observation.meta['telescope']!r})")

# --------------------------------------------------------------- inference --
# Mis-calibrated model: same instrument, wrong gain. Train ONLY the gain.
model = pipeline.replace_stage("gain", GainOperator(gain=jnp.array(1.0)))
spec = jax.tree.map(lambda _: False, model)
spec = eqx.tree_at(lambda p: p["gain"].gain, spec, replace=True)
forward, params0 = build_forward_fn(model, state, filter_spec=spec)

calibrator = GradientCalibrator(learning_rate=2e-7, n_steps=200)
params_fit, losses = calibrator.fit(forward, params0, observation.data)

fitted_gain = jax.tree.leaves(params_fit)[0]
print(f"calibration: gain 1.000 -> {float(fitted_gain):.4f}  (truth 1.100)")
print(f"loss: {float(losses[0]):.3e} -> {float(losses[-1]):.3e}")
