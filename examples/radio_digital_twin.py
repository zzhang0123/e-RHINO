"""Radio telescope digital twin — end-to-end demo.

Builds the forward model following the element taxonomy (see DESIGN.md):

    astrophysical sum -> ionosphere            (astro branch)
    + ground pickup + RFI                      (antenna temperature)
    -> instrument chain -> backend             (raw data products)

then demonstrates the inference seam: gradient calibration of the gain
against "observed" data, without touching the forward model's internals.

Run:  uv run python examples/radio_digital_twin.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from erhino import Coordinates, Environment, Pipeline, State, SumOperator
from erhino.inference import GradientCalibrator, build_forward_fn
from erhino.radio import (
    ADCOperator,
    BackendOperator,
    BeamOperator,
    CWCalibrationOperator,
    EMIOperator,
    FlaggingOperator,
    ForegroundOperator,
    GainOperator,
    GlobalSignalOperator,
    GroundPickupOperator,
    IonosphereOperator,
    NoiseOperator,
    NoiseWaveOperator,
    PointSourceOperator,
    ReceiverOperator,
    RFIOperator,
    SystemTemperatureOperator,
)

N_TIME, N_FREQ = 128, 32

# ------------------------------------------------------------ observation --
state = State(
    coords=Coordinates(
        time=jnp.linspace(0.0, 60.0, N_TIME),           # seconds
        freq=jnp.linspace(60e6, 85e6, N_FREQ),          # Hz (demo values)
    ),
    env=Environment(temperature=jnp.array(280.0)),      # couples into ground pickup
    key=jax.random.key(0),
    meta={"telescope": "generic-dish", "obs_id": "demo-001"},
)

# ------------------------------------------- antenna temperature assembly --
astro = Pipeline(
    SumOperator(
        GlobalSignalOperator(depth=jnp.array(0.2), centre=jnp.array(72e6),
                             width=jnp.array(5e6)),
        ForegroundOperator(amplitude=jnp.array(1.0e3), spectral_index=jnp.array(2.5),
                           ref_freq=70e6),
        PointSourceOperator(level=jnp.array(2.0)),
        names=("signal", "foregrounds", "point_sources"),
    ),
    IonosphereOperator(delta=jnp.array(0.01), ref_freq=70e6),
    names=("sky", "ionosphere"),
)

t_ant = SumOperator(
    astro,
    GroundPickupOperator(coupling=jnp.array(0.01), t_ground=jnp.array(300.0)),
    RFIOperator(amplitude=jnp.array(2.0e3), occupancy=0.01),
    names=("astro", "ground", "rfi"),
)

# ------------------------------------------------------- instrument chain --
# Chain order mirrors RHINO paper Eq. 6, P_rec = g (T_ant + T_nw + T_cw) + T_n:
# sky-side tsys before the reflection terms; CW tone BEFORE bandpass/gain so
# it tracks gain drift; thermal noise after the gain.
twin = Pipeline(
    t_ant,
    BeamOperator(solid_angle=jnp.array(0.8)),
    SystemTemperatureOperator(t_sys=jnp.array(150.0)),  # sky-side (atmosphere/spill)
    NoiseWaveOperator(t_unc=jnp.array(1.0), t_cos=jnp.array(0.5),
                      t_sin=jnp.array(0.2), t_zero=jnp.array(2.0),
                      gamma_re=jnp.array(0.05), gamma_im=jnp.array(0.02)),
    CWCalibrationOperator(amplitude=jnp.array(500.0), tone_freq=80e6),
    ReceiverOperator(bandpass=jnp.ones(N_FREQ)),
    GainOperator(gain=jnp.array(1.1)),                  # the "true" gain to recover
    NoiseOperator(sigma=jnp.array(0.5)),
    EMIOperator(amplitude=jnp.array(0.3), period=8),
    ADCOperator(scale=jnp.array(1.0), n_bits=14),
    FlaggingOperator(threshold=2.5e3),
    BackendOperator(n_chunk=4),
    names=("t_ant", "beam", "tsys", "noise_wave", "cw_tone", "receiver",
           "gain", "noise", "emi", "adc", "flagging", "backend"),
)

observation = eqx.filter_jit(twin)(state)
print(f"simulated waterfall: {observation.data.shape}  "
      f"(flags: {int(observation.aux['flags'].sum())} samples, "
      f"meta preserved: {observation.meta['telescope']!r})")

# --------------------------------------------------------------- inference --
# Mis-calibrated model: same instrument, wrong gain. Train ONLY the gain.
model = twin.replace_stage("gain", GainOperator(gain=jnp.array(1.0)))
spec = jax.tree.map(lambda _: False, model)
spec = eqx.tree_at(lambda p: p["gain"].gain, spec, replace=True)
forward, params0 = build_forward_fn(model, state, filter_spec=spec)

calibrator = GradientCalibrator(learning_rate=2e-7, n_steps=200)
params_fit, losses = calibrator.fit(forward, params0, observation.data)

fitted_gain = jax.tree.leaves(params_fit)[0]
print(f"calibration: gain 1.000 -> {float(fitted_gain):.4f}  (truth 1.100)")
print(f"loss: {float(losses[0]):.3e} -> {float(losses[-1]):.3e}")
