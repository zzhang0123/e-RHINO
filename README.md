# e-RHINO

**erhino** is a general-purpose, extensible, *differentiable* scientific
pipeline framework built on [JAX](https://github.com/jax-ml/jax) and
[Equinox](https://github.com/patrick-kidger/equinox).

Its first application is a **digital twin of a single-dish radio telescope**
(the eventual target instrument is RHINO, a horn antenna for the 21 cm global
signal — for now the radio operators model a *generic* single dish), and the
core is domain-agnostic by construction.

> **Core principle: everything is an Operator acting on a State.**

```
State_in ──▶ Operator ──▶ State_out
```

- **State** — an immutable JAX pytree carrying the complete scientific
  context: signal data, coordinates, environment telemetry, metadata, PRNG
  keys. Fields that don't enter the forward model still ride along for
  diagnostics, correlation studies, and reproducibility.
- **Operator** — a pure `State -> State` transformation (`equinox.Module`);
  its array fields are differentiable parameters for free.
- **Pipeline** — an ordered, named *sequential* composition of operators;
  itself an operator, so pipelines nest.
- **SumOperator** — a *parallel additive* composition: branches produce
  independent contributions on the same grid (e.g. an antenna temperature
  assembled from sky + ground + RFI components), each with its own PRNG
  subkey.

Because everything is a pytree, an entire instrument model is one function you
can `jit`, `grad`, and `vmap`.

## Install

```bash
git clone https://github.com/zzhang0123/e-RHINO
cd e-RHINO
uv sync                      # or: pip install -e ".[numpyro]"
```

Requires Python ≥ 3.11, `jax ≥ 0.5`, `equinox ≥ 0.13`.

## Usage

### Forward modelling

The forward model is assembled following the element taxonomy: astrophysical
components *sum*, the ionosphere distorts that sum, terrestrial contributions
add on top, and the instrument chain is *sequential* (ordering per the RHINO
system equation `P_rec = g (T_ant + T_nw + T_cw) + T_n` — the CW tone joins
before bandpass/gain so it tracks gain drift):

```python
import jax, jax.numpy as jnp, equinox as eqx
from erhino import State, Coordinates, Pipeline, SumOperator
from erhino.radio import (
    GlobalSignalOperator, ForegroundOperator, PointSourceOperator,   # sky
    IonosphereOperator, GroundPickupOperator, RFIOperator,           # environment
    BeamOperator, SystemTemperatureOperator, NoiseWaveOperator,      # instrument
    CWCalibrationOperator, ReceiverOperator, GainOperator,
    NoiseOperator, EMIOperator, ADCOperator,
    FlaggingOperator, BackendOperator,                               # backend
)

state = State(
    coords=Coordinates(time=jnp.linspace(0, 60, 128),
                       freq=jnp.linspace(60e6, 85e6, 32)),
    key=jax.random.key(0),
    meta={"telescope": "generic-dish", "obs_id": "demo-001"},
)

astro = Pipeline(
    SumOperator(
        GlobalSignalOperator(depth=jnp.array(0.2), centre=jnp.array(72e6),
                             width=jnp.array(5e6)),
        ForegroundOperator(amplitude=jnp.array(1e3),
                           spectral_index=jnp.array(2.5), ref_freq=70e6),
        PointSourceOperator(level=jnp.array(2.0)),
        names=("signal", "foregrounds", "point_sources"),
    ),
    IonosphereOperator(delta=jnp.array(0.01), ref_freq=70e6),
    names=("sky", "ionosphere"),
)

t_ant = SumOperator(
    astro,
    GroundPickupOperator(coupling=jnp.array(0.01), t_ground=jnp.array(300.0)),
    RFIOperator(amplitude=jnp.array(2e3), occupancy=0.01),
    names=("astro", "ground", "rfi"),
)

twin = Pipeline(
    t_ant,
    BeamOperator(solid_angle=jnp.array(0.8)),
    SystemTemperatureOperator(t_sys=jnp.array(150.0)),   # sky-side (atmosphere/spill)
    NoiseWaveOperator(t_unc=jnp.array(1.0), t_cos=jnp.array(0.5),
                      t_sin=jnp.array(0.2), t_zero=jnp.array(2.0),
                      gamma_re=jnp.array(0.05), gamma_im=jnp.array(0.02)),
    CWCalibrationOperator(amplitude=jnp.array(500.0), tone_freq=80e6),
    ReceiverOperator(bandpass=jnp.ones(32)),
    GainOperator(gain=jnp.array(1.1)),
    NoiseOperator(sigma=jnp.array(0.5)),
    EMIOperator(amplitude=jnp.array(0.3), period=8),
    ADCOperator(scale=jnp.array(1.0), n_bits=14),
    FlaggingOperator(threshold=2.5e3),
    BackendOperator(n_chunk=4),
    names=("t_ant", "beam", "tsys", "noise_wave", "cw_tone", "receiver",
           "gain", "noise", "emi", "adc", "flagging", "backend"),
)

observation = eqx.filter_jit(twin)(state)     # simulated waterfall + aux flags
```

### Inference / calibration (a separate layer)

Calibration never lives inside the forward model. The seam is
`build_forward_fn`, which turns a pipeline into `f(params) -> prediction`
via the Equinox partition/combine idiom:

```python
from erhino.inference import build_forward_fn, GradientCalibrator

# train ONLY the gain; freeze everything else
spec = jax.tree.map(lambda _: False, twin)
spec = eqx.tree_at(lambda p: p["gain"].gain, spec, replace=True)
forward, params0 = build_forward_fn(twin, state, filter_spec=spec)

params_fit, losses = GradientCalibrator(learning_rate=2e-7, n_steps=200).fit(
    forward, params0, observation.data
)
```

Run the full demo: `uv run python examples/radio_digital_twin.py`.

### Key conventions

- **Immutability everywhere**: `state.replace(...)`, `state.with_data(...)`,
  `pipeline.replace_stage(name, op)` return new objects; nothing mutates.
- **Metadata rule**: strings/labels go in `state.meta` (static — changing them
  recompiles); numbers/arrays go in `state.aux` / `env` / `coords` (traced,
  differentiable).
- **PRNG protocol**: operators consume randomness with
  `subkey, state = state.next_key()` and return the advanced state — one seed
  reproduces an entire run.
- **Angles**: degrees in public APIs, radians internally (RHINO family
  convention).

## Status

The architecture is complete and fully tested (jit+grad+vmap end-to-end).
`erhino.radio` is organized by the element taxonomy of a single-dish
experiment (see `DESIGN.md`):

| Subpackage | Elements |
|---|---|
| `radio.sky` | 21 cm global signal, diffuse foregrounds, point sources; **modular sky machinery**: `SkyModel × SkyProjector` (limTOD bridge, projection matrices, m-mode) |
| `radio.environment` | ionosphere, ground pickup, RFI |
| `radio.instrument` | beam, system temperature, noise-wave/reflection terms, bandpass, gain, CW calibration tone, thermal noise, self-EMI, ADC; calibration application |
| `radio.backend` | flagging (threshold + MomentRFI bridge), averaging |
| `radio.filters` | sidereal-repeat, sky-space (CG map-making), fringe-rate/delay filters |

Analysis on calibrated data uses the same Pipeline formalism (see
`examples/sky_projection_and_filters.py`): snapshot raw data
(`SnapshotOperator`), apply calibration, flag with MomentRFI, filter — all
differentiable except flagging (which is boolean by nature and bridges to
numpy MomentRFI via `pure_callback`).

The physics is **deliberately placeholder**: every operator implements
trivial-but-runnable math that establishes the contract (shapes, PRNG
consumption, differentiability, linearity in calibration parameters), to be
replaced by ports from limTOD — the single-dish TOD simulator that will
itself be rewritten in JAX + Equinox. Instrument-specific parameters
(RHINO's band, beam, receiver) arrive later as concrete configurations, not
framework assumptions.

See [DESIGN.md](DESIGN.md) for the architecture decisions and roadmap.

No CI is configured yet; run `uv run pytest` and `uv run ruff check` locally
before pushing.

## License

MIT
