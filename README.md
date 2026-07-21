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
- **Pipeline** — an ordered, named composition of operators; itself an
  operator, so pipelines nest.

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

```python
import jax, jax.numpy as jnp, equinox as eqx
from erhino import State, Coordinates, Pipeline
from erhino.radio import (SkyOperator, BeamOperator, SystemTemperatureOperator,
                          ReceiverOperator, GainOperator, NoiseOperator,
                          ADCOperator, BackendOperator)

state = State(
    coords=Coordinates(time=jnp.linspace(0, 60, 128),
                       freq=jnp.linspace(60e6, 85e6, 32)),
    key=jax.random.key(0),
    meta={"telescope": "RHINO", "obs_id": "demo-001"},
)

pipeline = Pipeline(
    SkyOperator(amplitude=jnp.array(1e3)),
    BeamOperator(solid_angle=jnp.array(0.8)),
    SystemTemperatureOperator(t_sys=jnp.array(150.0)),
    ReceiverOperator(bandpass=jnp.ones(32)),
    GainOperator(gain=jnp.array(1.0)),
    NoiseOperator(sigma=jnp.array(0.1)),
    ADCOperator(scale=jnp.array(1.0), n_bits=14),
    BackendOperator(n_chunk=4),
    names=("sky", "beam", "tsys", "receiver", "gain", "noise", "adc", "backend"),
)

observation = eqx.filter_jit(pipeline)(state)     # simulated waterfall
```

### Inference / calibration (a separate layer)

Calibration never lives inside the forward model. The seam is
`build_forward_fn`, which turns a pipeline into `f(params) -> prediction`
via the Equinox partition/combine idiom:

```python
from erhino.inference import build_forward_fn, GradientCalibrator

# train ONLY the gain; freeze everything else
spec = jax.tree.map(lambda _: False, pipeline)
spec = eqx.tree_at(lambda p: p["gain"].gain, spec, replace=True)
forward, params0 = build_forward_fn(pipeline, state, filter_spec=spec)

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

The architecture is complete and fully tested (129 tests; jit+grad+vmap
end-to-end). The physics is **deliberately placeholder**: every
`erhino.radio` operator documents the generic single-dish model it will hold
(beam-convolved sky, 1/f gain, radiometer-equation noise), to be ported from
[limTOD](https://github.com/zzhang0123) — the single-dish TOD simulator that
will itself be rewritten in JAX + Equinox. Instrument-specific parameters
(RHINO's band, beam, receiver) arrive later as concrete configurations, not
framework assumptions.

See [DESIGN.md](DESIGN.md) for the architecture decisions and roadmap.

No CI is configured yet; run `uv run pytest` and `uv run ruff check` locally
before pushing.

## License

MIT
