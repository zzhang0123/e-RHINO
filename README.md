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
  assembled from beam-convolved sky + ground components), each with its own
  PRNG subkey.

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

### Forward modelling — composition is implicit in the signal path

The canonical single-dish signal-path graph (`erhino.radio.RADIO_GRAPH`)
knows how elements compose: provide a *set* of operators and `assemble`
lights up the connected sub-path they induce and compiles it to the
equivalent `Pipeline`/`SumOperator` nesting. Absent transforms are skipped
as identity, junctions materialize as sums, and partial models come free:

```python
from erhino.radio import (assemble, GlobalSignalOperator, ForegroundOperator,
                          SkyOperator, IonosphereOperator, BeamOperator)

sky = assemble(GlobalSignalOperator(...), ForegroundOperator(...))
#  ≡ SumOperator(signal, foregrounds)                     — just the sky

part = assemble(SkyOperator(...), IonosphereOperator(...), BeamOperator(...))
#  ≡ Pipeline(sky, ionosphere, beam)                      — beam-convolved sky only

print(part)               # lit nodes + skipped-as-identity nodes
print(part.to_mermaid())  # lit/dim signal-path rendering (mermaid)
open("signal_path.html", "w").write(part.to_html())   # standalone lit/dim page
```

Switched calibration is a first-class graph concept: provide
`CalLoadOperator` alongside the antenna chain and the `receiver_input`
*selector* node switches between them per time sample, driven by the cycle
in `coords.extra["receiver_input"]` (0 = antenna, 1 = load). Without a load
the selector passes through at zero cost.

The same physical effect may enter at different stages in different forms —
the graph reserves *equivalent-entry leaves* for each (e.g. ground spill as
a pre-beam field to convolve, or as a post-beam effective temperature via
`ground_pickup`/`t_sys_extra`). See DESIGN.md D11.

The full digital twin is one `assemble` call over the element set — see
`examples/radio_digital_twin.py` for the complete runnable version:

```python
import jax, jax.numpy as jnp, equinox as eqx
from erhino import State, Coordinates
from erhino.radio import assemble  # + the operator imports

state = State(
    coords=Coordinates(time=jnp.linspace(0, 60, 128),
                       freq=jnp.linspace(60e6, 85e6, 32)),
    key=jax.random.key(0),
    meta={"telescope": "generic-dish", "obs_id": "demo-001"},
)

twin = assemble(
    GlobalSignalOperator(...), ForegroundOperator(...), PointSourceOperator(...),
    IonosphereOperator(...), RFIOperator(...),          # pre-beam field
    BeamOperator(...),                                  # shared chromatic beam
    GroundPickupOperator(...),                          # post-beam effective temp
    SystemTemperatureOperator(...), NoiseWaveOperator(...),
    CWCalibrationOperator(...), ReceiverOperator(...), GainOperator(...),
    NoiseOperator(...), EMIOperator(...), ADCOperator(...),
    FlaggingOperator(...), BackendOperator(...),
)

observation = eqx.filter_jit(twin)(state)     # simulated waterfall + aux flags
```

Explicit `Pipeline`/`SumOperator` composition remains first-class —
`assemble` merely compiles to it (the equivalence is regression-tested
bitwise), with the chain order pinned by the element taxonomy and the RHINO
system equation `P_rec = g (T_ant + T_nw + T_cw) + T_n`.

### Inference / calibration (a separate layer)

The inference layer is complete: gradient and Adam calibrators, a real
NumPyro bridge (pytree priors + semantic site names + masked likelihood +
posterior predictive), Fisher/delta-method uncertainty forecasts, Monte
Carlo pushforward, and neural surrogate stages (`NeuralOperator`) — all
through one seam, `build_forward_fn`. See
`examples/bayesian_and_uncertainty.py` and `examples/neural_surrogate.py`.


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
| `radio.sky` | 21 cm global signal, diffuse foregrounds, point sources; **modular sky machinery**: `SkyModel × SkyProjector` — native differentiable limTOD engine (`NativeLimTODProjector`, via `pip install -e '<limTOD>[jax]'`), numpy-limTOD oracle bridge, projection matrices, m-mode |
| `radio.environment` | ionosphere, ground pickup, RFI |
| `radio.instrument` | beam, system temperature, noise-wave/reflection terms, bandpass, gain, CW calibration tone, thermal noise, self-EMI, ADC; calibration application |
| `radio.backend` | flagging (threshold + MomentRFI bridge), averaging |
| `radio.filters` | sidereal-repeat, sky-space (CG map-making), fringe-rate/delay filters |

Analysis on calibrated data uses the same Pipeline formalism (see
`examples/sky_projection_and_filters.py` for snapshot -> apply-calibration ->
sidereal/sky-space filtering): all steps are differentiable except flagging,
which is boolean by nature and bridges to numpy MomentRFI via
`pure_callback`.

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
