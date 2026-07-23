# e-RHINO Architecture

Design record for the differentiable scientific pipeline framework. The goal
is a **reusable framework**, not a one-off simulator: the radio telescope
digital twin is the first Pipeline, not the design center.

## Layering

```
erhino.core        State / Operator / Pipeline     (domain-agnostic)
erhino.radio       instrument operators            (placeholder physics)
erhino.inference   likelihood / calibration        (treats pipelines as data)
```

**Hard rule:** `erhino.core` never imports from `radio` or `inference`. If the
framework proves reusable beyond radio astronomy, `core` graduates to its own
package by moving one directory.

## Decisions

### D1 — State is an `eqx.Module` with a static/traced split

Traced leaves (differentiable, vmappable): `data`, `coords`, `env`, `aux`,
`key`. Static (treedef, jit-cache key): `meta` as a hashable `FrozenMapping`.

The user-facing rule is one sentence: *strings and labels in `meta`
(recompiles on change); numbers and arrays in `aux`/`env`/`coords` (traced).*
Two explicit channels instead of one ambiguous one — and `FrozenMapping`
rejects unhashable values at construction, so jit-cache corruption is
impossible by construction.

### D2 — Functional updates via `dataclasses.replace`

`state.replace(...)` re-runs converters and `__check_init__` on every update,
so validation is preserved along the whole pipeline. `eqx.tree_at` remains the
tool for surgical deep edits (e.g. one parameter inside a nested pipeline).
Validation is *structural only* (types, dtypes, ndim) — never traced values —
so it is jit-safe.

### D3 — PRNG protocol: `subkey, state = state.next_key()`

Randomness is data flowing through the state. Operators that draw randomness
must return the advanced state. Consequences: one seed reproduces an entire
run; keys are never reused; `vmap` over a batch of keys gives independent
realisations of the whole instrument.

### D4 — One abstract base, no hierarchy

`AbstractOperator` has exactly one abstract method (`__call__`). There are no
intermediate base classes; shared behaviour lives in helpers and composition.
Differentiable parameters need zero registration machinery — Equinox already
makes every array field a leaf that `eqx.partition` / `eqx.filter_grad` can
select.

`requires` / `provides` ClassVar tuples are a declarative contract
(documentation today, the hook for a future `pipeline.validate()`).

### D5 — Pipeline is an Operator (composite pattern)

Same `State -> State` signature, so pipelines nest freely. Execution is a
Python loop that unrolls under jit — correct for heterogeneous stages. A
`lax.scan` over homogeneous operator stacks is a complementary future pattern.
`run_with_intermediates()` is a separate diagnostics method, not a flag, so
the operator contract stays uniform.

### D6 — Inference treats the Pipeline as data

`build_forward_fn(pipeline, state_template, filter_spec)` partitions the
pipeline into (trainable params, static skeleton) and closes over the
template, exposing `f(params) -> prediction`. Gradient calibrators, NumPyro,
and future neural surrogates all connect through this one seam; calibration
never contaminates the instrument description.

## Element taxonomy → module map

`erhino.radio` mirrors the element taxonomy of a single-dish global-signal
experiment (source: `assets/elements.rtf`, local reference material — the
`assets/` folder is gitignored because it contains an unpublished draft).

```
Raw data elements                          Module
─────────────────────────────────────────  ─────────────────────────────────
Astrophysical
  21cm global signal (const LST, smooth ν) radio/sky/global_signal.py
  diffuse foregrounds (LST & ν variable)   radio/sky/foregrounds.py
  bright point sources (beam-diluted)      radio/sky/point_sources.py
Environmental
  ionosphere (distorts astro signal)       radio/environment/ionosphere.py
  ground pickup (sidelobes, T_ambient)     radio/environment/ground.py
  RFI (narrow+wideband, stochastic)        radio/environment/rfi.py
Instrumental
  beam (convolution, chromatic)            radio/instrument/beam.py
  DI gains (1/f + slower drifts)           radio/instrument/gain.py
  reflections + bandpass                   radio/instrument/receiver.py
  noise-wave T/Γ terms (GCR draft Eq. 1)   radio/instrument/noise_wave.py
  calibration signals (CW tone, loads)     radio/instrument/calibration.py
  self-generated EMI (comb-like)           radio/instrument/emi.py
  thermal noise (radiometer, T_sys)        radio/instrument/noise.py
  digitisation artifacts                   radio/instrument/adc.py
Processing
  flagging (MomentRFI)                     radio/backend/flagging.py
  averaging / integration                  radio/backend/averaging.py
```

Composition follows the physics: astrophysical components sum
(`SumOperator`), the ionosphere distorts that sum, terrestrial contributions
add on top, and the instrument chain is sequential (`Pipeline`). The chain
order mirrors RHINO paper Eq. 6, `P_rec = g (T_ant + T_nw + T_cw) + T_n`:
sky-side temperatures enter before the reflection/noise-wave terms, the CW
tone joins *before* bandpass and gain (it tracks gain drift only if it
passes through the gain), and thermal noise is added after the gain:

    astro = Pipeline(SumOperator(signal, foregrounds, point_sources), ionosphere)
    t_ant = SumOperator(astro, ground, rfi)
    twin  = Pipeline(t_ant, beam, tsys, noise_wave, cw_tone, bandpass, gain,
                     noise, emi, adc, flagging, averaging)

Identified pain points (beam uncertainties, foreground spectra, low-level
unflagged RFI, ground spill) are exactly where differentiable parameters +
marginalisation will matter most — each placeholder docstring records the
intended modelling strategy (beam-null degrees of freedom, moment expansion,
stochastic RFI variance, modulated topographic template).

## Roadmap (physics to port into the placeholder contracts)

The radio operators model a **generic single-dish radio telescope**. The
primary source for real physics is **limTOD** (the in-house single-dish TOD
simulator), which will itself be rewritten in JAX + Equinox; until then the
bodies stay placeholders. Instrument-specific parameters (e.g. RHINO's band,
horn beam, receiver noise-wave / reflection specs) enter later as concrete
operator *configurations*, never as framework assumptions.

Note: argosim was considered as a base but not used — it targets
interferometric arrays, while RHINO is single-dish; limTOD is the right
upstream.

| Operator | Real model (generic single dish) | Source |
|---|---|---|
| GlobalSignalOperator | physical 21 cm models (troughs, physical params) | — |
| ForegroundOperator | uncertain spectral-index maps, moment expansion | limTOD, MERS |
| PointSourceOperator | source catalogue through sidelobes | limTOD |
| IonosphereOperator | chromatic absorption/refraction, time-variable | — |
| GroundPickupOperator | topographic template, alt/az modulation, beam-coupled | EM sims |
| RFIOperator | stochastic process model (night-to-night variance) | MomentRFI |
| BeamOperator | primary-beam convolution (harmonic alm rotation, ZYZ) | limTOD (TIBEC for full-Stokes) |
| SystemTemperatureOperator | receiver temperature, atmosphere | instrument configs |
| ReceiverOperator | bandpass; reflection/impedance effects | instrument configs |
| NoiseWaveOperator | full Eq. 1 with F factor; T/Γ per frequency | noise-wave GCR draft |
| CWCalibrationOperator | tone shape/stability, switched reference loads | RHINO paper Sect. 4 |
| GainOperator | g(t) with 1/f flicker fluctuations | limTOD, hydra-tod |
| NoiseOperator | radiometer equation, 1/f covariance | limTOD, hydra-tod |
| EMIOperator | characterised switching-harmonic combs | lab measurements |
| ADCOperator | true quantization + straight-through estimator | — |
| FlaggingOperator | MomentRFI flags informing the noise covariance | MomentRFI |
| BackendOperator | integration, waterfall products | — |

Plus: NumPyro bridge (`to_numpyro_model`), GCR sampling of noise-wave
parameters (draft Eq. 28), optax-based calibrators, uncertainty propagation,
neural surrogate operators, multi-experiment support.

## Known deferred issues

- `data` is any pytree; the radio convention is a single `(n_time, n_freq)`
  array. Multi-stream data (TOD + CW tone) will adopt a dict-of-arrays
  convention — no State change needed.
- No enforced `data`↔`coords` consistency invariant (shape-changing operators
  must update coords manually; `BackendOperator` demonstrates the contract).
  Future `pipeline.validate()` can check `requires`/`provides`.
- `run_with_intermediates` keeps every stage's state in memory — diagnostics
  only.
- Typed PRNG keys (`jax.random.key`) are assumed throughout (jax ≥ 0.5).
