# e-RHINO Architecture

Design record for the differentiable scientific pipeline framework. The goal
is a **reusable framework**, not a one-off simulator: the radio telescope
digital twin is the first Pipeline, not the design center.

## Layering

```
erhino.core        State / Operator / Pipeline / SumOperator   (domain-agnostic)
erhino.radio       single-dish operators, organized by element  (placeholder physics)
                   taxonomy: sky / environment / instrument / backend
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

### D6 — SumOperator: parallel additive composition, source-only semantics

Physical models are sums of independent components; `SumOperator` makes that
a first-class combinator alongside sequential `Pipeline`. Semantics chosen
deliberately narrow: branches are *source-type* operators producing
contributions on the shared coordinate grid; input `data` is stripped to
`None` before entering each branch (enforced, not merely documented — a
branch that tries to read caller data fails loudly), and branch writes to
`coords`/`env`/`meta`/`aux` are discarded (parallel writes have no
well-defined merge). Each branch receives its own PRNG subkey split
off the main chain, so stochastic branches draw independent randomness and
one seed reproduces the whole sum. Accumulation is leafwise
(`jax.tree.map`), with loud trace-time errors on shape or pytree-structure
mismatches and on branches producing no data — silent tuple concatenation
and NumPy broadcasting were real failure modes caught in review.

### D7 — Inference treats the Pipeline as data

`build_forward_fn(pipeline, state_template, filter_spec)` partitions the
pipeline into (trainable params, static skeleton) and closes over the
template, exposing `f(params) -> prediction`. Gradient calibrators, NumPyro,
and future neural surrogates all connect through this one seam; calibration
never contaminates the instrument description.

### D8 — Modular sky: SkyModel × SkyProjector

The sky term factorizes into *what the sky is* (`AbstractSkyModel`:
params → `(n_freq, n_pix)` maps; differentiable amplitudes / spectral
indices / moment coefficients) and *how it is seen* (`AbstractSkyProjector`:
maps → `(n_time, n_freq)`; `forward` + `adjoint` for linear engines),
composed by `SkySourceOperator`. Either half swaps independently, so the
same sky can be observed through limTOD beam convolution, a precomputed
projection matrix, or m-mode transfer matrices — and the same engine serves
different skies. Three engines form a maturity ladder, now complete:
`LimTODProjector` (pure_callback oracle — jit-safe, not differentiable) →
`MatrixProjector` (offline `generate_sky2sys_projection` matrix — fully
differentiable for fixed pointing/beam, RHINO's drift-scan case) →
`NativeLimTODProjector` (**delivered**: pure JAX via the `limtod_jax`
package, general pointing, differentiable w.r.t. both sky and beam alms,
exact adjoint; contract and status in `docs/limtod-port-contract.md`).
Linear projectors expose `adjoint` (verified by dot-product tests) because
map-making reuses it (D9).

### D9 — Filters are linear projections; raw data survives via snapshots

Sidereal-repeat extraction, sky-space (Wiener/map-making) filtering, and
fringe-rate/delay filtering are all projections `P d`; `AbstractLinearFilter`
fixes the shared semantics (`mode="extract"` → `P d`, `"remove"` → `d − P d`)
and concrete filters supply `P`. `SkySpaceFilter` solves the regularised
normal equations with matrix-free CG (`lax.custom_linear_solve` under the
hood), reusing the forward model's projector adjoint — so filters are
differentiable and their transfer functions can be marginalised in inference.
Filters run on calibrated data (`ApplyCalibrationOperator`) in ordinary
analysis Pipelines; `State.checkpoint(name)` / `SnapshotOperator` preserve
raw data beforehand (zero-copy — JAX arrays are immutable).

### D10 — Host-callback boundary policy

`jax.pure_callback` into numpy packages is used in exactly two situations:
(a) *permanently*, for inherently non-differentiable steps — RFI flagging via
MomentRFI (`MomentRFIFlaggingOperator`), where the output is boolean and a
gradient is meaningless; and (b) *temporarily*, as a correctness oracle for
physics awaiting a native port (`LimTODProjector`). Callbacks must never sit
inside a gradient path; the flags they produce flow to inference through
`MaskedGaussianLikelihood` (zero weight on flagged samples) and to
`SkySpaceFilter` noise weighting. Existing `aux["flags"]` are always passed
as MomentRFI's `prior_mask` so flaggers compose instead of clobbering.

### D11 — Composition is implicit in the signal path: graph-guided assembly

The canonical signal-path graph (`erhino/radio/graph.py`, rendered by
`Assembly.to_mermaid`) makes explicit composition unnecessary:
`assemble(*operators)` compiles a *set* of operator instances into the
Pipeline/SumOperator nesting induced on the graph — absent sources are
pruned, absent transforms contract to identity, junctions materialize as
SumOperator when two or more live branches converge (the upstream trunk
becomes branch 0). The folder is a *compiler*: the result is an ordinary
composite wrapped in `Assembly` (an operator carrying static lit/skipped
metadata), so jit/grad/`build_forward_fn`/`tree_at` are untouched.

Rules hardened by adversarial review:

- **Determinism**: junction branch order is the graph's edge declaration
  order, never call-site order — same provided set ⇒ identical tree, PRNG
  stream, and jit cache entry (regression-tested bitwise).
- **Source provenance**: every materialized Sum branch must contain a live
  source; a transform-rooted branch is an assembly-time `AssemblyError`, not
  a NoneType crash inside physics code.
- **Caller-data regimes**: a sourced assembly rejects caller `state.data`
  (it would be silently discarded); a source-free assembly is a transform
  chain that *requires* caller data. Both checks are structural (jit-safe).
- **Junctions are never operator slots**; multi-instance is allowed on
  `many=True` nodes (sibling Sum branches for sources, call-order chaining
  for the `filters` node).
- **Placement** is declared on the operator class (`graph_node` ClassVar,
  MRO-inherited so subclasses keep their base's slot), with `At(node, op)`
  as the per-instance escape hatch. Assembly metadata is hashable
  (`lit: tuple[str, ...]` + graph name); graph objects never enter the
  pytree.

**Equivalent-entry leaves**: the same physical effect may enter the chain at
different stages in different forms, so the graph reserves placeholder
leaves for each form even when no operator ships yet — ground spill as a
pre-beam *field* (`ground_field`, convolved by the shared beam node) or as a
post-beam *effective temperature* (`ground_pickup`, generic `t_sys_extra`);
the astro path as component fields through `beam` or pre-convolved via
`observed_astro_sky` (`SkySourceOperator`). The shared `beam` node stays a
single differentiable object (the #1 marginalisation target) rather than
fragmenting into per-operator copies.

Deferred: switched calibration loads need a *selector* junction kind
(replaces the antenna branch on the switching cycle) — a future SignalGraph
extension; region-coverage (one operator spanning several nodes) is not yet
modeled — `observed_astro_sky` covers today's case without it.

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
  calibration application                  radio/instrument/calibration.py
  sidereal / sky-space / Fourier filters   radio/filters/
Modular sky machinery (D8)
  sky models (params -> maps)              radio/sky/model.py
  projection engines (maps -> TOD)         radio/sky/projection.py
  composed sky slot                        radio/sky/source.py
Graph-guided assembly (D11)
  SignalGraph template + folder            core/graph.py
  canonical single-dish graph              radio/graph.py
```

Composition follows the physics, per the canonical signal-path graph
(`erhino/radio/graph.py`, D11): astrophysical components sum
(`SumOperator`), the ionosphere distorts that sum, RFI joins as a *pre-beam
field* (it enters through the sidelobes and is convolved by the shared beam
node), ground pickup joins as a *post-beam effective temperature*, and the
instrument chain is sequential (`Pipeline`). The chain order mirrors RHINO
paper Eq. 6, `P_rec = g (T_ant + T_nw + T_cw) + T_n`: sky-side temperatures
enter before the reflection/noise-wave terms, the CW tone joins *before*
bandpass and gain (it tracks gain drift only if it passes through the gain),
and thermal noise is added after the gain:

    astro = Pipeline(SumOperator(signal, foregrounds, point_sources), ionosphere)
    field = Pipeline(SumOperator(astro, rfi_field), beam)
    t_ant = SumOperator(field, ground_pickup)
    twin  = Pipeline(t_ant, atmosphere, noise_wave, cw_tone, bandpass, gain,
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
| BeamOperator | primary-beam convolution (harmonic alm rotation, ZYZ) | limTOD (TIBEC for full-Stokes); port task book: `docs/limtod-port-contract.md` |
| SystemTemperatureOperator | sky-side: atmosphere, ground spill (receiver temp lives in noise-wave T_0 / post-gain noise) | instrument configs |
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
- `SumOperator` discards branch writes to `coords`/`env`/`meta`/`aux` (by
  design, see D6) — an operator that must publish auxiliary output (e.g.
  flags) belongs in the sequential chain, not in a Sum branch.
- Typed PRNG keys (`jax.random.key`) are assumed throughout (jax ≥ 0.5).
