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
| SkyOperator | sky maps + spectral models, observed along pointing | limTOD (MERS for foregrounds) |
| BeamOperator | primary-beam convolution (harmonic alm rotation, ZYZ) | limTOD (TIBEC for full-Stokes) |
| SystemTemperatureOperator | receiver temperature, ground spill, atmosphere | limTOD / instrument configs |
| ReceiverOperator | bandpass; later reflection/impedance effects | instrument configs |
| GainOperator | g(t) with 1/f flicker fluctuations | limTOD, hydra-tod |
| NoiseOperator | radiometer equation, 1/f covariance | limTOD, hydra-tod |
| ADCOperator | true quantization + straight-through estimator | — |
| BackendOperator | integration, RFI flagging, waterfall products | MomentRFI |

Plus: NumPyro bridge (`to_numpyro_model`), optax-based calibrators,
uncertainty propagation, neural surrogate operators, multi-experiment support.

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
