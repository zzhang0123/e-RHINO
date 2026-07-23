# Changelog

## 0.1.0 (unreleased)

Initial architecture of the differentiable scientific pipeline framework.

### Integration seams (added after initial architecture)

- **Modular sky** (`erhino.radio.sky`): `AbstractSkyModel` (params → maps) ×
  `AbstractSkyProjector` (maps → TOD, with `adjoint` for linear engines),
  composed by `SkySourceOperator`. Engines: `MatrixProjector` (precomputed
  `generate_sky2sys_projection` matrix — differentiable today),
  `LimTODProjector` (pure_callback oracle into numpy limTOD),
  `MModeProjector` (m-mode transfer, drift scans). Port task book for the
  native JAX limTOD rewrite: `docs/limtod-port-contract.md`.
- **Native limTOD projector** (`NativeLimTODProjector`): the port contract
  delivered — pure-JAX sky→TOD chain (Wigner rotation + harmonic beam sum
  from the `limtod_jax` package in the limTOD repo), general pointing,
  jit/vmap-safe, differentiable w.r.t. both sky maps and beam alms, exact
  adjoint for `SkySpaceFilter` map-making. Matches numpy
  `generate_TOD_sky(..., truncate_frac_thres=0.0)`; enable x64 for
  quantitative accuracy. Optional dependency: `pip install -e '<limTOD>[jax]'`.
- **MomentRFI** (`erhino.radio.backend`): `MomentRFIFlaggingOperator`
  (host-callback into `IterativeSurfaceFitter`; existing flags become
  `prior_mask`) + `MaskedGaussianLikelihood` (flags → noise covariance).
- **Filters** (`erhino.radio.filters`): `AbstractLinearFilter`
  (extract/remove projection semantics) with `SiderealFilter` (day-repeating
  subspace), `SkySpaceFilter` (CG map-make/reproject through any linear sky
  projector), `FourierBandFilter` (fringe-rate/delay bands); plus
  `ApplyCalibrationOperator` and raw-data preservation via
  `State.checkpoint` / `SnapshotOperator`.

### Core (`erhino.core`)

- `State`: immutable pytree container (traced `data`/`coords`/`env`/`aux`/`key`,
  static hashable `meta` via `FrozenMapping`); functional updates
  (`replace`/`with_data`) and the PRNG protocol
  (`subkey, state = state.next_key()`).
- `AbstractOperator` / `LambdaOperator`: the universal `State -> State`
  contract with declarative `requires`/`provides`.
- `Pipeline`: sequential named composition (composite pattern — nests freely);
  `run_with_intermediates`, `replace_stage`, name/index access.
- `SumOperator`: parallel additive composition for source-type branches;
  per-branch PRNG subkeys; leafwise pytree accumulation with loud trace-time
  errors on shape/structure mismatch and dataless branches.

### Radio (`erhino.radio`) — placeholder physics, real contracts

- Reorganized by the single-dish element taxonomy:
  `sky/` (uniform, global signal, foregrounds, point sources),
  `environment/` (ionosphere, ground pickup, RFI),
  `instrument/` (beam, sky-side system temperature, noise-wave/reflection
  terms, CW calibration tone, bandpass, gain, thermal noise, EMI, ADC),
  `backend/` (flagging, averaging). Flat `erhino.radio` API preserved.
- Chain ordering follows the RHINO system equation
  `P_rec = g (T_ant + T_nw + T_cw) + T_n`: CW tone before bandpass/gain
  (it tracks gain drift only through the gain); sky-side temperatures before
  the reflection/noise-wave terms.
- `NoiseWaveOperator` preserves linearity in `t_nw = (T_unc, T_cos, T_sin)` —
  the `d = H t_nw` structure GCR sampling relies on.

### Inference (`erhino.inference`)

- `build_forward_fn(pipeline, state_template, filter_spec)`: the single seam
  between forward models and inference (Equinox partition/combine).
- `Likelihood` protocol + `GaussianLikelihood`; minimal working
  `GradientCalibrator`; `to_numpyro_model` stub (NumPyro optional extra).

### Project

- src layout, hatchling, uv-native; pytest with 80% coverage floor
  (currently ~97%); ruff clean; runnable end-to-end demo
  (`examples/radio_digital_twin.py`) including gradient recovery of a known
  gain.
