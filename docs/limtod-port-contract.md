# limTOD → JAX port contract

> **STATUS: DELIVERED** (2026-07-23, merged as `NativeLimTODProjector`).
> The port lives in the limTOD repo as the `limtod_jax` package
> (`angles` / `wigner` / `alm` / `hpx` / `core` / `projection` modules, with
> its own float64 test suite carrying the 1e-6 oracle guarantees below).
> e-RHINO's adapter is `erhino.radio.sky.NativeLimTODProjector`
> (`src/erhino/radio/sky/native.py`; lazy import — install the engine with
> `pip install -e '<limTOD>[jax]'`). e-RHINO-side contract tests:
> `tests/radio/test_sky_abstraction.py::TestNativeLimTODProjector`
> (float32-scale; the healpy-dependent oracle comparisons auto-skip where
> numpy limTOD is not installed).
>
> One deliberate semantic pin: the native chain equals numpy
> `generate_TOD_sky(..., truncate_frac_thres=0.0)` — the default `1e-10`
> map truncation is a *nonlinear* cleanup step and is excluded from the
> linear contract. Enable `jax_enable_x64` for quantitative work: the
> map↔alm transforms carry O(10%) error in float32.
>
> The rest of this document is retained as the acceptance specification the
> implementation satisfies (and the reference for future extensions, e.g.
> full-Stokes).

Task book for the agents rewriting limTOD's sky→TOD machinery in
JAX + Equinox. The goal: replace e-RHINO's oracle bridge
(`erhino.radio.sky.projection.LimTODProjector`, a `jax.pure_callback` into
numpy limTOD) with a **native, differentiable** projector satisfying the same
interface.

## Where the port fits

e-RHINO defines the seam; the port fills it:

```
erhino.radio.sky.AbstractSkyProjector          # the interface (already exists)
    forward(sky, coords) -> (n_time, n_freq)   # observe the sky
    adjoint(tod, coords) -> (n_freq, n_pix)    # transpose (linear projectors)

today:   LimTODProjector  = pure_callback -> numpy limTOD   (oracle, not differentiable)
         MatrixProjector  = precomputed generate_sky2sys_projection matrix (differentiable, fixed pointing)
target:  NativeLimTODProjector = pure JAX                    (differentiable, general pointing)
```

The rewritten functions should live in the new JAX limTOD package (not in
e-RHINO); e-RHINO will add a thin adapter (~15 lines) once they exist.

## Functions to port (source: `limTOD/limTOD/simulator.py`)

Priority order; each numpy original is the correctness oracle for its port.

| # | Original | Role | Difficulty |
|---|---|---|---|
| 1 | `zyz_of_pointing(LST_deg, lat_deg, az_deg, el_deg, selfrot_deg)` | pointing → ZYZ Euler angles | trivial (pure trig) |
| 2 | `zyzyz2zyz(...)` | Euler-angle composition | trivial |
| 3 | `_rotate_healpix_map(alm, psi, theta, phi, ...)` — the alm rotation | rotate beam alms to equatorial pointing | **hard** — this is the crux |
| 4 | `_beam_weighted_sum(beam_map, sky_map, normalize)` | beam·sky dot product | trivial |
| 5 | `generate_TOD_sky(beam_map, sky_map, LST..., az..., el..., selfrot...)` | full chain over pointings | composition of 1–4 |
| 6 | (phase 2, optional) `generate_sky2sys_projection(...)` | native projection-matrix builder | moderate |

**Recommended formulation for #3+#5**: stay in harmonic space — rotate the
beam alms per pointing (Wigner-D application) and take the harmonic dot
product with the sky alms, instead of round-tripping through maps
(`alm2map` per sample). Candidate kernels: `s2fft`'s Wigner-d machinery
(allowed as an optional dependency), or an explicit Wigner-d recursion.
Whichever is chosen, **verify the rotation convention numerically against
`healpy.rotate_alm` as used by `_rotate_healpix_map` — do not trust sign/order
conventions on paper.**

## Target signatures (pure functions, all jit/vmap/grad-safe)

```python
def zyz_of_pointing(lst_deg, lat_deg, az_deg, el_deg, selfrot_deg) -> tuple[Array, Array, Array]:
    """Degrees in (public API), radians out (psi, theta, phi) — RHINO family convention."""

def rotate_alm(alm: Array, psi: Array, theta: Array, phi: Array, *, lmax: int) -> Array:
    """Wigner rotation of spherical-harmonic coefficients. lmax STATIC."""

def beam_weighted_sum(beam_alm: Array, sky_alm: Array, *, normalize: bool = False) -> Array:
    """Harmonic-space beam-weighted average -> scalar antenna temperature."""

def generate_tod_sky(beam_alm, sky_alm, zyz_angles, *, lmax: int) -> Array:
    """(n_time, 3) angles -> (n_time,) TOD. Implemented as vmap/scan over #3+#4."""
```

An Equinox adapter then satisfies the e-RHINO interface:

```python
class NativeLimTODProjector(AbstractSkyProjector):
    beam_alms: Array                      # traced — differentiable beam params later
    lat_deg: float = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    def forward(self, sky, coords): ...   # map2alm (static grid) -> generate_tod_sky per freq
    def adjoint(self, tod, coords): ...   # the exact transpose of forward (required!)
```

## Hard requirements

1. **Purity** — no global state, no mutation; safe under `jit`, `vmap`,
   `grad`. No value-dependent Python control flow (shape/ndim checks fine).
2. **Differentiability** — gradients w.r.t. BOTH `sky_alm` and `beam_alm`
   must be finite and correct (check one component against finite
   differences). The forward map is linear in each — exploit that.
3. **vmap** — over the time axis (pointings) and the frequency axis.
4. **Precision** — Wigner recursions accumulate error: compute in float64
   (tests run with `jax.config.update("jax_enable_x64", True)`); never
   hardcode float32 (a float32 hardcode was already caught once in e-RHINO
   review — follow ambient dtype).
5. **Static vs traced** — `nside`/`lmax`: static ints. Angles, alms, maps:
   traced. No hashing of arrays.
6. **Dependencies** — jax + equinox; `s2fft` permitted as an extra; **no
   healpy inside the JAX path** (healpy stays in tests as the oracle).
7. **Conventions** — degrees public / radians internal; ZYZ Euler order
   matching `zyz_of_pointing`; HEALPix RING ordering; `normalize_beam`
   semantics identical to numpy limTOD.

## Acceptance tests (ship with the port; all must pass)

1. **Oracle equivalence**: for `nside ∈ {8, 16}`, random beam & sky maps,
   pointing sets that include the extremes — zenith (`el=90`), low elevation,
   `lat ∈ {53.24, 0, -90}`, `LST ∈ {0, 179.9, 359.9}` — the native
   `generate_tod_sky` matches `limTOD.simulator.generate_TOD_sky` to
   rel. err < 1e-6 in float64. *Extreme corners are mandatory, not optional:
   failure modes concentrate at boundaries (zenith gimbal, poles), and a
   moderate-parameter probe will miss them.*
2. **Adjoint dot-test**: `<forward(x), y> == <x, adjoint(y)>` to rel. 1e-6
   (this is what e-RHINO's `SkySpaceFilter` map-making relies on).
3. **jit**: `eqx.filter_jit` compiles; a second call with same shapes does
   not retrace.
4. **grad**: finite, nonzero, and matches finite differences (1e-4 rel) for
   one sky and one beam component.
5. **vmap consistency**: vmapped-over-frequency equals the Python loop.
6. Tests live in the new package's suite as parametrized pytest
   (see e-RHINO `tests/radio/test_sky_abstraction.py::TestLimTODProjector::
   test_oracle_matches_limtod` for the oracle-test pattern to reuse).

## Out of scope (for this port)

Full-Stokes beams (TIBEC integration — later), horizontal masks, MPI
(`mpiutil`), gain/noise generation (`TODSim.generate_TOD` — e-RHINO models
those as separate operators), map-making (`HPW_filter` — e-RHINO's
`SkySpaceFilter` already covers it in JAX).
