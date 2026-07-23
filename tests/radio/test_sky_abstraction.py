"""Tests for the modular sky abstraction: SkyModel x SkyProjector x SkySourceOperator."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from erhino import Pipeline, State
from erhino.core.errors import StateValidationError
from erhino.radio import GainOperator
from erhino.radio.sky import (
    LimTODProjector,
    MatrixProjector,
    MModeProjector,
    PowerLawSkyModel,
    SkySourceOperator,
    UniformSkyModel,
)

N_TIME, N_FREQ, N_PIX = 8, 4, 6


@pytest.fixture
def key():
    return jax.random.key(0)


def dot(a, b):
    return jnp.sum(a * b)


class TestSkyModels:
    def test_uniform_shape_and_value(self):
        sky = UniformSkyModel(amplitude=jnp.array(5.0), n_pix=N_PIX)(jnp.linspace(60e6, 85e6, 4))
        assert sky.shape == (4, N_PIX)
        assert jnp.all(sky == 5.0)

    def test_power_law(self):
        freq = jnp.linspace(60e6, 85e6, N_FREQ)
        model = PowerLawSkyModel(
            amplitude=jnp.arange(1.0, N_PIX + 1.0),
            spectral_index=jnp.array(2.5),
            ref_freq=70e6,
            n_pix=N_PIX,
        )
        sky = model(freq)
        assert sky.shape == (N_FREQ, N_PIX)
        expected = (freq[2] / 70e6) ** (-2.5) * 3.0
        assert jnp.allclose(sky[2, 2], expected)

    def test_validation(self):
        with pytest.raises(StateValidationError, match="n_pix"):
            UniformSkyModel(amplitude=jnp.array(1.0), n_pix=0)
        with pytest.raises(StateValidationError, match="ref_freq"):
            PowerLawSkyModel(
                amplitude=jnp.array(1.0), spectral_index=jnp.array(2.5),
                ref_freq=-1.0, n_pix=N_PIX,
            )


class TestMatrixProjector:
    @pytest.mark.parametrize("shape", [(N_TIME, N_PIX), (N_FREQ, N_TIME, N_PIX)])
    def test_forward_shape(self, key, shape, coords):
        A = jax.random.normal(key, shape)
        sky = jnp.ones((N_FREQ, N_PIX))
        out = MatrixProjector(matrix=A).forward(sky, coords)
        assert out.shape == (N_TIME, N_FREQ)

    @pytest.mark.parametrize("shape", [(N_TIME, N_PIX), (N_FREQ, N_TIME, N_PIX)])
    def test_adjoint_dot_identity(self, key, shape, coords):
        """<A x, y> == <x, A^T y> — the property SkySpaceFilter relies on."""
        k1, k2, k3 = jax.random.split(key, 3)
        proj = MatrixProjector(matrix=jax.random.normal(k1, shape))
        x = jax.random.normal(k2, (N_FREQ, N_PIX))
        y = jax.random.normal(k3, (N_TIME, N_FREQ))
        lhs = dot(proj.forward(x, coords), y)
        rhs = dot(x, proj.adjoint(y, coords))
        assert jnp.allclose(lhs, rhs, rtol=1e-5)

    def test_pixel_mismatch_raises(self, coords):
        proj = MatrixProjector(matrix=jnp.ones((N_TIME, N_PIX)))
        with pytest.raises(StateValidationError, match="pixels"):
            proj.forward(jnp.ones((N_FREQ, N_PIX + 1)), coords)

    def test_bad_ndim_rejected(self):
        with pytest.raises(StateValidationError, match="matrix"):
            MatrixProjector(matrix=jnp.ones(N_PIX))


class TestMModeProjector:
    @pytest.fixture
    def transfer(self, key):
        re, im = jax.random.normal(key, (2, N_FREQ, N_TIME, N_PIX))
        return re + 1j * im

    def test_forward_real_and_shaped(self, transfer, coords):
        out = MModeProjector(transfer=transfer).forward(jnp.ones((N_FREQ, N_PIX)), coords)
        assert out.shape == (N_TIME, N_FREQ)
        assert jnp.isrealobj(out)

    def test_adjoint_dot_identity(self, transfer, key, coords):
        proj = MModeProjector(transfer=transfer)
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (N_FREQ, N_PIX))
        y = jax.random.normal(k2, (N_TIME, N_FREQ))
        lhs = dot(proj.forward(x, coords), y)
        rhs = dot(x, proj.adjoint(y, coords))
        assert jnp.allclose(lhs, rhs, rtol=1e-4)

    def test_nm_mismatch_raises(self, transfer, coords):
        bad = MModeProjector(transfer=transfer[:, :-1, :])
        with pytest.raises(StateValidationError, match="n_m"):
            bad.forward(jnp.ones((N_FREQ, N_PIX)), coords)


class TestSkySourceOperator:
    @pytest.fixture
    def source(self, key):
        return SkySourceOperator(
            sky_model=PowerLawSkyModel(
                amplitude=jnp.ones(N_PIX), spectral_index=jnp.array(2.5),
                ref_freq=70e6, n_pix=N_PIX,
            ),
            projector=MatrixProjector(matrix=jax.random.normal(key, (N_TIME, N_PIX))),
        )

    def test_produces_data(self, source, template_state):
        out = source(template_state)
        assert out.data.shape == (N_TIME, N_FREQ)

    def test_jit_and_grad_reach_sky_params(self, source, template_state):
        pipe = Pipeline(source, GainOperator(gain=jnp.array(1.1)), names=("sky", "gain"))
        out = eqx.filter_jit(pipe)(template_state)

        def loss(pipe):
            return jnp.mean((pipe(template_state).data - out.data * 1.05) ** 2)

        grads = eqx.filter_grad(loss)(pipe)
        g_index = grads["sky"].sky_model.spectral_index
        g_amp = grads["sky"].sky_model.amplitude
        assert jnp.isfinite(g_index) and g_index != 0
        assert jnp.all(jnp.isfinite(g_amp)) and jnp.any(g_amp != 0)

    def test_projector_swaps_independently(self, source, template_state, key):
        """The modularity contract: same sky model, different engine."""
        re, im = jax.random.normal(key, (2, N_FREQ, N_TIME, N_PIX))
        swapped = eqx.tree_at(
            lambda s: s.projector, source, MModeProjector(transfer=re + 1j * im)
        )
        out = swapped(template_state)
        assert out.data.shape == (N_TIME, N_FREQ)
        assert isinstance(swapped.sky_model, PowerLawSkyModel)  # sky untouched

    def test_type_validation(self, key):
        with pytest.raises(StateValidationError, match="sky_model"):
            SkySourceOperator(
                sky_model=jnp.ones(3),
                projector=MatrixProjector(matrix=jnp.ones((N_TIME, N_PIX))),
            )
        with pytest.raises(StateValidationError, match="projector"):
            SkySourceOperator(
                sky_model=UniformSkyModel(amplitude=jnp.array(1.0), n_pix=N_PIX),
                projector=jnp.ones(3),
            )

    def test_requires_coords(self, source):
        with pytest.raises(StateValidationError, match="coords"):
            source(State())


class TestLimTODProjector:
    def test_validates_coordinate_requirements(self, template_state):
        proj = LimTODProjector(beam_maps=jnp.ones((N_FREQ, N_PIX)), lat_deg=53.2)
        with pytest.raises(StateValidationError, match="pointing"):
            proj.forward(jnp.ones((N_FREQ, N_PIX)), template_state.coords)

    def test_oracle_matches_limtod(self, template_state):
        """End-to-end oracle test — runs only where limTOD is installed."""
        limtod = pytest.importorskip("limTOD.simulator")
        import numpy as np

        nside = 4
        n_pix = 12 * nside**2
        rng = np.random.default_rng(0)
        beam = jnp.asarray(rng.random((2, n_pix)))
        sky = jnp.asarray(rng.random((2, n_pix)))
        coords = template_state.coords.replace(
            freq=template_state.coords.freq[:2],
            pointing=jnp.tile(jnp.array([[0.0, 90.0]]), (N_TIME, 1)),
            extra={"lst_deg": jnp.linspace(0.0, 30.0, N_TIME)},
        )
        proj = LimTODProjector(beam_maps=beam, lat_deg=53.2)
        out = proj.forward(sky, coords)
        direct = limtod.generate_TOD_sky(
            np.asarray(beam[0]), np.asarray(sky[0]),
            np.linspace(0.0, 30.0, N_TIME), 53.2,
            np.zeros(N_TIME), np.full(N_TIME, 90.0), np.zeros(N_TIME),
        )
        assert out.shape == (N_TIME, 2)
        assert jnp.allclose(out[:, 0], jnp.asarray(direct), rtol=1e-5)
