"""Tests for graph-guided assembly on the canonical single-dish graph."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import erhino.radio as radio
from erhino import Pipeline, SumOperator
from erhino.core.graph import AssemblyError, At
from erhino.radio import (
    RADIO_GRAPH,
    ADCOperator,
    BeamOperator,
    ForegroundOperator,
    GainOperator,
    GlobalSignalOperator,
    GroundPickupOperator,
    IonosphereOperator,
    MatrixProjector,
    NoiseOperator,
    PointSourceOperator,
    PowerLawSkyModel,
    RFIOperator,
    SkyOperator,
    SkySourceOperator,
    assemble,
)

N_TIME, N_FREQ = 8, 4


def ops():
    """Fresh operator instances (deterministic parameters)."""
    return {
        "gs": GlobalSignalOperator(
            depth=jnp.array(0.2), centre=jnp.array(72e6), width=jnp.array(5e6)
        ),
        "fg": ForegroundOperator(
            amplitude=jnp.array(1e3), spectral_index=jnp.array(2.5), ref_freq=70e6
        ),
        "ps": PointSourceOperator(level=jnp.array(2.0)),
        "io": IonosphereOperator(delta=jnp.array(0.01), ref_freq=70e6),
        "rf": RFIOperator(amplitude=jnp.array(100.0), occupancy=0.05),
        "bm": BeamOperator(solid_angle=jnp.array(0.8)),
        "gd": GroundPickupOperator(coupling=jnp.array(0.01), t_ground=jnp.array(300.0)),
        "gn": GainOperator(gain=jnp.array(1.1)),
        "ns": NoiseOperator(sigma=jnp.array(0.5)),
        "ad": ADCOperator(scale=jnp.array(1.0), n_bits=14),
    }


class TestUserExamples:
    def test_sky_components_auto_sum(self, template_state):
        """User example 1: {21cm, foregrounds} -> the sky sum, nothing else."""
        o = ops()
        asm = assemble(o["gs"], o["fg"])
        assert isinstance(asm.operator, SumOperator)
        assert asm.operator.names == ("global_signal", "foregrounds")
        hand = SumOperator(o["gs"], o["fg"], names=("global_signal", "foregrounds"))
        assert jnp.array_equal(asm(template_state).data, hand(template_state).data)

    def test_beam_convolved_sky_part(self, template_state):
        """User example 2: {sky, ionosphere, beam} -> just that part of Tsys."""
        o = ops()
        sky = SkyOperator(amplitude=jnp.array(1e3))
        asm = assemble(sky, o["io"], o["bm"])
        hand = Pipeline(sky, o["io"], o["bm"], names=("uniform_sky", "ionosphere", "beam"))
        assert jnp.array_equal(asm(template_state).data, hand(template_state).data)
        assert asm.lit == ("uniform_sky", "ionosphere", "beam")


class TestCanonicalTopology:
    def test_rfi_field_passes_through_beam(self, template_state):
        """rfi_field is a pre-beam field: the beam factor applies to it."""
        o = ops()
        asm = assemble(o["rf"], o["bm"])
        direct = o["rf"](template_state).data
        assert jnp.allclose(asm(template_state).data, 0.8 * direct)

    def test_sky_source_enters_after_beam(self, template_state):
        """observed_astro_sky is post-beam: a standalone beam never touches it."""
        key = jax.random.key(5)
        source = SkySourceOperator(
            sky_model=PowerLawSkyModel(
                amplitude=jnp.ones(6), spectral_index=jnp.array(2.5),
                ref_freq=70e6, n_pix=6,
            ),
            projector=MatrixProjector(matrix=jax.random.normal(key, (N_TIME, 6))),
        )
        o = ops()
        asm = assemble(source, o["gd"])
        assert isinstance(asm.operator, SumOperator)
        assert asm.operator.names == ("observed_astro_sky", "ground_pickup")
        hand = SumOperator(source, o["gd"], names=("observed_astro_sky", "ground_pickup"))
        assert jnp.array_equal(asm(template_state).data, hand(template_state).data)

    def test_multi_component_foregrounds(self, template_state):
        """Two foreground components on one many=True node sum as siblings."""
        f1 = ForegroundOperator(
            amplitude=jnp.array(1e3), spectral_index=jnp.array(2.5), ref_freq=70e6
        )
        f2 = ForegroundOperator(
            amplitude=jnp.array(10.0), spectral_index=jnp.array(2.1), ref_freq=70e6
        )
        asm = assemble(f1, f2)
        expected = f1(template_state).data + f2(template_state).data
        assert jnp.allclose(asm(template_state).data, expected)

    def test_transform_rooted_branch_rejected(self):
        """{ionosphere, ground_pickup}: iono has no live source -> clear error."""
        o = ops()
        with pytest.raises(AssemblyError, match="no live source"):
            assemble(o["io"], o["gd"])

    def test_full_set_equals_hand_built_bitwise(self, template_state):
        o = ops()
        asm = assemble(*o.values())
        astro = Pipeline(
            SumOperator(o["gs"], o["fg"], o["ps"],
                        names=("global_signal", "foregrounds", "point_sources")),
            o["io"],
            names=("astro_sum", "ionosphere"),
        )
        field = SumOperator(astro, o["rf"], names=("astro_sum", "rfi_field"))
        upto_beam = Pipeline(field, o["bm"], names=("field_sum", "beam"))
        t_ant = SumOperator(upto_beam, o["gd"], names=("field_sum", "ground_pickup"))
        hand = Pipeline(
            t_ant, o["gn"], o["ns"], o["ad"],
            names=("t_ant_sum", "gain", "noise", "adc"),
        )
        assert eqx.tree_equal(asm.operator, hand)
        assert jnp.array_equal(asm(template_state).data, hand(template_state).data)

    def test_processing_segment_rides_along(self, template_state):
        from erhino.radio import FlaggingOperator, SiderealFilter

        o = ops()
        asm = assemble(
            SkyOperator(amplitude=jnp.array(1e3)), o["ns"],
            FlaggingOperator(threshold=2e3),
            SiderealFilter(n_days=2, mode="remove"),
        )
        out = asm(template_state)
        assert "flags" in out.aux
        assert out.data.shape == (N_TIME, N_FREQ)


class TestRegistryCompleteness:
    def test_every_concrete_operator_is_placeable(self):
        """Every exported radio operator class has a valid graph_node."""
        import inspect

        from erhino.core.operator import AbstractOperator

        missing = []
        for name in radio.__all__:
            obj = getattr(radio, name)
            if not (inspect.isclass(obj) and issubclass(obj, AbstractOperator)):
                continue
            if inspect.isabstract(obj) or name.startswith("Abstract"):
                continue
            node = getattr(obj, "graph_node", None)
            if node is None or node not in RADIO_GRAPH.nodes:
                missing.append(name)
        assert not missing, f"operators without a valid graph_node: {missing}"

    def test_reserved_leaves_exist(self):
        """The equivalent-entry placeholder leaves are part of the template."""
        for leaf in ("ground_field", "t_sys_extra"):
            assert RADIO_GRAPH.nodes[leaf].reserved

    def test_t_sys_extra_accepts_at_injection(self, template_state):
        asm = assemble(
            SkyOperator(amplitude=jnp.array(1e3)),
            BeamOperator(solid_angle=jnp.array(0.8)),
            At("t_sys_extra", GroundPickupOperator(
                coupling=jnp.array(0.02), t_ground=jnp.array(300.0)
            )),
        )
        assert isinstance(asm.operator, SumOperator)
        assert jnp.all(jnp.isfinite(asm(template_state).data))


class TestRendering:
    def test_mermaid_marks_segments_and_lit(self):
        o = ops()
        asm = assemble(o["gs"], o["gn"])
        mm = asm.to_mermaid()
        assert "class global_signal lit" in mm
        assert "class gain lit" in mm
        assert "class ionosphere wire" in mm  # traversed as identity
        assert "class flagging dim" in mm
