"""Tests for SumOperator: parallel additive composition."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from erhino.core.combinators import SumOperator
from erhino.core.errors import PipelineError
from erhino.core.operator import AbstractOperator
from erhino.core.pipeline import Pipeline
from erhino.core.state import State


class Constant(AbstractOperator):
    """Source-type test operator: creates data, ignoring existing data."""

    value: jax.Array

    def __call__(self, state: State) -> State:
        return state.with_data(self.value * jnp.ones(3))


class NoisySource(AbstractOperator):
    """Source-type operator that consumes randomness."""

    def __call__(self, state: State) -> State:
        subkey, state = state.next_key()
        return state.with_data(jax.random.normal(subkey, (3,)))


class TestConstruction:
    def test_empty_rejected(self):
        with pytest.raises(PipelineError, match="at least one"):
            SumOperator()

    def test_non_operator_rejected(self):
        with pytest.raises(PipelineError, match="AbstractOperator"):
            SumOperator(lambda s: s)

    def test_names(self):
        op = SumOperator(Constant(value=jnp.array(1.0)), Constant(value=jnp.array(2.0)))
        assert op.names == ("constant", "constant_2")
        named = SumOperator(Constant(value=jnp.array(1.0)), names=("fg",))
        assert named.names == ("fg",)

    def test_access(self):
        op = SumOperator(
            Constant(value=jnp.array(1.0)),
            Constant(value=jnp.array(2.0)),
            names=("a", "b"),
        )
        assert len(op) == 2
        assert op["b"].value == 2.0
        assert op[0].value == 1.0


class TestExecution:
    def test_sums_branch_contributions(self):
        op = SumOperator(Constant(value=jnp.array(1.0)), Constant(value=jnp.array(2.0)))
        out = op(State())
        assert jnp.array_equal(out.data, jnp.full(3, 3.0))

    def test_replaces_existing_data(self):
        """SumOperator is a source combinator: input data is not accumulated."""
        op = SumOperator(Constant(value=jnp.array(1.0)))
        out = op(State(data=jnp.full(3, 100.0)))
        assert jnp.array_equal(out.data, jnp.ones(3))

    def test_preserves_context(self):
        op = SumOperator(Constant(value=jnp.array(1.0)))
        out = op(State(meta={"telescope": "demo"}))
        assert out.meta["telescope"] == "demo"

    def test_branches_can_be_pipelines(self):
        """Composite pattern: a branch may itself be a Pipeline (e.g. sky -> ionosphere)."""
        double = Pipeline(
            Constant(value=jnp.array(2.0)),
            # multiply existing data by 3 via a nested operator
            _Scale(factor=jnp.array(3.0)),
        )
        op = SumOperator(double, Constant(value=jnp.array(1.0)))
        out = op(State())
        assert jnp.array_equal(out.data, jnp.full(3, 7.0))  # 2*3 + 1


class _Scale(AbstractOperator):
    factor: jax.Array

    def __call__(self, state: State) -> State:
        return state.with_data(state.data * self.factor)


class _PyTreeSource(AbstractOperator):
    """Source producing a dict-of-streams payload (documented State.data form)."""

    value: jax.Array

    def __call__(self, state: State) -> State:
        return state.with_data({"tod": self.value * jnp.ones(3), "tone": self.value * jnp.ones(2)})


class _ShapedSource(AbstractOperator):
    n: int = eqx.field(static=True)

    def __call__(self, state: State) -> State:
        return state.with_data(jnp.ones((self.n, 2)))


class TestDataPytreeSemantics:
    """Regression tests: accumulation must be leafwise, loud on mismatch."""

    def test_dict_payloads_sum_leafwise(self):
        op = SumOperator(_PyTreeSource(value=jnp.array(1.0)), _PyTreeSource(value=jnp.array(2.0)))
        out = op(State())
        assert jnp.array_equal(out.data["tod"], jnp.full(3, 3.0))
        assert jnp.array_equal(out.data["tone"], jnp.full(2, 3.0))

    def test_tuple_payloads_sum_not_concatenate(self):
        class TupleSource(AbstractOperator):
            v: jax.Array

            def __call__(self, state):
                return state.with_data((self.v * jnp.ones(2), self.v * jnp.ones(4)))

        out = SumOperator(TupleSource(v=jnp.array(1.0)), TupleSource(v=jnp.array(2.0)))(State())
        assert isinstance(out.data, tuple) and len(out.data) == 2  # NOT a 4-tuple
        assert jnp.array_equal(out.data[0], jnp.full(2, 3.0))

    def test_shape_mismatch_raises_not_broadcasts(self):
        op = SumOperator(_ShapedSource(n=8), _ShapedSource(n=1), names=("full", "avg"))
        with pytest.raises(PipelineError, match="shape"):
            op(State())

    def test_structure_mismatch_raises(self):
        op = SumOperator(
            Constant(value=jnp.array(1.0)),
            _PyTreeSource(value=jnp.array(1.0)),
            names=("array", "dict"),
        )
        with pytest.raises(PipelineError, match="structure"):
            op(State())

    @pytest.mark.parametrize("none_first", [True, False])
    def test_none_data_branch_raises_regardless_of_order(self, none_first):
        passthrough = _Identity()
        source = Constant(value=jnp.array(1.0))
        branches = (passthrough, source) if none_first else (source, passthrough)
        op = SumOperator(*branches, names=("a", "b"))
        with pytest.raises(PipelineError, match="produced no data"):
            op(State())  # state.data is None; passthrough contributes None

    def test_branches_never_see_caller_data(self):
        """D6 enforced: branch input data is stripped to None."""
        from erhino.core.operator import LambdaOperator

        probe = LambdaOperator(
            fn=lambda s: s.with_data(jnp.ones(3) if s.data is None else jnp.full(3, 999.0))
        )
        out = SumOperator(probe)(State(data=jnp.full(3, 5.0)))
        assert jnp.array_equal(out.data, jnp.ones(3))


class TestReplaceBranch:
    def test_swaps_and_preserves_names(self):
        op = SumOperator(
            Constant(value=jnp.array(1.0)), Constant(value=jnp.array(2.0)),
            names=("a", "b"),
        )
        op2 = op.replace_branch("b", Constant(value=jnp.array(10.0)))
        assert jnp.array_equal(op2(State()).data, jnp.full(3, 11.0))
        assert op2.names == op.names
        assert op["b"].value == 2.0  # original untouched

    def test_rejects_non_operator(self):
        op = SumOperator(Constant(value=jnp.array(1.0)))
        with pytest.raises(PipelineError, match="AbstractOperator"):
            op.replace_branch(0, lambda s: s)


class _Identity(AbstractOperator):
    def __call__(self, state: State) -> State:
        return state


class TestPRNGSemantics:
    def test_branches_get_independent_keys(self):
        """Two stochastic branches must NOT reuse the same key."""
        op = SumOperator(NoisySource(), NoisySource())
        single = SumOperator(NoisySource())
        out2 = op(State(key=jax.random.key(0)))
        out1 = single(State(key=jax.random.key(0)))
        # If keys were reused, the two-branch sum would be exactly 2x one branch.
        assert not jnp.allclose(out2.data, 2.0 * out1.data)

    def test_output_key_advances(self):
        op = SumOperator(Constant(value=jnp.array(1.0)))
        s = State(key=jax.random.key(0))
        out = op(s)
        assert not jnp.array_equal(
            jax.random.key_data(out.key), jax.random.key_data(s.key)
        )

    def test_reproducible(self):
        op = SumOperator(NoisySource(), NoisySource())
        a = op(State(key=jax.random.key(7)))
        b = op(State(key=jax.random.key(7)))
        assert jnp.array_equal(a.data, b.data)

    def test_keyless_state_works_for_deterministic_branches(self):
        op = SumOperator(Constant(value=jnp.array(1.0)))
        out = op(State())
        assert out.key is None


class TestJaxTransforms:
    def test_jit_and_grad(self):
        op = SumOperator(Constant(value=jnp.array(1.0)), Constant(value=jnp.array(2.0)))
        out = eqx.filter_jit(op)(State())
        assert jnp.array_equal(out.data, jnp.full(3, 3.0))

        def loss(op):
            return jnp.sum(op(State()).data)

        grads = eqx.filter_grad(loss)(op)
        assert jnp.allclose(grads.branches[0].value, 3.0)  # d(sum(v*ones(3)))/dv
        assert jnp.allclose(grads.branches[1].value, 3.0)

    def test_is_an_operator_and_nests_in_pipeline(self):
        sky = SumOperator(Constant(value=jnp.array(1.0)), Constant(value=jnp.array(2.0)))
        full = Pipeline(sky, _Scale(factor=jnp.array(10.0)))
        out = full(State())
        assert jnp.array_equal(out.data, jnp.full(3, 30.0))
