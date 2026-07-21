# e-RHINO

**erhino** is a general-purpose, extensible, *differentiable* scientific pipeline
framework built on [JAX](https://github.com/jax-ml/jax) and
[Equinox](https://github.com/patrick-kidger/equinox).

Its first application is a **digital twin of the RHINO radio telescope** (a large
pyramidal horn antenna targeting the 21 cm global signal at 60–85 MHz), but the
core is domain-agnostic by construction.

> **Core principle: everything is an Operator acting on a State.**

Full usage documentation is under construction — see `DESIGN.md` (forthcoming)
and `examples/` for intended usage.

## Install (development)

```bash
uv sync
```

## License

MIT
