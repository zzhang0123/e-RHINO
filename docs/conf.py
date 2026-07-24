"""Sphinx configuration for the DIRT documentation (furo + MyST + autodoc)."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "DIRT"
author = "Zheng Zhang"
copyright = "2026, Zheng Zhang"

try:
    from dirt import __version__ as release
except ImportError:  # pragma: no cover - docs build without the package
    release = "0.0.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "numpyro": ("https://num.pyro.ai/en/stable", None),
}

html_theme = "furo"
html_title = "DIRT — Differentiable Instrument Response Twin"

exclude_patterns = ["_build"]

# The signal-path page is generated from the live template at build time,
# so the rendered graph can never drift from the code.
_SIGNAL_PATH_PAGE = pathlib.Path(__file__).parent / "signal-path.md"
try:
    from dirt.radio import RADIO_GRAPH

    _mermaid = RADIO_GRAPH.to_mermaid()
    _SIGNAL_PATH_PAGE.write_text(
        "# The canonical signal path\n\n"
        "The single-antenna template every assembly lights up — generated "
        "from `dirt.radio.RADIO_GRAPH` at documentation build time. `(+)` "
        "nodes are sum junctions, `(sw)` the antenna/cal-load selector; see "
        "the [tour](tour.md#4-graph-assembly) for the assembly rules and "
        "[the operator catalog](operators.md) for what lives at each node.\n\n"
        "```{mermaid}\n" + _mermaid + "\n```\n"
    )
except ImportError:  # pragma: no cover
    _SIGNAL_PATH_PAGE.write_text(
        "# The canonical signal path\n\n(dirt is not importable in this "
        "build environment; graph rendering skipped.)\n"
    )
