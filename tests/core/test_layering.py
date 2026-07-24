"""Guard the architectural hard rule: dirt.core is domain-agnostic.

DESIGN.md: "dirt.core never imports from radio or inference. If the
framework proves reusable beyond radio astronomy, core graduates to its own
package by moving one directory." This test makes that promise mechanical.
"""

import pathlib

import dirt.core

CORE_DIR = pathlib.Path(dirt.core.__file__).parent
# Match actual import statements, not prose mentions in docstrings.
FORBIDDEN = (
    "from dirt.radio",
    "import dirt.radio",
    "from dirt.inference",
    "import dirt.inference",
)


def test_core_never_imports_domain_layers():
    offenders = []
    for path in CORE_DIR.glob("*.py"):
        source = path.read_text()
        for banned in FORBIDDEN:
            if banned in source:
                offenders.append(f"{path.name}: contains {banned!r}")
    assert not offenders, (
        "dirt.core must stay domain-agnostic (extractable), but:\n"
        + "\n".join(offenders)
    )
