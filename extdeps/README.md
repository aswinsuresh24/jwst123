# External dependencies (`extdeps`)

Vendored / customized third-party packages used by **jwst123**.

| Path | Package | Role |
| --- | --- | --- |
| [`jhat/`](jhat/) | **jhat** (custom) | JWST/HST Alignment Tool used by `jwst123.alignment` |

These packages are **not** installed from PyPI by default. Install them from this
tree (see the root `README.md`) so the alignment pipeline gets the repository’s
custom JHAT build.
