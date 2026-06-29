"""Sphinx configuration for the opus_rs Python API documentation.

Docstrings are single-sourced from the Rust ``///`` comments: PyO3 emits them
as ``__doc__`` on the compiled module, and ``autodoc`` reads them here. The
``napoleon`` extension renders the NumPy-style sections (Parameters/Returns/
Raises/Examples) the docstrings are written in. Build with::

    pip install opus_rs[docs] && sphinx-build -W docs/python docs/python/_build
"""

import importlib.metadata

project = "opus_rs"
author = "Jack Geraghty"
try:
    release = importlib.metadata.version("opus_rs")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "alabaster"
templates_path = ["_templates"]
exclude_patterns = ["_build"]
