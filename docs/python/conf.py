import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath("."))

project = "ruopus"
copyright = "2026, Jack Geraghty"
author = "Jack Geraghty"
try:
    release = importlib.metadata.version("ruopus")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_parser",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme_options = {
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#1b4f72",
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
