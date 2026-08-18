"""Sphinx configuration for the Suite3D documentation."""

from __future__ import annotations

from datetime import date

project = "Suite3D"
author = "Suite3D contributors"
copyright = f"{date.today().year}, {author}"

extensions: list[str] = []
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_title = "Suite3D documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_extra_path = [".nojekyll"]
