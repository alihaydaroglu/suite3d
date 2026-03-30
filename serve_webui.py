"""Launch the suite3d web UI for job monitoring.

Usage:
    panel serve serve_webui.py --show
"""
from webui.webui import ui  # noqa: F401

ui.servable()
