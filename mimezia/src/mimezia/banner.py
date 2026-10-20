"""The notice that says which mode is in force."""
from __future__ import annotations

COLOURS = {"mock": "#b00020", "off": "#555555", "real": "#0b6e4f"}


def banner(mode: str, text: str) -> str:
    """Show ``text`` loudly: an HTML box in Jupyter, a starred line elsewhere. Returns the text."""
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            raise ImportError
        from IPython.display import HTML, display
        c = COLOURS.get(mode, "#333333")
        display(HTML(f'<div style="border:3px solid {c};padding:8px;font-weight:bold;color:{c}">{text}</div>'))
    except Exception:
        print("*" * 20, text, "*" * 20, flush=True)
    return text
