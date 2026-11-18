"""Shared bits for the game's quick CFD: paths to Vegeta's studies, the air, and a YAML writer with comments.

THIS IS FOR A GAME: coarse meshes, few iterations, the numbers are good to ~10..20 %. Every script writes one YAML
file whose comments say what each number is and how far to trust it.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT / "labs" / "aero" / "exp", ROOT / "labs" / "land" / "designs"):   # golf_gun, sonic_ball, pekari_golf
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = HERE / "output"
RHO, NU, A_SOUND = 1.204, 1.51e-5, 343.2          # air at 20 C, sea level


def num(x, nd=4):
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int,)):
        return str(x)
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return f'"{x}"'


class Yaml:
    """A tiny YAML writer that keeps a comment beside every value (PyYAML drops comments)."""

    def __init__(self, title, about):
        self.lines = [f"# {title}", *[f"# {a}" for a in about], f"# written {_dt.datetime.now().isoformat(timespec='seconds')}", ""]
        self.ind = 0

    def section(self, key, comment=None):
        self.lines.append("  " * self.ind + f"{key}:" + (f"  # {comment}" if comment else ""))
        self.ind += 1
        return self

    def end(self):
        self.ind -= 1
        return self

    def kv(self, key, value, comment=None, nd=4):
        if isinstance(value, str) and not value.startswith("["):
            v = f'"{value}"'
        elif isinstance(value, (list, tuple)):
            v = "[" + ", ".join(num(x, nd) for x in value) + "]"
        else:
            v = num(value, nd)
        self.lines.append("  " * self.ind + f"{key}: {v}" + (f"  # {comment}" if comment else ""))
        return self

    def blank(self):
        self.lines.append("")
        return self

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n")
        print(f"wrote {path}")
        return path


def args(argv, default_out):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(default_out), help="the YAML file to write")
    ap.add_argument("--processors", type=int, default=2)
    ap.add_argument("--dry", action="store_true", help="prepare nothing, run nothing: the YAML with the hand values and NOT RUN")
    return ap.parse_args(argv)
