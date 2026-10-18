#!/usr/bin/env python3
"""Book kestrel-x (code X): the X-C's bare CFD sweeps — the shared kestrel script on the book's campaign module (kestrel_xc_campaigns).

    .venv/bin/python workflows/kestrel_x/airframe_cfd.py --variant X-C --chapter sweep [--list | --prepare-only]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kestrel"))
from airframe_cfd import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(book="kestrel-x"))
