#!/usr/bin/env python3
"""Book kestrel-x (code X): the X-C panel spar's FEA — the shared kestrel script on the book's campaign module (kestrel_xc_campaigns).

    .venv/bin/python labs/aero/campaigns/side_labs/kestrel_x/panel_fea.py --variant X-C --chapter panel [--list | --prepare-only]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kestrel"))
from structure_fea import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(book="kestrel-x"))
