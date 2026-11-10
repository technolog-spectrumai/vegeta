#!/usr/bin/env python3
"""Book emerillon (code E): the printed arm's FEA — the shared kestrel script on the book's campaign module (emerillon_campaigns).

    .venv/bin/python labs/aero/campaigns/side_labs/emerillon/arm_fea.py --variant ER2 --chapter arm [--list | --prepare-only]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kestrel"))
from structure_fea import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(book="emerillon"))
