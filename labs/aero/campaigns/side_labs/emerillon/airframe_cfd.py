#!/usr/bin/env python3
"""Book emerillon (code E): ER1 / ER2 airframe CFD — the shared kestrel script on the book's campaign module (emerillon_campaigns).

    .venv/bin/python labs/aero/campaigns/side_labs/emerillon/airframe_cfd.py --variant ER2 --chapter drag [--list | --prepare-only]
    .venv/bin/python labs/aero/campaigns/side_labs/emerillon/airframe_cfd.py --variant ER2 --chapter alpha --quality fine --processors 8
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kestrel"))
from airframe_cfd import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(book="emerillon"))
