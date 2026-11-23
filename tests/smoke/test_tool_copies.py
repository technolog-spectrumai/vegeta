"""The tools' deliberate copies (result.py, _process.py, cache.py, _progress.py, _watermark.py) stay identical: each tool
is standalone, so they are copied, not shared — a fix goes into the master and ``python tools/sync_copies.py --write``."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import sync_copies  # noqa: E402


def test_copies_are_identical():
    assert sync_copies.drift() == []
