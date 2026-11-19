"""Resume support for the ARIGO-1 runners: the YAML each writes is also its checkpoint.

A run reads its YAML back and skips every item already finished there. The YAML records a fingerprint of the
design and model files (``designs/arigo1*.py``); if they changed since, the old results are set aside
(``<name>.stale.yaml``) and the run starts over, so results from an older design never mix with new ones.
``--fresh`` starts over by hand.
"""
import hashlib
import json
import time
from pathlib import Path

import yaml

DESIGNS = Path(__file__).resolve().parents[1] / "designs"


GLOBS = ["arigo1*.py"]          # ARIGO-2's runners add "arigo2*.py" (it is built on ARIGO-1's files)


def fingerprint() -> str:
    h = hashlib.sha256()
    for f in sorted({f for g in GLOBS for f in DESIGNS.glob(g)}):
        if f.name != "arigo1_drawings.py":
            h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()[:16]


def load(path: Path, fresh: bool) -> dict | None:
    """The previous run's document, or None (none, ``--fresh``, unreadable, or from another design)."""
    if fresh or not path.exists():
        return None
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        print(f"  {path.name} unreadable ({exc}); starting over")
        return None
    if doc.get("fingerprint") != fingerprint():
        stale = path.with_suffix(".stale.yaml")
        path.rename(stale)
        print(f"  the design/model changed since {path.name} was written: kept as {stale.name}, starting over")
        return None
    return doc


def save(path: Path, doc: dict):
    """Written to a temporary file and renamed: a kill mid-write never leaves a broken checkpoint."""
    doc["fingerprint"] = fingerprint()
    doc["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(json.loads(json.dumps(doc, default=float)), sort_keys=False, width=140))
    tmp.replace(path)


def eta(done, total, t0):
    """'[k/N, elapsed, about X left]' from the mean time per finished item."""
    el = time.time() - t0
    left = el / done * (total - done) if done else float("nan")
    f = lambda s: "?" if s != s else (f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min" if s >= 90 else f"{s:.0f} s")
    return f"[{done}/{total}, {f(el)} elapsed, ~{f(left)} left]"
