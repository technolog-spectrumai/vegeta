"""The MuJoCo autopilot's tuned settings per course (``races/autopilot.json``, written by ``races.tune``)."""
from __future__ import annotations

import json
import time
from pathlib import Path

FILE = Path(__file__).resolve().parent / "autopilot.json"


def load() -> dict:
    return json.loads(FILE.read_text()) if FILE.exists() else {}


def tuned(course: str) -> dict:
    """The tuned keyword arguments of ``races.sim.run_mujoco`` for the course, or ``{}`` (its defaults)."""
    return dict((load().get(course) or {}).get("settings") or {})


def record(course: str, entry: dict) -> None:
    """Store a course's result (``{"settings": {...} | None, ...}``); a finished lap never gives way to a DNF. Under the
    same ``objective`` only a better score (else a faster nominal lap) replaces an entry; a result of another objective
    (the robust score against the nominal lap's) replaces it, because the two do not compare."""
    d = load()
    old = d.get(course) or {}
    if old.get("settings") and not entry.get("settings"):
        return
    nominal = "nominal: the nominal lap alone"
    same = old.get("objective", nominal) == entry.get("objective", nominal)        # another objective replaces: the scores do not compare
    rank = lambda e: e.get("score", e.get("lap time [s]", 1e9)) if "score" in old and "score" in entry else e.get("lap time [s]", 1e9)
    if old.get("settings") and entry.get("settings") and same and rank(old) <= rank(entry):
        return
    d[course] = {**entry, "date": time.strftime("%Y-%m-%d")}
    FILE.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
