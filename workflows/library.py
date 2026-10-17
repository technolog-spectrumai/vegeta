"""library — the books and chapters of Vegeta's workflows.

A **book** is the part or assembly that is simulated (the Kestrel airframe, the Kestrel frame…): one design module,
one set of variants, one results folder. A **chapter** is the scenario or problem solved on it (the bare sweep, the
stall angles, the cruise with the rotor disks, the frame at 4 g…): one named list of cases. Every run lands under
``workflows/runs/<book>/<chapter>/<run id>/`` and every cached case under ``workflows/cache/<book>/``. A chapter may
have **prerequisites — chapters of the same book only** — that must have a completed, compatible run for the same
variant before it is allowed to start (``--ignore-prerequisites`` overrides, and says so in the manifest).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import os as _os
RUNS = Path(_os.environ.get("VEGETA_WORKFLOW_RUNS") or ROOT / "workflows" / "runs")
CACHE = Path(_os.environ.get("VEGETA_WORKFLOW_CACHE") or ROOT / "workflows" / "cache")


def _kc():
    import sys
    p = str(ROOT / "notebooks" / "designs")
    if p not in sys.path:
        sys.path.insert(0, p)
    import kestrel_campaigns
    return kestrel_campaigns


BOOKS = {
    "kestrel_airframe": {
        "title": "Kestrel airframe — the whole aircraft with the propellers as rotor disks (Aeromant / OpenFOAM)",
        "part": "the airframe of Kestrel-4Q-1 / 4Q-2 / 4Q-3 (designs/kestrel.py), propellers as actuator disks",
        "script": "workflows/kestrel/airframe_cfd.py",
        "needs": "OpenFOAM (openfoam.com or .org); 4 cores recommended",
        "time": "screening case 10–20 min, fine case ~1 h",
        "variants": ("4Q-1", "4Q-2", "4Q-3"),
        "options": [("--variant", ("4Q-1", "4Q-2", "4Q-3"), "4Q-2"), ("--quality", ("screening", "fine"), "screening"), ("--processors", None, "4")],
        "chapters": {
            "sweep": {"prerequisites": ()},
            "sweep-fine": {"prerequisites": ()},
            "stall": {"prerequisites": ()},
            "cruise": {"prerequisites": ("sweep",)},
            "transitions": {"prerequisites": ("sweep",)},
            "blown": {"prerequisites": ("sweep",)},
            "all": {"prerequisites": ()},
            "sections": {"prerequisites": ()},
        },
        "describe": lambda: _kc().CFD_CAMPAIGNS,
    },
    "kestrel_frame": {
        "title": "Kestrel frame — the carbon truss and the printed spar–post joint (Talos: Gmsh + CalculiX)",
        "part": "the spars, posts and diagonals as solid tubes; the printed joint",
        "script": "workflows/kestrel/structure_fea.py",
        "needs": "CalculiX ccx on the PATH, Gmsh (python module)",
        "time": "a frame case a few minutes; the whole book under an hour on 4 threads",
        "variants": ("4Q-1", "4Q-2", "4Q-3"),
        "options": [("--variant", ("4Q-1", "4Q-2", "4Q-3"), "4Q-2"), ("--threads", None, "4")],
        "chapters": {"frame": {"prerequisites": ()}, "joint": {"prerequisites": ("frame",)}, "all": {"prerequisites": ()}},
        "describe": lambda: _kc().FEA_CAMPAIGNS,
    },
}
IMPORTS = {"import_legacy": {"title": "import the first CFD campaign's results file into kestrel_airframe / sweep (once)", "script": "workflows/kestrel/import_legacy.py"}}


def chapters(book: str) -> dict:
    """``{chapter: {"description", "prerequisites"}}``."""
    b = BOOKS[book]
    desc = b["describe"]()
    return {c: {"description": desc.get(c, ""), "prerequisites": tuple(b["chapters"][c]["prerequisites"])} for c in b["chapters"]}


def run_dir(book: str, chapter: str, label: str) -> Path:
    from runkit import stamp, slug
    return RUNS / book / chapter / f"{stamp()}_{slug(label)}"


def runs_of(book: str, chapter: str | None = None, *, variant: str | None = None) -> list[dict]:
    """The runs of a book (and chapter), newest first, with their variant and state."""
    from runkit import list_runs, read_json
    out = []
    base = RUNS / book
    if not base.is_dir():
        return out
    for ch in sorted(base.iterdir()):
        if not ch.is_dir() or (chapter and ch.name != chapter):
            continue
        for r in list_runs(ch):
            m = read_json(Path(r["run"]) / "manifest.json") or {}
            v = (m.get("config") or {}).get("variant")
            if variant and v != variant:
                continue
            out.append({**r, "book": book, "chapter": ch.name, "variant": v, "quality": (m.get("config") or {}).get("quality")})
    return sorted(out, key=lambda r: r.get("created_at") or "", reverse=True)


def latest_completed(book: str, chapter: str, *, variant: str, expected: dict | None = None) -> dict | None:
    """The newest completed run of the chapter for the variant whose fingerprint is compatible with ``expected``
    (when given) — what a prerequisite check and a notebook's default selection look for."""
    kc = _kc() if expected is not None else None
    for r in runs_of(book, chapter, variant=variant):
        if r["state"] != "completed" or not r["summary"]:
            continue
        if expected is not None:
            from runkit import read_json
            m = read_json(Path(r["run"]) / "manifest.json") or {}
            if kc.compare(expected, m.get("design"))["level"] == "incompatible":
                continue
        return r
    return None


def check_prerequisites(book: str, chapter: str, *, variant: str, expected: dict | None = None) -> list[str]:
    """The prerequisite chapters of ``chapter`` (same book) that have **no** completed compatible run for the
    variant — with the command that would make one. Empty when the chapter may start."""
    missing = []
    for pre in BOOKS[book]["chapters"][chapter]["prerequisites"]:
        if latest_completed(book, pre, variant=variant, expected=expected) is None:
            missing.append(f"chapter {pre!r} of book {book!r} has no completed run for {variant}: "
                           f".venv/bin/python {BOOKS[book]['script']} --variant {variant} --chapter {pre}")
    return missing


def all_runs() -> list[dict]:
    out = []
    for b in BOOKS:
        out += runs_of(b)
    return sorted(out, key=lambda r: r.get("created_at") or "", reverse=True)
