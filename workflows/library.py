"""library — the books and chapters of Vegeta's workflows.

A **book** is a design (a part or assembly we simulate): ``kestrel`` is the book of the Kestrel aircraft, code **K**.
Its **chapters** are the numbered studies on it — CFD studies, FEA studies, whatever the design needs — chosen by
code: ``K1`` is the first chapter of ``kestrel``. Every run lands under ``workflows/runs/<book>/<chapter>/<run id>/``
and every cached case under ``workflows/cache/<book>/``. A chapter may have **prerequisites — chapters of the same
book only** — that must have a completed, compatible run for the same variant before it starts
(``--ignore-prerequisites`` overrides, and says so in the manifest).

    kestrel  (K)  — Kestrel-4Q biplane quadrotor tailsitter
      CFD, the airframe with the propellers as rotor disks (Aeromant)
        K1  airframe-sweep      the bare airframe 0–90°
        K2  airframe-stall      8–16°: CL max                              [after K1? no]
        K3  airframe-cruise     cruise + stall point, rotor disks           [after K1]
        K4  airframe-transitions the transition points, rotor disks         [after K1]
        ...
      FEA, the carbon frame and the printed joint (Talos)
        K8  frame               A, B, C, modes
        K9  joint                                                           [after K8]
"""
from __future__ import annotations

import os as _os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(_os.environ.get("VEGETA_WORKFLOW_RUNS") or ROOT / "workflows" / "runs")
CACHE = Path(_os.environ.get("VEGETA_WORKFLOW_CACHE") or ROOT / "workflows" / "cache")


def _kc():
    import sys
    p = str(ROOT / "notebooks" / "designs")
    if p not in sys.path:
        sys.path.insert(0, p)
    import kestrel_campaigns
    return kestrel_campaigns


KINDS = {
    "cfd": {"title": "CFD — the airframe with the propellers as rotor disks (Aeromant / OpenFOAM)", "script": "workflows/kestrel/airframe_cfd.py", "workflow": "airframe_cfd",
            "needs": "OpenFOAM (openfoam.com or .org); 4 cores recommended", "time": "screening case 10–20 min, fine case ~1 h",
            "options": [("--variant", ("4Q-1", "4Q-2", "4Q-3"), "4Q-2"), ("--quality", ("screening", "fine"), "screening"), ("--processors", None, "4")]},
    "fea": {"title": "FEA — the carbon frame and the printed spar–post joint (Talos: Gmsh + CalculiX)", "script": "workflows/kestrel/structure_fea.py", "workflow": "structure_fea",
            "needs": "CalculiX ccx on the PATH, Gmsh (python module)", "time": "a frame case a few minutes; the whole frame under an hour on 4 threads",
            "options": [("--variant", ("4Q-1", "4Q-2", "4Q-3"), "4Q-2"), ("--threads", None, "4")]},
}

BOOKS = {
    "kestrel": {
        "code": "K",
        "title": "Kestrel-4Q — the zero-stagger biplane quadrotor tailsitter (variants 4Q-1, 4Q-2, 4Q-3)",
        "design": "notebooks/designs/kestrel.py and kestrel_campaigns.py; the notebook 43_kestrel_study reads the runs",
        "variants": ("4Q-1", "4Q-2", "4Q-3"),
        "chapters": {                                        # order = the chapter numbers K1, K2, …
            "airframe-sweep": {"kind": "cfd", "campaign": "sweep", "prerequisites": ()},
            "airframe-stall": {"kind": "cfd", "campaign": "stall", "prerequisites": ()},
            "airframe-cruise": {"kind": "cfd", "campaign": "cruise", "prerequisites": ("airframe-sweep",)},
            "airframe-transitions": {"kind": "cfd", "campaign": "transitions", "prerequisites": ("airframe-sweep",)},
            "airframe-blown": {"kind": "cfd", "campaign": "blown", "prerequisites": ("airframe-sweep",)},
            "airframe-sweep-fine": {"kind": "cfd", "campaign": "sweep-fine", "prerequisites": ()},
            "airframe-all": {"kind": "cfd", "campaign": "all", "prerequisites": ()},
            "sections": {"kind": "cfd", "campaign": "sections", "prerequisites": ()},
            "frame": {"kind": "fea", "campaign": "frame", "prerequisites": ()},
            "joint": {"kind": "fea", "campaign": "joint", "prerequisites": ("frame",)},
            "frame-and-joint": {"kind": "fea", "campaign": "all", "prerequisites": ()},
        },
        "describe": lambda: {**{k: v for k, v in _kc().CFD_CAMPAIGNS.items()}, **{f"fea:{k}": v for k, v in _kc().FEA_CAMPAIGNS.items()}},
    },
}
IMPORTS = {"import_legacy": {"title": "import the first CFD campaign's results file into kestrel / airframe-sweep (once)", "script": "workflows/kestrel/import_legacy.py"}}


# ----------------------------------------------------------------------------------------------- lookups
def chapters(book: str) -> dict:
    """``{chapter: {"code", "n", "kind", "campaign", "description", "prerequisites", "script"}}`` in chapter order."""
    b = BOOKS[book]
    try:
        desc = b["describe"]()
    except Exception:
        desc = {}
    out = {}
    for n, (name, c) in enumerate(b["chapters"].items(), 1):
        key = c["campaign"] if c["kind"] == "cfd" else f"fea:{c['campaign']}"
        out[name] = {"code": f"{b['code']}{n}", "n": n, "kind": c["kind"], "campaign": c["campaign"], "description": desc.get(key, ""),
                     "prerequisites": tuple(c["prerequisites"]), "script": KINDS[c["kind"]]["script"], "workflow": KINDS[c["kind"]]["workflow"]}
    return out


def by_code(code: str) -> tuple[str, str, dict]:
    """``"K2"`` → ``(book, chapter name, chapter info)``; also accepts ``"kestrel/airframe-stall"`` or ``"kestrel 2"``."""
    s = code.strip()
    m = re.fullmatch(r"([A-Za-z]+)\s*(\d+)", s)
    if m:
        letter, n = m.group(1).upper(), int(m.group(2))
        for book, b in BOOKS.items():
            if b["code"].upper() == letter or book.upper() == letter:
                for name, info in chapters(book).items():
                    if info["n"] == n:
                        return book, name, info
                raise KeyError(f"book {book!r} has chapters {b['code']}1–{b['code']}{len(b['chapters'])}, not {s}")
        raise KeyError(f"no book with code {letter!r} (books: {', '.join(f'{k} = {v[chr(99) + chr(111) + chr(100) + chr(101)]}' for k, v in BOOKS.items())})")
    if "/" in s:
        book, name = s.split("/", 1)
        if book in BOOKS and name in BOOKS[book]["chapters"]:
            return book, name, chapters(book)[name]
    raise KeyError(f"unknown chapter {code!r}: use a code such as K1 (workflows/cli.py list)")


def chapter_of(book: str, kind: str, campaign: str) -> str | None:
    """The chapter name a script's ``--chapter <campaign>`` of ``kind`` belongs to (the run folder)."""
    for name, c in BOOKS[book]["chapters"].items():
        if c["kind"] == kind and c["campaign"] == campaign:
            return name
    return None


def tree(book: str | None = None, *, runs: bool = True) -> str:
    """The books as a tree: book, then the kinds, then the numbered chapters with their prerequisites and run counts."""
    lines = []
    for bname, b in BOOKS.items():
        if book and bname != book:
            continue
        lines.append(f"{bname}  ({b['code']})  {b['title']}")
        chs = chapters(bname)
        kinds = []
        for info in chs.values():
            if info["kind"] not in kinds:
                kinds.append(info["kind"])
        for ki, kind in enumerate(kinds):
            last_kind = ki == len(kinds) - 1
            lines.append(f"{'└── ' if last_kind else '├── '}{KINDS[kind]['title']}")
            items = [(n, i) for n, i in chs.items() if i["kind"] == kind]
            for j, (name, info) in enumerate(items):
                branch = ("    " if last_kind else "│   ") + ("└── " if j == len(items) - 1 else "├── ")
                pre = "  [after " + ", ".join(chs[p]["code"] for p in info["prerequisites"]) + "]" if info["prerequisites"] else ""
                rc = ""
                if runs:
                    rs = runs_of(bname, name)
                    done = sum(1 for r in rs if r["state"] == "completed")
                    rc = f"  ({done} completed run{'s' if done != 1 else ''}{', ' + str(len(rs) - done) + ' other' if len(rs) - done else ''})" if rs else ""
                lines.append(f"{branch}{info['code']:4s} {name:22s} {info['description']}{pre}{rc}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- runs
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
    chs = chapters(book)
    for ch in sorted(base.iterdir()):
        if not ch.is_dir() or (chapter and ch.name != chapter):
            continue
        for r in list_runs(ch):
            m = read_json(Path(r["run"]) / "manifest.json") or {}
            v = (m.get("config") or {}).get("variant")
            if variant and v != variant:
                continue
            out.append({**r, "book": book, "chapter": ch.name, "code": chs.get(ch.name, {}).get("code", "?"), "variant": v, "quality": (m.get("config") or {}).get("quality")})
    return sorted(out, key=lambda r: r.get("created_at") or "", reverse=True)


def latest_completed(book: str, chapter: str, *, variant: str, expected: dict | None = None) -> dict | None:
    """The newest completed run of the chapter for the variant whose fingerprint is compatible with ``expected``."""
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
    """The prerequisite chapters (same book) that have **no** completed compatible run for the variant, each with the
    command that would make one. Empty when the chapter may start."""
    chs = chapters(book)
    missing = []
    for pre in BOOKS[book]["chapters"][chapter]["prerequisites"]:
        if latest_completed(book, pre, variant=variant, expected=expected) is None:
            info = chs[pre]
            missing.append(f"{info['code']} ({pre}) has no completed run for {variant}:  wf {info['code']} --variant {variant}   "
                           f"(= {info['script']} --variant {variant} --chapter {info['campaign']})")
    return missing


def all_runs() -> list[dict]:
    out = []
    for b in BOOKS:
        out += runs_of(b)
    return sorted(out, key=lambda r: r.get("created_at") or "", reverse=True)
