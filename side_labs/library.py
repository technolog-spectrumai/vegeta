"""library — the books and chapters of Vegeta's workflows.

A **book** is a design (a part or assembly we simulate): ``kestrel`` is the book of the Kestrel aircraft, code **K**.
Its **chapters** are the numbered studies on it — CFD studies, FEA studies, whatever the design needs — chosen by
code: ``K1`` is the first chapter of ``kestrel``. Every run lands under ``side_labs/runs/<book>/<chapter>/<run id>/``
and every cached case under ``side_labs/cache/<book>/``. A chapter may have **prerequisites — chapters of the same
book only** — that must have a completed, compatible run for the same variant before it starts
(``--ignore-prerequisites`` overrides, and says so in the manifest).

The books, chapters, kinds and imports are declared once, in ``side_labs/flow.py`` (Yanus's registry); ``KINDS``,
``BOOKS`` and ``IMPORTS`` here are generated from it in the shape ``wf`` and the GUI read.

    kestrel  (K)  — Kestrel-4Q biplane quadrotor tailsitter
      CFD, the airframe with the propellers as rotor disks (Aeromant)
        K1  airframe-sweep      the bare airframe 0–90°
        K2  airframe-stall      8–16°: CL max
        K3  airframe-cruise     cruise + stall point, rotor disks           [after K1]
        K4  airframe-transitions the transition points, rotor disks         [after K1]
        ...
        K8  sections            the two finalist sections as slabs
      FEA, the carbon frame and the printed joint (Talos)
        K9  frame               A, B, C, modes
        K10 joint                                                           [after K9]
        K11 frame-and-joint
"""
from __future__ import annotations

import os as _os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(_os.environ.get("VEGETA_WORKFLOW_RUNS") or ROOT / "side_labs" / "runs")
CACHE = Path(_os.environ.get("VEGETA_WORKFLOW_CACHE") or ROOT / "side_labs" / "cache")


def _kc():
    import sys
    p = str(ROOT / "designs")
    if p not in sys.path:
        sys.path.insert(0, p)
    import kestrel_campaigns
    return kestrel_campaigns


def _flow():
    """The one registry, ``side_labs/flow.py`` (Yanus): the dicts below are views of it, in the shape ``wf`` and the
    GUI read."""
    import sys
    try:
        from vegeta.yanus import load_flow
    except ImportError:
        sys.path.insert(0, str(ROOT / "vegeta-yanus" / "src"))
        from vegeta.yanus import load_flow
    return load_flow(ROOT / "side_labs" / "flow.py")


FLOW = _flow()
KINDS_BY_BOOK = {name: {k: kind.as_dict() for k, kind in b.kinds.items()} for name, b in FLOW.books.items()}
KINDS = dict(KINDS_BY_BOOK["kestrel"])                       # the first book's kinds (kept for the callers that knew one book); kind_of(book, kind) for the rest


def kind_of(book: str, kind: str) -> dict:
    return KINDS_BY_BOOK[book][kind]
BOOKS = {name: {"code": b.code, "title": b.title, "design": b.design, "variants": tuple(b.variants),
                "chapters": {ch.name: {"kind": ch.kind.name, "campaign": ch.campaign, "prerequisites": ch.prerequisites} for ch in b.chapters.values()},
                "describe": b.describe or (lambda: {})}
         for name, b in FLOW.books.items()}
IMPORTS = {name: imp.as_dict() for name, imp in FLOW.imports.items()}


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
                     "prerequisites": tuple(c["prerequisites"]), "script": KINDS_BY_BOOK[book][c["kind"]]["script"], "workflow": KINDS_BY_BOOK[book][c["kind"]]["workflow"]}
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
    raise KeyError(f"unknown chapter {code!r}: use a code such as K1 (side_labs/cli.py list)")


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
            lines.append(f"{'└── ' if last_kind else '├── '}{KINDS_BY_BOOK[bname][kind]['title']}")
            items = [(n, i) for n, i in chs.items() if i["kind"] == kind]
            for j, (name, info) in enumerate(items):
                branch = ("    " if last_kind else "│   ") + ("└── " if j == len(items) - 1 else "├── ")
                pre = "  [after " + ", ".join(chs[p]["code"] for p in info["prerequisites"]) + "]" if info["prerequisites"] else ""
                mark, rc = " ", ""
                if runs:
                    cs = chapter_status(bname, name)
                    mark = cs["mark"] if cs["runs"] else " "
                    rc = f"  [{cs['summary']}]" if cs["runs"] else ""
                lines.append(f"{branch}{mark} {info['code']:4s} {name:22s} {info['description']}{pre}{rc}")
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


STATE_MARK = {"completed": "✔", "running": "▶", "partial": "◐", "cancelled": "◐", "prepared": "○", "failed": "✘"}


def chapter_status(book: str, chapter: str) -> dict:
    """What happened to a chapter so far, from its runs: per variant the **latest** run's state (a chapter counts as
    done for a variant when its latest run completed), the counts of completed / running / failed / other runs, and a
    one-line summary with marks (✔ completed, ▶ running, ◐ partial or cancelled, ○ prepared, ✘ failed)."""
    from runkit import read_json
    rs = runs_of(book, chapter)
    latest: dict[str, dict] = {}
    for r in rs:                                             # newest first
        st = r["state"] or "?"
        if st == "running":
            pid = (read_json(Path(r["run"]) / "status.json") or {}).get("pid")
            try:
                import os
                os.kill(int(pid), 0)
            except (OSError, TypeError, ValueError):
                st = "failed"                                # the process is gone: it did not finish
            r = {**r, "state": st}
        latest.setdefault(r.get("variant") or "?", r)
    counts = {"completed": 0, "running": 0, "failed": 0, "other": 0}
    for r in rs:
        counts["completed" if r["state"] == "completed" else "running" if r["state"] == "running" else "failed" if r["state"] == "failed" else "other"] += 1
    parts = []
    for v, r in sorted(latest.items()):
        parts.append(f"{STATE_MARK.get(r['state'], '?')} {v} {r['state']}")
    overall = ("running" if any(r["state"] == "running" for r in latest.values()) else "completed" if latest and all(r["state"] == "completed" for r in latest.values())
               else "failed" if latest and any(r["state"] == "failed" for r in latest.values()) else "partial" if latest else "none")
    return {"runs": len(rs), "latest": latest, "counts": counts, "overall": overall, "mark": STATE_MARK.get(overall, "—") if latest else "—",
            "summary": " · ".join(parts) if parts else "not run"}


def all_runs() -> list[dict]:
    out = []
    for b in BOOKS:
        out += runs_of(b)
    return sorted(out, key=lambda r: r.get("created_at") or "", reverse=True)


# ----------------------------------------------------------------------------------------------- Yanus: needs and pins
def needs_status(book: str, chapter: str) -> dict:
    """The chapter's needs on this machine (Yanus): ``{"ok", "missing": [labels], "line": "✔ OpenFOAM v2406 · ✘ ccx"}``."""
    from vegeta.yanus import check_need
    ch = FLOW.books[book].chapters[chapter]
    res = [check_need(n) for n in ch.all_needs()]
    return {"ok": all(r.ok for r in res), "missing": [r.need.label for r in res if not r.ok], "fixes": [r.fix for r in res if not r.ok],
            "line": " · ".join(f"{'✔' if r.ok else '✘'} {r.need.label}" + (f" {r.version}" if r.ok and r.version and r.version != "found" else "") for r in res)}
