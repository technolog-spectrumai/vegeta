"""select — find a saved workflow run and check it is usable. Reads runkit's files only; never starts anything.

    sel = yanus.find("K1", variant="4Q-2", fingerprint=FP, quality="fine")        # the newest usable run
    sel = yanus.find("K1", variant="4Q-2", run="workflows/runs/kestrel/airframe-sweep/<id>")   # this run or nothing
    display(sel); sel.require(); sel.summary; sel.provenance()

A run is usable when its ``manifest.json`` is readable (runkit schema), it belongs to the chapter's workflow and the
variant, ``status.json`` says ``completed`` and ``summary.json`` reads, the requested quality is the run's, it was not
made under the mocks (Mimesis's ``"mock": true``), and the book's compatibility check does not call it incompatible.
Without a fingerprint (or a check) the run is accepted as **unverified**, never as compatible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .contract import Chapter, Flow
from .registry import current
from .runs import RunView, read_run, runs_of

MOCK_QUALITIES = ("debug", "mock")


class RunUnusable(RuntimeError):
    """A required run is missing or unusable; the message says why and which command would produce it."""


def runs_root(flow: Flow) -> Path:
    return Path(os.environ.get("VEGETA_WORKFLOW_RUNS") or Path(flow.root) / "workflows" / "runs")


def _rel(flow: Flow, p) -> str:
    try:
        return str(Path(p).resolve().relative_to(Path(flow.root).resolve()))
    except ValueError:
        return str(p)


def command(ch: Chapter, variant: str | None, quality: str | None) -> str:
    """The ``wf`` command that would produce such a run (printed, never run here)."""
    return (f"wf {ch.code}" + (f" --variant {variant}" if variant else "")
            + (f" --quality {quality}" if quality and ch.kind.takes_option("--quality") else ""))


@dataclass
class Selection:
    """What ``find`` chose for one request, or why nothing is usable."""
    chapter: Chapter
    variant: str | None
    quality: tuple
    explicit: str | None
    run: RunView | None = None
    ok: bool = False
    reasons: list = field(default_factory=list)            # why the chosen (or explicit) run is refused, or why none was found
    compatibility: str = ""                                # compatible | warnings | unverified
    compat_reasons: list = field(default_factory=list)
    skipped: list = field(default_factory=list)            # (run id, reason) of the newer runs passed over
    root: str = ""
    command: str = ""

    @property
    def summary(self) -> dict | None:
        return self.run.summary if self.ok else None

    @property
    def manifest(self) -> dict | None:
        return self.run.manifest if self.ok else None

    @property
    def label(self) -> str:
        return f"{self.chapter.code} {self.chapter.name}" + (f", {self.variant}" if self.variant else "") + \
            (f", {'/'.join(self.quality)}" if self.quality else "")

    def require(self) -> dict:
        """The summary; ``RunUnusable`` with the reasons and the command when the run is not usable."""
        if self.ok:
            return self.run.summary
        raise RunUnusable(self.message())

    def message(self) -> str:
        what = f"the run {self.explicit}" if self.explicit else "no usable run"
        lines = [f"{self.label}: {what} — " + "; ".join(self.reasons or ["unusable"])]
        lines += [f"    skipped {rid}: {why}" for rid, why in self.skipped[:5]]
        lines.append(f"  produce one with:  {self.command}   (a workflow; nothing is run from here)")
        return "\n".join(lines)

    def provenance(self) -> dict:
        """What a report records about this input: the request, the run read (or why none), its state and checks."""
        m = (self.run.manifest or {}) if self.run else {}
        tv = m.get("tool_versions") or {}
        return {"request": {"chapter": self.chapter.ref, "code": self.chapter.code, "variant": self.variant,
                            "quality": list(self.quality) or None, "run": self.explicit, "mode": "explicit" if self.explicit else "latest"},
                "usable": self.ok, "run": _rel_root(self) if self.run else None, "run_id": self.run.run_id if self.run else None,
                "created_at": m.get("created_at"), "state": self.run.state if self.run else None,
                "quality": (m.get("config") or {}).get("quality"), "compatibility": self.compatibility or None,
                "compatibility_reasons": list(self.compat_reasons), "git": tv.get("git"), "host": m.get("host"),
                "reasons": list(self.reasons), "command": None if self.ok else self.command}

    def row(self) -> dict:
        r = self.run
        return {"input": self.label, "selection": ("explicit " if self.explicit else "latest ") + (r.run_id if r and r.manifest else "—"),
                "status": ("usable" if self.ok else "NOT USABLE: " + "; ".join(self.reasons)),
                "compatibility": self.compatibility + (": " + "; ".join(self.compat_reasons) if self.compat_reasons else ""),
                "provenance": r.provenance() if r and r.manifest else "", "command": "" if self.ok else self.command}

    def text(self) -> str:
        if self.ok:
            r = self.run
            return (f"✔ {self.label}: {'explicit' if self.explicit else 'latest'} run {r.run_id} — {_rel_root(self)}\n"
                    f"    {r.provenance()}; compatibility {self.compatibility}" + (f" ({'; '.join(self.compat_reasons)})" if self.compat_reasons else ""))
        return "✘ " + self.message()

    __str__ = text

    def _repr_html_(self) -> str:
        import html
        return f"<pre>{html.escape(self.text())}</pre>"


def _rel_root(sel: Selection) -> str:
    return sel.root or str(sel.run.root)


def check(rv: RunView, ch: Chapter, *, variant: str | None, quality: tuple, fingerprint: dict | None) -> tuple[bool, list, str, list]:
    """``(usable, reasons, compatibility, compatibility reasons)`` of one run directory for the request."""
    if rv.problem:
        return False, [rv.problem], "", []
    m = rv.manifest
    cfg = m.get("config") or {}
    if rv.workflow != ch.kind.workflow:
        return False, [f"a {rv.workflow} run, not {ch.kind.workflow}"], "", []
    if m.get("mock") is True or cfg.get("quality") in MOCK_QUALITIES:      # Mimesis's label: stub numbers
        return False, ["made under the mocks (vegeta.mock / Mimesis): stub numbers, not a result"], "", []
    if variant and rv.variant != variant:
        return False, [f"a run of {rv.variant}, not {variant}"], "", []
    if rv.state != "completed":
        hint = {"failed": "failed — read its run.log", "partial": "stopped by its budget — resume it (wf resume)",
                "cancelled": "cancelled — resume it (wf resume)", "running": "still running", "prepared": "prepared only, no solver ran"}
        return False, [f"run state '{rv.state}': " + ("its process is gone — resume it (wf resume)" if rv.extra.get("dead") else hint.get(rv.state, "not completed"))], "", []
    if rv.summary is None:
        return False, ["summary.json " + ("is not readable JSON" if (rv.root / "summary.json").exists() else "is missing")], "", []
    if quality:
        q = cfg.get("quality")
        if q is None:
            return False, [f"the run records no quality, {'/'.join(quality)} requested"], "", []
        if q not in quality:
            return False, [f"quality {q}, {'/'.join(quality)} requested"], "", []
    if fingerprint is None or ch.book.compatible is None:
        why = "no fingerprint given" if fingerprint is None else f"book {ch.book.name} has no compatibility check"
        return True, [], "unverified", [why]
    c = ch.book.compatible(fingerprint, m.get("design"), ch)
    level, why = c.get("level", ""), list(c.get("reasons") or [])
    if level == "incompatible":
        return False, ["incompatible: " + "; ".join(why)], level, why
    return True, [], level, why


def find(chapter, *, variant: str | None = None, fingerprint: dict | None = None, quality=None, run=None,
         flow: Flow | None = None, root=None) -> Selection:
    """The run a notebook reads for ``chapter`` (``"K1"``, ``"kestrel/airframe-sweep"``): ``run`` = that directory
    (checked; an unusable one is reported, never replaced), else the newest usable run under the runs folder. Never
    raises for a missing or unusable run (``Selection.ok`` / ``require()``) and never starts a computation. ``root`` =
    the runs folder (default ``$VEGETA_WORKFLOW_RUNS``, else ``workflows/runs``)."""
    flow = current(flow)
    ch = flow.chapter(chapter)
    qual = (quality,) if isinstance(quality, str) else tuple(quality or ())
    sel = Selection(ch, variant, qual, None if run is None else str(run), command=command(ch, variant, qual[0] if qual else None))
    base = Path(root) if root else runs_root(flow)
    if run is not None:
        p = Path(run).expanduser()
        if not p.is_absolute():
            p = next((c for c in (Path(flow.root) / p, Path.cwd() / p) if c.exists()), Path(flow.root) / p)
        sel.run = read_run(p)
        sel.root = _rel(flow, p)
        sel.ok, sel.reasons, sel.compatibility, sel.compat_reasons = check(sel.run, ch, variant=variant, quality=qual, fingerprint=fingerprint)
        imp = next((i for i in flow.imports.values() if i.marker and i.marker in p.name), None)
        if imp is not None and not sel.ok and sel.run.problem.startswith("not found"):
            sel.command = f"wf {imp.name}   ({imp.title})"         # the run came from an import script, not a chapter
        return sel
    if str(base).rstrip("/").endswith("-mock"):
        sel.reasons = [f"the runs folder is the mocks' ({base}): VEGETA_WORKFLOW_RUNS points at mock runs"]
        return sel
    for rv in runs_of(base, ch.book.name, ch.name, variant=variant):
        ok, why, compat, cwhy = check(rv, ch, variant=variant, quality=qual, fingerprint=fingerprint)
        if ok:
            sel.run, sel.ok, sel.compatibility, sel.compat_reasons, sel.root = rv, True, compat, cwhy, _rel(flow, rv.root)
            return sel
        sel.skipped.append((rv.run_id, "; ".join(why)))
    sel.reasons = [f"none of the {len(sel.skipped)} run(s) under {_rel(flow, base / ch.book.name / ch.name)} is usable" if sel.skipped
                   else f"no run under {_rel(flow, base / ch.book.name / ch.name)}" + (f" for {variant}" if variant else "")]
    return sel


def table(selections):
    """Several selections as one table (a pandas DataFrame when pandas is there, else text)."""
    sels = list(selections.values()) if isinstance(selections, dict) else list(selections)
    try:
        import pandas as pd
    except ImportError:
        return "\n".join(s.text() for s in sels)
    return pd.DataFrame([s.row() for s in sels]).set_index("input")
