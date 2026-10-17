"""gate — the two faces of the door: ``enter`` (needs and takes, before a unit starts) and ``Gate.leave`` (gives,
when it ends). A gate checks **everything** before it says anything, then either opens (``gate.inputs``) or raises
one ``GateClosed`` listing every missing required item with the command that produces it. It never produces
anything itself."""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import pins as _pins
from .contract import Chapter, Data, File, Flow, Notebook, Run, Unit
from .needs import check_need
from .registry import current
from .runs import RunView, read_run, runs_of
from .stamps import check_file, file_schema, missing_keys, read_stamp, sha256_file, write_stamp


class GateClosed(RuntimeError):
    """A required input is missing, unfinished, incompatible or stale — or a promised output was not written. The
    message lists every one with the command that fixes it."""

    def __init__(self, message: str, rows: list | None = None):
        super().__init__(message)
        self.rows = rows or []


@dataclass
class Row:
    input: str
    kind: str                       # need | run | file | give
    role: str                       # required | optional | need
    selection: str = "—"
    status: str = ""
    provenance: str = ""
    compatibility: str = ""
    usable: bool = False
    fix: str = ""
    key: str = ""

    def as_dict(self) -> dict:
        return {"input": self.input, "role": self.role, "selection": self.selection, "status": self.status, "provenance": self.provenance,
                "compatibility": self.compatibility, "usable": self.usable, "fix": self.fix}


def strict_pins() -> bool:
    """``YANUS_STRICT_PINS=1``: a notebook reads pinned runs only (an unpinned run take is NOT RUN / closes the gate)."""
    return os.environ.get("YANUS_STRICT_PINS", "").strip().lower() in ("1", "true", "on", "yes")


def runs_root(flow: Flow) -> Path:
    return Path(os.environ.get("VEGETA_WORKFLOW_RUNS") or Path(flow.root) / "workflows" / "runs")


def _rel(flow: Flow, p) -> str:
    try:
        return str(Path(p).resolve().relative_to(Path(flow.root).resolve()))
    except ValueError:
        return str(p)


def run_command(ch: Chapter, variant: str | None) -> str:
    return f"wf {ch.code}" + (f" --variant {variant}" if variant else "")


def notebook_command(nb) -> str:
    return f"scripts/run_notebooks.sh {nb.path.split('/', 1)[1]}"


# ================================================================================================= one run take
class _Fingerprints:
    def __init__(self, given: dict | None):
        self.given = dict(given or {})
        self.cache: dict = {}

    def get(self, book, variant):
        if variant in self.given:
            return self.given[variant]
        if book.fingerprint is None or variant is None:
            return None
        k = (book.name, variant)
        if k not in self.cache:
            self.cache[k] = book.fingerprint(variant)
        return self.cache[k]


def judge_run(rv: RunView, ch: Chapter, variant: str | None, *, states=("completed",), expected=None, deep: bool = True) -> tuple[bool, str, str]:
    """``(usable, status, compatibility)`` of one run directory for chapter ``ch``."""
    if rv.problem:
        return False, rv.problem, ""
    if ch.kind and rv.workflow and rv.workflow != ch.kind.workflow:
        return False, f"a {rv.workflow} run, not {ch.kind.workflow}", ""
    if variant and rv.variant and rv.variant != variant:
        return False, f"a run of {rv.variant}, not {variant}", ""
    if rv.state not in states or (rv.state == "completed" and rv.summary is None):
        hint = ("its process is gone — resume it with wf resume" if rv.extra.get("dead") else
                "still running or interrupted — resume it with wf resume" if rv.state in ("running", "prepared", "cancelled", "partial") else "failed — read run.log")
        return False, f"run state '{rv.state}' (no completed summary.json): {hint}", ""
    compat = ""
    if deep and expected is not None and ch.book.compatible is not None:
        c = ch.book.compatible(expected, (rv.manifest or {}).get("design"), ch)
        compat = c.get("level", "") + (": " + "; ".join(c.get("reasons") or []) if c.get("reasons") else "")
        if c.get("level") == "incompatible":
            return False, "incompatible: " + "; ".join(c.get("reasons") or []), compat
    elif not deep:
        compat = "not checked (--deep)"
    c = (rv.summary or {}).get("counts", {})
    st = f"{rv.state}" + (f": {c.get('solved', 0)} of {c.get('n', 0)} solved" if c else "")
    return True, st, compat


def candidates(flow: Flow, ch: Chapter, variant: str | None, *, states=("completed",), expected=None, deep: bool = True) -> list[RunView]:
    """The runs of the chapter that would be usable, newest first (for suggestions — never selected by themselves)."""
    out = []
    for rv in runs_of(runs_root(flow), ch.book.name, ch.name, variant=variant):
        if judge_run(rv, ch, variant, states=states, expected=expected, deep=deep)[0]:
            out.append(rv)
    return out


# ================================================================================================= the gate
@dataclass
class Gate:
    unit: Unit
    flow: Flow
    rows: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    selections: dict = field(default_factory=dict)
    started_ts: float = 0.0
    screening: bool = False
    variant: str | None = None
    left: list = field(default_factory=list)
    debug: bool = False

    @property
    def closed(self) -> list[Row]:
        return [r for r in self.rows if r.role in ("required", "need") and not r.usable]

    @property
    def ok(self) -> bool:
        return not self.closed

    def table(self):
        """The gate as a pandas DataFrame (index = input) when pandas is there, else the text."""
        try:
            import pandas as pd
        except ImportError:
            return self.text()
        return pd.DataFrame([r.as_dict() for r in self.rows]).set_index("input") if self.rows else pd.DataFrame()

    def text(self) -> str:
        head = f"{self.unit.kind_of_unit} {self.unit.ref}" + (" — screening: every take optional" if self.screening else "") + \
            (" — DEBUG: mocked numbers, outputs stamped as such" if self.debug else "")
        lines = [head]
        w = max([len(r.input) for r in self.rows] + [10])
        for r in self.rows:
            mark = "✔" if r.usable else ("✘" if r.role in ("required", "need") else "·")
            lines.append(f"  {mark} {r.input:{w}s}  {r.role:8s}  {r.status}" + (f"  [{r.compatibility}]" if r.compatibility and r.usable else ""))
            if not r.usable and r.fix:
                lines.append(f"    {'':{w}s}  → {r.fix}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.text()

    def _repr_html_(self) -> str:
        t = self.table()
        return t.to_html() if hasattr(t, "to_html") else "<pre>" + str(t) + "</pre>"

    def message(self) -> str:
        bad = self.closed
        lines = [f"{self.unit.kind_of_unit} {self.unit.ref}: the gate is closed — {len(bad)} required item(s) missing or unusable "
                 "(Yanus never produces them in their place):"]
        for r in bad:
            lines.append(f"  • {r.input}: {r.status}" + (f"\n      fix:  {r.fix}" if r.fix else ""))
        if isinstance(self.unit, Notebook) and self.unit.screening_env and not self.screening:
            lines.append(f"  For a screening pass without them: {self.unit.screening_env}=1 (the results are then labelled unvalidated).")
        return "\n".join(lines)

    # ---- the exit face
    def leave(self, *, design_fingerprint: dict | None = None) -> list[dict]:
        """Every file the unit gives must exist, be written during this execution and carry its keys; each then gets
        its stamp. Raises ``GateClosed`` listing every missing or old output — except an ``optional`` give, and in
        DEBUG (mocks) where a missing output is reported, not raised; what DEBUG writes is stamped ``debug``."""
        root = Path(self.flow.root)
        bad, stamps, rows = [], [], []
        needs = {r.input.split(" ", 1)[-1]: r.status for r in self.rows if r.kind == "need" and r.usable}
        inputs = []
        for r in self.rows:
            if r.kind == "run" and r.usable:
                inputs.append({"kind": "run", "key": r.key, "path": r.selection})
            elif r.kind in ("file", "data") and r.usable:
                f = root / r.selection
                st = read_stamp(f)
                inputs.append({"kind": "file", "key": r.key, "path": r.selection, "sha256": (st or {}).get("sha256") or sha256_file(f)})
        for g in self.unit.gives:
            f = root / g.path
            row = Row(f"gives {g.path}", "give", "required", selection=g.path)
            if not f.exists():
                row.status = "not written"
            elif f.stat().st_mtime < self.started_ts - 2:
                row.status = f"not written by this execution (last written {time.strftime('%Y-%m-%d %H:%M', time.localtime(f.stat().st_mtime))})"
            elif missing_keys(f, g.keys):
                row.status = "missing keys: " + ", ".join(missing_keys(f, g.keys))
            elif g.schema is not None and f.suffix == ".json" and file_schema(f) not in (None, g.schema):
                row.status = f"schema {file_schema(f)} in the file, {g.schema} declared"
            else:
                row.usable = True
                stamps.append(write_stamp(f, producer=self.unit.name, kind=self.unit.kind_of_unit, schema=g.schema, root=root, needs=needs,
                                          inputs=inputs, design_fingerprint=design_fingerprint, debug=self.debug))
                row.status = "written and stamped"
            if not row.usable:
                if g.optional or self.debug:
                    row.role = "optional"
                    row.status += " (optional output)" if g.optional else " (DEBUG: not required)"
                else:
                    row.fix = "the cell that writes it did not run or failed: read the notebook's output"
                    bad.append(row)
            rows.append(row)
        self.left = rows
        if bad:
            raise GateClosed(f"{self.unit.kind_of_unit} {self.unit.ref}: {len(bad)} promised output(s) missing:\n" +
                             "\n".join(f"  • {r.selection}: {r.status}" for r in bad), bad)
        return stamps


def enter(unit, *, flow: Flow | None = None, screening: bool | None = None, fingerprints: dict | None = None, variant: str | None = None,
          deep: bool = True, raise_closed: bool = True, needs_required: bool = True, display: bool = False, debug: bool | None = None) -> Gate:
    """Check the unit's needs and takes (all of them), then open the gate or raise ``GateClosed``.

    For a **notebook** each ``Run`` take reads the run **pinned** in ``notebooks/pins/<unit>.json``; with no pin, the
    newest usable run (finished, compatible), named as such in the table (``YANUS_STRICT_PINS=1``: pins only); for a **chapter** the newest usable run of each prerequisite chapter counts (the script's
    own check). ``fingerprints`` = ``{variant: fingerprint}`` already built (else the book builds them, only when
    ``deep``). ``screening`` (default: the notebook's ``screening_env`` = 1) makes every take optional. ``debug``
    (default: ``VEGETA_DEBUG`` on, as ``vegeta.mock`` reads it) = the numbers are mocked: inputs made in DEBUG are
    accepted, outputs are stamped as DEBUG and a missing one does not fail."""
    flow = current(flow)
    u = flow.unit(unit) if isinstance(unit, str) else unit
    if screening is None:
        screening = bool(isinstance(u, Notebook) and u.screening_env and os.environ.get(u.screening_env) == "1")
    if debug is None:
        debug = os.environ.get("VEGETA_DEBUG", "").strip().lower() in ("1", "true", "on", "yes")
    g = Gate(u, flow, started_ts=time.time(), screening=screening, variant=variant, debug=bool(debug))
    fps = _Fingerprints(fingerprints)
    root = Path(flow.root)
    pins = _pins.read_pins(root, u.name) if isinstance(u, Notebook) else {}

    for n in u.all_needs():
        r = check_need(n)
        g.rows.append(Row(f"{type(n).__name__.lower()} {n.label}", "need", "need" if needs_required else "optional", selection="this machine",
                          status=(r.version or "found") if r.ok else r.detail or "not found", usable=r.ok, fix=r.fix, key=f"need:{n.label}"))

    for t in u.takes:
        role = "required" if (t.required and not screening) else "optional"
        if isinstance(t, Run):
            auto = ""
            ch = flow.run_take_chapter(t, u)
            v = t.variant or variant
            key = t.key() if "/" in t.chapter else f"{ch.ref}" + (f"@{v}" if v else "")
            row = Row(f"{ch.code} {ch.name}" + (f", {v}" if v else ""), "run", role, key=key, fix=run_command(ch, v))
            expected = fps.get(ch.book, v) if deep else None
            if isinstance(u, Notebook):
                val = pins.get(key)
                if val is None:
                    cand = candidates(flow, ch, v, states=t.states, expected=expected, deep=deep)
                    if cand and not strict_pins():
                        rv = cand[0]                               # no pin: the newest usable run, said so in the table
                        row.selection = _rel(flow, rv.root)
                        auto = f"newest of {len(cand)} usable run(s), not pinned (yn pin to freeze it)"
                    else:
                        row.status = "not pinned" if key not in pins else "pinned to nothing"
                        if cand:
                            row.fix = f"yn pin {u.name} {ch.code}" + (f" --variant {v}" if v else "") + f" {_rel(flow, cand[0].root)}"
                            row.status += f" — {len(cand)} usable run(s) on disk, newest {cand[0].run_id} (YANUS_STRICT_PINS=1: pins only)"
                        g.inputs[key] = None
                        g.selections[key] = None
                        g.rows.append(row)
                        continue
                else:
                    path = _pins.resolve(root, val)
                    row.selection = val
                    rv = read_run(path)
                    if rv.problem.startswith("not found"):
                        for imp in flow.imports.values():
                            if imp.marker and imp.marker in path.name:
                                row.fix = f"wf {imp.name}   ({imp.title})"
            else:
                cand = candidates(flow, ch, v, states=t.states, expected=expected, deep=deep)
                if not cand:
                    row.status = f"no {'/'.join(t.states)} compatible run for {v or 'any variant'}"
                    g.inputs[key] = None
                    g.rows.append(row)
                    continue
                rv = cand[0]
                row.selection = _rel(flow, rv.root)
            ok, status, compat = judge_run(rv, ch, v, states=t.states, expected=expected, deep=deep)
            row.usable, row.status, row.compatibility = ok, status + (f" — {auto}" if auto else ""), compat
            row.provenance = rv.provenance() if rv.manifest else ""
            g.inputs[key] = rv.summary if ok else None
            g.selections[key] = rv.root if ok else None
        elif isinstance(t, File):
            key = t.key()
            row = Row(f"{t.path.rsplit('/', 1)[-1]} ({t.producer})", "file", role, selection=t.path, key=key)
            prod = flow.unit(t.producer)
            c = check_file(t.path, root=root, min_schema=t.min_schema, keys=t.keys, producer=prod.name, debug=g.debug)
            row.usable, row.status = c["ok"], c["status"]
            st = c["stamp"] or {}
            row.provenance = f"{st.get('producer')} {st.get('written_at', '')[:16]} (git {st.get('git')})" if st else ""
            row.fix = run_command(prod, None) if isinstance(prod, Chapter) else notebook_command(prod)
            g.inputs[key] = c["path"] if c["ok"] else None
        elif isinstance(t, Data):
            key = t.key()
            f = root / t.path
            role = "optional" if (t.cache or not t.required or screening) else "required"
            row = Row(f"{t.path.rsplit('/', 1)[-1]} (data)", "data", role, selection=t.path, key=key)
            if not f.exists():
                row.status = ("missing — a cache: computed on first use" + (f" ({t.note})" if t.note else "")) if t.cache else "missing"
                row.usable = False
                row.fix = "" if t.cache else (t.note or f"a committed input: restore it with git checkout -- {t.path}")
            elif missing_keys(f, t.keys):
                row.status = "missing keys: " + ", ".join(missing_keys(f, t.keys))
                row.fix = t.note or f"the file is incomplete: restore it with git checkout -- {t.path}"
            elif f.stat().st_size == 0:
                row.status = "empty file"
                row.fix = t.note or f"restore it with git checkout -- {t.path}"
            else:
                row.usable = True
                row.status = f"on file ({time.strftime('%Y-%m-%d %H:%M', time.localtime(f.stat().st_mtime))}, {f.stat().st_size / 1024:.0f} kB)" + (", a cache" if t.cache else "")
            if t.cache and f.exists() and not row.usable:
                row.role = "required"                          # a cache that is there but broken would fail its reader
            g.inputs[key] = f if row.usable else None
        else:
            continue
        g.rows.append(row)

    if display:
        _show(g)
    if raise_closed and not g.ok:
        raise GateClosed(g.message(), g.closed)
    return g


def _show(g: Gate) -> None:
    try:
        from IPython.display import display as _d
        _d(g.table())
    except ImportError:
        print(g.text())


# ================================================================================================= the script side
def check_chapter(chapter, *, variant: str | None, expected: dict | None = None, ignore: bool = False, flow: Flow | None = None,
                  exit_code: int = 2, out=None) -> dict:
    """The entry gate of a workflow script: its needs (reported; the script decides — a missing solver still allows
    ``--prepare-only``) and its prerequisite chapters (a usable run of each, for the variant). Exits ``exit_code``
    when a prerequisite is missing and not ``ignore``; returns ``{"missing": [...], "needs": {...}}`` for the
    manifest."""
    flow = current(flow)
    ch = flow.chapter(chapter) if isinstance(chapter, str) else chapter
    out = out or sys.stdout
    g = enter(ch, flow=flow, variant=variant, fingerprints={variant: expected} if expected is not None else None,
              deep=expected is not None, raise_closed=False, needs_required=False)
    missing = [f"{r.input} has no usable run: {r.status}:  {r.fix}" for r in g.rows if r.kind == "run" and not r.usable]
    needs = {r.input: (r.status if r.usable else "missing") for r in g.rows if r.kind == "need"}
    for r in g.rows:
        if r.kind == "need" and not r.usable:
            print(f"note: {r.input} not found on this machine ({r.fix})", file=out)
    if missing and not ignore:
        print(f"chapter {ch.code} ({ch.name}) of book {ch.book.name!r} needs a completed compatible run of its prerequisite chapter(s) first:", file=out)
        for m in missing:
            print("  - " + m, file=out)
        print("(or --ignore-prerequisites, recorded in the manifest)", file=out)
        sys.exit(exit_code)
    return {"missing": missing, "needs": needs}
