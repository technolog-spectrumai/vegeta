"""auto — the one face of Yanus that drives. A notebook flagged ``auto=True`` in the registry (its ``yanus_auto``
flag), or any gate entered with ``auto=True`` / ``YANUS_AUTO=1``, **runs by itself what it is missing**: the chapters
whose runs it takes (their prerequisite chapters first) and the notebooks whose files it takes — each in
**screening quality** (the kind's ``screening`` arguments: coarse meshes, few iterations) — and then opens with
whatever is there. An auto gate never raises ``GateClosed``: what could not be produced (a solver that is not on the
machine, a script that failed) is NOT RUN, said so in the table and in ``gate.auto``.

Everything runs in the foreground, one after the other, with its output on the terminal / in the notebook cell;
``YANUS_AUTO_MINUTES`` caps every chapter run (``--max-minutes``). Nested notebooks inherit ``YANUS_AUTO=1`` and so
produce their own inputs the same way.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .contract import Chapter, File, Flow, Notebook, Run, Unit
from .needs import check_need


def wanted(unit: Unit, auto: bool | None) -> bool:
    """Auto mode on? The argument, else ``YANUS_AUTO`` (1/0), else the notebook's flag."""
    if auto is not None:
        return bool(auto)
    env = os.environ.get("YANUS_AUTO", "").strip().lower()
    if env in ("1", "true", "on", "yes"):
        return True
    if env in ("0", "false", "off", "no"):
        return False
    return bool(getattr(unit, "auto", False))


@dataclass
class Step:
    """One thing auto mode runs (or cannot): a chapter for a variant, or a notebook."""
    unit: Unit
    variant: str | None = None
    because: str = ""                   # the take that asked for it
    argv: list = field(default_factory=list)
    state: str = "planned"              # planned | skipped | running | done | failed
    reason: str = ""
    returncode: int | None = None
    seconds: float = 0.0

    @property
    def label(self) -> str:
        if isinstance(self.unit, Chapter):
            return f"{self.unit.code} {self.unit.name}" + (f" --variant {self.variant}" if self.variant else "")
        return f"notebook {self.unit.name}"

    def as_dict(self) -> dict:
        return {"step": self.label, "because": self.because, "state": self.state, "reason": self.reason, "returncode": self.returncode,
                "seconds": round(self.seconds, 1), "command": " ".join(self.argv)}


# ================================================================================================= the plan
def chapter_argv(flow: Flow, ch: Chapter, variant: str | None, *, python: str | None = None) -> list[str]:
    k = ch.kind
    argv = [python or sys.executable, str(Path(flow.root) / k.script), "--chapter", ch.campaign]
    if variant:
        argv += ["--variant", variant]
    argv += list(k.screening)
    minutes = os.environ.get("YANUS_AUTO_MINUTES", "").strip()
    if minutes:
        argv += ["--max-minutes", minutes]
    return argv


def notebook_argv(flow: Flow, nb: Notebook, *, python: str | None = None, out_dir: str | None = None) -> list[str]:
    out = out_dir or tempfile.mkdtemp(prefix=f"yanus_auto_{nb.name}_")
    return [python or sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", str(Path(flow.root) / nb.path),
            "--output-dir", out, "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3"]


def missing_needs(unit: Unit) -> list[str]:
    return [n.label for n in unit.all_needs() if not check_need(n).ok]


def plan(flow: Flow, gate, *, python: str | None = None) -> list[Step]:
    """The steps that would produce the gate's unusable takes, in the order they must run: a chapter's prerequisite
    chapters before it, a producer notebook as one step (it produces its own inputs itself, in auto mode). A step
    whose needs are not on this machine is ``skipped`` with the reason; nothing is planned twice."""
    steps: list[Step] = []
    seen: set = set()

    def add_chapter(ch: Chapter, variant: str | None, because: str, depth: int = 0) -> None:
        key = ("ch", ch.ref, variant)
        if key in seen or depth > 12:
            return
        seen.add(key)
        for t in ch.takes:                                     # the prerequisites first (same book)
            if isinstance(t, Run):
                pre = flow.run_take_chapter(t, ch)
                add_chapter(pre, t.variant or variant, f"prerequisite of {ch.code}", depth + 1)
        st = Step(ch, variant, because, chapter_argv(flow, ch, variant, python=python))
        miss = missing_needs(ch)
        if miss:
            st.state, st.reason = "skipped", "not on this machine: " + ", ".join(miss)
        steps.append(st)

    def add_notebook(nb: Notebook, because: str) -> None:
        key = ("nb", nb.name)
        if key in seen:
            return
        seen.add(key)
        st = Step(nb, None, because, notebook_argv(flow, nb, python=python))
        miss = missing_needs(nb)
        if miss:
            st.state, st.reason = "skipped", "not on this machine: " + ", ".join(miss)
        steps.append(st)

    unit = gate.unit
    for r, t in zip(gate.take_rows, [t for t in unit.takes if isinstance(t, (Run, File))]):
        if r.usable:
            continue
        if isinstance(t, Run):
            add_chapter(flow.run_take_chapter(t, unit), t.variant or gate.variant, f"{unit.ref} takes {r.input}")
        elif isinstance(t, File) and t.producer:
            prod = flow.unit(t.producer)
            if isinstance(prod, Chapter):
                add_chapter(prod, gate.variant, f"{unit.ref} takes {t.path}")
            elif isinstance(prod, Notebook) and prod is not unit:
                add_notebook(prod, f"{unit.ref} takes {t.path}")
    # a prerequisite whose own run is already there is not re-run: the gate re-checks every take after the drive,
    # the chapter scripts refuse nothing (a usable run of a prerequisite makes it a no-op for the gate, not for the
    # script), so drop prerequisite steps that already have a usable run
    from .gate import candidates
    out = []
    for st in steps:
        if isinstance(st.unit, Chapter) and st.because.startswith("prerequisite of") and candidates(flow, st.unit, st.variant, deep=False):
            st.state, st.reason = "skipped", "a completed run is already there"
        out.append(st)
    return out


# ================================================================================================= the drive
def run_step(step: Step, flow: Flow, *, log=print, runner=None) -> Step:
    """Run one step in the foreground; ``runner(argv, env, cwd) -> returncode`` replaces ``subprocess`` in tests."""
    env = {**os.environ, "YANUS_AUTO": "1", "PYTHONUNBUFFERED": "1"}
    if isinstance(step.unit, Notebook) and step.unit.screening_env:
        env.setdefault(step.unit.screening_env, "1")
    step.state = "running"
    log(f"yanus auto: ▶ {step.label}  ({step.because})\n  $ {' '.join(step.argv)}")
    t0 = time.time()
    try:
        if runner is not None:
            rc = runner(step.argv, env, str(flow.root))
        else:
            rc = subprocess.call(step.argv, env=env, cwd=str(flow.root))
    except (OSError, KeyboardInterrupt) as exc:
        rc = 130 if isinstance(exc, KeyboardInterrupt) else 1
        step.reason = str(exc)
    step.seconds = time.time() - t0
    step.returncode = rc
    if rc == 0:
        step.state = "done"
    elif rc == 3:
        step.state, step.reason = "done", step.reason or "partial: some cases did not solve (usable as far as it goes)"
    else:
        step.state, step.reason = "failed", step.reason or {2: "the script's gate closed (a prerequisite or an option)", 4: "nothing solved", 130: "interrupted"}.get(rc, f"exit code {rc}")
    log(f"yanus auto: {'✔' if step.state == 'done' else '✘'} {step.label}: {step.state}" + (f" — {step.reason}" if step.reason else "") + f" ({step.seconds / 60:.1f} min)")
    return step


def drive(flow: Flow, gate, *, log=print, runner=None, python: str | None = None) -> list[Step]:
    steps = plan(flow, gate, python=python)
    todo = [s for s in steps if s.state == "planned"]
    if not steps:
        return steps
    log(f"yanus auto: {gate.unit.kind_of_unit} {gate.unit.ref} is missing {len(gate.take_rows) - sum(r.usable for r in gate.take_rows)} take(s); "
        f"{len(todo)} step(s) to run in screening quality" + (f", {len(steps) - len(todo)} cannot run here" if len(steps) != len(todo) else ""))
    for s in steps:
        if s.state == "skipped":
            log(f"yanus auto: · {s.label}: skipped — {s.reason}")
    for s in todo:
        run_step(s, flow, log=log, runner=runner)
    return steps


def report(steps: list[Step]) -> str:
    if not steps:
        return "yanus auto: nothing to produce — every take was usable"
    lines = ["yanus auto: what ran for this gate"]
    for s in steps:
        mark = {"done": "✔", "failed": "✘", "skipped": "·"}.get(s.state, "▶")
        lines.append(f"  {mark} {s.label:40s} {s.state}" + (f" — {s.reason}" if s.reason else "") + (f" ({s.seconds / 60:.1f} min)" if s.seconds else ""))
    return "\n".join(lines)
