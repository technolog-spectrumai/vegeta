"""A controlled design campaign: the AI proposes parameter values, Vegeta builds and analyses them as
new revisions in a ``vegeta.core`` workspace, and the loop stops on explicit limits.

What the loop may do, and nothing else:
- change **parameter values** of the design (never its source, never loads, materials or analyses);
- only within the design's parameter ranges, optional tighter ``bounds`` and the ``free`` list;
- only after the ``approval`` policy says yes (``"ask"`` prompts the engineer; a callable is a
  written policy; ``"auto"`` is an explicit opt-in);
- only until a limit is hit: iterations, tokens, wall time, no progress (``patience``), or a file
  named ``STOP`` in the campaign directory.

Every step is appended to ``<workspace>/campaigns/<name>/events.jsonl`` and the state is rewritten to
``campaign.json``; every candidate is an ordinary immutable revision with its evaluations. The
campaign never labels revisions: choosing the design stays with the engineer.
"""
from __future__ import annotations

import json
import math
import operator
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .session import utc_now

_OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,60}$")

INSTRUCTION = (
    "You are running a design campaign. Propose the NEXT candidate as kind='parameters' with only the "
    "parameters you want to change (subset of the free parameters, within their ranges). Use the "
    "iteration table to reason about sensitivities: every row was built and analysed. Aim for a design "
    "that satisfies all criteria and is best on the objective; make one well-reasoned move per step. "
    "Do not change the source. If you believe the best feasible design has been found or no further "
    "useful move exists, answer kind='answer' and say why."
)


@dataclass(frozen=True)
class Criterion:
    """A requirement on one metric, e.g. ``Criterion("fea.static.safety_factor_yield", ">=", 2.0)``.

    Metric names: ``geometry.<measurement>`` (volume, surface_area, ...) and ``<kind>.<analysis>.<metric>``.
    """

    metric: str
    op: str
    value: float

    def __post_init__(self):
        if self.op not in _OPS:
            raise ValueError(f"criterion op must be one of {sorted(_OPS)}")

    def check(self, metrics: dict) -> bool | None:
        v = metrics.get(self.metric)
        return None if v is None else bool(_OPS[self.op](v, self.value))

    def __str__(self) -> str:
        return f"{self.metric} {self.op} {self.value:g}"


@dataclass(frozen=True)
class Objective:
    """What "better" means among feasible candidates: ``Objective("geometry.volume", "min")``."""

    metric: str
    sense: str = "min"

    def __post_init__(self):
        if self.sense not in ("min", "max"):
            raise ValueError("objective sense must be 'min' or 'max'")

    def better(self, a: float, b: float | None) -> bool:
        return b is None or (a < b if self.sense == "min" else a > b)

    def __str__(self) -> str:
        return f"{self.sense} {self.metric}"


@dataclass(frozen=True)
class Budget:
    """Hard limits. The loop stops at the first one reached (after finishing the running step)."""

    max_iterations: int = 6          # proposals asked for (the starting revision is not counted)
    max_tokens: int | None = 300_000  # input + output tokens reported by the proposer
    max_minutes: float | None = None  # wall time of run()
    patience: int = 3                # stop after this many proposals without a new best feasible design

    def __post_init__(self):
        if self.max_iterations < 1 or self.patience < 1:
            raise ValueError("max_iterations and patience must be >= 1")


@dataclass(frozen=True)
class Analysis:
    """One analysis run on every candidate, through the revision API.

    ``kind="fea"``: ``factory(rev) -> talos.StructuralModel``; ``kind="cfd"``: ``factory(rev, workdir)
    -> aeromant.CFDCase``; ``kind="print"``: ``settings`` and ``orientation`` (mellonia).
    """

    kind: str
    name: str
    factory: Callable | None = None
    settings: Any = None
    orientation: Any = None
    options: dict = field(default_factory=dict)   # passed to run_fea / run_cfd / run_print

    def __post_init__(self):
        if self.kind not in ("fea", "cfd", "print"):
            raise ValueError("analysis kind must be 'fea', 'cfd' or 'print'")
        if self.kind in ("fea", "cfd") and self.factory is None:
            raise ValueError(f"{self.kind} analysis {self.name!r} needs a factory")
        if self.kind == "print" and (self.settings is None or self.orientation is None):
            raise ValueError(f"print analysis {self.name!r} needs settings and orientation")

    def run(self, rev):
        if self.kind == "fea":
            return rev.run_fea(self.name, self.factory, **self.options)
        if self.kind == "cfd":
            return rev.run_cfd(self.name, self.factory, **self.options)
        return rev.run_print(self.name, self.settings, self.orientation, **self.options)


def _scalars(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}.{k}": float(v) for k, v in (metrics or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}


def _design_source(spec: str, dd) -> str:
    target = spec.partition(":")[0]
    if target.endswith(".py"):
        return Path(target).read_text()
    import inspect

    try:
        return inspect.getsource(inspect.getmodule(type(dd)))
    except (OSError, TypeError):  # pragma: no cover
        return f"(source of {spec} not available)"


class Campaign:
    """Propose → approve → branch → generate → analyse → record, until a limit is reached.

    ``start`` is an existing revision (the baseline); ``proposer`` is any ``vegeta.ai.Proposer``
    (``ClaudeProposer`` or your own). ``run()`` is the only call that does work; it can be called again
    on the same workspace and name to continue a stopped campaign with a new budget.
    """

    def __init__(self, workspace, start, *, analyses: Sequence[Analysis], criteria: Sequence[Criterion],
                 objective: Objective, proposer, budget: Budget | None = None,
                 approval: str | Callable[[dict], bool] = "ask", free: Sequence[str] | None = None,
                 bounds: dict[str, tuple[float, float]] | None = None, goal: str = "", name: str = "campaign"):
        if not _NAME.match(name):
            raise ValueError("campaign name: letters, digits, '_', '-', '.' only")
        if approval not in ("ask", "auto") and not callable(approval):
            raise ValueError("approval must be 'ask', 'auto' or a callable(proposal) -> bool")
        self.workspace, self.start = workspace, start
        self.analyses, self.criteria, self.objective = list(analyses), list(criteria), objective
        self.proposer, self.budget, self.approval = proposer, budget or Budget(), approval
        self.bounds = dict(bounds or {})
        self.goal, self.name = goal, name
        self.dir = Path(workspace.root) / "campaigns" / name
        dd = start.design.load()
        names = [p.name for p in dd.params]
        self.free = list(free) if free is not None else [p.name for p in dd.params if p.type in (int, float)]
        unknown = [n for n in list(self.free) + list(self.bounds) if n not in names]
        if unknown:
            raise ValueError(f"unknown parameters {unknown}; the design has {names}")
        self.iterations: list[dict] = []
        self.tokens = 0
        self.stopped: str | None = None
        self.history: list[dict] = []
        self._load()

    # -- persistence ------------------------------------------------------------------------
    def _load(self):
        state = self.dir / "campaign.json"
        if state.is_file():
            rec = json.loads(state.read_text())
            if rec["start"] != self.start.id:
                raise ValueError(f"campaign {self.name!r} exists with start {rec['start']}, not {self.start.id}")
            self.iterations, self.tokens, self.history = rec["iterations"], rec["tokens"], rec.get("history", [])

    def _save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        rec = {"name": self.name, "start": self.start.id, "design": self.start.design_name, "goal": self.goal,
               "criteria": [asdict(c) for c in self.criteria], "objective": asdict(self.objective),
               "budget": asdict(self.budget), "free": self.free, "bounds": self.bounds,
               "approval": self.approval if isinstance(self.approval, str) else getattr(self.approval, "__name__", "callable"),
               "proposer": self.proposer.describe(), "analyses": [{"kind": a.kind, "name": a.name} for a in self.analyses],
               "tokens": self.tokens, "stopped": self.stopped, "best": self.best_id,
               "iterations": self.iterations, "history": self.history[-20:], "updated_at": utc_now()}
        tmp = self.dir / "campaign.json.tmp"
        tmp.write_text(json.dumps(rec, indent=2, default=str))
        tmp.replace(self.dir / "campaign.json")

    def _event(self, event: str, **data):
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "events.jsonl", "a") as fh:
            fh.write(json.dumps({"event": event, "at": utc_now(), **data}, default=str) + "\n")

    # -- evaluation -------------------------------------------------------------------------
    def evaluate(self, rev) -> tuple[dict, list[str]]:
        """Generate (if needed) and run every analysis not yet recorded on ``rev``; return its metrics."""
        messages: list[str] = []
        if not rev.is_generated:
            g = rev.generate()
            if not g.ok:
                return {}, [f"generate failed: {m}" for m in g.messages]
        metrics = _scalars("geometry", (rev.geometry_summary() or {}).get("metrics", {}))
        for a in self.analyses:
            ev = rev.evaluation(a.kind, a.name) or a.run(rev)
            if not ev.ok:
                messages += [f"{a.kind}.{a.name} {ev.status}: {m}" for m in ev.messages[:3]]
            metrics.update(_scalars(f"{a.kind}.{a.name}", ev.metrics))
        return metrics, messages

    def assess(self, metrics: dict) -> dict:
        checks = {str(c): c.check(metrics) for c in self.criteria}
        feasible = all(v is True for v in checks.values())
        return {"checks": checks, "feasible": feasible, "objective": metrics.get(self.objective.metric)}

    # -- views --------------------------------------------------------------------------------
    @property
    def best_id(self) -> str | None:
        best = None
        for it in self.iterations:
            if it.get("feasible") and it.get("objective") is not None and (
                    best is None or self.objective.better(it["objective"], best["objective"])):
                best = it
        return best["revision"] if best else None

    @property
    def best(self):
        return self.workspace.revision(self.best_id) if self.best_id else None

    def _incumbent(self) -> dict:
        """The row the next proposal starts from: the best feasible one, else the latest evaluated one."""
        bid = self.best_id
        rows = [it for it in self.iterations if it.get("revision")]
        return next(it for it in rows if it["revision"] == bid) if bid else rows[-1]

    def table(self):
        """One row per step (pandas DataFrame when pandas is installed, else a list of dicts)."""
        rows = []
        for it in self.iterations:
            row = {"step": it["step"], "revision": it.get("revision"), "status": it["status"],
                   "feasible": it.get("feasible"), self.objective.metric: it.get("objective")}
            row.update({k: v for k, v in (it.get("changes") or {}).items()})
            row.update({str(c): (it.get("metrics") or {}).get(c.metric) for c in self.criteria})
            row["summary"] = it.get("summary", "")
            rows.append(row)
        try:
            import pandas as pd

            return pd.DataFrame(rows).set_index("step")
        except ImportError:  # pragma: no cover
            return rows

    def plot(self):
        """Objective and criterion metrics per step; feasible candidates filled, the best starred."""
        import matplotlib.pyplot as plt

        rows = [it for it in self.iterations if it.get("metrics")]
        n = 1 + len(self.criteria)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False)
        steps = [it["step"] for it in rows]
        for ax, (metric, crit) in zip(axes[0], [(self.objective.metric, None)] + [(c.metric, c) for c in self.criteria]):
            ys = [it["metrics"].get(metric, float("nan")) for it in rows]
            ax.plot(steps, ys, color="#8a96a3", lw=1, zorder=1)
            for s, y, it in zip(steps, ys, rows):
                ax.scatter([s], [y], s=60, zorder=2, color="#2e7d32" if it["feasible"] else "white",
                           edgecolors="#2e7d32" if it["feasible"] else "#c62828", linewidths=1.5)
                if it["revision"] == self.best_id:
                    ax.scatter([s], [y], marker="*", s=260, color="#f9a825", zorder=3)
            if crit is not None:
                ax.axhline(crit.value, color="#c62828", ls="--", lw=1)
                ax.set_title(str(crit), fontsize=10)
            else:
                ax.set_title(f"objective: {self.objective}", fontsize=10)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
        fig.suptitle(f"campaign {self.name}: filled = meets all criteria, ★ = best")
        fig.tight_layout()
        return fig

    # -- the loop -----------------------------------------------------------------------------
    def context(self) -> dict:
        """What the proposer sees: the design (source, ranges, incumbent values) plus the campaign."""
        inc = self._incumbent()
        rev = self.workspace.revision(inc["revision"])
        dd = rev.design.load()
        values = dd.resolve(**rev.params)
        params = []
        for p in dd.params:
            lo, hi = self.bounds.get(p.name, (p.min, p.max))
            params.append({"name": p.name, "value": values[p.name], "units": p.units, "min": lo, "max": hi,
                           "description": (p.description + ("" if p.name in self.free else " [FIXED: do not change]")).strip()})
        lines = [f"Goal: {self.goal}" if self.goal else "",
                 "Criteria (all must hold): " + "; ".join(str(c) for c in self.criteria),
                 f"Objective among feasible designs: {self.objective}",
                 f"Free parameters: {', '.join(self.free)}",
                 f"Incumbent (the values shown above): {inc['revision']} "
                 f"({'feasible' if inc.get('feasible') else 'NOT feasible'})",
                 "Iterations so far (every row was built and analysed by Vegeta):"]
        keys = [self.objective.metric] + [c.metric for c in self.criteria if c.metric != self.objective.metric]
        for it in self.iterations:
            m = it.get("metrics") or {}
            vals = ", ".join(f"{k}={m[k]:.4g}" for k in keys if k in m)
            prm = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in (it.get("parameters") or {}).items()
                            if k in self.free)
            lines.append(f"- step {it['step']} {it.get('revision') or '-'} [{it['status']}"
                         f"{', feasible' if it.get('feasible') else ''}] {prm} -> {vals or it.get('reason', '')}")
        spec = rev.design.resolved_source()
        return {"path": spec, "source": _design_source(spec, dd),
                "parameters": params, "measurements": {k: v for k, v in inc.get("metrics", {}).items() if k.startswith("geometry.")},
                "notes": "\n".join(l for l in lines if l),
                "campaign": {"criteria": [asdict(c) for c in self.criteria], "objective": asdict(self.objective),
                             "free": self.free, "bounds": self.bounds, "iterations": self.iterations,
                             "incumbent": inc["revision"]}}

    def _check_proposal(self, data: dict, base_params: dict, dd) -> tuple[dict, str | None]:
        if data.get("kind") != "parameters":
            return {}, f"kind={data.get('kind')!r}: a campaign only accepts parameter changes"
        changes = dict(data.get("parameters") or {})
        if not changes:
            return {}, "no parameters proposed"
        fixed = [k for k in changes if k not in self.free]
        if fixed:
            return {}, f"parameters {fixed} are not free in this campaign (free: {self.free})"
        for k, v in changes.items():
            if k in self.bounds:
                lo, hi = self.bounds[k]
                if not (lo <= v <= hi):
                    return {}, f"{k}={v} is outside the campaign bounds [{lo}, {hi}]"
        try:
            values = dd.resolve(**{**base_params, **changes})
        except ValueError as exc:
            return {}, str(exc)
        for it in self.iterations:
            if it.get("parameters") == values:
                return {}, f"these values were already evaluated in step {it['step']} ({it.get('revision')})"
        return values, None

    def _approve(self, proposal: dict) -> bool:
        if self.approval == "auto":
            return True
        if callable(self.approval):
            return bool(self.approval(proposal))
        print(f"\n[{self.name}] step {proposal['step']}: {proposal['summary']}\n  changes: {proposal['changes']}"
              f"\n  rationale: {proposal['rationale']}")
        return input("  run this candidate? [y/N] ").strip().lower() in ("y", "yes")

    def _stop_reason(self, t0: float, since_best: int, asked: int) -> str | None:
        b = self.budget
        if (self.dir / "STOP").exists():
            return "STOP file found"
        if asked >= b.max_iterations:
            return f"max_iterations ({b.max_iterations}) reached"
        if b.max_tokens is not None and self.tokens >= b.max_tokens:
            return f"token budget ({b.max_tokens}) used: {self.tokens}"
        if b.max_minutes is not None and (time.monotonic() - t0) / 60 >= b.max_minutes:
            return f"time budget ({b.max_minutes} min) used"
        if since_best >= b.patience:
            return f"no new best feasible design in {b.patience} proposals"
        return None

    def run(self, progress: bool = True) -> "Campaign":
        """Run the loop until a limit is reached. Returns ``self`` (see ``table()``, ``best``, ``stopped``)."""
        t0 = time.monotonic()
        say = print if progress else (lambda *a, **k: None)
        self.stopped = None
        if not self.iterations:
            say(f"[{self.name}] step 0: evaluating the start revision {self.start.id}")
            metrics, msgs = self.evaluate(self.start)
            self._append({"step": 0, "revision": self.start.id, "status": "evaluated", "summary": "start",
                          "parameters": self.start.design.load().resolve(**self.start.params), "changes": {},
                          "metrics": metrics, "messages": msgs, **self.assess(metrics)})
        asked, since_best = 0, 0
        while True:
            reason = self._stop_reason(t0, since_best, asked)
            if reason:
                break
            step = self.iterations[-1]["step"] + 1
            ctx = self.context()
            inc = self._incumbent()
            base = self.workspace.revision(inc["revision"])
            data, usage = self.proposer.propose(ctx, INSTRUCTION, self.history)
            asked += 1
            self.tokens += int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
            self.history.append({"user": f"(campaign step {step})", "assistant": json.dumps(data)})
            row = {"step": step, "revision": None, "summary": data.get("summary", ""), "rationale": data.get("rationale", ""),
                   "base": base.id, "usage": usage}
            if data.get("kind") == "answer":
                self._append({**row, "status": "proposer_stopped", "reason": data.get("rationale", "")})
                reason = f"the proposer ended the campaign: {data.get('summary', '')}"
                break
            values, problem = self._check_proposal(data, base.params, base.design.load())
            row["changes"] = dict(data.get("parameters") or {})
            if problem:
                say(f"[{self.name}] step {step}: refused — {problem}")
                self.history.append({"user": f"(Vegeta refused step {step}: {problem})", "assistant": '{"kind": "answer"}'})
                self._append({**row, "status": "refused", "reason": problem})
                since_best += 1
                continue
            if not self._approve({**row, "values": values}):
                self._append({**row, "status": "declined", "reason": "declined by the approval policy"})
                reason = "declined by the approval policy (stop means stop)"
                break
            changed = {k: v for k, v in values.items() if base.params.get(k, None) != v and k in row["changes"]}
            rev = base.branch(note=f"campaign {self.name} step {step}: {row['summary']}"[:200], **changed)
            say(f"[{self.name}] step {step}: {rev.id} {changed} — {row['summary']}")
            metrics, msgs = self.evaluate(rev)
            prev_best = self.best_id
            self._append({**row, "revision": rev.id, "status": "evaluated", "parameters": values,
                          "metrics": metrics, "messages": msgs, **self.assess(metrics)})
            if msgs:
                self.history.append({"user": f"(analysis messages for step {step}: {msgs[:3]})", "assistant": '{"kind": "answer"}'})
            since_best = 0 if self.best_id != prev_best else since_best + 1
        self.stopped = reason
        self._event("stopped", reason=reason, best=self.best_id, tokens=self.tokens)
        self._save()
        say(f"[{self.name}] stopped: {reason}. Best feasible: {self.best_id or 'none'}; tokens used: {self.tokens}")
        return self

    def _append(self, row: dict):
        self.iterations.append(row)
        self._event("step", **row)
        self._save()

    def __repr__(self) -> str:
        return (f"<Campaign {self.name}: {len(self.iterations)} steps, best {self.best_id}, "
                f"stopped={self.stopped!r}>")
