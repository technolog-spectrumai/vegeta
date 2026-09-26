"""The bounded prompt-to-3D loop: prompt -> plan -> generate -> run -> check -> render -> review -> revise.

::

    s = Session("a small stool with three legs", "_runs/fidia/stool", agent=demo_agent())
    s.run()                        # until done or a limit; one line per revision
    s.feedback("make the seat blue"); s.run()
    s.best.files()["glb"]          # the best valid revision's GLB

One iteration makes one revision (``rev-NNN/``, sealed by ``revision.json``). The model plans once (again only when
feedback asks to re-plan), then writes and revises code; the code runs in the sandbox; Vegeta's checks decide
validity; the review (a vision call on the contact sheet) scores it and is skipped when the build or a check
failed. A revision is **done** when it is valid, its export round-trip is clean, the reviewer accepts it with a
score >= ``accept_score``, every acceptance check passes and no user feedback is waiting.

It stops when done, or at a limit (``max_iterations`` and ``max_minutes`` per ``run()`` call, ``max_tokens`` over the
whole run directory, ``patience`` valid revisions without a new best), on ``cancel()``, a ``STOP`` file, a notebook
interrupt, a declined approval, a failed model call, or when the reviewer gives up. ``best/`` is always a copy of
the best **valid** revision: one that has seen the latest user feedback before an older one, then the higher
score, fewer warnings, the earlier revision; ``pin_best()`` overrides the choice.
``run.json`` (rewritten atomically) and ``events.jsonl`` (append-only) make a run resumable: ``Session.open(dir)``.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from vegeta.ai import Cancelled, ProviderError, Usage

from .agent import describe_agent
from .checks import check_export, check_parts, summarize
from .export import export_scene, reimport_check
from .mesh import bounds, load_parts
from .render import render_views
from .revision import Revision
from .sandbox import Sandbox, execute


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Limits:
    """Hard limits; the loop stops at the first one reached (between stages, never mid-write)."""

    max_iterations: int = 6             # revisions per run() call
    max_minutes: float | None = 20.0    # wall time per run() call
    max_tokens: int | None = 400_000    # input + output tokens over the whole run directory
    patience: int = 3                   # valid revisions without a new best before stopping
    accept_score: int = 8               # the reviewer's score needed for "done" (0-10)
    max_triangles: int = 200_000        # triangle budget of a model (a check)
    render_size: int = 512              # px per view; the contact sheet is 3 x 2 views

    def __post_init__(self):
        if self.max_iterations < 1 or self.patience < 1:
            raise ValueError("max_iterations and patience must be >= 1")
        if not 0 <= self.accept_score <= 10:
            raise ValueError("accept_score must be between 0 and 10")


class _Stop(Exception):
    """Ends the current revision early (cancel, decline, model failure); ``status`` goes into its record."""

    def __init__(self, reason: str, status: str, stage: str | None = None):
        super().__init__(reason)
        self.reason, self.status, self.stage = reason, status, stage


class Session:
    """A prompt-to-3D run in ``out_dir`` (created, or resumed if it holds a run of the same prompt)."""

    def __init__(self, prompt: str, out_dir: str | Path, *, agent, limits: Limits | None = None,
                 sandbox: Sandbox | None = None, approve: str | Callable[[dict], bool] = "auto",
                 on_iteration: Callable[[Revision], Any] | None = None, progress: bool = True):
        if not prompt or not prompt.strip():
            raise ValueError("the prompt is empty")
        if approve not in ("auto", "ask") and not callable(approve):
            raise ValueError("approve must be 'auto', 'ask' or a callable")
        self.prompt = prompt.strip()
        self.dir = Path(out_dir)
        self.agent = agent
        self.limits = limits or Limits()
        self.sandbox = sandbox or Sandbox()
        self.approve = approve
        self.on_iteration = on_iteration
        self.progress = progress
        self._cancel = threading.Event()
        self._stopped_by: str | None = None
        if (self.dir / "run.json").is_file():
            self.state = json.loads((self.dir / "run.json").read_text())
            if self.state["prompt"] != self.prompt:
                raise ValueError(f"{self.dir} holds a run for another prompt: {self.state['prompt']!r}")
            self.usage = Usage(**{k: self.state["usage"][k] for k in ("input_tokens", "output_tokens", "calls", "model")})
            self._event("resumed", revisions=len(self.state["revisions"]), tokens=self.usage.tokens)
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "prompt.txt").write_text(self.prompt + "\n")
            self.usage = Usage()
            self.state = {"prompt": self.prompt, "created_at": utc_now(), "status": "new", "stopped": None,
                          "done": False, "plan_version": 0, "replan": False, "feedback": [], "feedback_used": 0,
                          "revisions": [], "best": None, "pinned": None}
            self._event("created", prompt=self.prompt)
        self._save()

    @classmethod
    def open(cls, out_dir: str | Path, *, agent, limits: Limits | None = None, **kwargs) -> "Session":
        """Resume the run in ``out_dir`` (its prompt, and its limits unless new ones are given)."""
        state = json.loads((Path(out_dir) / "run.json").read_text())
        return cls(state["prompt"], out_dir, agent=agent, limits=limits or Limits(**state.get("limits", {})), **kwargs)

    # -- state -----------------------------------------------------------------------------------------------------
    def _save(self) -> None:
        self.state.update(updated_at=utc_now(), limits=asdict(self.limits), sandbox=self.sandbox.to_dict(),
                          agent=describe_agent(self.agent), usage=self.usage.to_dict())
        tmp = self.dir / "run.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=2, default=str))
        tmp.replace(self.dir / "run.json")

    def _event(self, event: str, **data) -> None:
        with open(self.dir / "events.jsonl", "a") as fh:
            fh.write(json.dumps({"event": event, "at": utc_now(), **data}, default=str) + "\n")

    def _say(self, text: str) -> None:
        if self.progress:
            print(text, flush=True)

    @property
    def plan(self) -> dict | None:
        f = self.dir / "plan.json"
        return json.loads(f.read_text()) if f.is_file() else None

    @property
    def revisions(self) -> list[Revision]:
        return [Revision(self.dir / r["name"]) for r in self.state["revisions"]]

    @property
    def best(self) -> Revision | None:
        """The best valid revision (its own directory; ``best/`` holds a copy)."""
        return Revision(self.dir / self.state["best"]) if self.state["best"] else None

    @property
    def best_dir(self) -> Path | None:
        return self.dir / "best" if (self.dir / "best").is_dir() else None

    @property
    def done(self) -> bool:
        return bool(self.state["done"]) and not self.state["feedback"]

    @property
    def stopped(self) -> str | None:
        return self.state["stopped"]

    @property
    def tokens(self) -> int:
        return self.usage.tokens

    # -- the user's controls -----------------------------------------------------------------------------------------
    def feedback(self, text: str, *, replan: bool = False) -> None:
        """Queue feedback for the next revision (``replan=True`` also re-plans first). Clears "done"."""
        if not text or not text.strip():
            raise ValueError("feedback is empty")
        self.state["feedback"].append({"text": text.strip(), "replan": bool(replan), "at": utc_now()})
        self.state["replan"] = self.state["replan"] or bool(replan)
        self._event("feedback", text=text.strip(), replan=replan)
        self._save()

    def cancel(self) -> None:
        """Stop as soon as possible: abandons a running model call or build (safe from another thread)."""
        self._cancel.set()

    def pin_best(self, revision: str | int | Revision | None) -> None:
        """Make ``revision`` (valid only) the best output whatever the scores; ``None`` unpins."""
        if revision is None:
            self.state["pinned"] = None
        else:
            name = revision.name if isinstance(revision, Revision) else (
                f"rev-{revision:03d}" if isinstance(revision, int) else str(revision))
            row = next((r for r in self.state["revisions"] if r["name"] == name), None)
            if row is None:
                raise ValueError(f"no revision {name}")
            if not row.get("valid"):
                raise ValueError(f"{name} is not valid; only valid revisions can be the best output")
            self.state["pinned"] = name
        self._event("pinned", revision=self.state["pinned"])
        self._update_best()
        self._save()

    # -- the loop --------------------------------------------------------------------------------------------------
    def _stop_reason(self, t0: float, made: int) -> str | None:
        lim = self.limits
        if (self.dir / "STOP").exists():
            return "STOP file found (delete it to continue)"
        if self._cancel.is_set():
            return "cancelled by the user"
        if self.done:
            return f"done: {self.state['best']} accepted"
        if made >= lim.max_iterations:
            return f"max_iterations ({lim.max_iterations}) reached"
        if lim.max_tokens is not None and self.tokens + self._reserve() > lim.max_tokens:
            return f"token budget: {self.tokens} of {lim.max_tokens} used, not enough left for another call"
        if lim.max_minutes is not None and (time.monotonic() - t0) / 60 >= lim.max_minutes:
            return f"time limit ({lim.max_minutes:g} min) reached"
        best = self.state["best"]
        if best:
            n_best = int(best.split("-")[1])
            since = sum(1 for r in self.state["revisions"] if r.get("valid") and r["number"] > n_best)
            if since >= lim.patience:
                return f"no better revision in {since} valid revisions (patience {lim.patience})"
        return None

    def run(self, progress: bool | None = None) -> "Session":
        """Iterate until done or stopped (see the module docstring). Returns ``self``."""
        if progress is not None:
            self.progress = progress
        t0, made = time.monotonic(), 0
        self._cancel.clear()
        self._stopped_by = None
        self.state.update(status="running", stopped=None)
        self._save()
        reason = None
        try:
            while True:
                reason = self._stop_reason(t0, made)
                if reason:
                    break
                rev = self.step()
                made += 1
                if self._stopped_by:
                    reason = self._stopped_by
                    break
                if self.on_iteration is not None:
                    answer = self.on_iteration(rev)
                    if answer is False:
                        reason = "stopped by on_iteration"
                        break
                    if isinstance(answer, str) and answer.strip():
                        self.feedback(answer)
        except KeyboardInterrupt:
            reason = "interrupted by the user (state saved)"
        self.state.update(status="done" if self.done else "stopped", stopped=reason)
        self._event("stopped", reason=reason, best=self.state["best"], tokens=self.tokens)
        self._save()
        cost = self.usage.cost_usd()
        self._say(f"stopped: {reason}. Best: {self.state['best'] or 'none'}; tokens {self.tokens}"
                  + (f" (~${cost:.2f})" if cost is not None else ""))
        return self

    def step(self) -> Revision:
        """Make one revision (plan first if needed). Never raises for a failed build; a cancelled, declined or
        failed model call seals the revision with that status and sets the stop reason for ``run()``."""
        self._stopped_by = None
        n = len(self.state["revisions"]) + 1
        rev = Revision(self.dir / f"rev-{n:03d}")
        if rev.path.exists():  # left by a crash before sealing: keep it aside, never reuse it
            aside = self.dir / "abandoned" / f"{rev.name}-{int(time.time())}"
            aside.parent.mkdir(exist_ok=True)
            rev.path.rename(aside)
        rev.path.mkdir(parents=True)
        feedback = list(self.state["feedback"])
        t0, used0 = time.monotonic(), self.usage
        rec: dict[str, Any] = {"number": n, "name": rev.name, "started_at": utc_now(), "status": "running", "valid": False,
                               "score": None, "verdict": None, "done": False, "summary": "", "error": None,
                               "problems": [], "fails": None, "warnings": None, "feedback": [f["text"] for f in feedback],
                               "plan_version": self.state["plan_version"], "cancelled_during": None}
        self._event("revision_started", revision=rev.name)
        try:
            self._revise(rev, rec, feedback)
        except _Stop as stop:
            rec.update(status=stop.status, error=str(stop), cancelled_during=stop.stage)
            self._stopped_by = stop.reason
        except KeyboardInterrupt:
            rec.update(status="cancelled", error="interrupted by the user", cancelled_during=rec.get("stage"))
            self._finish(rev, rec, t0, used0, feedback)
            raise
        self._finish(rev, rec, t0, used0, feedback)
        return rev

    def _finish(self, rev: Revision, rec: dict, t0: float, used0: Usage, feedback: list) -> None:
        rec.pop("stage", None)
        used = self.usage.tokens - used0.tokens
        rec.update(duration_s=round(time.monotonic() - t0, 2), tokens=used, finished_at=utc_now())
        if rec["status"] not in ("cancelled", "declined", "agent_error") or rec.get("generated"):
            # the feedback reached a generation: it is consumed; otherwise it waits for the next revision
            self.state["feedback"] = self.state["feedback"][len(feedback):]
            self.state["feedback_used"] = self.state.get("feedback_used", 0) + len(feedback)
        rec["epoch"] = self.state.get("feedback_used", 0)  # how much of the user's feedback this revision has seen
        rev.seal(rec)
        row = {k: rec[k] for k in ("number", "name", "status", "valid", "score", "verdict", "done", "summary",
                                   "fails", "warnings", "tokens", "duration_s", "epoch")}
        self.state["revisions"].append(row)
        self.state["done"] = bool(rec["done"])
        self._update_best()
        self._event("revision", **row)
        self._save()
        best = " *best*" if self.state["best"] == rev.name else ""
        score = f", score {rec['score']}/10 {rec['verdict']}" if rec["score"] is not None else ""
        warns = f", {rec['warnings']} warning(s)" if rec.get("warnings") else ""
        extra = f" — {rec['error']}" if rec.get("error") and not rec["valid"] else ""
        self._say(f"{rev.name}: {rec['status']}, {'valid' if rec['valid'] else 'invalid'}{score}{warns}{best}"
                  f" ({rec['duration_s']:.0f} s, {used} tokens) {rec['summary']}{extra}")

    def _reserve(self) -> int:
        return int(getattr(self.agent, "reserve", 0) or 0)

    def _call(self, stage: str, fn, *args, **kwargs):
        """One model call: budget and cancel checked first, usage counted, cancellation and failures turned into stops."""
        lim = self.limits
        if lim.max_tokens is not None and self.tokens + self._reserve() > lim.max_tokens:
            raise _Stop(f"token budget: {self.tokens} of {lim.max_tokens} used before {stage}", "cancelled", stage)
        self._check_cancel(stage)
        try:
            reply = fn(*args, cancel=self._cancel, **kwargs)
        except Cancelled:
            raise _Stop("cancelled by the user", "cancelled", stage) from None
        except ProviderError as exc:
            raise _Stop(f"the model call failed during {stage}: {exc}", "agent_error", stage) from None
        self.usage = self.usage + reply.usage
        self._event("call", stage=stage, usage=reply.usage.to_dict(), duration_s=round(reply.duration_s, 2))
        return reply.data

    def _check_cancel(self, stage: str) -> None:
        if self._cancel.is_set():
            raise _Stop("cancelled by the user", "cancelled", stage)
        if (self.dir / "STOP").exists():
            raise _Stop("STOP file found (delete it to continue)", "cancelled", stage)

    def _write(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, indent=2, default=str))

    def _revise(self, rev: Revision, rec: dict, feedback: list[dict]) -> None:
        texts = [f["text"] for f in feedback]
        # 1. plan (once; again when feedback asks for it)
        if self.plan is None or self.state["replan"]:
            rec["stage"] = "plan"
            replan_texts = [f["text"] for f in feedback if f["replan"]]
            plan = self._call("plan", self.agent.plan, self.prompt, feedback=replan_texts, previous=self.plan)
            self.state["plan_version"] += 1
            self.state["replan"] = False
            (self.dir / "plans").mkdir(exist_ok=True)
            self._write(self.dir / "plans" / f"plan-{self.state['plan_version']:03d}.json", plan)
            self._write(self.dir / "plan.json", plan)
            rec["plan_version"] = self.state["plan_version"]
            self._event("plan", version=self.state["plan_version"], object=plan.get("object"), parts=len(plan["parts"]))
            self._say(f"plan v{self.state['plan_version']}: {plan.get('object')} — "
                      f"{', '.join(p['name'] for p in plan['parts'])}; size {plan.get('size_mm')} mm")
        plan = self.plan
        # 2. generate
        rec["stage"] = "generate"
        gen = self._call("generate", self.agent.generate, self._brief(rec["number"], plan, texts))
        rec["generated"] = True
        rec["summary"] = gen["summary"]
        self._write(rev.path / "generation.json", gen)
        if not self._approved(rec["number"], gen):
            raise _Stop("declined by the approval policy", "declined", "approve")
        # 3. run in the sandbox
        rec["stage"] = "execute"
        self._check_cancel("execute")
        ex = execute(gen["source"], rev.path, sandbox=self.sandbox, cancel=self._cancel)
        rec["status"] = ex.status
        self._event("executed", revision=rev.name, status=ex.status, error=ex.error, duration_s=round(ex.duration_s, 2))
        if ex.status == "cancelled":
            raise _Stop("cancelled by the user", "cancelled", "execute")
        if not ex.ok:
            rec["error"] = ex.error
            return
        # 4. checks, renders, export + re-import
        rec["stage"] = "checks"
        parts = load_parts(rev.path)
        checks = check_parts(parts, plan, contacts=ex.contacts, max_triangles=self.limits.max_triangles)
        render = render_views(parts, rev.path / "renders", size=self.limits.render_size, title=plan.get("object", ""))
        export_scene(parts, rev.path / "export", step=rev.path / "model.step")
        report = reimport_check(rev.path / "export", parts)
        checks += check_export(report)
        summary = summarize(checks)
        self._write(rev.path / "checks.json", {"summary": summary, "checks": [c.to_dict() for c in checks],
                                                "render": render})
        rec.update(valid=summary["valid"], fails=summary["fails"], warnings=summary["warnings"],
                   problems=summary["problems"])
        if not summary["valid"]:
            return  # errors go straight back to the generator; no review for a broken model
        # 5. review (advice only)
        rec["stage"] = "review"
        lo, hi = bounds(parts)
        brief = {"prompt": self.prompt, "plan": plan, "revision": rec["number"], "feedback": texts,
                 "problems": summary["problems"], "measured": {"size_mm": [round(float(v), 1) for v in hi - lo],
                                                               "parts": [p.name for p in parts],
                                                               "triangles": int(sum(len(p.triangles) for p in parts))}}
        review = self._call("review", self.agent.review, brief, (rev.path / "renders" / "sheet.png").read_bytes())
        self._write(rev.path / "review.json", review)
        rec.update(score=review["score"], verdict=review["verdict"])
        accepted = all(a["ok"] for a in review["acceptance"])
        pending = len(self.state["feedback"]) > len(feedback)
        rec["done"] = (review["verdict"] == "accept" and review["score"] >= self.limits.accept_score and accepted
                       and report.get("ok", False) and not pending)
        if review["verdict"] == "give_up":
            self._stopped_by = f"the reviewer gave up: {review.get('summary', '')}"

    def _approved(self, n: int, gen: dict) -> bool:
        info = {"revision": n, "summary": gen["summary"], "addresses": gen["addresses"], "source": gen["source"]}
        if self.approve == "auto":
            return True
        if callable(self.approve):
            return bool(self.approve(info))
        print(f"\nrevision {n}: {gen['summary']}\n  addresses: {gen['addresses']}\n  "
              f"{len(gen['source'].splitlines())} lines of code (see rev-{n:03d}/generation.json)")
        return input("  run this code in the sandbox? [y/N] ").strip().lower() in ("y", "yes")

    def _brief(self, n: int, plan: dict, feedback: list[str]) -> dict:
        """What the modeller sees: the plan, feedback and the revision to build on (with what went wrong)."""
        rows = [r for r in self.state["revisions"] if (self.dir / r["name"] / "design.py").is_file()
                and r["status"] not in ("cancelled", "declined")]
        base, why = (rows[-1] if rows else None), ""
        best = next((r for r in self.state["revisions"] if r["name"] == self.state["best"]), None)
        if base and best and not base["valid"] and len(rows) >= 2 and not rows[-2]["valid"]:
            base, why = best, "the best valid revision; the last two attempts failed, so start again from here"
        previous = None
        if base:
            r = Revision(self.dir / base["name"])
            ex = r.execution or {}
            tb = ex.get("traceback") or ""
            previous = {"name": r.name, "why": why, "status": base["status"], "error": ex.get("error"),
                        "traceback": "\n".join(tb.splitlines()[-15:]), "problems": r.record.get("problems", []),
                        "review": r.review, "source": r.source}
        return {"prompt": self.prompt, "plan": plan, "revision": n, "feedback": feedback, "previous": previous,
                "best": {"name": best["name"], "score": best["score"]} if best else None}

    # -- best ------------------------------------------------------------------------------------------------------
    def _update_best(self) -> None:
        valid = [r for r in self.state["revisions"] if r.get("valid")]
        if self.state.get("pinned"):
            name = self.state["pinned"]
        elif valid:
            # a revision that has seen more of the user's feedback wins; then score, fewer warnings, earlier
            name = max(valid, key=lambda r: (r.get("epoch", 0), r["score"] if r["score"] is not None else -1,
                                             -(r["warnings"] or 0), -r["number"]))["name"]
        else:
            name = None
        if name == self.state["best"] and (name is None or (self.dir / "best").is_dir()):
            return
        self.state["best"] = name
        if name:
            tmp, old = self.dir / "best.tmp", self.dir / "best.old"
            for d in (tmp, old):
                if d.exists():
                    shutil.rmtree(d)
            shutil.copytree(self.dir / name, tmp)
            (tmp / "BEST").write_text(f"{name}\n")
            if (self.dir / "best").exists():
                (self.dir / "best").rename(old)
            tmp.rename(self.dir / "best")
            if old.exists():
                shutil.rmtree(old)
            self._event("best", revision=name)

    # -- views -----------------------------------------------------------------------------------------------------
    def table(self):
        """One row per revision (pandas DataFrame when pandas is installed, else a list of dicts)."""
        rows = [{**r, "best": r["name"] == self.state["best"]} for r in self.state["revisions"]]
        try:
            import pandas as pd

            cols = ["name", "status", "valid", "fails", "warnings", "score", "verdict", "done", "best", "epoch", "tokens",
                    "duration_s", "summary"]
            return pd.DataFrame(rows, columns=cols).set_index("name") if rows else pd.DataFrame(columns=cols)
        except ImportError:
            return rows

    def report(self) -> str:
        """A short text report of the run."""
        cost = self.usage.cost_usd()
        lines = [f"prompt: {self.prompt}", f"status: {self.state['status']} ({self.state['stopped'] or 'not stopped'})",
                 f"revisions: {len(self.state['revisions'])}, best: {self.state['best'] or 'none'}"
                 + (f" (pinned)" if self.state.get("pinned") else ""),
                 f"tokens: {self.tokens} in {self.usage.calls} calls" + (f", ~${cost:.2f}" if cost is not None else "")]
        if self.state["feedback"]:
            lines.append(f"feedback waiting: {[f['text'] for f in self.state['feedback']]}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<fidia.Session {self.dir}: {len(self.state['revisions'])} revisions, best {self.state['best']}, "
                f"{self.state['status']}>")
