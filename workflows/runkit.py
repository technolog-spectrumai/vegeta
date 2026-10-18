"""runkit — the run-directory contract of Vegeta's long-running analyses (the ``workflows/`` scripts).

A workflow is a thin script that prepares explicit inputs, runs the selected solver cases, gathers the artifacts,
parses the results and saves them. This module gives every such script the same **file-based** bookkeeping, nothing
more — no scheduler, no service, no database, no DAG:

``<run directory>/``
    ``manifest.json``   schema version, run id, timestamps, the command, the configuration and its hash, the input
                        files and their hashes, the design identity, tool versions, the artifact paths
    ``status.json``     ``prepared`` / ``running`` / ``completed`` / ``failed`` / ``cancelled``, updated atomically, with
                        the case counts, the case in progress and the last heartbeat (process alive, solver log advancing)
    ``run.log``         timestamped, continuously flushed: every case and stage start and finish, the solver commands
                        the tools report, PIDs, elapsed times, the heartbeats
    ``summary.json``    written **only when the run completed** its list of cases (atomically); a prepared, partial
                        (budget), failed or cancelled run leaves ``summary.partial.json`` instead, so it stays
                        inspectable but never reads as a completed, usable result
    ``cases/<name>/``   the native solver case (OpenFOAM directory, Gmsh mesh, CalculiX files, logs) and ``case.json``
                        (the case's status, key, timings, the tool's result as JSON)

Numerical convergence is recorded separately from the process's exit: a solver that ran to the end without reaching
its residual target is ``process: ok``, ``converged: False``; the engineering module decides the status (``solved``,
``diverged``, ``FAILED``) and the summary carries both. A heartbeat says the process is alive and whether the solver's
log grew since the last one; it never invents a percentage and never kills a quiet solver.

Interruption: ``Run.signals()`` turns SIGINT / SIGTERM into the ``cancel`` event the Vegeta tools accept (they terminate
their child process); the run ends ``cancelled`` with its partial results on disk and exit code 130. Resume: ``Run.open``
on an existing directory reuses the cases that finished (same case key), reruns the rest; it never promises a solver
checkpoint restart.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

SCHEMA = 1
STATES = ("prepared", "running", "completed", "partial", "failed", "cancelled")
FINISHED_CASE = ("solved", "diverged", "FAILED")          # a case the solver finished with: reused on resume
EXIT = {"completed": 0, "prepared": 0, "partial": 3, "failed": 1, "cancelled": 130, "nothing solved": 4}
LOG_PATTERNS = ("log.*", "*.sta", "ccx.log", "*.cvg", "gmsh.log")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def sha256_file(path, chunk: int = 1 << 20) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def write_json_atomic(path, obj) -> Path:
    """Write ``obj`` next to ``path`` and rename it into place: a reader never sees a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str, allow_nan=True))
    os.replace(tmp, path)
    return path


def read_json(path) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def mock_active() -> bool:
    """Whether this process runs under ``vegeta.mock`` (``VEGETA_DEBUG`` on, or the mocks installed): its run is then
    labelled ``"mock": true`` in the manifest (Mimesis's convention) and is never read as an engineering result."""
    if os.environ.get("VEGETA_DEBUG", "").strip().lower() in ("1", "true", "on", "yes"):
        return True
    m = sys.modules.get("vegeta.mock")
    return bool(m is not None and m.installed())


def tool_versions(extra: dict | None = None) -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        from importlib.metadata import version
        for dist in ("vegeta-cli", "numpy", "pandas", "cadquery", "gmsh"):
            try:
                out[dist] = version(dist)
            except Exception:
                pass
    except Exception:
        pass
    try:
        root = Path(__file__).resolve().parents[1]
        out["git"] = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout.strip()
        out["git_dirty"] = bool(dirty)
    except Exception:
        pass
    out.update(extra or {})
    return out


def children_pids(pid: int | None = None) -> list[int]:
    """The live child processes of ``pid`` (Linux ``/proc``): the solver a tool started, for the heartbeat."""
    pid = pid or os.getpid()
    out = []
    proc = Path("/proc")
    if not proc.is_dir():
        return out
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if ppid == pid:
            out.append(int(d.name))
    return sorted(out)


def newest_log(case_dir, patterns: Iterable[str] = LOG_PATTERNS) -> Path | None:
    """The most recently modified solver log under ``case_dir`` (the native tools' ``log.simpleFoam``, CalculiX's ``ccx.log``…)."""
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        return None
    best, best_t = None, -1.0
    for pat in patterns:
        for p in case_dir.rglob(pat):
            try:
                t = p.stat().st_mtime
            except OSError:
                continue
            if t > best_t:
                best, best_t = p, t
    return best


def last_progress_line(log: Path, max_bytes: int = 65536) -> str:
    """The last line of the log that reads as native solver progress: OpenFOAM's ``Time = N`` or ``ExecutionTime``,
    CalculiX's iteration / increment lines, Gmsh's percentage — else the last non-empty line."""
    try:
        with log.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            tail = f.read().decode(errors="replace").splitlines()
    except OSError:
        return ""
    keys = ("Time = ", "ExecutionTime", "increment", "iteration", "Solving for", "residual", "%", "Meshing")
    for line in reversed(tail):
        s = line.strip()
        if s and any(k in s for k in keys):
            return s[:160]
    for line in reversed(tail):
        if line.strip():
            return line.strip()[:160]
    return ""


@dataclass
class Budget:
    """What an engineer allows a run: ``max_cases`` cases started, ``max_minutes`` of wall clock before no new case
    is started (a running case finishes), ``case_timeout_s`` per solver case (passed to the tool)."""
    max_cases: int | None = None
    max_minutes: float | None = None
    case_timeout_s: float | None = None


@dataclass
class CaseRecord:
    name: str
    key: str
    status: str = "NOT RUN"            # solved | diverged | FAILED | cancelled | NOT RUN | running
    process: str | None = None         # ok | failed | cancelled | None (not run)
    converged: bool | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    note: str = ""
    metrics: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    reused: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class Heartbeat:
    """A daemon thread that, every ``interval`` seconds while a case runs, logs that this process is alive, which
    child processes it has, the newest solver log's size and last progress line, and whether that log grew since the
    last beat (``advancing``). The ``status.json`` carries the same fields. A quiet log is reported, never acted on."""

    def __init__(self, run: "Run", interval: float = 30.0):
        self.run = run
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._case: str | None = None
        self._dir: Path | None = None
        self._t0 = 0.0
        self._last_size = -1
        self._quiet_since = 0.0

    def start(self, case: str, case_dir: Path) -> None:
        self.stop()
        self._case, self._dir, self._t0 = case, Path(case_dir), time.monotonic()
        self._last_size, self._quiet_since = -1, time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def beat(self) -> dict:
        now = time.monotonic()
        log = newest_log(self._dir) if self._dir else None
        size = log.stat().st_size if log else -1
        advancing = log is not None and size != self._last_size
        if advancing:
            self._quiet_since = now
        self._last_size = size
        info = {"at": utc_now(), "case": self._case, "pid": os.getpid(), "children": children_pids(), "case_elapsed_s": round(now - self._t0, 1),
                "solver_log": str(log.relative_to(self.run.root)) if log else None, "solver_log_bytes": size if log else None,
                "last_progress": last_progress_line(log) if log else "", "advancing": bool(advancing), "quiet_for_s": round(now - self._quiet_since, 1)}
        return info

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                info = self.beat()
                if info["solver_log"]:
                    growth = "advancing" if info["advancing"] else f"no new output for {info['quiet_for_s']:.0f} s"
                    tail = f"{info['solver_log']} {info['solver_log_bytes'] / 1024:.0f} kB, {growth}" + (f" — {info['last_progress']}" if info["last_progress"] else "")
                else:
                    tail = "no solver log yet"
                self.run.log(f"heartbeat: {info['case']} alive (pid {info['pid']}, children {info['children'] or '-'}), {info['case_elapsed_s']:.0f} s into the case; {tail}", echo=False)
                self.run.status(heartbeat=info)
            except Exception as exc:                          # a heartbeat must never break a run
                try:
                    self.run.log(f"heartbeat error: {exc!r}", echo=False)
                except Exception:
                    pass


class Run:
    """One run directory. ``Run.create`` starts a new one, ``Run.open`` resumes an existing one."""

    def __init__(self, root: Path, manifest: dict, *, heartbeat_s: float = 30.0, echo: bool = True):
        self.root = Path(root)
        self.manifest = manifest
        self.cases_dir = self.root / "cases"
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self._log = (self.root / "run.log").open("a", buffering=1, encoding="utf-8")
        self._echo = echo
        self.echo_fn = print                      # a progress bar replaces it with tqdm.write so lines do not break the bar
        self.cancel = threading.Event()
        self.heartbeat = Heartbeat(self, heartbeat_s)
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._status: dict = read_json(self.root / "status.json") or {}
        self.records: dict[str, CaseRecord] = {}
        self._signals_installed = False

    # ---------------------------------------------------------------------------------------------- create / open
    @classmethod
    def create(cls, root, *, workflow: str, campaign: str, config: dict, design: dict | None = None, inputs: dict | None = None,
               versions: dict | None = None, planned_cases: list[str] | None = None, heartbeat_s: float = 30.0, echo: bool = True) -> "Run":
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"{root} is not empty: choose a new --out, or --resume it")
        root.mkdir(parents=True, exist_ok=True)
        run_id = root.name
        manifest = {"schema": SCHEMA, "run_id": run_id, "workflow": workflow, "campaign": campaign, "created_at": utc_now(), "host": socket.gethostname(),
                    "pid": os.getpid(), "command": shlex.join(sys.argv), "cwd": os.getcwd(), "config": config, "config_hash": sha256_json(config),
                    "inputs": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in (inputs or {}).items()}, "design": design or {},
                    "tool_versions": tool_versions(versions), "planned_cases": list(planned_cases or []), "resumed": [], "mock": mock_active(),
                    "artifacts": {"run_log": "run.log", "status": "status.json", "cases": "cases/", "summary": None}}
        write_json_atomic(root / "manifest.json", manifest)
        run = cls(root, manifest, heartbeat_s=heartbeat_s, echo=echo)
        run.log(f"run {run_id} created: workflow {workflow}, campaign {campaign}, {len(planned_cases or [])} case(s) planned, config {manifest['config_hash'][:12]}")
        run.log(f"command: {manifest['command']}")
        run.status("prepared", cases={"planned": len(planned_cases or []), "done": 0, "reused": 0, "failed": 0, "running": None})
        return run

    @classmethod
    def open(cls, root, *, config: dict | None = None, heartbeat_s: float = 30.0, echo: bool = True) -> "Run":
        """Resume: the manifest must exist and, when ``config`` is given, hash to the same configuration (the same
        campaign, quality, design…) — else a ``ValueError`` names the differing keys."""
        root = Path(root)
        manifest = read_json(root / "manifest.json")
        if not manifest:
            raise FileNotFoundError(f"{root} has no manifest.json: not a run directory")
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"{root}: manifest schema {manifest.get('schema')} is not {SCHEMA}")
        if config is not None and sha256_json(config) != manifest.get("config_hash"):
            old = manifest.get("config", {})
            diff = sorted(k for k in set(old) | set(config) if old.get(k) != config.get(k))
            raise ValueError(f"{root} was run with a different configuration (differs in {diff}); start a new run instead of resuming")
        manifest.setdefault("resumed", []).append({"at": utc_now(), "command": shlex.join(sys.argv), "pid": os.getpid()})
        manifest["pid"] = os.getpid()
        manifest["mock"] = bool(manifest.get("mock")) or mock_active()      # resumed under the mocks: the whole run is mock
        write_json_atomic(root / "manifest.json", manifest)
        run = cls(root, manifest, heartbeat_s=heartbeat_s, echo=echo)
        for d in sorted(run.cases_dir.iterdir()) if run.cases_dir.is_dir() else []:
            rec = read_json(d / "case.json")
            if rec:
                run.records[rec["name"]] = CaseRecord(**{k: v for k, v in rec.items() if k in CaseRecord.__dataclass_fields__})
        fin = [r for r in run.records.values() if r.status in FINISHED_CASE]
        run.status(cases={"planned": len(manifest.get("planned_cases", [])), "done": len(fin), "reused": 0,
                          "failed": sum(r.status in ("FAILED", "diverged") for r in fin), "running": None}, current_case=None)
        run.log(f"run {manifest['run_id']} resumed: {len(run.records)} case record(s) on disk, {len(fin)} finished")
        return run

    # ---------------------------------------------------------------------------------------------- log / status
    def log(self, msg: str, *, echo: bool | None = None) -> None:
        line = f"{utc_now()} {msg}"
        with self._lock:
            self._log.write(line + "\n")
            self._log.flush()
        if (self._echo if echo is None else echo):
            self.echo_fn(line)

    def status(self, state: str | None = None, **fields) -> dict:
        with self._lock:
            if state is not None:
                if state not in STATES:
                    raise ValueError(f"state {state!r} not in {STATES}")
                self._status["state"] = state
                if state == "running" and "started_at" not in self._status:
                    self._status["started_at"] = utc_now()
                if state in ("completed", "partial", "failed", "cancelled"):
                    self._status["finished_at"] = utc_now()
            self._status.update(fields)
            self._status.update(updated_at=utc_now(), pid=os.getpid(), elapsed_s=round(time.monotonic() - self._started, 1), run_id=self.manifest["run_id"])
            write_json_atomic(self.root / "status.json", self._status)
            return dict(self._status)

    def add_artifact(self, name: str, path) -> None:
        p = Path(path)
        rel = str(p.relative_to(self.root)) if p.is_absolute() and self.root in p.parents else str(p)
        self.manifest.setdefault("artifacts", {})[name] = rel
        write_json_atomic(self.root / "manifest.json", self.manifest)

    def set_design(self, design: dict) -> None:
        self.manifest["design"] = design
        write_json_atomic(self.root / "manifest.json", self.manifest)

    # ---------------------------------------------------------------------------------------------- signals
    def signals(self):
        """``with run.signals():`` — SIGINT / SIGTERM set the cancel event (the tools stop their solver); a second
        signal raises ``KeyboardInterrupt``."""
        run = self

        class _Ctx:
            def __enter__(self_):
                self_.old = {}

                def handler(signum, frame):
                    if run.cancel.is_set():
                        raise KeyboardInterrupt
                    run.log(f"signal {signal.Signals(signum).name} received: cancelling after the running solver stops (again to abort hard)")
                    run.cancel.set()
                for s in (signal.SIGINT, signal.SIGTERM):
                    try:
                        self_.old[s] = signal.signal(s, handler)
                    except ValueError:                        # not the main thread
                        pass
                return run

            def __exit__(self_, *exc):
                for s, h in self_.old.items():
                    signal.signal(s, h)
                return False
        return _Ctx()

    # ---------------------------------------------------------------------------------------------- cases
    def case_dir(self, name: str) -> Path:
        return self.cases_dir / slug(name)

    def reusable(self, name: str, key: str, *, rerun_failed: bool = False) -> CaseRecord | None:
        """The finished record of ``name`` when it was made with the same case ``key`` (same geometry, values, quality)."""
        rec = self.records.get(name)
        if rec is None or rec.key != key or rec.status not in FINISHED_CASE:
            return None
        if rerun_failed and rec.status == "FAILED":
            return None
        return rec

    def run_case(self, name: str, key: str, fn: Callable[[Path, threading.Event], dict], *, rerun_failed: bool = False, case_dir=None) -> CaseRecord:
        """Run one case through ``fn(case_dir, cancel) -> dict`` (keys ``status``, ``process``, ``converged``, ``note``,
        ``metrics``, ``artifacts``: what the engineering module read from the tool's result) under the heartbeat;
        record it in ``cases/<name>/case.json`` (or ``case_dir``, when the tool chooses the directory). A finished case
        with the same key is reused, not rerun."""
        prev = self.reusable(name, key, rerun_failed=rerun_failed)
        if prev is not None:
            prev.reused = True
            c = dict(self._status.get("cases", {}))
            c["reused"] = c.get("reused", 0) + 1
            self.status(cases=c)
            self.log(f"case {name}: reused ({prev.status}, {prev.duration_s or 0:.0f} s on {prev.finished_at})")
            return prev
        d = Path(case_dir) if case_dir is not None else self.case_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        rec = CaseRecord(name=name, key=key, status="running", started_at=utc_now())
        self.records[name] = rec
        write_json_atomic(d / "case.json", rec.to_dict())
        self.status(cases={**self._status.get("cases", {}), "running": name}, current_case=name)
        self.log(f"case {name}: start (dir {d.relative_to(self.root) if self.root in d.parents else d})")
        t0 = time.monotonic()
        self.heartbeat.start(name, d)
        try:
            out = fn(d, self.cancel) or {}
            rec.status = out.get("status", "FAILED")
            rec.process = out.get("process")
            rec.converged = out.get("converged")
            rec.note = str(out.get("note", "") or "")
            rec.metrics = out.get("metrics", {}) or {}
            rec.artifacts = {k: str(v) for k, v in (out.get("artifacts", {}) or {}).items()}
        except Exception as exc:                              # a failure is a result, never the end of the run
            rec.status, rec.process, rec.note = "FAILED", "failed", f"{type(exc).__name__}: {str(exc)[:300]}"
            self.log(f"case {name}: exception {rec.note}\n{traceback.format_exc()}", echo=False)
        finally:
            self.heartbeat.stop()
        if self.cancel.is_set() and rec.status not in FINISHED_CASE:
            rec.status, rec.process = "cancelled", "cancelled"
        rec.finished_at, rec.duration_s = utc_now(), round(time.monotonic() - t0, 1)
        write_json_atomic(d / "case.json", rec.to_dict())
        c = dict(self._status.get("cases", {}))
        c["done"] = c.get("done", 0) + 1
        c["failed"] = c.get("failed", 0) + (rec.status in ("FAILED", "diverged"))
        c["running"] = None
        self.status(cases=c, current_case=None, last_case={"name": name, "status": rec.status, "duration_s": rec.duration_s})
        self.log(f"case {name}: {rec.status} in {rec.duration_s:.0f} s" + (f" (converged {rec.converged})" if rec.converged is not None else "") + (f" — {rec.note}" if rec.note else ""))
        return rec

    # ---------------------------------------------------------------------------------------------- finish
    def finish(self, summary: dict, *, state: str, note: str = "") -> int:
        """Close the run: ``completed`` publishes ``summary.json`` atomically; ``partial`` (stopped by its budget with
        cases left), ``failed``, ``cancelled`` and ``prepared`` (no solver ran) write ``summary.partial.json``. Returns
        the process exit code."""
        summary = dict(summary)
        summary.setdefault("run_id", self.manifest["run_id"])
        summary["state"] = state
        summary["finished_at"] = utc_now()
        summary["elapsed_s"] = round(time.monotonic() - self._started, 1)
        summary["cases"] = {n: r.to_dict() for n, r in self.records.items()} | summary.get("cases", {})
        if state == "completed":
            path = write_json_atomic(self.root / "summary.json", summary)
            partial = self.root / "summary.partial.json"
            if partial.exists():
                partial.unlink()
        else:
            path = write_json_atomic(self.root / "summary.partial.json", summary)
        self.add_artifact("summary", path)
        self.status(state, note=note)
        n = sum(r.status == "solved" for r in self.records.values())
        self.log(f"run {state}: {n} solved, {sum(r.status == 'diverged' for r in self.records.values())} diverged, "
                 f"{sum(r.status == 'FAILED' for r in self.records.values())} failed, {sum(r.status == 'cancelled' for r in self.records.values())} cancelled, "
                 f"{sum(r.status == 'NOT RUN' for r in self.records.values())} not run; {summary['elapsed_s']:.0f} s; {path.name}" + (f" — {note}" if note else ""))
        self._log.flush()
        if state == "completed" and self.records and n == 0 and any(r.status in ("FAILED", "diverged") for r in self.records.values()):
            return EXIT["nothing solved"]
        return EXIT[state]

    def close(self) -> None:
        self.heartbeat.stop()
        try:
            self._log.close()
        except Exception:
            pass


def slug(name: str) -> str:
    """A case name as a directory name (ASCII, no spaces or slashes; the full name stays in case.json)."""
    out = []
    for ch in name:
        if (ch.isascii() and ch.isalnum()) or ch in "-_.+":
            out.append(ch)
        elif ch in " /,:;()[]|":
            out.append("_")
        else:
            out.append("")
    s = "".join(out).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s[:120] or "case"


# --------------------------------------------------------------------------------------------------- reading runs
def load_run(root) -> dict:
    """What a notebook reads: ``{"root", "manifest", "status", "summary"}`` — ``summary`` is the completed
    ``summary.json`` or ``None`` (a ``summary.partial.json`` is returned under ``"partial"``, never as the summary)."""
    root = Path(root)
    return {"root": root, "manifest": read_json(root / "manifest.json"), "status": read_json(root / "status.json"),
            "summary": read_json(root / "summary.json"), "partial": read_json(root / "summary.partial.json")}


def list_runs(base, workflow: str | None = None) -> list[dict]:
    """The runs under ``base`` (one directory each), newest first: run id, workflow, campaign, state, created, cases."""
    base = Path(base)
    out = []
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir(), reverse=True):
        m = read_json(d / "manifest.json")
        if not m or (workflow and m.get("workflow") != workflow):
            continue
        s = read_json(d / "status.json") or {}
        out.append({"run": str(d), "run_id": m.get("run_id"), "workflow": m.get("workflow"), "campaign": m.get("campaign"), "state": s.get("state"),
                    "created_at": m.get("created_at"), "cases": s.get("cases", {}), "design": (m.get("design") or {}).get("name"), "summary": (d / "summary.json").is_file()})
    return out


def add_common_arguments(ap) -> None:
    """The arguments every workflow script shares."""
    ap.add_argument("--out", default=None, help="the run directory (default workflows/runs/<workflow>/<timestamp>_<campaign>)")
    ap.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue an interrupted run: finished cases are reused, the rest run")
    ap.add_argument("--rerun-failed", action="store_true", help="with --resume: run the cases that FAILED again (a diverged or solved case is kept)")
    ap.add_argument("--prepare-only", action="store_true", help="write the cases and the manifest, run no solver (status 'prepared')")
    ap.add_argument("--list", action="store_true", help="print the campaign's cases and exit; nothing is written")
    ap.add_argument("--cases", default=None, help="run only the cases whose name contains one of these comma-separated texts")
    ap.add_argument("--max-cases", type=int, default=None, help="start at most this many cases")
    ap.add_argument("--max-minutes", type=float, default=None, help="start no new case after this much wall clock")
    ap.add_argument("--case-timeout", type=float, default=None, help="seconds a solver may take per case (the tool stops it)")
    ap.add_argument("--heartbeat", type=float, default=30.0, help="seconds between heartbeats in run.log / status.json")
    ap.add_argument("--quiet", action="store_true", help="log to run.log only, not to the terminal")
    ap.add_argument("--no-progress", action="store_true", help="no progress bars (they are off anyway when the output is not a terminal, e.g. detached)")
    ap.add_argument("--no-cache", action="store_true", help="solve every case even when an earlier run of this book solved it (the cache under workflows/cache/)")
    ap.add_argument("--ignore-prerequisites", action="store_true", help="start the chapter although a prerequisite chapter of the book has no completed run")


def select_cases(names: list[str], filt: str | None) -> list[str]:
    if not filt:
        return list(names)
    wanted = [w.strip() for w in filt.split(",") if w.strip()]
    return [n for n in names if any(w in n for w in wanted)]


def default_out(workflow: str, campaign: str, base=None) -> Path:
    """``workflows/runs/<book>/<chapter>/<timestamp>_<label>`` (``workflow`` is the book, ``campaign`` the chapter
    when given as ``"book/chapter"``; a bare name keeps the flat layout)."""
    base = Path(base) if base else Path(__file__).resolve().parent / "runs" / workflow
    return base / f"{stamp()}_{slug(campaign)}"


class Cache:
    """A content-addressed store of **finished** cases across runs of the same book: ``workflows/cache/<book>/<key>.json``
    holds the case record (status, metrics, artifacts pointing at the run that solved it) — a chapter asked for a
    case it has already solved, in any earlier run, reads it back instead of solving again. ``enabled=False`` turns
    it off for a run (``--no-cache``); ``clean`` empties it. Only ``solved`` and ``diverged`` cases are kept (a FAILED
    case is retried next time)."""

    KEEP = ("solved", "diverged")

    def __init__(self, book: str, base=None, *, enabled: bool = True):
        self.dir = Path(base or os.environ.get("VEGETA_WORKFLOW_CACHE") or Path(__file__).resolve().parent / "cache") / book
        self.enabled = enabled
        self.hits = 0

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        rec = read_json(self.path(key))
        if rec and rec.get("status") in self.KEEP:
            self.hits += 1
            return rec
        return None

    def put(self, key: str, record: dict) -> None:
        if not self.enabled or record.get("status") not in self.KEEP:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.path(key), {**record, "cached_at": utc_now()})

    @staticmethod
    def stats(base=None) -> dict:
        base = Path(base or os.environ.get("VEGETA_WORKFLOW_CACHE") or Path(__file__).resolve().parent / "cache")
        out = {}
        if not base.is_dir():
            return out
        for b in sorted(base.iterdir()):
            if b.is_dir():
                files = list(b.glob("*.json"))
                out[b.name] = {"cases": len(files), "kB": round(sum(f.stat().st_size for f in files) / 1024, 1)}
        return out

    @staticmethod
    def clean(base=None, book: str | None = None) -> int:
        import shutil
        base = Path(base or os.environ.get("VEGETA_WORKFLOW_CACHE") or Path(__file__).resolve().parent / "cache")
        n = 0
        if not base.is_dir():
            return 0
        for b in base.iterdir():
            if b.is_dir() and (book is None or b.name == book):
                n += len(list(b.glob("*.json")))
                shutil.rmtree(b)
        return n
