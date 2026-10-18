"""Launching a chapter and following its run — the GUI's only moving parts.

A chapter's script is started **detached** (its own session, like ``nohup``): closing the window never stops a
run. The GUI then follows the run through the files the contract writes — ``status.json`` for the state, the case
counts and the heartbeat, ``run.log`` for the text — exactly as ``wf status`` and ``wf tail`` do. Stopping sends
SIGINT to the pid in ``status.json``: the tool stops its solver, the run ends ``cancelled``, finished cases stay.
"""
from __future__ import annotations

import os
import shlex
import signal
import sys
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QTimer, pyqtSignal

import library
import runkit

ROOT = library.ROOT


def command_for(info: dict, options: dict, *, mode: str = "run", limits: str = "", cache: bool = True) -> list[str]:
    """The argv of a chapter run (without the interpreter): the script, ``--chapter``, the options, the mode flags."""
    argv = [info["script"], "--chapter", info["campaign"]]
    for flag, value in options.items():
        if value not in (None, ""):
            argv += [flag, str(value)]
    if mode == "prepare":
        argv.append("--prepare-only")
    if mode == "list":
        argv.append("--list")
    if not cache:
        argv.append("--no-cache")
    if limits.strip():
        argv += shlex.split(limits)
    return argv


def alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


class RunFollower(QObject):
    """Polls one run directory: emits ``status`` (the status.json dict) and ``log_lines`` (new lines of run.log)."""

    status = pyqtSignal(dict)
    log_lines = pyqtSignal(list)
    finished = pyqtSignal(dict)

    def __init__(self, parent: QObject | None = None, interval_ms: int = 1500) -> None:
        super().__init__(parent)
        self.run_dir: Path | None = None
        self._pos = 0
        self._last_state = None
        self.timer = QTimer(self)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self.poll)

    def follow(self, run_dir: Path | None, *, from_start: bool = True) -> None:
        self.run_dir = Path(run_dir) if run_dir else None
        self._pos = 0
        self._last_state = None
        if self.run_dir is not None:
            if not from_start:
                log = self.run_dir / "run.log"
                self._pos = log.stat().st_size if log.exists() else 0
            self.timer.start()
            self.poll()
        else:
            self.timer.stop()

    def poll(self) -> None:
        if self.run_dir is None:
            return
        log = self.run_dir / "run.log"
        if log.exists():
            with log.open("rb") as f:
                f.seek(self._pos)
                chunk = f.read()
                self._pos = f.tell()
            if chunk:
                self.log_lines.emit(chunk.decode(errors="replace").splitlines())
        st = runkit.read_json(self.run_dir / "status.json") or {}
        if st:
            if st.get("state") == "running" and not alive(st.get("pid")):
                st = {**st, "state": "running? (pid gone)"}
            self.status.emit(st)
            if st.get("state") in ("completed", "partial", "failed", "cancelled", "prepared") and self._last_state != st.get("state"):
                self.finished.emit(st)
            self._last_state = st.get("state")


class Launcher(QObject):
    """Starts a chapter's script detached and finds the run directory it creates."""

    launched = pyqtSignal(str, object)        # the command line, the run directory (Path or None yet)
    failed = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.python = sys.executable
        self.out_dir = library.RUNS

    def launch(self, argv: list[str], *, label: str) -> tuple[bool, str, Path | None]:
        """Start ``python <argv>`` in its own session; the terminal output goes to ``workflows/runs/<label>_<time>.out``.
        Returns (ok, command line, the new run directory when it already exists)."""
        before = {r["run"] for r in library.all_runs()}
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out = self.out_dir / f"{runkit.slug(label)}_{runkit.stamp()}.out"
        proc = QProcess()
        proc.setWorkingDirectory(str(ROOT))
        proc.setProgram(self.python)
        proc.setArguments([str(ROOT / argv[0]), *argv[1:], "--no-progress"])
        proc.setStandardOutputFile(str(out))
        proc.setStandardErrorFile(str(out), QProcess.OpenModeFlag.Append)
        ok, pid = proc.startDetached()
        cmd = shlex.join([".venv/bin/python", *argv])
        if not ok:
            self.failed.emit(f"could not start: {cmd}")
            return False, cmd, None
        new = None
        for _ in range(40):                                  # the run directory appears once the cases are built (a few seconds)
            time.sleep(0.25)
            after = [r for r in library.all_runs() if r["run"] not in before]
            if after:
                new = Path(after[0]["run"])
                break
        self.launched.emit(cmd, new)
        return True, cmd, new

    @staticmethod
    def find_new_run(before: set[str]) -> Path | None:
        after = [r for r in library.all_runs() if r["run"] not in before]
        return Path(after[0]["run"]) if after else None


def stop_run(run_dir: Path) -> str:
    st = runkit.read_json(Path(run_dir) / "status.json") or {}
    pid = st.get("pid")
    if st.get("state") != "running" or not alive(pid):
        return f"{Path(run_dir).name}: no running process (state {st.get('state')})"
    os.kill(int(pid), signal.SIGINT)
    return f"SIGINT sent to pid {pid}: the solver stops, the run ends 'cancelled', finished cases are kept (resume later)"


def resume_argv(run_dir: Path) -> list[str] | None:
    m = runkit.read_json(Path(run_dir) / "manifest.json") or {}
    kind = m.get("kind") or {"airframe_cfd": "cfd", "structure_fea": "fea"}.get(m.get("workflow"))
    if kind not in library.KINDS:
        return None
    return [library.kind_of(m.get("book", "kestrel"), kind)["script"], "--resume", str(run_dir)]


def notebook_line(run_dir: Path) -> str:
    pins = library.pin_commands(run_dir)
    if pins:
        return "; ".join(pins)
    m = runkit.read_json(Path(run_dir) / "manifest.json") or {}
    cfg = m.get("config") or {}
    rel = Path(run_dir).relative_to(ROOT) if ROOT in Path(run_dir).parents else Path(run_dir)
    if m.get("workflow") == "airframe_cfd":
        ch = m.get("chapter") or cfg.get("chapter")
        camp = (library.BOOKS.get(m.get("book"), {}).get("chapters", {}).get(ch) or {}).get("campaign") or cfg.get("campaign", "sweep")
        if camp == "legacy-import":
            camp = "sweep"
        return f'RUNS["airframe_cfd"]["{cfg.get("variant")}"]["{camp}"] = "{rel}"'
    if m.get("workflow") == "structure_fea":
        return f'RUNS["structure_fea"]["{cfg.get("variant")}"] = "{rel}"'
    return f"# {m.get('workflow')}: not read by the notebook"
