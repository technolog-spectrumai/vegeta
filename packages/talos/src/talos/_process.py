"""Run external commands and record everything about them (copied per package by design)."""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .result import CommandRecord


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(
    command: Sequence[str | os.PathLike],
    cwd: str | os.PathLike,
    *,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    log_name: str | None = None,
    on_output: Callable[[str], None] | None = None,
    cancel: threading.Event | None = None,
) -> CommandRecord:
    """Run ``command`` in ``cwd``; never raises for tool problems.

    ``env`` entries are added to the current environment. ``on_output`` receives each stdout line
    (used for progress reporting). Setting ``cancel`` terminates the process. When ``log_name`` is
    given, ``<cwd>/<log_name>.log`` receives the command, return code, stdout and stderr.
    """
    cmd = [str(c) for c in command]
    cwd = Path(cwd)
    started_at = utc_now()
    t0 = time.monotonic()
    full_env = dict(os.environ)
    if env:
        full_env.update({k: str(v) for k, v in env.items()})

    def finish(returncode, stdout="", stderr="", error=None) -> CommandRecord:
        rec = CommandRecord(
            command=cmd, cwd=str(cwd), returncode=returncode, duration_s=time.monotonic() - t0,
            stdout=stdout, stderr=stderr, started_at=started_at, error=error,
        )
        if log_name:
            rec.log_file = str(_write_log(cwd / f"{log_name}.log", rec))
        return rec

    if not cwd.is_dir():
        return finish(None, error=f"working directory does not exist: {cwd}")
    exe = cmd[0]
    if shutil.which(exe, path=full_env.get("PATH")) is None and not Path(exe).is_file():
        return finish(None, error=f"executable '{exe}' not found on PATH")

    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, errors="replace", bufsize=1,
        )
    except OSError as exc:
        return finish(None, error=f"could not start '{exe}': {exc}")

    out: list[str] = []
    err: list[str] = []

    def pump(stream, sink, callback):
        for line in stream:
            sink.append(line)
            if callback is not None:
                try:
                    callback(line)
                except Exception:  # progress callbacks must never break a run
                    pass
        stream.close()

    threads = [
        threading.Thread(target=pump, args=(proc.stdout, out, on_output), daemon=True),
        threading.Thread(target=pump, args=(proc.stderr, err, None), daemon=True),
    ]
    for t in threads:
        t.start()

    error = None
    while proc.poll() is None:
        if cancel is not None and cancel.is_set():
            error = "cancelled"
        elif timeout is not None and time.monotonic() - t0 > timeout:
            error = f"timed out after {timeout} s"
        if error:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            break
        time.sleep(0.05)
    for t in threads:
        t.join()
    return finish(None if error else proc.returncode, "".join(out), "".join(err), error)


def _write_log(path: Path, rec: CommandRecord) -> Path:
    text = (
        f"# command:    {' '.join(rec.command)}\n"
        f"# cwd:        {rec.cwd}\n"
        f"# started_at: {rec.started_at}\n"
        f"# duration_s: {rec.duration_s:.3f}\n"
        f"# returncode: {rec.returncode}\n"
        f"# error:      {rec.error}\n"
        f"\n===== stdout =====\n{rec.stdout}\n===== stderr =====\n{rec.stderr}"
    )
    try:
        path.write_text(text)
    except OSError:
        pass
    return path


def describe_failure(rec: CommandRecord, tail: int = 15) -> str:
    """One readable message for a failed command, including the end of its output."""
    name = Path(rec.command[0]).name if rec.command else "command"
    if rec.error:
        head = f"{name}: {rec.error}"
    else:
        head = f"{name} exited with code {rec.returncode}"
    lines = (rec.stderr.strip() or rec.stdout.strip()).splitlines()[-tail:]
    detail = "\n".join(lines)
    if rec.log_file:
        head += f" (log: {rec.log_file})"
    return head + (f"\n{detail}" if detail else "")
