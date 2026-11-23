"""Run the geometry runner in a walled-off subprocess: clean environment, resource limits, own process group.

Based on the ``run_command`` every Vegeta tool keeps a copy of (never raises, records everything, terminates on
timeout or cancel); this copy also strips the environment (no API keys reach generated code), applies rlimits
and kills the whole process group.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .result import CommandRecord

KEEP_ENV = ("PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "SYSTEMROOT")
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_environment(home: str | os.PathLike, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """A minimal environment: PATH and locale only, a private HOME/TMPDIR, single-threaded maths, no secrets."""
    env = {k: v for k, v in os.environ.items() if k in KEEP_ENV and not any(w in k.upper() for w in SECRET_WORDS)}
    env.update({"HOME": str(home), "TMPDIR": str(home), "OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg", "PYTHONNOUSERSITE": "1"})
    env.update({k: str(v) for k, v in (extra or {}).items() if not any(w in k.upper() for w in SECRET_WORDS)})
    return env


def _limits(memory_mb: int | None, cpu_s: int | None, file_mb: int | None):
    def apply():
        import resource

        for which, value in ((resource.RLIMIT_AS, memory_mb and memory_mb * 1024 * 1024),
                             (resource.RLIMIT_CPU, cpu_s), (resource.RLIMIT_FSIZE, file_mb and file_mb * 1024 * 1024)):
            if value:
                try:
                    resource.setrlimit(which, (int(value), int(value)))
                except (ValueError, OSError):
                    pass
    return apply


def _kill_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(3)
            return
        except subprocess.TimeoutExpired:
            continue


def run_walled(command: Sequence[str | os.PathLike], cwd: str | os.PathLike, *, env: Mapping[str, str],
               timeout: float | None = None, cancel: threading.Event | None = None, memory_mb: int | None = None,
               cpu_s: int | None = None, file_mb: int | None = None, log_path: str | os.PathLike | None = None) -> CommandRecord:
    """Run ``command`` with exactly ``env``, rlimits and its own process group; never raises."""
    cmd = [str(c) for c in command]
    started_at = utc_now()
    t0 = time.monotonic()

    def finish(returncode, stdout="", stderr="", error=None) -> CommandRecord:
        rec = CommandRecord(command=cmd, cwd=str(cwd), returncode=returncode, duration_s=time.monotonic() - t0,
                            stdout=stdout, stderr=stderr, started_at=started_at, error=error)
        if log_path:
            try:
                Path(log_path).write_text(f"# command:    {' '.join(cmd)}\n# returncode: {returncode}\n# error:      {error}\n"
                                          f"# duration_s: {rec.duration_s:.2f}\n\n===== stdout =====\n{stdout}\n===== stderr =====\n{stderr}")
                rec.log_file = str(log_path)
            except OSError:
                pass
        return rec

    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=dict(env), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL, text=True, errors="replace", start_new_session=True,
                                preexec_fn=_limits(memory_mb, cpu_s, file_mb))
    except OSError as exc:
        return finish(None, error=f"could not start {cmd[0]}: {exc}")
    out: list[str] = []
    err: list[str] = []
    pumps = [threading.Thread(target=lambda s=s, k=k: k.extend(s), daemon=True) for s, k in ((proc.stdout, out), (proc.stderr, err))]
    for t in pumps:
        t.start()
    error = None
    while proc.poll() is None:
        if cancel is not None and cancel.is_set():
            error = "cancelled"
        elif timeout is not None and time.monotonic() - t0 > timeout:
            error = f"timed out after {timeout:g} s"
        if error:
            _kill_group(proc)
            break
        time.sleep(0.05)
    for t in pumps:
        t.join(2)
    return finish(proc.returncode if error is None else None, "".join(out), "".join(err), error)
