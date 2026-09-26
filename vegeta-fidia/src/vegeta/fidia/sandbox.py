"""Run model-written design code out of process and read back what it built.

``execute(source, outdir)`` screens the source (``contract.validate_source``), writes it to ``outdir/design.py``
and runs ``python -I -m vegeta.fidia.runner`` in a fresh temporary directory with a clean environment (no API
keys), a memory cap, a CPU and file-size limit, a wall-clock timeout and a cancel event, then loads
``result.json`` and ``parts.npz`` (never pickles). The walls are resource limits and a speed bump, not a security
boundary: add a ``wrapper`` such as ``("unshare", "-rn")`` or run inside a container for that.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._process import clean_environment, run_walled
from .contract import validate_source

STATUSES = ("ok", "rejected", "build_error", "timeout", "cancelled", "resource_limit")
# what a killed runner's stderr says when it ran out of memory (RLIMIT_AS turns into these, not MemoryError)
_MEMORY_WORDS = ("MemoryError", "std::bad_alloc", "Cannot allocate memory", "out of memory")


@dataclass
class Sandbox:
    """Resource walls for one execution. ``wrapper`` is prepended to the command (e.g. ``("unshare", "-rn")``)."""

    timeout_s: float = 120.0
    memory_mb: int | None = 4096
    file_mb: int | None = 256
    python: str = sys.executable
    wrapper: Sequence[str] = ()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "wrapper": list(self.wrapper)}


@dataclass
class ExecResult:
    """What the runner reported: status, parts (metadata; arrays stay in ``parts.npz``) and measurements."""

    status: str
    parts: list[dict] = field(default_factory=list)
    error: str | None = None
    traceback: str | None = None
    duration_s: float = 0.0
    parameters: dict = field(default_factory=dict)
    bbox_min: list[float] | None = None
    bbox_max: list[float] | None = None
    tessellation: dict = field(default_factory=dict)
    contacts: dict = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    outdir: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def brep(self) -> list[dict]:
        return [p["brep"] for p in self.parts]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        if self.ok:
            names = ", ".join(p["name"] for p in self.parts)
            return f"ok: {len(self.parts)} part(s) [{names}] in {self.duration_s:.1f} s"
        return f"{self.status}: {self.error}"


def execute(source: str, outdir: str | Path, *, sandbox: Sandbox | None = None, parameters: Mapping | None = None,
            cancel: threading.Event | None = None) -> ExecResult:
    """Screen and run ``source``; write ``design.py``, ``execution.json``, ``runner.log`` (+ runner outputs) in ``outdir``."""
    sb = sandbox or Sandbox()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    design = out / "design.py"
    design.write_text(source)
    problems = validate_source(source)
    if problems:
        return _save(out, ExecResult("rejected", error="; ".join(problems), outdir=str(out)))
    for stale in ("result.json", "parts.npz", "model.step"):
        (out / stale).unlink(missing_ok=True)
    work = Path(tempfile.mkdtemp(prefix="fidia-run-"))
    try:
        command = [*sb.wrapper, sb.python, "-I", "-m", "vegeta.fidia.runner", str(design.resolve()), str(out.resolve())]
        if parameters:
            (work / "params.json").write_text(json.dumps(dict(parameters)))
            command.append(str(work / "params.json"))
        env = clean_environment(work)  # -I ignores PYTHON* variables anyway; the child imports from site-packages
        rec = run_walled(command, work, env=env, timeout=sb.timeout_s, cancel=cancel, memory_mb=sb.memory_mb,
                         cpu_s=int(sb.timeout_s) + 5 if sb.timeout_s else None, file_mb=sb.file_mb,
                         log_path=out / "runner.log")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    res = ExecResult("ok", command=rec.command, duration_s=rec.duration_s, outdir=str(out))
    if rec.error == "cancelled":
        res.status, res.error = "cancelled", "cancelled by the user"
    elif rec.error and rec.error.startswith("timed out"):
        res.status, res.error = "timeout", f"the design took longer than {sb.timeout_s:g} s to build"
    elif not (out / "result.json").is_file():
        memory = any(w in rec.stderr for w in _MEMORY_WORDS) or rec.returncode in (-9, -6, 137, 134)
        res.status = "resource_limit" if memory else "build_error"
        tail = "\n".join(rec.stderr.strip().splitlines()[-15:])
        res.error = (f"the runner stopped without a result (return code {rec.returncode}"
                     f"{', probably out of memory' if memory else ''})" + (f": {rec.error}" if rec.error else ""))
        res.traceback = tail or None
    else:
        data = json.loads((out / "result.json").read_text())
        res.status = data.get("status", "build_error")
        for key in ("parts", "error", "traceback", "parameters", "bbox_min", "bbox_max", "tessellation",
                    "contacts"):
            if data.get(key) is not None:
                setattr(res, key, data[key])
        if res.status == "build_error" and any(w in (res.error or "") for w in _MEMORY_WORDS):
            res.status = "resource_limit"
    if res.status not in STATUSES:
        res.status, res.error = "build_error", f"unknown runner status {res.status!r}"
    return _save(out, res)


def _save(out: Path, res: ExecResult) -> ExecResult:
    (out / "execution.json").write_text(json.dumps(res.to_dict(), indent=2, default=str))
    return res
