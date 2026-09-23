"""Slice an STL with the PrusaSlicer command line."""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Sequence

from ._process import describe_failure, run_command, utc_now
from .gcode import read_gcode
from .result import Result
from .settings import Orientation, PrintSettings


def prusaslicer_version(executable: Sequence[str]) -> str | None:
    try:
        out = subprocess.run([*executable, "--help"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.strip().splitlines()[0] if out.strip() else ""
    return first or None


def slice_stl(
    stl: str | Path,
    settings: PrintSettings,
    orientation: Orientation,
    workdir: str | Path,
    *,
    executable: str | Sequence[str] = "prusa-slicer",
    timeout: float | None = None,
    cancel: threading.Event | None = None,
) -> Result:
    """Slice ``stl`` with explicit settings and an engineer-chosen orientation.

    Writes into ``workdir``: ``inputs/<stl>``, ``profiles/*.ini``, ``<stem>.gcode``,
    ``effective_config.ini``, ``prusaslicer.log`` and ``summary.json``.
    """
    t0 = time.monotonic()
    stl = Path(stl)
    workdir = Path(workdir)
    exe = [executable] if isinstance(executable, str) else list(executable)
    if not isinstance(settings, PrintSettings):
        raise ValueError("settings must be a mellonia.PrintSettings")
    if not isinstance(orientation, Orientation):
        raise ValueError("orientation must be a mellonia.Orientation (use Orientation() for 'as modelled')")
    res = Result(kind="mellonia.slice", metadata={
        "stl": str(stl), "settings_name": settings.name, "settings_source": settings.source,
        "settings": {"printer": settings.printer, "filament": settings.filament, "print": settings.print},
        "orientation": {"rotate_x": orientation.rotate_x, "rotate_y": orientation.rotate_y,
                        "rotate_z": orientation.rotate_z},
        "started_at": utc_now(),
    })

    def finish() -> Result:
        res.duration_s = time.monotonic() - t0
        res.artifacts["summary"] = workdir / "summary.json"
        res.save_json(res.artifacts["summary"])
        return res

    workdir.mkdir(parents=True, exist_ok=True)
    if not stl.is_file():
        res.fail(f"STL not found: {stl}")
        return finish()
    inputs = workdir / "inputs"
    inputs.mkdir(exist_ok=True)
    local_stl = inputs / stl.name
    if local_stl.resolve() != stl.resolve():
        shutil.copy2(stl, local_stl)
    res.artifacts["input_stl"] = local_stl
    inis = settings.write(workdir / "profiles")
    for p in inis:
        res.artifacts[f"{p.stem}_ini"] = p
    gcode = workdir / f"{stl.stem}.gcode"
    if gcode.exists():
        gcode.unlink()  # never report a stale file as this run's output

    cmd = [*exe, "--export-gcode"]
    for p in inis:
        cmd += ["--load", str(p.resolve())]
    cmd += orientation.cli_args()
    cmd += ["--output", str(gcode.resolve()), str(local_stl.resolve())]
    rec = run_command(cmd, workdir, timeout=timeout, log_name="prusaslicer", cancel=cancel)
    res.execution.append(rec)
    if rec.log_file:
        res.artifacts["log"] = Path(rec.log_file)
    res.metadata["tool_versions"] = {"prusaslicer": prusaslicer_version(exe) if rec.error is None else None}
    if rec.error == "cancelled":
        res.fail("slicing cancelled by user", status="cancelled")
        return finish()
    if not rec.ok:
        res.fail(describe_failure(rec))
        return finish()
    if not gcode.is_file() or gcode.stat().st_size == 0:
        res.fail("PrusaSlicer exited successfully but wrote no G-code\n" + describe_failure(rec))
        return finish()
    res.artifacts["gcode"] = gcode

    info = read_gcode(gcode)
    if info.config:
        eff = workdir / "effective_config.ini"
        eff.write_text("# full configuration PrusaSlicer used (from the G-code footer)\n"
                       + "".join(f"{k} = {v}\n" for k, v in sorted(info.config.items())))
        res.artifacts["effective_config"] = eff
        unknown = sorted(k for k in settings.merged() if k not in info.config)
        if unknown:
            res.fail(f"PrusaSlicer does not know setting(s) {unknown} (typo?) — they were ignored")
        defaulted = len(set(info.config) - set(settings.merged()))
        res.messages.append(f"{defaulted} settings were not given explicitly and use PrusaSlicer defaults; "
                            "see effective_config.ini")
    else:
        res.messages.append("G-code has no embedded configuration block; settings could not be verified")
    m = info.metrics()
    m["slicing_status"] = "ok" if res.ok else "settings error"
    res.metrics = m
    if m["filament_used_g"] is None:
        res.messages.append("filament mass not reported: set filament_density to get grams")
    if m["filament_cost"] is None:
        res.messages.append("filament cost not reported: set filament_cost to get cost")
    if m["layer_count"] == 0:
        res.fail("G-code contains no layers")
    return finish()
