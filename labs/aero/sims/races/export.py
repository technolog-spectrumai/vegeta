"""One race run on disk: the time series (CSV), the events, and a summary with its provenance.

New runs go to ``races/runs/notebook/<course>/<stamp>_<simulator>/`` (ignored by git), never over committed data:

    timeseries.csv   one row per recorded step, the units in ``UNITS``
    events.json      the referee's events (time, text)
    summary.json     status (finished / DNF and why), collisions, resets, lap time, elapsed simulated time, wall time,
                     the referee's statistics, the gate splits, the provenance (course, model and configuration
                     hashes, seed, timestep, simulator and its version, git commit), the artifacts (paths, videos and
                     any rendering error)

``runs()`` lists what is on disk; ``read_lap()`` gives a run's history and events back for ``races.analysis``."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from . import courses, refquad

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / "labs" / "aero" / "sims" / "races" / "runs" / "notebook"
SCHEMA = 1
COLUMNS = ["t", "x", "y", "z", "vx", "vy", "vz", "thrust_N", "power_W", "speed", "qw", "qx", "qy", "qz", "p", "q", "r"]
UNITS = {"t": "s (simulated)", "x": "m", "y": "m", "z": "m", "vx": "m/s", "vy": "m/s", "vz": "m/s", "thrust_N": "N (total)", "power_W": "W (electrical)",
         "speed": "m/s", "qw": "attitude quaternion", "qx": "attitude quaternion", "qy": "attitude quaternion", "qz": "attitude quaternion",
         "p": "rad/s (body roll rate)", "q": "rad/s (body pitch rate)", "r": "rad/s (body yaw rate)"}


def _sha(data) -> str:
    if not isinstance(data, (bytes, str)):
        data = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def simulator_version(simulator: str) -> str:
    if simulator == "mujoco":
        import mujoco
        return f"mujoco {mujoco.__version__}"
    return "races.mock (kinematic flyer on the racing line)"


def provenance(course, *, simulator: str, config: dict, seed: int | None, timestep: float) -> dict:
    """What the run depended on, as hashes and values: the course (its MJCF scene and parameters), the aircraft model
    (mass budget and the drive table), the run's configuration."""
    drive = refquad.DRIVE_JSON.read_bytes() if refquad.DRIVE_JSON.exists() else b""
    return {"course": course.slug, "course_sha256": _sha(courses.to_mjcf(course, with_quad=False)), "course_params": course.params,
            "model": "reference 5-inch quad (races.refquad)", "model_sha256": _sha({"budget": refquad.BUDGET, "mass": refquad.MASS_KG, "drive": _sha(drive)}),
            "config": config, "config_sha256": _sha(config), "seed": seed, "timestep_s": timestep,
            "simulator": simulator, "simulator_version": simulator_version(simulator), "git_commit": _commit()}


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_run(course, lap: dict, *, simulator: str, config: dict, seed: int | None, timestep: float, wall_s: float,
              videos: dict | None = None, video_errors: dict | None = None, root: Path = RUNS) -> Path:
    """Write one run; returns its folder. ``lap`` is what ``mock.run_mock`` or ``sim.run_mujoco`` returned."""
    d = Path(root) / course.slug / f"{time.strftime('%Y%m%d_%H%M%S')}_{simulator}"
    n = 1
    while d.exists():
        d = d.with_name(f"{d.name.rsplit('#', 1)[0]}#{n}"); n += 1
    d.mkdir(parents=True)
    H = np.asarray(lap["history"], dtype=float)
    cols = COLUMNS[:H.shape[1]] if H.ndim == 2 else COLUMNS[:0]
    np.savetxt(d / "timeseries.csv", H, delimiter=",", header=",".join(cols), comments="", fmt="%.6g")
    if lap.get("pilot") is not None and len(lap["pilot"]):                     # the MuJoCo autopilot's own signals (races.sim.PILOT_COLUMNS)
        from .sim import PILOT_COLUMNS
        np.savetxt(d / "autopilot.csv", np.asarray(lap["pilot"], dtype=float), delimiter=",", header=",".join(PILOT_COLUMNS), comments="", fmt="%.6g")
    (d / "events.json").write_text(json.dumps(_plain([list(e) for e in lap["events"]]), indent=1))
    st = dict(lap["stats"])
    st.pop("controller", None)
    finished = st.get("status") == "finished"
    summary = {"schema": SCHEMA, "course": course.slug, "name": course.name, "kind": course.kind, "length_m": course.length(),
               "status": "finished" if finished else "DNF", "dnf_reason": None if finished else (st.get("reason") or st.get("status") or "no lap"),
               "valid_lap": finished, "collisions": st.get("collisions"), "resets": st.get("resets"), "penalty_s": st.get("penalty [s]"),
               "lap_time_s": st.get("lap time [s]") if finished else None, "elapsed_sim_time_s": float(H[-1, 0] - H[0, 0]) if len(H) else 0.0, "wall_time_s": wall_s,
               "stats": st, "gate_times": lap.get("gate times"), "controller": lap["stats"].get("controller"),
               "provenance": provenance(course, simulator=simulator, config=config, seed=seed, timestep=timestep),
               "units": {c: UNITS[c] for c in cols},
               "artifacts": {"timeseries": "timeseries.csv", "events": "events.json", **({"autopilot": "autopilot.csv"} if lap.get("pilot") is not None and len(lap["pilot"]) else {}),
                             "videos": {k: str(v) for k, v in (videos or {}).items() if v}, "video_errors": video_errors or {}}}
    (d / "summary.json").write_text(json.dumps(_plain(summary), indent=1, ensure_ascii=False))
    return d


def runs(course: str | None = None, root: Path = RUNS) -> list[dict]:
    """Every run on disk (oldest first), each summary with its folder under ``dir``."""
    out = []
    for f in sorted(Path(root).glob(f"{course or '*'}/*/summary.json")):
        try:
            s = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        s["dir"] = str(f.parent)
        out.append(s)
    return out


def read_lap(run: dict) -> dict:
    """The history (as recorded) and the events of one run."""
    d = Path(run["dir"])
    H = np.loadtxt(d / "timeseries.csv", delimiter=",", skiprows=1, ndmin=2)
    events = [tuple(e) for e in json.loads((d / "events.json").read_text())]
    return {"history": H, "events": events}
