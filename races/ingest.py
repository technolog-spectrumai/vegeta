"""One file to hand over: ``ingest_race.json`` — every race run under ``side_labs/runs/races/`` digested into a single
compact JSON: per course the newest completed run (its manifest essentials, the referee's statistics, the manoeuvre
metrics recomputed from the trajectory, the gate splits with the speed and power over each split, the events, a
downsampled trajectory, the collisions, the videos' names and sizes, the tail of run.log) plus every other run's
state and failure note, the mock baseline for the same courses, and a cross-course comparison. No videos, no
scene files: those stay where they are.

    python -m races ingest                                  # side_labs/runs/races → ingest_race.json
    python -m races ingest --runs <dir> --out <file> --hz 2 --no-mock
"""
from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from . import courses, metrics, mock, refquad

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "side_labs" / "runs" / "races"


def _read(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    return x


def _state(run_dir: Path) -> tuple:
    s = _read(run_dir / "summary.json")
    if s:
        return s.get("state", "completed"), s
    p = _read(run_dir / "summary.partial.json")
    if p:
        return p.get("state", "partial"), p
    st = _read(run_dir / "status.json") or {}
    return st.get("state", "unknown"), None


def downsample(H: np.ndarray, hz: float) -> dict:
    """The trajectory at ``hz`` samples per second, rounded: t, x, y, z, speed, thrust, power (+ tilt from the attitude when recorded)."""
    if len(H) == 0:
        return {}
    t = H[:, 0]
    step = max(int(round(1.0 / hz / max(np.median(np.diff(t)), 1e-3))), 1)
    S = H[::step]
    out = {"t": np.round(S[:, 0], 2).tolist(), "x": np.round(S[:, 1], 2).tolist(), "y": np.round(S[:, 2], 2).tolist(), "z": np.round(S[:, 3], 2).tolist(),
           "speed": np.round(S[:, 9], 2).tolist(), "thrust_N": np.round(S[:, 7], 2).tolist(), "power_W": np.round(S[:, 8], 1).tolist()}
    if H.shape[1] >= 17:
        tilt = [math.degrees(math.acos(max(-1.0, min(1.0, metrics._quat_to_R(q)[2, 2])))) for q in S[:, 10:14]]
        out["tilt_deg"] = np.round(tilt, 1).tolist()
        out["body_rates_deg_s"] = np.round(np.degrees(S[:, 14:17]), 1).tolist()
    return out


def splits(H: np.ndarray, events: list, t_start: float | None) -> list:
    """Per zone-to-zone split: the time, the distance flown, the mean and peak speed, the mean and peak power, the energy."""
    zone_events = [(float(t), s) for t, s in events if str(s).startswith("zone")]
    if not zone_events:
        return []
    t = H[:, 0]
    out, t_prev, name_prev = [], (t_start if t_start is not None else float(t[0])), "start"
    for tz, label in zone_events:
        m = (t >= t_prev) & (t <= tz)
        if m.sum() >= 2:
            seg = H[m]
            d = float(np.linalg.norm(np.diff(seg[:, 1:4], axis=0), axis=1).sum())
            dt = float(np.diff(seg[:, 0]).sum())
            out.append({"from": name_prev, "to": label.split("'")[1] if "'" in label else label, "t_at_zone_s": round(tz, 2), "split_s": round(tz - t_prev, 2), "distance_m": round(d, 1),
                        "mean_speed": round(d / dt, 2) if dt > 0 else None, "peak_speed": round(float(seg[:, 9].max()), 2), "mean_power_W": round(float(seg[:, 8].mean()), 1),
                        "peak_power_W": round(float(seg[:, 8].max()), 1), "energy_Wh": round(float(np.trapezoid(seg[:, 8], seg[:, 0]) / 3600) if hasattr(np, "trapezoid") else float(np.trapz(seg[:, 8], seg[:, 0]) / 3600), 4),
                        "min_altitude_m": round(float(seg[:, 3].min()), 2), "max_altitude_m": round(float(seg[:, 3].max()), 2)})
        t_prev, name_prev = tz, label.split("'")[1] if "'" in label else label
    return out


def digest_run(run_dir: Path, *, hz: float = 2.0, log_lines: int = 30) -> dict:
    run_dir = Path(run_dir)
    state, summary = _state(run_dir)
    manifest = _read(run_dir / "manifest.json") or {}
    status = _read(run_dir / "status.json") or {}
    cases = {}
    for cd in sorted((run_dir / "cases").iterdir()) if (run_dir / "cases").is_dir() else []:
        cj = _read(cd / "case.json") or {}
        cases[cd.name] = {"status": cj.get("status"), "note": cj.get("note"), "elapsed_s": cj.get("elapsed_s") or cj.get("duration_s"),
                          "files": {f.name: f.stat().st_size for f in cd.iterdir() if f.is_file() and f.name != "case.json"}}
    log = (run_dir / "run.log").read_text().splitlines()[-log_lines:] if (run_dir / "run.log").exists() else []
    out = {"run": run_dir.name, "course": manifest.get("campaign") or manifest.get("chapter"), "state": state, "created_at": manifest.get("created_at"), "config": manifest.get("config"),
           "tool_versions": manifest.get("tool_versions"), "status": status, "cases": cases, "log_tail": log}
    stats = _read(run_dir / "cases" / "race" / "stats.json")
    tr = run_dir / "cases" / "race" / "trajectory.npz"
    if summary:
        out["summary"] = {k: summary.get(k) for k in ("mode", "quality", "rules", "counts", "videos", "note", "elapsed_s", "finished_at")}
    if stats:
        out["referee"] = stats.get("stats")
        out["manoeuvre_recorded"] = stats.get("manoeuvre")
        out["gate_times"] = stats.get("gate times")
        out["events"] = stats.get("events")
    if tr.exists():
        H = np.load(tr)["history"]
        slug = out["course"]
        course = courses.build(slug) if slug in courses.COURSES else None
        path = mock.schedule(course) if course is not None else None
        out["trajectory_shape"] = list(H.shape)
        out["manoeuvre"] = metrics.from_history(H, path=path)
        t_start = next((float(t) for t, s in (stats or {}).get("events", []) if s == "start"), None)
        out["splits"] = splits(H, (stats or {}).get("events", []), t_start)
        out["trajectory"] = downsample(H, hz)
        out["collisions"] = [{"t": float(t), "what": s} for t, s in (stats or {}).get("events", []) if "collision" in str(s) or "reset" in str(s)]
        if course is not None:
            near = []
            for row in H[::max(len(H) // 400, 1)]:
                d, name = course.nearest_obstacle(row[1:4])
                near.append((d, name, row[0]))
            d, name, tt = min(near)
            out["closest_approach"] = {"distance_m": round(float(d), 2), "obstacle": name, "t": round(float(tt), 2)}
    return _plain(out)


def ingest(runs_dir: Path = DEFAULT_RUNS, *, hz: float = 2.0, with_mock: bool = True, log_lines: int = 30) -> dict:
    runs_dir = Path(runs_dir)
    doc = {"schema": 1, "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(), "runs_dir": str(runs_dir), "aircraft": refquad.summary(),
           "envelope": metrics.envelope(), "literature": metrics.LITERATURE, "courses": {}, "comparison": {}, "other_runs": []}
    for slug in courses.COURSES:
        cdir = runs_dir / slug
        if not cdir.is_dir():
            doc["courses"][slug] = {"state": "no runs"}
            continue
        runs = sorted([d for d in cdir.iterdir() if d.is_dir()], key=lambda d: d.name)
        completed = [d for d in runs if _state(d)[0] == "completed" and (d / "cases" / "race" / "trajectory.npz").exists()]
        best = completed[-1] if completed else None
        c = courses.build(slug)
        entry = {"course": {"name": c.name, "kind": c.kind, "length_m": round(c.length(), 1), "zones": [z.name for z in c.zones], "params": c.params}}
        if best is not None:
            entry["run"] = digest_run(best, hz=hz, log_lines=log_lines)
        else:
            entry["run"] = None
        for d in runs:
            if d is best:
                continue
            st, s = _state(d)
            note = (s or {}).get("note") or ((_read(d / "cases" / "race" / "case.json") or {}).get("note"))
            doc["other_runs"].append({"course": slug, "run": d.name, "state": st, "note": note})
        if with_mock:
            r = mock.run_mock(c)
            entry["mock_baseline"] = {"stats": _plain(r["stats"]), "manoeuvre": _plain(metrics.from_history(r["history"], path=r["path"]))}
        doc["courses"][slug] = entry
    keys = ["status", "lap time [s]", "penalty [s]", "zones passed", "collisions", "resets", "energy [Wh]", "Wh per km", "max speed [m/s]", "mean speed [m/s]", "peak power [W]", "mean power [W]"]
    mkeys = ["peak lateral acceleration [g]", "peak acceleration along the path [g]", "peak deceleration along the path [g]", "peak load factor (thrust / weight carried)", "least turn radius above 3 m/s [m]",
             "peak roll rate [deg/s]", "peak pitch rate [deg/s]", "peak yaw rate [deg/s]", "peak tilt [deg]", "tracking error to the racing line, RMS [m]", "share of the lap at ≥ 50 % thrust"]
    for slug, e in doc["courses"].items():
        r = e.get("run")
        if r and r.get("referee"):
            doc["comparison"][slug] = {"mujoco": {k: r["referee"].get(k) for k in keys} | {k: (r.get("manoeuvre") or {}).get(k) for k in mkeys},
                                       "mock": {k: e.get("mock_baseline", {}).get("stats", {}).get(k) for k in keys} if with_mock else None,
                                       "closest_approach": r.get("closest_approach")}
        else:
            doc["comparison"][slug] = {"mujoco": None, "mock": {k: e.get("mock_baseline", {}).get("stats", {}).get(k) for k in keys} if with_mock and e.get("mock_baseline") else None}
    return doc


def write(doc: dict, out: Path) -> Path:
    out = Path(out)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=_plain))
    return out


# ================================================================================================= scanning for runs
SCAN_ROOTS = (ROOT / "side_labs" / "runs" / "races", ROOT / "races" / "runs", ROOT / "races" / "runs" / "notebook")


def scan(roots=SCAN_ROOTS, *, extra=()) -> "pd.DataFrame":
    """Every race run found where the workflows (``side_labs/runs/races/<course>/<run>/``) and the thin CLI
    (``races/runs/<course>_<aircraft>_<mode>/``) write them, plus any ``extra`` folders: course, run, state, mode,
    quality, created, whether the lap's trajectory and stats are there, the videos, the sizes, the failure note."""
    import pandas as pd
    rows = []
    for root in list(roots) + list(extra):
        root = Path(root)
        if not root.is_dir():
            continue
        for d in sorted(root.rglob("*")):
            if not d.is_dir():
                continue
            is_run = (d / "manifest.json").exists() or (d / "stats.json").exists() or (d / "trajectory.npz").exists()
            if not is_run or d.parent.name == "cases":                     # a case folder inside a run is not a run
                continue
            man = _read(d / "manifest.json") or {}
            state, summ = _state(d) if (d / "manifest.json").exists() else (("cli run", None) if (d / "stats.json").exists() else ("files", None))
            tr = d / "cases" / "race" / "trajectory.npz" if (d / "cases").is_dir() else d / "trajectory.npz"
            st = d / "cases" / "race" / "stats.json" if (d / "cases").is_dir() else d / "stats.json"
            videos = sorted(p.name for p in d.rglob("*.mp4")) + sorted(p.name for p in d.rglob("*.gif"))
            note = (summ or {}).get("note") or (_read(d / "cases" / "race" / "case.json") or {}).get("note") if (d / "cases").is_dir() else None
            course = man.get("campaign") or man.get("chapter") or (d.parent.name if d.parent.name in courses.COURSES else d.name.split("_")[0])
            rows.append({"course": course, "run": str(d.relative_to(ROOT)) if ROOT in d.parents else str(d), "state": state, "mode": (summ or {}).get("mode") or (man.get("config") or {}).get("mock") and "mock" or ("mujoco" if man else ""),
                         "quality": (summ or {}).get("quality") or (man.get("config") or {}).get("quality"), "created": man.get("created_at"), "trajectory": tr.exists(), "stats": st.exists(),
                         "videos": ", ".join(videos), "size [MB]": round(sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e6, 1), "note": (note or "")[:120]})
    df = pd.DataFrame(rows, columns=["course", "run", "state", "mode", "quality", "created", "trajectory", "stats", "videos", "size [MB]", "note"])
    return df.sort_values(["course", "run"]).reset_index(drop=True) if len(df) else df


def expected_locations() -> dict:
    """Where the files are supposed to be, for the scan's report when nothing is found."""
    return {"workflow runs (wf R1–R4)": str(ROOT / "side_labs" / "runs" / "races" / "<course>" / "<timestamp>_refquad_<quality>" / "cases" / "race" / "{trajectory.npz, stats.json, map.png, scene.xml}"),
            "workflow videos": str(ROOT / "side_labs" / "runs" / "races" / "<course>" / "<run>" / "cases" / "video-<camera>" / "<course>_<camera>.mp4"),
            "thin CLI runs (python -m races run)": str(ROOT / "races" / "runs" / "<course>_refquad_<mode>" / "{trajectory.npz, stats.json, map.png, replay.gif}"),
            "ingest file": str(ROOT / "ingest_race.json"), "notebook 50's export": str(ROOT / "designs" / "data" / "four_races.json")}
