"""SOUND EVALUATION from CFD: the multirotors' propellers and the Pekari recoilless cannon, one structured result file
per run plus charts. Run  labs/aero/sims/sound_eval.sh --help  for this text.

WHAT IT RUNS (targets; default all four)
  quad   Drongo quadcopter (4 x 5x4.3 tri-blade): one rotor in OpenFOAM (rotor_mrf_static / _oblique) per speed,
         trimmed, one rpm correction -> Gutin tones + vortex noise; 1 rotor and all 4 rotors
  hexa   Drako hexacopter (6 x T-Motor 30x10.5), the same
  can1   Pekari recoilless cannon, 8 mm steel ball at 280 m/s: one shot, rhoCentralFoam wedge, 37 microphones at 1 m
  can5   the same cannon, 5 balls --interval s apart: NO extra CFD -- the single shot's CFD recording is overlapped in
         code, each copy delayed by its firing time and scaled by its source strength (the 0-D model's peak exit
         mass flow at that shot's sagged plenum pressure). can1 and can5 share the one CFD case.
  Speeds (quad, hexa): --speeds, km/h, default 0,20,40,60,80.
  Cost on a laptop: a rotor case ~5-20 min (screening) on 8 cores; a cannon shot (1.1 M cells, ~144k steps) several
  hours on 8 cores -- --can-mesh coarse (~0.2 M cells) for a first look.  Needs OpenFOAM (source its bashrc first).

MULTICORE
  -j N / --cpu N        cores in all (default: every core of the machine)
  -p K / --parallel K   CFD cases run at the same time, each on N/K cores (default 1: one case on all N cores).
                        The cannon's 5 shots and the rotor speeds are independent: -j 12 -p 3 runs three at once.

USAGE
  labs/aero/sims/sound_eval.sh                         everything, all cores
  labs/aero/sims/sound_eval.sh -j 12 quad hexa         the multirotors only, 12 cores
  labs/aero/sims/sound_eval.sh -j 16 -p 4 quad hexa    four rotor speeds at a time on 4 cores each
  labs/aero/sims/sound_eval.sh --no-cfd                dry run: blade-element rotors; cannon cases written, not run
  labs/aero/sims/sound_eval.sh --fresh                 ignore solved cases (otherwise reused)

RESTARTING
  Ctrl-C any time and run the same command again: solved cases come back from their OpenFOAM folders
  (labs/aero/sims/output/sound_cfd/...), an interrupted cannon shot continues from its last written time.

OUTPUT
  ONE file per run:  labs/aero/sims/output/runs/sound_eval_<YYYYmmdd-HHMMSS>_<host>.yaml
    settings, for every target and speed/shot: operating point (rpm, thrust, torque, ball speed, plenum), the
    source of each number (CFD / fallback), sound metrics at the listener (LAeq, LApeak... see below), the
    cannon's directivity (peak dB at 1 m on the ring, 0-180 deg), the WAV paths, and a KPI section.
  WAVs:   output/sound_wav/<run>/drongo_quad_<kmh>kmh_1prop.wav, _4props.wav, drako_hexa_..._6props.wav,
          pekari_1shot_10m_90deg.wav, pekari_5shots_10m_90deg.wav
  Charts: next to the YAML (sound_charts.py):  <stem>_levels.png, <stem>_speed.png, <stem>_spectra.png,
          <stem>_cannon.png, <stem>_kpi.png
  Several machines -> one file:  python labs/aero/sims/sound_eval.py --merge runs/sound_eval_*.yaml -o all.yaml

METRICS (at the listener; all from the WAV's pressure signal, before the 16-bit scaling)
  LAeq_dB (A-weighted energy average over the signal: 5 s for rotors, the burst for the cannon), LZpeak_dB (peak
  unweighted), LAFmax_dB (A-weighted, 125 ms fast), SEL_dBA (sound exposure, cannon), peak_Pa, dominant_hz
  (strongest A-weighted spectral line), sharpness proxy hf_share (energy above 2 kHz / total, A-weighted).
"""
from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import re
import threading

import numpy as np
import yaml

try:
    from tqdm import tqdm
except ImportError:                                            # pip install tqdm for the bar; runs without it
    tqdm = None


class Progress:
    """One total bar over every CFD case of the run (a rotor speed point or a cannon shot = 1). A running cannon shot
    counts its fraction: the solver's time over the case's end time, read from its log every few seconds."""

    def __init__(self, total):
        self.done, self.live, self.lock = 0.0, {}, threading.Lock()
        self.bar = tqdm(total=total, unit="case", dynamic_ncols=True, smoothing=0.05,
                        bar_format="{l_bar}{bar}| {n:.2f}/{total} cases [{elapsed}<{remaining}] {postfix}") if tqdm else None
        self.stop = threading.Event()
        if self.bar:
            threading.Thread(target=self._poll, daemon=True).start()

    def track(self, key, log, end):
        with self.lock:
            self.live[key] = (Path(log), end, 0.0)

    def finish(self, key, label=""):
        with self.lock:
            self.live.pop(key, None)
            self.done += 1
        self._draw(label)

    def _poll(self):
        while not self.stop.wait(5.0):
            with self.lock:
                for k, (log, end, _) in list(self.live.items()):
                    self.live[k] = (log, end, _sim_time(log) / end if end else 0.0)
            self._draw()

    def _draw(self, label=""):
        if not self.bar:
            return
        with self.lock:
            n = self.done + sum(min(f, 0.999) for _, _, f in self.live.values())
            running = ", ".join(f"{k} {f:.0%}" for k, (_, _, f) in self.live.items())
        self.bar.n = round(min(n, self.bar.total), 3)
        if label or running:
            self.bar.set_postfix_str(running or label, refresh=False)
        self.bar.refresh()

    def close(self):
        self.stop.set()
        if self.bar:
            self.bar.close()


def _sim_time(log: Path) -> float:
    """The last 'Time = ' in a solver log (read from its tail)."""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 20000))
            m = re.findall(rb"^Time = ([0-9.eE+-]+)", f.read(), re.M)
        return float(m[-1]) if m else 0.0
    except OSError:
        return 0.0


PROGRESS: Progress | None = None


def print(*a, **k):                                            # noqa: A001 - keep the bar at the bottom
    import builtins
    if tqdm and PROGRESS and PROGRESS.bar:
        tqdm.write(" ".join(map(str, a)))
    else:
        builtins.print(*a, **k)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in [ROOT / "scripts/prop_sound", ROOT / "labs/aero/exp", ROOT / "labs/aero/designs", *sorted(ROOT.glob("libs/*/src"))]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = HERE / "output"
TARGETS = ("quad", "hexa", "can1", "can5")
CAN_MESH = {"fine": (0.5, 5.0), "coarse": (1.0, 10.0), "smoke": (2.0, 25.0)}   # (exit cell, far cell) mm


# -- metrics ----------------------------------------------------------------------------------------------------------
def a_weight_db(f):
    f = np.maximum(np.asarray(f, float), 1e-3)
    f2 = f * f
    ra = 12194 ** 2 * f2 * f2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2))
    return 20 * np.log10(ra) + 2.0


def a_filter(x, rate):
    F = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), 1 / rate)
    return np.fft.irfft(F * 10 ** (a_weight_db(fr) / 20), len(x)), fr, np.abs(F) ** 2


def metrics(x, rate, impulsive=False) -> dict:
    x = np.asarray(x, float)
    db = lambda p2: round(10 * math.log10(max(p2, 1e-30) / 4e-10), 1)
    xa, fr, P = a_filter(x, rate)
    w = int(0.125 * rate)
    fast = np.sqrt(np.convolve(xa ** 2, np.ones(w) / w, mode="same")) if len(xa) > w else np.abs(xa)
    PA = P * 10 ** (a_weight_db(fr) / 10)
    k = int(np.argmax(PA[1:])) + 1
    active = xa[np.abs(xa) > 1e-3 * np.abs(xa).max()] if impulsive and np.abs(xa).max() > 0 else xa
    out = dict(LAeq_dB=db(float(np.mean(active ** 2))), LZpeak_dB=db(float(np.max(x ** 2))),
               LAFmax_dB=db(float(np.max(fast) ** 2)), peak_Pa=round(float(np.abs(x).max()), 3),
               dominant_hz=round(float(fr[k]), 1), hf_share=round(float(PA[fr > 2000].sum() / max(PA.sum(), 1e-30)), 3))
    if impulsive:
        out["SEL_dBA"] = db(float(np.sum(xa ** 2) / rate))       # re 1 s
    return out


def spectrum(x, rate, bands=1 / 3):
    """Third-octave band levels (dB, unweighted) 25 Hz - 20 kHz, for the charts."""
    F = np.abs(np.fft.rfft(x)) ** 2 * 2 / (len(x) ** 2)
    fr = np.fft.rfftfreq(len(x), 1 / rate)
    centres = 1000 * 2 ** (np.arange(-16, 14) * bands)
    rows = []
    for fc in centres:
        if fc * 2 ** (bands / 2) > rate / 2:
            break
        sel = (fr >= fc * 2 ** (-bands / 2)) & (fr < fc * 2 ** (bands / 2))
        rows.append([round(float(fc), 1), round(10 * math.log10(max(F[sel].sum() / 2, 1e-30) / 4e-10), 1)])
    return rows


# -- multirotors ------------------------------------------------------------------------------------------------------
def run_rotors(target, args, env, wav_dir, pool_cores):
    import prop_sound as ps
    m = ps.machine(target)
    speeds = [float(s) for s in args.speeds.split(",") if s.strip()]
    print(f"[{target}] {m['name']}: {m['n']} rotors, {m['mass']:.2f} kg, propeller D {m['prop'].diameter * 1000:.0f} mm, "
          f"speeds {speeds} km/h", flush=True)
    workdir = OUT / "sound_cfd" / m["name"]
    if args.fresh and workdir.exists():
        import shutil
        shutil.rmtree(workdir)
    pa = Namespace(out=workdir, quality=args.quality, procs=pool_cores, iterate=args.iterate,
                   distance=args.prop_distance, angle=args.prop_angle, no_cfd=args.no_cfd)
    (workdir / "cfd").mkdir(parents=True, exist_ok=True)       # point() puts its cases in <out>/cfd
    if not args.no_cfd:
        ps.stl(m, workdir / "cfd")                              # the blade once, before cases run side by side
    with ThreadPoolExecutor(max(1, args.parallel)) as ex:
        def pt(k):
            r = ps.point(m, k, pa, env)
            PROGRESS and PROGRESS.finish(f"{target} {k:g}", f"{target} {k:g} km/h done")
            return r
        pts = list(ex.map(pt, speeds))
    rows, sigs = [], {}
    for p in pts:
        k = int(round(p["speed_kmh"]))
        one = ps.synth(p, seconds=args.seconds, seed=k)
        offs = np.linspace(-0.015, 0.015, m["n"])
        allr = sum(ps.synth(p, 1 + o, args.seconds, seed=1000 * k + i) for i, o in enumerate(offs))
        sigs[(k, 1)], sigs[(k, m["n"])] = one, allr
        row = {kk: v for kk, v in p.items() if not kk.startswith("_")}
        row["tones"] = [{"hz": round(f, 1), "dB": round(L, 1)} for f, L in p["_tones"]]
        row["one_rotor"] = metrics(one, ps.RATE)
        row["all_rotors"] = metrics(allr, ps.RATE)
        row["spectrum_one"] = spectrum(one, ps.RATE)
        row["spectrum_all"] = spectrum(allr, ps.RATE)
        rows.append(row)
        print(f"  [{target}] {k} km/h: rpm {row['rpm']}, {row['source']}, LAeq one {row['one_rotor']['LAeq_dB']} "
              f"/ all {row['all_rotors']['LAeq_dB']} dB(A)", flush=True)
    full = 1.2 * max(float(np.abs(s).max()) for s in sigs.values())
    for (k, n), s in sigs.items():
        f = wav_dir / f"{m['name']}_{k:03d}kmh_{n}prop{'s' if n > 1 else ''}.wav"
        ps.write_wav(f, s, full)
        for r in rows:
            if int(round(r["speed_kmh"])) == k:
                r.setdefault("wav", {})["one_rotor" if n == 1 else "all_rotors"] = str(f.relative_to(OUT))
    return {"kind": "multirotor", "machine": m["name"], "rotors": m["n"], "mass_kg": round(m["mass"], 3),
            "propeller": getattr(m["prop"], "name", ""), "diameter_mm": round(m["prop"].diameter * 1000, 1),
            "listener": {"distance_m": args.prop_distance, "angle_from_axis_deg": args.prop_angle},
            "quality": args.quality if not args.no_cfd else "blade element (no CFD)", "wav_full_scale_Pa": round(full, 4),
            "points": rows}


# -- cannon -----------------------------------------------------------------------------------------------------------
def run_cannon(target, args, wav_dir, pool_cores):
    import bearing_recoilless as br
    import bearing_sound as bs
    import bearing_sound_cfd as bc
    import sonic_flow as sf
    shots = 1 if target == "can1" else 5
    fine, far = CAN_MESH[args.can_mesh]
    gun, grid = bc.Gun(), bc.Grid(fine, far)
    d = br.design(verbose=False)
    tag = f"{shots}shot{'s' if shots > 1 else ''}"
    root = OUT / "sound_cfd" / f"pekari_shot_{args.can_mesh}"       # one CFD case for can1 and can5
    print(f"[{target}] Pekari: {d['p_bar_g']:.1f} bar g, throat {d['throat_mm']:.2f} mm, {d['v']:.0f} m/s; "
          f"mesh {args.can_mesh} {bc.cell_count(gun, grid):,} cells, dt {bc.time_step(grid):.3g} s", flush=True)
    hist = bs.burst(shots, args.interval, args.refill, d)
    cases = [root / "shot1"]
    peak = lambda h: float(np.max(h["mdot_muzzle"] + h["mdot_back"]))
    scale = [peak(h) / peak(hist[0]) for h in hist]                # each ball's source strength vs the CFD shot

    def one(i):
        st = _one(i)
        PROGRESS and PROGRESS.finish(f"{target} shot{i + 1}", f"{target} shot {i + 1} {st}")
        return st

    def _one(i):
        c, h = cases[i], hist[i]
        done = list((c / "postProcessing/mics").glob("*/p"))
        started = (c / "log.rhoCentralFoam").exists() or (c / "processor0").exists()
        if args.fresh or not c.exists():
            bc.write_case(c, h, gun, grid)
            done, started = [], False
        if args.no_cfd:
            return "written (no CFD)"
        if done and _finished(c):
            print(f"  [{target}] shot {i + 1}: solved before, reused", flush=True)
            return "CFD (reused)"
        t0 = time.time()
        if PROGRESS:
            import json
            PROGRESS.track(f"{target} shot{i + 1}", c / "log.rhoCentralFoam", json.loads((c / "shot.json").read_text())["end_time"])
        print(f"  [{target}] shot {i + 1}: rhoCentralFoam on {pool_cores} cores ...", flush=True)
        if started and pool_cores > 1 and (c / "processor0").exists():
            sf.resume_case(c, processors=pool_cores)
        else:
            sf.run_case(c, steps=bc.STEPS, processors=pool_cores)
        print(f"  [{target}] shot {i + 1}: done in {(time.time() - t0) / 3600:.2f} h", flush=True)
        return "CFD"

    with ThreadPoolExecutor(max(1, args.parallel)) as ex:
        status = list(ex.map(one, range(len(cases))))
    res = {"kind": "cannon", "name": "Pekari recoilless 8 mm", "shots": shots, "interval_s": args.interval,
           "refill_s": args.refill, "design": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()},
           "mesh": {"level": args.can_mesh, "exit_cell_mm": fine, "far_cell_mm": far, "cells": bc.cell_count(gun, grid),
                    "delta_t_s": bc.time_step(grid)},
           "listener": {"distance_m": args.can_distance, "angle_from_muzzle_deg": args.can_angle, "gun_height_m": args.height,
                        "ear_m": args.ear},
           "cases": [str(c.relative_to(OUT)) for c in cases], "status": status,
           "burst_method": "one CFD shot, copies delayed and scaled by the 0-D peak exit mass flow" if shots > 1 else "CFD",
           "per_shot": [{"shot": i + 1, "time_s": round(i * args.interval, 3), "plenum_bar_g": round(h["p_gauge"], 2),
                         "ball_m_s": round(h["v"], 1), "energy_J": round(0.5 * br.PROJ["mass"] * h["v"] ** 2, 1),
                         "air_g": round(h["air_kg"] * 1e3, 2), "barrel_ms": round(h["barrel_s"] * 1e3, 2)}
                        for i, h in enumerate(hist)]}
    if not all(s.startswith("CFD") for s in status):
        res["sound"] = None
        return res
    seconds = args.interval * (shots - 1) + 0.6
    total = np.zeros(int(seconds * bc.RATE))
    c = cases[0]
    base = bc.listener_signal(c, gun, distance=args.can_distance, angle=args.can_angle, height=args.height,
                              ear=args.ear, seconds=seconds)
    t, p = bc.read_mics(c)
    p = p - p[: max(5, len(t) // 200)].mean(axis=0)
    for i in range(shots):
        s = base[: len(base) - int(round(i * args.interval * bc.RATE))] * scale[i]
        res["per_shot"][i]["source_scale"] = round(scale[i], 4)
        k = int(round(i * args.interval * bc.RATE))
        total[k:k + len(s)] += s[: len(total) - k]
        res["per_shot"][i].update(metrics(s, bc.RATE, impulsive=True))
        res["per_shot"][i]["directivity_1m"] = [
            {"deg": a, "peak_dB": round(20 * math.log10(max(scale[i] * float(np.abs(p[:, j]).max()), 1e-9) / 20e-6), 1)}
            for j, (a, _, _) in enumerate(gun.mics())]
        if i == 0:
            n = min(len(t), int(0.03 / max(t[1] - t[0], 1e-9)))
            step = max(1, n // 1500)
            res["trace_1m"] = {"deg": [0, 90, 180], "t_ms": [round(float(x) * 1e3, 4) for x in t[:n:step]],
                               "p_Pa": [[round(float(x), 2) for x in p[:n:step, j]] for j in (0, 18, 36)]}
    full = 1.1 * max(float(np.abs(total).max()), 1e-6)
    f = wav_dir / f"pekari_{tag}_{args.can_distance:g}m_{args.can_angle:g}deg.wav"
    bs.write_wav(f, total, full)
    res["sound"] = {"wav": str(f.relative_to(OUT)), "wav_full_scale_Pa": round(full, 3),
                    **metrics(total, bc.RATE, impulsive=True), "spectrum": spectrum(total, bc.RATE),
                    "trace_listener": {"t_ms": [round(i / bc.RATE * 1e3, 3) for i in range(0, len(total), 48)],
                                       "p_Pa": [round(float(x), 3) for x in total[::48]]}}
    print(f"  [{target}] LZpeak {res['sound']['LZpeak_dB']} dB, SEL {res['sound']['SEL_dBA']} dB(A) at "
          f"{args.can_distance:g} m -> {f}", flush=True)
    return res


def _finished(case: Path) -> bool:
    import json
    import bearing_sound_cfd as bc
    try:
        end = json.loads((case / "shot.json").read_text())["end_time"]
        t, _ = bc.read_mics(case)
        return t[-1] >= end * 0.999
    except Exception:
        return False


# -- KPI, merge, main -------------------------------------------------------------------------------------------------
def kpi(results: dict) -> dict:
    """0-100, quieter is better, min-max across what the file holds: rotors by all-rotor LAeq at each speed (mean),
    and the cannon's SEL per shot."""
    out = {}
    rot = {r["machine"]: float(np.mean([p["all_rotors"]["LAeq_dB"] for p in r["points"]]))
           for r in results.values() if r.get("kind") == "multirotor"}
    can = {f"pekari_{r['shots']}": r["sound"]["SEL_dBA"] - 10 * math.log10(r["shots"])
           for r in results.values() if r.get("kind") == "cannon" and r.get("sound")}
    for name, d in (("multirotor_mean_LAeq_dB", rot), ("cannon_SEL_per_shot_dBA", can)):
        if d:
            lo, hi = min(d.values()), max(d.values())
            out[name] = {k: {"value": round(v, 1), "score": round(100 * (hi - v) / (hi - lo), 1) if hi > lo else 100.0}
                         for k, v in d.items()}
    return out


def merge(files, out):
    data = {"merged_from": [str(f) for f in files], "results": {}}
    for f in files:
        d = yaml.safe_load(Path(f).read_text())
        data.setdefault("settings", d.get("settings"))
        data["results"].update(d.get("results", {}))
    data["kpi"] = kpi(data["results"])
    Path(out).write_text(yaml.safe_dump(data, sort_keys=False, width=140))
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", metavar="TARGET", help=" | ".join(TARGETS) + " (default all)")
    ap.add_argument("-j", "--cpu", type=int, default=os.cpu_count() or 4, help="cores in all")
    ap.add_argument("-p", "--parallel", type=int, default=1, help="CFD cases at the same time (each on cpu/parallel cores)")
    ap.add_argument("--speeds", default="0,20,40,60,80", help="multirotor speeds, km/h")
    ap.add_argument("--quality", default="screening", choices=("smoke", "screening", "fine"), help="rotor CFD mesh")
    ap.add_argument("--iterate", type=int, default=1, help="rpm corrections to the trim thrust")
    ap.add_argument("--seconds", type=float, default=5.0, help="rotor WAV length")
    ap.add_argument("--prop-distance", type=float, default=10.0)
    ap.add_argument("--prop-angle", type=float, default=60.0, help="from the rotor axis, deg")
    ap.add_argument("--can-mesh", default="fine", choices=tuple(CAN_MESH))
    ap.add_argument("--interval", type=float, default=0.12, help="s between balls (can5)")
    ap.add_argument("--refill", type=float, default=0.25, help="plenum refill time constant, s (assumed)")
    ap.add_argument("--can-distance", type=float, default=10.0)
    ap.add_argument("--can-angle", type=float, default=90.0, help="from the muzzle direction, deg")
    ap.add_argument("--height", type=float, default=1.0, help="cannon height above ground, m")
    ap.add_argument("--ear", type=float, default=1.6)
    ap.add_argument("--no-cfd", action="store_true", help="dry run (rotors: blade element; cannon: cases only)")
    ap.add_argument("--fresh", action="store_true", help="ignore solved cases")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--merge", nargs="+", metavar="YAML", help="merge result files (with -o) and exit")
    ap.add_argument("-o", "--output", help="merge output")
    args = ap.parse_args(argv)
    if args.merge:
        return merge(args.merge, args.output or "sound_eval_merged.yaml")
    targets = args.targets or list(TARGETS)
    bad = [t for t in targets if t not in TARGETS]
    if bad:
        ap.error(f"unknown target(s) {bad}: choose from {TARGETS}")
    args.parallel = max(1, min(args.parallel, args.cpu))
    cores = max(1, args.cpu // args.parallel)
    env = None
    if not args.no_cfd:
        from vegeta import aeromant
        env = aeromant.OpenFOAMEnvironment.detect()
        if env.flavor() is None:
            sys.exit("OpenFOAM not found: source its bashrc (or ./install_local.sh), or run with --no-cfd")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    host = socket.gethostname().split(".")[0]
    stem = f"sound_eval_{stamp}_{host}"
    wav_dir = OUT / "sound_wav" / stem
    wav_dir.mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    path = OUT / "runs" / f"{stem}.yaml"
    print(f"sound_eval: {' '.join(targets)} | {args.cpu} cores, {args.parallel} case(s) at a time on {cores} | -> {path}")
    doc = {"run": {"started": stamp, "host": host, "cpu": args.cpu, "parallel": args.parallel, "cores_per_case": cores,
                   "targets": targets, "openfoam": None if env is None else str(env.flavor())},
           "settings": {k: v for k, v in vars(args).items() if k not in ("merge", "output", "targets")}, "results": {}}
    global PROGRESS
    n_speeds = len([x for x in args.speeds.split(",") if x.strip()])
    PROGRESS = Progress(sum(n_speeds if t in ("quad", "hexa") else 1 for t in targets))
    t0 = time.time()
    for tgt in targets:
        try:
            r = run_rotors(tgt, args, env, wav_dir, cores) if tgt in ("quad", "hexa") else run_cannon(tgt, args, wav_dir, cores)
        except KeyboardInterrupt:
            print("interrupted: run the same command again to go on"); break
        except Exception as e:                                        # one target failing keeps the others
            r = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{tgt}] FAILED: {r['error']}", flush=True)
        doc["results"][tgt] = r
        doc["kpi"] = kpi(doc["results"])
        doc["run"]["elapsed_h"] = round((time.time() - t0) / 3600, 3)
        path.write_text(yaml.safe_dump(doc, sort_keys=False, width=140))     # after every target: a crash keeps the rest
    PROGRESS.close()
    print("wrote", path)
    if not args.no_charts:
        import sound_charts
        sound_charts.charts(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
