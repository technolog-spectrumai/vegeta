"""ASTOR-V and ASTOR-BOOST (the variable-sweep, net-dropping ASTOR): every simulation from one script, under one
progress bar.

    labs/aero/sims/astor.sh -j 12                     # everything: CFD, then stability, then the net drop
    labs/aero/sims/astor.sh -j 12 --cfd-only          # one phase: --cfd-only | --stability-only | --netdrop-only
    labs/aero/sims/astor.sh -j 12 --skip-cfd          # stability and the net drop (on the lattice model, or the CFD already installed)
    labs/aero/sims/astor.sh -j 12 --design boost      # one design: v (ASTOR-V) or boost (ASTOR-BOOST); default both
    labs/aero/sims/astor.sh --dry-run                 # list what would run, and stop

``-j N`` is the CPU cores for every phase: OpenFOAM's MPI processes in the CFD (used as given); worker processes in
the stability and net-drop phases, lowered if the machine cannot carry them (cores - 2; ~1 GB of free RAM each;
``--force-cpu`` insists). One bar counts every CFD case, stability job and net-drop sortie together, with the time
left; finished work is kept, so a restart continues (``--fresh`` starts over). Each phase writes ONE timestamped
file ``output/runs/astor_v_<phase>_<time>_<host>.yaml`` (paste them back). A phase that fails does not stop the next.

1. **cfd** — per design, at 12, 35 and 60 deg of sweep, an angle-of-attack sweep (``designs/astor_v_cfd.CASES``; 40
   cases). Answers: does the sliding carriage hold the static margin with the body in the flow; the drag per sweep
   and the pod's cost; the maximum lift spread. Installed as ``designs/data/astor_v_cfd_screening.yaml``
   (``--no-install`` not to): the later phases then use its corrections.
2. **stability** — per design, at 12 deg / 25 m/s, 35 deg / 40 m/s and 60 deg / 60 m/s: trim, the linear modes,
   doublets, the roll on the tailerons, a gust, the stall and its recovery; the stall and top speed per sweep; one
   performance card per design with the wing sweeping by airspeed. To read: Dutch-roll damping and directional
   stability (the fin is small and under the tail), the roll rate on tailerons, the stall spread, the top speed tucked.
3. **netdrop** — sorties against Armageddon's bandit quadcopter (it wanders at 6 m/s, runs and jinks at 7 m/s when
   it hears ASTOR) and against a 50 m/s fixed-wing drone: the capture rate per sortie and per net, for net radii of
   1.0 and 1.5 m; ASTOR-BOOST with its fan. ``--sorties N`` (default 20 per case), ``--latency``, ``--noise``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGNS = HERE.parent / "designs"
sys.path.insert(0, str(DESIGNS))
sys.path.append(str(HERE))

import astor_run as R                                                # noqa: E402  (the shared runner machinery)

R.rs.GLOBS = ["arigo*.py", "astor*.py", "raptors*.py"]
DATA = DESIGNS / "data" / "astor_v_cfd_screening.yaml"
STATES = ((12.0, 25.0), (35.0, 40.0), (60.0, 60.0))                  # (sweep deg, test speed m/s)
NAMES = {"v": "ASTOR-V", "boost": "ASTOR-BOOST"}


def _meta(phase, a):
    return {"aircraft": "ASTOR-V / ASTOR-BOOST", "phase": phase, "designs": a.designs, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": os.uname().nodename, "commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(), "cpu": a.cpu,
            "cfd_installed": DATA.exists()}


# ------------------------------------------------------------------------------------------------ 1. CFD
def phase_cfd(a) -> int:
    import astor_v_cfd as cf
    quality = "fine" if a.full_cfd else a.quality
    todo_all = [(d, s, al) for d in a.designs for s in cf.CASES for al in cf.CASES[s][2]]
    if a.dry_run:
        for d, s, al in todo_all:
            print(f"cfd  {NAMES[d]:12s} sweep {s:4.0f} deg  V {cf.CASES[s][0]:4.0f} m/s  alpha {al:+5.1f}")
        print(len(todo_all), "cases")
        return 0
    R.OUT.mkdir(parents=True, exist_ok=True)
    path = R.OUT / f"astor_v_cfd{'_fine' if a.full_cfd else ''}.yaml"
    doc = R.rs.load(path, a.fresh) or {"aircraft": "ASTOR-V / ASTOR-BOOST", "quality": quality, "configurations": {}}
    for d in a.designs:
        for s in cf.CASES:
            c = doc["configurations"].setdefault(cf.key(d, s), {"design": d, "sweep_deg": s, "speed_m_s": cf.CASES[s][0], "reference": cf.reference(), "rows": []})
            c["rows"] = [r for r in c["rows"] if r.get("status") == "solved"]
    done = {(c["design"], c["sweep_deg"], float(r["alpha_deg"])) for c in doc["configurations"].values() for r in c["rows"]}
    todo = [x for x in todo_all if (x[0], x[1], float(x[2])) not in done]
    print(f"ASTOR-V CFD ({quality}): {len(todo_all) - len(todo)} cases already solved (kept), {len(todo)} to run on {a.cpu} MPI process(es)")
    bar = R._bar(len(todo_all), len(todo_all) - len(todo), "astor-v cfd")
    ok = 0
    for d, s, al in todo:
        err = ""
        try:
            res = cf.case(a.workdir, d, s, alpha_deg=al, quality=quality).ensure(run=True, processors=a.cpu, timeout=a.timeout)
        except Exception as exc:                                     # noqa: BLE001
            res, err = None, f"{type(exc).__name__}: {exc}"
        r = cf.row(d, s, al, res)
        if res is None:
            r["messages"] = [err[:300]]
        c = doc["configurations"][cf.key(d, s)]
        c["rows"].append(r)
        c["rows"].sort(key=lambda x: x["alpha_deg"])
        c["summary"] = cf.summary(c["rows"])
        ok += r["status"] == "solved"
        msg = f"{NAMES[d]} sweep {s:.0f} alpha {al:+.1f}: {r['status']}" + (f"  Cl {r['Cl']:+.3f} Cd {r['Cd']:.4f} Cm {r['Cm_cg']:+.3f}" if r["status"] == "solved"
                                                                             else "  " + (err or "; ".join(r.get("messages", [])))[:160])
        if bar:
            bar.set_postfix_str(msg[:60]); bar.update(1)
        print("  " + msg)
        R.rs.save(path, doc)
    if bar:
        bar.close()
    usable = [k for k, c in doc["configurations"].items() if sum(r.get("status") == "solved" for r in c["rows"]) >= 4]
    installed = False
    if usable and not a.no_install and not a.full_cfd:
        shutil.copy(path, DATA)
        installed = True
        print(f"installed -> {DATA}  ({len(usable)} configurations; phases 2 and 3 now use the CFD's corrections)")
    run = R._write("cfd", {"run": {**_meta("cfd", a), "quality": quality, "installed": installed},
                           "summary": {k: c.get("summary") for k, c in doc["configurations"].items()},
                           "results": {"astor_v_cfd": doc}}, "astor_v")
    print("->", run)
    print(f"{'configuration':18s} {'CD min':>8s} {'drag cm2':>9s} {'CL max':>7s} {'margin':>7s}")
    for k, c in doc["configurations"].items():
        s = c.get("summary") or {}
        if s.get("solved", 0) >= 3:
            print(f"{k:18s} {s.get('CD_min', float('nan')):8.4f} {s.get('drag_area_cm2', float('nan')):9.1f} {s.get('CL_max', float('nan')):7.2f} {s.get('static_margin', float('nan')):7.3f}")
    return 0 if ok == len(todo) else 1


# ------------------------------------------------------------------------------------------------ workers
def _job(job):
    kind = job[0]
    t = time.time()
    try:
        import arigo1_model as am
        import arigo1_mujoco as amj
        import astor_model as AM
        import astor_v_flight as VF
        if kind == "stab":
            name, sweep, Vt, k = job[1], job[2], job[3], job[4]

            class Fixed(amj.Sim):                                    # the wing held at this sweep for the tests
                def __init__(self, variant=None, propulsion=None, mesh=None, bandit=False):
                    import mujoco
                    import numpy as np
                    self.C = VF.get(name, sweep)
                    self.m = mujoco.MjModel.from_xml_string(amj.mjcf(self.C, mesh, bandit))
                    self.d = mujoco.MjData(self.m)
                    self.bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "arigo")
                    self.controls = np.array([0.0, 0.0, 0.0, 0.5])
                    self.wind = np.zeros(3)
                    self.info = {}
            old = amj.Sim
            amj.Sim = Fixed
            try:
                out = amj.stability_tests(f"astor_{name}", "tractor_lowcam", Vt, k)
            finally:
                amj.Sim = old
            C = VF.get(name, sweep)
            out.update({"aircraft": NAMES[name], "sweep_deg": sweep, "stall_m_s": round(VF.stall(C), 1), "top_speed_m_s": round(AM.top_speed(C), 1),
                        "static_margin": round(C["static_margin"], 3), "Cnb": round(C["Cnb"], 4), "aero_source": C["aero_source"]})
        elif kind == "card":
            import arigo_perf as P
            name = job[1]
            sf = lambda bandit=False: VF.SimV(name, bandit=bandit)
            out = P.card(NAMES[name], VF.get(name, VF.SWEEPS[0]), sf, lambda n, sim, record=None: amj.manoeuvre(n, sim, record), 30.0)
            out["speeds_by_sweep"] = {f"{s:.0f}": {"stall_m_s": round(VF.stall(VF.get(name, s)), 1), "top_speed_m_s": round(AM.top_speed(VF.get(name, s)), 1)}
                                      for s in VF.SWEEPS}
        elif kind == "net":
            import astor_netdrop as ND
            name, bandit, radius, seed0, n, lat, noise = job[1:8]
            out = ND.campaign(name, bandit, n=n, seed0=seed0, net_radius=radius, latency_s=lat, noise_m=noise, boost=(name == "boost"))
        else:
            out = {"error": f"unknown job {kind}"}
    except Exception as exc:                                          # noqa: BLE001
        import traceback
        out = {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-900:]}
    out["seconds"] = round(time.time() - t, 1)
    return "|".join(str(x) for x in job), out


# ------------------------------------------------------------------------------------------------ 2. stability
def phase_stability(a) -> int:
    jobs = [("stab", d, s, Vt, k) for d in a.designs for s, Vt in STATES for k in (1.0, 0.35)] + [("card", d) for d in a.designs]
    if a.dry_run:
        print("\n".join(" ".join(str(x) for x in j) for j in jobs)); print(len(jobs), "jobs")
        return 0
    res = R._run_jobs(jobs, a, "astor_v_stability_cache.yaml", "astor-v stability", _job)
    runs = {k.replace("|", "_"): v for k, v in res.items() if k.startswith("stab")}
    cards = {k.split("|")[1]: v for k, v in res.items() if k.startswith("card")}
    run = R._write("stability", {"run": _meta("stability", a), "results": {"astor_v_mujoco": {"runs": runs}, "astor_v_perf": {"cards": cards}}}, "astor_v")
    print(f"{'run':30s} {'stall':>6s} {'top':>6s} {'margin':>7s} {'SP z':>6s} {'DR z':>6s} {'roll deg/s':>11s} {'stall loss m':>13s}")
    for k, r in runs.items():
        if "error" in r:
            print(f"{k:30s} ERROR {r['error']}"); continue
        lm = r["linear_modes"]
        print(f"{k:30s} {r['stall_m_s']:6.1f} {r['top_speed_m_s']:6.1f} {r['static_margin']:7.3f} {lm['short_period']['zeta']:6.2f} {lm['dutch_roll']['zeta']:6.3f} "
              f"{str((r.get('full_aileron_roll') or {}).get('steady_roll_rate_deg_s')):>11s} {str((r.get('stall_and_release') or {}).get('altitude_lost_m')):>13s}")
    for d, c in cards.items():
        print(d, c.get("error") or {x: c["summary"].get(x) for x in ("mass_kg", "min_speed_m_s", "max_speed_m_s", "max_climb_m_s", "roll_rate_deg_s", "takeoff_time_s", "endurance_min")}, c.get("speeds_by_sweep"))
    print("->", run)
    return 0


# ------------------------------------------------------------------------------------------------ 3. net drop
def phase_netdrop(a) -> int:
    per = 1                                                          # one sortie per job: the progress bar counts sorties
    jobs = [("net", d, b, r, seed, per, a.latency, a.noise) for d in a.designs for b in ("armageddon", "fast") for r in (1.0, 1.5)
            for seed in range(a.sorties)]
    if a.dry_run:
        print("\n".join(" ".join(str(x) for x in j) for j in jobs[:6])); print("...", len(jobs), "sorties in", len(jobs) // max(a.sorties, 1), "cases")
        return 0
    tally = {"n": 0, "caught": 0}

    def note(key, out):                                              # shown on the bar: sorties flown and bandits caught so far
        tally["n"] += 1
        tally["caught"] += sum(1 for r in out.get("runs", []) if r.get("caught"))
        return f"caught {tally['caught']}/{tally['n']}"
    res = R._run_jobs(jobs, a, "astor_v_netdrop_cache.yaml", "astor-v net drop", _job, note=note, unit="sortie")
    table = {}
    for k, v in res.items():
        if "error" in v:
            print(k, "ERROR", v["error"], v.get("trace", "")[-300:]); continue
        key = f"{v['design']} | {v['bandit']} | net radius {v['settings']['net_radius']} m"
        t = table.setdefault(key, {"design": v["design"], "bandit": v["bandit"], "net_radius_m": v["settings"]["net_radius"], "sorties": 0,
                                   "caught": 0, "nets": 0, "first_net": 0, "no_drop": 0, "crashes": 0, "runs": []})
        t["sorties"] += v["sorties"]; t["nets"] += v["nets_dropped"]; t["first_net"] += v["caught_with_first_net"]
        t["no_drop"] += v["no_drop_sorties"]; t["crashes"] += v["crashes"]
        t["caught"] += sum(1 for r in v["runs"] if r["caught"]); t["runs"] += v["runs"]
    for t in table.values():
        t["capture_rate_per_sortie"] = round(t["caught"] / max(t["sorties"], 1), 3)
        t["captures_per_net"] = round(t["caught"] / max(t["nets"], 1), 3)
    run = R._write("netdrop", {"run": {**_meta("netdrop", a), "sorties_per_case": a.sorties, "latency_s": a.latency, "noise_m": a.noise},
                               "summary": {k: {x: v[x] for x in v if x != "runs"} for k, v in table.items()},
                               "results": {"astor_v_netdrop": table}}, "astor_v")
    print(f"{'case':46s} {'sorties':>8s} {'caught':>7s} {'per sortie':>11s} {'per net':>8s} {'no drop':>8s} {'crashes':>8s}")
    for k, t in table.items():
        print(f"{k:46s} {t['sorties']:8d} {t['caught']:7d} {t['capture_rate_per_sortie']:11.2f} {t['captures_per_net']:8.2f} {t['no_drop']:8d} {t['crashes']:8d}")
    print("->", run)
    return 0


def counts(a) -> dict:
    """How many units each phase has: CFD cases, stability jobs, net-drop sorties."""
    import astor_v_cfd as cf
    return {"cfd": len(a.designs) * sum(len(c[2]) for c in cf.CASES.values()), "stability": len(a.designs) * (len(STATES) * 2 + 1),
            "netdrop": len(a.designs) * 4 * a.sorties}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="astor.sh", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("-j", "--cpu", type=int, default=4, help="CPU cores for every phase (default 4); see above")
    ap.add_argument("--force-cpu", action="store_true", help="stability and netdrop: use -j exactly as given, without the cap")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cfd-only", action="store_true", help="only the CFD")
    g.add_argument("--stability-only", action="store_true", help="only stability and performance")
    g.add_argument("--netdrop-only", action="store_true", help="only the net drop")
    g.add_argument("--skip-cfd", action="store_true", help="stability and the net drop, no CFD")
    ap.add_argument("--design", choices=("v", "boost"), action="append", help="v (ASTOR-V) or boost (ASTOR-BOOST); default: both")
    ap.add_argument("--fresh", action="store_true", help="forget the saved progress of the phases that run")
    ap.add_argument("--dry-run", action="store_true", help="list the cases and jobs and stop")
    ap.add_argument("--quality", default="screening", help="cfd: mesh quality")
    ap.add_argument("--full-cfd", action="store_true", help="cfd: the fine mesh; kept apart, not installed")
    ap.add_argument("--no-install", action="store_true", help="cfd: do not install the result for the later phases")
    ap.add_argument("--timeout", type=float, default=3 * 3600.0, help="cfd: seconds per case")
    ap.add_argument("--workdir", type=Path, default=HERE.parent / "_runs" / "astor_v_cfd", help="cfd: the cases' folder")
    ap.add_argument("--sorties", type=int, default=20, help="netdrop: sorties per case (default 20)")
    ap.add_argument("--latency", type=float, default=0.10, help="netdrop: the camera's latency [s]")
    ap.add_argument("--noise", type=float, default=0.3, help="netdrop: the camera's position noise [m]")
    a = ap.parse_args(argv)
    a.designs = a.design or ["v", "boost"]
    phases = (["cfd"] if a.cfd_only else ["stability"] if a.stability_only else ["netdrop"] if a.netdrop_only
              else ["stability", "netdrop"] if a.skip_cfd else ["cfd", "stability", "netdrop"])
    fns = {"cfd": phase_cfd, "stability": phase_stability, "netdrop": phase_netdrop}
    n = counts(a)
    if a.dry_run:
        for ph in phases:
            print(f"---- {ph}: {n[ph]} units")
            fns[ph](a)
        print("total:", sum(n[ph] for ph in phases), "units in", len(phases), "phase(s)")
        return 0
    total = sum(n[ph] for ph in phases)
    try:
        from tqdm import tqdm
        import builtins
        bar = tqdm(total=total, desc="astor", unit="unit", dynamic_ncols=True, smoothing=0.05)
        R.GLOBAL_BAR = bar
        plain = builtins.print
        builtins.print = lambda *x, **k: tqdm.write(" ".join(str(v) for v in x))       # lines above the bar, not through it
    except ImportError:
        bar, plain = None, None
        print("(tqdm is not installed in this Python: plain progress lines; pip install tqdm for the bar)")
    t0, ended = time.time(), []
    try:
        for i, ph in enumerate(phases, 1):
            print(f"================ ASTOR {i}/{len(phases)}: {ph}  (-j {a.cpu}) ================")
            t = time.time()
            before = bar.n if bar is not None else 0
            try:
                rc = fns[ph](a)
            except Exception as exc:                                  # noqa: BLE001  a phase that fails does not stop the next
                rc = f"{type(exc).__name__}: {exc}"[:200]
            if bar is not None and bar.n < before + n[ph]:
                bar.update(before + n[ph] - bar.n)                    # what the phase could not do still leaves the bar true to the rest
            ended.append((ph, rc, time.time() - t))
    finally:
        if bar is not None:
            import builtins
            bar.close()
            builtins.print = plain
            R.GLOBAL_BAR = None
    print(f"================ ASTOR: done in {(time.time() - t0) / 60:.0f} min ================")
    for ph, rc, dt_ in ended:
        print(f"  {ph:10s} {'ok' if rc == 0 else 'ended with ' + str(rc)}  ({dt_ / 60:.0f} min)")
    print(f"  results: {R.OUT / 'runs'}/astor_v_*.yaml  (paste the newest of each phase back)")
    return 0 if all(rc == 0 for _, rc, _ in ended) else 1


if __name__ == "__main__":
    sys.exit(main())
