"""ASTOR: every simulation from one script, under one progress bar. The product is ASTOR-TWIN (two wing motors on a
fixed wing (12 deg; the outer panels plug in), the net launcher with three nets; ``designs/astor_twin``; the default since
2026-11-22): its CFD, stability and net-shot sorties. ASTOR-T (tube launched, talons), ASTOR-V and ASTOR-BOOST are
its predecessors, kept for comparison (``--design t | v | boost``).

    labs/aero/sims/astor.sh -j 12                     # everything: CFD, then stability, then the net drop
    labs/aero/sims/astor.sh -j 12 --cfd-only          # one phase: --cfd-only | --stability-only | --netdrop-only
    labs/aero/sims/astor.sh -j 12 --skip-cfd          # stability and the net drop (on the lattice model, or the CFD already installed)
    labs/aero/sims/astor.sh -j 12 --design boost      # one design: v (ASTOR-V) or boost (ASTOR-BOOST); default both
    labs/aero/sims/astor.sh -j 12 --design t          # ASTOR-T (tube launched, designs/astor_t): its CFD, stability, net drop and launch
    labs/aero/sims/astor.sh -j 12 --launch-only       # ASTOR-T out of its tube: 216 launches (exit speed x elevation x wind x late wings or motor)
    labs/aero/sims/astor.sh -j 12 --design twin       # ASTOR-TWIN (two wing motors, the net launcher; designs/astor_twin): CFD, stability, net-shot sorties
    labs/aero/sims/astor.sh -j 12 --design twin --cfd-only | --stability-only ;  --netshot-only (ASTOR-TWIN's sorties alone)
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
import math
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGNS = HERE.parent / "designs"
sys.path.insert(0, str(DESIGNS))
sys.path.append(str(HERE))

import astor_run as R                                                # noqa: E402  (the shared runner machinery)

R.GLOBS_ACTIVE = ["arigo*.py", "astor*.py", "raptors*.py"]
DATA = DESIGNS / "data" / "astor_v_cfd_screening.yaml"
STATES = ((12.0, 25.0), (35.0, 40.0), (60.0, 60.0))                  # (sweep deg, test speed m/s)
STATES_BY_DESIGN = {"twin": ((12.0, 30.0), (12.0, 45.0), (12.0, 60.0))}   # ASTOR-TWIN: the wing fixed at 12 deg, three speeds (it stalls at 23 m/s with its three nets)


CFD_STATES = {"twin": ["twin_agile", "twin_agile_flap"]}             # what "--design twin" means to the CFD phase (astor_v_cfd.DESIGNS)
FLAP_STATES = {"twin": ((12.0, 24.0),)}                              # stability with every flap down: (sweep, speed)


def states(design):
    return STATES_BY_DESIGN.get(design, STATES)
NAMES = {"v": "ASTOR-V", "boost": "ASTOR-BOOST", "t": "ASTOR-T", "twin": "ASTOR-TWIN", "twin_stowed": "TWIN tube 0", "twin_agile": "TWIN 12%/4%", "twin_agile_flap": "TWIN flaps 20"}


def _inputs() -> str:
    """A short hash of the installed CFD data: the flight phases' saved progress is kept per CFD result (the code's
    own fingerprint does not see data files), so a new CFD run is never answered with the old one's flights."""
    import hashlib
    return hashlib.sha256(DATA.read_bytes()).hexdigest()[:8] if DATA.exists() else "nocfd"


def _meta(phase, a):
    return {"aircraft": "ASTOR-V / ASTOR-BOOST", "phase": phase, "designs": a.designs, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": os.uname().nodename, "commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(), "cpu": a.cpu,
            "cfd_installed": DATA.exists(), "cfd_data": _inputs(), "code": R.fingerprint()}


# ------------------------------------------------------------------------------------------------ 1. CFD
def phase_cfd(a) -> int:
    import astor_v_cfd as cf
    quality = "fine" if a.full_cfd else a.quality
    a.designs = [x for d in a.designs for x in CFD_STATES.get(d, [d])]       # a design's CFD is its states' (ASTOR-TWIN: flaps up, flaps down)
    todo_all = [(d, s, al) for d in a.designs for s in cf.sweeps(d) for al in cf.alphas(d, s)]
    if a.dry_run:
        for d, s, al in todo_all:
            print(f"cfd  {NAMES[d]:12s} sweep {s:4.0f} deg  V {cf.CASES[s][0]:4.0f} m/s  alpha {al:+5.1f}")
        print(len(todo_all), "cases")
        return 0
    R.OUT.mkdir(parents=True, exist_ok=True)
    path = R.OUT / f"astor_v_cfd{'_fine' if a.full_cfd else ''}.yaml"
    doc = R._load(path, a.fresh) or {"aircraft": "ASTOR-V / ASTOR-BOOST", "quality": quality, "configurations": {}}
    for d in a.designs:
        for s in cf.sweeps(d):
            c = doc["configurations"].setdefault(cf.key(d, s), {"design": d, "sweep_deg": s, "speed_m_s": cf.CASES[s][0], "reference": cf.reference(d, s), "rows": []})
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
        R._save(path, doc)
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

            flap = 1.0 if len(job) > 5 else 0.0

            class Fixed(VF.Fixed):                                   # the wing held at this sweep, the alpha limiter on
                def __init__(self, variant=None, propulsion=None, mesh=None, bandit=False):
                    super().__init__(name, sweep, mesh, bandit, flap=flap)
            old = amj.Sim
            amj.Sim = Fixed
            try:
                out = amj.stability_tests(f"astor_{name}", "tractor_lowcam", Vt, k)
            finally:
                amj.Sim = old
            C = VF.with_flap(VF.get(name, sweep), flap)
            out["flaps_deg"] = VF.FLAP_DEG * flap
            k = VF.SAS.get(name)
            if k:                                                    # the modes with the dampers, as equivalent damping derivatives (the washout left out)
                Ca = {**C, "Cnr": C["Cnr"] - abs(C["Cndr"]) * k["yaw_k"] * 57.3 * 2 * Vt / C["b"], "Cmq": C["Cmq"] + C["Cmde"] * k["pitch_k"] * k["pitch_v_ref"] / Vt * 57.3 * 2 * Vt / C["c"]}
                out["linear_modes_bare_airframe"] = out["linear_modes"]
                out["linear_modes"] = {kk: {a_: (round(b_, 4) if isinstance(b_, float) else b_) for a_, b_ in r.items()} for kk, r in am.name_modes(am.modes(Ca, Vt)).items()}
                out["verdict"].update({"dutch_roll_zeta": out["linear_modes"].get("dutch_roll", {}).get("zeta"), "spiral_time_to_double_s": out["linear_modes"].get("spiral", {}).get("t_half_or_double_s")})
                out["augmentation"] = {**k, "note": "yaw damper, pitch damper and g limit in the flight controller; linear_modes are with them"}
            out.update({"aircraft": NAMES[name], "sweep_deg": sweep, "stall_m_s": round(VF.stall(C), 1), "top_speed_m_s": round(AM.top_speed(C), 1),
                        "static_margin": round(C["static_margin"], 3), "Cnb": round(C["Cnb"], 4), "mass_kg": round(C["mass"], 3),
                        "alpha_limiter": {"on": True, "limit_deg": round(math.degrees(C["alpha_stall"]) - C.get("alpha_margin_deg", VF.ALPHA_MARGIN_DEG), 1)},
                        "Clda_per_deg": {"ailerons": round(C["Clda_aileron"], 5), "tail": round(C["Clda_tail"], 5)}, "aero_source": C["aero_source"]})
        elif kind == "card":
            import arigo_perf as P
            name = job[1]
            sf = lambda bandit=False: VF.SimV(name, bandit=bandit)
            out = P.card(NAMES[name], VF.get(name, VF.SWEEPS[0]), sf, lambda n, sim, record=None: amj.manoeuvre(n, sim, record), 30.0)
            out["speeds_by_sweep"] = {f"{s:.0f}": {"stall_m_s": round(VF.stall(VF.get(name, s)), 1), "top_speed_m_s": round(AM.top_speed(VF.get(name, s)), 1)}
                                      for s in VF.sweeps(name)}
        elif kind == "shot":
            import astor_netshot as NS
            out = NS.netshot(seed=job[1])
        elif kind == "launch":
            import astor_tube_launch as TL
            out = TL.launch(job[1], v_exit=job[2], elevation_deg=job[3], headwind=job[4], crosswind=job[5], gust=job[6], open_s=job[7], motor_delay_s=job[8])
            out.pop("history_t_x_z_V_theta_alpha_sweep_bank", None)
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
    jobs = ([("stab", d, s, Vt, k) for d in a.designs for s, Vt in states(d) for k in (1.0, 0.35)]
            + [("stab", d, s, Vt, 1.0, "flaps") for d in a.designs for s, Vt in FLAP_STATES.get(d, ())] + [("card", d) for d in a.designs])
    if a.dry_run:
        print("\n".join(" ".join(str(x) for x in j) for j in jobs)); print(len(jobs), "jobs")
        return 0
    res = R._run_jobs(jobs, a, f"astor_v_stability_cache_{_inputs()}.yaml", "astor-v stability", _job)
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
    jobs = [("net", d, b, r, seed, per, a.latency, a.noise) for d in a.designs for b in a.bandits for r in (1.0, 1.5)
            for seed in range(a.sorties)]
    if a.dry_run:
        print("\n".join(" ".join(str(x) for x in j) for j in jobs[:6])); print("...", len(jobs), "sorties in", len(jobs) // max(a.sorties, 1), "cases")
        return 0
    tally = {"n": 0, "caught": 0}

    def note(key, out):                                              # shown on the bar: sorties flown and bandits caught so far
        tally["n"] += 1
        tally["caught"] += sum(1 for r in out.get("runs", []) if r.get("caught"))
        return f"caught {tally['caught']}/{tally['n']}"
    res = R._run_jobs(jobs, a, f"astor_v_netdrop_cache_{_inputs()}.yaml", "astor-v net drop", _job, note=note, unit="sortie")
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


# ------------------------------------------------------------------------------------------------ 4. tube launch
LAUNCH_SPEEDS, LAUNCH_ELEVATIONS = (14.0, 16.0, 18.0, 20.0), (30.0, 40.0, 50.0)
LAUNCH_WINDS = ((0.0, 0.0, 0.0), (-5.0, 0.0, 0.0), (-8.0, 0.0, 0.0), (10.0, 0.0, 3.0), (0.0, 8.0, 0.0), (0.0, 0.0, 4.0))     # head (+ against), cross, gust down
LAUNCH_MECH = ((0.3, 0.2), (0.8, 0.2), (0.3, 1.0))                                                                          # wings open in, motor starts after [s]


def _launch_jobs(a):
    return [("launch", "t", v, e, w[0], w[1], w[2], m[0], m[1]) for v in LAUNCH_SPEEDS for e in LAUNCH_ELEVATIONS for w in LAUNCH_WINDS for m in LAUNCH_MECH]


def phase_launch(a) -> int:
    """ASTOR-T out of its tube: exit speed x elevation x wind x how late the wings and the motor are."""
    jobs = _launch_jobs(a)
    if a.dry_run:
        print("\n".join(" ".join(str(x) for x in j) for j in jobs[:5])); print("...", len(jobs), "launches")
        return 0
    res = R._run_jobs(jobs, a, f"astor_t_launch_cache_{_inputs()}.yaml", "astor-t tube launch", _job, unit="launch")
    rows = [v for v in res.values()]
    table = {}
    for v in LAUNCH_SPEEDS:
        rs_ = [x for x in rows if "error" not in x and x["v_exit_m_s"] == v]
        table[f"{v:.0f} m/s"] = {"launches": len(rs_), "ok": sum(1 for x in rs_ if x["ok"]),
                                  "worst_sink_m": max((x["sink_below_muzzle_m"] for x in rs_), default=None),
                                  "failed": [{k: x[k] for k in ("elevation_deg", "headwind_m_s", "crosswind_m_s", "gust_down_m_s", "open_s", "sink_below_muzzle_m")} | {"motor_delay_s": j[8]}
                                             for j, x in ((j, res["|".join(str(q) for q in j)]) for j in jobs) if "error" not in x and x["v_exit_m_s"] == v and not x["ok"]]}
    import astor_t as T
    run = R._write("launch", {"run": _meta("launch", a), "summary": table, "launcher": T.launcher(), "results": {"astor_t_launch": rows}}, "astor_v")
    for k, t in table.items():
        print(f"exit {k}: {t['ok']}/{t['launches']} fly away, worst sink {t['worst_sink_m']} m")
    for k, x in res.items():
        if "error" in x:
            print(k, "ERROR", x["error"])
    print("->", run)
    return 0


def phase_netshot(a) -> int:
    """ASTOR-TWIN's net-launcher sortie against astor_hunt's population of quadcopters (astor_netshot)."""
    jobs = [("shot", seed) for seed in range(a.sorties * 3)]
    if a.dry_run:
        print(f"... {len(jobs)} net-shot sorties")
        return 0
    res = R._run_jobs(jobs, a, f"astor_twin_netshot_cache_{_inputs()}.yaml", "astor-twin net shot", _job, unit="sortie")
    rows = [v for v in res.values() if "error" not in v]
    by, nets = {}, {}
    for v in rows:
        b = by.setdefault(v["quadcopter"]["behaviour"], [0, 0])
        b[0] += v["win"]; b[1] += 1
        if v["win"]:
            nets[v["nets_fired"]] = nets.get(v["nets_fired"], 0) + 1
    k, n = sum(v["win"] for v in rows), len(rows)
    summ = {"sorties": n, "netted": k, "win_rate": round(k / max(n, 1), 3), "astor_lost": sum(v["astor_lost"] for v in rows),
            "by_behaviour": {b: {"netted": v[0], "sorties": v[1], "rate": round(v[0] / v[1], 2)} for b, v in by.items()}, "netted_with_net_number": nets,
            "errors": {kk: v["error"] for kk, v in res.items() if "error" in v}}
    run = R._write("netshot", {"run": _meta("netshot", a), "summary": summ, "results": {"astor_twin_netshot": rows}}, "astor_v")
    print(summ)
    print("->", run)
    return 0


def counts(a) -> dict:
    """How many units each phase has: CFD cases, stability jobs, net-drop sorties."""
    import astor_v_cfd as cf
    return {"netshot": a.sorties * 3, "cfd": sum(len(cf.alphas(d, s)) for d0 in a.designs for d in CFD_STATES.get(d0, [d0]) for s in cf.sweeps(d)), "stability": sum(len(states(d)) * 2 + 1 + len(FLAP_STATES.get(d, ())) for d in a.designs),
            "netdrop": len(a.designs) * 2 * len(a.bandits) * a.sorties, "launch": len(_launch_jobs(a))}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="astor.sh", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("-j", "--cpu", type=int, default=4, help="CPU cores for every phase (default 4); see above")
    ap.add_argument("--force-cpu", action="store_true", help="stability and netdrop: use -j exactly as given, without the cap")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cfd-only", action="store_true", help="only the CFD")
    g.add_argument("--stability-only", action="store_true", help="only stability and performance")
    g.add_argument("--netdrop-only", action="store_true", help="only the net drop")
    g.add_argument("--launch-only", action="store_true", help="only ASTOR-T's tube launch (exit speed x elevation x wind x late wings or motor)")
    g.add_argument("--netshot-only", action="store_true", help="only ASTOR-TWIN's net-launcher sorties (3 x --sorties)")
    g.add_argument("--skip-cfd", action="store_true", help="stability and the net drop, no CFD")
    ap.add_argument("--design", choices=("v", "boost", "t", "twin"), action="append", help="twin (ASTOR-TWIN: the ASTOR product, the default), or the earlier t (ASTOR-T), v (ASTOR-V), boost (ASTOR-BOOST)")
    ap.add_argument("--fresh", action="store_true", help="forget the saved progress of the phases that run")
    ap.add_argument("--dry-run", action="store_true", help="list the cases and jobs and stop")
    ap.add_argument("--quality", default="screening", help="cfd: mesh quality")
    ap.add_argument("--full-cfd", action="store_true", help="cfd: the fine mesh; kept apart, not installed")
    ap.add_argument("--no-install", action="store_true", help="cfd: do not install the result for the later phases")
    ap.add_argument("--timeout", type=float, default=3 * 3600.0, help="cfd: seconds per case")
    ap.add_argument("--workdir", type=Path, default=HERE.parent / "_runs" / "astor_v_cfd", help="cfd: the cases' folder")
    ap.add_argument("--sorties", type=int, default=20, help="netdrop: sorties per case (default 20)")
    ap.add_argument("--bandit", choices=("fast", "armageddon", "both"), default="fast",
                    help="netdrop: the bandit; default fast (three runs gave no release in 240 sorties against the slow jinking Armageddon bandit)")
    ap.add_argument("--latency", type=float, default=0.10, help="netdrop: the camera's latency [s]")
    ap.add_argument("--noise", type=float, default=0.3, help="netdrop: the camera's position noise [m]")
    ap.add_argument("--tube-stowed", action="store_true", help="cfd only: ASTOR-TWIN with the launch tube level along the belly (0 deg) instead of 15 deg down; "
                    "8 cases, stored beside the 15 deg ones (twin_stowed_sweep12), the flight model untouched")
    ap.add_argument("--agility-cfd", action="store_true", help="cfd only: ASTOR-TWIN with the agility study's 12 % thick, 4 % cambered section, clean and with the "
                    "flaperons and inboard flaps down 20 deg (tube level); 16 cases, stored beside the others, the flight model untouched")
    a = ap.parse_args(argv)
    a.designs = a.design or ["twin"]
    if a.tube_stowed or a.agility_cfd:
        a.designs = (["twin_stowed"] if a.tube_stowed else []) + (["twin_agile", "twin_agile_flap"] if a.agility_cfd else [])
        a.cfd_only, a.stability_only, a.netshot_only = True, False, False
    a.bandits = ["fast", "armageddon"] if a.bandit == "both" else [a.bandit]
    phases = (["cfd"] if a.cfd_only else ["stability"] if a.stability_only else ["netdrop"] if a.netdrop_only else ["launch"] if a.launch_only
              else ["stability", "netdrop"] if a.skip_cfd else ["cfd", "stability", "netdrop"])
    if "t" in a.designs and not (a.cfd_only or a.stability_only or a.netdrop_only or a.launch_only or a.netshot_only):
        phases.append("launch")
    if a.tube_stowed or a.agility_cfd:
        phases = ["cfd"]
    if "twin" in a.designs:
        if len(a.designs) > 1:
            ap.error("--design twin is run on its own (its mission phase is the net launcher's, not the dropped nets')")
        if a.netdrop_only or a.launch_only:
            ap.error("ASTOR-TWIN has no dropped-net or tube-launch phase: --cfd-only, --stability-only or --netshot-only")
        phases = [ph for ph in phases if ph not in ("netdrop", "launch")]
        if not (a.cfd_only or a.stability_only):
            phases.append("netshot")
    if a.netshot_only:
        a.designs, phases = ["twin"], ["netshot"]
    fns = {"cfd": phase_cfd, "stability": phase_stability, "netdrop": phase_netdrop, "launch": phase_launch, "netshot": phase_netshot}
    n = counts(a)
    if a.dry_run:
        for ph in phases:
            print(f"---- {ph}: {n[ph]} units")
            fns[ph](a)
        print("total:", sum(n[ph] for ph in phases), "units in", len(phases), "phase(s)")
        return 0
    total = sum(n[ph] for ph in phases)
    m = _meta("", a)
    print(f"code: commit {m['commit']}, fingerprint {m['code']}, CFD data {m['cfd_data']}  (saved results are reused only for this code; --fresh to rerun all)")
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
