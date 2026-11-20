"""ASTOR (the propeller dart, ``designs/astor.py``): its three simulation phases. Each is restartable and writes ONE
timestamped file ``output/runs/astor_<phase>_<time>_<host>.yaml`` (paste it back), with charts next to it.

    labs/aero/sims/astor_cfd.sh -j 12          # 1. CFD (OpenFOAM, tens of minutes): the polar, stall, static margin
    labs/aero/sims/astor_stability.sh -j 12    # 2. stability (MuJoCo, ~2 min): modes, doublets, gust, stall, at 30 and 60 m/s
    labs/aero/sims/astor_missions.sh -j 12     # 3. missions (~10 min): performance card, fast-target chases, launch
                                               #    success and operating wind, the ASTOR KPIs, per pack

1. **cfd** — the whole airframe, propeller off: an angle-of-attack sweep at the 30 m/s transit and three small angles
   at the 65 m/s dash (the zero-lift drag that sets the top speed). When at least six angles have solved, the result
   is installed as ``designs/data/astor_cfd_screening.yaml`` and phases 2 and 3 use it (``--no-install`` to keep the
   lattice model). ``-j`` here is OpenFOAM's MPI processes; ``--full-cfd`` runs the fine mesh.
2. **stability** — trim, the linear modes, hands-off, elevator and rudder doublets, aileron pulse and roll, a 3 m/s
   gust, the stall and its recovery, a 60-degree turn; at 30 and 60 m/s, with the model's post-stall pitching moment
   and a weak one (0.35): the bounds.
3. **missions** — for each pack option: the performance card (speeds, climb, turns, roll/pitch/yaw, the chases, the
   mission metrics); tail chases of 30 and 50 m/s fixed-wing targets from 300 m (five starts, the median); rail
   launch success and the operating wind (Monte Carlo; fixed and turning rail); the sizing requirement (a 50 m/s
   target from behind); the ASTOR KPIs and gates (``reports/arigo_catalog/KPIS.md``).

Phases 2 and 3 run on the lattice model until the CFD is installed; the run file says which (``aero_source``).
``-j`` for phases 2 and 3 is capped to what the machine can carry (cores - 2, free RAM); ``--force-cpu`` insists.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime as dt
import json
import shutil
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGNS = HERE.parent / "designs"
sys.path.insert(0, str(DESIGNS))
sys.path.append(str(HERE))

import yaml                                                          # noqa: E402

import arigo1_resume as rs                                           # noqa: E402

rs.GLOBS = ["arigo*.py", "astor*.py", "buteo*.py", "raptors*.py"]
OUT = HERE / "output"
RUNS = HERE.parent / "_runs" / "astor_cfd"
DATA = DESIGNS / "data" / "astor_cfd_screening.yaml"


def _clean(x):
    return json.loads(json.dumps(x, default=float))


def _bar(total, done, desc):
    try:
        from tqdm import tqdm
        return tqdm(total=total, initial=done, desc=desc, unit="job")
    except ImportError:
        return None


def _write(phase, doc, name="astor"):
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    p = OUT / "runs" / f"{name}_{phase}_{stamp}_{socket.gethostname()}.yaml"
    p.write_text(yaml.safe_dump(_clean(doc), sort_keys=False, width=140))
    return p


def _meta(phase, a):
    import astor_model as M
    C = M.get()
    return {"aircraft": "ASTOR", "phase": phase, "started": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "host": socket.gethostname(), "commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(),
            "cpu": a.cpu, "aero_source": C["aero_source"], "mass_kg": round(C["mass"], 3), "pack": C["mass_budget"]["pack"]["name"]}


# ------------------------------------------------------------------------------------------------ 1. CFD
def phase_cfd(a, cf=None, name="astor", data=None, meta=None) -> int:
    if cf is None:
        import astor_cfd as cf
    data = data or DATA
    meta = meta or _meta
    quality = "fine" if a.full_cfd else a.quality
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_cfd{'_fine' if a.full_cfd else ''}.yaml"
    doc = rs.load(path, a.fresh) or {"aircraft": name.upper(), "speed_m_s": cf.SPEED, "dash_speed_m_s": cf.DASH_SPEED, "quality": quality,
                                     "rows": [], "dash_rows": []}
    ref = cf.reference()
    doc["reference"] = {**ref, "cg_mm": list(ref["cg_mm"])}
    doc["rows"] = [r for r in doc.get("rows", []) if r.get("status") == "solved"]
    doc["dash_rows"] = [r for r in doc.get("dash_rows", []) if r.get("status") == "solved"]
    todo = [("rows", al, cf.SPEED) for al in a.alphas if float(al) not in {float(r["alpha_deg"]) for r in doc["rows"]}]
    todo += [("dash_rows", al, cf.DASH_SPEED) for al in cf.DASH_ALPHAS if float(al) not in {float(r["alpha_deg"]) for r in doc["dash_rows"]}]
    total = len(a.alphas) + len(cf.DASH_ALPHAS)
    print(f"{name.upper()} CFD ({quality}): {total - len(todo)} cases already solved (kept), {len(todo)} to run on {a.cpu} MPI process(es)")
    bar = _bar(total, total - len(todo), f"{name} cfd")
    ok = 0
    for key, al, V in todo:
        err = ""
        try:
            res = cf.case(a.workdir, alpha_deg=al, V=V, quality=quality).ensure(run=True, processors=a.cpu, timeout=a.timeout)
        except Exception as exc:                                     # noqa: BLE001
            res, err = None, f"{type(exc).__name__}: {exc}"
        r = cf.row(al, V, res)
        if res is None:
            r["messages"] = [err[:300]]
        doc[key].append(r)
        doc[key].sort(key=lambda x: x["alpha_deg"])
        doc["summary"] = cf.summary(doc["rows"])
        ok += r["status"] == "solved"
        msg = f"V {V:.0f} alpha {al:+5.1f}: {r['status']}" + (f"  Cl {r['Cl']:+.3f} Cd {r['Cd']:.4f} Cm {r['Cm_cg']:+.3f}" if r["status"] == "solved" else f" {err[:90]}")
        if bar:
            bar.set_postfix_str(msg[:60]); bar.update(1)
        print("  " + msg)
        rs.save(path, doc)
    if bar:
        bar.close()
    solved = [r for r in doc["rows"] if r.get("status") == "solved"]
    installed = False
    if len(solved) >= 6 and not a.no_install and not a.full_cfd:
        shutil.copy(path, data)
        installed = True
        print(f"installed -> {data}  (phases 2 and 3 now use the CFD)")
    run = _write("cfd", {"run": {**meta("cfd", a), "quality": quality, "installed": installed, "solved": len(solved),
                                 "dash_solved": len(doc["dash_rows"])}, "results": {f"{name}_cfd": doc}}, name)
    for p in charts_cfd(run, doc):
        print("->", p)
    print("->", run)
    return 0 if ok == len(todo) else 1


def charts_cfd(run: Path, doc) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in doc.get("rows", []) if r.get("status") == "solved"]
    if len(rows) < 2:
        return []
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), facecolor="#fcfcfb")
    al = [r["alpha_deg"] for r in rows]
    for ax, k, km, lab in zip(axs, ("Cl", "Cd", "Cm_cg"), ("Cl", "Cd", "Cm"), ("CL", "CD", "Cm about the CG")):
        ax.plot(al, [r[k] for r in rows], color="#eda100", lw=2, marker="o", label="CFD")
        ax.plot(al, [r["model"][km] for r in rows], color="#52514e", lw=1.5, ls="--", label="lattice + hand")
        ax.set_title(f"{doc.get('aircraft', 'ASTOR')}: {lab} at {doc.get('speed_m_s', 30):.0f} m/s", fontsize=10, loc="left")
        ax.set_xlabel("angle of attack (deg)", fontsize=8); ax.grid(color="#e4e3df"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = run.with_name(run.stem + "_polar.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    return [out]


# ------------------------------------------------------------------------------------------------ workers (phases 2, 3)
def _job(job):
    kind = job[0]
    t = time.time()
    try:
        if kind == "stab":
            import astor_mujoco as SM
            out = SM.stability_tests(V=job[1], post_stall_cm_factor=job[2])
        elif kind == "card":
            import arigo_perf as P
            import astor_model as M
            import astor_mujoco as SM
            pack = job[1]
            sf = lambda bandit=False: SM.SimAstor(bandit=bandit, C=M.get(pack))
            C = M.get(pack)
            out = P.card("ASTOR", C, sf, SM.manoeuvre, M.CRUISE)
            out["aircraft"] = f"ASTOR, {pack}"
            out["fast_chases"] = {n: SM.fast_chase(sf, n) for n in SM.FAST_CHASES}
            for n, r in out["fast_chases"].items():
                out["summary"][f"{n}_time_to_5m_s"] = r["time_to_5m_s"]
                out["summary"][f"{n}_catch_rate"] = r["catch_rate"]
            out["requirement"] = M.requirement(C)
        elif kind == "robust":
            import astor_model as M
            import astor_mujoco as SM
            import raptors_robust as RB
            pack, turning, samples = job[1], job[2], job[3]
            sf = lambda bandit=False: SM.SimAstor(bandit=bandit, C=M.get(pack))
            out = RB.robustness(sf, M.get(pack), "rail", "parachute", n_launch=samples, n_land=max(4, samples // 4),
                                turning_station=turning)
        else:
            out = {"error": f"unknown job {kind}"}
    except Exception as exc:                                          # noqa: BLE001
        import traceback
        out = {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-800:]}
    out["seconds"] = round(time.time() - t, 1)
    return "|".join(str(x) for x in job), out


def _run_jobs(jobs, a, cache_name, desc, job_fn=None):
    job_fn = job_fn or _job
    import raptors_explore as RX
    OUT.mkdir(parents=True, exist_ok=True)
    cache_p = OUT / cache_name
    cache = rs.load(cache_p, a.fresh) or {"what": desc, "jobs": {}}
    key = lambda j: "|".join(str(x) for x in j)
    todo = [j for j in jobs if key(j) not in cache["jobs"] or "error" in cache["jobs"][key(j)]]
    n_req = a.cpu
    cpu, why = RX.safe_workers(a.cpu, a.force_cpu)
    if cpu < n_req:
        print(f"-j {n_req} lowered to {cpu} ({why}; ~1 GB per worker). --force-cpu to insist.")
    print(f"{desc}: {len(jobs) - len(todo)} jobs cached, {len(todo)} to run on {cpu} process(es)")
    bar = _bar(len(jobs), len(jobs) - len(todo), desc)

    def done(k, out):
        cache["jobs"][k] = _clean(out)
        rs.save(cache_p, cache)
        if bar:
            bar.set_postfix_str(k[:50]); bar.update(1)
        else:
            print("  done", k)
    if cpu > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(cpu, initializer=RX._init_worker, maxtasksperchild=4) as pool:
            for k, out in pool.imap_unordered(job_fn, todo, chunksize=1):
                done(k, out)
    else:
        for j in todo:
            done(*job_fn(j))
    if bar:
        bar.close()
    return {key(j): cache["jobs"][key(j)] for j in jobs}


# ------------------------------------------------------------------------------------------------ 2. stability
def phase_stability(a) -> int:
    import astor_mujoco as SM
    jobs = [("stab", V, k) for V in SM.V_TESTS for k in (1.0, 0.35)]
    res = _run_jobs(jobs, a, "astor_stability_cache.yaml", "astor stability")
    runs = {f"V{float(k.split('|')[1]):.0f}_cm{float(k.split('|')[2]):g}": v for k, v in res.items()}
    run = _write("stability", {"run": _meta("stability", a), "results": {"astor_mujoco": {"runs": runs}}})
    print(f"{'run':12s} {'SP zeta':>8s} {'DR zeta':>8s} {'spiral t2 (s)':>14s} {'stall loss (m)':>15s} {'recovers':>9s}")
    for k, r in runs.items():
        if "error" in r:
            print(f"{k:12s} ERROR {r['error']}")
            continue
        lm, st = r["linear_modes"], r.get("stall_and_release") or {}
        print(f"{k:12s} {lm['short_period']['zeta']:8.2f} {lm['dutch_roll']['zeta']:8.3f} {lm['spiral']['t_half_or_double_s']:14.1f} "
              f"{str(st.get('altitude_lost_m')):>15s} {str((r.get('verdict') or {}).get('recovers_from_stall_hands_off')):>9s}")
    print("->", run)
    return 0


# ------------------------------------------------------------------------------------------------ 3. missions
def phase_missions(a) -> int:
    import arigo_charts as AC
    import astor as A
    packs = a.packs or list(A.PACKS)
    jobs = [("card", pk) for pk in packs]
    jobs += [("robust", pk, True, a.samples) for pk in packs] + [("robust", A.DEFAULT_PACK, False, a.samples)]
    res = _run_jobs(jobs, a, "astor_missions_cache.yaml", "astor missions")
    import raptors_robust as RB
    cards = {}
    for pk in packs:
        c = res[f"card|{pk}"]
        if "error" in c:
            print("card", pk, "ERROR", c["error"], c.get("trace", "")[-300:])
            continue
        rb = res.get(f"robust|{pk}|True|{a.samples}") or {}
        if "error" not in rb and rb:
            c["robust"] = rb
            c["summary"].update(RB.summary(rb))
        if pk == A.DEFAULT_PACK:
            fx = res.get(f"robust|{pk}|False|{a.samples}") or {}
            if fx and "error" not in fx:
                c["robust_fixed_rail"] = fx
        cards[f"astor_{pk.replace(' ', '_')}"] = c
    MK = AC.mission_kpi(cards)
    items = AC.MISSIONS["astor"][1]
    table = {}
    for k, c in cards.items():
        s = c["summary"]
        table[k] = {"name": c["aircraft"], "score": MK[k]["astor"],
                    "gates": {g: bool(f(s.get(g))) for g, f in AC.GATES["astor"]},
                    "kpis": [{"kpi": lab, "value": s.get(key), "threshold": thr, "objective": obj, "score": MK[k]["per_metric"].get(key)}
                             for key, lab, thr, obj in items if key in s]}
    doc = {"run": {**_meta("missions", a), "packs": packs, "samples": a.samples}, "kpi": table,
           "results": {"astor_missions": {"cards": cards}}}
    run = _write("missions", doc)
    for p in charts_missions(run, doc):
        print("->", p)
    print("->", run)
    for k, t in table.items():
        s = cards[k]["summary"]
        print(f"  {t['name']:28s} score {t['score']:5.1f}  gates {t['gates']}  mass {s['mass_kg']:.2f} kg  top {s['max_speed_m_s']} m/s  "
              f"full power {s['full_power_min']} min  50 m/s chase {s.get('fast_50_time_to_5m_s')} s")
    return 0


def charts_missions(run: Path, doc) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    T = doc["kpi"]
    if not T:
        return []
    out = []
    cols = ["#eda100", "#2a78d6", "#1baf7a", "#eb6834"]
    labs = [x["kpi"] for x in next(iter(T.values()))["kpis"]]
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(labs) + 1.6), facecolor="#fcfcfb")
    y = np.arange(len(labs))
    h = 0.8 / len(T)
    for i, (k, t) in enumerate(T.items()):
        sc = {x["kpi"]: x["score"] or 0 for x in t["kpis"]}
        ax.barh(y - 0.4 + h * (i + 0.5), [sc.get(l, 0) for l in labs], height=h, color=cols[i % 4], label=f"{t['name']} ({t['score']:.0f})")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7); ax.invert_yaxis(); ax.set_xlim(0, 105)
    ax.set_xlabel("score (0 = threshold, 100 = objective)", fontsize=8); ax.grid(axis="x", color="#e4e3df"); ax.set_axisbelow(True)
    ax.set_title("ASTOR: KPI scores per pack", fontsize=11, loc="left"); ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    out.append(run.with_name(run.stem + "_kpi.png")); fig.savefig(out[-1], dpi=110); plt.close(fig)
    # launch and recovery success over wind (default pack): turning and fixed rail
    cards = doc["results"]["astor_missions"]["cards"]
    c = next((c for c in cards.values() if "robust_fixed_rail" in c), None)
    if c and "robust" in c:
        fig, ax = plt.subplots(figsize=(7.5, 4), facecolor="#fcfcfb")
        for rb, col, lab in ((c["robust"], "#eda100", "turning rail"), (c["robust_fixed_rail"], "#52514e", "fixed rail")):
            w = [p["wind_m_s"] for p in rb["per_wind"]]
            ax.plot(w, [100 * p["launch"]["rate"] for p in rb["per_wind"]], color=col, lw=2, marker="o", label=f"launch, {lab}")
        w = [p["wind_m_s"] for p in c["robust"]["per_wind"]]
        ax.plot(w, [100 * p["recovery"]["rate"] for p in c["robust"]["per_wind"]], color="#e34948", lw=2, ls="--", marker="s", label="parachute recovery")
        ax.axhline(95, color="#52514e", lw=1, ls=":")
        ax.set_xlabel("steady wind (m/s), gusts 1.5x", fontsize=8); ax.set_ylabel("success (%)", fontsize=8); ax.set_ylim(0, 105)
        ax.set_title("ASTOR: launch and recovery success over wind (dotted: 95 %)", fontsize=10, loc="left")
        ax.grid(color="#e4e3df"); ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        out.append(run.with_name(run.stem + "_wind.png")); fig.savefig(out[-1], dpi=110); plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ main
def main(argv=None, phase=None) -> int:
    ap = argparse.ArgumentParser(prog=f"astor_{phase or '<phase>'}.sh", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    if phase is None:
        ap.add_argument("phase", choices=("cfd", "stability", "missions"))
    ap.add_argument("-j", "--cpu", type=int, default=4, help="processes (cfd: OpenFOAM MPI; others: workers, capped to the machine)")
    ap.add_argument("--force-cpu", action="store_true", help="phases 2 and 3: use -j as given")
    ap.add_argument("--fresh", action="store_true", help="forget the phase's saved progress")
    ap.add_argument("--alphas", type=float, nargs="*", help="cfd: the angles of the 30 m/s sweep")
    ap.add_argument("--quality", default="screening", help="cfd: mesh quality (screening)")
    ap.add_argument("--full-cfd", action="store_true", help="cfd: the fine mesh (hours); kept apart, not installed")
    ap.add_argument("--no-install", action="store_true", help="cfd: do not install the result into designs/data")
    ap.add_argument("--timeout", type=float, default=3 * 3600.0, help="cfd: seconds per case")
    ap.add_argument("--workdir", type=Path, default=RUNS, help="cfd: the cases' folder")
    ap.add_argument("--packs", nargs="*", help="missions: pack names (default: every option in astor.PACKS)")
    ap.add_argument("--samples", type=int, default=24, help="missions: Monte Carlo launches per envelope")
    a = ap.parse_args(argv)
    phase = phase or a.phase
    if phase == "cfd":
        import astor_cfd as cf
        a.alphas = a.alphas or list(cf.ALPHAS)
        return phase_cfd(a)
    return phase_stability(a) if phase == "stability" else phase_missions(a)


if __name__ == "__main__":
    sys.exit(main())
