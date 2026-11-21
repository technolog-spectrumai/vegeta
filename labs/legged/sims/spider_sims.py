"""The spider simulations (``./spider.sh`` runs this): cases run in parallel on ``-j`` CPU cores, results written as
YAML — one file per simulation with every case and a summary, plus ``index.yaml``.

    python labs/legged/sims/spider_sims.py list
    python labs/legged/sims/spider_sims.py run all -j 8 [--quick] [--out runs/spider/<stamp>]
    python labs/legged/sims/spider_sims.py run arachne-climb salty-jump -j 4

Simulations:

* ``arachne-climb``     — MuJoCo dynamics of ARACHNE on a wall: gaits × surfaces × payload × adhesion scatter, plus a
  failed foot (``arachne_climb``);
* ``arachne-statics``   — Monte Carlo of the load sharing and attachment margins: surface × foot module × which feet
  are attached × heading, with foot stiffness, placement and payload scatter (``arachne_statics``);
* ``arachne-gaits``     — the gaits' kinematic checks: attached feet, adhesion margin, ankle tilt, IK, power and
  runtime per gait × surface × payload (``arachne_gaits``);
* ``spider-actuation``  — electric vs fluid-pressure joints over robot mass 0.1–3 kg (``spider_actuation``);
* ``salty-jump``        — SALTY's hydraulic catapult: charge pressure × bore × lever × line × gas volume
  (``salty_jump``).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
ROOT = LAB.parent.parent


def _paths():
    for p in (LAB / "designs", LAB / "common", HERE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


_paths()


# --------------------------------------------------------------------------------------------- YAML (no dependency)
def _scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return ".nan"
        if math.isinf(v):
            return ".inf" if v > 0 else "-.inf"
        return repr(round(v, 6))
    s = str(v)
    if s == "" or any(c in s for c in ":#{}[],&*!|>'\"%@`\n") or s.strip() != s or s.lower() in (
            "yes", "no", "true", "false", "null", "on", "off", "~"):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    return s


def to_yaml(obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return pad + "{}\n"
        out = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{_scalar(k)}:\n{to_yaml(v, indent + 1)}")
            else:
                out.append(f"{pad}{_scalar(k)}: {_scalar(v) if not isinstance(v, (dict, list)) else ('{}' if isinstance(v, dict) else '[]')}\n")
        return "".join(out)
    if isinstance(obj, (list, tuple)):
        if not obj:
            return pad + "[]\n"
        out = []
        for v in obj:
            if isinstance(v, (dict, list)) and v:
                body = to_yaml(v, indent + 1)
                out.append(f"{pad}- {body.lstrip()}")
            else:
                out.append(f"{pad}- {_scalar(v)}\n")
        return "".join(out)
    return pad + _scalar(obj) + "\n"


def _plain(x):
    """numpy scalars and tuples → plain Python, for the YAML."""
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return [_plain(v) for v in x.tolist()]
    except ImportError:
        pass
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


# --------------------------------------------------------------------------------------------- the simulations
def _climb_cases(quick):
    cases = []
    payloads = (20.0,) if quick else (0.0, 20.0, 40.0)
    seeds = (0,) if quick else (0, 1, 2)
    plan = [("climb", "glass"), ("climb_fast", "glass"), ("climb_careful", "varnished wood"), ("turn", "glass")]
    if not quick:
        plan += [("climb", "acrylic"), ("climb_fast", "acrylic"), ("climb", "varnished wood"),
                 ("climb_careful", "glass"), ("climb_careful", "bark"), ("crab", "bark"), ("climb_careful", "brick")]
    for (g, s), m, sd in itertools.product(plan, payloads, seeds):
        cases.append({"gait": g, "surface": s, "payload_g": m, "seed": sd, "adhesion_scatter": 0.0 if sd == 0 else 0.3,
                      "cycles": 1 if quick else 3})
    faults = [("climb", "glass", "L1"), ("climb", "glass", "R4"), ("climb_careful", "varnished wood", "L1")]
    for g, s, f in (faults[:1] if quick else faults):
        cases.append({"gait": g, "surface": s, "payload_g": 20.0, "seed": 0, "adhesion_scatter": 0.0,
                      "fault_foot": f, "cycles": 1 if quick else 3})
    if not quick:
        for ns in (0, 10):                                                # the contact model's sensitivity
            cases.append({"gait": "climb", "surface": "glass", "payload_g": 20.0, "seed": 0, "adhesion_scatter": 0.0,
                          "noslip": ns, "cycles": 3})
    return cases


def _climb_run(case):
    import arachne_climb as ac
    return ac.run(**case)


def _climb_summary(results):
    out = {}
    for r in results:
        if r.get("error"):
            continue
        k = f"{r['gait']} on {r['surface']}" + (f" (foot {r['fault_foot']} failed)" if r.get("fault_foot") else "") + \
            (f" (noslip {r['noslip']})" if r.get("noslip", 4) != 4 else "")
        e = out.setdefault(k, {"runs": 0, "falls": 0, "speed_mm_s": [], "slip_max_mm": 0.0, "detached_steps": 0,
                               "power_w": [], "planned_speed_mm_s": r.get("planned_speed_mm_s")})
        e["runs"] += 1
        e["falls"] += int(r["fell"])
        e["speed_mm_s"].append(r["speed_mm_s"])
        e["slip_max_mm"] = max(e["slip_max_mm"], r["slip_max_mm"])
        e["detached_steps"] += r["detached_steps"]
        e["power_w"].append(r["power_w"])
    for e in out.values():
        e["speed_mm_s"] = round(sum(e["speed_mm_s"]) / len(e["speed_mm_s"]), 2)
        e["power_w"] = round(sum(e["power_w"]) / len(e["power_w"]), 2)
        e["verdict"] = "FALLS" if e["falls"] else ("slips" if e["slip_max_mm"] > 5 or e["detached_steps"] > 100 else "holds")
    return out


def _statics_cases(quick):
    surf_mod = [("glass", "pad"), ("acrylic", "pad"), ("varnished wood", "pad"), ("sanded wood", "hybrid"),
                ("sanded wood", "spine"), ("painted wall", "hybrid"), ("bark", "spine"), ("brick", "spine")]
    attached = {"8": [], "7 (front foot up)": ["L1"], "7 (rear foot up)": ["R4"], "6 (wave pair up)": ["L1", "R3"],
                "4 (tetrapod)": ["R1", "L2", "R3", "L4"]}
    headings = (0.0, 90.0, 180.0)
    cases = []
    for (s, m), (name, up), h, tail in itertools.product(surf_mod, attached.items(), headings, (False, True)):
        if tail and m == "pad":
            continue
        if quick and (h == 90.0 or name.startswith("7 (rear")):
            continue
        cases.append({"surface": s, "module": m, "attached": name, "lifted": up, "heading_deg": h, "tail": tail,
                      "n": 60 if quick else 400})
    return cases


def _statics_run(case):
    import numpy as np
    import arachne_robot as ar
    import arachne_specs as sp
    import arachne_statics as st
    import motion_kit as mk
    kin = mk.Kin(ar.arachne())
    att = [f for f in sp.LEGS if f not in case["lifted"]]
    nominal = st.check(kin, att, surface=case["surface"], module=case["module"], heading_deg=case["heading_deg"],
                       tail=case["tail"])
    mc = st.monte_carlo(kin, att, n=case["n"], seed=1, surface=case["surface"], module=case["module"],
                        heading_deg=case["heading_deg"], tail=case["tail"])
    return {k: case[k] for k in ("surface", "module", "attached", "heading_deg", "tail")} | {
        "margin_nominal": round(float(nominal["margin"]), 3), "worst_foot": nominal["worst_foot"],
        "margin_p05": round(float(np.percentile(mc, 5)), 3), "margin_min": round(float(mc[0]), 3),
        "meets_rule_2": bool(np.percentile(mc, 5) >= 2.0), "draws": case["n"]}


def _statics_summary(results):
    best = {}
    for r in results:
        if r.get("error"):
            continue
        k = f"{r['surface']} / {r['module']}{' + tail' if r['tail'] else ''}"
        e = best.setdefault(k, {})
        e.setdefault(r["attached"], []).append(r["margin_p05"])
    return {k: {a: round(min(v), 3) for a, v in e.items()} for k, e in best.items()}


def _gaits_cases(quick):
    import arachne_gaits as ag
    cases = []
    for g, (_, _, _, _, _, surfaces) in ag.GAITS.items():
        surfs = list(surfaces) + ([] if quick else (["varnished wood"] if g == "climb" else []))
        for s in surfs:
            for m in ((20.0,) if quick else (0.0, 20.0, 40.0)):
                cases.append({"gait": g, "surface": s, "payload_g": m})
    return cases


def _gaits_run(case):
    import arachne_gaits as ag
    import arachne_robot as ar
    import motion_kit as mk
    kin = mk.Kin(ar.arachne(case["payload_g"]))
    m = ag.wall_gait(kin, case["gait"], surface=case["surface"])
    keep = ("speed_mm_s", "yaw_rate_deg_s", "power_w", "runtime_min", "servo_current_mean_a", "module", "tail")
    return dict(case, **{k: m.params[k] for k in keep if k in m.params}, **m.check, period_s=m.period)


def _gaits_summary(results):
    return {f"{r['gait']} on {r['surface']} ({r['payload_g']:.0f} g)": {
        "margin": r["adhesion_margin_min"], "attached_min": r["attached_min"], "runtime_min": r["runtime_min"],
        "ok": bool(r["adhesion_margin_min"] >= 2.0 and r["ankle_tilt_max_deg"] <= 18.0)}
        for r in results if not r.get("error")}


def _actuation_cases(quick):
    masses = (0.1, 0.18, 0.5, 1.5, 3.0) if quick else (0.1, 0.15, 0.18, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    return [{"mass_kg": m, "climbs": c} for m in masses for c in (True, False)]


def _actuation_run(case):
    import spider_actuation as sa
    m = case["mass_kg"]
    key = f"sweep {m:g} kg"
    sa.ROBOTS[key] = {"mass": m, "reach": 0.075 * (m / 0.18) ** (1 / 3), "joints": 24, "stance": 6,
                      "climbs": case["climbs"], "role": "isometric scaling of ARACHNE-2"}
    t = sa.compare(key)
    sup = sa.extension_support(key)
    feasible = [a for a in t if t[a]["ok"]]
    best = min(feasible, key=lambda a: t[a]["mass_g"]) if feasible else None
    return dict(case, reach_m=round(sa.ROBOTS[key]["reach"], 4), hold_nm=round(sa.joint_demand(key)["hold_nm"], 4),
                lightest=best, pressure_support=round(sup["fraction_of_hold"], 3),
                architectures={a: {"mass_g": round(v["mass_g"], 1), "share": round(v["share"], 3),
                                   "walk_w": round(v["walk_w"], 2), "hold_w": round(v["hold_w"], 2),
                                   "seals": v["seals"], "ok": v["ok"]} for a, v in t.items()})


def _actuation_summary(results):
    return {f"{r['mass_kg']:g} kg ({'climber' if r['climbs'] else 'walker'})": {
        "lightest": r["lightest"], "pressure_carries": r["pressure_support"],
        "spider_hybrid_extra_g": round(r["architectures"]["spider"]["mass_g"] - r["architectures"]["tendon"]["mass_g"], 1),
        "spider_hybrid_walk_power_saving": round(1 - r["architectures"]["spider"]["walk_w"] /
                                                 r["architectures"]["tendon"]["walk_w"], 3)
        if r["architectures"]["tendon"]["walk_w"] == r["architectures"]["tendon"]["walk_w"] else None}
        for r in results if not r.get("error")}


def _salty_cases(quick):
    if quick:
        grid = itertools.product((12.0, 16.0, 20.0), (10.0,), (7.5,), (2.0, 3.0), (4.0,), (1.0,))
    else:
        grid = itertools.product((8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0), (8.0, 10.0, 12.0), (5.0, 7.5, 10.0),
                                 (1.5, 2.0, 3.0, 4.0), (2.0, 4.0, 8.0), (1.0, 1.4))
    return [{"p_charge_bar": p, "bore_mm": b, "lever_ratio": l, "line_id_mm": d, "gas_cm3": g, "polytropic": n}
            for p, b, l, d, g, n in grid]


def _salty_run(case):
    import salty_jump as sj
    return dict(case, **sj.simulate(**case))


def _salty_summary(results):
    ok = [r for r in results if not r.get("error") and r["took_off"] and r["distance_m"] >= 1.0]
    out = {"cases": len(results), "reach_1m": len(ok)}
    if ok:
        low = min(ok, key=lambda r: (r["p_charge_bar"], -r["efficiency"]))
        eff = max(ok, key=lambda r: r["efficiency"])
        out["lowest_pressure_for_1m"] = {k: low[k] for k in ("p_charge_bar", "bore_mm", "lever_ratio", "line_id_mm",
                                                            "gas_cm3", "polytropic", "distance_m", "takeoff_s",
                                                            "efficiency", "power_peak_w", "recharge_s")}
        out["most_efficient_1m"] = {k: eff[k] for k in ("p_charge_bar", "bore_mm", "lever_ratio", "line_id_mm",
                                                       "gas_cm3", "polytropic", "distance_m", "efficiency")}
    far = max((r for r in results if not r.get("error")), key=lambda r: r["distance_m"])
    out["farthest"] = {k: far[k] for k in ("p_charge_bar", "bore_mm", "lever_ratio", "line_id_mm", "gas_cm3",
                                          "polytropic", "distance_m", "power_peak_w")}
    return out


SIMS = {
    "arachne-climb": (_climb_cases, _climb_run, _climb_summary, "MuJoCo dynamics of ARACHNE climbing a wall"),
    "arachne-statics": (_statics_cases, _statics_run, _statics_summary, "load sharing and attachment margins, "
                                                                        "Monte Carlo"),
    "arachne-gaits": (_gaits_cases, _gaits_run, _gaits_summary, "the gaits' kinematic checks"),
    "spider-actuation": (_actuation_cases, _actuation_run, _actuation_summary, "electric vs fluid-pressure joints "
                                                                               "over robot mass"),
    "salty-jump": (_salty_cases, _salty_run, _salty_summary, "SALTY's hydraulic catapult"),
}


def _worker(args):
    name, case = args
    _paths()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t = time.time()
    try:
        r = _plain(SIMS[name][1](dict(case)))
    except Exception as e:                                                # a failed case is a result, not a crash
        r = dict(case, error=f"{type(e).__name__}: {e}")
    r["wall_s"] = round(time.time() - t, 2)
    return r


def _git():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run(names, jobs, out, quick=False):
    out.mkdir(parents=True, exist_ok=True)
    meta = {"started": _dt.datetime.now().isoformat(timespec="seconds"), "jobs": jobs, "quick": quick,
            "host": platform.node(), "python": platform.python_version(), "vegeta_commit": _git()}
    index = {"meta": meta, "simulations": {}}
    ctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
    for name in names:
        cases_fn, _, summary_fn, what = SIMS[name]
        cases = cases_fn(quick)
        print(f"[{name}] {len(cases)} cases on {jobs} core(s) — {what}", flush=True)
        t0 = time.time()
        results = []
        with ctx.Pool(jobs) as pool:
            for i, r in enumerate(pool.imap(_worker, [(name, c) for c in cases]), 1):
                results.append(r)
                if i == len(cases) or i % max(1, len(cases) // 20) == 0:
                    print(f"  {i}/{len(cases)}  {time.time() - t0:6.1f} s", flush=True)
        errors = [r for r in results if r.get("error")]
        doc = {"simulation": name, "about": what, "meta": dict(meta, cases=len(cases), errors=len(errors),
                                                              wall_s=round(time.time() - t0, 1)),
               "summary": _plain(summary_fn(results)), "cases": results}
        path = out / f"{name}.yaml"
        path.write_text(to_yaml(doc))
        index["simulations"][name] = {"file": path.name, "cases": len(cases), "errors": len(errors),
                                      "wall_s": round(time.time() - t0, 1)}
        print(f"[{name}] -> {path}" + (f"  ({len(errors)} errors)" if errors else ""), flush=True)
    (out / "index.yaml").write_text(to_yaml(index))
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spider.sh", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the simulations and their case counts")
    r = sub.add_parser("run", help="run simulations")
    r.add_argument("sims", nargs="+", help=f"simulation names or 'all': {', '.join(SIMS)}")
    r.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="CPU cores to use")
    r.add_argument("--quick", action="store_true", help="a small grid (a few minutes) to check the setup")
    r.add_argument("--out", default=None, help="output folder (default runs/spider/<date-time>)")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for n, (cf, _, _, what) in SIMS.items():
            print(f"{n:18s} {len(cf(False)):5d} cases ({len(cf(True))} with --quick)  {what}")
        return 0
    names = list(SIMS) if "all" in a.sims else a.sims
    bad = [n for n in names if n not in SIMS]
    if bad:
        ap.error(f"unknown simulation(s) {bad}; choose from {list(SIMS)} or 'all'")
    out = Path(a.out) if a.out else ROOT / "runs" / "spider" / _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run(names, max(1, a.jobs), out, a.quick)
    print(f"done: {out}/index.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
