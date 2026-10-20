"""Validation of the tuned autopilot in the machine-learning sense: the tuner (``races.tune``) fits the settings on one
condition, the nominal lap, and this module scores them on held-out conditions it never flew — the generalisation
gap between the training score and the validation score.

**Held-out scenarios**, drawn per family from a seeded generator (``SEED``; the tuner's swarm seeds are 0 and 1 and
it never perturbs the world), ``n`` per family:

==============  ========================================================================================
nominal         the tuning condition itself (the training score: must reproduce ``autopilot.json``)
settings ±2 %   every knob × U(0.98, 1.02) — the tolerance of the tuned point (a knife edge fails here)
start ±0.5 m    the start moved by U(±0.5) m horizontally, U(±0.2) m vertically
mass ±5 %       the quad's mass and inertia × U(0.95, 1.05); the controller keeps the nominal values
thrust ±5 %     every rotor's thrust × U(0.95, 1.05) (motors, propellers, battery sag)
wind ≤ 2 m/s    a steady horizontal wind, a random heading, U(0, 2) m/s
combined        all of the above at once
==============  ========================================================================================

**Metrics per family:** the clean rate (finished with no collision — the target), the finish rate, the median and
p90 lap time of the finished laps, and the gap (median finished lap − the nominal lap). **Grade** per family on the
clean rate: ``pass`` ≥ 0.8, ``weak`` ≥ 0.5, else ``fail``; a course passes when every family passes. The thresholds
are a stated convention (``GRADES``), not a physical requirement."""
from __future__ import annotations

import numpy as np

SEED = 20261020
FAMILIES = ("nominal", "settings ±2 %", "start ±0.5 m", "mass ±5 %", "thrust ±5 %", "wind ≤ 2 m/s", "combined")
GRADES = (("pass", 0.8), ("weak", 0.5))


def _draw(family: str, rng) -> tuple[dict, dict]:
    """(factors on the settings, the world's scenario) for one held-out lap."""
    mult, sc = {}, {}
    every = family == "combined"
    if family == "settings ±2 %" or every:
        mult = {k: float(rng.uniform(0.98, 1.02)) for k in ("safety", "lookahead_s", "omega_pos", "omega_att", "tilt_max_deg", "v_cap")}
    if family == "start ±0.5 m" or every:
        sc["start_offset"] = [float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.2, 0.2))]
    if family == "mass ±5 %" or every:
        sc["mass_scale"] = float(rng.uniform(0.95, 1.05))
    if family == "thrust ±5 %" or every:
        sc["thrust_scale"] = float(rng.uniform(0.95, 1.05))
    if family == "wind ≤ 2 m/s" or every:
        a, w = float(rng.uniform(0.0, 2 * np.pi)), float(rng.uniform(0.0, 2.0))
        sc["wind"] = [w * np.cos(a), w * np.sin(a), 0.0]
    return mult, sc


def scenarios(n: int, seed: int = SEED, families=FAMILIES) -> list[dict]:
    """The held-out set: ``n`` laps per family (one for ``nominal``, which is deterministic), the same for a seed."""
    out = []
    for fi, fam in enumerate(FAMILIES):
        if fam not in families:
            continue
        rng = np.random.default_rng([seed, fi])                    # each family its own stream: adding one changes no other
        for i in range(1 if fam == "nominal" else n):
            mult, sc = _draw(fam, rng)
            out.append({"family": fam, "i": i, "mult": mult, "scenario": sc})
    return out


def apply(settings: dict, mult: dict) -> dict:
    s = dict(settings)
    for k, f in mult.items():
        if s.get(k) is not None:
            s[k] = round(s[k] * f, 4)
    return s


def _fly(job):
    slug, kw, sc = job
    from races import courses, sim
    try:
        return sim.run_mujoco(courses.COURSES[slug](), scenario=sc, **kw)["stats"]
    except Exception as e:                                         # a crash is a result
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}


def fly(slug: str, settings: dict, sc: list[dict], jobs: int = 1, control_dt: float = 0.004) -> list[dict]:
    """Every held-out lap of one course; rows in the order of ``sc`` (so ``-j`` changes nothing but the time)."""
    work = [(slug, {"control_dt": control_dt, **apply(settings, s["mult"])}, s["scenario"]) for s in sc]
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(jobs, len(work))) as ex:
            stats = list(ex.map(_fly, work))
    else:
        stats = [_fly(w) for w in work]
    rows = []
    for s, st in zip(sc, stats):
        fin = st.get("status") == "finished"
        rows.append({"family": s["family"], "i": s["i"], "status": st.get("status"), "reason": st.get("reason") or "",
                     "lap time [s]": float(st["lap time [s]"]) if fin else None, "collisions": int(st.get("collisions") or 0),
                     "zones passed": st.get("zones passed"), "mult": s["mult"], "scenario": s["scenario"]})
    return rows


def grade(clean_rate: float) -> str:
    for g, lo in GRADES:
        if clean_rate >= lo - 1e-12:
            return g
    return "fail"


def summarise(rows: list[dict]) -> dict:
    """Per family: n, clean and finish rates, median / p90 lap, the gap to the nominal lap, the grade."""
    nominal = next((r["lap time [s]"] for r in rows if r["family"] == "nominal"), None)
    fams = {}
    for fam in dict.fromkeys(r["family"] for r in rows):
        R = [r for r in rows if r["family"] == fam]
        laps = [r["lap time [s]"] for r in R if r["lap time [s]"] is not None]
        clean = sum(1 for r in R if r["lap time [s]"] is not None and r["collisions"] == 0) / len(R)
        med = float(np.median(laps)) if laps else None
        fams[fam] = {"n": len(R), "clean rate": round(clean, 3), "finish rate": round(len(laps) / len(R), 3),
                     "median lap [s]": round(med, 2) if med is not None else None,
                     "p90 lap [s]": round(float(np.percentile(laps, 90)), 2) if laps else None,
                     "gap [s]": round(med - nominal, 2) if med is not None and nominal is not None else None,
                     "grade": grade(clean)}
    held = [r for r in rows if r["family"] != "nominal"]
    overall = sum(1 for r in held if r["lap time [s]"] is not None and r["collisions"] == 0) / len(held) if held else None
    worst = min(fams.values(), key=lambda f: f["clean rate"]) if fams else None
    return {"nominal lap [s]": round(nominal, 2) if nominal is not None else None,
            "held-out clean rate": round(overall, 3) if overall is not None else None,
            "verdict": "pass" if fams and all(f["grade"] == "pass" for f in fams.values()) else ("fail" if any(f["grade"] == "fail" for f in fams.values()) else "weak"),
            "weakest family": next((k for k, v in fams.items() if v is worst), None),
            "families": fams}
