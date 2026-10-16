"""prop_lab — one propeller alone in CFD, in the configurations Kestrel flies (notebook 38), and the calibration it
hands to the Kestrel notebooks.

Kestrel's airframe CFD runs its four propellers as **rotor disks** (blade-element sources fed with Boreas's blade rows
and section polar); its flight model runs on Boreas's BEMT drive tables and De Young's propeller normal force. All of
it rests on the generic thin-airfoil section Boreas assumes for a 9-inch blade. prop_lab measures the propeller
itself — the 9×6 resolved, blades and hub, in a rotating frame (Aeromant's ``rotor_mrf_static`` / ``rotor_mrf`` /
``rotor_mrf_oblique``) — at:

- the **hover** (static thrust at three speeds, the hover rpm among them),
- **axial inflow** 4–24 m/s (the cruise, the climb, the maximum-speed end: the advance-ratio sweep),
- **inflow at an angle** to the axis, 15–90° (the transition: the propeller at an angle of attack gives a normal
  force and a pitching moment of its own, and its thrust falls off differently from the axial case).

Each case is read beside the blade-element prediction at the same point. The **calibration** is two numbers for the
blade section — the lift-slope scale ``cl_scale`` and the drag scale ``cd_scale`` that make BEMT match the solved
axial and static cases in thrust and torque at once — and one for the normal force, ``cn_factor`` (the CFD's
normal-force slope over De Young's, over the oblique cases). ``export`` writes them with the whole case table to
``data/prop_lab.json``; ``kestrel_systems.propulsion`` applies the section scales (so the drive tables, ``rpm_for``
and the rotor disks' polar tables carry the measured propeller), ``kestrel_flight.prop_cn_alpha`` the normal-force
factor. A failed or diverged case is a result: listed, counted, left out of the fit.

**Noise** (``noise``, ``noise_table``): the blade-passing frequency and its harmonics from Gutin's steady-loading tones
on the thrust and torque the CFD measured (BEMT's where it did not), the broadband vortex noise (Hubbard), the
A-weighted total at a listener's distance and angle; ``render`` / ``render_sweep`` write the sound as WAV files — one
rotor in the hover, in cruise, at a transition point, the four rotors of the aircraft in the hover (their rpm split
by the allocation: the beats), and the transition as a glide from the hover rpm to the cruise rpm.

Frame: the rotor axis is +x, the inflow comes along +x (axial) or in the x–z plane at ``inflow_angle`` (oblique:
the cross-flow along +z); the CAD propeller (``cad``) is built about z and turned onto x (``stl``). Steady MRF: no
blade passing, no tip vortex; a steady frame averages the once-per-revolution loading of an inclined inflow.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import kestrel_systems as ks

PROP_LAB_JSON = Path(__file__).resolve().parent / "data" / "prop_lab.json"
RHO, NU = ks.RHO0, 1.46e-5
QUALITY = {
    "screening": dict(cells_per_diameter=6.0, surface_level=3, near_level=2, rotor_level=2, wake_level=1, iterations=400, residual_target=1e-4),
    "fine": dict(cells_per_diameter=8.0, surface_level=4, near_level=3, rotor_level=3, wake_level=2, iterations=800, residual_target=1e-5),
}
RPM_HOVER_REF = 4.15             # N: one rotor's share of the 1.69 kg aircraft in the hover (the rpm for it comes from Boreas)
T_CRUISE_REF = 0.8               # N: one rotor's cruise thrust at 16 m/s with round struts


# ------------------------------------------------------------------------------------------------- the propeller
def geometry() -> dict:
    """The 9×6's CAD numbers from ``kestrel_systems.COMPONENTS`` (mm): the planform Boreas uses, a 16 × 10 mm hub."""
    c = ks.COMPONENTS["propeller"]
    return dict(diameter=c["diameter_in"] * 25.4, pitch=c["pitch_in"] * 25.4, blades=c["blades"], hub_diameter=16.0, hub_height=10.0, bore=5.0,
                chord_root=c["chord_root_m"] * 1000, chord_max=c["chord_max_m"] * 1000, chord_tip=c["chord_tip_m"] * 1000,
                thickness=0.10, camber=0.04, stations=12)


def cad():
    """Dedalus's parametric propeller with the 9×6's numbers (axis z, blade 1 along +x)."""
    from vegeta.dedalus.examples import Propeller
    return Propeller().generate(**geometry())


def stl(workdir, tolerance: float = 0.05) -> Path:
    """The propeller turned onto the rotor frame's axis (+x) and written as STL (mm), once per workdir."""
    from vegeta import dedalus
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "prop_9x6_axis_x.stl"
    if not path.exists():
        g = cad()
        gx = dedalus.Geometry.from_cadquery(g.shape.rotate((0, 0, 0), (0, 1, 0), 90), name="prop_axis_x")
        gx.export_stl(str(path), tolerance=tolerance)
    return path


# ------------------------------------------------------------------------------------------------- blade element
def airfoil(calibrated: bool = False):
    """Boreas's section for the blade: the generic one, or the one prop_lab calibrated (``kestrel_systems.blade_airfoil``)."""
    return ks.blade_airfoil(calibrated=calibrated)


def bemt(rpm: float, V: float = 0.0, *, calibrated: bool = False):
    """Boreas's operating point of the 9×6 at ``rpm`` and axial inflow ``V``."""
    from vegeta import boreas
    sys_ = ks.propulsion()
    return boreas.bemt.solve(sys_.prop, airfoil(calibrated), rpm, V, RHO)


def rpm_hover() -> float:
    return ks.rpm_for(RPM_HOVER_REF, 0.0)


def rpm_cruise() -> float:
    return ks.rpm_for(T_CRUISE_REF, 16.0)


def deyoung_cn_alpha(thrust: float, V: float) -> float:
    """De Young's normal-force slope dC_N/dα on q·A, as the transition model uses it (``kestrel_flight.prop_cn_alpha``
    without prop_lab's factor)."""
    import kestrel_flight as kf
    return kf.prop_cn_alpha(thrust, V, calibrated=False)


# ------------------------------------------------------------------------------------------------- the configurations
def configurations(*, rpm_main: float = 8000.0, rpm_high: float = 10000.0) -> dict:
    """The cases by name: the hover at three speeds (the hover rpm among them), the axial sweep at ``rpm_main`` with
    two points at ``rpm_high`` and the cruise point at its own rpm, the oblique cases at ``rpm_main`` at two speeds."""
    out = {}
    r_h, r_c = rpm_hover(), rpm_cruise()
    for rpm, lab in ((5000.0, "5000 rpm"), (r_h, f"{r_h:.0f} rpm (the hover)"), (9000.0, "9000 rpm")):
        out[f"hover, {lab}"] = {"kind": "hover", "template": "rotor_mrf_static", "rpm": rpm, "airspeed": 0.0, "inflow_angle": 0.0}
    for V in (4.0, 8.0, 12.0, 16.0, 20.0, 24.0):
        out[f"axial, {V:g} m/s, {rpm_main:.0f} rpm"] = {"kind": "axial", "template": "rotor_mrf", "rpm": rpm_main, "airspeed": V, "inflow_angle": 0.0}
    for V in (12.0, 20.0):
        out[f"axial, {V:g} m/s, {rpm_high:.0f} rpm"] = {"kind": "axial", "template": "rotor_mrf", "rpm": rpm_high, "airspeed": V, "inflow_angle": 0.0}
    out[f"cruise, 16 m/s, {r_c:.0f} rpm"] = {"kind": "axial", "template": "rotor_mrf", "rpm": r_c, "airspeed": 16.0, "inflow_angle": 0.0}
    for V, angles in ((8.0, (15.0, 30.0, 45.0, 60.0, 90.0)), (14.0, (15.0, 30.0, 45.0, 60.0))):
        for a in angles:
            out[f"oblique, {V:g} m/s at {a:g}°, {rpm_main:.0f} rpm"] = {"kind": "oblique", "template": "rotor_mrf_oblique", "rpm": rpm_main, "airspeed": V, "inflow_angle": a}
    return out


def case(workdir, name: str, spec: dict, *, quality: str = "screening", env=None):
    """One Aeromant case (prepared, not run)."""
    from vegeta import aeromant
    pr = ks.propulsion().prop
    params = dict(rpm=spec["rpm"], diameter=pr.diameter, kinematic_viscosity=NU, density=RHO, rotation=1, **QUALITY[quality])
    if spec["template"] == "rotor_mrf":
        params["airspeed"] = spec["airspeed"]
    elif spec["template"] == "rotor_mrf_oblique":
        params.update(airspeed=spec["airspeed"], inflow_angle=spec["inflow_angle"])
    env = env or aeromant.OpenFOAMEnvironment.detect()
    tag = name.replace(", ", "_").replace(" ", "").replace("°", "deg").replace("(", "").replace(")", "")
    return aeromant.CFDCase(spec["template"], str(stl(Path(workdir) / "stl")), params, workdir=str(Path(workdir) / f"{tag}_{quality}"),
                            geometry_units="mm", environment=env)


def plan(workdir, specs: dict | None = None, *, quality: str = "screening", env=None) -> dict:
    specs = specs or configurations()
    return {n: case(workdir, n, s, quality=quality, env=env) for n, s in specs.items()}


# ------------------------------------------------------------------------------------------------- the table
SANE = 1e3                       # N, N·m: a thrust or torque beyond this from a 9-inch propeller is a diverged solver


def _note(r) -> str:
    msgs = [m for m in getattr(r, "messages", []) if m]
    return msgs[-1][:160] if msgs else ""


def table(results: dict | None, specs: dict | None = None) -> "pd.DataFrame":
    """One row per configuration: the point, the status (``solved`` / ``diverged`` / ``FAILED`` / ``NOT RUN``), the
    CFD's thrust, torque, shaft power, efficiency or figure of merit, normal force and pitching moment, beside the
    blade-element prediction at the same point (thrust, torque, power; De Young's C_Nα for the oblique cases) and
    the ratios. A failure is a result."""
    import pandas as pd
    specs = specs or configurations()
    results = results or {}
    pr = ks.propulsion().prop
    D = pr.diameter
    rows = {}
    for name, s in specs.items():
        V, a = s["airspeed"], math.radians(s["inflow_angle"])
        v_ax = V * math.cos(a)
        op = bemt(s["rpm"], v_ax)
        n = s["rpm"] / 60
        q = 0.5 * RHO * max(V, 0.1) ** 2
        A = math.pi * (D / 2) ** 2
        row = {"kind": s["kind"], "template": s["template"], "rpm": s["rpm"], "V [m/s]": V, "angle [deg]": s["inflow_angle"], "J": v_ax / (n * D),
               "Re (0.75 R)": 0.75 * D / 2 * 2 * math.pi * n * 0.9 * pr.chord[len(pr.chord) * 3 // 4] / NU,
               "BEMT T [N]": op.thrust, "BEMT Q [N·m]": op.torque, "BEMT P [W]": op.power,
               "BEMT η / FM": op.efficiency if V > 0 else op.figure_of_merit,
               "De Young CNα [1/rad]": deyoung_cn_alpha(op.thrust, V) if s["kind"] == "oblique" else np.nan}
        r = results.get(name)
        if r is None or not getattr(r, "ok", False):
            msg = "NOT RUN" if r is None or any("NOT RUN" in m for m in getattr(r, "messages", [])) else "FAILED"
            rows[name] = dict(row, status=msg, note="" if r is None else _note(r))
            continue
        m = r.metrics
        T, Q, P = (np.nan if m.get(k) is None else float(m[k]) for k in ("thrust_N", "torque_Nm", "power_W"))
        sane = all(np.isfinite(x) for x in (T, Q)) and abs(T) <= SANE and abs(Q) <= SANE
        note = "" if sane else f"diverged: |T| {abs(T):.3g} N, |Q| {abs(Q):.3g} N·m"
        if sane and m.get("converged") is False:
            note = f"not converged in {m.get('iterations')} iterations"
        N = float(m["normal_force_N"]) if m.get("normal_force_N") is not None else np.nan
        My = float(m["pitching_moment_Nm"]) if m.get("pitching_moment_Nm") is not None else np.nan
        cn_alpha = N / (q * A) / math.sin(a) if s["kind"] == "oblique" and a > 0 and np.isfinite(N) else np.nan
        rows[name] = dict(row, status="solved" if sane else "diverged", **{"CFD T [N]": T, "CFD Q [N·m]": Q, "CFD P [W]": P,
                          "CFD η / FM": m.get("efficiency") if V > 0 else m.get("figure_of_merit"),
                          "CFD N [N]": N, "CFD My [N·m]": My, "CFD CNα [1/rad]": cn_alpha,
                          "T CFD / BEMT": T / op.thrust if op.thrust else np.nan, "Q CFD / BEMT": Q / op.torque if op.torque else np.nan,
                          "CNα CFD / De Young": cn_alpha / row["De Young CNα [1/rad]"] if s["kind"] == "oblique" and row["De Young CNα [1/rad]"] else np.nan,
                          "thrust scatter": m.get("thrust_std_lastN"), "iterations": m.get("iterations"), "converged": m.get("converged"),
                          "cells": m.get("mesh_cells")}, note=note)
    return pd.DataFrame(rows).T


def summary(tab) -> dict:
    st = tab["status"].astype(str)
    out = {k: int((st == k).sum()) for k in ("solved", "diverged", "FAILED", "NOT RUN")}
    out["n"] = len(st)
    out["by kind"] = {k: int(((st == "solved") & (tab["kind"] == k)).sum()) for k in ("hover", "axial", "oblique")}
    out["line"] = (f"{out['solved']} of {out['n']} cases solved (hover {out['by kind']['hover']}, axial {out['by kind']['axial']}, "
                   f"oblique {out['by kind']['oblique']}), {out['diverged']} diverged, {out['FAILED']} failed, {out['NOT RUN']} not run")
    return out


# ------------------------------------------------------------------------------------------------- the calibration
def fit_section(tab, *, kinds=("hover", "axial")) -> dict:
    """The two scales of the blade section — ``cl_scale`` on the lift slope, ``cd_scale`` on the profile drag — that make
    BEMT reproduce the solved hover and axial cases' thrust and torque together (least squares on the log ratios).
    ``None`` when fewer than two cases solved."""
    from scipy.optimize import least_squares
    from vegeta import boreas
    sel = tab[(tab["status"] == "solved") & tab["kind"].isin(kinds)] if "CFD T [N]" in tab.columns else tab.iloc[0:0]
    if len(sel) < 2:
        return {"cl_scale": None, "cd_scale": None, "n_cases": int(len(sel)), "rms_log_ratio_before": None, "rms_log_ratio_after": None}
    sys_ = ks.propulsion()
    base = ks.blade_airfoil(calibrated=False)
    pts = [(float(r["rpm"]), float(r["V [m/s]"]) * math.cos(math.radians(float(r["angle [deg]"]))), float(r["CFD T [N]"]), float(r["CFD Q [N·m]"]))
           for _, r in sel.iterrows()]

    def resid(x):
        af = boreas.Airfoil(name=base.name, cl_alpha=base.cl_alpha * x[0], alpha0_deg=base.alpha0_deg, cl_max=base.cl_max, cd0=base.cd0 * x[1], k=base.k)
        out = []
        for rpm, v, T, Q in pts:
            op = boreas.bemt.solve(sys_.prop, af, rpm, v, RHO)
            out += [math.log(max(T, 1e-6) / max(op.thrust, 1e-6)), math.log(max(Q, 1e-9) / max(op.torque, 1e-9))]
        return np.array(out)

    r0 = resid([1.0, 1.0])
    sol = least_squares(resid, [1.0, 1.0], bounds=([0.3, 0.1], [2.0, 5.0]))
    return {"cl_scale": float(sol.x[0]), "cd_scale": float(sol.x[1]), "n_cases": len(pts),
            "rms_log_ratio_before": float(np.sqrt(np.mean(r0 ** 2))), "rms_log_ratio_after": float(np.sqrt(np.mean(sol.fun ** 2)))}


def fit_normal_force(tab) -> dict:
    """``cn_factor``: the median of the CFD's normal-force slope over De Young's across the solved oblique cases
    (``None`` without any), and the table of C_Nα against the angle and the advance ratio."""
    if "CNα CFD / De Young" not in tab.columns:
        return {"cn_factor": None, "n_cases": 0, "points": []}
    sel = tab[(tab["status"] == "solved") & (tab["kind"] == "oblique")]
    ratios = [float(v) for v in sel["CNα CFD / De Young"].astype(float) if np.isfinite(v)]
    rows = [{"angle_deg": float(r["angle [deg]"]), "V": float(r["V [m/s]"]), "J": float(r["J"]), "cn_alpha_cfd": float(r["CFD CNα [1/rad]"]),
             "cn_alpha_deyoung": float(r["De Young CNα [1/rad]"]), "thrust_ratio": float(r["T CFD / BEMT"])} for _, r in sel.iterrows()]
    return {"cn_factor": float(np.median(ratios)) if ratios else None, "n_cases": len(ratios), "points": rows}


def export(tab, path=PROP_LAB_JSON, *, meta: dict | None = None, keep_previous: bool = True) -> dict:
    """The calibration and the whole table to ``data/prop_lab.json``. A pass that solved nothing keeps a previous
    calibration (the file's statuses are refreshed under ``last_listing``)."""
    sec, nf = fit_section(tab), fit_normal_force(tab)
    rows = {str(k): {kk: (None if vv is None or (isinstance(vv, float) and not np.isfinite(vv)) else (vv.item() if hasattr(vv, "item") else vv))
                     for kk, vv in r.items()} for k, r in tab.to_dict("index").items()}
    nt = noise_table(tab)
    noise_rows = {str(k): {kk: (vv.item() if hasattr(vv, "item") else vv) for kk, vv in r.items()} for k, r in nt.to_dict("index").items()}
    doc = {"meta": meta or {}, "summary": {k: v for k, v in summary(tab).items() if k != "line"},
           "calibration": {"cl_scale": sec["cl_scale"], "cd_scale": sec["cd_scale"], "cn_factor": nf["cn_factor"],
                           "section_fit": sec, "normal_force_fit": nf}, "cases": rows, "noise": {"listener": LISTENER, "rows": noise_rows}}
    path = Path(path)
    prev = load(path)
    if keep_previous and sec["cl_scale"] is None and nf["cn_factor"] is None and prev and prev.get("calibration", {}).get("cl_scale") is not None:
        prev["last_listing"] = {"meta": meta or {}, "summary": doc["summary"]}
        path.write_text(json.dumps(prev, indent=1, default=str))
        return prev
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=str))
    return doc


def load(path=PROP_LAB_JSON) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def calibration(path=PROP_LAB_JSON) -> dict:
    """What Kestrel applies: ``{"cl_scale", "cd_scale", "cn_factor", "on_file": bool, "meta"}`` — unit scales and
    ``on_file`` False when there is no calibration yet."""
    doc = load(path)
    cal = (doc or {}).get("calibration", {})
    on = cal.get("cl_scale") is not None or cal.get("cn_factor") is not None
    return {"cl_scale": cal.get("cl_scale") or 1.0, "cd_scale": cal.get("cd_scale") or 1.0, "cn_factor": cal.get("cn_factor") or 1.0,
            "on_file": on, "meta": (doc or {}).get("meta", {}), "n_solved": (doc or {}).get("summary", {}).get("solved", 0)}


# ------------------------------------------------------------------------------------------------- noise and sound
LISTENER = {"distance_m": 10.0, "angle_deg": 60.0}      # 10 m off, 60° from the axis: a bystander watching it hover
RATE = 44100


def noise(rpm: float, V: float, thrust: float, torque: float, *, distance: float | None = None, angle_deg: float | None = None,
          harmonics: int = 8) -> dict:
    """One propeller's noise at a listener: Gutin's tones of the steady loading (the blade-passing frequency B·n and
    its harmonics; ``thrust`` and ``torque`` as measured or predicted), the broadband vortex noise with its peak
    frequency, the totals in dB and dB(A). Steady loading only: the once-per-revolution load cycle of an inclined
    inflow adds loading harmonics the estimate does not carry."""
    from vegeta.boreas import noise as bn
    d = LISTENER["distance_m"] if distance is None else distance
    a = LISTENER["angle_deg"] if angle_deg is None else angle_deg
    pr = ks.propulsion().prop
    g = bn.gutin_harmonics(pr, max(thrust, 0.0), max(torque, 0.0), rpm, d, a, harmonics=harmonics)
    v = bn.vortex_noise(pr, rpm, d, V)
    tones = [(float(f), float(L)) for f, L in zip(g["frequency_hz"], g["spl_db"])]
    levels = [L for _, L in tones] + [v["spl_db"]]
    weights = bn.a_weighting(np.array([f for f, _ in tones] + [v["peak_hz"]]))
    total = bn.add_levels(levels)
    total_a = bn.add_levels([L + float(w) for L, w in zip(levels, weights)])
    return {"rpm": rpm, "V": V, "thrust": thrust, "torque": torque, "blade_pass_hz": g["blade_pass_hz"], "tones": tones,
            "tonal_db": g["total_tonal_db"], "broadband_db": v["spl_db"], "broadband_peak_hz": v["peak_hz"],
            "band": (max(0.3 * v["peak_hz"], 200.0), min(3.0 * v["peak_hz"], 0.45 * RATE)), "total_db": total, "total_dba": total_a,
            "distance_m": d, "angle_deg": a}


def noise_table(tab) -> "pd.DataFrame":
    """One row per configuration: the blade-passing frequency, the first three tones, the broadband, the totals —
    on the CFD's thrust and torque where the case solved, on BEMT's otherwise (the ``source`` column says which)."""
    import pandas as pd
    rows = {}
    for name, r in tab.iterrows():
        solved = str(r.get("status")) == "solved" and "CFD T [N]" in tab.columns and np.isfinite(float(r["CFD T [N]"]))
        T = float(r["CFD T [N]"]) if solved else float(r["BEMT T [N]"])
        Q = float(r["CFD Q [N·m]"]) if solved else float(r["BEMT Q [N·m]"])
        nz = noise(float(r["rpm"]), float(r["V [m/s]"]), T, Q)
        rows[name] = {"source": "CFD" if solved else "BEMT", "rpm": r["rpm"], "V [m/s]": r["V [m/s]"], "angle [deg]": r["angle [deg]"],
                      "BPF [Hz]": nz["blade_pass_hz"], "tone 1 [dB]": nz["tones"][0][1], "tone 2 [dB]": nz["tones"][1][1], "tone 3 [dB]": nz["tones"][2][1],
                      "tonal [dB]": nz["tonal_db"], "broadband [dB]": nz["broadband_db"], "broadband peak [Hz]": nz["broadband_peak_hz"],
                      "total [dB]": nz["total_db"], "total [dB(A)]": nz["total_dba"]}
    return pd.DataFrame(rows).T


def _synth(tones, broadband_db, band, *, seconds, seed, p_ref=20e-6):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    x = np.zeros_like(t)
    for f, L in tones:
        if 0 < f < RATE / 2 and np.isfinite(L):
            x += math.sqrt(2) * p_ref * 10 ** (L / 20) * np.sin(2 * math.pi * f * t + rng.uniform(0, 2 * math.pi))
    if np.isfinite(broadband_db):
        n = rng.standard_normal(len(t))
        F = np.fft.rfft(n)
        fr = np.fft.rfftfreq(len(t), 1 / RATE)
        F[(fr < band[0]) | (fr > band[1])] = 0
        nb = np.fft.irfft(F, len(t))
        nb *= p_ref * 10 ** (broadband_db / 20) / max(float(np.std(nb)), 1e-30)
        x += nb
    return x


def _fade(x, seconds: float = 0.05):
    t = np.arange(len(x)) / RATE
    return x * np.minimum(1.0, np.minimum(t, t[-1] - t) / seconds)


def signal(nz: dict, *, seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """A sound-pressure signal [Pa] of one propeller's noise (``noise``): the tones at their levels, the broadband as
    band-passed noise around its peak."""
    return _fade(_synth(nz["tones"], nz["broadband_db"], nz["band"], seconds=seconds, seed=seed))


def signal_rotors(nzs: list, *, seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """Several rotors heard together (each at its own rpm: the beats between them), their pressures added."""
    return _fade(sum(_synth(nz["tones"], nz["broadband_db"], nz["band"], seconds=seconds, seed=seed + i) for i, nz in enumerate(nzs)))


def signal_sweep(points: list, *, seconds: float = 6.0, seed: int = 0, p_ref: float = 20e-6) -> np.ndarray:
    """The sound of a glide through ``points`` (``noise`` dicts, spread evenly over ``seconds``): every tone's
    frequency and level interpolated in time, the phase integrated so the pitch slides without a click; the
    broadband's level interpolated too."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    knots = np.linspace(0, seconds, len(points))
    x = np.zeros_like(t)
    n_h = min(len(p["tones"]) for p in points)
    for h in range(n_h):
        f = np.interp(t, knots, [p["tones"][h][0] for p in points])
        L = np.interp(t, knots, [p["tones"][h][1] for p in points])
        phase = 2 * math.pi * np.cumsum(f) / RATE + rng.uniform(0, 2 * math.pi)
        x += math.sqrt(2) * p_ref * 10 ** (L / 20) * np.sin(phase)
    Lb = np.interp(t, knots, [p["broadband_db"] for p in points])
    band = (min(p["band"][0] for p in points), max(p["band"][1] for p in points))
    n = rng.standard_normal(len(t))
    F = np.fft.rfft(n)
    fr = np.fft.rfftfreq(len(t), 1 / RATE)
    F[(fr < band[0]) | (fr > band[1])] = 0
    nb = np.fft.irfft(F, len(t))
    nb /= max(float(np.std(nb)), 1e-30)
    x += nb * p_ref * 10 ** (Lb / 20)
    return _fade(x)


def write_wav(path, x: np.ndarray, full_scale_pa: float) -> Path:
    """16-bit mono WAV at ``RATE``; one ``full_scale_pa`` for every file of a comparison, so the louder one is louder."""
    import wave
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(x) / full_scale_pa, -1.0, 1.0)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(RATE)
        f.writeframes((pcm * 32767).astype("<i2").tobytes())
    return path


def hover_rotor_rpms(mass_kg: float = 1.686, split: float = 0.015) -> list:
    """The four rotors' speeds in a steady hover with a small differential (``split`` of the thrust on the diagonal
    pairs, the heading held against a disturbance): what makes four rotors beat rather than hum."""
    T = mass_kg * ks.G / 4
    return [ks.rpm_for(T * (1 + split), 0.0), ks.rpm_for(T * (1 - split), 0.0), ks.rpm_for(T * (1 - split), 0.0), ks.rpm_for(T * (1 + split), 0.0)]


def render(out_dir, tab, *, mass_kg: float = 1.686, seconds: float = 3.0) -> dict:
    """The WAV files: one rotor in the hover (at the hover rpm), in cruise, at a transition point (8 m/s at 45°);
    the four rotors in the hover (their rpm split: the beats); the transition as a glide from the hover to the cruise
    through the oblique points. The thrust and torque from the CFD where a matching case solved, else from BEMT.
    Returns ``{name: {"path", "total_db", "total_dba", "seconds"}}``, every file at one full scale."""
    out_dir = Path(out_dir)
    def pick(kind, **sel):
        rows = tab[tab["kind"] == kind]
        for k, v in sel.items():
            rows = rows[np.isclose(rows[k].astype(float), v)]
        if len(rows) == 0:
            return None
        r = rows.iloc[0]
        solved = str(r["status"]) == "solved" and "CFD T [N]" in tab.columns and np.isfinite(float(r["CFD T [N]"]))
        return noise(float(r["rpm"]), float(r["V [m/s]"]), float(r["CFD T [N]"] if solved else r["BEMT T [N]"]), float(r["CFD Q [N·m]"] if solved else r["BEMT Q [N·m]"]))
    r_h, r_c = rpm_hover(), rpm_cruise()
    hov = pick("hover", rpm=r_h) or noise(r_h, 0.0, bemt(r_h).thrust, bemt(r_h).torque)
    cru = pick("axial", rpm=r_c) or noise(r_c, 16.0, bemt(r_c, 16.0).thrust, bemt(r_c, 16.0).torque)
    tr = pick("oblique", **{"V [m/s]": 8.0, "angle [deg]": 45.0}) or noise(8000.0, 8.0, bemt(8000.0, 8.0 * math.cos(math.pi / 4)).thrust, bemt(8000.0, 8.0 * math.cos(math.pi / 4)).torque)
    four = [noise(r, 0.0, bemt(r).thrust, bemt(r).torque) for r in hover_rotor_rpms(mass_kg)]
    glide = [hov] + [pick("oblique", **{"V [m/s]": 8.0, "angle [deg]": a}) or noise(8000.0, 8.0, bemt(8000.0, 8.0 * math.cos(math.radians(a))).thrust, bemt(8000.0, 8.0 * math.cos(math.radians(a))).torque)
                     for a in (60.0, 45.0, 30.0, 15.0)] + [cru]
    sigs = {"one_rotor_hover": signal(hov, seconds=seconds, seed=1), "one_rotor_cruise_16ms": signal(cru, seconds=seconds, seed=2),
            "one_rotor_transition_8ms_45deg": signal(tr, seconds=seconds, seed=3), "four_rotors_hover": signal_rotors(four, seconds=seconds, seed=4),
            "transition_hover_to_cruise": signal_sweep(glide, seconds=2 * seconds, seed=5)}
    full = 1.2 * max(float(np.abs(x).max()) for x in sigs.values())
    out = {}
    for name, x in sigs.items():
        path = write_wav(out_dir / f"{name}.wav", x, full)
        rms = float(np.sqrt(np.mean(x ** 2)))
        from vegeta.boreas import noise as bn
        out[name] = {"path": str(path), "seconds": len(x) / RATE, "rms_db": bn.spl(rms), "full_scale_pa": full}
    out["one_rotor_hover"].update(total_db=hov["total_db"], total_dba=hov["total_dba"])
    out["one_rotor_cruise_16ms"].update(total_db=cru["total_db"], total_dba=cru["total_dba"])
    out["four_rotors_hover"].update(total_db=float(10 * math.log10(sum(10 ** (nz["total_db"] / 10) for nz in four))),
                                    rpms=[nz["rpm"] for nz in four], beat_hz=[abs(four[0]["blade_pass_hz"] - four[1]["blade_pass_hz"])])
    return out
