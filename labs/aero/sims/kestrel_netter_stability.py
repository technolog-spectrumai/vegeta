#!/usr/bin/env python3
"""Kestrel-4Q-2N — the flight checks that need no solver: static stability, hover authority, a rotor out, the hover
in a wind, cruise manoeuvrability, the transitions through the stall, and the net shot's recoil. Run by
``labs/aero/sims/kestrel.sh`` (``--stability``); no OpenFOAM, no CalculiX: the reduced-order models of notebooks 40–43.

    python3 labs/aero/sims/kestrel_netter_stability.py            # ~3-5 min: the tables and the transition figures
    python3 labs/aero/sims/kestrel_netter_stability.py --quick    # the tables and the nominal transitions only
    python3 labs/aero/sims/kestrel_netter_stability.py --matrix   # + the back transition's remedy matrix (~15 min)

What it computes (``kestrel_netter.vehicle_model``: the launcher's masses, 4Q-2's polar on the stretched pod, the
10x7 / 2814 drive):

* **static stability**: the neutral point with the pod and the propellers at cruise thrust, the static margin
  (``kestrel_flight.neutral_point``) — the launcher moves 1.6 kg into the pod, the battery trims the CG;
* **hover**: the attainable moments and angular accelerations with the launcher's inertia
  (``kestrel_control.hover_authority``), one rotor out (``motor_out``), the broadside hover in 2–10 m/s of wind
  (``hover_wind``);
* **cruise**: pitch / yaw / roll authority and the turn against airspeed (``cruise_manoeuvrability``);
* **transitions** (``kestrel_transition.simulate``): hover → cruise and cruise → hover at the 3 s schedule in still
  air, the back transition into a 6 m/s headwind and with the crow brake; with ``--matrix`` the remedy × schedule ×
  headwind table (``kestrel2.remedy_matrix``);
* **the net shot**: the recoil (``kestrel_netter.shot``) as a speed step: the back transition entered at the cruise
  speed less the recoil (the shot fired just before the pitch-up, the worst case for the stall), and the hover's
  sink after a shot fired straight up, caught by the rotors (thrust/weight).

Writes ``labs/aero/sims/output/kestrel_netter/``: ``stability.json`` (every table), the CSVs and the transition
figures. Exit 0 when the static margin is positive, hover thrust/weight ≥ 2 and every nominal transition passes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))

import numpy as np  # noqa: E402

OUT = ROOT / "labs" / "aero" / "sims" / "output" / "kestrel_netter"


def _f(x):
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    return x


def _df_rows(df) -> dict:
    return {str(k): _f(v) for k, v in df.to_dict(orient="index").items()}


def _cases(V: float, dv: float, quick: bool, airbrake: float) -> dict:
    cases = {("forward", "as is, still air"): dict(kind="forward"),
             ("back", "as is, still air"): dict(kind="back"),
             ("back", "headwind 6 m/s"): dict(kind="back", headwind=6.0),
             ("back", f"after a net shot (entered at {V - dv:.1f} m/s)"): dict(kind="back", V_entry=V - dv)}
    if not quick:
        cases[("back", f"crow brake ±{airbrake:g}°")] = dict(kind="back", airbrake_deg=airbrake)
        cases[("back", "after a net shot, crow brake")] = dict(kind="back", V_entry=V - dv, airbrake_deg=airbrake)
        cases[("forward", "headwind 6 m/s")] = dict(kind="forward", headwind=6.0)
    return cases


def units(*, quick: bool = False, matrix: bool = False) -> int:
    """The jobs the global bar counts: the tables (1), each transition, the remedy matrix (1)."""
    return 1 + len(_cases(16.0, 0.0, quick, 0.0)) + (1 if matrix else 0)


def main(argv=None, progress=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="the tables and the nominal transitions only")
    ap.add_argument("--matrix", action="store_true", help="also the back transition's remedy matrix (~15 min)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    import kestrel_control as kc
    import kestrel_flight as kf
    import kestrel_netter as kn
    import kestrel_transition as kt
    import kestrel2 as k2
    import pandas as pd

    report = {"aircraft": "Kestrel-4Q-2N", "source": "labs/aero/sims/kestrel_netter_stability.py", "methods": {}}
    fit = kn.checks()
    report["fit_checks"] = fit
    print("fit checks:", fit or "all hold")

    print("vehicle (the drive table is built with Boreas on the first run) ...")
    veh = kn.vehicle_model()
    c = kn.cg()
    report["vehicle"] = {"mass_kg": veh.mass, "x_cg_m": veh.x_cg, "z_cg_m": veh.z_cg, "Ixx": veh.Ixx, "Iyy": veh.Iyy, "Izz": c["Izz"],
                         "battery_x_m": c["battery_x"], "polar": _f(veh.pol.__dict__), "drive": kn.DRIVE_KEY}

    # ---- static stability
    V = veh.V_cruise
    D = k2.drag(veh, V)
    npt = kf.neutral_point(veh.p, V=V, thrust_each=D / 4, x_cg=veh.x_cg)
    report["static"] = _f(npt)
    report["methods"]["static"] = "kestrel_flight.neutral_point: the lattice's wings, the pod (Multhopp), the propellers' normal force at cruise thrust"
    print(f"static margin {npt['static_margin']:.3f} chords (neutral point {npt['x_np'] * 1e3:.0f} mm, CG {veh.x_cg * 1e3:.0f} mm)")

    # ---- hover
    T_hover = veh.weight / 4
    T_max = float(veh.drive.t_max(0.0))
    report["hover"] = {"thrust_each_N": T_hover, "T_max_N": T_max, "thrust_over_weight": 4 * T_max / veh.weight}
    inertia = np.diag([veh.Ixx, veh.Iyy, c["Izz"]])
    ha = kc.hover_authority(veh.p, mass_kg=veh.mass, cg=(veh.x_cg, veh.z_cg), drive=veh.drive, inertia=inertia)
    mo = kc.motor_out(veh.p, mass_kg=veh.mass, cg=(veh.x_cg, veh.z_cg), drive=veh.drive)
    hw = kc.hover_wind(veh.p, mass_kg=veh.mass, cg=(veh.x_cg, veh.z_cg), drive=veh.drive)
    report["hover"]["authority"] = _df_rows(ha)
    report["hover"]["motor_out"] = _df_rows(mo)
    report["hover"]["wind"] = _df_rows(hw)
    report["methods"]["hover"] = "kestrel_control: the rotors' allocation (differential thrust, reaction torque), the drive's thrust limits"
    for name, df in (("hover_authority", ha), ("motor_out", mo), ("hover_wind", hw)):
        df.to_csv(out / f"{name}.csv")
    print(f"hover thrust/weight {4 * T_max / veh.weight:.2f}")

    # ---- cruise
    cm = kc.cruise_manoeuvrability(veh)
    cm.to_csv(out / "cruise_manoeuvrability.csv")
    report["cruise"] = _df_rows(cm)
    report["methods"]["cruise"] = "kestrel_control.cruise_manoeuvrability: rotors' differential thrust and the elevons against airspeed"

    # ---- the net shot
    shot = kn.shot(veh.mass)
    dv = shot["recoil_m_s"]
    a_up = (4 * T_max - veh.weight) / veh.mass
    report["shot"] = {**_f({k: v for k, v in shot.items() if k != "shots"}), "shots": _f(shot["shots"]),
                      "hover_sink_after_a_shot_up_m": dv ** 2 / (2 * a_up), "hover_recovery_s": dv / a_up,
                      "cruise_speed_after_a_shot_m_s": V - dv, "stall_m_s": math.sqrt(2 * veh.weight / (kt.RHO * veh.L["S_ref"] * veh.pol.cl_max))}
    report["methods"]["shot"] = "kestrel_netter.shot: the cartridge's impulse on the aircraft (closed breech); the hover sink from the rotors' spare thrust"
    print(f"net shot: recoil {dv:.2f} m/s; hover sink {report['shot']['hover_sink_after_a_shot_up_m']:.2f} m; cruise after a shot {V - dv:.1f} m/s (stall {report['shot']['stall_m_s']:.1f})")

    # ---- transitions
    if progress:
        progress("tables")
    cases = _cases(V, dv, args.quick, k2.AIRBRAKE_DEG)
    rows, eps = {}, {}
    for (kind, name), kw in cases.items():
        kw = dict(kw)
        kind = kw.pop("kind")
        t1 = time.monotonic()
        ep = kt.simulate(veh, kind, t_tr=3.0, **kw)
        rows[(kind, name)] = dict(ep.metrics)
        eps[(kind, name)] = ep
        print(f"  {kind:8s} {name:45s} pass={ep.metrics.get('pass')}  ({time.monotonic() - t1:.0f} s)")
        if progress:
            progress(f"{kind} {name}")
    tdf = pd.DataFrame(rows).T
    tdf.index.names = ["direction", "case"]
    tdf.to_csv(out / "transitions.csv")
    report["transitions"] = {f"{k[0]} | {k[1]}": _f(v) for k, v in rows.items()}
    report["methods"]["transitions"] = "kestrel_transition.simulate (vegeta.aeromant.transitions): the reduced-order tailsitter model, 3 s pitch schedule"

    if args.matrix:
        print("remedy matrix ...")
        mat = k2.remedy_matrix(veh, progress=True)
        df = mat["table"] if isinstance(mat, dict) else mat[0]
        df.to_csv(out / "remedy_matrix.csv")
        report["remedy_matrix"] = {" | ".join(map(str, k)) if isinstance(k, tuple) else str(k): _f(v) for k, v in df.to_dict(orient="index").items()}
        if progress:
            progress("remedy matrix")

    if not args.no_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plot = getattr(kt, "plot_episode", None)
            for (kind, name), ep in eps.items():
                f = out / f"transition_{kind}_{name.split(' (')[0].replace(' ', '_').replace('/', '')}.png"
                if plot is not None:
                    fig = plot(ep, title=f"Kestrel-4Q-2N, {kind}: {name}")
                    fig.savefig(f, dpi=110)
                    plt.close(fig)
                else:
                    h = getattr(ep, "history", None) or getattr(ep, "log", None)
                    if h is None:
                        break
                    fig, ax = plt.subplots(figsize=(8, 4))
                    for key in ("alpha_deg", "theta_deg", "V", "z"):
                        if key in h:
                            ax.plot(h["t"], h[key], label=key)
                    ax.legend()
                    ax.set_title(f"Kestrel-4Q-2N, {kind}: {name}")
                    fig.savefig(f, dpi=110)
                    plt.close(fig)
        except Exception as e:                                # the figures are a convenience; the tables are the result
            report["figures"] = f"not drawn: {e!r}"

    ok = (npt["static_margin"] > 0 and 4 * T_max / veh.weight >= 2.0
          and all(bool(r.get("pass", False)) for (k, n), r in rows.items() if n == "as is, still air"))
    report["verdict"] = {"static_margin_positive": bool(npt["static_margin"] > 0), "hover_thrust_over_weight_ge_2": bool(4 * T_max / veh.weight >= 2.0),
                         "nominal_transitions_pass": bool(all(bool(r.get("pass", False)) for (k, n), r in rows.items() if n == "as is, still air")),
                         "ok": bool(ok), "elapsed_s": time.monotonic() - t0}
    (out / "stability.json").write_text(json.dumps(_f(report), indent=1, default=str) + "\n")
    print(f"written {out / 'stability.json'} ({time.monotonic() - t0:.0f} s); verdict {'ok' if ok else 'NOT ok'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
