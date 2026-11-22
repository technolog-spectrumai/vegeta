"""Propeller sound of a multirotor at several flight speeds, from CFD: one rotor alone and all the machine's rotors.

For each speed (km/h): trim the machine (the body tilts forward until the rotors' thrust carries the weight and the
body's drag), give each rotor its share of the thrust, solve one rotor in OpenFOAM with its blades resolved (Aeromant's
``rotor_mrf_static`` in the hover, ``rotor_mrf_oblique`` in forward flight: the air meets the tilted disc at 90° − tilt),
correct the rpm once to the thrust the trim needs (thrust ~ rpm²) and solve again, then build the sound from the CFD's
thrust and torque with ``rotor_noise``: steady and unsteady loading tones (Lowson; the arm under the rotor, the edgewise
asymmetry in forward flight, ingested turbulence) and BPM trailing-edge broadband in 1/3 octaves, at a
listener ``--distance`` m away and ``--angle`` degrees from the rotor axis. The machine's sound adds its rotors with
their small rpm differences (the beating every multirotor has) and random phases.

Writes to ``--out``: ``<machine>_<kmh>kmh_1prop.wav``, ``<machine>_<kmh>kmh_<n>props.wav`` (all files of one run on
one common scale: louder is louder), ``<machine>_sound.csv`` / ``.json`` (rpm, thrust, torque, levels, the source of
each point: CFD or the blade-element fallback). Re-running reuses solved cases (the CFD folders under ``--out/cfd``).

Machines (dev_arigo's designs): ``quad`` = Drongo (250 mm quad, 5x4.3 tri-blade, notebook 08), ``hexa`` = Drako (heavy-lift
hexacopter, T-Motor 30x10.5). ``--no-cfd`` skips OpenFOAM and uses blade-element theory (a quick dry run).
"""
from __future__ import annotations

import argparse, csv, json, math, sys, time, wave
from pathlib import Path

import numpy as np

RHO, NU, G, RATE = 1.225, 1.46e-5, 9.81, 44100
QUALITY = {   # as prop_lab (labs/aero/designs/prop_lab.py): coarser meshes than these gave unusable propeller numbers
    "screening": dict(cells_per_diameter=10.0, surface_level=4, near_level=3, rotor_level=3, wake_level=2, iterations=600, residual_target=1e-4),
    "fine": dict(cells_per_diameter=14.0, surface_level=5, near_level=4, rotor_level=4, wake_level=3, iterations=1000, residual_target=1e-5),
    "smoke": dict(cells_per_diameter=6.0, surface_level=3, near_level=2, rotor_level=2, wake_level=1, iterations=60, residual_target=1e-4),
}


def machine(name: str) -> dict:
    """Mass, rotor count, drag area, the Boreas propeller and section, and the CAD numbers of the same blade."""
    if name == "quad":
        import drongo_robot as dr
        prop, airfoil, _, _ = dr.boreas_system()
        mass = sum(dr.mass_budget(cad=dr.CAD).values()) / 1000.0
        cad = dict(hub_diameter=12.0, hub_height=6.0, bore=5.0, thickness=0.10, camber=0.04, stations=10)
        return dict(name="drongo_quad", n=4, mass=mass, cda=dr.DRAG_AREA_M2, prop=prop, airfoil=airfoil, cad=cad,
                    pitch_m=0.0254 * 4.3, chords=(0.010, 0.016, 0.006),
                    noise=dict(arm_width_m=0.010, arm_gap_m=0.008), section="naca4409")   # thin moulded blade        # 10 mm carbon arm, motor bell gap
    if name == "hexa":
        import drako_robot as dk
        from drako import Drako
        prop, airfoil, _, _ = dk.boreas_system()
        p = dk.design_params()
        s = Drako.sizing(p)
        k = dk.CHORD_FIT
        cad = dict(hub_diameter=40.0, hub_height=16.0, bore=10.0, thickness=0.09, camber=0.04, stations=10)
        return dict(name="drako_hexa", n=6, mass=s["tow_design_kg"], cda=0.25 + 0.15, prop=prop, airfoil=airfoil, cad=cad,
                    pitch_m=0.0254 * 10.5, chords=(0.045 * k, 0.065 * k, 0.025 * k),
                    noise=dict(arm_width_m=0.030, arm_gap_m=0.040), section="clarky")     # carbon heavy-lift blade        # 30 mm arm tube, 40 mm below the disc
    raise SystemExit(f"unknown machine {name!r}: quad or hexa")


def trim(m: dict, V: float) -> dict:
    """Level flight at V: tilt so that thrust balances weight and drag; thrust per rotor; the inflow's angle to the axis."""
    W, D = m["mass"] * G, 0.5 * RHO * V * V * m["cda"]
    tilt = math.atan2(D, W)
    return {"tilt_deg": math.degrees(tilt), "thrust_per_rotor_N": math.hypot(W, D) / m["n"],
            "inflow_angle_deg": 90.0 - math.degrees(tilt), "axial_inflow_m_s": V * math.sin(tilt)}


def stl(m: dict, workdir: Path) -> Path:
    """The machine's propeller in CAD (Dedalus's parametric propeller with the Boreas planform), axis turned to +x."""
    from vegeta import dedalus
    from vegeta.dedalus.examples import Propeller
    path = workdir / f"{m['name']}_prop_axis_x.stl"
    if not path.exists():
        workdir.mkdir(parents=True, exist_ok=True)
        cr, cm, ct = m["chords"]
        g = Propeller().generate(diameter=m["prop"].diameter * 1000, pitch=m["pitch_m"] * 1000, blades=m["prop"].blades,
                                 chord_root=cr * 1000, chord_max=cm * 1000, chord_tip=ct * 1000, **m["cad"])
        gx = dedalus.Geometry.from_cadquery(g.shape.rotate((0, 0, 0), (0, 1, 0), 90), name="prop_axis_x")
        gx.export_stl(str(path), tolerance=0.02 if m["prop"].diameter < 0.3 else 0.1)
    return path


def cfd_point(m, V, tr, rpm, workdir, quality, procs, env):
    """One solved case (reused if solved before): (thrust N, torque N m) or None when it failed."""
    from vegeta import aeromant
    params = dict(rpm=rpm, diameter=m["prop"].diameter, kinematic_viscosity=NU, density=RHO, rotation=1, **QUALITY[quality])
    if V > 0:
        template = "rotor_mrf_oblique"
        params.update(airspeed=V, inflow_angle=tr["inflow_angle_deg"])
    else:
        template = "rotor_mrf_static"
    wd = workdir / f"{m['name']}_{V * 3.6:03.0f}kmh_{rpm:.0f}rpm_{quality}"
    case = aeromant.CFDCase(template, str(stl(m, workdir)), params, workdir=str(wd), geometry_units="mm", environment=env)
    old = case.results() if (wd / "log.solver").exists() else None
    if old is not None and old.ok:
        res = old
        print(f"    reused {wd.name}")
    else:
        if not case.prepare(overwrite=True).ok:
            return None
        t0 = time.time()
        res = case.run(processors=procs)
        print(f"    {wd.name}: {'ok' if res.ok else 'FAILED'} in {(time.time() - t0) / 60:.1f} min")
    if not res.ok or res.metrics.get("thrust_N") is None:
        print("    ", res.messages[-2:])
        return None
    return float(res.metrics["thrust_N"]), float(res.metrics["torque_Nm"])


def point(m, kmh, args, env):
    from vegeta import boreas
    V = kmh / 3.6
    tr = trim(m, V)
    T_need = tr["thrust_per_rotor_N"]
    op = boreas.rpm_for_thrust(m["prop"], m["airfoil"], T_need, tr["axial_inflow_m_s"], RHO)
    af = section_airfoil(m, op.rpm, V, args)
    if af is not None:                                   # NeuralFoil polar at this point's Reynolds number, solved again
        op = boreas.rpm_for_thrust(m["prop"], af, T_need, tr["axial_inflow_m_s"], RHO)
    rpm, T, Q = op.rpm, op.thrust, op.torque
    src = f"blade element{' + NeuralFoil' if af is not None else ''} (no CFD)"
    alpha = (np.asarray(op.r), np.asarray(op.alpha_deg))
    if not args.no_cfd:
        r = cfd_point(m, V, tr, rpm, args.out / "cfd", args.quality, args.procs, env)
        for _ in range(args.iterate):
            if r is None or r[0] <= 0:
                break
            rpm = rpm * math.sqrt(T_need / r[0])                       # thrust ~ rpm^2: one correction to the trim thrust
            r = cfd_point(m, V, tr, rpm, args.out / "cfd", args.quality, args.procs, env)
        if r is not None and r[0] > 0:
            T, Q, src = r[0], r[1], f"CFD ({args.quality})"
        else:
            src = "blade element (the CFD case failed)"
    out = acoustics(m, kmh, tr, rpm, T, Q, T_need, src, args, alpha)
    if af is not None:
        out["airfoil"] = af.describe()
    return out


def use_neuralfoil(args) -> bool:
    import nf_airfoil
    return getattr(args, "airfoil", "auto") != "simple" and nf_airfoil.available()


def section_airfoil(m, rpm, V, args):
    """NeuralFoil airfoil of the machine's section at the Reynolds number of 0.75 R, or None (simple polar)."""
    if not use_neuralfoil(args):
        if getattr(args, "airfoil", "auto") == "neuralfoil":
            sys.exit("--airfoil neuralfoil: pip install neuralfoil")
        return None
    import nf_airfoil
    return nf_airfoil.NeuralFoilAirfoil(m.get("section", "naca4409"), nf_airfoil.reynolds(m["prop"], rpm, V))


def acoustics(m, kmh, tr, rpm, T, Q, T_need, src, args, alpha=None):
    """The rotor's sound at the listener (rotor_noise: steady + unsteady loading tones, BPM broadband)."""
    import rotor_noise as rn
    V = kmh / 3.6
    dstar = None
    if use_neuralfoil(args):                             # the real trailing-edge boundary layer of every strip
        import nf_airfoil
        sec = m.get("section", "naca4409")
        a_of = (lambda r: float(np.interp(r, *alpha))) if alpha is not None else (lambda r: 4.0)
        dstar = lambda r, U, ch, Re: nf_airfoil.te_dstar(sec, a_of(r), Re, ch)
    res = rn.rotor(m["prop"], max(T, 0.0), max(Q, 0.0), rpm, args.distance, args.angle, V, tr["tilt_deg"],
                   rn.NoiseParams(**m.get("noise", {})), dstar)
    tones = res["tones"]
    bb = res["broadband"]
    bb_db = 10 * math.log10(sum(10 ** (L / 10) for _, L in bb))
    peak_hz = max(bb, key=lambda x: x[1])[0]
    one = rn.dba(res)
    g = {"blade_pass_hz": m["prop"].blades * rpm / 60}
    vn = {"spl_db": bb_db, "peak_hz": peak_hz}
    return {"machine": m["name"], "speed_kmh": kmh, "rotors": m["n"], "tilt_deg": round(tr["tilt_deg"], 2),
            "thrust_needed_N": round(T_need, 3), "rpm": round(rpm), "thrust_N": round(T, 3), "torque_Nm": round(Q, 4), "source": src,
            "blade_pass_hz": round(g["blade_pass_hz"], 1), "broadband_db": round(vn["spl_db"], 1), "broadband_peak_hz": round(vn["peak_hz"]),
            "one_rotor_dBA": round(one, 1), "all_rotors_dBA": round(one + 10 * math.log10(m["n"]), 1),
            "noise_model": "Lowson steady+unsteady loading (arm, edgewise, turbulence) + BPM TBL-TE",
            "_tones": tones, "_bb": bb, "_band": (150.0, 0.45 * RATE)}


def synth(p, rpm_scale=1.0, seconds=5.0, seed=0):
    """A rotor's sound pressure [Pa]: its tones (scaled in frequency for an rpm offset) and band-passed broadband."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    x = np.zeros_like(t)
    for f, L in p["_tones"]:
        f = f * rpm_scale
        if f < RATE / 2:
            x += math.sqrt(2) * 20e-6 * 10 ** (L / 20) * np.sin(2 * math.pi * f * t + rng.uniform(0, 2 * math.pi))
    F = np.fft.rfft(rng.standard_normal(len(t)))
    fr = np.fft.rfftfreq(len(t), 1 / RATE)
    if "_bb" in p:                                   # shaped to the 1/3-octave spectrum, each band at its level
        bb = np.array(p["_bb"])
        for fc, L in bb:
            sel = (fr >= fc * 2 ** (-1 / 6) * rpm_scale) & (fr < fc * 2 ** (1 / 6) * rpm_scale)
            if sel.any() and fc * rpm_scale < RATE / 2:
                band = np.zeros_like(F); band[sel] = F[sel]
                nb = np.fft.irfft(band, len(t))
                x += nb * 20e-6 * 10 ** (L / 20) / max(float(np.std(nb)), 1e-30)
    else:
        lo, hi = p["_band"]
        F[(fr < lo) | (fr > hi)] = 0
        nb = np.fft.irfft(F, len(t))
        x += nb * 20e-6 * 10 ** (p["broadband_db"] / 20) / max(float(np.std(nb)), 1e-30)
    fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.05)
    return x * fade


def write_wav(path, x, full):
    pcm = np.clip(x / full, -1, 1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(RATE)
        f.writeframes((pcm * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("machine", choices=("quad", "hexa"))
    ap.add_argument("--speeds", default="0,20,40,60", help="km/h, comma separated")
    ap.add_argument("--quality", default="screening", choices=tuple(QUALITY))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--iterate", type=int, default=1, help="rpm corrections to the trim thrust (each one more CFD case)")
    ap.add_argument("--distance", type=float, default=10.0, help="listener distance [m]")
    ap.add_argument("--angle", type=float, default=60.0, help="listener angle from the rotor axis [deg]")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=Path("prop_sounds"))
    ap.add_argument("--no-cfd", action="store_true", help="blade-element theory only (dry run)")
    ap.add_argument("--airfoil", default="auto", choices=("auto", "neuralfoil", "simple"),
                    help="section polar: NeuralFoil when installed (auto), or the simple polar")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    m = machine(args.machine)
    env = None
    if not args.no_cfd:
        from vegeta import aeromant
        env = aeromant.OpenFOAMEnvironment.detect()
        if env.flavor() is None:
            sys.exit("OpenFOAM not found: install it (./install_local.sh) or run with --no-cfd")
    print(f"{m['name']}: {m['n']} rotors, {m['mass']:.2f} kg, propeller {m['prop'].name} D {m['prop'].diameter * 1000:.0f} mm")
    pts = []
    for kmh in [float(s) for s in args.speeds.split(",") if s.strip()]:
        print(f"  {kmh:.0f} km/h ...", flush=True)
        pts.append(point(m, kmh, args, env))
        print(f"    rpm {pts[-1]['rpm']}, thrust {pts[-1]['thrust_N']} N ({pts[-1]['source']}), {pts[-1]['one_rotor_dBA']} dB(A) one rotor")
    sig = {}
    for p in pts:
        k = int(round(p["speed_kmh"]))
        sig[(k, 1)] = synth(p, seconds=args.seconds, seed=k)
        offsets = np.linspace(-0.015, 0.015, m["n"])                 # the controller's rpm spread: the rotors beat
        sig[(k, m["n"])] = sum(synth(p, 1 + o, args.seconds, seed=1000 * k + i) for i, o in enumerate(offsets))
    full = 1.2 * max(np.abs(s).max() for s in sig.values())
    for (k, n), s in sig.items():
        f = args.out / f"{m['name']}_{k:03d}kmh_{n}prop{'s' if n > 1 else ''}.wav"
        write_wav(f, s, full)
        print("  wrote", f)
    rows = [{k: v for k, v in p.items() if not k.startswith("_")} for p in pts]
    (args.out / f"{m['name']}_sound.json").write_text(json.dumps({"listener": {"distance_m": args.distance, "angle_deg": args.angle},
                                                                   "quality": args.quality, "points": rows}, indent=1))
    with open(args.out / f"{m['name']}_sound.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
