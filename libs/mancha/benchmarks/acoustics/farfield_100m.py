"""Propagate Mancha's near-field rotor solutions to 100 m and compare: levels (dB, dB(A)), tones, directivity.

    python benchmarks/acoustics/farfield_100m.py RUN.npz [RUN.npz ...] [--diameter-m 0.254] [--distance 100]
                                                  [--out results/acoustics/farfield]

RUN.npz is a cache written by ``rotor_fwh.py`` (blade loads, voxel faces, and the converged density and
frame-relative velocity fields).

**Physical scale.** The lattice rotor is taken as an air propeller of diameter ``--diameter-m`` with the
simulated tip Mach number (c = 343 m/s, ρ = 1.2 kg/m³): the cell is ``dx = D/D_cells``, the step
``dt = dx (1/√3) / 343`` (lattice sound speed → 343 m/s); velocities scale by dx/dt, forces by ρ dx⁴/dt²,
pressures by ρ (dx/dt)². The Reynolds number stays the simulation's (tens): the levels are those of this
viscous test rotor at that rpm, not of a real propeller of that size.

**Methods compared at the observers** (all through FW-H, Farassat 1A, medium at rest, rotor and observers
translating at −U — the wind-tunnel inflow):

1. *Solid surface*: loading from Mancha's per-link blade forces, thickness from the blade voxel faces.
2. *Permeable surface (closed)*: Mancha's own near-field flow — density, velocity, pressure — on a closed
   cylinder (radius 1.4 R, from 0.5 D upstream to 0.75 D downstream of the rotor plane) fixed in the lab: the
   near-field solution itself propagated, including the volume (wake) sources inside it.
3. *Permeable surface (open)*: the same without the downstream end cap, which the wake crosses (the classic
   end-cap problem): the spread between 2 and 3 is an uncertainty band of the permeable result.
4. *Gutin*: compact steady loading with Mancha's thrust and torque at 0.8 R (no thickness noise).

Outputs: ``farfield_<run>.json`` (SPL of each blade-passing harmonic, tonal OASPL in dB and dB(A), per angle
and method) and figures (spectra, directivity, waveforms).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rotor_fwh import BLADES, trilinear  # noqa: E402

from mancha.acoustics import SourceSurface, fwh, gutin, harmonics, rigid_rotation, rotate_x  # noqa: E402

C_AIR, RHO_AIR, P_REF = 343.0, 1.2, 20e-6
CS = 1.0 / math.sqrt(3.0)
HARMONICS = 5
ANGLES = list(range(10, 171, 10))


def a_weighting(f: np.ndarray) -> np.ndarray:
    """A-weighting [dB] (IEC 61672-1)."""
    f = np.asarray(f, dtype=float)
    f2 = f * f
    ra = 12194.0 ** 2 * f2 * f2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
                                   * (f2 + 12194.0 ** 2))
    return 20 * np.log10(ra) + 2.0


def spl(p_rms) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.asarray(p_rms, dtype=float), 1e-30) / P_REF)


def cylinder(radius, x0, x1, centre, n_az=72, step=1.0, caps=(True, True)):
    """Panels of a closed (or open) cylinder about the x axis: centres, outward normals, areas (lattice units)."""
    phi = (np.arange(n_az) + 0.5) * 2 * np.pi / n_az
    dphi = 2 * np.pi / n_az
    nx = max(2, int(round((x1 - x0) / step)))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    X, P = np.meshgrid(xs, phi, indexing="ij")
    pts = [np.stack([X, centre[1] + radius * np.cos(P), centre[2] + radius * np.sin(P)], -1).reshape(-1, 3)]
    nrm = [np.stack([0 * X, np.cos(P), np.sin(P)], -1).reshape(-1, 3)]
    area = [np.full(X.size, radius * dphi * (x1 - x0) / nx)]
    nr = max(2, int(round(radius / step)))
    redge = np.linspace(0.0, radius, nr + 1)
    for xc, sign, on in ((x0, -1.0, caps[0]), (x1, 1.0, caps[1])):
        if not on:
            continue
        rc = 0.5 * (redge[1:] + redge[:-1])
        R_, P_ = np.meshgrid(rc, phi, indexing="ij")
        pts.append(np.stack([np.full(R_.size, xc), centre[1] + (R_ * np.cos(P_)).ravel(),
                             centre[2] + (R_ * np.sin(P_)).ravel()], -1))
        nrm.append(np.tile([sign, 0.0, 0.0], (R_.size, 1)))
        area.append(np.repeat(0.5 * (redge[1:] ** 2 - redge[:-1] ** 2) * dphi, n_az))
    return np.concatenate(pts), np.concatenate(nrm), np.concatenate(area)


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        run = {k: d[k] for k in d.files}
    run.update(json.loads(str(run.pop("meta"))))
    if "velocity" not in run:
        raise SystemExit(f"{path} has no velocity field — rerun rotor_fwh.py to regenerate the cache")
    return run


def propagate(run: dict, diameter_m: float, distance: float, per_rev: int) -> dict:
    D, om, U = run["D"], run["omega"], run["U"]
    R = D / 2
    cy, cz = run["centre"]
    x0 = run["x0"]
    centre = np.array([x0, cy, cz])
    axis_pt = np.array([0.0, cy, cz])
    dx = diameter_m / D
    dt = dx * CS / C_AIR
    vs = dx / dt
    fs = RHO_AIR * dx ** 4 / dt ** 2
    ps = RHO_AIR * vs ** 2
    om_p = om / dt
    rpm = om_p * 60 / (2 * np.pi)
    bpf = BLADES * om_p / (2 * np.pi)
    U_p = U * vs
    V = np.array([-U_p, 0.0, 0.0])
    period = 2 * np.pi / om_p
    tau = np.arange(4 * per_rev) * period / per_rev               # physical source time [s]
    ang = om_p * tau

    def to_m(p):
        return (p - centre) * dx

    surfaces = {}
    kin = rigid_rotation(to_m(run["link_pos"]), None, om_p, tau)
    loads = rotate_x(-run["link_force"] * fs, ang)
    surfaces["loading"] = SourceSurface("forces", tau, kin["y"], kin["v"], np.ones(len(loads[0])), force=loads,
                                        vdot=kin["vdot"])
    kf = rigid_rotation(to_m(run["face_pos"]), run["face_normal"], om_p, tau)
    surfaces["thickness"] = SourceSurface("solid", tau, kf["y"], kf["v"], np.full(len(run["face_pos"]), dx * dx),
                                          n=kf["n"], p=np.zeros(kf["y"].shape[:2]), vdot=kf["vdot"],
                                          ndot=kf["ndot"])
    rho_f, u_f = run["density"], run["velocity"]
    for name, caps in (("permeable_closed", (True, True)), ("permeable_open", (True, False))):
        pts, nrm, area = cylinder(1.4 * R, x0 - 0.5 * D, x0 + 0.75 * D, centre, caps=caps)
        inframe = rotate_x(pts - axis_pt, -ang) + axis_pt           # the lab-fixed panels seen from the frame
        rho = trilinear(rho_f, inframe)
        u_rel = np.stack([trilinear(u_f[k], inframe) for k in range(3)], -1)
        r_frame = inframe - axis_pt
        u_abs_frame = u_rel + np.stack([np.zeros_like(rho), -om * r_frame[..., 2], om * r_frame[..., 1]], -1)
        u_lab = np.einsum("tij,tej->tei", np.stack([np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)],
                                                              [0, math.sin(a), math.cos(a)]]) for a in ang]),
                          u_abs_frame)
        u_med = (u_lab - np.array([U, 0.0, 0.0])) * vs             # medium (air at rest) frame, m/s
        y = np.broadcast_to(to_m(pts), (len(tau),) + pts.shape)
        zero = np.zeros_like(y)
        surfaces[name] = SourceSurface("permeable", tau, y, zero, area * dx * dx, n=np.broadcast_to(nrm, y.shape),
                                       p=(rho - 1.0) / 3.0 * ps, rho=rho * RHO_AIR, u=u_med, vdot=zero,
                                       ndot=np.zeros_like(y))
    T, Q = run["thrust"] * fs, run["torque"] * fs * dx
    results = {"angles": ANGLES, "frequencies_hz": [m * bpf for m in range(1, HARMONICS + 1)], "methods": {}}
    series = {}
    for method in ("solid", "solid_loading", "solid_thickness", "permeable_closed", "permeable_open", "gutin"):
        results["methods"][method] = {"spl_db": [], "oaspl_db": [], "oaspl_dba": []}
    for deg in ANGLES:
        th = math.radians(deg)
        obs = np.array([[-distance * math.cos(th), distance * math.sin(th), 0.0]])  # θ from −x, the thrust side
        sig = {}
        fl = fwh(surfaces["loading"], obs, C_AIR, RHO_AIR, V)
        t = fl["t"]
        sig["solid_loading"] = fl["total"][0]
        sig["solid_thickness"] = fwh(surfaces["thickness"], obs, C_AIR, RHO_AIR, V, t_obs=t)["total"][0]
        sig["solid"] = sig["solid_loading"] + sig["solid_thickness"]
        for name in ("permeable_closed", "permeable_open"):
            sig[name] = fwh(surfaces[name], obs, C_AIR, RHO_AIR, V, t_obs=t)["total"][0]
        for name, p in sig.items():
            h = harmonics(t, p, bpf, HARMONICS)
            _store(results["methods"][name], h, bpf)
        g = gutin(abs(T), abs(Q), om_p, BLADES, 0.8 * R * dx, distance, th, C_AIR, HARMONICS)
        _store(results["methods"]["gutin"], g, bpf)
        if deg in (30, 90):
            series[deg] = (t - t[0], sig)
    results["case"] = {"diameter_m": diameter_m, "rpm": rpm, "bpf_hz": bpf, "tip_speed_m_s": om_p * R * dx,
                       "tip_mach": om_p * R * dx / C_AIR, "inflow_m_s": U_p, "thrust_N": T, "torque_Nm": Q,
                       "distance_m": distance, "cell_mm": dx * 1e3, "time_step_us": dt * 1e6,
                       "chord_reynolds_simulated": run.get("viscosity") and math.hypot(0.75 * om * R, U) * 0.2 * D /
                       run["viscosity"]}
    return results, series


def _store(rec: dict, h: np.ndarray, bpf: float) -> None:
    f = bpf * np.arange(1, len(h) + 1)
    levels = spl(h)
    rec["spl_db"].append(levels.tolist())
    rec["oaspl_db"].append(float(10 * np.log10(np.sum(10 ** (levels / 10)))))
    rec["oaspl_dba"].append(float(10 * np.log10(np.sum(10 ** ((levels + a_weighting(f)) / 10)))))


LABELS = {"solid": "FW-H solid (Mancha blade surface)", "permeable_closed": "FW-H permeable, closed (Mancha flow)",
          "permeable_open": "FW-H permeable, open end", "gutin": "Gutin (Mancha T, Q)",
          "solid_loading": "  solid: loading part", "solid_thickness": "  solid: thickness part"}
STYLE = {"solid": ("C0", "o-"), "permeable_closed": ("C3", "s-"), "permeable_open": ("C3", "s--"),
         "gutin": ("k", "x:"), "solid_loading": ("C2", "-"), "solid_thickness": ("C1", "-")}


def plot(results: dict, series: dict, out: Path, tag: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case = results["case"]
    f = np.array(results["frequencies_hz"])
    ang = results["angles"]
    head = (f"{tag}: {case['diameter_m'] * 1000:.0f} mm rotor, {case['rpm']:.0f} rpm, BPF {case['bpf_hz']:.1f} Hz, "
            f"tip Mach {case['tip_mach']:.2f}, at {case['distance_m']:.0f} m")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, deg in zip(axes[0], (90, 30)):
        i = ang.index(deg)
        w = 0.18
        for k, m in enumerate(("solid", "permeable_closed", "permeable_open", "gutin")):
            lv = np.array(results["methods"][m]["spl_db"][i])
            ax.bar(np.arange(len(f)) + (k - 1.5) * w, lv, w, color=STYLE[m][0], alpha=0.9 if k != 2 else 0.5,
                   label=LABELS[m])
        ax.set_xticks(np.arange(len(f)), [f"{x:.0f} Hz\n{j + 1}×BPF" for j, x in enumerate(f)], fontsize=8)
        ax.set(ylabel="SPL [dB re 20 µPa]", title=f"Tone spectrum at {deg}° from the axis ahead")
        lo = min(min(results["methods"][m]["spl_db"][i]) for m in ("solid", "permeable_closed", "gutin"))
        ax.set_ylim(bottom=max(lo - 10, -60))
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].legend(fontsize=7)
    ax = axes[1][0]
    for m in ("solid", "permeable_closed", "permeable_open", "gutin", "solid_loading", "solid_thickness"):
        c, st = STYLE[m]
        ax.plot(ang, [x[0] for x in results["methods"][m]["spl_db"]], st, color=c, label=LABELS[m],
                ms=4, alpha=0.6 if m.startswith("solid_") else 1)
    ax.set(xlabel="angle from the axis ahead [deg]", ylabel="SPL at 1×BPF [dB]", title="Directivity of the first tone")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1][1]
    t, sig = series[90]
    n = int(2 * (1 / case["bpf_hz"]) / (t[1] - t[0]))
    for m in ("solid", "permeable_closed", "permeable_open"):
        c, st = STYLE[m]
        ax.plot(t[:n] * 1e3, (sig[m][:n] - sig[m][:n].mean()) * 1e3, st.replace("o", "").replace("s", ""),
                color=c, label=LABELS[m])
    ax.set(xlabel="time [ms]", ylabel="p' [mPa]", title="Waveform at 90° (two blade passages)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle(head, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"farfield_{tag}.png", dpi=140)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--diameter-m", type=float, default=0.254)
    ap.add_argument("--distance", type=float, default=100.0)
    ap.add_argument("--per-rev", type=int, default=360)
    ap.add_argument("--out", default="results/acoustics/farfield")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in map(Path, args.runs):
        tag = path.stem.replace("rotor_", "")
        results, series = propagate(load(path), args.diameter_m, args.distance, args.per_rev)
        (out / f"farfield_{tag}.json").write_text(json.dumps(results, indent=1))
        plot(results, series, out, tag)
        c = results["case"]
        print(f"{tag}: {c['rpm']:.0f} rpm, BPF {c['bpf_hz']:.1f} Hz, T = {c['thrust_N']:.3g} N, "
              f"Q = {c['torque_Nm']:.3g} N·m")
        i = ANGLES.index(90)
        for m, rec in results["methods"].items():
            print(f"  {m:18s} at 90°: tones {np.round(rec['spl_db'][i], 1)} dB, OASPL {rec['oaspl_db'][i]:.1f} dB, "
                  f"{rec['oaspl_dba'][i]:.1f} dB(A)")


if __name__ == "__main__":
    main()
