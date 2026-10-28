"""FW-H on a Mancha rotor: near field against Mancha's own pressure, far field against Gutin.

    python benchmarks/acoustics/rotor_fwh.py [--diameter 24] [--tip-speed 0.06] [--advance 0.5]
                                             [--viscosity 0.005] [--out results/acoustics]

The rotor: two flat, twisted blades (pitch/diameter 0.8, chord 0.2 D, 2 cells thick, from 0.2 R to R) on a short
hub, in an open-water box (inflow U = J n D at x−, pressure outlet at x+, far-field velocity faces laterally with a
viscous sponge along them — without it the box edges, where the frame speed is highest, go unstable).
Mancha solves it in the rotating frame (regularized BGK) to a steady state; thrust and torque are monitored.

Three checks follow, all in lattice units (ρ0 = 1, c = 1/√3 — Mancha is weakly compressible, so its pressure
field is itself an acoustic solution):

1. **Near field, FW-H vs Mancha.** Lab-fixed probes around the rotor (in the rotor plane outside the tips,
   upstream, beside the slipstream). Mancha's lab-frame pressure history at a probe is its steady rotating-frame
   field read along the circle the probe describes in that frame. FW-H uses the blade surface from the same run:
   loading from the momentum-exchange force of every wall link (point forces at the link midpoints), thickness
   from the voxel faces moving with Ω×r. The medium is at rest in the frame where rotor and probes translate at
   −U (FW-H with a uniform translation). Compared: the first BPF harmonics and the fluctuating waveform.
2. **Far field, FW-H vs Gutin.** At 20 wavelengths of the blade-passing frequency, FW-H on the same surface
   against Gutin's formula with Mancha's thrust and torque at the conventional effective radius 0.8 R (and at the
   loading-weighted radius), loading only — Gutin has no thickness noise.
3. The thickness/loading split and the directivity, for the record.

Limits: the voxel staircase, a laminar Reynolds number (chord Re ≈ 40), the far field reached through FW-H only
(the box is far smaller than a wavelength; Mancha's own pressure there is near-field).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mancha import RegularizedBGK, RotatingFrame, Simulation, Sponge, __version__  # noqa: E402
from mancha.acoustics import SourceSurface, fwh, gutin, harmonics, rigid_rotation, rotate_x  # noqa: E402
from mancha.diagnostics import solid_faces  # noqa: E402

CS = 1.0 / math.sqrt(3.0)
BLADES = 2


def rotor_mask(shape, centre, radius, x0, chord, thickness, pitch, blades=BLADES, hub_ratio=0.2):
    """Voxels of a rotor turning in +θ (right hand about +x) with axial inflow along +x: flat blades on helices of
    pitch ``pitch`` (geometric angle β = atan(P / 2πr)), mid-chord on the generator, plus a hub cylinder."""
    x = np.arange(shape[0], dtype=float)[:, None, None] - x0
    y = np.arange(shape[1], dtype=float)[None, :, None] - centre[0]
    z = np.arange(shape[2], dtype=float)[None, None, :] - centre[1]
    r = np.sqrt(y ** 2 + z ** 2)
    theta = np.arctan2(z, y)
    rhub = hub_ratio * radius
    hub = (r <= rhub) & (np.abs(x) <= 0.15 * 2 * radius)
    mask = np.zeros(hub.shape, bool)
    beta = np.arctan2(pitch, 2 * np.pi * np.maximum(r, 1e-9))
    for k in range(blades):
        phi = np.angle(np.exp(1j * (theta - 2 * np.pi * k / blades)))
        s = r * phi
        X = -s * np.cos(beta) + x * np.sin(beta)             # along the chord, leading edge at X = −c/2
        Y = -s * np.sin(beta) - x * np.cos(beta)             # normal to the chord (towards the suction side)
        mask |= (np.abs(X) <= chord / 2) & (np.abs(Y) <= thickness / 2) & (r >= rhub) & (r <= radius)
    return mask | hub, mask & ~hub


def run_mancha(args, cache: Path) -> dict:
    if cache.exists():
        with np.load(cache, allow_pickle=False) as d:
            return {k: d[k] for k in d.files} | json.loads(str(d["meta"]))
    D = args.diameter
    R = D / 2
    nx, ny = int(4.5 * D), int(args.box * D) // 2 * 2
    sponge = Sponge(width=max(4.0, D / 4), viscosity=0.05)    # lateral faces: the frame speed is highest there
    centre = ((ny - 1) / 2, (ny - 1) / 2)
    x0 = 1.5 * D
    omega = args.tip_speed / R
    U = args.advance * omega * D / (2 * np.pi)                # J = U / (n D), n = Ω / 2π
    mask, blades = rotor_mask((nx, ny, ny), centre, R, x0, 0.2 * D, 2.0, 0.8 * D)
    frame = RotatingFrame(omega, center=centre)
    sim = Simulation(shape=(nx, ny, ny), collision=RegularizedBGK(3 * args.viscosity + 0.5), frame=frame,
                     velocity=(U, 0, 0), check_interval=250, sponge=sponge)
    sim.set_boundary("x-", "velocity", velocity=(U, 0, 0)).set_boundary("x+", "pressure", density=1.0)
    for face in ("y-", "y+", "z-", "z+"):
        sim.set_boundary(face, "velocity", velocity=(U, 0, 0))
    sim.set_solid(mask)
    t0 = time.perf_counter()
    history = []
    last = None
    while sim.steps_done < args.max_steps:
        sim.run(500)
        F, Mo = sim.solid_load(part=blades)                   # blades only: the hub's load is axisymmetric
        T, Q = -F[0], -Mo[0]
        history.append((sim.steps_done, T, Q))
        print(f"step {sim.steps_done}: T = {T:.5g}, Q = {Q:.5g}", flush=True)
        if last is not None and abs(T - last[0]) < args.tol * abs(T) and abs(Q - last[1]) < args.tol * abs(Q):
            break
        last = (T, Q)
    pos, Fl = sim.solid_links(part=blades)
    fc, fn = solid_faces(sim.solid)
    rho = sim.density()
    out = {"pressure": (rho - 1.0) / 3.0, "density": rho, "velocity": sim.velocity(),   # velocity: frame-relative
           "link_pos": pos, "link_force": Fl, "face_pos": fc,
           "face_normal": fn, "mask": mask, "blades": blades, "history": np.array(history)}
    meta = {"D": D, "omega": omega, "U": U, "centre": list(centre), "x0": x0, "shape": [nx, ny, ny],
            "thrust": float(history[-1][1]), "torque": float(history[-1][2]), "steps": sim.steps_done,
            "wall_time_s": time.perf_counter() - t0, "viscosity": args.viscosity, "tip_speed": args.tip_speed,
            "advance": args.advance}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, meta=np.array(json.dumps(meta)), **out)
    return out | meta


def trilinear(field: np.ndarray, pts: np.ndarray) -> np.ndarray:
    i0 = np.floor(pts).astype(int)
    f = pts - i0
    out = np.zeros(pts.shape[:-1])
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1]) * \
                    (f[..., 2] if dz else 1 - f[..., 2])
                out += w * field[i0[..., 0] + dx, i0[..., 1] + dy, i0[..., 2] + dz]
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--diameter", type=int, default=24)
    ap.add_argument("--tip-speed", type=float, default=0.04)
    ap.add_argument("--box", type=float, default=3.0, help="box width in diameters (lateral)")
    ap.add_argument("--advance", type=float, default=0.5)
    ap.add_argument("--viscosity", type=float, default=0.005)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=2e-4)
    ap.add_argument("--per-rev", type=int, default=720, help="FW-H source samples per revolution")
    ap.add_argument("--out", default="results/acoustics")
    args = ap.parse_args(argv)
    out = Path(args.out)
    run = run_mancha(args, out / f"rotor_D{args.diameter}.npz")
    D, om, U = run["D"], run["omega"], run["U"]
    R = D / 2
    cy, cz = run["centre"]
    centre = np.array([0.0, cy, cz])
    x0 = run["x0"]
    T, Q = run["thrust"], run["torque"]
    period = 2 * np.pi / om
    bpf = BLADES * om / (2 * np.pi)
    tau = np.arange(4 * args.per_rev) * period / args.per_rev
    V = np.array([-U, 0.0, 0.0])                              # rotor and probes translate at −U in the medium

    # FW-H sources from the run: loading on every wall link (force on the fluid = −force on the body) and
    # thickness on every voxel face, all turning with Ω about the axis.
    kin = rigid_rotation(run["link_pos"], None, om, tau, centre)
    Fb = rotate_x(-run["link_force"], om * tau)
    loading = SourceSurface("forces", tau, kin["y"], kin["v"], np.ones(len(run["link_pos"])), force=Fb,
                            vdot=kin["vdot"])
    kf = rigid_rotation(run["face_pos"], run["face_normal"], om, tau, centre)
    thickness = SourceSurface("solid", tau, kf["y"], kf["v"], np.ones(len(run["face_pos"])), n=kf["n"],
                              p=np.zeros(kf["y"].shape[:2]), vdot=kf["vdot"], ndot=kf["ndot"])

    # 1. near field: probes fixed in the lab, Mancha's pressure read along the probe's circle in the frame
    probes = {"rotor plane, 1.25 R": (x0, 1.25 * R, 0.0), "rotor plane, 1.5 R": (x0, 1.5 * R, 0.0),
              "rotor plane, 1.75 R": (x0, 1.75 * R, 0.0), "upstream 0.5 R, r = R": (x0 - 0.5 * R, R, 0.0),
              "downstream 1 R, r = 1.6 R": (x0 + R, 1.6 * R, 0.0)}
    near, near_series = {}, {}
    for name, (px, pr, pa) in probes.items():
        X = np.array([px, cy + pr * np.cos(pa), cz + pr * np.sin(pa)])
        fl = fwh(loading, X[None, :], CS, 1.0, V)
        ft = fwh(thickness, X[None, :], CS, 1.0, V, t_obs=fl["t"])
        t = fl["t"]
        # Mancha at the same lab times: the rotor has turned by Ωt, so the lab-fixed probe sits at R(−Ωt) X in
        # the rotating frame (no realignment — phase is compared too)
        inframe = rotate_x((X - centre)[None, :], -om * t)[:, 0, :] + centre
        p_m = trilinear(run["pressure"], inframe)
        p_f, p_l, p_t = fl["total"][0] + ft["total"][0], fl["total"][0], ft["total"][0]
        h_m, h_f = harmonics(t, p_m, bpf, 3), harmonics(t, p_f, bpf, 3)
        wm, wf = p_m - p_m.mean(), p_f - p_f.mean()
        near[name] = {"mancha_bpf_rms": h_m.tolist(), "fwh_bpf_rms": h_f.tolist(),
                      "ratio_first_harmonic": float(h_f[0] / h_m[0]),
                      "fwh_thickness_share_first_harmonic": float(harmonics(t, p_t, bpf, 1)[0] / h_f[0]),
                      "waveform_rms_mancha": float(np.sqrt(np.mean(wm ** 2))),
                      "waveform_rms_difference": float(np.sqrt(np.mean((wm - wf) ** 2)))}
        near_series[name] = (t, wm, wf, p_l - p_l.mean(), p_t - p_t.mean())
        print(f"{name:27s} BPF1: Mancha {h_m[0]:.3e}  FW-H {h_f[0]:.3e}  ratio {h_f[0] / h_m[0]:.3f}; waveform rms "
              f"difference {near[name]['waveform_rms_difference'] / near[name]['waveform_rms_mancha']:.2f} of "
              f"Mancha's", flush=True)

    # 2. far field against Gutin (loading only), on the side the thrust points to (−x) and behind
    lam = CS / bpf
    dist = 20 * lam
    r_t = np.linalg.norm(run["link_pos"][:, 1:] - centre[1:], axis=1)
    Ft = -run["link_force"]                                   # on the fluid
    tang = np.stack([np.zeros_like(r_t), -(run["link_pos"][:, 2] - cz), run["link_pos"][:, 1] - cy], 1) / r_t[:, None]
    f_tan = (Ft * tang).sum(1)
    r_w = float((np.abs(f_tan) * r_t).sum() / np.abs(f_tan).sum())   # loading-weighted radius of the drag force
    far = []
    for deg in range(10, 171, 20):
        th = math.radians(deg)
        X = centre + np.array([x0 - dist * math.cos(th), dist * math.sin(th), 0.0])   # θ from −x (ahead)
        fl = fwh(loading, X[None, :], CS, 1.0, V)
        ft = fwh(thickness, X[None, :], CS, 1.0, V, t_obs=fl["t"])
        hl = harmonics(fl["t"], fl["total"][0], bpf, 3)
        f0 = fwh(loading, X[None, :], CS, 1.0)                     # hover (no translation), as Gutin assumes
        h0 = harmonics(f0["t"], f0["total"][0], bpf, 3)
        ht = harmonics(ft["t"], ft["total"][0], bpf, 3)
        g08 = gutin(T, Q, om, BLADES, 0.8 * R, dist, th, CS, 3)
        gw = gutin(T, Q, om, BLADES, r_w, dist, th, CS, 3)
        far.append({"angle_deg": deg, "fwh_loading": hl.tolist(), "fwh_loading_static": h0.tolist(),
                    "fwh_thickness": ht.tolist(),
                    "gutin_0.8R": g08.tolist(), "gutin_weighted_radius": gw.tolist()})
        print(f"far field θ={deg:3d}°: FW-H loading BPF1 {hl[0]:.3e} (static {h0[0]:.3e}), Gutin(0.8R) {g08[0]:.3e} "
              f"(ratio static/Gutin {h0[0] / g08[0]:.3f}), Gutin(r_w={r_w / R:.2f}R) ratio {h0[0] / gw[0]:.3f}; "
              f"thickness {ht[0]:.3e}", flush=True)

    record = {"mancha_version": __version__, "rotor": {k: run[k] for k in ("D", "omega", "U", "thrust", "torque",
                                                                              "steps", "viscosity", "tip_speed",
                                                                              "advance", "wall_time_s")},
              "tip_mach": float(om * R / CS), "chord_reynolds_0.75R": float(
                  math.hypot(0.75 * om * R, U) * 0.2 * D / run["viscosity"]),
              "links": int(len(run["link_pos"])), "faces": int(len(run["face_pos"])),
              "loading_weighted_radius_over_R": r_w / R, "far_field_distance_wavelengths": 20,
              "near_field": near, "far_field": far}
    (out / "rotor_fwh.json").write_text(json.dumps(record, indent=1, default=float))
    plot(out, run, near_series, far, period, R, T, Q)
    print(f"-> {out / 'rotor_fwh.json'} and figures rotor_*.png")


def plot(out: Path, run: dict, near_series: dict, far: list, period: float, R: float, T: float, Q: float) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    D = run["D"]
    names = list(near_series)
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 2.0 * len(names)), sharex=True)
    for ax, name in zip(axes, names):
        t, wm, wf, wl, wt = near_series[name]
        rev = (t - t[0]) / period
        ax.plot(rev, wm, "-", lw=2.4, color="0.7", label="Mancha (direct, compressible LBM)")
        ax.plot(rev, wf, "--", lw=1.3, color="C0", label="FW-H from Mancha's blade surface")
        ax.plot(rev, wl, ":", lw=1, color="C2", label="  loading part")
        ax.plot(rev, wt, ":", lw=1, color="C1", label="  thickness part")
        ax.set_title(name, fontsize=9)
        ax.set_ylabel("p' [lattice]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("revolutions (lab time, same rotor angle in both)")
    fig.suptitle(f"Near field around the rotor: Mancha vs FW-H (D = {D} cells, T = {T:.3g}, Q = {Q:.3g})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "rotor_near.png", dpi=140)
    plt.close(fig)

    ang = [f["angle_deg"] for f in far]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for m, ax in enumerate(axes):
        ax.semilogy(ang, [f["fwh_loading_static"][m] for f in far], "-", color="C0", label="FW-H loading (hover)")
        ax.semilogy(ang, [f["fwh_loading"][m] for f in far], "--", color="C0", label="FW-H loading (with inflow)")
        ax.semilogy(ang, [f["fwh_thickness"][m] for f in far], "-", color="C1", label="FW-H thickness")
        ax.semilogy(ang, [f["gutin_0.8R"][m] for f in far], "o", mfc="none", color="k", label="Gutin, R_e = 0.8 R")
        ax.semilogy(ang, [f["gutin_weighted_radius"][m] for f in far], "x", color="k",
                    label="Gutin, loading-weighted R_e")
        ax.set_title(f"{m + 1} × BPF", fontsize=9)
        ax.set_xlabel("angle from the axis ahead [deg]")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("p rms [lattice] at 20 wavelengths")
    axes[0].legend(fontsize=7)
    fig.suptitle("Far field: FW-H on Mancha's blade loads vs Gutin with Mancha's thrust and torque", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "rotor_far.png", dpi=140)
    plt.close(fig)

    # one-page summary: waveform at one probe, BPF amplitude at every probe, far-field directivity
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})
    key = "rotor plane, 1.5 R" if "rotor plane, 1.5 R" in near_series else names[0]
    t, wm, wf, wl, wt = near_series[key]
    rev = (t - t[0]) / period
    axes[0].plot(rev, wm, "-", lw=2.4, color="0.6", label="Mancha (direct LBM pressure)")
    axes[0].plot(rev, wf, "--", lw=1.4, color="C0", label="FW-H from Mancha's blade surface")
    axes[0].set(xlabel="revolutions", ylabel="p' [lattice]", title=f"Near field, probe {key}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    amps_m = [harmonics(near_series[n][0], near_series[n][1], 2.0 / period, 1)[0] for n in names]
    amps_f = [harmonics(near_series[n][0], near_series[n][2], 2.0 / period, 1)[0] for n in names]
    xs = np.arange(len(names))
    axes[1].bar(xs - 0.2, amps_m, 0.4, color="0.6", label="Mancha")
    axes[1].bar(xs + 0.2, amps_f, 0.4, color="C0", label="FW-H")
    for i, (a, b) in enumerate(zip(amps_m, amps_f)):
        axes[1].text(i + 0.2, b, f"{b / a:.2f}", ha="center", va="bottom", fontsize=7)
    axes[1].set_xticks(xs, [n.replace(", ", "\n") for n in names], fontsize=7)
    axes[1].set(ylabel="BPF amplitude, rms [lattice]", title="Near field, every probe (label: FW-H / Mancha)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    axes[2].semilogy(ang, [f["fwh_loading_static"][0] for f in far], "-", color="C0", label="FW-H loading")
    axes[2].semilogy(ang, [f["fwh_thickness"][0] for f in far], "-", color="C1", label="FW-H thickness")
    axes[2].semilogy(ang, [f["gutin_0.8R"][0] for f in far], "o", mfc="none", color="k", label="Gutin (same T, Q)")
    axes[2].set(xlabel="angle from the axis ahead [deg]", ylabel="BPF rms at 20 wavelengths [lattice]",
                title="Far field, 1 × BPF")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, which="both")
    fig.suptitle(f"Rotor D = {D} cells: FW-H vs Mancha's own pressure vs Gutin — no measured data exist for this "
                 "synthetic rotor", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "rotor_summary.png", dpi=140)
    plt.close(fig)

    h = np.asarray(run["history"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(h[:, 0] / period, h[:, 1], "o-", ms=3, label="thrust T")
    ax.plot(h[:, 0] / period, h[:, 2], "s-", ms=3, label="torque Q")
    ax.set(xlabel="revolutions", ylabel="lattice units", title="Blade loads while Mancha converges")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rotor_history.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
