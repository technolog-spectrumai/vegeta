"""FW-H verification figures: exact fields, panel convergence and Gutin.

    python benchmarks/acoustics/verify_fwh.py [--out results/acoustics]

1. ``fwh_exact.png`` — pressure histories from FW-H on a permeable sphere (r = 0.3 m) around a dipole, the
   sphere at rest and turning at Mach 0.26, against the exact field, at a far and a near observer.
2. ``fwh_convergence.png`` — the maximum error against the number of panels.
3. ``fwh_gutin.png`` — directivity of the first three blade-passing harmonics: FW-H with two rotating compact
   forces against Gutin's formula (and Vegeta's ``boreas.noise.gutin_harmonics`` when Vegeta is importable).
Writes ``verify_fwh.json`` with the numbers behind the figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mancha.acoustics import (  # noqa: E402
    SourceSurface,
    acoustic_field,
    dipole_potential,
    fwh,
    gutin,
    harmonics,
    rigid_rotation,
    sphere_panels,
)

C, RHO0 = 340.0, 1.2
OMEGA = 2 * np.pi * 200.0
TAU = np.linspace(0, 0.03, 900)
OBS = np.array([[2.0, 0.5, 1.0], [-0.6, 0.2, 0.1]])
OBS_NAMES = ["far observer (2.3 m)", "near observer (0.64 m)"]


def permeable(phi, rotation, n_theta=40, n_phi=80):
    pts, nrm, dS = sphere_panels(0.3, n_theta, n_phi)
    k = rigid_rotation(pts, nrm, rotation, TAU)
    rho, u, p = acoustic_field(phi, k["y"], TAU[:, None], RHO0, C)
    s = SourceSurface("permeable", TAU, k["y"], k["v"], dS, n=k["n"], p=p, rho=rho, u=u, vdot=k["vdot"],
                      ndot=k["ndot"])
    r = fwh(s, OBS, C, RHO0)
    exact = acoustic_field(phi, OBS[:, None, :], r["t"][None, :], RHO0, C)[2]
    return r, exact


def gutin_directivity():
    B, R, rpm = 2, 0.12, 6000.0
    om = rpm * 2 * np.pi / 60
    re, thrust, torque = 0.8 * R, 4.0, 0.08
    per = 720
    tau = np.arange(5 * per) * (2 * np.pi / om) / per
    ang = 2 * np.pi * np.arange(B) / B
    k = rigid_rotation(np.stack([0 * ang, re * np.cos(ang), re * np.sin(ang)], 1), None, om, tau)
    e_theta = k["v"] / np.linalg.norm(k["v"], axis=-1, keepdims=True)
    force = np.zeros_like(k["y"])
    force[..., 0] = -thrust / B
    force += torque / (B * re) * e_theta
    s = SourceSurface("forces", tau, k["y"], k["v"], np.ones(B), force=force, vdot=k["vdot"])
    angles = np.arange(10, 171, 10)
    f, g = [], []
    for deg in angles:
        th = np.radians(deg)
        r = fwh(s, [[30 * np.cos(th), 30 * np.sin(th), 0.0]], C, RHO0)
        f.append(harmonics(r["t"], r["total"][0], B * om / (2 * np.pi), 3))
        g.append(gutin(thrust, torque, om, B, re, 30.0, th, C, 3))
    boreas = None
    try:
        sys.path[:0] = [str(ROOT.parents[0] / "vegeta-cli" / "src"), str(ROOT.parents[0] / "ogma" / "src")]
        from vegeta.boreas import Medium, Propeller, gutin_harmonics

        prop = Propeller("verification", B, (0.03, 0.06, R), (0.02, 0.02, 0.015), (30, 20, 12))
        boreas = [gutin_harmonics(prop, thrust, torque, rpm, 30.0, float(d), medium=Medium("air", RHO0, C),
                                  harmonics=3)["p_rms_pa"] for d in angles]
    except ImportError:
        pass
    return angles, np.array(f), np.array(g), None if boreas is None else np.array(boreas)


def main(argv=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="results/acoustics")
    out = Path(ap.parse_args(argv).out)
    out.mkdir(parents=True, exist_ok=True)
    record = {}

    phi = dipole_potential(1e-3, OMEGA, C, axis=(0, 0.6, 0.8))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for col, rot in enumerate((0.0, 300.0)):
        r, exact = permeable(phi, rot)
        for row in range(2):
            ax = axes[row, col]
            t_ms = (r["t"] - r["t"][0]) * 1e3
            ax.plot(t_ms, exact[row], "-", lw=2.2, color="0.75", label="exact")
            ax.plot(t_ms, r["total"][row], "--", lw=1.2, color="C0", label="FW-H total")
            ax.plot(t_ms, r["thickness"][row], ":", lw=1, color="C1", label="thickness term")
            ax.plot(t_ms, r["loading"][row], ":", lw=1, color="C2", label="loading term")
            err = np.abs(r["total"][row] - exact[row]).max() / np.abs(exact[row]).max()
            record[f"exact_rotation{int(rot)}_{row}"] = err
            ax.set_title(f"{OBS_NAMES[row]}, surface {'at rest' if rot == 0 else 'turning, M = 0.26'}\n"
                         f"max error {err:.1e}", fontsize=9)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel("p' [Pa]")
            if row == 1:
                ax.set_xlabel("observer time [ms]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("FW-H (permeable sphere) vs exact field of a 200 Hz dipole")
    fig.tight_layout()
    fig.savefig(out / "fwh_exact.png", dpi=140)
    plt.close(fig)

    panels, errs = [], []
    for nt, nph in ((6, 12), (10, 20), (20, 40), (40, 80)):
        r, exact = permeable(phi, 300.0, nt, nph)
        panels.append(nt * nph)
        errs.append(float(np.abs(r["total"] - exact).max() / np.abs(exact).max()))
    record["convergence"] = {"panels": panels, "errors": errs}
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.loglog(panels, errs, "o-", label="FW-H, turning sphere")
    ax.loglog(panels, errs[0] * (np.array(panels) / panels[0]) ** -1.0, "--", color="0.5",
              label="second order (∝ 1/panels)")
    ax.set(xlabel="panels on the sphere", ylabel="max relative error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("Panel convergence (floor: time interpolation)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fwh_convergence.png", dpi=140)
    plt.close(fig)

    angles, f, g, b = gutin_directivity()
    record["gutin"] = {"angles": angles.tolist(), "fwh": f.tolist(), "gutin": g.tolist(),
                       "boreas": None if b is None else b.tolist()}
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in range(3):
        ax.semilogy(angles, f[:, m], "-", color=f"C{m}", label=f"FW-H, {m + 1}×BPF")
        ax.semilogy(angles, g[:, m], "o", mfc="none", color=f"C{m}", label=f"Gutin, {m + 1}×BPF")
        if b is not None:
            ax.semilogy(angles, b[:, m], "x", color=f"C{m}", ms=5, label=f"Boreas, {m + 1}×BPF")
    ax.set(xlabel="angle from the axis ahead [deg]", ylabel="p rms [Pa] at 30 m")
    ax.set_title("Two-blade compact rotor, 6000 rpm, T = 4 N, Q = 0.08 N·m", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "fwh_gutin.png", dpi=140)
    plt.close(fig)
    (out / "verify_fwh.json").write_text(json.dumps(record, indent=1))
    print("figures in", out)


if __name__ == "__main__":
    main()
