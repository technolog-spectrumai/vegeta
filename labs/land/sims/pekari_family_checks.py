#!/usr/bin/env python3
"""Pekari-2M, Pumba and Calydon — the checks behind docs/pekari_pumba.md, one run: the headline numbers, the lift capacity
against reach (arm and stability limits, with and without Pumba's stabilizers), the drive on grades, the
manipulator workspaces, the MuJoCo stand tests (folded / deployed with the rated loads) and the two platforms to
scale (CAD, with a 1.8 m person for scale).

    python3 labs/land/sims/pekari_family_checks.py                 # all, figures to sims/output/
    python3 labs/land/sims/pekari_family_checks.py --no-mujoco --no-cad
    python3 labs/land/sims/pekari_family_checks.py --docs          # also copy the figures to docs/figures/

Writes ``sims/output/pekari_family.json`` and ``pekari_family_*.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "land" / "designs"))

import pekari_arms as pa  # noqa: E402
import pekari_family as pf  # noqa: E402
import pekari_2m_robot as p2  # noqa: E402
import pumba_robot as pu  # noqa: E402
import calydon_robot as cr  # noqa: E402

OUT = ROOT / "labs" / "land" / "sims" / "output"
DOCS = ROOT / "docs" / "figures"
PLATFORMS = (p2.PEKARI_2M, pu.PUMBA, cr.CALYDON)


def numbers() -> dict:
    out = {}
    for pl in PLATFORMS:
        s = pf.summary(pl)
        rows = {}
        for m in pl.mounts:
            R = pa.reach(m.arm)
            for st in ((False, True) if pl.stabilizers else (False,)):
                for yaw_name, yaw in (("front", 0.0), ("side", math.copysign(math.pi / 2, m.y))):
                    key = f"{m.name} {yaw_name}{' stabilized' if st else ''}"
                    rows[key] = [(round(r, 3), {k: (round(v, 1) if isinstance(v, float) and math.isfinite(v) else v)
                                                for k, v in pf.lift_capacity(pl, m.name, r, yaw, stabilizers_deployed=st).items()})
                                 for r in [R * f for f in (0.4, 0.55, 0.7, 0.85, 1.0)]]
        out[pl.name] = {
            "summary": s, "mass_budget": pf.mass_budget(pl),
            "drive": {g: pf.drive_check(pl, grade_deg=g, v_ms=0.3) for g in (0, 10, 20, 30)},
            "power": pf.power(pl), "bom": pf.bom(pl), "bom_total_eur": pf.bom_total(pl), "capabilities": {k: (sorted(v) if isinstance(v, frozenset) else v)
                                                    for k, v in pf.capabilities(pl).items()},
            "arms": {m.name: {"mass_kg": pa.total_mass(m.arm), "reach_m": pa.reach(m.arm), "dof": pa.arm_dof(m.arm),
                              "peak": pa.load_curve(m.arm, 7), "hold": pa.load_curve(m.arm, 7, "hold")} for m in pl.mounts},
            "lift": rows,
        }
    out["WILK reference (manufacturer-reported)"] = pu.WILK_REFERENCE
    return out


def stand_tests() -> dict:
    out = {}
    cases = [(p2.PEKARI_2M, "folded", {}, False), (p2.PEKARI_2M, "deployed", {"L": 5.0, "R": 5.0}, False),
             (pu.PUMBA, "folded", {}, False), (pu.PUMBA, "deployed", {"main": 30.0, "aux": 5.0}, False),
             (pu.PUMBA, "deployed", {"main": 30.0, "aux": 5.0}, True),
             (cr.CALYDON, "folded", {}, False), (cr.CALYDON, "deployed", {"main": 60.0, "aux": 8.0}, False),
             (cr.CALYDON, "deployed", {"main": 60.0, "aux": 8.0}, True)]
    for pl, pose, loads, st in cases:
        key = f"{pl.name} {pose}{' loads ' + str(loads) if loads else ''}{' stabilizers' if st else ''}"
        out[key] = pf.stand_test(pl, pose, loads=loads, stabilizers_deployed=st)
    return out


def fig_lift(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, pl in zip(axes, PLATFORMS):
        for m in pl.mounts:
            R = pa.reach(m.arm)
            rs = [R * (0.35 + 0.65 * i / 24) for i in range(25)]
            ax.plot(rs, [pa.rated_load(m.arm, r) for r in rs], label=f"{m.name}: arm (peak / 1.5)")
            ax.plot(rs, [pa.rated_load(m.arm, r, "hold") for r in rs], "--", label=f"{m.name}: arm continuous")
            for st in ((False, True) if pl.stabilizers else (False,)):
                yaw = math.copysign(math.pi / 2, m.y)
                ax.plot(rs, [min(pf.TIP_RATIO * pf.tipping_load(pl, m.name, r, yaw, stabilizers_deployed=st), 500) for r in rs],
                        ":", label=f"{m.name}: 75 % tipping, sideways{' + stabilizers' if st else ''}")
        ax.set_title(f"{pl.name}: what an arm may lift")
        ax.set_xlabel("horizontal reach from the shoulder [m]")
        ax.set_ylabel("load in the gripper [kg]")
        ax.set_ylim(0, {"Pekari-2M": 15, "Pumba": 80, "Calydon": 160}[pl.name])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def fig_workspace(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, pl in zip(axes, PLATFORMS):
        for m, c in zip(pl.mounts, ("tab:orange", "tab:blue")):
            sh = pf.shoulder(pl, m)
            for elev, mk, lab in ((-math.pi / 2, "s", "tool down"), (0.0, ".", "tool level")):
                pts = pa.workspace(m.arm, elev, n=45, z_span=(-sh[2], {"Pekari-2M": 1.0, "Pumba": 1.4, "Calydon": 1.6}[pl.name]))
                ax.scatter([m.x + r for r, _ in pts], [sh[2] + z for _, z in pts], s=4, marker=mk, color=c, alpha=0.35,
                           label=f"{m.name} ({lab})")
        g = pf.geometry(pl)
        ax.add_patch(plt.Rectangle((-pl.chassis["hull_length"] / 2e3, pl.chassis["ground_clearance"] / 1e3),
                                   pl.chassis["hull_length"] / 1e3, pl.chassis["hull_height"] / 1e3, color="0.3"))
        ax.plot([g["x_rear"], g["x_front"]], [0, 0], "k", lw=3)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{pl.name}: TCP workspace in the arm's plane (side view)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.legend(fontsize=7, markerscale=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def fig_scale(path):
    """The two platforms to scale from their CAD (folded and deployed): orthographic side views (x, z) of the
    tessellated solids, shaded by the facets' facing, with a 1.8 m person for scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import PolyCollection

    import calydon
    import pekari_2m
    import pumba

    shapes = []
    for cls, pl, color in ((pekari_2m.Pekari2M, p2.PEKARI_2M, (0.95, 0.45, 0.05)), (pumba.Pumba, pu.PUMBA, (0.98, 0.78, 0.05)),
                             (calydon.Calydon, cr.CALYDON, (0.85, 0.60, 0.10))):
        for pose, stab in (("folded", "stowed"), ("deployed", "deployed")):
            g = cls().generate(arm_pose=pose, stabilizers=stab)
            shp = g.shape.val() if hasattr(g.shape, "val") else g.shape
            verts, tris = shp.tessellate(2.0, 0.5)
            v = np.array([(p.x, p.y, p.z) for p in verts]) / 1e3
            shapes.append((f"{pl.name}, {pose}", v, np.array(tris), np.array(color)))
    fig, ax = plt.subplots(figsize=(20, 3.6))
    x0 = 0.0
    for name, v, tris, color in shapes:
        dx = x0 - v[:, 0].min()
        tri = v[tris]                                             # (n, 3, 3)
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        order = np.argsort(tri[:, :, 1].mean(axis=1))[::-1]      # far side (+y) first: the near side on top
        light = 0.45 + 0.55 * np.clip(-nrm[:, 1] * 0.6 + nrm[:, 2] * 0.5 + nrm[:, 0] * 0.2, 0, 1)
        polys = tri[order][:, :, [0, 2]] + np.array([dx, 0.0])
        ax.add_collection(PolyCollection(polys, facecolors=np.clip(np.outer(light[order], color), 0, 1), linewidths=0))
        ax.text(x0 + (v[:, 0].max() - v[:, 0].min()) / 2, v[:, 2].max() + 0.08, name, ha="center", fontsize=9)
        x0 += v[:, 0].max() - v[:, 0].min() + 0.5
    ax.plot([x0 + 0.2, x0 + 0.2], [0, 1.8], color="0.25", lw=8, solid_capstyle="butt")
    ax.text(x0 + 0.2, 1.86, "1.8 m", ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlim(-0.2, x0 + 0.6)
    ax.set_ylim(-0.05, 2.05)
    ax.set_aspect("equal")
    ax.set_xlabel("m")
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_title("Pekari-2M, Pumba and Calydon to scale (side views of the CAD; stabilizers deployed with the arms)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-mujoco", action="store_true")
    ap.add_argument("--no-cad", action="store_true")
    ap.add_argument("--docs", action="store_true", help="copy the figures to docs/figures/")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    res = numbers()
    if not a.no_mujoco:
        res["stand_tests"] = stand_tests()
    (OUT / "pekari_family.json").write_text(json.dumps(res, indent=1, default=str))
    figs = [("pekari_family_lift.png", fig_lift), ("pekari_family_workspace.png", fig_workspace)]
    if not a.no_cad:
        figs.append(("pekari_family_scale.png", fig_scale))
    for name, fn in figs:
        fn(OUT / name)
        if a.docs:
            DOCS.mkdir(parents=True, exist_ok=True)
            shutil.copy(OUT / name, DOCS / name)
    for pl in PLATFORMS:
        s = res[pl.name]["summary"]
        print(f"{pl.name}: {s['mass_empty_kg']:.0f} kg empty ({pu.chassis_mass(pl):.0f} kg without arms and stabilizers), {s['mass_loaded_kg']:.0f} kg loaded, "
              f"{s['overall_m'][0]:.2f} x {s['overall_m'][1]:.2f} x {s['overall_m'][2]:.2f} m folded, "
              f"{s['v_max_m_s'] * 3.6:.1f} km/h, {s['endurance_h']:.1f} h")
    for k, v in res.get("stand_tests", {}).items():
        print(f"  stand {k}: tilt {v['tilt_deg']:.3f} deg, tracks carry {v['track_share']:.0%}, arm error {v['arm_err_deg']:.2f} deg")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
