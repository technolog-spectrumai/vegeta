"""Halka solved in Fulgur (``docs/plasma.md``, P3): the ring section of a preset meshed with Gmsh and the corona
solved on FiPy — the field with its space charge, the ions, the current and the force on the air — against the 1-D
model and the video's numbers.

    python halka_fulgur.py [out.png]        # the sweep and the figure (minutes; seconds under vegeta.mock)

``solve_section(preset, V)`` is the unit of work; ``sweep`` runs the voltages, ``offset_sweep`` the convergence
offset dz (the lever the 2-D solve found: thrust = I dz / mu). Under ``vegeta.mock`` the solve returns the 1-D
model's numbers, labelled mock.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from vegeta.fulgur import Air, DRY
from vegeta.fulgur.ehd import CoronaProblem, ring_section
from vegeta.fulgur.thruster import PRESETS

from halka import Halka

G = 9.81
EDGE_R = 0.1e-3                 # the emitter's effective edge radius (the teeth as an axisymmetric edge)
DOMAIN = dict(r_outer_k=1.6, z_lo=-0.06, z_hi=0.10)


def rings_of(preset: str, dz: float | None = None) -> list:
    """[(radius, role, z)] of the preset in metres, the grounds' offset replaced by ``dz`` when given."""
    t = PRESETS[preset]
    return [(r.radius, r.role, (dz if dz is not None else r.z) if r.role == "ground" else 0.0) for r in t.rings]


def section(preset: str = "mark4c", dz: float | None = None, edge_r: float = EDGE_R, lc: float = 1.0, params=None):
    """The Gmsh section of the preset: Halka's strip and ground sections, cells ``lc`` times the default size."""
    p = {q.name: q.default for q in Halka.parameters}
    p.update(params or {})
    t = PRESETS[preset]
    strip_h, ground_h = p["strip_h"] * 1e-3, p["ground_h"] * 1e-3
    ground_w = (p["ground_w"] if t.name != "mark4" else p["ground_t"]) * 1e-3
    return ring_section(rings_of(preset, dz), r_outer=DOMAIN["r_outer_k"] * t.outer_radius, z_lo=DOMAIN["z_lo"], z_hi=DOMAIN["z_hi"],
                        strip_h=strip_h, strip_t=max(p["strip_t"] * 1e-3, 2 * edge_r), ground_w=ground_w, ground_h=ground_h,
                        edge_r=edge_r, lc_edge=15e-6 * lc, lc_ground=0.5e-3 * lc, lc_far=6e-3 * lc)


def solve_section(preset: str, V: float, air: Air = DRY, dz: float | None = None, edge_r: float = EDGE_R, lc: float = 1.0,
                  problem=None, params=None) -> dict:
    """The corona of the preset at ``V``: current, thrust (the axial force on the air), the radial force, power,
    thrust per watt, the onset voltage of the modelled edge; ``problem`` reuses a built ``CoronaProblem``."""
    if problem is None:
        mesh, em, gr, edge = section(preset, dz, edge_r, lc, params)
        problem = CoronaProblem(mesh, em, gr, air.peek_field(edge_r, 0.9), air=air, geometry="planar_axi", inject=edge)
    problem.rho.setValue(0.0)                                   # the Laplace field of the modelled edge: its onset
    problem._solve_poisson(1e3)
    onset = problem.e_onset / problem.field_at(problem.inject) * 1e3
    r = problem.solve(V)
    return {"preset": preset, "V": V, "I": r.current, "thrust": r.force_z, "force_r": r.force_r, "power": r.power,
            "thrust_per_watt": r.thrust_per_watt, "onset": onset, "converged": r.converged, "iterations": r.iterations,
            "cells": problem.mesh.numberOfCells, "dz": dz, "edge_r": edge_r, "problem": problem}


def sweep(preset: str, voltages, air: Air = DRY, dz: float | None = None, lc: float = 1.0) -> list:
    out, prob = [], None
    for V in voltages:
        r = solve_section(preset, V, air, dz, lc=lc, problem=prob)
        prob = r["problem"]
        out.append(r)
    return out


def offset_sweep(preset: str, V: float, dzs, air: Air = DRY, lc: float = 1.0) -> list:
    return [solve_section(preset, V, air, dz, lc=lc) for dz in dzs]


def one_d(preset: str, V: float, air: Air = DRY, dz: float | None = None) -> dict:
    t = PRESETS[preset]
    if dz is not None:
        from dataclasses import replace
        t = replace(t, rings=tuple(replace(r, z=dz if r.role == "ground" else 0.0) for r in t.rings))
    return t.estimate(V, air)


# --- the mock: the 1-D model's numbers at once
def _mock_solve_section(preset, V, air=DRY, dz=None, edge_r=EDGE_R, lc=1.0, problem=None, params=None):
    from vegeta import mock
    e = one_d(preset, V, air, dz)
    return mock.label({"preset": preset, "V": V, "I": e["I"], "thrust": e["thrust"], "force_r": 0.0, "power": e["power"],
                       "thrust_per_watt": e["thrust_per_watt"], "onset": e["onset"], "converged": True, "iterations": 0,
                       "cells": 0, "dz": dz, "edge_r": edge_r, "problem": None})


try:
    from vegeta import mock as _mock
    _mock.on_install(lambda: _mock.patch(sys.modules[__name__], "solve_section", _mock_solve_section))
except ImportError:                                          # the toolkit without vegeta-cli
    pass


# --- the figure
def _tri(problem):
    import matplotlib.tri as mtri
    c = np.asarray(problem.mesh.cellCenters) * 1e3
    tri = mtri.Triangulation(c[0], c[1])
    # drop the triangles that bridge an electrode hole (long edges across it)
    x, y = c[0][tri.triangles], c[1][tri.triangles]
    edge = np.max([np.hypot(x[:, i] - x[:, (i + 1) % 3], y[:, i] - y[:, (i + 1) % 3]) for i in range(3)], axis=0)
    tri.set_mask(edge > 9.0)
    return tri


def main(out="halka_fulgur.png", preset="mark4c", voltages=(14e3, 17e3, 20e3, 23e3, 26e3), dzs=(0.008, 0.012, 0.016, 0.020)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BLUE, ORANGE, GREY = "#1f2328", "#6e7781", "#2f6db3", "#c8711a", "#9aa4ad"
    rows = sweep(preset, voltages)
    off = offset_sweep(preset, 20e3, dzs)
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3)
    # the fields at 20 kV
    r20 = min(rows, key=lambda r: abs(r["V"] - 20e3))
    prob = r20["problem"]
    if prob is not None:
        tri = _tri(prob)
        phi = np.asarray(prob.phi.value) / 1e3
        rho = np.asarray(prob.rho.value) * 1e3
        E = np.asarray(prob._E_cell()); emag = np.hypot(E[0], E[1]) / 1e6
        for k, (val, title, cmap, vmax) in enumerate([(phi, f"potential [kV] at {r20['V']/1e3:.0f} kV", "Blues", None),
                                                      (rho, "ion charge density [mC/m³]", "Oranges", np.percentile(rho[rho > 0], 99) if (rho > 0).any() else None),
                                                      (emag, "field [MV/m]; above 0.5 a streamer can propagate", "Greys", 2.0)]):
            ax = fig.add_subplot(gs[0, k])
            cf = ax.tripcolor(tri, val, cmap=cmap, vmin=0, vmax=vmax, shading="gouraud")
            fig.colorbar(cf, ax=ax, shrink=0.8)
            for rr, role, z in rings_of(preset):
                ax.plot([rr * 1e3], [z * 1e3], "s" if role == "ground" else "^", color=INK if role == "ground" else ORANGE, ms=5)
            ax.set_xlim(0, 160); ax.set_ylim(-40, 70); ax.set_aspect("equal")
            ax.set_xlabel("r [mm]"); ax.set_ylabel("z [mm] (air flows +z)"); ax.set_title(title, fontsize=10, loc="left", color=INK)
    # thrust and power vs V: 2-D and 1-D
    ax = fig.add_subplot(gs[1, 0])
    Vk = [r["V"] / 1e3 for r in rows]
    ax.plot(Vk, [r["thrust"] / G * 1e3 for r in rows], "o-", color=BLUE, lw=2, label="Fulgur 2-D solve")
    ax.plot(Vk, [one_d(preset, r["V"])["thrust"] / G * 1e3 for r in rows], "s--", color=GREY, lw=1.5, label="1-D model (I dz/μ)")
    ax.axhline(25, color=ORANGE, lw=1, ls=":"); ax.text(Vk[0], 25.6, "video: 25 gf on the pendulum", color=ORANGE, fontsize=8)
    ax.set_xlabel("voltage [kV]"); ax.set_ylabel("thrust [gf]"); ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", color="#e6e8eb")
    ax.set_title(f"{preset}: thrust", loc="left", color=INK, fontsize=10)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(Vk, [r["power"] for r in rows], "o-", color=BLUE, lw=2, label="Fulgur 2-D")
    ax.plot(Vk, [one_d(preset, r["V"])["power"] for r in rows], "s--", color=GREY, lw=1.5, label="1-D")
    ax.set_xlabel("voltage [kV]"); ax.set_ylabel("power [W]"); ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", color="#e6e8eb")
    ax.set_title("power drawn by the corona", loc="left", color=INK, fontsize=10)
    ax = fig.add_subplot(gs[1, 2])
    dzk = [r["dz"] * 1e3 for r in off]
    ax.bar(dzk, [r["thrust_per_watt"] * 1e3 for r in off], width=2.6, color=BLUE)
    for r in off:
        ax.text(r["dz"] * 1e3, r["thrust_per_watt"] * 1e3 + 0.1, f"{r['thrust']/G*1e3:.0f} gf\n{r['power']:.0f} W", ha="center", fontsize=8, color=MUTED)
    ax.set_xlabel("grounds' axial offset dz [mm] (the convergence)"); ax.set_ylabel("thrust per watt [mN/W] at 20 kV")
    ax.set_title("the lever: thrust = I dz / μ, T/P = dz / (μV)", loc="left", color=INK, fontsize=10); ax.grid(axis="y", color="#e6e8eb"); ax.set_axisbelow(True)
    fig.suptitle(f"Halka {preset} in Fulgur: the corona solved on the ring section (FiPy, {r20['cells']} cells); onset of the modelled edge {r20['onset']/1e3:.1f} kV", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    return out, rows, off


if __name__ == "__main__":
    out, rows, off = main(*sys.argv[1:2])
    for r in rows:
        print(f"{r['V']/1e3:5.1f} kV  I={r['I']*1e3:6.2f} mA  T={r['thrust']/G*1e3:5.1f} gf  P={r['power']:6.1f} W  T/P={r['thrust_per_watt']*1e3:.2f} mN/W  conv={r['converged']}")
    for r in off:
        print(f"dz={r['dz']*1e3:4.0f} mm at 20 kV  I={r['I']*1e3:5.2f} mA  T={r['thrust']/G*1e3:5.1f} gf  P={r['power']:5.1f} W  T/P={r['thrust_per_watt']*1e3:.2f} mN/W")
    print(out)
