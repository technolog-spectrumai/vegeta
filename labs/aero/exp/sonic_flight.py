"""Vertical flight of what leaves the blowdown pipe (notebook 51): kinetic energy at the muzzle and along the next 50 m
of climb, under gravity and air drag.

m dv/dt = -m g - ½ ρ v² Cd(M) A, with v up and dh = v dt. The air is still at sea level (ρ 1.2 kg/m³, a 343 m/s).

Drag coefficients on the frontal area, against Mach. They are rough engineering values, not measured here:
- **sphere**: 0.47 subcritical, rising through the transonic range to ~0.95 at Mach 1 (sphere drag curves, e.g.
  Hoerner, *Fluid-Dynamic Drag*, ch. 16; Miller & Bailey 1979).
- **cylinder + head, nose first**: forebody + blunt base + skin friction. Base drag 0.12 subsonic, rising to ~0.25 at
  Mach 1. Hemisphere forebody ~0.02 subsonic, paraboloid ~0.01, both rising past Mach 0.8 (the hemisphere more).
  Skin friction 0.004 per unit of wetted / frontal area (4 L/D). Hoerner chs. 3, 13, 16 for the orders of magnitude.
- **tumbling**: a long body without fins is statically unstable nose first (uniform rubber has its centre of mass
  mid-body). The worst case is broadside: Cd 1.2 on the side area D·L, averaged with the nose-first value.

The point is a comparison between shapes, and with the ±30 % these values carry the ranking is what to trust.
"""
from __future__ import annotations

import math

import numpy as np

G, RHO, A_SOUND = 9.81, 1.2, 343.0


def _ramp(M, sub, sonic, m0=0.6, m1=1.0):
    """sub below m0, sonic from m1, smooth (cosine) in between."""
    s = np.clip((M - m0) / (m1 - m0), 0.0, 1.0)
    return sub + (sonic - sub) * 0.5 * (1 - np.cos(np.pi * s))


def cd_sphere(M):
    return _ramp(M, 0.47, 0.95, 0.5, 1.0)


def cd_body(M, head: str, l_over_d: float):
    nose = _ramp(M, 0.02, 0.25, 0.8, 1.1) if head == "hemisphere" else _ramp(M, 0.01, 0.12, 0.85, 1.1)
    base = _ramp(M, 0.12, 0.25, 0.6, 1.0)
    return nose + base + 0.004 * 4 * l_over_d


def drag_area(ball, M, tumbling: bool = False) -> float:
    """Cd·A [m²] of a ``sonic_ball.Ball`` at Mach M."""
    D = ball.diameter * 1e-3
    A = math.pi / 4 * D * D
    if ball.shape == "sphere":
        return cd_sphere(M) * A
    cda = cd_body(M, ball.shape, ball.L / ball.diameter) * A
    if tumbling:
        cda = 0.5 * (cda + 1.2 * D * ball.L * 1e-3)
    return cda


def climb(ball, v0: float, height: float = 50.0, tumbling: bool = False, dt: float = 2e-4) -> dict:
    """Vertical climb from the muzzle at v0 [m/s]: h, v, kinetic energy, time; stops at ``height`` or the apex."""
    m = ball.mass
    h, v, t = 0.0, v0, 0.0
    rows = [(t, h, v)]
    while v > 0 and h < height:
        def acc(v):
            return -G - 0.5 * RHO * v * v * drag_area(ball, v / A_SOUND, tumbling) / m
        k1 = acc(v); k2 = acc(v + 0.5 * dt * k1); k3 = acc(v + 0.5 * dt * k2); k4 = acc(v + dt * k3)
        dv = dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        h += dt * (v + 0.5 * dv)
        v += dv
        t += dt
        rows.append((t, h, v))
    a = np.array(rows)
    out = {"t": a[:, 0], "h": a[:, 1], "v": np.maximum(a[:, 2], 0.0)}
    out["E"] = 0.5 * m * out["v"] ** 2
    out["reached"] = out["h"][-1] >= height - 1e-6
    out["apex"] = None if out["reached"] else out["h"][-1]
    return out


def energy_at(fl: dict, heights) -> np.ndarray:
    """Kinetic energy [J] at the given heights; 0 past the apex."""
    return np.array([np.interp(z, fl["h"], fl["E"]) if z <= fl["h"][-1] else 0.0 for z in heights])


# -- every body, one comparison --------------------------------------------------------------------------------------
def exit_states(root=None, contact_pressure: float = 0.15e6) -> dict:
    """Exit speed of each body: the coupled CFD's gas force plus rubber friction where its run has finished ("CFD"),
    else the 0-D gun model scaled by the CFD/0-D ratio of the finished runs ("0-D, calibrated")."""
    from pathlib import Path
    import sonic_ball as sb, sonic_flow as sf
    root = Path(root or sb.HERE / "output" / "sonic_ball")
    bodies = {f"ball Ø{d:g} mm": (d, sb.bore_setup(d)[1], root / f"sweep_air_d{d:g}") for d in sb.SWEEP_DIAMETERS}
    for head in ("hemisphere", "paraboloid"):
        for ld in sb.LD_VALUES:
            b = sb.projectile(head, ld, contact_pressure=contact_pressure)
            bodies[f"{head} L/D {ld}"] = (10.0, b, root / f"shape_air_{head}_ld{ld}")
    out, ratios, pending = {}, [], []
    for name, (d, b, case) in bodies.items():
        rig = sb.bore_setup(d)[0]
        travel = (rig.pipe_length - b.rear_x - b.diameter) * 1e-3
        g = sb.gun(sf.AIR, rig, ball=b)
        if (case / "ball_done.json").exists():
            x, F = sb.force_vs_travel(case)
            v = exit_with_friction_(x, F, b, travel)
            out[name] = dict(ball=b, v=v, source="CFD + friction")
            ratios.append(v / g["u"][-1] if g["u"][-1] > 0 else 1.0)
        elif b.shape == "hemisphere" and all((root / f"shape_air_paraboloid_ld{ld}" / "ball_done.json").exists() for ld in sb.LD_VALUES):
            # in the bore only the mass matters: the gas force of the paraboloid runs, interpolated to this mass
            para = {"mass": [], "x": [], "F": []}
            for ld in sb.LD_VALUES:
                x, F = sb.force_vs_travel(root / f"shape_air_paraboloid_ld{ld}")
                para["mass"].append(sb.projectile("paraboloid", ld).mass); para["x"].append(x); para["F"].append(F)
            x, F = sb.force_for_mass(para, b.mass)
            out[name] = dict(ball=b, v=exit_with_friction_(x, F, b, travel), source="CFD (mass-interpolated) + friction")
        else:
            pending.append((name, b, g["u"][-1]))
    k = float(np.mean(ratios)) if ratios else 1.0
    for name, b, v0 in pending:
        out[name] = dict(ball=b, v=k * v0, source=f"0-D × {k:.2f}")
    return out


def exit_with_friction_(x, F, b, travel):
    import sonic_ball as sb
    return sb.exit_with_friction(x, F, b.mass, b.friction, travel)


def compare_plots(states: dict, select=None, height: float = 50.0, tumbling: bool = False, title: str = ""):
    """Two figures, every selected body on the same axes: velocity vs height and kinetic energy vs height, the exit
    values in the legend. ``select``: names (or substrings) to include; default all."""
    import matplotlib.pyplot as plt
    names = [n for n in states if select is None or any(s in n for s in select)]
    cmaps = {"ball": plt.get_cmap("Greens"), "hemisphere": plt.get_cmap("Blues"), "paraboloid": plt.get_cmap("Oranges")}
    groups = {g: [n for n in names if n.startswith(g)] for g in cmaps}
    figs = []
    for qty, unit in (("v", "velocity [m/s]"), ("E", "kinetic energy [J]")):
        fig, ax = plt.subplots(figsize=(11, 6.2))
        for g, ns in groups.items():
            for i, n in enumerate(ns):
                s = states[n]
                f = climb(s["ball"], s["v"], height, tumbling and s["ball"].shape != "sphere")
                c = cmaps[g](0.35 + 0.65 * i / max(1, len(ns) - 1))
                lab = f"{n}: {s['ball'].mass*1e3:.2f} g, exit {s['v']:.0f} m/s, {f['E'][0]:.1f} J" + ("" if s["source"].startswith("CFD") else f" [{s['source']}]")
                ls = "-" if s["source"].startswith("CFD") else "--"
                ax.plot(f["h"], f[qty], color=c, ls=ls, lw=1.8, label=lab)
                if not f["reached"]:
                    ax.plot(f["h"][-1], 0, marker="x", color=c, ms=7)
        ax.set_xlim(0, height); ax.set_ylim(bottom=0)
        ax.set_xlabel("height above the muzzle [m]"); ax.set_ylabel(unit)
        ax.set_title((title or "Vertical climb after the muzzle") + (" — tumbling" if tumbling else " — nose first")
                     + f": {'velocity' if qty == 'v' else 'kinetic energy'} (× = apex below 50 m)", loc="left", fontsize=11)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
        ax.grid(True, color="#e4e3df")
        fig.tight_layout()
        figs.append(fig)
    return figs


def compare_html(states: dict, path, height: float = 50.0) -> None:
    """A self-contained interactive page (Plotly): velocity and kinetic energy vs height for every body, nose first
    and tumbling; click legend entries to show or hide bodies, double-click to isolate one."""
    import json
    traces = {"v": [], "E": []}
    groups = {"ball": ["#a1d99b", "#41ab5d", "#006d2c", "#00441b"],
              "hemisphere": ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b", "#041f4a"],
              "paraboloid": ["#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#a63603", "#7f2704", "#5a1b02"]}
    count = {g: 0 for g in groups}
    for n, s in states.items():
        g = n.split()[0]
        col = groups[g][count[g] % len(groups[g])]; count[g] += 1
        for tumbling in (False, True):
            if tumbling and s["ball"].shape == "sphere":
                continue
            f = climb(s["ball"], s["v"], height, tumbling)
            k = max(1, len(f["h"]) // 300)
            name = f"{n}" + (" (tumbling)" if tumbling else "")
            hover = f"{n}<br>{s['ball'].mass*1e3:.2f} g, exit {s['v']:.0f} m/s, {f['E'][0]:.1f} J<br>{s['source']}"
            for q in ("v", "E"):
                traces[q].append(dict(x=[round(x, 3) for x in f["h"][::k].tolist()], y=[round(y, 4) for y in f[q][::k].tolist()],
                                      name=name, legendgroup=n, mode="lines", text=hover,
                                      hovertemplate="%{text}<br>h %{x:.1f} m: %{y:.2f}<extra></extra>",
                                      line=dict(color=col, width=2, dash="dot" if tumbling else ("solid" if s["source"].startswith("CFD") else "dash")),
                                      visible=("legendonly" if tumbling else True)))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Projectile climb comparison</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>body{{font-family:system-ui,sans-serif;margin:16px;background:#fcfcfb;color:#0b0b0b}} .p{{height:560px;max-width:1200px}}
p{{color:#52514e;max-width:1100px}}</style></head><body>
<h2>What leaves the tube: vertical climb, 0–{height:g} m</h2>
<p>Rubber bodies from the 50 bar blowdown pipe. Click a legend entry to show or hide a body; double-click to isolate it.
Tumbling variants (dotted) are hidden at start. Solid lines: exit speed from the coupled CFD + rubber friction. Dashed: CFD not finished yet, 0-D model calibrated on the finished runs.</p>
<div id="v" class="p"></div><div id="E" class="p"></div>
<script>
const T = {json.dumps(traces)};
const lay = (t, y) => ({{title: t, xaxis: {{title: 'height above the muzzle [m]', range: [0, {height}]}}, yaxis: {{title: y, rangemode: 'tozero'}},
  paper_bgcolor: '#fcfcfb', plot_bgcolor: '#fcfcfb', legend: {{font: {{size: 10}}}}, margin: {{t: 50}}}});
Plotly.newPlot('v', T.v, lay('Velocity vs height', 'velocity [m/s]'), {{responsive: true}});
Plotly.newPlot('E', T.E, lay('Kinetic energy vs height', 'kinetic energy [J]'), {{responsive: true}});
</script></body></html>"""
    from pathlib import Path
    Path(path).write_text(html)
