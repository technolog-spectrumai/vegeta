"""GAVILAN's pictures (notebook 38): the 3D render (matplotlib, from the CadQuery tessellation: flying, from above,
from the side, and standing on its fins) and the three-view with its dimensions.

    python gavilan_drawings.py [folder]        # writes gavilan_3d.png and gavilan_three_view.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import gavilan as gv

COLOURS = {"fuselage": ("#8d99a6", 1.0), "emitter": ("#111111", 1.0), "body": ("#8d99a6", 1.0), "wing": ("#c9a66b", 1.0), "tail": ("#6b4f2a", 1.0), "pylons": ("#3b3f45", 1.0),
           "camera": ("#111111", 1.0), "props": ("#7fb2e5", 0.30)}
FILL = {"boom": "#6b4f2a", "body": "#8d99a6", "wing": "#c9a66b", "fins": "#6b4f2a", "pylons": "#3b3f45", "pods": "#3b3f45", "camera": "#111111"}


def _tris(shape, tol=0.8):
    out = []
    for s in shape.solids().vals():
        verts, tris = s.tessellate(tol, 0.3)
        v = np.array([(q.x, q.y, q.z) for q in verts])
        out += [v[list(t)] for t in tris]
    return out


def render_f(out="gavilan_f_3d.png", q=None, name="Gavilan-F — fixed wing, twin tractor, the emitter in the nose"):
    """The fixed wing from three sides."""
    q = gv.f_resolve(q)
    parts = gv.GavilanF().parts(q)
    fig = plt.figure(figsize=(16, 6.2))
    for k, (elev, azim, title) in enumerate([(24, -128, "flying, from the front quarter"), (90, -90, "from above"), (0, -90, "from the side")]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for part, shape in parts.items():
            col, alpha = COLOURS[part]
            ax.add_collection3d(Poly3DCollection(_tris(shape), facecolors=col, edgecolors=col, linewidths=0.0, alpha=alpha, shade=alpha == 1.0))
        h, cx = 0.5 * q["span"], -q["nose_length"] + q["fuselage_length"] / 2
        ax.set_xlim(cx - h, cx + h); ax.set_ylim(-h, h); ax.set_zlim(-h - 30, h - 30)
        ax.set_box_aspect((1, 1, 1), zoom=1.5); ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=11)
    fig.suptitle(f"{name}\nspan {q['span']:.0f} mm, length {q['fuselage_length']:.0f} mm, two 7-inch tractors at y ±{q['nacelle_y']:.0f} mm; "
                 f"the emitter's tube on the body axis: muzzle at the nose, rear jet out of the tail", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def draw(ax, p, elev, azim, title, standing=False):
    """One 3D view; ``standing``: nose up on the fins (+x turned to −z, the feet on z = 0)."""
    L = p["length"]
    for name, shape in gv.Gavilan().parts(p).items():
        col, alpha = COLOURS[name]
        if standing:
            shape = shape.rotate((0, 0, 0), (0, 1, 0), 90).translate((0, 0, L))      # +x (aft) to −z: the nose up
        ax.add_collection3d(Poly3DCollection(_tris(shape), facecolors=col, edgecolors=col, linewidths=0.0, alpha=alpha, shade=alpha == 1.0))
    h = L * 0.52
    cx = (0, 0, L / 2) if standing else (L / 2, 0, 0)
    ax.set_xlim(cx[0] - h, cx[0] + h); ax.set_ylim(cx[1] - h, cx[1] + h); ax.set_zlim(cx[2] - h, cx[2] + h)
    ax.set_box_aspect((1, 1, 1), zoom=1.55); ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=11)


def render(out="gavilan_3d.png", p=None, caption=None, name="GAVILAN"):
    p = gv.resolve(p)
    L = gv.Gavilan.layout(p)
    fig = plt.figure(figsize=(15, 11))
    views = [(24, -128, "flying, from the front quarter", False, {}), (90, -90, "from above", False, {}),
             (0, -90, "from the side", False, {}), (14, -60, "standing on its fins, the camera on the horizon", True, {"camera_tilt_deg": 90.0})]
    for k, (elev, azim, title, standing, over) in enumerate(views):
        draw(fig.add_subplot(2, 2, k + 1, projection="3d"), {**p, **over}, elev, azim, title, standing)
    fig.suptitle(caption or (f"{name}\nfour fixed motors, two above and two below the wing; span {p['span']:.0f} mm, "
                             f"length {p['length']:.0f} mm, wing {L['S_mm2'] / 1e6:.3f} m² (aspect ratio {L['AR']:.1f}), "
                             f"7-inch propellers; the only servo tilts the camera"), fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def three_view(out="gavilan_three_view.png", p=None, name="GAVILAN"):
    p = gv.resolve(p)
    o, L = gv.outline(p), gv.Gavilan.layout(p)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.4), gridspec_kw={"width_ratios": [o["length"], o["length"], o["width"] * 1.15]})
    for a, view, label in zip(ax, ("top", "side", "front"), ("top (x aft, y right)", "side (x aft, z up)", "front (y right, z up)")):
        for key in ("boom", "fins", "wing", "pylons", "body", "pods", "camera"):
            for poly in o[view].get(key, []):
                a.fill(poly[:, 0], poly[:, 1], color=FILL[key], alpha=0.55 if key in ("wing", "fins") else 0.9, ec="k", lw=0.6)
        for (x0, y0), (x1, y1) in o[view].get("prop_lines", []):
            a.plot([x0, x1], [y0, y1], color="#1f77b4", lw=2)
        for c in o[view].get("props", []):
            a.plot(c[:, 0], c[:, 1], color="#1f77b4", lw=1.0)
        a.set_aspect("equal"); a.grid(alpha=0.25); a.set_title(label, fontsize=10); a.set_xlabel("[m]")
    m = 0.06
    ax[0].annotate("", (0, -o["span"] / 2 - m), (o["length"], -o["span"] / 2 - m), arrowprops=dict(arrowstyle="<->"))
    ax[0].text(o["length"] / 2, -o["span"] / 2 - m - 0.012, f"length {p['length']:.0f} mm", ha="center", va="top", fontsize=9)
    ax[0].annotate("", (-m, -o["span"] / 2), (-m, o["span"] / 2), arrowprops=dict(arrowstyle="<->"))
    ax[0].text(-m - 0.012, 0, f"span {p['span']:.0f} mm", ha="right", va="center", rotation=90, fontsize=9)
    ax[0].set_xlim(-0.14, o["length"] + 0.04); ax[0].set_ylim(-o["span"] / 2 - 0.12, o["span"] / 2 + 0.05)
    ax[1].annotate("", (o["length"] + 0.03, -L["foot_half_width"] / 1e3), (o["length"] + 0.03, L["foot_half_width"] / 1e3), arrowprops=dict(arrowstyle="<->"))
    ax[1].text(o["length"] + 0.04, 0, f"feet {2 * L['foot_half_width']:.0f} mm", rotation=90, va="center", fontsize=9)
    ax[1].set_xlim(-0.05, o["length"] + 0.09)
    ax[2].text(0, -o["height"] / 2 - 0.03, f"propellers {p['prop_diameter']:.0f} mm, axes at y ±{p['motor_y']:.0f}, z ±{p['motor_z']:.0f} mm\n"
               f"tips: {L['tip_to_wing']:.0f} mm to the wing, {L['tip_to_tip']:.0f} mm to each other, {L['tip_to_body']:.0f} mm to the body",
               ha="center", va="top", fontsize=8)
    ax[2].set_ylim(-o["height"] / 2 - 0.11, o["height"] / 2 + 0.03)
    fig.suptitle(f"{name} three-view — wing {L['S_mm2'] / 1e6:.3f} m², aspect ratio {L['AR']:.2f}, root chord {L['c_root']:.0f} mm, "
                 f"tip chord {L['c_tip']:.0f} mm, body ⌀ {p['body_diameter']:.0f} mm", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------- the comparison
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")   # fixed order, one per aircraft
CHARTED = ("merlin", "peregrine", "xwing", "emerillon", "emerillon_armed", "gavilan_a", "gavilan_f", "gavilan_f_broad")
SHORT = {"merlin": "Merlin", "peregrine": "Peregrine B", "xwing": "X-wing X-A", "emerillon": "Émerillon ER2",
         "emerillon_armed": "Émerillon ER2 + emitter", "gavilan_a": "Gavilan-A + emitter", "gavilan_f": "Gavilan-F + emitter",
         "gavilan_f_broad": "Gavilan-F broad + emitter", "gavilan_a_unarmed": "Gavilan-A, no emitter", "gavilan_b": "Gavilan-B + emitter"}


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)


def turn_chart(out, machines, speeds=None):
    """Turn rate and turn radius against speed, instantaneous and sustained, one line per aircraft."""
    import gavilan_flight as gf
    speeds = np.arange(6.0, 40.5, 1.0) if speeds is None else speeds
    keys = [k for k in CHARTED if k in machines]
    data = {k: [gf.turn(machines[k], float(V)) for V in speeds] for k in keys}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5), sharex=True, facecolor=SURFACE)
    panels = (("rate_inst_deg_s", "instantaneous turn rate", "deg/s"), ("rate_sus_deg_s", "sustained turn rate (the speed holds)", "deg/s"),
              ("radius_inst_m", "instantaneous turn radius", "m"), ("radius_sus_m", "sustained turn radius", "m"))
    for a, (col, title, unit) in zip(ax.flat, panels):
        _style(a, title, ylabel=unit)
        for k, c in zip(keys, SERIES):
            y = np.array([r[col] for r in data[k]], float)
            y[(~np.isfinite(y)) | (y <= 0)] = np.nan
            a.plot(speeds, y, color=c, lw=2, label=SHORT[k])
        for V in (10, 15, 25):
            a.axvline(V, color=MUTED, lw=0.7, ls=(0, (2, 3)))
        if "radius" in col:
            a.set_ylim(0, 40)
    for a in ax[1]:
        a.set_xlabel("airspeed [m/s]", fontsize=9, color=MUTED)
    ax[0, 0].legend(fontsize=8.5, frameon=False, ncol=2, labelcolor=INK)
    fig.suptitle("Turning against speed — a line stops where the aircraft cannot hold a level turn (below its stall); dotted: 10, 15, 25 m/s",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return out


def bar_chart(out, rows):
    """Six comparisons as horizontal bars: one bar per aircraft, the value written at its end."""
    by = {r["key"]: r for r in rows}
    keys = [k for k in CHARTED if k in by]
    panels = (("static thrust / weight", "static thrust / weight", "{:.1f}", None),
              ("min speed [m/s]", "slowest level flight [m/s] (0: it hovers)", "{:.0f}", None),
              ("top speed [m/s]", "top speed [m/s]", "{:.0f}", None),
              ("150 m climb [s]", "ground to 150 m [s]", "{:.1f}", None),
              ("reverse at 15 m/s [s]", "reverse direction at 15 m/s [s]", "{:.1f}", None),
              ("re-attack [s]", "next firing pass, hovering bandit [s] (0: it holds it)", "{:.1f}", None))
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.2), facecolor=SURFACE)
    y = np.arange(len(keys))[::-1]
    for a, (col, title, fmt, _) in zip(ax.flat, panels):
        _style(a, title)
        vals = [by[k][col] for k in keys]
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        top = max(finite) if finite else 1.0
        for yi, k, v, c in zip(y, keys, vals, SERIES):
            ok = v is not None and np.isfinite(v)
            a.barh(yi, v if ok else 0.0, height=0.62, color=c)
            a.text((v if ok else 0.0) + 0.015 * top, yi, fmt.format(v) if ok else "none", va="center", fontsize=8.5, color=INK)
        a.set_yticks(y); a.set_yticklabels([SHORT[k] for k in keys], fontsize=8.5, color=INK)
        a.set_xlim(0, 1.14 * top); a.grid(axis="y", visible=False)
    fig.suptitle("The fleet on the chase's numbers — as designed; '+ emitter' carries one 0.6 kg recoilless emitter", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return out


COLOR_OF = {"merlin": SERIES[0], "peregrine": SERIES[1], "xwing": SERIES[2], "emerillon": SERIES[3], "gavilan_b": SERIES[4],
            "gavilan_a": SERIES[5], "gavilan_f": SERIES[6], "gavilan_f_broad": SERIES[7]}       # the colour follows the aircraft
NAME_OF = {"merlin": "Merlin", "peregrine": "Peregrine B", "xwing": "X-wing X-A", "emerillon": "Émerillon ER2", "gavilan_a": "Gavilan-A",
           "gavilan_b": "Gavilan-B", "gavilan_f": "Gavilan-F", "gavilan_f_broad": "Gavilan-F broad"}


def kpi_bars(out, rows, group, title):
    """One panel per KPI of ``group``: a bar per aircraft, the value at its end, the best one marked."""
    import gavilan_flight as gf
    kpis = [k for k in gf.KPIS if k["group"] == group]
    cols = 3
    fig, ax = plt.subplots(-(-len(kpis) // cols), cols, figsize=(16.5, 3.5 * -(-len(kpis) // cols) + 0.7), facecolor=SURFACE, squeeze=False)
    y = np.arange(len(rows))[::-1]
    for a, k in zip(ax.flat, kpis):
        _style(a, k["label"])
        vals = [r[k["key"]] for r in rows]
        have = [v for v in vals if v is not None]
        top = max(have) if have and max(have) > 0 else 1.0
        best = max(have) if k["better"] == "high" else min(have)
        for yi, r, v in zip(y, rows, vals):
            a.barh(yi, v if v is not None else 0.0, height=0.62, color=COLOR_OF[r["key"]])
            txt = "cannot" if v is None else (f"{v:.0f}" if abs(v) >= 20 else f"{v:.1f}" if abs(v) >= 2 else f"{v:.2f}")
            mark = "  ◀ best" if v is not None and abs(v - best) < 1e-9 else ""
            a.text((v or 0.0) + 0.015 * top, yi, txt + mark, va="center", fontsize=8.5, color=INK)
        a.set_yticks(y); a.set_yticklabels([NAME_OF[r["key"]] for r in rows], fontsize=8.5, color=INK)
        a.set_xlim(0, 1.25 * top); a.grid(axis="y", visible=False)
        a.text(0.0, -0.13, ("higher is better" if k["better"] == "high" else "lower is better") + " — " + k["why"], transform=a.transAxes,
               fontsize=8, color=MUTED, va="top")
    for a in ax.flat[len(kpis):]:
        a.set_visible(False)
    fig.suptitle(title, fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.2)
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return out


def kpi_score_bars(out, rows):
    """The two group scores side by side: each KPI scaled to the fleet's best, plain mean, equal weights."""
    import gavilan_flight as gf
    sc = gf.kpi_scores(rows)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE)
    for a, g, title in zip(ax, ("bandit", "general"), ("against bandits: mean of the 9 KPIs", "general: mean of the 6 KPIs")):
        _style(a, title)
        order = sorted(rows, key=lambda r: sc[r["key"]][g])
        for yi, r in enumerate(order):
            v = sc[r["key"]][g]
            a.barh(yi, v, height=0.62, color=COLOR_OF[r["key"]])
            a.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
        a.set_yticks(range(len(order))); a.set_yticklabels([NAME_OF[r["key"]] for r in order], fontsize=9, color=INK)
        a.set_xlim(0, 1.1); a.grid(axis="y", visible=False)
    fig.suptitle("Scores, unarmed — each KPI scaled so the fleet's best is 1 and 'cannot' is 0, then averaged with equal weights",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return out


def kpi_charts(folder=".", machines=None):
    import gavilan_flight as gf
    folder = Path(folder)
    rows = gf.kpi_table(machines)
    return [kpi_bars(folder / "gavilan_kpi_bandit.png", rows, "bandit", "KPIs against bandits — every aircraft unarmed"),
            kpi_bars(folder / "gavilan_kpi_general.png", rows, "general", "General KPIs — every aircraft unarmed"),
            kpi_score_bars(folder / "gavilan_kpi_scores.png", rows)], rows


def comparison_charts(folder=".", machines=None):
    import gavilan_flight as gf
    folder = Path(folder)
    machines = machines or gf.fleet()
    rows = gf.scorecard(machines)
    return [turn_chart(folder / "gavilan_fleet_turns.png", machines), bar_chart(folder / "gavilan_fleet_bars.png", rows)], rows


def main(folder="."):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for key in gv.VARIANTS:
        tag = "" if key == "A" else f"_{key.lower()}"
        out += [render(folder / f"gavilan{tag}_3d.png", gv.variant(key), name=gv.VARIANT_NAME[key]),
                three_view(folder / f"gavilan{tag}_three_view.png", gv.variant(key), name=gv.VARIANT_NAME[key])]
    out.append(render_f(folder / "gavilan_f_3d.png"))
    return out


if __name__ == "__main__":
    print(*main(*sys.argv[1:]), sep="\n")
