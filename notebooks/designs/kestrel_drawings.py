"""Kestrel-4Q — the drawings of notebook 38 (matplotlib only).

- ``front_view``: looking at the nose — both wings, the pod, the posts and the X diagonals, the four propeller discs
  with their senses of rotation (the diagonal pairing), the CG.
- ``side_view``: the profile — one leading-edge station for both wings (zero stagger), the propeller plane, the CG,
  the neutral point, the feet and the ground when standing.
- ``episode_figure``: one transition against time — attitude and its reference, the angles of attack (free stream and
  the blown span), the airspeed and the height, the rotor thrusts against their limit, the pitch moments, the elevon,
  the roll check.
- ``path_figure``: transitions in the vertical plane with the aircraft's profile drawn at its attitude along the way.
- ``polar_figure``: the full-range polar (and the CFD's points when there are any).
"""
from __future__ import annotations

import math

import numpy as np

import kestrel

COL = {"wing": "#9fb8d0", "edge": "#40556b", "truss": "#202020", "disc": "#d0a040", "cg": "#c03030", "np": "#2a6fb0",
       "fwd": "#2a6fb0", "back": "#c06000"}


def front_view(p=None, *, cg=None, spins=(1, -1, -1, 1), ax=None):
    import matplotlib.pyplot as plt
    L = kestrel.layout(p)
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5))
    b2, h, ym, yin, R = L["span"] / 2, L["h"], L["motor_y"], L["pod_width"] / 2, L["R_prop"]
    t = 0.12 * L["chord"]
    for z in (h, -h):
        ax.add_patch(plt.Rectangle((-b2, z - t / 2), 2 * b2, t, color=COL["wing"], ec=COL["edge"]))
    ax.add_patch(plt.Rectangle((-yin, -h), 2 * yin, 2 * h, color="#e8e8e8", ec=COL["edge"], label="pod"))
    for s in (1, -1):
        ax.plot([s * ym, s * ym], [-h, h], color=COL["truss"], lw=2.5)
        ax.plot([s * yin, s * ym], [h, -h], color=COL["truss"], lw=1.5)
        ax.plot([s * yin, s * ym], [-h, h], color=COL["truss"], lw=1.5)
    for (name, m), sp in zip(L["motors"].items(), spins):
        ax.add_patch(plt.Circle((m["y"], m["z"]), R, color=COL["disc"], alpha=0.25))
        ax.add_patch(plt.Circle((m["y"], m["z"]), 0.017, color="#333"))
        ax.annotate("", xy=(m["y"] + 0.6 * R * (1 if sp > 0 else -1), m["z"] + 0.6 * R), xytext=(m["y"] - 0.6 * R * (1 if sp > 0 else -1), m["z"] + 0.6 * R),
                    arrowprops=dict(arrowstyle="->", color="#805000"))
        ax.text(m["y"], m["z"] - 0.045, name, ha="center", va="top", fontsize=9)
    if cg is not None:
        ax.plot(cg.get("y_cg", 0.0), cg["z_cg"], "x", color=COL["cg"], ms=12, mew=3, label="CG")
    ax.set(aspect="equal", xlabel="y [m] (right)", ylabel="z [m]", title="Kestrel-4Q, looking at the nose (arrows: the blades' motion at the disc's top)")
    ax.legend(loc="lower right", fontsize=8)
    return ax


def side_view(p=None, *, x_cg=None, x_np=None, ax=None):
    import matplotlib.pyplot as plt
    from kestrel import naca4
    pr = kestrel.resolve(p)
    L = kestrel.layout(p)
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5))
    c, xle, h = L["chord"], L["x_le"], L["h"]
    up, lo = naca4(pr["camber"], pr["camber_pos"], pr["thickness"])
    sec = np.array(up + lo[::-1])
    for z in (h, -h):
        ax.fill(xle + c * sec[:, 0], z + c * sec[:, 1], color=COL["wing"], ec=COL["edge"])
    ax.add_patch(plt.Rectangle((0, -0.49 * L["gap"]), L["pod_length"], 0.98 * L["gap"], color="#e8e8e8", ec=COL["edge"], zorder=0))
    xt, tc = L["x_feet"], L["fin_tip_chord"]
    fin = np.array([(L["fin_root_x"], h), (xt - tc, h + L["fin_height"]), (xt, h + L["fin_height"]), (xt, -h - L["ventral_height"]),
                    (xt - tc, -h - L["ventral_height"]), (L["fin_root_x"], -h)])
    ax.fill(fin[:, 0], fin[:, 1], color="#cfd8e0", ec=COL["edge"], zorder=0)
    for z in (h, -h):
        ax.plot([L["prop_x"]] * 2, [z - L["R_prop"], z + L["R_prop"]], color=COL["disc"], lw=3)
        ax.plot([xle + 0.6 * c, xt], [z, z], color=COL["truss"], lw=1.2)
    ax.axvline(xle, color="#888", ls=":", lw=1)
    ax.text(xle, h + L["R_prop"] + 0.02, "both leading edges\n(zero stagger)", ha="center", fontsize=8)
    ax.axvline(xt, color="#8d7b5e", lw=2)
    ax.text(xt + 0.01, 0, "the ground,\nstanding", fontsize=8, va="center")
    if x_cg is not None:
        ax.plot(x_cg, 0, "x", color=COL["cg"], ms=12, mew=3, label=f"CG, x = {x_cg * 1000:.0f} mm")
    if x_np is not None:
        ax.plot(x_np, 0, "o", color=COL["np"], ms=8, label=f"neutral point, x = {x_np * 1000:.0f} mm")
    ax.set(aspect="equal", xlabel="x [m] (aft from the nose tip)", ylabel="z [m]", title="Kestrel-4Q, side view (level flight; the thrust points to −x)")
    ax.legend(loc="lower left", fontsize=8)
    return ax


def episode_figure(ep, veh, title=None):
    import matplotlib.pyplot as plt
    g = ep.log
    t = g["t"]
    fig, axs = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    tr = g["mode"] == "transition"

    def shade(a):
        if tr.any():
            a.axvspan(t[tr][0], t[tr][-1], color="#f0e0c0", alpha=0.5, lw=0)
    a = axs[0, 0]; shade(a)
    a.plot(t, np.degrees(g["theta"]), label="pitch θ"); a.plot(t, np.degrees(g["theta_ref"]), "--", label="reference")
    a.set(ylabel="deg", title="attitude (90° = hovering nose-up)"); a.legend(fontsize=8)
    a = axs[0, 1]; shade(a)
    ok = g["V"] > 1.0
    a.plot(t[ok], np.degrees(np.abs(g["alpha"][ok])), label="free stream")
    a.plot(t[ok], np.degrees(np.abs(g["alpha_blown_up"][ok])), label="blown span, upper wing")
    a.plot(t[ok], np.degrees(np.abs(g["alpha_blown_lo"][ok])), label="blown span, lower wing")
    a.axhline(math.degrees(veh.pol.alpha_stall), color="k", ls=":", lw=1, label="stall")
    a.set(ylabel="deg", title="angle of attack", ylim=(0, 100)); a.legend(fontsize=8)
    a = axs[1, 0]; shade(a)
    a.plot(t, g["V"], label="airspeed [m/s]"); a2 = a.twinx(); a2.plot(t, g["Z"], color="#555", label="height [m]")
    a.set(ylabel="m/s", title="airspeed and height"); a2.set_ylabel("m"); a.legend(loc="upper left", fontsize=8); a2.legend(loc="lower right", fontsize=8)
    a = axs[1, 1]; shade(a)
    a.plot(t, g["T_up"], label="upper rotor"); a.plot(t, g["T_lo"], label="lower rotor"); a.plot(t, g["T_max"], "k:", label="T_max")
    a.axhline(veh.drive.T_min, color="k", ls="--", lw=0.8, label="idle")
    a.set(ylabel="N", title="rotor thrust (each)"); a.legend(fontsize=8)
    a = axs[2, 0]; shade(a)
    a.plot(t, g["M_aero"], label="aerodynamic"); a.plot(t, g["M_cmd"], label="commanded"); a.plot(t, g["M_got"], "--", label="allocated")
    a2 = a.twinx(); a2.plot(t, np.degrees(g["delta"]), color="#8a2be2", lw=0.8, label="elevon [deg]"); a2.set_ylabel("deg")
    a.set(xlabel="t [s]", ylabel="N·m", title="pitch moments about the CG"); a.legend(loc="upper left", fontsize=8); a2.legend(loc="lower right", fontsize=8)
    a = axs[2, 1]; shade(a)
    a.plot(t, g["roll_demand"], label="one-sided stall's rolling moment")
    a.plot(t, g["roll_authority"], label="authority (rotors + elevons)")
    a.plot(t, g["roll_authority_rotors"], ":", label="rotors only")
    a.set(xlabel="t [s]", ylabel="N·m", title="roll check (not simulated)"); a.legend(fontsize=8)
    fig.suptitle(title or f"{ep.vehicle}: {ep.kind} transition, {ep.t_tr:g} s, headwind {ep.headwind:g} m/s — {'PASS' if ep.metrics['pass'] else 'FAIL'}")
    fig.tight_layout()
    return fig


def _profile(L):
    """The level aircraft's side profile (pod box, the two wings' chords, the propeller discs) about the CG, in metres."""
    c, xle, h = L["chord"], L["x_le"], L["h"]
    return [np.array([[xle, h], [xle + c, h]]), np.array([[xle, -h], [xle + c, -h]]),
            np.array([[0, -0.45 * L["gap"]], [L["pod_length"], -0.45 * L["gap"]], [L["pod_length"], 0.45 * L["gap"]], [0, 0.45 * L["gap"]], [0, -0.45 * L["gap"]]]),
            np.array([[L["prop_x"], h - L["R_prop"]], [L["prop_x"], h + L["R_prop"]]]),
            np.array([[L["prop_x"], -h - L["R_prop"]], [L["prop_x"], -h + L["R_prop"]]])]


def path_figure(eps, veh, *, every_s=0.5, scale=4.0, x_max=None):
    """The flight paths (earth x, z), one row per episode, with the aircraft's profile at its attitude every
    ``every_s`` s (drawn ``scale`` × full size so it reads at the path's scale)."""
    import matplotlib.pyplot as plt
    L = veh.L
    fig, axs = plt.subplots(len(eps), 1, figsize=(15, 3.4 * len(eps)), squeeze=False)
    prof = _profile(L)
    for ax, ep in zip(axs[:, 0], eps):
        g = ep.log
        col = COL["fwd"] if ep.kind == "forward" else COL["back"]
        ax.plot(g["X"], g["Z"], color=col, lw=1.5, label=f"{ep.kind}, {ep.t_tr:g} s")
        dt = g["t"][1] - g["t"][0]
        for i in range(0, len(g["t"]), max(1, int(every_s / dt))):
            th = g["theta"][i]
            ct, st = math.cos(th), math.sin(th)
            for seg in prof:
                bx = seg[:, 0] - veh.x_cg                    # body x aft
                bz = seg[:, 1] - veh.z_cg
                X = g["X"][i] + scale * (-bx * ct - bz * st)
                Z = g["Z"][i] + scale * (-bx * st + bz * ct)
                ax.plot(X, Z, color=col, lw=0.8, alpha=0.8)
        ax.set(aspect="equal", ylabel="height [m]", title=f"{ep.vehicle}: {ep.kind}, {ep.t_tr:g} s, headwind {ep.headwind:g} m/s "
               f"(aircraft drawn {scale:g}× size, every {every_s:g} s)")
        if x_max:
            ax.set_xlim(-5, x_max)
    axs[-1, 0].set_xlabel("x [m]")
    fig.tight_layout()
    return fig


def polar_figure(pol, cfd=None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    a = np.radians(np.linspace(0, 180, 361))
    cl, cd, cm = pol.coefficients(a)
    ax[0].plot(np.degrees(a), cl, label="CL"); ax[0].plot(np.degrees(a), cd, label="CD"); ax[0].plot(np.degrees(a), cm, label="Cm (¼ chord)")
    ax[0].axvline(math.degrees(pol.alpha_stall), color="k", ls=":", lw=1)
    ax[0].set(xlabel="α [deg]", title="the model's full-range polar (per wing area)"); ax[0].legend(fontsize=8)
    aa = np.radians(np.linspace(-5, 25, 121))
    cl2, cd2, _ = pol.coefficients(aa)
    ax[1].plot(np.degrees(aa), cl2, label="CL"); ax[1].plot(np.degrees(aa), cl2 / np.maximum(cd2, 1e-6) / 10, label="L/D ÷ 10")
    ax[1].set(xlabel="α [deg]", title="through the stall"); ax[1].legend(fontsize=8)
    if cfd is not None and len(cfd):
        sv = cfd[(cfd["status"] == "solved") & (~cfd["rotor disks"].astype(bool))]
        if len(sv):
            ax[0].plot(sv["α [deg]"].astype(float), sv["CL"].astype(float), "o", label="CFD, bare airframe CL")
            ax[0].plot(sv["α [deg]"].astype(float), sv["CD"].astype(float), "s", label="CFD, bare airframe CD")
            ax[0].legend(fontsize=8)
    return ax


def _tri_plot(ax, g, a, b, tol=0.3, color=None):
    from matplotlib.collections import PolyCollection
    v, t = g.tessellate(tol)
    tris = v[t]
    ax.add_collection(PolyCollection(tris[:, :, [a, b]], facecolor=color or COL["wing"], edgecolor=COL["edge"], linewidths=0.05))


def fairing_figure(p=None, V: float = 16.0):
    """Kestrel-4Q-2's faired struts: the sections to scale (the round tube, its sleeve), the drag per metre of strut at
    ``V``, one bay seen from above and from the front, and the whole aircraft with the faired truss."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import kestrel_flight as kf
    pr = kestrel.resolve(p)
    q = 0.5 * kf.RHO * V ** 2
    fig = plt.figure(figsize=(16, 10))
    # 1. the sections
    ax = fig.add_subplot(2, 3, 1)
    for k, (d, dy) in enumerate(((pr["post_diameter"], 0.0), (pr["diagonal_diameter"], -22.0))):
        c, pts = kestrel.fairing_section(d, pr["fairing_tc"], pr["fairing_wall"])
        P = np.array(pts + pts[:1])
        ax.fill(P[:, 0], P[:, 1] + dy, color="#cfd8e0", ec=COL["edge"])
        ax.add_patch(plt.Circle((0, dy), d / 2, color="#202020"))
        ax.add_patch(plt.Circle((-30, dy), d / 2, color="#202020"))
        ax.text(-30, dy - d / 2 - 2, f"round Ø{d:g}", ha="center", va="top", fontsize=8)
        ax.text(c * 0.7 + 1, dy, f"faired: chord {c:.0f} mm,\nthickness {d + 2 * pr['fairing_wall']:.1f} mm", va="center", fontsize=8)
    ax.annotate("", xy=(-10, 14), xytext=(-42, 14), arrowprops=dict(arrowstyle="->", color="#2a6fb0", lw=2))
    ax.text(-26, 16, "air (cruise; the slipstream in the hover)", ha="center", fontsize=8, color="#2a6fb0")
    ax.set(aspect="equal", xlim=(-45, 45), ylim=(-36, 22), xlabel="x [mm] (along the pod's axis)", title="the strut sections, to scale")
    # 2. the drag per metre
    ax = fig.add_subplot(2, 3, 4)
    rows = []
    for d, name in ((pr["post_diameter"], "post Ø8"), (pr["diagonal_diameter"], "diagonal Ø6")):
        rows += [(f"{name}\nround", q * kf.CD_CYLINDER * d / 1000), (f"{name}\nfaired", q * kf.CD_FAIRED * d / 1000)]
    ax.bar([r[0] for r in rows], [r[1] for r in rows], color=["#c06000", "#2a6fb0"] * 2)
    for i, (_, v) in enumerate(rows):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    dr, dfa = kf.drag_buildup(p), kf.drag_buildup(p, faired=True)
    ax.set(ylabel="N per metre of strut", title=f"drag per metre at {V:g} m/s — the whole truss {q * dr['bracing_m2']:.2f} N → {q * dfa['bracing_m2']:.2f} N\n"
           f"(Cd0 {dr['Cd0']:.3f} → {dfa['Cd0']:.3f})")
    # 3–4. one bay, from above and from the front
    d = kestrel.Kestrel4Q()
    over = {k: v for k, v in pr.items() if k not in kestrel.PER_CALL}
    over["strut_fairing"] = True
    truss = d.generate(**over, part="truss")
    wing = d.generate(**over, part="wing")
    nac = d.generate(**over, part="nacelle")
    ax = fig.add_subplot(2, 3, 2)
    _tri_plot(ax, wing, 0, 1, color="#e6edf3"); _tri_plot(ax, nac, 0, 1, color="#bbbbbb"); _tri_plot(ax, truss, 0, 1, color="#2b2b2b")
    ax.set(aspect="equal", xlim=(100, 360), ylim=(30, 520), xlabel="x [mm]", ylabel="y [mm]",
           title="the right bay from above:\npost and diagonals' sleeves, chords along x")
    ax = fig.add_subplot(2, 3, 3)
    _tri_plot(ax, d.generate(**over, part="airframe"), 1, 2, color="#cfd8e0"); _tri_plot(ax, truss, 1, 2, color="#2b2b2b")
    ax.set(aspect="equal", xlim=(0, 520), ylim=(-170, 170), xlabel="y [mm]", ylabel="z [mm]", title="from the front: barely thicker than the tubes (9.6 / 7.6 mm)")
    # 5. the right bay in 3-D, close up (the propeller discs left out)
    import matplotlib.gridspec as gs
    ax = fig.add_subplot(gs.GridSpec(2, 3, figure=fig)[1, 1:], projection="3d")
    import cadquery as cq
    from vegeta.dedalus import Geometry
    box = cq.Workplane("XY").box(420, 520, 700).translate((210, 260, 0))
    cut = lambda g: Geometry(cq.Workplane("XY").add(g.shape).intersect(box).val())
    rest = cut(d.generate(**over, part="airframe"))
    v, t = rest.tessellate(0.4)
    ax.add_collection3d(Poly3DCollection(v[t], facecolor=COL["wing"], edgecolor=COL["edge"], linewidths=0.02, alpha=0.45))
    v, t = cut(truss).tessellate(0.2)
    ax.add_collection3d(Poly3DCollection(v[t], facecolor="#3a3a3a", edgecolor="#111", linewidths=0.05))
    ax.set(xlim=(0, 520), ylim=(0, 520), zlim=(-260, 260), title="Kestrel-4Q-2: the right bay with the faired post and diagonals")
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=-125); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.tight_layout()
    return fig


def front_view_4q2(p=None, *, cg=None, ax=None):
    """Looking at the nose with the Part-II options drawn: the elevons shaded on the outer panels, the bracing as
    chosen (``bracing``: x / posts / none), the faired struts as thicker blades."""
    import matplotlib.pyplot as plt
    pr = kestrel.resolve(p)
    L = kestrel.layout(p)
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5))
    b2, h, ym, yin, R = L["span"] / 2, L["h"], L["motor_y"], L["pod_width"] / 2, L["R_prop"]
    t = 0.12 * L["chord"]
    for z in (h, -h):
        ax.add_patch(plt.Rectangle((-b2, z - t / 2), 2 * b2, t, color=COL["wing"], ec=COL["edge"]))
        if L["elevons"]:
            for s in (1, -1):
                y0, y1 = s * L["elevon_y_in"], s * b2
                ax.add_patch(plt.Rectangle((min(y0, y1), z - t / 2 - 0.004), abs(y1 - y0), t + 0.008, color="#8a2be2", alpha=0.45, lw=0))
    ax.add_patch(plt.Rectangle((-yin, -h), 2 * yin, 2 * h, color="#e8e8e8", ec=COL["edge"]))
    lw_post = 2.5 if not pr["strut_fairing"] else 4.5
    lw_diag = 1.5 if not pr["strut_fairing"] else 3.0
    for s in (1, -1):
        if L["posts"]:
            ax.plot([s * ym, s * ym], [-h, h], color=COL["truss"], lw=lw_post)
        if L["diagonals"]:
            ax.plot([s * yin, s * ym], [h, -h], color=COL["truss"], lw=lw_diag)
            ax.plot([s * yin, s * ym], [-h, h], color=COL["truss"], lw=lw_diag)
    for name, m in L["motors"].items():
        ax.add_patch(plt.Circle((m["y"], m["z"]), R, color=COL["disc"], alpha=0.25))
        ax.add_patch(plt.Circle((m["y"], m["z"]), 0.017, color="#333"))
    if cg is not None:
        ax.plot(cg.get("y_cg", 0.0), cg["z_cg"], "x", color=COL["cg"], ms=12, mew=3)
    tags = [f"bracing: {L['bracing']}", "faired struts" if pr["strut_fairing"] else "round struts", "elevons" if L["elevons"] else "no elevons"]
    ax.set(aspect="equal", xlabel="y [m]", ylabel="z [m]", title="Kestrel-4Q-2 options, looking at the nose — " + ", ".join(tags))
    return ax


def xwing_front_view(L, *, cg=None, ax=None):
    """The X-wing looking at the nose: four panels at ±45°, the discs at their tips, the pod, the elevons shaded."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    t = 0.12 * L["chord"]
    phi = L["phi"]
    for sy, sz in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
        e = np.array([sy * math.cos(phi), sz * math.sin(phi)])
        n = np.array([-e[1], e[0]])
        for r0, r1, col, al in ((L["r0"], L["r1"], COL["wing"], 1.0), (L["elevon_y_in"], L["r1"], "#8a2be2", 0.45)):
            a, b = r0 * e, r1 * e
            quad = np.array([a + n * t / 2, b + n * t / 2, b - n * t / 2, a - n * t / 2])
            ax.fill(quad[:, 0], quad[:, 1], color=col, ec=COL["edge"] if al == 1.0 else None, alpha=al, lw=0.8 if al == 1.0 else 0)
    ax.add_patch(plt.Circle((0, 0), L["pod_width"] / 2, color="#e8e8e8", ec=COL["edge"]))
    for name, m in L["motors"].items():
        ax.add_patch(plt.Circle((m["y"], m["z"]), L["R_prop"], color=COL["disc"], alpha=0.25))
        ax.add_patch(plt.Circle((m["y"], m["z"]), 0.017, color="#333"))
        ax.text(m["y"] * 1.12, m["z"] * 1.12, name, ha="center", va="center", fontsize=9)
    if cg is not None:
        ax.plot(0.0, cg["z_cg"], "x", color=COL["cg"], ms=12, mew=3)
    ax.set(aspect="equal", xlabel="y [m]", ylabel="z [m]", title=f"{L['variant']}: looking at the nose")
    return ax
