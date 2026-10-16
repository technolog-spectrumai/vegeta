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
