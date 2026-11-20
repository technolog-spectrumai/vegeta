"""Figures for the ARIGO catalogue (``build.py``): CAD renders (whole aircraft, every part), MuJoCo screenshots,
and plots — from the stability run file (``arigo_charts``) and computed here (CFD polars, mass budgets, pack trades,
chase trajectories, launches). Everything is written to ``<out>/fig/``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AERO = HERE.parents[1]
sys.path.insert(0, str(AERO / "designs"))
sys.path.append(str(AERO / "sims"))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FAMILY = {"ARIGO-0": "#2a78d6", "ARIGO-1": "#eb6834", "ARIGO-2": "#1baf7a", "ARIGO-20": "#eda100", "ARIGO-0 strut": "#e87ba4"}
WING = {"goshawk": "#2a78d6", "zero": "#eb6834", "hybrid": "#1baf7a"}
PART_COL = {"fuselage": "#8d99a6", "wing": "#c9a66b", "tail": "#6b4f2a", "camera": "#111111", "motors": "#3b3f45",
            "props": "#7fb2e5", "hook": "#c0392b", "blades": "#1f2a33"}


# ------------------------------------------------------------------------------------------------ products
def products():
    """The catalogue's products: key, name, the CAD class and parameters, the sim factory, the cruise speed."""
    import arigo1 as a1
    import arigo1_mujoco as amj
    import arigo2 as a2
    import arigo2_mujoco as m2j
    import arigo20 as a20
    import arigo20_mujoco as m20j
    return {
        "arigo-1": dict(name="ARIGO-1", cls=a1.Arigo1, q=a1.variant("hybrid", "pusher"), card="arigo-1_hybrid",
                        sim=lambda bandit=False, mesh=None: amj.Sim("hybrid", "pusher", mesh=mesh, bandit=bandit), V=18.0),
        "arigo-0": dict(name="ARIGO-0", cls=a1.Arigo1, q=a1.variant("hybrid", "tractor_lowcam"), card="arigo-0_hybrid",
                        sim=lambda bandit=False, mesh=None: amj.Sim("hybrid", "tractor_lowcam", mesh=mesh, bandit=bandit), V=18.0),
        "arigo-1-zero": dict(name="ARIGO-1 zero", cls=a1.Arigo1, q=a1.variant("zero", "pusher"), card="arigo-1_zero",
                             sim=lambda bandit=False, mesh=None: amj.Sim("zero", "pusher", mesh=mesh, bandit=bandit), V=18.0),
        "arigo-20": dict(name="ARIGO-20", cls=a20.Arigo20, q=a20.variant(), card="arigo-20",
                         sim=lambda bandit=False, mesh=None: m20j.Sim20(mesh=mesh, bandit=bandit), V=30.0),
        "arigo-2": dict(name="ARIGO-2", cls=a2.Arigo2, q=a2.variant("hybrid"), card="arigo-2_hybrid",
                        sim=lambda bandit=False, mesh=None: m2j.Sim2("hybrid", mesh=mesh, bandit=bandit), V=18.0),
        "arigo-0-strut": dict(name="ARIGO-0 chin strut", cls=a1.Arigo1, q=a1.variant("hybrid", "tractor"), card="arigo-0_strut_hybrid",
                              sim=lambda bandit=False, mesh=None: amj.Sim("hybrid", "tractor", mesh=mesh, bandit=bandit), V=18.0),
    }


# ------------------------------------------------------------------------------------------------ CAD (VTK)
def _mesh(shape, tol=0.5):
    import pyvista as pv
    pts, faces, off = [], [], 0
    for so in shape.solids().vals():
        v, t = so.tessellate(tol, 0.2)
        pts += [(q.x, q.y, q.z) for q in v]
        faces += [x for tri in t for x in (3, tri[0] + off, tri[1] + off, tri[2] + off)]
        off += len(v)
    return pv.PolyData(np.array(pts), np.array(faces))


def render(out, items, elev=22.0, azim=-130.0, size=(1400, 1000), zoom=1.15):
    """``items``: [(cadquery shape, colour, opacity)] rendered together, parallel projection, white background."""
    import pyvista as pv
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("white")
    centre = None
    for shape, col, op in items:
        m = _mesh(shape)
        pl.add_mesh(m, color=col, opacity=op, smooth_shading=False, specular=0.25)
        centre = m.center if centre is None else centre
    pl.enable_parallel_projection()
    e, a = math.radians(elev), math.radians(azim)
    bounds = pl.bounds
    c = ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2)
    d = 3000.0
    pos = (c[0] + d * math.cos(e) * math.cos(a), c[1] + d * math.cos(e) * math.sin(a), c[2] + d * math.sin(e))
    pl.camera_position = [pos, c, (0, 0, 1) if abs(elev) < 89 else (-1, 0, 0)]
    pl.reset_camera()
    pl.camera.zoom(zoom)
    pl.screenshot(str(out))
    pl.close()
    autocrop(out)
    return out


def autocrop(path, pad=24):
    """Trim the white margin around a render (keeps ``pad`` pixels)."""
    img = plt.imread(path)
    ink = np.where((img[..., :3] < 0.985).any(axis=2))
    if len(ink[0]):
        y0, y1 = max(0, ink[0].min() - pad), min(img.shape[0], ink[0].max() + pad)
        x0, x1 = max(0, ink[1].min() - pad), min(img.shape[1], ink[1].max() + pad)
        plt.imsave(path, img[y0:y1, x0:x1])


def cad_figures(fig: Path, keys=("arigo-1", "arigo-0", "arigo-20", "arigo-2", "arigo-0-strut"), log=print):
    """Per product: the whole aircraft (hero view and a four-view sheet), and each part on its own."""
    P = products()
    out = {}
    for k in keys:
        pr = P[k]
        p = pr["cls"]().resolve(**{**pr["q"], "show_props": True})
        parts = pr["cls"]().parts(p)
        if "props" not in parts and "_props" in dir(pr["cls"]) and k != "arigo-20":
            parts["props"] = pr["cls"]()._props(p)
        allp = [(s, PART_COL.get(n, "#888888"), 0.35 if n == "props" else 1.0) for n, s in parts.items()]
        out[k] = {"hero": render(fig / f"cad_{k}_hero.png", allp, 24, -128, zoom=1.25)}
        views = []
        for name, (e, a) in (("front quarter", (22, -130)), ("from above", (90, -90)), ("from the side", (0, -90)), ("from the front", (0, 180))):
            views.append((render(fig / f"cad_{k}_{name.replace(' ', '_')}.png", allp, e, a, size=(900, 700)), name))
        sheet = fig / f"cad_{k}_views.png"
        f, axs = plt.subplots(2, 2, figsize=(11, 8.2), facecolor="white")
        for ax, (img, t) in zip(axs.flat, views):
            ax.imshow(plt.imread(img)); ax.set_axis_off(); ax.set_title(t, fontsize=11)
        f.tight_layout(); f.savefig(sheet, dpi=130); plt.close(f)
        out[k]["views"] = sheet
        out[k]["parts"] = {}
        for n, s in parts.items():
            if n == "fuselage":                           # shown in the whole-aircraft views only
                continue
            out[k]["parts"][n] = render(fig / f"cad_{k}_part_{n}.png", [(s, PART_COL.get(n, "#888888"), 1.0)], 24, -128, size=(900, 700), zoom=1.1)
        log(f"   CAD {k}: {len(parts)} parts")
    # the family line-up
    f, axs = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="white")
    for ax, k in zip(axs, ("arigo-0", "arigo-1", "arigo-20")):
        ax.imshow(plt.imread(out[k]["hero"])); ax.set_axis_off(); ax.set_title(P[k]["name"], fontsize=14, color=fam(P[k]["name"]))
    f.tight_layout(); f.savefig(fig / "family_lineup.png", dpi=130); plt.close(f)
    out["lineup"] = fig / "family_lineup.png"
    return out


# ------------------------------------------------------------------------------------------------ MuJoCo screenshots
def screenshots(fig: Path, keys=("arigo-1", "arigo-0", "arigo-20"), log=print):
    """Frames from the simulator: a hand launch, the urban chase (bandit in red) and a break turn, per product."""
    import mujoco
    import arigo1_mujoco as amj
    P = products()
    out = {}
    mdir = fig / "mesh"
    mdir.mkdir(exist_ok=True)
    for k in keys:
        pr = P[k]
        stl = mdir / f"{k}.stl"
        if not stl.exists():
            pr["cls"]().generate(**{**pr["q"], "show_props": True}).export_stl(str(stl), tolerance=0.5)
        shots = {}
        for scen, times in (("takeoff", (0.4, 1.5)), ("urban", (4.0, 9.0)), ("break_turn", (3.0,))):
            sim = pr["sim"](bandit=(scen == "urban"), mesh=str(stl))
            r = mujoco.Renderer(sim.m, 540, 960)
            cam = mujoco.MjvCamera()
            cam.distance, cam.elevation = (3.2 if scen != "urban" else 6.5), -14.0
            grabs, st = [], {"psi": None}

            def rec(s, times=times, grabs=grabs, cam=cam, r=r, st=st):
                t = s.d.time
                sd = s.state()
                vh = sd["v_world"][:2]
                psi = math.degrees(math.atan2(vh[1], vh[0])) if np.linalg.norm(vh) > 1 else 0.0
                st["psi"] = psi if st["psi"] is None else st["psi"] + 0.15 * ((psi - st["psi"] + 180) % 360 - 180)
                for tt in times:
                    if abs(t - tt) < 0.006 and len(grabs) < len(times):
                        cam.lookat[:] = sd["pos"]
                        cam.azimuth = st["psi"] + 180.0 - 25.0
                        r.update_scene(s.d, camera=cam)
                        grabs.append(r.render().copy())
            if scen == "takeoff":
                # takeoff() has no record hook: fly it by hand with the same rules
                Vs = math.sqrt(sim.C["mass"] * amj.am.G / (0.5 * amj.am.RHO * sim.C["S"] * sim.C["CL_MAX"]))
                V0 = amj.HAND_RELEASE_M_S if k != "arigo-20" else 1.3 * Vs
                sim.start(max(V0, 1.15 * Vs), amj.RELEASE_HEIGHT_M)
                sim.d.qvel[:3] = (V0, 0.0, 0.0)
                mujoco.mj_forward(sim.m, sim.d)
                ap = amj.Autopilot(sim)
                amj.run(sim, max(times) + 0.1, lambda t, s: ap(phi_c=0.0, theta_c=math.radians(12.0) if s.state()["V"] > 1.25 * Vs else 0.0,
                                                              throttle=1.0), rec)
            elif scen == "urban":
                amj.chase(sim, "urban", record=rec, V=max(16.0, 1.25 * Vs_of(sim)))
            else:
                amj.manoeuvre("break_turn", sim, rec, V=amj.ENTRY["break_turn"][0] + (8.0 if k == "arigo-20" else 0.0))
            for i, img in enumerate(grabs):
                pth = fig / f"shot_{k}_{scen}_{i}.png"
                plt.imsave(pth, img)
                shots.setdefault(scen, []).append(pth)
            r.close()
        out[k] = shots
        log(f"   screenshots {k}: {sum(len(v) for v in shots.values())}")
    return out


def Vs_of(sim):
    import arigo1_model as am
    C = sim.C
    return math.sqrt(C["mass"] * am.G / (0.5 * am.RHO * C["S"] * C["CL_MAX"]))


# ------------------------------------------------------------------------------------------------ plots
def _style(ax, title, xl=None, yl=None):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if xl:
        ax.set_xlabel(xl, fontsize=8, color=INK2)
    if yl:
        ax.set_ylabel(yl, fontsize=8, color=INK2)


def polars(fig: Path):
    """CL, CD and Cm (about the CG) over alpha: the CFD tables the models use, per product, its three wings."""
    import arigo1_model as am
    import arigo20_model as m20
    import arigo2_model as m2
    rows = [("ARIGO-1", "pusher"), ("ARIGO-0", "tractor_lowcam"), ("ARIGO-2", "twin")]
    f, axs = plt.subplots(4, 3, figsize=(13, 13), facecolor=SURFACE)
    for i, (nm, prop) in enumerate(rows + [("ARIGO-20", "edf")]):
        for w in (("goshawk", "zero", "hybrid") if prop != "edf" else ("",)):
            C = m20.get() if prop == "edf" else (m2.coefficients(w) if prop == "twin" else am.coefficients(w, prop))
            t = C.get("cfd")
            if t is None:
                continue
            col = WING.get(w, FAMILY["ARIGO-20"])
            lab = w or "ARIGO-20"
            axs[i, 0].plot(t["alpha_deg"], t["CL"], color=col, lw=2, marker="o", ms=4, label=lab)
            axs[i, 1].plot(t["alpha_deg"], t["CD"], color=col, lw=2, marker="o", ms=4, label=lab)
            axs[i, 2].plot(t["alpha_deg"], t["Cm"] + C.get("Cm_rig", 0.0), color=col, lw=2, marker="o", ms=4, label=lab)
        for j, (q, u) in enumerate((("Lift CL", "CL"), ("Drag CD", "CD"), ("Pitching moment Cm (CG, rigged)", "Cm"))):
            _style(axs[i, j], f"{nm}: {q}", "angle of attack (deg)", u)
            if axs[i, j].lines:
                axs[i, j].legend(fontsize=7, frameon=False)
    f.suptitle("CFD polars (screening RANS, propellers off) as used by the flight models", x=0.01, ha="left", fontsize=12)
    f.tight_layout(rect=(0, 0, 1, 0.97)); f.savefig(fig / "cfd_polars.png", dpi=120); plt.close(f)
    return fig / "cfd_polars.png"


def mass_budgets(fig: Path, cards: dict):
    """Stacked mass bars per product (lead configuration): structure, propulsion, avionics, margin, pack."""
    groups = {"structure": ("wing", "tail", "fuselage", "skid", "blister", "strut", "nacelle", "pylon"),
              "propulsion": ("motor", "ESC", "propeller", "fan"), "servos": ("servo",),
              "avionics": ("controller", "computer", "camera", "radio", "wiring"), "margin": ("margin",)}
    cols = {"structure": "#8d99a6", "propulsion": "#3b3f45", "servos": "#6250d6", "avionics": "#1baf7a", "margin": "#e4b363", "pack": "#e34948"}
    keys = [k for k in ("arigo-0_hybrid", "arigo-1_hybrid", "arigo-1_zero", "arigo-2_hybrid", "arigo-20") if k in cards and "point_mass" in cards[k]]
    f, ax = plt.subplots(figsize=(11, 0.6 * len(keys) + 1.6), facecolor=SURFACE)
    for yi, k in enumerate(keys):
        mb = cards[k]["point_mass"].get("mass_budget") or {}
        sums = dict.fromkeys(cols, 0.0)
        for it in mb.get("items", []):
            g = next((g for g, words in groups.items() if any(w.lower() in it["item"].lower() for w in words)), "structure")
            sums[g] += it["g"]
        sums["pack"] = (mb.get("pack") or {}).get("g", 0.0)
        left = 0.0
        for g, v in sums.items():
            ax.barh(yi, v, left=left, color=cols[g], edgecolor=SURFACE, linewidth=2, label=g if yi == 0 else None)
            if v > 90:
                ax.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white")
            left += v
        ax.text(left, yi, f"  {left:.0f} g", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([cards[k]["aircraft"] for k in keys], fontsize=8); ax.invert_yaxis()
    _style(ax, "Mass budgets (default pack), grams", "g")
    ax.legend(ncol=6, fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.02))
    f.tight_layout(); f.savefig(fig / "mass_budgets.png", dpi=130); plt.close(f)
    return fig / "mass_budgets.png"


def pack_trades(fig: Path, cards: dict):
    keys = [k for k in ("arigo-0_hybrid", "arigo-1_hybrid", "arigo-1_zero", "arigo-20") if k in cards and "point_mass" in cards[k]]
    f, axs = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    for k in keys:
        tr = cards[k]["point_mass"].get("pack_trade") or []
        nm = cards[k]["aircraft"]
        col = FAMILY.get(nm.split()[0], "#888")
        ls = "--" if "zero" in nm else "-"
        axs[0].plot([r["total_kg"] for r in tr], [r["endurance_min"] for r in tr], color=col, ls=ls, marker="o", lw=2, label=nm)
        axs[1].plot([r["total_kg"] for r in tr], [r["stall_m_s"] for r in tr], color=col, ls=ls, marker="o", lw=2, label=nm)
    _style(axs[0], "Endurance against take-off mass, over the pack options", "take-off mass (kg)", "endurance (min)")
    _style(axs[1], "Stall speed against take-off mass", "take-off mass (kg)", "stall (m/s)")
    for ax in axs:
        ax.legend(fontsize=8, frameon=False)
    f.tight_layout(); f.savefig(fig / "pack_trades.png", dpi=130); plt.close(f)
    return fig / "pack_trades.png"


def chase_tracks(fig: Path, keys=("arigo-1", "arigo-0", "arigo-20"), log=print):
    """Each chase scenario flown by each product (proportional navigation): the tracks seen from above, and the
    distance to the bandit over time. Computed here with the current models."""
    import arigo1_mujoco as amj
    P = products()
    out, summary = {}, {}
    for scen in amj.CHASES:
        t_b, bx, by, *_ = amj.bandit_path(scen)
        f, axs = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
        axs[0].plot(bx, by, color="#e34948", lw=2.5, label="bandit")
        for k in keys:
            sim = P[k]["sim"](bandit=True)
            h, s = amj.chase(sim, scen, V=max(amj.CHASES[scen]["V"], 1.25 * Vs_of(sim)))
            col = fam(P[k]["name"])
            axs[0].plot(h["x"], h["y"], color=col, lw=1.8, label=P[k]["name"])
            bxi = np.interp(h["t"], t_b, bx)
            byi = np.interp(h["t"], t_b, by)
            d = np.sqrt((h["x"] - bxi) ** 2 + (h["y"] - byi) ** 2 + (h["z"] - amj.CHASES[scen]["h"]) ** 2)
            axs[1].plot(h["t"], d, color=col, lw=1.8, label=P[k]["name"])
            summary.setdefault(scen, {})[k] = s
        axs[1].axhline(5.0, color=INK2, lw=1, ls=":")
        axs[1].set_yscale("log")
        _style(axs[0], f"{scen.replace('_', ' ')} chase: tracks from above (m)", "x (m)", "y (m)")
        axs[0].set_aspect("equal", adjustable="datalim")
        _style(axs[1], "distance to the bandit (log scale; dotted: 5 m)", "time (s)", "m")
        for ax in axs:
            ax.legend(fontsize=8, frameon=False)
        f.tight_layout(); f.savefig(fig / f"chase_{scen}.png", dpi=130); plt.close(f)
        out[scen] = fig / f"chase_{scen}.png"
        log(f"   chase {scen}: done")
    return out, summary


def launches(fig: Path, keys=("arigo-1", "arigo-0", "arigo-20", "arigo-2")):
    """Hand launch: height and speed over the first seconds, per product."""
    import mujoco
    import arigo1_mujoco as amj
    P = products()
    f, axs = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    res = {}
    for k in keys:
        sim = P[k]["sim"]()
        Vs = Vs_of(sim)
        sim.start(max(amj.HAND_RELEASE_M_S, 1.15 * Vs), amj.RELEASE_HEIGHT_M)
        sim.d.qvel[:3] = (amj.HAND_RELEASE_M_S, 0.0, 0.0)
        mujoco.mj_forward(sim.m, sim.d)
        ap = amj.Autopilot(sim)
        h = amj.run(sim, 6.0, lambda t, s: ap(phi_c=0.0, theta_c=math.radians(12.0) if s.state()["V"] > 1.25 * Vs else 0.0, throttle=1.0))
        col = fam(P[k]["name"])
        axs[0].plot(h["t"], h["z"], color=col, lw=2, label=P[k]["name"])
        axs[1].plot(h["t"], h["V"], color=col, lw=2, label=P[k]["name"])
        res[k] = {"min_height_m": float(h["z"].min())}
    axs[0].axhline(10.0, color=INK2, ls=":", lw=1)
    _style(axs[0], f"Hand launch at {amj.HAND_RELEASE_M_S:.0f} m/s from {amj.RELEASE_HEIGHT_M} m: height (dotted: safe 10 m)", "time (s)", "m")
    _style(axs[1], "airspeed", "time (s)", "m/s")
    for ax in axs:
        ax.legend(fontsize=8, frameon=False)
    f.tight_layout(); f.savefig(fig / "launch.png", dpi=130); plt.close(f)
    return fig / "launch.png", res


def stability_bars(fig: Path, run: dict):
    """Dutch-roll damping, short-period damping, spiral doubling time and stall height loss per configuration."""
    rows = []
    for n in ("arigo0_mujoco", "arigo1_mujoco", "arigo2_mujoco", "arigo20_mujoco"):
        for k, x in ((run.get("results", {}).get(n) or {}).get("runs") or {}).items():
            if not k.endswith("cm1") or "error" in x:
                continue
            fam = {"arigo0": "ARIGO-0", "arigo1": "ARIGO-1", "arigo2": "ARIGO-2", "arigo20": "ARIGO-20"}[n[:-7]]
            nm = fam + " " + k.replace("_cm1", "").replace("_tractor_lowcam", "").replace("_pusher", "").replace("_tractor", " strut")
            if fam == "ARIGO-20":
                nm = "ARIGO-20"
            lm = x["linear_modes"]
            rows.append((nm, lm["dutch_roll"]["zeta"], lm["short_period"]["zeta"], lm["spiral"]["t_half_or_double_s"],
                         x["stall_and_release"].get("altitude_lost_m") or 0))
    if not rows:
        return None
    f, axs = plt.subplots(1, 4, figsize=(16, 0.32 * len(rows) + 1.8), facecolor=SURFACE, sharey=True)
    y = np.arange(len(rows))
    for ax, j, title in zip(axs, (1, 2, 3, 4), ("Dutch roll damping ratio", "Short-period damping ratio", "Spiral: time to double (s)",
                                               "Stall: height lost (m)")):
        cols = [FAMILY.get("ARIGO-0 strut" if "strut" in r[0] else r[0].split()[0], "#888") for r in rows]
        ax.barh(y, [r[j] for r in rows], color=cols, height=0.7, edgecolor=SURFACE, linewidth=2)
        _style(ax, title)
    axs[0].set_yticks(y); axs[0].set_yticklabels([r[0] for r in rows], fontsize=7); axs[0].invert_yaxis()
    f.tight_layout(); f.savefig(fig / "stability_bars.png", dpi=130); plt.close(f)
    return fig / "stability_bars.png"


def fam(name):
    """The family colour of a product or configuration name."""
    return FAMILY["ARIGO-0 strut"] if "strut" in name else FAMILY.get(name.split()[0], "#888888")
