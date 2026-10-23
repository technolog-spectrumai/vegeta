"""Plots and the summary of the sonic blowdown study (``sonic_flow``): CFD histories against the 0-D model, Mach fields
of the jet, the CAD views. ``report(cases_dir, results_dir)`` writes PNGs, ``summary.json`` and ``histories_<gas>.csv``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sonic_flow as sf

# the reference categorical palette (light): colour = gas, line style = model
COLORS = {"air": "#2a78d6", "co2": "#eb6834"}
LABEL = {"air": "Air", "co2": "CO$_2$"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style():
    import matplotlib.pyplot as plt                  # no backend forced: inline in a notebook, Agg headless
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2.0, "savefig.dpi": 150, "savefig.bbox": "tight"})
    return plt


def clean(h: dict) -> dict:
    """Mass-weighted averages are meaningless while hardly anything crosses the plane (the first microseconds, before
    the start-up shock reaches it): there they take the area averages."""
    h = dict(h)
    for plane in ("exit", "valve"):
        still = np.abs(h[f"{plane}_mdot"]) < 1e-3
        h[f"{plane}_U"] = np.where(still, h[f"{plane}_U_area"], h[f"{plane}_U"])
        h[f"{plane}_Ma"] = np.where(still, h[f"{plane}_Ma_area"], h[f"{plane}_Ma"])
        h[f"{plane}_T_mass"] = np.where(still, h[f"{plane}_T"], h[f"{plane}_T_mass"])
    return h


def _decimate(h: dict, n: int = 2000) -> dict:
    step = max(1, len(h["t"]) // n)
    return {k: v[::step] for k, v in h.items()}


PANELS = [  # key in CFD histories, key in 0-D, title, unit, scale
    ("exit_U", "U", "Exit gas velocity (mass-averaged)", "m/s", 1.0),
    ("exit_Ma", None, "Exit Mach number", "–", 1.0),
    ("exit_mdot", "mdot", "Mass flow at the pipe exit", "kg/s", 1.0),
    ("released_mass", "released_mass", "Gas released", "g", 1e3),
    ("exit_p", "p_star", "Exit static pressure (0-D: throat p*)", "bar", 1e-5),
    ("exit_T_mass", "T_star", "Exit static temperature (0-D: throat T*)", "K", 1.0),
    ("container_p", "p", "Container pressure", "bar", 1e-5),
    ("container_T", "T", "Container temperature", "K", 1.0),
]


def plot_velocity(cfd: dict, zero_d: dict, out: Path):
    plt = _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    for g, h in cfd.items():
        ax.plot(h["t"] * 1e3, h["exit_U"], color=COLORS[g], label=f"{LABEL[g]} — CFD (rhoCentralFoam)")
    for g, (ideal, real) in zero_d.items():
        ax.plot(ideal["t"] * 1e3, ideal["U"], color=COLORS[g], ls="--", lw=1.6, label=f"{LABEL[g]} — 0-D ideal gas, sonic throat")
        ax.plot(real["t"] * 1e3, real["U"], color=COLORS[g], ls=":", lw=2.0, label=f"{LABEL[g]} — 0-D real gas (CoolProp)")
    ax.set_xlabel("time after the valve opens [ms]")
    ax.set_ylabel("exit velocity [m/s]")
    ax.set_title("Exit gas velocity, 10 mm pipe, 50 bar container, valve open 0–10 ms", loc="left")
    ax.set_xlim(0, max(h["t"][-1] for h in cfd.values()) * 1e3)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.savefig(out / "exit_velocity.png")
    plt.close(fig)


def plot_dashboard(cfd: dict, zero_d: dict, out: Path):
    plt = _style()
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=True)
    for ax, (key, zkey, title, unit, k) in zip(axes.flat, PANELS):
        for g, h in cfd.items():
            ax.plot(h["t"] * 1e3, h[key] * k, color=COLORS[g], label=f"{LABEL[g]} CFD")
        if zkey:
            for g, (ideal, real) in zero_d.items():
                ax.plot(ideal["t"] * 1e3, ideal[zkey] * k, color=COLORS[g], ls="--", lw=1.4, label=f"{LABEL[g]} 0-D ideal")
                ax.plot(real["t"] * 1e3, real[zkey] * k, color=COLORS[g], ls=":", lw=1.8, label=f"{LABEL[g]} 0-D real")
        if key == "exit_Ma":
            ax.axhline(1.0, color=MUTED, lw=1, ls="-.")
            ax.text(0.2, 1.03, "Mach 1 (choked)", color=MUTED, fontsize=8, va="bottom")
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
    for ax in axes[-1]:
        ax.set_xlabel("time after the valve opens [ms]")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out / "blowdown_dashboard.png")
    plt.close(fig)


def plot_startup(cfd: dict, out: Path, until_ms: float = 1.0):
    """The first millisecond: the start-up shock reaching the exit, the supersonic overshoot, the settling to Mach 1."""
    plt = _style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for g, h in cfd.items():
        m = h["t"] * 1e3 <= until_ms
        t = h["t"][m] * 1e3
        axes[0].plot(t, h["exit_U"][m], color=COLORS[g], label=LABEL[g])
        axes[1].plot(t, h["exit_Ma"][m], color=COLORS[g], label=LABEL[g])
        axes[2].plot(t, h["exit_mdot"][m], color=COLORS[g], label=f"{LABEL[g]} exit")
        axes[2].plot(t, h["valve_mdot"][m], color=COLORS[g], ls="--", lw=1.4, label=f"{LABEL[g]} valve")
    axes[1].axhline(1.0, color=MUTED, lw=1, ls="-.")
    for ax, (title, unit) in zip(axes, (("Exit velocity", "m/s"), ("Exit Mach number", "–"), ("Mass flow", "kg/s"))):
        ax.set_title(f"{title} — first {until_ms:g} ms", loc="left")
        ax.set_xlabel("time [ms]")
        ax.set_ylabel(unit)
        ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "startup_transient.png")
    plt.close(fig)


def _times(case: Path) -> list[float]:
    ts = []
    for d in case.iterdir():
        try:
            t = float(d.name)
        except ValueError:
            continue
        if t > 0 and (d / "Ma").exists():
            ts.append(t)
    return sorted(ts)


def mach_slice(case: Path, time: float):
    """Cell centres (x, r) [m] and Mach number at one written time (pyvista's OpenFOAM reader)."""
    import pyvista as pv
    r = pv.OpenFOAMReader(str(case / "case.foam"))
    r.set_active_time_value(time)
    r.enable_all_cell_arrays()
    m = r.read()["internalMesh"]
    c = m.cell_centers().points
    return c[:, 0], np.hypot(c[:, 1], c[:, 2]), np.asarray(m.cell_data["Ma"])


def plot_mach_fields(cases: dict, out: Path, rig: sf.Rig = sf.Rig(), times_ms=(0.5, 1.0, 5.0, 10.0)):
    plt = _style()
    import matplotlib.tri as mtri
    gases = list(cases)
    fig, axes = plt.subplots(len(times_ms), len(gases), figsize=(7 * len(gases), 2.6 * len(times_ms)), squeeze=False)
    xe = rig.pipe_length * 1e-3
    levels = np.linspace(0, 3.6, 37)
    im = None
    for j, g in enumerate(gases):
        avail = _times(cases[g])
        for i, tm in enumerate(times_ms):
            ax = axes[i][j]
            t = min(avail, key=lambda v: abs(v - tm * 1e-3))
            x, rr, ma = mach_slice(cases[g], t)
            keep = (x > xe - 0.045) & (x < xe + 0.12) & (rr < 0.03)
            x, rr, ma = x[keep], rr[keep], ma[keep]
            X, Y, M = np.concatenate([x, x]) * 1e3, np.concatenate([rr, -rr]) * 1e3, np.concatenate([ma, ma])
            tri = mtri.Triangulation(X, Y)
            # drop triangles bridging the pipe wall (no gas there)
            xm, ym = X[tri.triangles].mean(1), np.abs(Y[tri.triangles]).mean(1)
            tri.set_mask((xm < rig.pipe_length) & (ym > rig.pipe_bore / 2))
            im = ax.tricontourf(tri, M, levels=levels, cmap="magma", extend="max")
            Rp = rig.pipe_bore / 2
            for s in (1, -1):
                ax.fill_between([rig.pipe_length - 45, rig.pipe_length], s * Rp, s * (Rp + rig.pipe_wall), color="#8a8984")
            ax.set_aspect("equal")
            ax.set_title(f"{LABEL[g]} — Mach number, t = {t * 1e3:.2f} ms", loc="left")
            ax.set_ylabel("r [mm]")
            ax.grid(False)
        axes[-1][j].set_xlabel("x [mm] (pipe exit at x = %g)" % rig.pipe_length)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("Mach number")
    fig.savefig(out / "mach_fields.png")
    plt.close(fig)


def plot_cad(cad_dir: Path, out: Path):
    from vegeta import dedalus
    plt = _style()
    for name in ("supply_rig", "fluid_volume"):
        step = cad_dir / f"{name}.step"
        if step.exists():
            fig = dedalus.plot_views(dedalus.load_step(step))
            fig.suptitle({"supply_rig": "Supply rig: 1 L container, 10 mm pipe, ball valve, exit flange",
                          "fluid_volume": "Fluid volume solved by the CFD (container + bore)"}[name])
            fig.savefig(out / f"cad_{name}.png")
            plt.close(fig)


def report(cases_dir: Path, out: Path, gases=("air", "co2"), cond: sf.Conditions = sf.Conditions()):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cases = {g: Path(cases_dir) / g for g in gases if (Path(cases_dir) / g / "postProcessing").exists()}
    cfd = {g: clean(sf.histories(c)) for g, c in cases.items()}
    zero_d = {g: (sf.blowdown(sf.GASES[g], cond=cond), sf.blowdown(sf.GASES[g], cond=cond, real=True, n=120)) for g in gases}
    plot_velocity(cfd, zero_d, out)
    plot_dashboard(cfd, zero_d, out)
    if cfd:
        plot_startup(cfd, out)
        plot_mach_fields(cases, out)
    plot_cad(out / "cad", out)
    summ = {"conditions": sf.asdict(cond), "rig": sf.asdict(sf.Rig()), "cfd": {}, "zero_d": {}}
    for g, h in cfd.items():
        summ["cfd"][g] = sf.summary(h, sf.GASES[g])
        d = _decimate(h)
        keys = ["t", "exit_U", "exit_Ma", "exit_mdot", "exit_p", "exit_T_mass", "valve_mdot", "container_p", "container_T",
                "container_mass", "released_mass"]
        np.savetxt(out / f"histories_{g}.csv", np.column_stack([d[k] for k in keys]), delimiter=",", fmt="%.6g",
                   header=",".join(keys), comments="")
    for g, (ideal, real) in zero_d.items():
        throat = sf.choked_real(sf.GASES[g].coolprop, cond.p0, cond.T0)
        summ["zero_d"][g] = {
            "ideal": {"gamma": sf.GASES[g].gamma, "exit_velocity_start_m_s": ideal["U"][0], "exit_velocity_end_m_s": ideal["U"][-1],
                      "mass_flow_start_kg_s": ideal["mdot"][0], "container_mass_g": ideal["mass"][0] * 1e3,
                      "released_mass_g": ideal["released_mass"][-1] * 1e3, "container_p_end_bar": ideal["p"][-1] / 1e5,
                      "container_T_end_K": ideal["T"][-1]},
            "real": {"exit_velocity_start_m_s": real["U"][0], "exit_velocity_end_m_s": real["U"][-1],
                     "mass_flow_start_kg_s": real["mdot"][0], "container_mass_g": real["mass"][0] * 1e3,
                     "released_mass_g": real["released_mass"][-1] * 1e3, "container_p_end_bar": real["p"][-1] / 1e5,
                     "container_T_end_K": real["T"][-1], "throat_T_K": throat[3], "throat_p_bar": throat[2] / 1e5,
                     "throat_vapour_quality": throat[4]},
        }
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    return summ
