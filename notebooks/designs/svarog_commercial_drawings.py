"""Svarog III's figures (notebook 47): the four lasers' beam quality against their limits, the collimated and the
diverging beam against distance, the energy budgets with their bands, the thin-plate thermal fields, the welding
comparison, the shielding irradiance against the MPE and the barriers' bounds against distance, and the verdict
matrices. Static Matplotlib; every function takes the module's tables and an optional ``ax``."""
from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import svarog_commercial as sc

COL = {"YLS-SM": "tab:blue", "FL2000C": "tab:orange", "FL2000CSM": "tab:green", "WFL-3000": "tab:red", "1kW": "grey", "2kW": "black"}
VERDICT_COL = {"PASS": "#4caf50", "INCOMPLETE": "#ffb300", "FAIL": "#e53935"}


def _ax(ax, w=8, h=4.5):
    return ax if ax is not None else plt.subplots(figsize=(w, h))[1]


def beam_quality_view(optic: dict, ax=None):
    """BPP against M² for each laser: the published limit as a bar (M² or BPP), the derived one as a point; the
    Gaussian/multimode boundary."""
    ax = _ax(ax)
    m2 = np.linspace(1.0, 10.0, 100)
    ax.plot(m2, [sc.bpp_from_m2(m) for m in m2], "k--", lw=1, label="BPP = M² λ/π at 1070 nm")
    for k, o in optic.items():
        ax.scatter([o["m2"]], [o["bpp_mm_mrad"]], s=70, color=COL.get(k, "k"), label=f"{k}: {o['basis']}", zorder=3)
        spec = sc.LASERS[k]
        if sc.known(spec.m2_max):
            ax.axvline(spec.m2_max, color=COL.get(k, "k"), lw=0.8, alpha=0.5)
        if sc.known(spec.bpp_max_mm_mrad):
            ax.axhline(spec.bpp_max_mm_mrad, color=COL.get(k, "k"), lw=0.8, alpha=0.5)
    ax.axvspan(1.0, sc.GAUSSIAN_M2_LIMIT, color="green", alpha=0.07, label="Gaussian description holds")
    ax.set_xlabel("M² (limit)"); ax.set_ylabel("BPP [mm·mrad] (limit)"); ax.set_title("beam quality: the published limits and what they imply")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    return ax


def propagation_view(optic: dict, ax=None, distances=None):
    """The raw collimated beam's radius against distance (geometric growth against the Gaussian one) and the beam past the focus."""
    ax = _ax(ax)
    d = np.linspace(0.5, 100.0, 300) if distances is None else np.asarray(distances)
    for k, o in optic.items():
        ax.plot(d, sc.beam_radius_at(o, d, "collimated") * 1e3, color=COL.get(k, "k"), lw=2, label=f"{k} collimated (θ {o['collimated_divergence'] * 1e6:.0f} µrad)")
        ax.plot(d, sc.beam_radius_at(o, d, "reflected") * 1e3, color=COL.get(k, "k"), lw=1, ls=":", label=f"{k} past the focus (θ {o['cone_half_angle'] * 1e3:.0f} mrad)")
    ax.set_yscale("log"); ax.set_xlabel("distance [m]"); ax.set_ylabel("beam radius [mm]"); ax.set_title("the stray beam against distance")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, ncol=2)
    return ax


def gaussian_vs_multimode_view(opt: dict, ax=None):
    """One laser's focus: the embedded Gaussian's w(z) against the geometric cone of the imaged core."""
    ax = _ax(ax, 7, 4)
    z = np.linspace(-4 * opt["rayleigh_length"], 4 * opt["rayleigh_length"], 300)
    g = opt["gaussian"]
    ax.plot(z * 1e3, g.w(z) * 1e6, lw=2, label=f"embedded Gaussian, M² {opt['m2']:.2f}: w₀ {opt['spot_gaussian'] * 1e6:.0f} µm")
    geo = np.maximum(opt["spot_geometric"], opt["cone_geometric"] * np.abs(z))
    ax.plot(z * 1e3, geo * 1e6, lw=2, ls="--", label=f"core imaged: {opt['spot_geometric'] * 1e6:.0f} µm, cone {opt['cone_geometric'] * 1e3:.0f} mrad")
    ax.set_xlabel("z from the focus [mm]"); ax.set_ylabel("radius [µm]"); ax.set_title(f"{opt['laser']}: {opt['regime']}", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    return ax


def energy_view(energy: dict, ax=None):
    """Stacked bars per laser: optical out, waste heat, chiller — with the low–high band on the total."""
    ax = _ax(ax, 9, 4.5)
    keys = list(energy)
    x = np.arange(len(keys))
    out = np.array([energy[k]["power_W"] for k in keys]) / 1e3
    heat = np.array([energy[k]["waste_heat_W"]["nominal"] for k in keys]) / 1e3
    chil = np.array([energy[k]["chiller_electrical_W"]["nominal"] for k in keys]) / 1e3
    tot_lo = np.array([energy[k]["total_electrical_W"]["low"] for k in keys]) / 1e3
    tot_hi = np.array([energy[k]["total_electrical_W"]["high"] for k in keys]) / 1e3
    ax.bar(x, out, color="gold", label="optical out")
    ax.bar(x, heat, bottom=out, color="tab:red", alpha=0.7, label="waste heat (electrical in − optical)")
    ax.bar(x, chil, bottom=out + heat, color="tab:blue", alpha=0.7, label="chiller electrical")
    ax.errorbar(x, out + heat + chil, yerr=[out + heat + chil - tot_lo, tot_hi - (out + heat + chil)], fmt="none", ecolor="k", capsize=4, label="assumptions' range")
    for i, k in enumerate(keys):
        c = sc.LASERS[k].cooling_kW
        if sc.known(c):
            ax.plot([i - 0.4, i + 0.4], [out[i] + c, out[i] + c], "k--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(keys); ax.set_ylabel("kW"); ax.set_title("energy balance (dashed: the published cooling capacity above the optical output)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    return ax


def thermal_field_view(case: dict, ax=None, extent_mm=(-12.0, 3.0, 0.0, 3.0)):
    """The thin-plate field of one weld case: isotherms around the moving source (the melt and the HAZ lines marked)."""
    ax = _ax(ax, 8, 3.2)
    pool = case["pool"]
    xi = np.linspace(extent_mm[0], extent_mm[1], 300) * 1e-3
    y = np.linspace(extent_mm[2], extent_mm[3], 120) * 1e-3
    XI, Y = np.meshgrid(xi, y)
    T = pool["field"](XI, Y)
    s = sc.SHEET_MATERIALS[case["family"]]["solid"]
    cf = ax.contourf(xi * 1e3, y * 1e3, T, levels=np.linspace(sc.th.T_ROOM, s.T_fail, 25), cmap="inferno")
    ax.contour(xi * 1e3, y * 1e3, T, levels=[sc.SHEET_MATERIALS[case["family"]]["T_haz"], s.T_melt], colors=["cyan", "white"], linewidths=1.2)
    plt.colorbar(cf, ax=ax, label="T [K]")
    ax.set_xlabel("ξ along the weld [mm] (source at 0, moving →)"); ax.set_ylabel("y [mm]")
    ax.set_title(f"{case['laser']}: {case['material']} {case['thickness'] * 1e3:g} mm at {case['speed'] * 60:.1f} m/min — white: melt, cyan: HAZ", fontsize=8)
    return ax


def welding_comparison_view(tables: dict, quantity: str = "v [m/min]", ax=None):
    """One welding quantity per laser against thickness, steel solid, aluminium dashed."""
    ax = _ax(ax)
    for k, df in tables.items():
        for fam, ls in (("steel", "-"), ("aluminium", "--")):
            sub = df[df["material"].str.contains("steel" if fam == "steel" else "aluminium")]
            ax.plot(sub["t [mm]"], sub[quantity], ls, marker="o", color=COL.get(k, "k"), label=f"{k} {fam}")
    ax.set_xlabel("sheet thickness [mm]"); ax.set_ylabel(quantity); ax.set_title(f"{quantity} per laser (model, unvalidated)"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    return ax


def exposure_view(optic: dict, ax=None):
    """The collimated beam's irradiance against distance per laser, the eye's and the skin's MPE at T1/T2/T3 as lines."""
    ax = _ax(ax)
    d = np.linspace(1.0, 200.0, 400)
    for k, o in optic.items():
        E = o["delivered_W"] / (math.pi * sc.beam_radius_at(o, d, "collimated") ** 2)
        ax.plot(d, E, color=COL.get(k, "k"), lw=2, label=f"{k} collimated")
        E2 = sc._a("workpiece_reflectivity").nominal * o["delivered_W"] / (math.pi * sc.beam_radius_at(o, d, "reflected") ** 2)
        ax.plot(d, E2, color=COL.get(k, "k"), lw=1, ls=":", label=f"{k} reflected, diverging")
    ax.axhline(float(sc.mpe_eye_1070(10.0)), color="k", lw=0.8, ls="--"); ax.text(150, float(sc.mpe_eye_1070(10.0)) * 1.3, "MPE eye ≥10 s (T1, T2, T3)", fontsize=7)
    ax.axhline(float(sc.mpe_eye_1070(1.0)), color="k", lw=0.5, ls=":"); ax.text(150, float(sc.mpe_eye_1070(1.0)) * 1.3, "MPE eye, 1 s", fontsize=7)
    ax.axhline(float(sc.mpe_skin_1070(100.0)), color="brown", lw=0.8, ls="--"); ax.text(150, float(sc.mpe_skin_1070(100.0)) * 1.3, "MPE skin ≥10 s", fontsize=7, color="brown")
    ax.set_yscale("log"); ax.set_xlabel("distance [m]"); ax.set_ylabel("mean irradiance [W/m²]"); ax.set_title("accidental exposure against distance (IEC 60825-1 MPE, 1070 nm)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6, ncol=2)
    return ax


def risk_vs_distance_view(df, ax=None, value="bound_ratio", kind="collimated"):
    """A barrier indicator against distance per laser and family (base variants): the thermal bound ratio (1 = the
    closed-form bound reaches the damage point) or the irradiance over the MPE."""
    ax = _ax(ax, 9, 4.5)
    sub = df[(df["beam"] == kind) & df["base"]]
    marks = {"polycarbonate": "s", "aluminium": "^", "wood": "o"}
    for (laser, fam), g in sub.groupby(["laser", "family"]):
        g = g.sort_values("distance_m")
        ax.plot(g["distance_m"], g[value], marker=marks.get(fam, "o"), color=COL.get(laser, "k"), lw=1, label=f"{laser} {fam}")
    if value == "bound_ratio":
        ax.axhline(1.0, color="k", ls="--", lw=1); ax.text(5, 1.1, "bound reaches the damage point", fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("distance [m]"); ax.set_ylabel(value); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6, ncol=3)
    ax.set_title("shielding indicator against distance — an indicator, not a verdict", fontsize=9)
    return ax


def verdict_matrix_view(vm, ax=None, title="verdicts"):
    """The verdict matrix as coloured cells (green PASS, amber INCOMPLETE, red FAIL)."""
    ax = _ax(ax, 8, 0.35 * len(vm) + 1.5)
    codes = {"PASS": 0, "INCOMPLETE": 1, "FAIL": 2}
    M = np.array([[codes.get(v, 1) for v in row] for row in vm.values])
    ax.imshow(M, cmap=ListedColormap([VERDICT_COL["PASS"], VERDICT_COL["INCOMPLETE"], VERDICT_COL["FAIL"]]), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(vm.columns))); ax.set_xticklabels([f"{c:g} m" for c in vm.columns])
    ax.set_yticks(range(len(vm))); ax.set_yticklabels([" / ".join(str(i) for i in idx) for idx in vm.index], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, vm.values[i, j], ha="center", va="center", fontsize=6)
    ax.set_title(title, fontsize=9)
    return ax


STYLE_HOLD = {1.0: ":", 3.0: "--", 5.0: "-", 10.0: "-."}


def required_thickness_view(df, axs=None, hold_times=(1.0, 3.0, 5.0, 10.0), endurance_rows=None):
    """Three panels (polycarbonate, aluminium, steel): the required sheet thickness against distance, one colour per laser,
    one line style per hold time (1 s dotted, 3 s dashed, 5 s solid, T1 dash-dot); the recorded endurance points (the
    1/2/3 mm sheets that held at least the hold time) as markers when given; guides at 1, 2, 3 mm."""
    fams = list(dict.fromkeys(df["family"]))
    if axs is None:
        _, axs = plt.subplots(1, len(fams), figsize=(6.5 * len(fams), 4.8), sharey=True)
    for ax, fam in zip(np.atleast_1d(axs), fams):
        sub = df[df["family"] == fam]
        for laser, g in sub.groupby("laser"):
            for t in hold_times:
                h = g[g["hold_s"] == t].sort_values("distance_m")
                ax.plot(h["distance_m"], h["required_mm"], STYLE_HOLD.get(t, "-"), color=COL.get(laser, "k"), lw=1.5, label=f"{laser}, {'T1 10 s' if t == 10.0 else f'{t:g} s'}")
        if endurance_rows is not None:
            e = sc.pd.DataFrame(endurance_rows)
            e = e[e["family"] == fam]
            for t in hold_times:
                ok = e[e["holds_s"] >= t]
                for laser, g in ok.groupby("laser"):
                    ax.scatter(g["distance_m"], g["thickness_mm"], marker="x", color=COL.get(laser, "k"), s=25, zorder=4)
        for mm in (1.0, 2.0, 3.0):
            ax.axhline(mm, color="grey", lw=0.6, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("distance [m]"); ax.grid(alpha=0.3, which="both")
        ax.set_title(f"{fam}: sheet required (closed-form bounds; × recorded sheets that held)", fontsize=8)
    np.atleast_1d(axs)[0].set_ylabel("required thickness [mm]")
    np.atleast_1d(axs)[-1].legend(fontsize=5, ncol=2)
    return axs
