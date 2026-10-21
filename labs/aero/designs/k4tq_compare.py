"""K4TQ — the designs side by side: K4TQ-A1 "Cathy", K4TQ-A2 (Cathy autonomous), K4TQ-F1 "Freddy" (front NACA 4415, rear
NACA 2415: the NeuralFoil study's sections) and K4TQ-F2 (Freddy autonomous), or any ``k4tq`` variant (E, C) and parameter set.

Everything comes from the design modules (``k4tq``, ``k4tq_sizing``, ``k4tq_sections``, ``k4tq_cad``): this module
only evaluates and tabulates. Masses are the CAD's where ``data/k4tq/cad_masses.json`` holds the design's geometry
(``cad_mass_status``). Labels: ESTIMATE / CALC as in Stage 1; MOCK where a mocked tool ran (the BEMT drive maps are
coarsened under the mocks; the section polars are NeuralFoil's, cached).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import k4tq
import k4tq_sizing as sz

DESIGNS = ("A1", "A2", "F1", "F2")
DATA = Path(__file__).resolve().parent / "data" / "k4tq"


def name(variant: str) -> str:
    nick = k4tq.NICKNAME.get(variant, "")
    return f"K4TQ-{variant}" + (f" ({nick})" if nick else "")


def spar_depths(p: dict) -> dict:
    """The section's depth at the spar station (30 % chord) on each wing's tip chord [mm] against the 18 mm the Ø16 spar
    and its skins need."""
    import k4tq_sections as ks
    L = k4tq.layout(p)
    out = {}
    for w in ("front", "rear"):
        depth = ks.thickness_at(k4tq.section_of(p, w)) * L["wings"][w]["tip_chord"] * 1e3
        out[w] = depth
    out["fits"] = min(out["front"], out["rear"]) >= ks.SPAR_DEPTH_M * 1e3
    return out


def evaluate(variant: str, *, corridor: bool = True, extended: bool = True, **over) -> dict:
    """One design through the Stage 1 analyses on its (CAD) masses."""
    veh = sz.vehicle(variant, **over)
    p = veh["p"]
    h = sz.hover(veh)
    c2, c4 = sz.cruise(veh, n_running=2), sz.cruise(veh, n_running=4)
    b2, b4 = sz.best_speeds(c2), sz.best_speeds(c4)
    m = sz.mission(veh, cr=c2, P_transition=h["electrical [W]"])
    cor = sz.corridor(veh, speeds=np.arange(0.0, 25.0, 2.0)) if corridor else None
    out = {"veh": veh, "hover": h, "cruise2": c2, "cruise4": c4, "best2": b2, "best4": b4, "mission": m, "corridor": cor,
           "corridor_summary": sz.corridor_summary(cor, veh) if cor is not None else None, "spar": spar_depths(p)}
    if extended:                                                   # R2: the radius with 0.5 kg (the extended mission)
        v5 = sz.vehicle(variant, payload_kg=0.5, **over)
        cr5 = sz.cruise(v5, n_running=2)
        P5 = sz.hover(v5)["electrical [W]"]
        lo, hi = 0.0, 60.0
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if sz.mission(v5, radius_km=mid, cr=cr5, P_transition=P5)["closes"]:
                lo = mid
            else:
                hi = mid
        out["radius 0.5 kg [km]"] = lo
    return out


def table(evals: dict) -> pd.DataFrame:
    rows = {}
    for v, e in evals.items():
        veh, p, h, b2, b4, m = e["veh"], e["veh"]["p"], e["hover"], e["best2"], e["best4"], e["mission"]
        cs = e["corridor_summary"] or {}
        rows[name(v)] = {
            "sections (front / rear)": f"{k4tq.section_of(p, 'front')} / {k4tq.section_of(p, 'rear')}",
            "autonomy (Jetson)": bool(p.get("autonomy")), "masses": p.get("cad_mass_status", ""), "take-off mass [kg]": veh["mass"]["mass_kg"],
            "electronics [W]": k4tq.avionics_power(p), "spar depth front / rear tip [mm]": f"{e['spar']['front']:.1f} / {e['spar']['rear']:.1f}",
            "Ø16 spar fits": e["spar"]["fits"], "CL max (aircraft)": veh["stall"]["CL max (aircraft)"], "stall speed [m/s]": sz.stall_speed(veh),
            "stalls first": veh["stall"]["stalls first"], "best L/D (2 motors)": b2["L/D max"], "best-range V [m/s]": b2["best range V [m/s]"],
            "cruise power [W]": sz._at(e["cruise2"], b2["best range V [m/s]"], "electrical [W]"), "still-air range [km]": b2["range [km]"],
            "endurance [min]": b2["endurance [min]"], "max level V [m/s]": b4["max level V [m/s]"], "hover [W]": h["electrical [W]"],
            "T/W (sagged)": h["T/W (sagged pack, 3.3 V/cell)"], "front motor in the hover": h["most loaded motor / its maximum"],
            "R3 closes (1 kg)": m["closes"], "R3 margin [Wh]": m["margin [Wh]"], "radius with 0.5 kg [km]": e.get("radius 0.5 kg [km]", math.nan),
            "transition peak [W]": cs.get("peak power [W]", math.nan), "wing-borne from [m/s]": cs.get("wing-borne (tilt < 8°) from [m/s]", math.nan),
        }
    df = pd.DataFrame(rows).T
    df.attrs["label"] = sz.label("CALC")
    return df


def figure(evals: dict, path: Path) -> Path:
    """Power against speed (cruise, two motors), the transition corridor's power, the section polars of the two wings,
    and the headline bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import k4tq_sections as ks
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=110)
    for v, e in evals.items():
        c = e["cruise2"][e["cruise2"]["feasible"]]
        axs[0, 0].plot(c.index, c["electrical [W]"], "o-", ms=3, label=name(v))
        if e["corridor"] is not None:
            cor = e["corridor"][e["corridor"]["feasible"]]
            axs[0, 1].plot(cor.index, cor["electrical [W]"], "o-", ms=3, label=name(v))
    axs[0, 0].set(xlabel="airspeed [m/s]", ylabel="electrical power [W]", title="cruise, rear pair stopped (the aircraft's polar through the drive)")
    axs[0, 1].set(xlabel="airspeed [m/s]", ylabel="electrical power [W]", title="the transition corridor (level flight, fuselage level)")
    cache = ks.load_cache()
    secs = sorted({(w, k4tq.section_of(e["veh"]["p"], w)) for e in evals.values() for w in ("front", "rear")})
    e0 = next(iter(evals.values()))
    for w, n in secs:
        Re = e0["veh"]["L"]["wings"][w]["mac"] * e0["best2"]["best range V [m/s]"] / ks.NU
        pol = ks.polar(n, Re, cache=cache)
        s = pol.attached()
        axs[1, 0].plot(pol.cd[s], pol.cl[s], ls="-" if w == "front" else "--", label=f"{w}: {n} (Re {Re:.2g})")
    axs[1, 0].set(xlabel="cd", ylabel="cl", xlim=(0, 0.035), title="the wings' sections at the cruise: losses against lift (NeuralFoil)")
    df = table(evals)
    ks_ = ["still-air range [km]", "R3 margin [Wh]", "stall speed [m/s]", "cruise power [W]"]
    x = np.arange(len(ks_))
    for i, (k, r) in enumerate(df.iterrows()):
        vals = [float(r[c]) for c in ks_]
        base = [float(df.iloc[0][c]) for c in ks_]
        axs[1, 1].bar(x + 0.2 * i - 0.3, [v / b if b else 0 for v, b in zip(vals, base)], width=0.2, label=k)
    axs[1, 1].axhline(1.0, color="k", lw=0.8)
    axs[1, 1].set_xticks(x, ks_, rotation=15)
    axs[1, 1].set(title=f"relative to {df.index[0]}")
    for ax in axs.flat:
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def verdict(df: pd.DataFrame) -> str:
    """Plain-language reading of the table: who wins what, and the hard constraints."""
    lines = []
    for k, r in df.iterrows():
        if not bool(r["Ø16 spar fits"]):
            lines.append(f"- **{k}**: the Ø16 spar does not fit its front tip section ({r['spar depth front / rear tip [mm]']} mm against 18) — "
                         "buildable only with a spar tapered to Ø12 at the tip (a structural check in Stage 4).")
    best_r = df["still-air range [km]"].astype(float).idxmax()
    low_vs = df["stall speed [m/s]"].astype(float).idxmin()
    lines.append(f"- longest range: **{best_r}** ({float(df.loc[best_r, 'still-air range [km]']):.1f} km); lowest stall: **{low_vs}** "
                 f"({float(df.loc[low_vs, 'stall speed [m/s]']):.1f} m/s).")
    for k, r in df.iterrows():
        if bool(r["autonomy (Jetson)"]):
            lines.append(f"- **{k}**: the companion computer and perception cost {float(r['electronics [W]']) - 19.0:.0f} W and the R3 margin falls to "
                         f"{float(r['R3 margin [Wh]']):.0f} Wh (T/W {float(r['T/W (sagged)']):.2f} on a sagged pack).")
    return "\n".join(lines)
