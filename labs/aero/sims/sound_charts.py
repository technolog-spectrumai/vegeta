"""Charts from a sound_eval result file (or a merged one), in the ARIGO charts' style (same palette, the quietest
three hatched per panel, densest = winner).

    python labs/aero/sims/sound_charts.py labs/aero/sims/output/runs/sound_eval_<time>_<host>.yaml

Writes next to it:
  <stem>_levels.png   bar panels: LAeq, LAFmax, LZpeak per configuration (rotor configs at their cruise speed,
                      the cannon per shot and burst)
  <stem>_speed.png    multirotors: LAeq (1 rotor / all rotors), rpm, blade-pass frequency vs speed
  <stem>_spectra.png  third-octave spectra: each machine at every speed (all rotors), the cannon at the listener
  <stem>_cannon.png   the cannon: pressure at 1 m (0/90/180 deg), at the listener, directivity, per-shot ball speed and
                      SEL over the burst
  <stem>_kpi.png      the KPI scores (0-100, quieter is better)
``sound_eval.sh`` calls it at the end of every run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a5cd1", "#4fb3c7"]
COLOUR = {"drongo_quad": "#2a78d6", "drako_hexa": "#eb6834", "pekari_1": "#1baf7a", "pekari_5": "#eda100"}
RANK_HATCH = ["xxxx", "///", "/"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title, unit=""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    if unit:
        ax.set_ylabel(unit, color=INK2, fontsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def _bars(ax, labels, values, colours, lower_better=True):
    v = np.array(values, float)
    order = np.argsort(v if lower_better else -v)
    hatch = {int(i): RANK_HATCH[r] for r, i in enumerate(order[:3])} if len(v) > 1 else {}
    for i, (l, x, c) in enumerate(zip(labels, v, colours)):
        ax.bar(i, x, color=c, edgecolor="white", hatch=hatch.get(i, ""), lw=0.8)
        ax.text(i, x, f"{x:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=7)
    lo = np.nanmin(v)
    ax.set_ylim(max(0, lo - 15), np.nanmax(v) + 6)


def _configs(res):
    """(label, colour, metrics dict) for the level bars."""
    rows = []
    for k, r in res.items():
        if r.get("kind") == "multirotor" and r.get("points"):
            c = COLOUR.get(r["machine"], PALETTE[len(rows) % len(PALETTE)])
            for p in r["points"]:
                v = int(round(p["speed_kmh"]))
                rows.append((f"{r['machine']} {v} km/h x{r['rotors']}", c, p["all_rotors"]))
        elif r.get("kind") == "cannon" and r.get("sound"):
            c = COLOUR.get(f"pekari_{r['shots']}", PALETTE[len(rows) % len(PALETTE)])
            rows.append((f"Pekari {r['shots']} shot{'s' if r['shots'] > 1 else ''}", c, r["sound"]))
    return rows


def charts(path) -> list[Path]:
    path = Path(path)
    doc = yaml.safe_load(path.read_text())
    res = {k: v for k, v in doc.get("results", {}).items() if isinstance(v, dict) and "error" not in v}
    stem = path.with_suffix("")
    out = []

    rows = _configs(res)
    if rows:
        fig, axs = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")
        for ax, (key, title) in zip(axs, (("LAeq_dB", "LAeq (energy average)"), ("LAFmax_dB", "LAFmax (125 ms fast)"),
                                          ("LZpeak_dB", "LZpeak (unweighted peak)"))):
            _style(ax, title, "dB")
            _bars(ax, [r[0] for r in rows], [r[2][key] for r in rows], [r[1] for r in rows])
        fig.suptitle(f"Sound levels at the listener - {path.name} (hatched: quietest three)", fontsize=11, color=INK)
        fig.tight_layout()
        out.append(_save(fig, stem, "levels"))

    rot = [r for r in res.values() if r.get("kind") == "multirotor" and r.get("points")]
    if rot:
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), facecolor="white")
        _style(axs[0], "LAeq vs speed", "dB(A)"); _style(axs[1], "rpm (trimmed)", "rpm"); _style(axs[2], "Blade-pass frequency", "Hz")
        for r in rot:
            c = COLOUR.get(r["machine"], PALETTE[0])
            v = [p["speed_kmh"] for p in r["points"]]
            axs[0].plot(v, [p["all_rotors"]["LAeq_dB"] for p in r["points"]], "-o", color=c, label=f"{r['machine']} all {r['rotors']}")
            axs[0].plot(v, [p["one_rotor"]["LAeq_dB"] for p in r["points"]], "--o", color=c, mfc="white", label=f"{r['machine']} 1 rotor")
            axs[1].plot(v, [p["rpm"] for p in r["points"]], "-o", color=c, label=r["machine"])
            axs[2].plot(v, [p["blade_pass_hz"] for p in r["points"]], "-o", color=c, label=r["machine"])
            for p in r["points"]:
                if not str(p["source"]).startswith("CFD"):
                    axs[1].plot(p["speed_kmh"], p["rpm"], "x", color=INK, ms=8)
        for ax in axs:
            ax.set_xlabel("speed [km/h]", color=INK2, fontsize=8)
            ax.legend(fontsize=7, frameon=False)
        fig.suptitle(f"Multirotors over speed ({rot[0]['listener']['distance_m']:g} m, "
                     f"{rot[0]['listener']['angle_from_axis_deg']:g} deg from the axis; x = not from CFD)", fontsize=11, color=INK)
        fig.tight_layout()
        out.append(_save(fig, stem, "speed"))

    specs = []
    for r in rot:
        for p in r["points"]:
            specs.append((r["machine"], f"{int(round(p['speed_kmh']))} km/h", p["spectrum_all"]))
    can = [r for r in res.values() if r.get("kind") == "cannon" and r.get("sound")]
    for r in can:
        specs.append((f"pekari_{r['shots']}", "", r["sound"]["spectrum"]))
    if specs:
        groups = list(dict.fromkeys(s[0] for s in specs))
        fig, axs = plt.subplots(1, len(groups), figsize=(5.4 * len(groups), 4.4), facecolor="white", squeeze=False)
        for ax, gname in zip(axs[0], groups):
            _style(ax, f"{gname}: third-octave spectrum", "dB (unweighted)")
            sel = [s for s in specs if s[0] == gname]
            for i, (_, lab, sp) in enumerate(sel):
                sp = np.array(sp)
                ax.semilogx(sp[:, 0], sp[:, 1], "-", color=PALETTE[i % len(PALETTE)], label=lab or gname, lw=1.4)
            ax.set_xlabel("frequency [Hz]", color=INK2, fontsize=8)
            ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        out.append(_save(fig, stem, "spectra"))

    for r in can:
        fig, axs = plt.subplots(2, 2, figsize=(13, 8), facecolor="white")
        tr = r.get("trace_1m")
        _style(axs[0, 0], "Shot 1 at 1 m (CFD microphones)", "p' [Pa]")
        if tr:
            for j, (deg, p) in enumerate(zip(tr["deg"], tr["p_Pa"])):
                axs[0, 0].plot(tr["t_ms"], p, color=PALETTE[j], lw=1, label=f"{deg} deg")
            axs[0, 0].legend(fontsize=7, frameon=False)
        axs[0, 0].set_xlabel("t [ms]", color=INK2, fontsize=8)
        tl = r["sound"]["trace_listener"]
        _style(axs[0, 1], f"At the listener ({r['listener']['distance_m']:g} m, {r['listener']['angle_from_muzzle_deg']:g} deg, ground)", "p' [Pa]")
        axs[0, 1].plot(tl["t_ms"], tl["p_Pa"], color=COLOUR.get(f"pekari_{r['shots']}", INK), lw=0.8)
        axs[0, 1].set_xlabel("t [ms]", color=INK2, fontsize=8)
        ax = fig.add_subplot(2, 2, 3, projection="polar")
        axs[1, 0].remove()
        for i, s in enumerate(r["per_shot"]):
            d = s.get("directivity_1m")
            if d:
                a = np.radians([x["deg"] for x in d])
                ax.plot(a, [x["peak_dB"] for x in d], color=PALETTE[i % len(PALETTE)], lw=1.2, label=f"shot {s['shot']}")
        ax.set_thetamin(0); ax.set_thetamax(180)
        top = max((x["peak_dB"] for s in r["per_shot"] for x in s.get("directivity_1m", [])), default=140)
        ax.set_rlim(top - 40, top + 3)
        ax.set_title("Peak dB at 1 m vs angle from the muzzle", fontsize=10, color=INK)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        _style(axs[1, 1], "Over the burst", "")
        n = [s["shot"] for s in r["per_shot"]]
        axs[1, 1].bar(np.array(n) - 0.2, [s["ball_m_s"] for s in r["per_shot"]], 0.4, color=PALETTE[0], label="ball [m/s]")
        a2 = axs[1, 1].twinx()
        a2.plot(n, [s.get("SEL_dBA", np.nan) for s in r["per_shot"]], "-o", color=PALETTE[1], label="SEL [dB(A)]")
        a2.plot(n, [s.get("LZpeak_dB", np.nan) for s in r["per_shot"]], "-s", color=PALETTE[2], label="LZpeak [dB]")
        a2.tick_params(colors=INK2, labelsize=8)
        axs[1, 1].set_xlabel("shot", color=INK2, fontsize=8)
        axs[1, 1].legend(fontsize=7, frameon=False, loc="lower left"); a2.legend(fontsize=7, frameon=False, loc="lower right")
        fig.suptitle(f"Pekari recoilless cannon, {r['shots']} shot(s), mesh {r['mesh']['level']} ({r['mesh']['cells']:,} cells)",
                     fontsize=11, color=INK)
        fig.tight_layout()
        out.append(_save(fig, stem, f"cannon{r['shots']}"))

    k = doc.get("kpi") or {}
    if k:
        fig, axs = plt.subplots(1, len(k), figsize=(6 * len(k), 4.2), facecolor="white", squeeze=False)
        for ax, (name, d) in zip(axs[0], k.items()):
            _style(ax, f"KPI: {name} (100 = quietest)", "score")
            labs = list(d)
            _bars(ax, labs, [d[l]["score"] for l in labs], [COLOUR.get(l, PALETTE[i]) for i, l in enumerate(labs)], lower_better=False)
            ax.set_ylim(0, 110)
        fig.tight_layout()
        out.append(_save(fig, stem, "kpi"))
    for p in out:
        print("  chart", p)
    return out


def _save(fig, stem, name):
    p = Path(f"{stem}_{name}.png")
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    return p


if __name__ == "__main__":
    for f in sys.argv[1:] or sys.exit(__doc__):
        charts(f)
