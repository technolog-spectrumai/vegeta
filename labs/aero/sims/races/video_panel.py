"""The side panel of the pilot videos (``gen_pilot_videos.sh``): what the autopilot asks for next to what the quad
does, so a video shows why a lap goes the way it goes. matplotlib (Agg), no OpenGL.

* **Header:** the tuned settings flown (constant for the lap) as bars, each normalised to its search range
  (``races.converge.BOUNDS``: 0 = the bottom, 1 = the top) so all six share one axis, the value above its bar and the
  name under the axis; then the autopilot's mode now (racing, a hover at the start
  or after a reset, returning to a missed gate, the drop guard's approach / align / descend — ``races.sim.PILOT_MODES``),
  the collisions so far and the next zone.
* **Strip charts** over the last ``WINDOW`` seconds, the cursor at now:
  1. speed: measured against the reference the autopilot commands (``v_ref``: the schedule through the carrot);
  2. tilt: measured against commanded (the desired force's angle after the tilt limit), the limit dashed;
  3. thrust in multiples of the weight: the rotor commands' sum against the applied thrust, hover (1 ×) dotted,
     the commands the drive could not give shaded; the drive's limit at the inflow is printed;
  4. the tracking error (|p − p_ref|) and the altitude.

The figure is built once; a frame only moves the window and the cursor and rewrites the header's live line. A lap
recorded before the autopilot's signals were (no ``pilot``) gets the physics only."""
from __future__ import annotations

import math

import numpy as np

from . import refquad

WINDOW = 10.0
MODE_COLOURS = {"racing": "#2a9d8f", "hover": "#d62828", "missed gate: back": "#6c757d",
                "drop: approach": "#e9a23b", "drop: align": "#e9a23b", "drop: descend": "#e9a23b"}
LABELS = {"safety": ("safety", ""), "lookahead_s": ("look-ahead", "s"), "omega_pos": ("ω position", "rad/s"),
          "omega_att": ("ω attitude", "rad/s"), "tilt_max_deg": ("tilt limit", "°"), "v_cap": ("speed cap", "m/s")}


def tilt_from_quaternion(H) -> np.ndarray:
    """The tilt from vertical [°] of each history row's attitude (w, x, y, z)."""
    q = np.asarray(H)[:, 10:14]
    return np.degrees(np.arccos(np.clip(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1.0, 1.0)))


class SidePanel:
    def __init__(self, course, result: dict, settings: dict | None, size: tuple, label: str = ""):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from races.sim import PILOT_MODES
        self.course, self.size, self.modes = course, (int(size[0]), int(size[1])), PILOT_MODES
        H = np.asarray(result["history"], dtype=float)
        P = result.get("pilot")
        P = np.asarray(P, dtype=float) if P is not None and len(P) else None
        self.t, self.P = H[:, 0], P
        ev = [(float(e[0]), str(e[1])) for e in result.get("events") or []]
        self.zone_t = [te for te, s in ev if s.startswith("zone")]
        self.hit_t = [te for te, s in ev if s.startswith("collision")]
        W = refquad.MASS_KG * 9.81
        tilt = tilt_from_quaternion(H) if H.shape[1] >= 14 else np.full(len(H), np.nan)
        tilt_max = (settings or {}).get("tilt_max_deg")

        w, h = self.size
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor="#101418")
        gs = self.fig.add_gridspec(6, 1, height_ratios=[0.62, 1.15, 1, 1, 1, 1], hspace=0.55, left=0.15, right=0.86, top=0.985, bottom=0.06)
        hd = self.fig.add_subplot(gs[0]); hd.axis("off")
        fs = max(6.0, h / 75)
        hd.text(0.0, 1.0, "AUTOPILOT" + (f"  ·  {label}" if label else ""), color="#e9c46a", fontsize=fs * 1.1, weight="bold", va="top", family="monospace")
        self.live = hd.text(0.0, 0.0, "", color="#e6e6e6", fontsize=fs, va="bottom", family="monospace", weight="bold")
        self._bars(self.fig.add_subplot(gs[1]), settings or {}, fs)
        self.axes = [self.fig.add_subplot(gs[i]) for i in range(2, 6)]
        for ax in self.axes:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#b0b0b0", labelsize=fs * 0.8)
            for sp in ax.spines.values():
                sp.set_color("#444")
            ax.grid(True, color="#333", lw=0.5)
        a1, a2, a3, a4 = self.axes
        a1.plot(self.t, H[:, 9], color="#8ecae6", lw=1.2, label="speed")
        if P is not None:
            a1.plot(P[:, 0], P[:, 1], color="#e9c46a", lw=1.0, ls="--", label="commanded")
        a1.set_ylabel("m/s", color="#b0b0b0", fontsize=fs * 0.8)
        a1.set_ylim(0, max(1.0, 1.1 * np.nanmax(np.r_[H[:, 9], P[:, 1] if P is not None else []])))
        a2.plot(self.t, tilt, color="#8ecae6", lw=1.2, label="tilt")
        if P is not None:
            a2.plot(P[:, 0], P[:, 3], color="#e9c46a", lw=1.0, ls="--", label="commanded")
        if tilt_max:
            a2.axhline(tilt_max, color="#d62828", lw=0.9, ls=":", label="limit")
        a2.set_ylabel("tilt °", color="#b0b0b0", fontsize=fs * 0.8)
        a2.set_ylim(0, max(10.0, 1.1 * np.nanmax(np.r_[tilt, [tilt_max or 0]])))
        a3.plot(self.t, H[:, 7] / W, color="#8ecae6", lw=1.2, label="applied")
        if P is not None:
            a3.plot(P[:, 0], P[:, 5] / W, color="#e9c46a", lw=1.0, ls="--", label="commanded")
            sat = P[:, 7] > 0.5
            if sat.any():
                a3.fill_between(P[:, 0], 0, 1, where=sat, transform=a3.get_xaxis_transform(), color="#d62828", alpha=0.25, lw=0, label="limited")
            self.t_max = P[:, 6] / W
        a3.axhline(1.0, color="#b0b0b0", lw=0.8, ls=":")
        a3.set_ylabel("× weight", color="#b0b0b0", fontsize=fs * 0.8)
        top = np.nanmax(np.r_[H[:, 7] / W, P[:, 5] / W if P is not None else []])
        a3.set_ylim(0, max(2.0, 1.15 * top))
        self.t_max_txt = a3.text(0.99, 0.92, "", transform=a3.transAxes, ha="right", va="top", color="#b0b0b0", fontsize=fs * 0.75)
        if P is not None:
            a4.plot(P[:, 0], P[:, 2], color="#f4a261", lw=1.2, label="tracking error")
            a4.set_ylim(0, max(0.5, 1.1 * np.nanmax(P[:, 2])))
        a4.set_ylabel("error m", color="#f4a261", fontsize=fs * 0.8)
        b4 = a4.twinx()
        b4.plot(self.t, H[:, 3], color="#8ecae6", lw=1.0, label="altitude")
        b4.set_ylim(0, max(1.0, 1.1 * np.nanmax(H[:, 3])))
        b4.tick_params(colors="#8ecae6", labelsize=fs * 0.8)
        b4.set_ylabel("alt m", color="#8ecae6", fontsize=fs * 0.8)
        for sp in b4.spines.values():
            sp.set_color("#444")
        self.axes.append(b4)
        for ax in (a1, a2, a3, a4):
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="upper left", fontsize=fs * 0.65, ncol=3, frameon=False, labelcolor="#d0d0d0", handlelength=1.5)
        a4.set_xlabel("s", color="#b0b0b0", fontsize=fs * 0.8)
        self.cursors = [ax.axvline(0, color="#ffffff", lw=0.8, alpha=0.7) for ax in self.axes[:4]]

    def _bars(self, ax, settings: dict, fs: float):
        """The settings flown: bars normalised to their search ranges, the value above, the name under the axis."""
        from races import converge
        keys = [k for k in converge.KEYS if k in settings]
        x = converge.to_x({k: settings[k] for k in keys})
        hts = [x[k] for k in keys]
        ax.set_facecolor("#161b22")
        ax.bar(range(len(keys)), hts, width=0.62, color="#457b9d", edgecolor="#8ecae6", lw=0.6)
        for i, (k, hgt) in enumerate(zip(keys, hts)):
            ax.text(i, hgt + 0.04, _fmt(k, settings[k]), ha="center", va="bottom", color="#e6e6e6", fontsize=fs * 0.72)
        ax.set_xticks(range(len(keys)), [LABELS[k][0].replace(" ", "\n", 1) for k in keys], fontsize=fs * 0.7, color="#d0d0d0")
        ax.set_ylim(0, 1.32)
        ax.set_yticks([0, 1], ["min", "max"], fontsize=fs * 0.65, color="#b0b0b0")
        ax.axhline(1.0, color="#444", lw=0.6, ls=":")
        ax.tick_params(axis="x", length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_color("#444")
        ax.set_title("settings flown (share of the search range)", color="#d0d0d0", fontsize=fs * 0.8, loc="right", pad=2)

    def _row(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.P[:, 0], t), 0, len(self.P) - 1))

    def frame(self, t: float) -> np.ndarray:
        """The panel at time ``t``: an RGB array of ``size``."""
        for ax in self.axes:
            ax.set_xlim(t - WINDOW, t + 0.02 * WINDOW)
        for c in self.cursors:
            c.set_xdata([t, t])
        passed = sum(z <= t for z in self.zone_t)
        nxt = self.course.zones[passed + 1].name if passed + 1 < len(self.course.zones) else "—"
        hits = sum(c <= t for c in self.hit_t)
        mode = self.modes[int(self.P[self._row(t), 8])] if self.P is not None else "—"
        self.live.set_text(f"mode {mode}   hits {hits}\nnext {nxt}")
        self.live.set_color(MODE_COLOURS.get(mode, "#e6e6e6"))
        if self.P is not None:
            self.t_max_txt.set_text(f"drive limit {self.t_max[self._row(t)]:.1f} × weight")
        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        return img[: self.size[1], : self.size[0]].copy()

    def close(self):
        import matplotlib.pyplot as plt
        plt.close(self.fig)


def _fmt(k: str, v) -> str:
    if v is None:
        return "none" if k == "v_cap" else "—"
    unit = LABELS[k][1]
    return f"{v:.3g}{' ' + unit if unit else ''}" if not (isinstance(v, float) and math.isnan(v)) else "—"
