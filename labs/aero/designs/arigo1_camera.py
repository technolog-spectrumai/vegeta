"""What a nose tractor's propeller does to the observation camera, and how much of it software can take back
(notebook 39). ``sims/arigo1_camera.sh`` runs it for the layouts and writes a YAML and pictures.

Geometry only (no optics, no image processing): rays over the camera's field of view (pinhole, level, looking
forward) meet the propeller's plane; a ray is blocked where it meets the spinner or a blade. The blades are the
generic planform of a 10x6 two-blade propeller (Boreas' ``Propeller.from_pitch``; chord 16 -> 24 -> 7 mm), seen along
the axis (each element's width chord x cos(pitch angle)).

Capture modes compared:

- **global shutter, synchronised**: the exposure triggered at a fixed blade phase (from the ESC's rotor position or
  eRPM telemetry): the blades are a fixed mask; the best phase is the one that leaves the image centre clearest;
- **two-frame fusion**: alternate frames at phases a quarter blade-pitch apart; a pixel is lost only if blocked in
  both;
- **global shutter, free-running**: the mask at a random phase each frame (its mean share; it moves);
- **motion blur**: a finite exposure smears each blade over omega x t_exp: the pixels touched at all;
- **rolling shutter**: rows read over ``readout`` seconds while the propeller turns: the blades become curved bands;
  triggered at a fixed phase the band pattern is static too, but there are blades x rev/s x readout of them.

Hand inputs, marked: the rpm range (cruise ~6-9k, full ~12k rpm for a 10x6 on 4S: +-20 %), the readout times
(typical CMOS 8-33 ms), the camera at the layout's position with a 110 x 80 deg view.
"""
from __future__ import annotations

import math

import numpy as np

import arigo1 as a1

RPMS = (6000.0, 9000.0, 12000.0)            # HAND
EXPOSURES_MS = (0.1, 0.2, 0.5, 1.0, 2.0)
READOUTS_MS = (8.0, 16.0, 33.0)            # HAND: fast / typical / slow CMOS rolling readout
CENTRE = (40.0, 30.0)                      # deg: the tracking window


def blade(prop_d_mm: float):
    """r [mm], projected half-width [rad] along the blade (viewed along the axis)."""
    from vegeta import boreas
    d_in = prop_d_mm / 25.4
    d, pitch = boreas.inches(d_in, 0.6 * d_in)
    pr = boreas.Propeller.from_pitch("generic", d, pitch, blades=2, chord_root_m=0.016, chord_max_m=0.024, chord_tip_m=0.007,
                                     mass_kg=0.015, rotor_mass_kg=0.05)
    r = np.asarray(pr.r) * 1000.0
    w = np.asarray(pr.chord) * 1000.0 * np.cos(np.radians(np.asarray(pr.beta_deg)))
    return r, np.arctan2(w / 2, r)


class View:
    """The camera's rays on the propeller plane for one layout."""

    def __init__(self, q, fov=(110.0, 80.0), shape=(240, 320), hub_r_mm=14.0, blades=2):
        p = a1.resolve(q)
        L = a1.Arigo1.layout(p)
        self.L, self.p, self.B, self.hub = L, p, blades, hub_r_mm
        H, W = shape
        hs = np.radians(np.linspace(-fov[0] / 2, fov[0] / 2, W))
        vs = np.radians(np.linspace(fov[1] / 2, -fov[1] / 2, H))       # row 0 at the top
        self.Hd, self.Vd = np.meshgrid(np.degrees(hs), np.degrees(vs))
        cx, cy, cz = L["camera"]
        dx = L["x_prop"] - cx                                          # < 0: the disc ahead (forward is -x)
        self.ahead = dx < 0
        t = dx / -1.0
        Hh, Vv = np.radians(self.Hd), np.radians(self.Vd)
        y = cy + t * np.tan(Hh)
        z = cz + t * np.tan(Vv)
        self.r = np.hypot(y, z)
        self.th = np.arctan2(z, y)
        R = L["prop_d"] / 2 if L["props"] else 0.0
        self.R = R
        self.in_disc = self.ahead & (self.r <= R) if R else np.zeros_like(self.r, bool)
        self.spinner = self.in_disc & (self.r <= hub_r_mm)
        rb, hw = blade(L["prop_d"] if R else 254.0)
        self.halfw = np.interp(self.r, rb, hw, left=hw[0], right=0.0)
        self.centre = (np.abs(self.Hd) <= CENTRE[0] / 2) & (np.abs(self.Vd) <= CENTRE[1] / 2)

    def mask(self, phase, extra=0.0):
        """Blocked pixels at blade phase [rad]; ``extra``: half the smear angle added to the blade's half-width."""
        m = self.spinner.copy()
        for k in range(self.B):
            d = (self.th - phase - 2 * math.pi * k / self.B + math.pi) % (2 * math.pi) - math.pi
            m |= self.in_disc & (np.abs(d) <= self.halfw + extra)
        return m

    def rolling(self, phase0, rpm, readout_s):
        """The rolling-shutter mask: row i read at i/H x readout."""
        H = self.r.shape[0]
        w = rpm / 60 * 2 * math.pi
        ph = phase0 + w * readout_s * (np.arange(H) / H)[:, None]
        m = self.spinner.copy()
        for k in range(self.B):
            d = (self.th - ph - 2 * math.pi * k / self.B + math.pi) % (2 * math.pi) - math.pi
            m |= self.in_disc & (np.abs(d) <= self.halfw)
        return m

    def share(self, m):
        return {"image": round(float(m.mean()), 4), "centre": round(float(m[self.centre].mean()), 4)}


def analyse(q, name="") -> dict:
    v = View(q)
    out = {"layout": name, "camera_mm": [float(x) for x in v.L["camera"]], "prop_diameter_mm": v.L["prop_d"],
           "behind_disc": v.share(v.in_disc), "spinner": v.share(v.spinner)}
    if not v.in_disc.any():
        out["note"] = "no propeller disc in the camera's view: nothing to remove"
        return out
    phases = np.radians(np.arange(0.0, 180.0, 2.0))
    shares = [v.share(v.mask(ph)) for ph in phases]
    free = {"image": round(float(np.mean([s["image"] for s in shares])), 4), "centre": round(float(np.mean([s["centre"] for s in shares])), 4)}
    k = int(np.argmin([s["centre"] for s in shares]))
    best = float(phases[k])
    pair = v.mask(best) & v.mask(best + math.pi / (2 * v.B))
    out.update({
        "free_running_global_mean": free,
        "synced_global_best_phase": {"phase_deg": round(math.degrees(best), 1), **v.share(v.mask(best))},
        "synced_global_worst_phase": max(shares, key=lambda s: s["centre"]),
        "two_frame_fusion_residual": v.share(pair),
        "motion_blur": [], "rolling_shutter": []})
    for rpm in RPMS:
        w = rpm / 60 * 2 * math.pi
        for te in EXPOSURES_MS:
            sweep = w * te / 1000
            out["motion_blur"].append({"rpm": rpm, "exposure_ms": te, "smear_deg": round(math.degrees(sweep), 1),
                                       "tip_smear_mm": round(v.R * sweep, 1), "touched": v.share(v.mask(best + sweep / 2, extra=sweep / 2))})
        for ro in READOUTS_MS:
            m = v.rolling(best, rpm, ro / 1000)
            out["rolling_shutter"].append({"rpm": rpm, "readout_ms": ro, "blade_passes_per_frame": round(v.B * rpm / 60 * ro / 1000, 2),
                                           "synced": v.share(m)})
    # a 1-degree target (a bandit at 25 m is ~0.6-1 deg) inside the tracking window: chance it is hidden in one frame,
    # and after two-frame fusion (pixel shares as probabilities, the target ~ a pixel cluster: an upper bound)
    out["target_hidden_probability"] = {"free_running_frame": free["centre"], "synced_best_phase": out["synced_global_best_phase"]["centre"],
                                        "two_frame_fusion": out["two_frame_fusion_residual"]["centre"]}
    out["_view"] = v
    out["_best"] = best
    return out


def picture(results: list, path):
    """One row per layout: the synced mask, the two-frame residual, a rolling-shutter mask (9000 rpm, 16 ms)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in results if "_view" in r]
    if not rows:
        return None
    fig, axs = plt.subplots(len(rows), 4, figsize=(16, 3.3 * len(rows)), squeeze=False)
    for ax_row, r in zip(axs, rows):
        v, b = r["_view"], r["_best"]
        ims = [(v.in_disc, "behind the disc"), (v.mask(b), f"synced global shutter ({v.share(v.mask(b))['image']:.1%})"),
               (v.mask(b) & v.mask(b + math.pi / (2 * v.B)), "two-frame fusion: lost"),
               (v.rolling(b, 9000.0, 0.016), "rolling shutter, 9000 rpm, 16 ms (synced)")]
        for ax, (m, t) in zip(ax_row, ims):
            ax.imshow(m, cmap="Greys", extent=(v.Hd.min(), v.Hd.max(), v.Vd.min(), v.Vd.max()))
            ax.add_patch(plt.Rectangle((-CENTRE[0] / 2, -CENTRE[1] / 2), CENTRE[0], CENTRE[1], fill=False, ec="tab:red", lw=1))
            ax.set_title(f"{r['layout']}: {t}", fontsize=9)
            ax.set_xlabel("deg")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
