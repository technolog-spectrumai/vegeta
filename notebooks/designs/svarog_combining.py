"""Svarog-combining (notebook 47): the combining of complete fibre-laser outputs on Hyperion's ``combining`` module —
the 1 kW and the 2 kW devices each built from 1, 2, 4 or 8 channels, incoherently, coherently (filled and tiled
apertures) and spectrally; the near and far fields, the power accounting, the beam metrics against one Gaussian of the
same power, the phase sensitivity and the losses. The channels are *laser outputs* (the 30/250 power stage's LP₀₁);
the pump-diode combining of the modules stays in ``svarog_industrial`` (``PumpModules``, ``SignalCombiner``).

Nothing here is about damage or targets: the far field is at the combiner's output, not on a workpiece.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from vegeta import hyperion as hy
from vegeta.hyperion import combining as cb

import svarog_industrial as si

WAVELENGTH = 1070e-9                                   # the Yb power stage's signal
MODULE_FIBRE = si.FIBRE_POWER                           # the 30/250 fibre whose LP₀₁ is a channel
MODULE_W0 = MODULE_FIBRE.mode_radius(WAVELENGTH)        # 11.8 µm (Marcuse)
MODULE_M2 = 1.0                                         # the stage's LP₀₁ taken diffraction-limited (a 30/250 LMA fibre measures 1.1–1.3: ``multimode_check``)
MODULE_M2_MEASURED = 1.3
DEVICES = {"1kW": 1000.0, "2kW": 2000.0}                # the delivered powers when 46's report is not at hand
CHANNEL_COUNTS = (1, 2, 4, 8)
SPECTRAL_SPACING = 3e-9                                 # 3 nm between colours — the Yb band 1060–1085 nm holds eight
GRATING_EFFICIENCY = 0.95                               # a multilayer-dielectric grating per channel
GRATING_DISPERSION = 2.0e-3 / 1e-9                      # [rad/m] a 1740 l/mm grating near Littrow at 1070 nm: 2 mrad per nm
RESIDUAL_FRACTIONS = (0.0, 0.05, 0.1, 0.25, 0.5)        # the share of it left uncorrected (alignment, the channels' linewidth)
GRID_N = 256


def powers_from(report: Path | None) -> dict:
    """{variant: W delivered} from notebook 46's report when it is there, else the nominal ones."""
    out = dict(DEVICES)
    if report is not None and Path(report).exists():
        rep = json.loads(Path(report).read_text())
        for k in out:
            if k in rep.get("industrial", {}) and rep["industrial"][k].get("delivered_W"):
                out[k] = float(rep["industrial"][k]["delivered_W"])
    return out


def channel_set(n: int, total_W: float, layout: str = "ring", method: str = "incoherent", losses=None, phases=None, w0: float = MODULE_W0,
                m2: float = MODULE_M2) -> list[cb.Channel]:
    """``n`` channels sharing ``total_W`` equally: side by side (``layout``) for incoherent and tiled-coherent combining,
    overlapped for filled-coherent and spectral; distinct wavelengths for spectral."""
    lay = "overlap" if method in ("coherent-filled", "spectral") else layout
    lams = [WAVELENGTH + (i - (n - 1) / 2) * SPECTRAL_SPACING for i in range(n)] if method == "spectral" else None
    return cb.channels(n, WAVELENGTH, total_W / n, w0, layout=lay, m2=m2, phases=phases, losses=losses, wavelengths=lams, name=f"{method[:3]}")


COMBINERS = {"incoherent": cb.IncoherentCombiner(), "coherent-filled": cb.CoherentCombiner(aperture="filled"),
             "coherent-tiled": cb.CoherentCombiner(aperture="tiled"), "spectral": cb.SpectralCombiner(efficiency_per_channel=GRATING_EFFICIENCY, min_separation=1e-9)}


def study(total_W: float, counts=CHANNEL_COUNTS, methods=None, n: int = GRID_N, margin: float = 12.0) -> dict:
    """{method: {N: Combined}} for one device's power, every method on one grid per N (so the maps compare); ``margin``
    waists of window around the channels (the far-field pixel is λ/window: 12 for the tables, 30 for finer maps)."""
    methods = methods or list(COMBINERS)
    out = {m: {} for m in methods}
    for k in counts:
        grid = cb.Grid.for_channels(channel_set(k, total_W, method="incoherent"), n, margin=margin)   # the widest set sizes the grid
        for m in methods:
            ch = channel_set(k, total_W, method=m)
            out[m][k] = COMBINERS[m].combine(ch, grid)
    return out


def equivalent(total_W: float, grid: cb.Grid) -> cb.Combined:
    """One Gaussian of the device's power on the module's beam — what a single laser of that power would give."""
    return cb.equivalent_gaussian(total_W, WAVELENGTH, MODULE_W0, grid, MODULE_M2)


def metrics_table(res: dict, total_W: float) -> list[dict]:
    """Rows (method, N, power out, efficiency, σx, σθx, BPP, M², brightness, on-axis far field, overlap) plus the
    equivalent Gaussian's row."""
    rows = []
    first = next(iter(next(iter(res.values())).values()))
    eq = equivalent(total_W, first.grid)
    rows.append({"method": "one Gaussian", "N": 1, **_row(eq)})
    for m, d in res.items():
        for k, r in d.items():
            rows.append({"method": m, "N": k, **_row(r)})
    return rows


def _row(r: cb.Combined) -> dict:
    return {"P_out [W]": r.power_out, "efficiency": r.efficiency, "σx [µm]": r.sigma_x * 1e6, "σθx [mrad]": r.sigma_tx * 1e3, "BPP [mm·mrad]": r.bpp_x * 1e6,
            "M²": r.m2_x, "brightness [W/(cm²·sr)]": r.brightness / 1e4, "on-axis [W/sr]": r.I_far_axis, "Parseval": r.conservation, "overlap": r.overlap,
            "aperture": r.extra.get("aperture", "")}


def phase_study(total_W: float, counts=(2, 4, 8), sigmas=(0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5), trials: int = 30, aperture: str = "filled") -> dict:
    """The coherent combiner's on-axis gain and efficiency against the rms phase error, per channel count."""
    out = {}
    for k in counts:
        ch = channel_set(k, total_W, method="coherent-filled" if aperture == "filled" else "coherent-tiled")
        out[k] = cb.phase_sensitivity(ch, cb.CoherentCombiner(aperture=aperture), sigmas, trials=trials, seed=k)
    return out


def loss_study(total_W: float, n: int = 4, losses=(0.0, 0.01, 0.02, 0.05, 0.1), one_channel: bool = False) -> dict:
    """Every method's power out and brightness when each channel loses ``loss`` on the way in (``one_channel``: only
    the first does — a bad splice, the others clean; the coherent filled sum then also pays the imbalance,
    1 − (Σ√P_i)²/(N ΣP_i), reported as ``imbalance [%]`` — second order, 0.05 % for one channel at −10 %)."""
    out = {m: [] for m in COMBINERS}
    grid = cb.Grid.for_channels(channel_set(n, total_W), GRID_N)
    for L in losses:
        ls = [L] + [0.0] * (n - 1) if one_channel else [L] * n
        for m, comb in COMBINERS.items():
            r = comb.combine(channel_set(n, total_W, method=m, losses=ls), grid)
            avail = sum(c.power_in for c in channel_set(n, total_W, method=m, losses=ls)) * (GRATING_EFFICIENCY if m == "spectral" else 1.0)
            out[m].append({"loss": L, "P_out [W]": r.power_out, "efficiency": r.efficiency, "brightness [W/(cm²·sr)]": r.brightness / 1e4,
                           "imbalance [%]": 100 * (1 - r.power_out / avail) if avail > 0 else math.nan})
    return out


def interference_pair(total_W: float, pitch_w0: float = 4.0, phase: float = 0.0, n: int = 512, margin: float = 36.0) -> cb.Combined:
    """Two tiled channels ``pitch_w0`` waists apart with a phase difference: the two-beam fringes (a wide window for a
    fine far-field pixel)."""
    ch = cb.channels(2, WAVELENGTH, total_W / 2, MODULE_W0, pitch=pitch_w0 * MODULE_W0, layout="line", m2=MODULE_M2, phases=[0.0, phase])
    return cb.CoherentCombiner(aperture="tiled").combine(ch, cb.Grid.for_channels(ch, n, margin=margin))


def spectral_residual(total_W: float, n: int = 4, fractions=RESIDUAL_FRACTIONS) -> list[dict]:
    """The spectral combiner's far-field width and brightness against the residual dispersion — the share of the
    grating's angular dispersion the alignment (or the channels' linewidth) leaves uncorrected."""
    rows = []
    for f in fractions:
        comb = cb.SpectralCombiner(efficiency_per_channel=GRATING_EFFICIENCY, min_separation=1e-9, dispersion=f * GRATING_DISPERSION)
        r = comb.combine(channel_set(n, total_W, method="spectral"))
        rows.append({"residual share": f, "residual [mrad/nm]": f * GRATING_DISPERSION * 1e-9 * 1e3, "pointing max [mrad]": r.extra["residual_pointing_max"] * 1e3,
                     "σθx [mrad]": r.sigma_tx * 1e3, "σθy [mrad]": r.sigma_ty * 1e3, "BPP x [mm·mrad]": r.bpp_x * 1e6, "brightness [W/(cm²·sr)]": r.brightness / 1e4})
    return rows


def multimode_check(total_W: float, counts=CHANNEL_COUNTS, m2: float = MODULE_M2_MEASURED) -> list[dict]:
    """The incoherent and spectral rows again with the measured M² of the LMA stage: the far field widens by M², the
    brightness falls by M⁴, the ranking does not change (the coherent combiner takes diffraction-limited channels only)."""
    rows = []
    for k in counts:
        for m in ("incoherent", "spectral"):
            r = COMBINERS[m].combine(channel_set(k, total_W, method=m, m2=m2))
            rows.append({"method": m, "N": k, "M² channel": m2, **_row(r)})
    return rows


def record(powers: dict, studies: dict, phases: dict, losses: dict, spectral: dict, note: str = "") -> dict:
    """The report: per device the metrics of every (method, N) and the equivalent Gaussian, the phase and loss studies."""
    def ps(d):
        return {str(k): {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in v.items()} for k, v in d.items()}
    return {"meta": {"module": {"fibre": MODULE_FIBRE.to_dict(), "w0": MODULE_W0, "m2": MODULE_M2, "wavelength": WAVELENGTH},
                     "spectral_spacing": SPECTRAL_SPACING, "grating_efficiency": GRATING_EFFICIENCY, "grid_n": GRID_N, "note": note},
            "devices": {k: {"power_W": powers[k], "metrics": metrics_table(studies[k], powers[k]),
                            "results": {m: {str(n): r.to_dict() for n, r in d.items()} for m, d in studies[k].items()}} for k in studies},
            "phase": {k: ps(v) for k, v in phases.items()}, "losses": losses, "spectral": spectral}
