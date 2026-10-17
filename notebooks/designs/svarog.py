"""Svarog — the CW diode-end-pumped Nd:YAG laser designed on Hyperion (notebook 46).

The device as numbers: the rod (`ROD`), the fibre-coupled pump and its imaging train (`Pump`, `pump_train`,
`pump_in_rod`), the output coupling (`T_CANDIDATES`), four cavity revisions (`REVISIONS`: A the textbook plano–concave
with a separate dichroic mirror, B the HR coating on the rod's pump face and a short arm, C a folded three-mirror cavity
with a concave fold at an angle, D the zone-II high-power cavity that the thermal lens opens) built with the pump's
thermal lens at the absorption centroid (`cavity`), the
laser's steady state at an operating point with the mode the cavity really has (`operating_point`, `power_curve`), the
stability window over the pump (`stability_sweep`), the comparison of the revisions (`compare`), the output beam and
its delivery train (`delivery_train`), the record (`design_record`). Everything is Hyperion; nothing is run."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from vegeta import hyperion as hy

LAM = 1064e-9                      # the laser line
LAM_P = 808e-9                     # the pump
LOSS_ROUND_TRIP = 0.012            # scatter and absorption per round trip besides the mirrors: the rod's two AR faces (0.2 % each),
                                   # bulk and coating losses — a clean small DPSS; measured ones range 0.5–3 %
STRESS_AND_BULGE = 0.2             # the thermal lens beyond dn/dT (stress birefringence, end bulging) as a fraction, Nd:YAG literature
ND_LINEWIDTH_HZ = 120e9            # Nd:YAG 1064 nm gain bandwidth (~0.45 nm): how many longitudinal modes fit

ROD = hy.Rod(hy.ND_YAG, radius=1.5e-3, length=0.010, name="rod")     # Ø3 mm × 10 mm, 1.0 at.% — absorbs 98 % of the pump
T_CANDIDATES = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20)


# ------------------------------------------------------------------------------------------------------- the pump
@dataclass(frozen=True)
class Pump:
    """A fibre-coupled diode bar: ``power_max`` [W] at 808 nm from a fibre of ``core_radius`` [m] and numerical aperture
    ``NA``, imaged onto the rod's face by a collimator ``f_collimator`` and a focusing lens ``f_focus`` (the spot is the
    core radius × f_focus/f_collimator; the beam's M² = π r NA/λ fixes how fast it then diverges inside the rod)."""
    power_max: float = 30.0
    core_radius: float = 100e-6
    NA: float = 0.22
    f_collimator: float = 0.025
    f_focus: float = 0.100
    name: str = "808 nm fibre-coupled diode"

    @property
    def magnification(self) -> float:
        return self.f_focus / self.f_collimator

    @property
    def spot_radius(self) -> float:
        """The pump's geometric image radius on the rod's entrance face."""
        return self.core_radius * self.magnification

    def beam(self) -> hy.GaussianBeam:
        return hy.GaussianBeam.from_fibre(LAM_P, self.core_radius, self.NA, name=self.name)

    def with_magnification(self, m: float) -> "Pump":
        return replace(self, f_focus=self.f_collimator * m)


PUMP = Pump()


def pump_train(pump: Pump) -> hy.Train:
    """Fibre end → collimator (one focal length away) → gap → focusing lens → the rod's face at its focus: a 1:M image."""
    return hy.Train([hy.FreeSpace(pump.f_collimator, name="fibre to collimator"), hy.ThinLens(pump.f_collimator, name="collimator"),
                     hy.FreeSpace(0.040, name="collimated"), hy.ThinLens(pump.f_focus, name="focus"), hy.FreeSpace(pump.f_focus, name="to the rod")],
                    name="pump train")


def pump_in_rod(pump: Pump, rod: hy.Rod = ROD) -> dict:
    """The pump beam at the rod's face (from the train) and inside the rod (n = YAG's at 808 nm, the beam's M²):
    w_p(z), the absorption-weighted effective radius, the radius where 1/e of the pump is left, and the trace."""
    tr = pump.beam().through(pump_train(pump))
    face = tr.out                                         # the beam at the rod's face, in air
    n_p = hy.YAG.n(LAM_P)
    inside = hy.GaussianBeam.from_q(face.q * n_p, LAM_P, n_p, face.m2, "pump in the rod")   # a flat interface: q scales with n
    z = np.linspace(0.0, rod.length, 201)
    w = inside.w(z)
    return {"trace": tr, "at_face": face, "inside": inside, "z": z, "w": w, "w_face": float(face.w(0.0)),
            "w_effective": hy.effective_pump_radius(inside.w, rod), "w_at_absorption_length": float(inside.w(rod.absorption_length)),
            "w_exit": float(inside.w(rod.length)), "rayleigh_in_rod": inside.rayleigh_range, "m2": face.m2}


def absorption_centroid(rod: hy.Rod = ROD) -> float:
    """Where the heat sits: the absorption-weighted mean depth of the pump in the rod."""
    a, L = rod.medium.alpha_pump, rod.length
    return (1.0 / a) - L * math.exp(-a * L) / (1.0 - math.exp(-a * L))


# ----------------------------------------------------------------------------------------------------- the cavities
@dataclass(frozen=True)
class Revision:
    """One cavity: ``build(f_thermal)`` returns the Train from the HR end to the output coupler with the rod split at the
    absorption centroid around a ThermalLens of focal length ``f_thermal``; ``T`` the coupler's transmission."""
    name: str
    title: str
    kind: str                       # "dichroic" (a separate flat HR before the rod), "rod_hr" (the HR coating on the rod's face), "folded"
    T: float
    hr_reflectivity: float = 0.999
    rod: hy.Rod = ROD
    note: str = ""
    params: dict = field(default_factory=dict)

    def rod_parts(self, f_thermal: float, pump_W: float) -> list:
        zc = absorption_centroid(self.rod)
        return [hy.Slab(zc, hy.YAG, name="rod a"), hy.ThermalLens(f_thermal, pump_W, name="thermal lens"), hy.Slab(self.rod.length - zc, hy.YAG, name="rod b")]

    def build(self, f_thermal: float = math.inf, pump_W: float = 0.0) -> hy.Train:
        p = self.params
        rod = self.rod_parts(f_thermal, pump_W)
        hr = hy.FlatMirror(self.hr_reflectivity, name="HR")
        if self.kind == "dichroic":
            return hy.Train([hr, hy.FreeSpace(p["gap"], name="gap"), *rod, hy.FreeSpace(p["L"] - p["gap"] - self.rod.length, name="arm"),
                             hy.CurvedMirror(p["R_oc"], 0.0, 1.0 - self.T, name="OC")], self.name)
        if self.kind == "rod_hr":
            return hy.Train([hr, *rod, hy.FreeSpace(p["L"] - self.rod.length, name="arm"), hy.CurvedMirror(p["R_oc"], 0.0, 1.0 - self.T, name="OC")], self.name)
        if self.kind == "folded":
            return hy.Train([hr, *rod, hy.FreeSpace(p["d1"] - self.rod.length, name="arm 1"), hy.CurvedMirror(p["R_fold"], p["angle"], 0.999, name="fold"),
                             hy.FreeSpace(p["d2"], name="arm 2"), hy.FlatMirror(1.0 - self.T, name="OC")], self.name)
        raise ValueError(self.kind)

    @property
    def folded(self) -> bool:
        return self.kind == "folded"

    @property
    def length(self) -> float:
        return self.build().length


REVISIONS = {
    "A": Revision("A", "textbook plano–concave: separate dichroic HR, L 150 mm, R_oc 500 mm", "dichroic", T=0.10,
                  params={"gap": 0.005, "L": 0.150, "R_oc": 0.500},
                  note="a flat dichroic HR (HT at 808 nm) 5 mm before the rod, a long arm, a gentle concave coupler: the layout of the "
                       "low-power textbook, which the thermal lens closes before the design pump"),
    "B": Revision("B", "compact zone I: HR on the rod's face, L 80 mm, R_oc 500 mm", "rod_hr", T=0.10, params={"L": 0.080, "R_oc": 0.500},
                  note="the HR/HT dichroic coating on the rod itself, a short arm: stable from cold to past the pump's maximum, the mode half the pump spot"),
    "C": Revision("C", "folded: HR on the rod, arm 150 mm, concave fold R 150 mm at 10°, arm 90 mm, flat OC", "folded", T=0.10,
                  params={"d1": 0.150, "R_fold": 0.150, "angle": 10.0, "d2": 0.090},
                  note="the fold mirror makes a second, nearly collimated arm (room for a Q-switch or a doubling crystal later); its angle costs astigmatism"),
    "D": Revision("D", "zone II, high power: HR on the rod, L 220 mm, R_oc 150 mm (L > R_oc)", "rod_hr", T=0.10, params={"L": 0.220, "R_oc": 0.150},
                  note="unstable cold; the thermal lens opens it above ~13 W and the mode in the rod is then two thirds of the pump spot — the "
                       "high-power design rule (Magni): the mode's size is set by the dioptric width of the window, not by the mirrors"),
}


def thermal_lens(pump_W: float, w_pump: float, rod: hy.Rod = ROD) -> hy.ThermalLens:
    return hy.thermal_lens(rod, pump_W, w_pump, STRESS_AND_BULGE)


def cavity(rev: Revision, pump_W: float, w_pump: float) -> hy.Resonator:
    """The revision's resonator at a pump power: the thermal lens at the absorption centroid."""
    f = thermal_lens(pump_W, w_pump, rev.rod).f
    return hy.Resonator(rev.build(f, pump_W), LAM, loss=LOSS_ROUND_TRIP, name=f"Svarog {rev.name}")


def mode_in_rod(res: hy.Resonator) -> dict:
    """The mode radius at the thermal-lens plane (the end of ``rod a``) in both planes, their geometric mean, the waist,
    the spot on the coupler and on the HR — nan when unstable."""
    out = {}
    for p, m in res.modes(samples=2).items():
        out[f"stable_{p}"] = m.stable
        out[f"w_rod_{p}"] = m.spot("rod a") if m.stable else math.nan
        out[f"w_oc_{p}"] = m.spot("OC") if m.stable else math.nan
        out[f"w_hr_{p}"] = m.spot("HR") if m.stable else math.nan
        out[f"w0_{p}"] = m.w0 if m.stable else math.nan
        out[f"m_{p}"] = m.m
    out["stable"] = out["stable_t"] and out["stable_s"]
    out["w_rod"] = math.sqrt(out["w_rod_t"] * out["w_rod_s"]) if out["stable"] else math.nan
    out["astigmatism"] = out["w_rod_t"] / out["w_rod_s"] if out["stable"] else math.nan
    return out


def operating_point(rev: Revision, pump_W: float, w_pump: float, T: float | None = None) -> dict:
    """The laser at one pump power with the mode its cavity has there (self-consistent in the thermal lens):
    the cavity's numbers, the Laser's table, the thermal lens, the rod's temperature."""
    res = cavity(rev, pump_W, w_pump)
    m = mode_in_rod(res)
    tl = thermal_lens(pump_W, w_pump, rev.rod)
    out = {"revision": rev.name, "pump_W": pump_W, "w_pump": w_pump, "f_thermal": tl.f, **m}
    if m["stable"]:
        laser = hy.Laser(rev.rod, w_pump, m["w_rod"], rev.T if T is None else T, LOSS_ROUND_TRIP + (1.0 - rev.hr_reflectivity), rev.name)
        out.update(laser.table(pump_W))
        out["laser"] = laser
        out["overlap_ratio"] = m["w_rod"] / w_pump
        out["pump_inside_mode"] = min(1.0, out["overlap_ratio"] ** 2)    # a top-hat pump: the share of it inside the mode radius
    else:
        out.update({"output_W": 0.0, "threshold_W": math.nan, "slope_efficiency": math.nan, "overlap_ratio": math.nan, "pump_inside_mode": math.nan})
    out["resonator"] = res
    out.update({f"rod_{k}": v for k, v in hy.rod_temperature_rise(rev.rod, pump_W, w_pump).items()})
    return out


def power_curve(rev: Revision, w_pump: float, pumps=None, T: float | None = None) -> dict:
    """Output, threshold and the mode radius against the pump power, the cavity re-solved at every point."""
    pumps = np.linspace(0.0, PUMP.power_max, 61) if pumps is None else np.asarray(pumps, dtype=float)
    rows = [operating_point(rev, float(p), w_pump, T) for p in pumps]
    keys = ("output_W", "threshold_W", "w_rod", "w_rod_t", "w_rod_s", "w_oc_t", "f_thermal", "m_t", "m_s", "overlap_ratio", "slope_efficiency")
    out = {"pump_W": pumps, "stable": np.array([r["stable"] for r in rows])}
    for k in keys:
        out[k] = np.array([r.get(k, math.nan) for r in rows], dtype=float)
    return out


def stability_sweep(rev: Revision, w_pump: float, pumps=None, plane: str = "t") -> dict:
    """Hyperion's sweep over the pump power: m, stability, waist, the spot on the rod, the HR and the OC."""
    pumps = np.linspace(0.0, 1.5 * PUMP.power_max, 91) if pumps is None else pumps
    return hy.Resonator.sweep(lambda p: cavity(rev, p, w_pump), pumps, elements=["HR", "rod a", "OC"], plane=plane)


def stability_window(rev: Revision, w_pump: float, pumps=None) -> dict:
    """The pump range over which both planes are stable (the first window from zero, and whether it is open at 0 W)."""
    sw_t, sw_s = stability_sweep(rev, w_pump, pumps, "t"), stability_sweep(rev, w_pump, pumps, "s")
    ok = sw_t["stable"] & sw_s["stable"]
    p = sw_t["values"]
    out = {"open_at_zero": bool(ok[0]), "p_low": math.nan, "p_high": math.nan, "windows": [], "sweep_t": sw_t, "sweep_s": sw_s}
    i = 0
    while i < len(ok):                                            # every contiguous stable run: [from, to] in watts
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            out["windows"].append((float(p[i]), float(p[j]) if j + 1 < len(ok) else math.inf))
            i = j + 1
        else:
            i += 1
    if out["windows"]:
        out["p_low"], out["p_high"] = out["windows"][0]
    return out


def window_around(window: dict, pump_W: float) -> tuple[float, float] | None:
    """The stable window that contains ``pump_W`` (from, to), or None."""
    for lo, hi in window["windows"]:
        if lo <= pump_W <= hi:
            return (lo, hi)
    return None


def compare(revs, pump_W: float, w_pump: float) -> dict:
    """One row per revision at the design pump: length, mode in the rod against the pump, overlap ratio, astigmatism,
    threshold, output, slope, the thermal lens, the stability window, the beam out (waist and divergence)."""
    rows = {}
    for rev in revs:
        op = operating_point(rev, pump_W, w_pump)
        win = stability_window(rev, w_pump)
        around = window_around(win, pump_W) or (math.nan, math.nan)
        res = op["resonator"]
        spec = res.spectrum()
        out = output_beam(res) if op["stable"] else None
        rows[rev.name] = {"title": rev.title, "length [mm]": rev.length * 1e3, "T": rev.T, "stable": op["stable"],
                          "mode in rod [µm]": op["w_rod"] * 1e6, "pump radius [µm]": w_pump * 1e6, "mode / pump": op["overlap_ratio"],
                          "pump inside the mode": op["pump_inside_mode"],
                          "astigmatism w_t/w_s": op["astigmatism"], "spot on OC [µm]": op.get("w_oc_t", math.nan) * 1e6,
                          "threshold [W]": op["threshold_W"], "slope": op["slope_efficiency"], "output [W]": op["output_W"],
                          "thermal lens f [mm]": op["f_thermal"] * 1e3, "stable from 0 W": win["open_at_zero"],
                          "window from [W]": around[0], "window to [W]": around[1],
                          "FSR [MHz]": spec["fsr"] / 1e6, "linewidth [MHz]": spec["linewidth"] / 1e6,
                          "modes under the gain line": ND_LINEWIDTH_HZ / spec["fsr"],
                          "beam out w0 [µm]": out["w0"] * 1e6 if out else math.nan, "beam out divergence [mrad]": out["divergence"] * 1e3 if out else math.nan}
    return rows


# ------------------------------------------------------------------------------------------------- the beam out
OC_SUBSTRATE = {"thickness": 0.006, "material": hy.FUSED_SILICA}


def output_beam(res: hy.Resonator, plane: str = "t") -> dict:
    """The beam leaving the output coupler: the cavity mode on the OC, then the OC substrate (a concave inner face and
    a flat outer face: a weak negative lens for a curved coupler) — the beam in air just outside, its waist, divergence."""
    m = res.mode(plane, samples=2)
    if not m.stable:
        raise ValueError(f"{res.name}: no output beam, the cavity is unstable ({m.note})")
    oc = res.train.elements[-1]
    inside = m.trace.out                                          # the beam at the OC, inside the cavity
    if isinstance(oc, hy.CurvedMirror):
        sub = hy.ThickLens(-oc.R, math.inf, OC_SUBSTRATE["thickness"], OC_SUBSTRATE["material"], name="OC substrate")
    else:
        sub = hy.Slab(OC_SUBSTRATE["thickness"], OC_SUBSTRATE["material"], name="OC substrate")
    tr = inside.through([sub], plane)
    beam = tr.out
    return {"beam": beam, "w_oc": inside.w0 if inside.z0 == 0 else float(inside.w(0.0)), "w0": beam.w0, "z0": beam.z0, "divergence": beam.divergence,
            "rayleigh_range": beam.rayleigh_range, "substrate": sub}


def delivery_train(beam: hy.GaussianBeam, w_collimated: float = 1.0e-3, f_focus: float = 0.050, standoff: float = 0.150) -> dict:
    """From the OC: a collimating lens one focal length past the beam's waist sized for a collimated radius
    ``w_collimated``, a stand-off, a focusing lens ``f_focus`` and the work spot: the train, its trace, the spot and the
    depth of focus (2 z_R) at the work plane."""
    f_c = math.pi * beam.w0 * w_collimated / (beam.m2 * LAM)     # a lens one focal length from the waist: w' = λ f / (π w0)
    gap = max(f_c + beam.z0, 0.005)                               # beam.z0 > 0: the waist lies ahead of the OC; the collimator one f past it (≥ 5 mm from the OC)
    train = hy.Train([hy.FreeSpace(gap, name="OC to collimator"), hy.ThinLens(f_c, name="collimator"), hy.FreeSpace(standoff, name="stand-off"),
                      hy.ThinLens(f_focus, name="focusing lens"), hy.FreeSpace(f_focus * 1.05, name="to the work")], name="delivery")
    tr = beam.through(train, samples=60)
    work = tr.waists[-1] if tr.waists else None
    return {"train": train, "trace": tr, "f_collimator": f_c, "collimator_at": gap, "w_collimated": tr.spot_at("focusing lens")["w"],
            "work_spot": work["w0"] if work else math.nan, "work_distance": (work["z"] - (gap + standoff)) if work else math.nan,
            "depth_of_focus": 2 * math.pi * work["w0"] ** 2 / (beam.m2 * LAM) if work else math.nan}


# ---------------------------------------------------------------------------------------------------- the record
def design_record(rev: Revision, pump: Pump, pump_W: float, w_pump: float) -> dict:
    """What the next notebook (the build) reads: the chosen revision's cavity at the design pump, the pump, the laser's
    numbers, the beam out and the delivery train — Hyperion objects where they are, numbers elsewhere."""
    op = operating_point(rev, pump_W, w_pump)
    out = output_beam(op["resonator"])
    dl = delivery_train(out["beam"])
    return {"name": "Svarog", "revision": rev.name, "title": rev.title, "wavelength": LAM, "pump_wavelength": LAM_P,
            "rod": rev.rod.to_dict(), "pump": {"power_max": pump.power_max, "core_radius": pump.core_radius, "NA": pump.NA,
                                                "f_collimator": pump.f_collimator, "f_focus": pump.f_focus, "spot_radius": pump.spot_radius,
                                                "w_effective": w_pump, "design_power": pump_W},
            "cavity": op["resonator"], "cavity_cold": hy.Resonator(rev.build(), LAM, loss=LOSS_ROUND_TRIP, name=f"Svarog {rev.name} cold"),
            "laser": {k: op[k] for k in ("threshold_W", "slope_efficiency", "output_W", "intracavity_W", "small_signal_gain", "w_rod", "w_rod_t", "w_rod_s",
                                        "f_thermal", "T", "loss", "heat_W", "rod_dT_centre_to_barrel") if k in op},
            "beam_out": out["beam"], "delivery": {"train": dl["train"], "f_collimator": dl["f_collimator"], "work_spot": dl["work_spot"],
                                                   "depth_of_focus": dl["depth_of_focus"], "w_collimated": dl["w_collimated"]}}
