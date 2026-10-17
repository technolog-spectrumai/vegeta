"""Svarog-industrial — the 1 kW welding laser and its guard (notebooks 48 and 49): the fibre MOPA scaled to a kilowatt on
Hyperion's `fibre` module, the 20 m process fibre and the welding head on `welding`, the welds it makes, and the stray
beam at 20 m against the barriers on `thermal` — with the closed-form screening that picks the one simulation worth
running per barrier.

Everything is numbers and Hyperion; nothing is run. What was built before (the rod, the fibre oscillator, the first
amplifier) stays as it is: this module composes them towards the industrial device."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from vegeta import hyperion as hy
from vegeta.hyperion import fibre as fb
from vegeta.hyperion import thermal as th
from vegeta.hyperion import welding as wd

import svarog_fiber as sf

LAM, LAM_P = sf.LAM, sf.LAM_P
POWER_TARGET = 1000.0                           # W at the workpiece
SEED_DEFAULT = 59.0                             # W: notebook 47's amplifier output when its report is not at hand

# ---------------------------------------------------------------------------------------------- the power stage
FIBRE_POWER = hy.Fibre(core_radius=15e-6, NA=0.06, clad_radius=125e-6, coating_radius=200e-6, clad_NA=0.46,
                       alpha_signal=sf.LOSS_DB_PER_KM / 1e3 / fb.DB, alpha_pump=sf.LOSS_DB_PER_KM / 1e3 / fb.DB, name="30/250 Yb double-clad")
ABSORPTION_POWER_DB_PER_M = 2.0                 # the datasheet's cladding absorption at 976 nm


@dataclass(frozen=True)
class PumpModules:
    """Wavelength-stabilised 976 nm pump modules of ``module_W`` on 200/220 µm, NA 0.22 pigtails into a (6+1)×1 combiner."""
    module_W: float = 220.0
    n_modules: int = 6
    combiner_efficiency: float = 0.95
    pigtail_core_radius: float = 100e-6
    pigtail_NA: float = 0.22
    name: str = "976 nm 220 W modules, (6+1)×1"

    @property
    def launched_max(self) -> float:
        return self.module_W * self.n_modules * self.combiner_efficiency

    def fits_the_cladding(self, fibre: hy.Fibre) -> dict:
        per_port = (self.pigtail_core_radius * self.pigtail_NA) ** 2
        clad = (fibre.clad_radius * fibre.clad_NA) ** 2
        return {"ports_that_fit": int(clad / per_port), "used": self.n_modules, "fits": self.n_modules * per_port <= clad}


PUMP_POWER = PumpModules()


def power_active(length: float) -> hy.ActiveFibre:
    yb = fb.YB_SILICA.with_(N=fb.ions_for_absorption(FIBRE_POWER, ABSORPTION_POWER_DB_PER_M, LAM_P))
    return hy.ActiveFibre(FIBRE_POWER, yb, length, name=f"{FIBRE_POWER.name}, {length:g} m")


def power_stage(length: float, direction: str = "counter") -> hy.FibreAmplifier:
    return hy.FibreAmplifier(power_active(length), LAM_P, LAM, direction, name=f"Svarog-industrial power stage, {length:g} m")


def power_stage_length_choice(seed_W: float, pump_W: float, lengths=None, within: float = 0.03, n: int = 200) -> dict:
    lengths = np.arange(4.0, 20.01, 1.0) if lengths is None else np.asarray(lengths, dtype=float)
    rows = []
    for L in lengths:
        s = power_stage(float(L)).solve(seed_W, pump_W, n)
        rows.append({"length": float(L), "output_W": s.output_W, "gain_db": s.gain_db, "unabsorbed_W": s.unabsorbed_W, "heat_W": s.heat_W,
                     "heat_peak_per_m": s.peak_heat_per_m, "inversion_max": float(s.inversion.max())})
    best = max(rows, key=lambda r: r["output_W"])
    chosen = next(r for r in rows if r["output_W"] >= (1.0 - within) * best["output_W"])
    return {"rows": rows, "best_length": best["length"], "chosen_length": chosen["length"], "chosen_output_W": chosen["output_W"], "rule": f"shortest within {within:.0%} of the best"}


def pump_for_target(length: float, seed_W: float, target_W: float = POWER_TARGET, n: int = 200) -> float:
    """The launched pump that puts ``target_W`` out of the stage (bisection), capped at the modules' maximum."""
    amp = power_stage(length)
    lo, hi = 0.0, PUMP_POWER.launched_max
    if amp.solve(seed_W, hi, n).output_W < target_W:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if amp.solve(seed_W, mid, n).output_W < target_W:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------------------------- heat exchange
COOLING_POWER = {"water-cooled spool (h 2000)": 2000.0, "cooled plate (h 500)": 500.0, "still air (h 10)": 10.0}


def thermal_budget(sol: fb.FibreSolution, length: float, delivery_loss_db: float = 0.3, lens_absorption: float = 0.002) -> dict:
    """Where the watts go: the heat in the active fibre (the pump's quantum defect) and the core temperature per cooling,
    the unabsorbed pump and the splice loss dumped in a cladding-mode stripper, the delivery fibre's loss, the two lenses
    of the head (each absorbing ``lens_absorption`` of the beam: a conduction estimate on BK7's centre rise)."""
    temps = {name: hy.core_temperature(FIBRE_POWER, sol.peak_heat_per_m, h=h)["dT_core_to_sink"] for name, h in COOLING_POWER.items()}
    stripper = sol.unabsorbed_W + sol.output_W * (1 - 10 ** (-0.1 / 10))          # the unabsorbed pump and ~0.1 dB of splice loss into the stripper
    delivered = sol.output_W * 10 ** (-delivery_loss_db / 10)
    lens_W = lens_absorption * delivered
    lens_dT = th.gaussian_steady_rise(th.Solid("BK7", 2510, 858, 1.114, 830, 0.0, 2000, 0.0, 1.0), lens_W, 0.012)
    return {"active_fibre_heat_W": sol.heat_W, "active_fibre_peak_W_per_m": sol.peak_heat_per_m, "core_dT_per_cooling": temps,
            "stripper_W": stripper, "delivery_loss_W": sol.output_W - delivered, "delivered_W": delivered, "lens_absorbed_W_each": lens_W,
            "lens_centre_dT": lens_dT, "wall_plug_pump_W": None}


# ---------------------------------------------------------------------------------------------- the delivery and the head
DELIVERY = wd.DeliveryFibre(core_radius=25e-6, NA=0.12, length=20.0, name="50/125 process fibre, 20 m")
HEAD = wd.WeldHead(f_collimator=0.100, f_focus=0.200, name="welding head 100/200")


def head_numbers(power_W: float) -> dict:
    spot = HEAD.spot_radius(DELIVERY)
    return {"bpp_mm_mrad": DELIVERY.bpp * 1e6, "m2": DELIVERY.m2(LAM), "collimated_radius": HEAD.collimated_radius(DELIVERY), "spot_radius": spot,
            "cone_half_angle": HEAD.cone_half_angle(DELIVERY), "rayleigh_length": HEAD.rayleigh_length(DELIVERY),
            "mean_irradiance_W_per_cm2": power_W / (math.pi * spot ** 2) / 1e4,
            "srs_delivery": hy.srs_threshold(hy.Fibre(DELIVERY.core_radius, DELIVERY.NA, 62.5e-6, alpha_signal=2.3e-3), DELIVERY.length, LAM)}


# ---------------------------------------------------------------------------------------------- the welds
WELD_MATERIALS = {"mild steel": {"solid": th.MILD_STEEL, "absorption": 0.7, "thicknesses": (1e-3, 2e-3, 3e-3, 4e-3, 6e-3)},
                  "aluminium": {"solid": th.ALUMINIUM, "absorption": 0.5, "thicknesses": (1e-3, 2e-3, 3e-3, 4e-3)}}
SPEEDS = np.array([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]) / 60.0       # m/s


def welds(power_W: float) -> dict:
    """Per material: the keyhole threshold at the head's spot, the penetration curve over the speeds, the process window
    per thickness (the fastest full-penetration speed and the speed with 20 % of depth in hand)."""
    spot = HEAD.spot_radius(DELIVERY)
    out = {}
    for name, m in WELD_MATERIALS.items():
        out[name] = {"keyhole": wd.keyhole_threshold(m["solid"], spot), "curve": wd.penetration_curve(m["solid"], power_W, SPEEDS, spot, absorption=m["absorption"]),
                     "window": wd.process_window(m["solid"], power_W, m["thicknesses"], spot, absorption=m["absorption"]),
                     "conduction_pool_300W": wd.rosenthal_pool(m["solid"], m["solid"].absorptivity * 300.0, 0.01)}
    return out


# ---------------------------------------------------------------------------------------------- the stray beam at 20 m
EXPOSURE = {"T1": 10.0, "T2": 100.0, "T3": 30000.0}             # IEC 60825-4 exposure classes [s]
DISTANCE = 20.0


@dataclass(frozen=True)
class StrayBeam:
    name: str
    power: float
    radius: float                                   # at the barrier
    shape: str = "tophat"
    note: str = ""

    @property
    def mean_irradiance(self) -> float:
        return self.power / (math.pi * self.radius ** 2)


def stray_beams(power_W: float, distance: float = DISTANCE) -> list[StrayBeam]:
    """The cases at ``distance``: the raw collimated beam (escaping before the focusing optic, or re-collimated by a
    reflector), the beam diverging past the focus, a keyhole back-reflection (a fifth of the power, diverging)."""
    collimated = HEAD.collimated_radius(DELIVERY) + DELIVERY.core_radius / HEAD.f_collimator * distance
    diverging = HEAD.cone_half_angle(DELIVERY) * distance
    return [StrayBeam("raw collimated beam", power_W, collimated, "tophat", "the worst case: the full power in a beam that stays small"),
            StrayBeam("diverging past the focus", power_W, diverging, "tophat", "the beam through the focus, or a flat workpiece's specular reflection"),
            StrayBeam("keyhole back-reflection", 0.2 * power_W, diverging, "tophat", "a fifth of the power thrown back out of the keyhole")]


@dataclass(frozen=True)
class Barrier:
    solid: th.Solid
    thickness: float

    @property
    def name(self) -> str:
        return f"{self.solid.name} {self.thickness * 1e3:g} mm"


BARRIERS = [Barrier(th.BLACK_PC, 5e-3), Barrier(th.BLACK_PC, 10e-3), Barrier(th.CLEAR_PC, 5e-3), Barrier(th.CLEAR_PC, 10e-3), Barrier(th.LASER_PC, 5e-3),
            Barrier(th.LASER_PC, 10e-3), Barrier(th.BRICK, 65e-3), Barrier(th.ALUMINIUM, 1.5e-3), Barrier(th.ALUMINIUM, 3e-3),
            Barrier(th.MILD_STEEL, 1.5e-3), Barrier(th.MILD_STEEL, 3e-3)]


def plate_steady_bound(solid: th.Solid, thickness: float, absorbed_W: float, radius: float, sink_radius: float) -> float:
    """An upper bound on a plate's steady centre rise under a spot of ``radius``, the edge at ``sink_radius`` held cold and
    no convection or radiation: ΔT ≤ P_abs (ln(R/w) + 1/2) / (2π k d) — the thin-plate Green's function; a thick slab
    sits lower still than the semi-infinite Gaussian rise, which is used when the plate is thicker than the spot."""
    if thickness > radius:
        return th.gaussian_steady_rise(solid, absorbed_W, radius)
    return absorbed_W * (math.log(sink_radius / radius) + 0.5) / (2 * math.pi * solid.k * thickness)


def screen(barrier: Barrier, beam: StrayBeam, T0: float = th.T_ROOM) -> dict:
    """The closed forms before any simulation: the transmitted share (a clear sheet fails at once), the semi-infinite
    time to surface damage, the lumped lower bound on the burn-through time (no lateral losses), the steady upper bound on
    the plate's rise (no losses but conduction). The decision: ``"fails: transparent"``, ``"survives: steady bound"`` (the
    bound stays under the damage point — nothing to simulate), ``"survives: energy bound"`` (even without any loss the
    burn-through takes longer than T3), or ``"simulate"``."""
    s, d = barrier.solid, barrier.thickness
    q = s.absorptivity * beam.mean_irradiance * (1.0 if beam.shape == "tophat" else 2.0)
    absorbed = s.absorptivity * beam.power
    transmitted = s.transmitted(d)
    damage_T = s.T_melt if s.failure == "melt" else s.T_fail
    t_surface = (damage_T - T0) ** 2 * math.pi * s.rho * s.c * s.k / (2 * q) ** 2 if q > 0 else math.inf
    t_lower = th.thin_plate_time_to(s, d, q, s.T_melt if s.failure == "melt" else s.T_fail) if q > 0 else math.inf
    if s.failure == "ablate" and q > 0:
        t_lower = s.enthalpy_to_fail(T0) * d / q
    rise_bound = plate_steady_bound(s, d, absorbed, beam.radius, max(8 * beam.radius, 12 * d, 0.02))
    steady_T = math.nan
    if transmitted > 0.01:
        decision = "fails: transparent"
    elif rise_bound < damage_T - T0:
        decision = "survives: steady bound"
    elif t_lower > EXPOSURE["T3"]:
        decision = "survives: energy bound"
    else:                                                            # the cheap steady solve with the losses in: does it ever get hot enough?
        st = th.steady_plate(s, d, beam.power, beam.radius, shape=beam.shape, h=10.0, nr=24, nz=_nz(d), r_max=_r_max(beam.radius, d))
        steady_T = st["T_max"]
        decision = "simulate" if st["reaches_damage"] else "survives: steady solve"
    return {"barrier": barrier.name, "beam": beam.name, "absorbed_irradiance_W_per_cm2": q / 1e4, "transmitted_fraction": transmitted,
            "t_surface_damage_bound": t_surface, "t_burn_through_lower_bound": t_lower, "steady_rise_bound_K": rise_bound, "damage_rise_K": damage_T - T0,
            "steady_T_max": steady_T, "decision": decision}


def _nz(thickness: float) -> int:
    return 8 if thickness >= 0.02 else 5


def _r_max(radius: float, thickness: float) -> float:
    return max(6 * radius, 12 * thickness, 0.05)


SIMULATION_CAP_S = 3000.0                                       # the transient's own cap: a front that has not crossed by then has stalled or is timed


def simulate(barrier: Barrier, beam: StrayBeam, t_max: float = SIMULATION_CAP_S, nr: int = 24, nz: int | None = None) -> th.HeatingResult:
    """The one transient worth running: the barrier under the beam until it fails, settles (the melt front stalled) or the cap."""
    return th.heat_plate(barrier.solid, barrier.thickness, beam.power, beam.radius, t_max, shape=beam.shape, h=10.0, nr=nr, nz=nz or _nz(barrier.thickness),
                         r_max=_r_max(beam.radius, barrier.thickness))


def verdict(scr: dict, sim: th.HeatingResult | None) -> dict:
    """Pass/fail per exposure class and the time the barrier holds, from the screening or the simulation."""
    if scr["decision"] == "fails: transparent":
        holds = 0.0
    elif scr["decision"].startswith("survives"):
        holds = math.inf
    elif sim is not None:
        # a front that stalled (steady) holds for good; one still moving at the cap is extrapolated from the centre's front speed
        if sim.failed:
            holds = sim.failure_time
        elif sim.steady:
            holds = math.inf
        else:
            front = sim.melt_depth + sim.ablated_depth
            n = max(2, len(front) // 5)
            speed = (front[-1] - front[-n]) / max(sim.t[-1] - sim.t[-n], 1e-9)
            holds = sim.t_end + (sim.thickness - front[-1]) / speed if speed > 0 else math.inf
    else:
        holds = math.nan
    out = {"holds_s": holds, "source": scr["decision"] if sim is None else ("simulation: " + (sim.failure_mode or ("steady" if sim.steady else "ran out of time")))}
    for cls, t in EXPOSURE.items():
        out[cls] = bool(holds > t)
    out["surface_damage_s"] = sim.surface_damage_time if sim is not None else scr["t_surface_damage_bound"]
    return out


def guard_study(power_W: float, barriers=None, beams=None, run: bool = True) -> dict:
    """Every barrier against every beam case: the screening for all, the simulation only where it decides, the verdict per
    class with the worst beam deciding the barrier's."""
    barriers = BARRIERS if barriers is None else barriers
    beams = stray_beams(power_W) if beams is None else beams
    rows, sims = [], {}
    for b in barriers:
        worst = None
        for beam in beams:
            scr = screen(b, beam)
            sim = None
            if scr["decision"] == "simulate" and run:
                sim = simulate(b, beam)
                sims[(b.name, beam.name)] = sim
            v = verdict(scr, sim)
            rows.append({**scr, **v, "simulated": sim is not None})
            if worst is None or v["holds_s"] < worst["holds_s"]:
                worst = {**v, "worst_beam": beam.name}
        rows[-1]["worst_for_barrier"] = worst
    return {"rows": rows, "simulations": sims, "beams": beams, "barriers": barriers}


# ---------------------------------------------------------------------------------------------- the record
def design_record(length: float, seed_W: float, pump_W: float, sol: fb.FibreSolution, budget: dict, weld: dict, guard: dict) -> dict:
    return {"name": "Svarog-industrial", "wavelength": LAM, "pump_wavelength": LAM_P, "target_W": POWER_TARGET,
            "power_stage": {"amplifier": power_stage(length).to_dict(), "fibre": FIBRE_POWER.to_dict(), "seed_W": seed_W, "pump_W": pump_W,
                            "pump": {"modules": PUMP_POWER.n_modules, "module_W": PUMP_POWER.module_W}, **sol.to_dict(),
                            "coil_radius": sf.coil_radius_for(FIBRE_POWER, length)},
            "thermal_budget": budget, "delivery": DELIVERY.to_dict(), "head": HEAD.to_dict(), "head_numbers": head_numbers(budget["delivered_W"]),
            "beam_at_work": HEAD.focused_beam(DELIVERY, LAM),
            "welds": {m: {"keyhole": w["keyhole"], "window": w["window"]} for m, w in weld.items()},
            "guard": {"distance": DISTANCE, "exposure": EXPOSURE, "beams": [vars(b) for b in guard["beams"]],
                      "verdicts": [{k: v for k, v in r.items()} for r in guard["rows"]]}}
