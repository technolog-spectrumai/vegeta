"""Zeppelin-Micro — the cheapest helium airship that lifts a daylight camera with a video link to a few hundred metres
over the sea, for a few hours (Stage V8.19). It never leaves the troposphere,
so it is a Zeppelin, not a Strato: the Strato name is kept for ships that float in the stratosphere. Two concepts, sized with the same physics and compared:

- **tethered** (an aerostat on a line from the vessel): no propulsion; the wind is held by the tether. Sized so that
  its free lift holds the tether within ``max_tether_angle`` of the vertical at the design wind, with the line's own
  weight and drag included; the camera runs on an onboard battery, the video goes by radio (an unpowered line: no
  copper in the tether, which would weigh ~10× the Dyneema).
- **free-flying** (a micro blimp): electric propellers that out-fly the design wind by ``speed_margin``, a battery
  for ``hours`` of station-keeping in the average wind, static lift = mass.

Physics from Ogma (atmosphere, gas, envelope) and ``strato_aero`` (drag build-up, actuator-disc propellers; at sea
level the hull Reynolds number is 10⁶–10⁷, inside the drag build-up's range). The maritime wind is a Weibull
distribution at 10 m (``og.wind.power_law`` profile above the sea, α = 0.11). **Estimates**: everything in
``CATALOGUE`` (materials, parts, prices — 2026 order-of-magnitude retail prices, not quotes), the wind statistics, the
tether drag coefficient, the free-lift margin. Costs are bill-of-materials plus a fabrication allowance; the report
compares concepts and design winds with them, it does not price a product.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

import ogma.wind as ow
from ogma import envelope as oe
from ogma import lifting_gas as lg
from ogma import stratosphere as st

import strato_aero as sa
import strato_geometry as sg

G0 = st.G0


@dataclass(frozen=True)
class Part:
    mass: float                   # kg (or kg per unit)
    cost: float                   # EUR (or EUR per unit)
    power: float = 0.0            # W
    note: str = ""


@dataclass(frozen=True)
class Catalogue:
    """Estimates (masses from typical products of the class; prices order-of-magnitude retail, EUR, 2026)."""
    envelopes: dict = field(default_factory=lambda: {
        "tpu-nylon": Part(0.12, 45.0, 0.01, "TPU-coated ripstop nylon, welded: 0.12 kg/m²; €15/m² fabric + €30/m² "
                          "fabrication; ~1 %/day of the helium lost (power field = leak fraction per day)"),
        "pe-film": Part(0.06, 12.0, 0.03, "UV-stabilised polyethylene film, taped: 0.06 kg/m²; cheap, fragile, weeks of "
                        "life; ~3 %/day lost"),
    })
    fin: Part = Part(0.20, 60.0, note="foam-core or inflated fins: kg and € per m² of planform")
    cameras: dict = field(default_factory=lambda: {
        "eo": Part(0.8, 1500.0, 12.0, "daylight EO camera on a 2-axis gimbal"),
        "eo-ir": Part(2.0, 4500.0, 25.0, "daylight EO + uncooled LWIR thermal camera on a stabilised 2-axis gimbal "
                      "(finds people in the water at night)"),
    })
    video_link: Part = Part(0.2, 500.0, 4.0, "digital video link to the vessel, 5–10 km")
    avionics_tethered: Part = Part(0.15, 150.0, 2.0, "power board, GPS/attitude for image tagging, beacon")
    avionics_free: Part = Part(0.35, 400.0, 6.0, "flight controller, GPS, telemetry, failsafe valve")
    battery: Part = Part(1 / 180.0, 0.6, note="Li-ion pack: kg/Wh (180 Wh/kg) and €/Wh")
    motor: Part = Part(0.6, 250.0, note="brushless motor + ESC + propeller + mount: kg/kW and €/kW")
    tether_line: Part = Part(0.0030, 0.6, note="Dyneema SK78 2.5 mm: kg/m, €/m; ~6 kN breaking")
    tether_breaking: float = 6000.0   # N
    winch: Part = Part(0.0, 900.0, note="small electric deck winch with slip ring-free drum (on the vessel)")
    mooring: Part = Part(0.1, 80.0, note="confluence lines and swivel")
    gondola: Part = Part(0.25, 150.0, note="printed or foam gondola with the camera mount")
    helium_eur_m3: float = 25.0       # €/m³ (balloon-grade, bottled; 15–40 in Europe)
    fabrication_overhead: float = 0.25   # + on the parts cost: assembly, tests, spares


CAT = Catalogue()


@dataclass(frozen=True)
class SeaWind:
    """Wind over open water (estimate): Weibull at 10 m with ``mean`` and shape ``k``; power-law profile above."""
    mean_10m: float = 7.5             # m/s, North Sea / Baltic annual mean order (5–9)
    k: float = 2.0
    alpha: float = 0.11               # shear exponent over the sea

    def at(self, speed_10m, z):
        return ow.power_law(np.asarray(speed_10m, float), z, 10.0, self.alpha)

    def fraction_below(self, speed_at_z: float, z: float) -> float:
        """Share of the hours whose wind at height ``z`` is below ``speed_at_z`` (Weibull CDF)."""
        v10 = speed_at_z / (z / 10.0) ** self.alpha
        lam = self.mean_10m / math.gamma(1 + 1 / self.k)
        return 1.0 - math.exp(-(v10 / lam) ** self.k)

    def mean_at(self, z: float) -> float:
        return self.mean_10m * (z / 10.0) ** self.alpha


SEA = SeaWind()


@dataclass(frozen=True)
class Mission:
    altitude: float = 200.0           # m above the sea (working height of the camera)
    hours: float = 4.0                # per deployment
    design_wind: float = 12.0         # m/s at the working height: the wind it must still work in
    deployments_per_year: int = 100


@dataclass(frozen=True)
class MicroDesign:
    concept: str = "tethered"         # "tethered" or "free"
    fineness: float = 3.0
    envelope: str = "tpu-nylon"
    fin_scale: float = 1.4            # aerostats carry big fins (weathervaning on the line)
    max_tether_angle: float = 45.0    # deg from the vertical at the design wind (tethered)
    free_lift_margin: float = 0.10    # of the gross lift, kept as positive free lift (gusts, cooling; tethered)
    speed_margin: float = 3.0         # m/s of airspeed above the design wind (free-flying)
    battery_reserve: float = 0.25     # of the battery (free-flying: return and landing)
    camera: str = "eo"                # "eo" (daylight) or "eo-ir" (daylight + thermal)


def _ship(length: float, d: MicroDesign) -> sg.Strato:
    lay = sg.Layout(fin_root_chord=sg.Layout.fin_root_chord * d.fin_scale, fin_tip_chord=sg.Layout.fin_tip_chord * d.fin_scale,
                    fin_span=sg.Layout.fin_span * d.fin_scale, solar_half_angle_deg=5.0)
    return sg.Strato(sg.Hull(length, length / d.fineness), lay)


def evaluate(length: float, d: MicroDesign, m: Mission = Mission(), cat: Catalogue = CAT, wind: SeaWind = SEA) -> dict:
    """Masses, lift, loads, power and cost of a ``length`` [m] hull for the mission (no sizing: see ``size``)."""
    ship = _ship(length, d)
    h = ship.hull
    air = st.state(m.altitude)
    env = cat.envelopes[d.envelope]
    q = air.dynamic_pressure(m.design_wind + (d.speed_margin if d.concept == "free" else 0.0))
    dp = max(1.2 * q, 50.0)                                   # keeps the nose round in the design wind
    hoop = float(np.max(oe.membrane_forces(*h.profile(401), dp)["hoop"]))
    gas = lg.fill_for_full(h.volume, air.pressure, air.temperature, dp, "helium", 0.98)
    gross = air.density * h.volume
    net = gross - gas                                         # kg of static lift
    fin_area = ship.fin_planform_area * ship.layout.fin_count
    cam = cat.cameras[d.camera]
    loads = cam.power + cat.video_link.power
    parts = {"envelope": (env.mass * h.surface_area * 1.1, env.cost * h.surface_area),
             "fins": (cat.fin.mass * fin_area, cat.fin.cost * fin_area),
             "gondola": (cat.gondola.mass, cat.gondola.cost),
             "camera": (cam.mass, cam.cost), "video link": (cat.video_link.mass, cat.video_link.cost)}
    out = {"concept": d.concept, "length": length, "diameter": h.diameter, "volume": h.volume, "area": h.surface_area,
           "gross_lift": gross, "gas_mass": gas, "net_lift": net, "superpressure": dp, "hoop_line_load": hoop,
           "helium_m3_stp": gas / lg.gas_density(101325.0, 288.15, "helium", 0.98)}
    if d.concept == "tethered":
        loads += cat.avionics_tethered.power
        parts["avionics"] = (cat.avionics_tethered.mass, cat.avionics_tethered.cost)
        wh = loads * m.hours / 0.85
        parts["battery"] = (cat.battery.mass * wh, cat.battery.cost * wh)
        parts["confluence lines"] = (cat.mooring.mass, cat.mooring.cost)
        theta = math.radians(d.max_tether_angle)
        line_len = m.altitude / math.cos(theta) * 1.10                         # 10 % for the catenary sag
        line_kg = cat.tether_line.mass * line_len
        # wind on the hull at the top, and on the line (Cd 1.1 on its diameter, profile-averaged q; half at the top)
        drag_hull = sa.drag(ship, m.design_wind, m.altitude).total
        q_line = np.mean([st.state(z).dynamic_pressure(float(wind.at(m.design_wind / (m.altitude / 10) ** wind.alpha, z)))
                          for z in np.linspace(5.0, m.altitude, 20)])
        drag_line = 1.1 * 0.0025 * line_len * q_line
        mass = sum(v[0] for v in parts.values())
        free = (net - mass) * G0                                              # N of free lift at the top
        horizontal = drag_hull + 0.5 * drag_line
        vertical = free - 0.5 * line_kg * G0                                  # the top carries about half the line
        angle = math.degrees(math.atan2(horizontal, max(vertical, 1e-9))) if vertical > 0 else 90.0
        tension = math.hypot(horizontal, max(vertical, 0.0))
        out.update({"system_mass": mass, "line_length": line_len, "line_mass": line_kg, "drag_hull": drag_hull,
                    "drag_line": drag_line, "free_lift_N": free, "tether_angle": angle, "tension": tension,
                    "tether_sf": cat.tether_breaking / max(tension, 1e-9), "battery_wh": wh, "propulsion_w": 0.0,
                    "feasible": (angle <= d.max_tether_angle and free >= d.free_lift_margin * gross * G0
                                 and cat.tether_breaking / max(tension, 1e-9) >= 5.0)})
        ground = {"winch": cat.winch.cost, "tether": cat.tether_line.cost * line_len}
    else:
        loads += cat.avionics_free.power
        parts["avionics"] = (cat.avionics_free.mass, cat.avionics_free.cost)
        v_air = m.design_wind + d.speed_margin
        p_max = sa.electric_power(ship, v_air, m.altitude)["power"]
        p_mean = sa.electric_power(ship, wind.mean_at(m.altitude), m.altitude)["power"]
        wh = (p_mean + loads) * m.hours / (1 - d.battery_reserve) / 0.95
        parts["propulsion"] = (cat.motor.mass * p_max / 1000, cat.motor.cost * p_max / 1000)
        parts["battery"] = (cat.battery.mass * wh, cat.battery.cost * wh)
        mass = sum(v[0] for v in parts.values())
        out.update({"system_mass": mass, "propulsion_w": p_max, "mean_power_w": p_mean, "battery_wh": wh,
                    "airspeed": v_air, "feasible": net >= mass})
        ground = {"charger and spares": 300.0}
    build = sum(v[1] for v in parts.values()) * (1 + cat.fabrication_overhead)
    fill = out["helium_m3_stp"] * cat.helium_eur_m3
    # both stay inflated through the year (on deck or in a shelter) and are topped up for the envelope's leak, plus
    # one full refill a year for repairs (estimate)
    yearly_gas = fill * (1 + env.power * 365)
    out.update({"masses": {k: v[0] for k, v in parts.items()}, "costs": {k: v[1] for k, v in parts.items()},
                "ground_costs": ground, "build_cost": build, "ground_cost": sum(ground.values()),
                "helium_fill_cost": fill, "helium_cost_per_year": yearly_gas, "loads_w": loads,
                "surplus_lift": net - mass, "operable_fraction": wind.fraction_below(m.design_wind, m.altitude)})
    return out


def size(d: MicroDesign, m: Mission = Mission(), cat: Catalogue = CAT, wind: SeaWind = SEA, lo: float = 1.0,
         hi: float = 40.0) -> dict:
    """The shortest feasible hull [m] (scan in 0.5 m steps, then bisection to 1 cm). Feasibility is not monotone in
    the length: a too-large aerostat overloads its tether, so the scan starts small. nan when nothing fits."""
    grid = np.arange(lo, hi + 1e-9, 0.5)
    first = next((L for L in grid if evaluate(L, d, m, cat, wind)["feasible"]), None)
    if first is None:
        return {"length": float("nan"), "feasible": False, "design": asdict(d), "mission": asdict(m)}
    a, b = max(lo, first - 0.5), first
    while b - a > 0.01:
        mid = 0.5 * (a + b)
        if evaluate(mid, d, m, cat, wind)["feasible"]:
            b = mid
        else:
            a = mid
    r = evaluate(b, d, m, cat, wind)
    r.update({"design": asdict(d), "mission": asdict(m)})
    return r


def cheapest(m: Mission = Mission(), cat: Catalogue = CAT, wind: SeaWind = SEA, camera: str = "eo") -> dict:
    """Both concepts with both envelopes, sized; the cheapest first-year cost (build + ground + a year of helium)."""
    rows = []
    for concept in ("tethered", "free"):
        for env in cat.envelopes:
            for fin in ((1.4, 1.0) if concept == "tethered" else (1.0, 0.7)):
                r = size(MicroDesign(concept=concept, envelope=env, fin_scale=fin, camera=camera), m, cat, wind)
                if r.get("feasible"):
                    r["first_year_cost"] = r["build_cost"] + r["ground_cost"] + r["helium_cost_per_year"]
                rows.append(r)
    ok = [r for r in rows if r.get("feasible")]
    return {"rows": rows, "best": min(ok, key=lambda r: r["first_year_cost"]) if ok else None}


def replace_mission(m: Mission, **kw) -> Mission:
    return replace(m, **kw)
