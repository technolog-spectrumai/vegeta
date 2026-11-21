"""Chuck: the self-driving firefighter quad. Kerchak's quad (``gorilla_quad``, ``gorilla_quad_autonomy``) with nobody
aboard: the seat becomes a water tank, and a pump, a water cannon (a monitor on a pan-tilt mount), a hose reel and a
hydrant suction hose put out the fires the bandits set in the city (Armageddon). This module is the water system's
design - the fires it is sized against, the tank, the pump, the jet, the hoses - and the numbers the game takes
(``game_spec``).

**The fires** (``FIRES``) - none exist in Armageddon or Montezuma yet, so the game's model is set here: a fire grows
as the standard t-squared fire (``Q = alpha t^2`` [kW], NFPA 72 / SFPE growth rates) to its peak; water on its base
takes heat away at ``COOLING_J_KG`` per kg times an application efficiency (a straight stream on the burning
surface: ``EFFICIENCY``), and the fire is out when the water's cooling exceeds its heat release long enough
(``KNOCKDOWN_S``); a fire left alone burns its fuel out. Arriving early matters: a car fire at 2 min is a third of
its peak.

**The pump**: an electric multistage centrifugal on the 72 V bus, ``PUMP['flow_l_min']`` at
``PUMP['pressure_bar']`` - sized so the cannon alone knocks down a bin fire at once and a car fire before its peak,
and the 300 L tank lasts ``tank_minutes``; a hydrant refills the tank through the suction hose in under a minute.

**The jet**: a smooth-bore nozzle; its diameter from Bernoulli at the pump's pressure less the line's losses; its
throw from a jet model (``throw``): the stream leaves at ``v = Cv sqrt(2 dp / rho)`` and flies as a coherent core
that breaks up and slows with distance (an effective slowing length tuned so a 6-7 mm stream at ~15 bar throws the
20-25 m handbooks give for straight streams). The reaction on the nozzle, ``rho Q v``, sizes the mount's servos.

    import chuck
    print(chuck.summary())
    chuck.ChuckDesign().run("out")        # CAD: part="water_system" (tank, pump, monitor, reel, piping), "tank", ...
"""
from __future__ import annotations

import json
import math

import cadquery as cq
import numpy as np

import gorilla_quad as gq
from vegeta.dedalus import Design, Parameter

__all__ = ["FIRES", "TANK", "PUMP", "NOZZLE", "HOSES", "MONITOR", "water_rate_to_suppress", "nozzle", "throw", "hose_loss_bar",
           "sizing", "summary", "game_spec", "ChuckDesign", "chuck_mesh"]

RHO = 1000.0
G = 9.81
COOLING_J_KG = 2.6e6          #: water from 20 °C to steam at 100 °C [J/kg]: 0.34 MJ to heat it, 2.26 MJ to boil it
EFFICIENCY = 0.5              #: the share of a straight stream on the fire's base that does that work (the rest runs off)
KNOCKDOWN_S = 4.0             #: the cooling must beat the heat release this long for the fire to go out

#: the fires the bandits set (the game's model): t-squared growth alpha [kW/s²] to a peak [kW], the fuel [MJ]
FIRES = {
    "bin": {"what": "a street rubbish bin, 240 L", "alpha_kW_s2": 0.047, "peak_kW": 300.0, "fuel_MJ": 120.0, "radius_m": 0.4},
    "pallets": {"what": "a stack of wooden pallets", "alpha_kW_s2": 0.047, "peak_kW": 1500.0, "fuel_MJ": 1200.0, "radius_m": 0.8},
    "car": {"what": "a passenger car", "alpha_kW_s2": 0.012, "peak_kW": 4000.0, "fuel_MJ": 6000.0, "radius_m": 2.2},
}

#: the tank in the seat's place: polyethylene, baffled, low between the axles
TANK = {"volume_l": 200.0, "size_m": (0.62, 0.90, 0.40), "pos": (-0.92, 0.0, 0.74), "shell_kg": 18.0, "baffles": 2}
#: Kerchak drives (the seat is his again): the tank sits over the rear axle in the trunk's place, 200 L - he and 300 L
#: would put 270 kg more on the quad than it was built for
#: the pump: an electric multistage centrifugal, its motor on the 72 V bus
PUMP = {"flow_l_min": 120.0, "pressure_bar": 16.0, "efficiency": 0.62, "motor_eff": 0.92, "mass_kg": 21.0,
        "pos": (0.70, 0.0, 0.40), "priming": "a self-priming first stage (dry suction from a hydrant's well or an open tank)"}
#: the monitor: a water cannon on a pan-tilt mount on the tank's front
MONITOR = {"pos": (1.02, 0.0, 0.84), "pan_deg": 170.0, "tilt_deg": (-15.0, 75.0), "slew_deg_s": 60.0, "mass_kg": 9.0,
           "pattern": "straight stream to 30° fog (an adjustable nozzle)"}
#: the hoses: the cannon's feed (rigid pipe and a short flexible), the reel's hose, the hydrant suction hose
HOSES = {"feed": {"length_m": 1.2, "bore_mm": 32.0}, "reel": {"length_m": 20.0, "bore_mm": 25.0, "kg_m": 0.55, "reel_d_m": 0.52},
         "suction": {"length_m": 3.0, "bore_mm": 50.0, "coupling": "Storz 52-C to a hydrant", "hydrant_flow_l_min": 400.0}}
#: Kerchak's firefighting tools, gorilla-sized, in the right-hand rack (where the bazooka rode) and on the left
TOOLS = {"fire axe": {"length_m": 1.10, "mass_kg": 4.2, "what": "a pick-head axe, a 1.1 m fibreglass helve for his fists"},
         "halligan bar": {"length_m": 1.05, "mass_kg": 6.0, "what": "a forcible-entry bar: adze, pick and fork, 1.05 m"},
         "extinguisher": {"length_m": 0.62, "mass_kg": 15.0, "what": "a 9 kg ABC powder extinguisher for car and electrical fires"},
         "helmet": {"length_m": 0.0, "mass_kg": 2.5, "what": "a firefighter's helmet sized for his head, a visor, a lamp"}}
RACK = {"pos": (-0.08, -0.60, 0.80), "extinguisher_pos": (0.10, 0.62, 0.62)}
KERCHAK = {"mass": gq.RIDER["mass"], "cg": (-0.05, 0.0, 0.99)}
#: Chuck's drive: the same four motors through a longer reduction - with Kerchak and the water the quad weighs ~1040 kg,
#: 43 % over the 729 kg it was sized for; 8.5:1 instead of 6:1 gives back the torque, and costs top speed
DRIVE_RATIO = 8.5
CV = 0.97                     #: a smooth-bore nozzle's velocity coefficient
JET_SLOWING_D = 1500.0        #: the jet's slowing length in nozzle diameters (tuned to 20-25 m straight-stream throws)


def fire_hrr(kind, t):
    """A fire's heat release [kW] ``t`` s after it was set (t-squared growth to its peak)."""
    f = FIRES[kind]
    return min(f["alpha_kW_s2"] * t * t, f["peak_kW"])


def water_rate_to_suppress(hrr_kW, efficiency=EFFICIENCY):
    """The water flow [L/min] on a fire's base that takes away its heat release."""
    return hrr_kW * 1e3 / (COOLING_J_KG * efficiency) / RHO * 1000 * 60


def hose_loss_bar(flow_l_min, length_m, bore_mm, roughness_m=1.5e-6):
    """Darcy-Weisbach friction loss [bar] in a hose (Swamee-Jain friction factor)."""
    q = flow_l_min / 60000.0
    a = math.pi * (bore_mm / 2000.0) ** 2
    v = q / a
    re = v * bore_mm / 1000.0 / 1.0e-6
    f = 0.25 / math.log10(roughness_m / (3.7 * bore_mm / 1000.0) + 5.74 / re ** 0.9) ** 2
    return f * length_m / (bore_mm / 1000.0) * RHO * v * v / 2 / 1e5


def nozzle(pressure_bar=None, flow_l_min=None, line_loss_bar=None):
    """The nozzle: (diameter [mm], jet speed [m/s], reaction [N]) for the pump's flow at its pressure less the feed's loss."""
    p = PUMP["pressure_bar"] if pressure_bar is None else pressure_bar
    qlm = PUMP["flow_l_min"] if flow_l_min is None else flow_l_min
    loss = hose_loss_bar(qlm, HOSES["feed"]["length_m"], HOSES["feed"]["bore_mm"]) + 0.3 if line_loss_bar is None else line_loss_bar
    v = CV * math.sqrt(2 * (p - loss) * 1e5 / RHO)
    q = qlm / 60000.0
    d = math.sqrt(4 * q / (math.pi * v))
    return d * 1000.0, v, RHO * q * v


NOZZLE = {}


def throw(elevation_deg, v0=None, d_mm=None, dt=2e-3):
    """The jet's reach [m] on level ground from the monitor's height, and its path [(x, z)]. The core flies as a
    slug that slows over a length growing with the jet's diameter (``L = JET_SLOWING_D d``) - tuned so the throw is the
    handbooks' 20-25 m for a straight stream of this size at this pressure, not a spray model."""
    d, v, _ = nozzle()
    v0 = v if v0 is None else v0
    d_mm = d if d_mm is None else d_mm
    L = JET_SLOWING_D * d_mm / 1000.0                 # the slowing length [m]: 10 m for a 6.9 mm stream
    a = math.radians(elevation_deg)
    p = np.array([0.0, MONITOR["pos"][2]])
    vel = v0 * np.array([math.cos(a), math.sin(a)])
    path = [p.copy()]
    while p[1] > 0.0 and p[0] < 200.0:
        s = float(np.linalg.norm(vel))
        acc = -s * vel / L - np.array([0.0, G])
        vel = vel + acc * dt
        p = p + vel * dt
        path.append(p.copy())
    return float(p[0]), np.array(path)


def sizing():
    """The water system's numbers, and the quad's with Kerchak driving and the water system and tools aboard."""
    d_mm, v, reaction = nozzle()
    q = PUMP["flow_l_min"] / 60000.0
    hydraulic_kW = q * PUMP["pressure_bar"] * 1e5 / 1e3
    electric_kW = hydraulic_kW / PUMP["efficiency"] / PUMP["motor_eff"]
    tank_min = TANK["volume_l"] / PUMP["flow_l_min"]
    reach = {a: round(throw(a)[0], 1) for a in (10, 20, 30, 40)}
    best = max(range(5, 60), key=lambda a: throw(a)[0])
    suppress = {k: round(water_rate_to_suppress(f["peak_kW"]), 1) for k, f in FIRES.items()}
    # the largest fire the pump puts out, and when each kind outgrows it
    hrr_max = PUMP["flow_l_min"] / 60000.0 * RHO * COOLING_J_KG * EFFICIENCY / 1e3
    outgrows = {k: (round(math.sqrt(hrr_max / f["alpha_kW_s2"])) if hrr_max < f["peak_kW"] else None) for k, f in FIRES.items()}
    water = {k: round(PUMP["flow_l_min"] / 60.0 * KNOCKDOWN_S + 0.0, 1) for k in FIRES}
    reel_loss = hose_loss_bar(PUMP["flow_l_min"] * 0.5, HOSES["reel"]["length_m"], HOSES["reel"]["bore_mm"])
    refill_s = TANK["volume_l"] / HOSES["suction"]["hydrant_flow_l_min"] * 60.0
    masses = {"water": TANK["volume_l"] * RHO / 1000.0, "tank": TANK["shell_kg"], "pump + motor + inverter": PUMP["mass_kg"] + 3.0,
              "monitor": MONITOR["mass_kg"], "reel + 20 m hose": 8.0 + HOSES["reel"]["length_m"] * HOSES["reel"]["kg_m"],
              "suction hose + coupling": 6.0, "piping, valves, foam inductor": 7.0,
              "tools (axe, halligan, extinguisher) + rack": sum(t["mass_kg"] for k, t in TOOLS.items() if k != "helmet") + 3.0}
    load = sum(masses.values())
    cg_z = (masses["water"] * (TANK["pos"][2] + TANK["size_m"][2] * 0.45) + masses["tank"] * (TANK["pos"][2] + TANK["size_m"][2] / 2)
            + masses["pump + motor + inverter"] * PUMP["pos"][2] + masses["monitor"] * MONITOR["pos"][2]
            + (load - masses["water"] - masses["tank"] - masses["pump + motor + inverter"] - masses["monitor"]) * 0.85) / load
    m_k = KERCHAK["mass"] + TOOLS["helmet"]["mass_kg"]
    x_load = (masses["water"] * TANK["pos"][0] + masses["tank"] * TANK["pos"][0] + masses["pump + motor + inverter"] * PUMP["pos"][0]
              + masses["monitor"] * MONITOR["pos"][0] + masses["reel + 20 m hose"] * TANK["pos"][0]) / (load - 13.0 - masses["tools (axe, halligan, extinguisher) + rack"])
    on = m_k + load
    cg = ((m_k * KERCHAK["cg"][0] + load * x_load) / on, 0.0, (m_k * KERCHAK["cg"][2] + load * cg_z) / on)
    quad = gq.sizing(rider_cg=cg, rider_mass=on)
    designed = gq.sizing()
    battery_Wh_per_tank = electric_kW * tank_min / 60 * 1000
    return {"nozzle_mm": round(d_mm, 2), "jet_m_s": round(v, 1), "reaction_N": round(reaction, 1), "hydraulic_kW": round(hydraulic_kW, 2),
            "pump_electric_kW": round(electric_kW, 2), "tank_minutes": round(tank_min, 2), "battery_Wh_per_tank": round(battery_Wh_per_tank),
            "tanks_per_battery_with_driving_reserve": round(0.5 * gq.sizing()["battery_Wh"] / battery_Wh_per_tank, 1),
            "throw_m_by_elevation_deg": reach, "best_elevation_deg": best, "max_throw_m": round(throw(best)[0], 1),
            "water_l_min_to_suppress_peak": suppress, "largest_fire_put_out_kW": round(hrr_max), "fire_outgrows_pump_at_s": outgrows,
            "reel_loss_bar_at_half_flow": round(reel_loss, 2), "feed_loss_bar": round(hose_loss_bar(PUMP["flow_l_min"], HOSES["feed"]["length_m"], HOSES["feed"]["bore_mm"]), 2),
            "hydrant_refill_s": round(refill_s), "load_kg": round(load, 1), "load_cg_z_m": round(cg_z, 3),
            "quad": {k: quad[k] for k in ("mass_total_kg", "cg_total_m", "ssf", "rollover_lat_g", "axle_load_kg")},
            "aboard_kg": round(on, 1), "designed_for_kg": round(designed["mass_rider_kg"], 1),
            "wheel_torque_needed_Nm": quad["wheel_torque_Nm"], "wheel_torque_designed_Nm": designed["wheel_torque_Nm"],
            "drive": _drive(quad, designed),
            "grade_with_designed_motors": round(math.tan(math.asin(min(1.0, designed["wheel_force_N"] * 4 / (quad["mass_total_kg"] * G) - 0.035 - 1.0 / G))) , 3),
            "masses_kg": masses, "monitor_servo_torque_Nm": round(reaction * 0.35 * 1.5, 1)}


def _drive(quad, designed):
    """The motors through ``DRIVE_RATIO``: the wheel torque, the steepest grade (1 m/s² to spare kept), the top speed."""
    r0 = designed["motor"]["ratio"]
    T_motor = designed["motor"]["peak_torque_Nm"]
    T_wheel = T_motor * DRIVE_RATIO * 0.95
    m = quad["mass_total_kg"]
    s = (4 * T_wheel / gq.GEOM["wheel_r"]) / (m * G) - gq.CRR - 1.0 / G
    return {"ratio": DRIVE_RATIO, "wheel_torque_Nm": round(T_wheel), "wheel_torque_needed_Nm": quad["wheel_torque_Nm"],
            "grade": round(math.tan(math.asin(max(-1.0, min(1.0, s)))), 3), "top_speed_kmh": round(gq.V_MAX * 3.6 * r0 / DRIVE_RATIO, 1)}


def summary():
    return json.dumps(sizing(), indent=1)


def game_spec():
    """What Armageddon needs to play Chuck's water system: the fires' model, the jet, the tank and the mount."""
    s = sizing()
    return {"name": "chuck", "source": "vegeta labs/legged/designs/chuck.py",
            "fires": {"model": "t-squared growth to a peak; out when the water's cooling (2.6 MJ/kg x efficiency) beats the heat release for knockdown_s",
                      "cooling_J_kg": COOLING_J_KG, "efficiency": EFFICIENCY, "knockdown_s": KNOCKDOWN_S, "kinds": FIRES},
            "tank_l": TANK["volume_l"], "flow_l_min": PUMP["flow_l_min"], "pressure_bar": PUMP["pressure_bar"],
            "jet": {"speed_m_s": s["jet_m_s"], "nozzle_mm": s["nozzle_mm"], "slowing_length_m": round(JET_SLOWING_D * s["nozzle_mm"] / 1000.0, 2),
                    "max_throw_m": s["max_throw_m"], "best_elevation_deg": s["best_elevation_deg"], "reaction_N": s["reaction_N"]},
            "monitor": MONITOR, "reel_m": HOSES["reel"]["length_m"], "hydrant_refill_s": s["hydrant_refill_s"],
            "pump_kW": s["pump_electric_kW"]}


# ---------------------------------------------------------------- CAD (mm, the quad's frame)
def _box(c, size, fillet=0.0):
    w = cq.Workplane("XY").box(*size).translate(c)
    return w.edges().fillet(fillet) if fillet else w


def _tube(a, b, r):
    return cq.Workplane(obj=gq._tube(a, b, r))


class ChuckDesign(Design):
    """Chuck's water system on the quad (the quad's frame, mm): ``tank``, ``pump``, ``monitor`` (pan base, tilt
    yoke, the nozzle), ``reel`` (the hose drum in the trunk), ``suction`` (the hydrant hose coiled on the left),
    ``piping``, or ``water_system`` (all)."""

    parameters = [Parameter("part", "water_system", choices=("water_system", "tank", "pump", "monitor", "reel", "suction", "piping", "tools", "helmet"),
                            description="what to build")]

    def tank(self, p):
        L, W, H = (v * 1000 for v in TANK["size_m"])
        x, y, z = (v * 1000 for v in TANK["pos"])
        body = _box((x, y, z + H / 2), (L, W, H), fillet=60)
        lid = cq.Workplane("XY").circle(90).extrude(30).translate((x + 180, 300, z + H)).edges(">Z").fillet(10)
        sight = cq.Workplane("XY").box(20, 6, H * 0.7).translate((x + 100, W / 2 + 2, z + H / 2))
        return body.union(lid).union(sight)

    def pump(self, p):
        x, y, z = (v * 1000 for v in PUMP["pos"])
        casing = cq.Workplane("YZ").circle(95).extrude(280).translate((x - 140, y, z))
        motor = cq.Workplane("YZ").circle(80).extrude(220).translate((x - 380, y, z)).edges().fillet(10)
        inlet = cq.Workplane("XY").circle(32).extrude(120).translate((x + 60, y, z - 160))
        outlet = cq.Workplane("XY").circle(24).extrude(160).translate((x - 40, y, z + 80))
        base = cq.Workplane("XY").box(560, 220, 14).translate((x - 120, y, z - 105))
        return casing.union(motor).union(inlet).union(outlet).union(base)

    def monitor(self, p):
        x, y, z = (v * 1000 for v in MONITOR["pos"])
        base = cq.Workplane("XY").circle(80).extrude(90).translate((x, y, z - 90))
        yoke = (cq.Workplane("XY").box(70, 170, 120).translate((x, y, z + 60))
                .cut(cq.Workplane("XY").box(80, 110, 90).translate((x, y, z + 85))))
        barrel = cq.Workplane("YZ").circle(26).extrude(420).translate((x - 60, y, z + 75)).rotate((x, y, z + 75), (x, y + 1, z + 75), -15)
        tip = cq.Workplane("YZ").circle(36).extrude(70).translate((x + 350, y, z + 75)).rotate((x, y, z + 75), (x, y + 1, z + 75), -15)
        return base.union(yoke).union(barrel).union(tip)

    def reel(self, p):
        """The hose reel on top of the tank, its axis across, the nozzle clipped where his right hand finds it."""
        x = TANK["pos"][0] * 1000
        R = HOSES["reel"]["reel_d_m"] / 2 * 1000
        z = (TANK["pos"][2] + TANK["size_m"][2]) * 1000 + R + 20
        drum = cq.Workplane("XZ").circle(R * 0.55).extrude(300, both=True).translate((x, 0, z))
        flanges = [cq.Workplane("XZ").circle(R).extrude(8, both=True).translate((x, s * 310, z)) for s in (-1, 1)]
        hose = cq.Workplane("XZ").circle(R * 0.92).circle(R * 0.56).extrude(290, both=True).translate((x, 0, z))
        stands = [cq.Workplane("XY").box(60, 20, R + 20).translate((x, s * 330, z - (R + 20) / 2)) for s in (-1, 1)]
        nozzle = cq.Workplane("YZ").circle(24).extrude(300).translate((x + 120, -360, z - 60))
        out = drum.union(hose).union(nozzle)
        for f in flanges + stands:
            out = out.union(f)
        return out

    def tools(self, p):
        """His tools: the axe and the Halligan bar lying in the right-hand rack's cradles, the extinguisher in a
        bracket on the left."""
        x0, y0, z0 = (v * 1000 for v in RACK["pos"])
        axe_l, hal_l = TOOLS["fire axe"]["length_m"] * 1000, TOOLS["halligan bar"]["length_m"] * 1000
        helve = cq.Workplane("YZ").circle(18).extrude(axe_l).translate((x0 - axe_l / 2, y0 - 40, z0 + 60))
        head = cq.Workplane("XY").box(60, 30, 300).translate((x0 + axe_l / 2 - 40, y0 - 40, z0 + 60)).edges("|Y").fillet(8)
        bar = cq.Workplane("YZ").circle(16).extrude(hal_l).translate((x0 - hal_l / 2, y0 + 40, z0 + 40))
        adze = cq.Workplane("XY").box(40, 70, 140).translate((x0 + hal_l / 2, y0 + 40, z0 + 70))
        fork = cq.Workplane("XY").box(90, 60, 20).translate((x0 - hal_l / 2 - 30, y0 + 40, z0 + 40))
        cradles = [cq.Workplane("YZ").circle(70).circle(60).extrude(40).translate((x0 + dx, y0, z0 + 50))
                   .cut(cq.Workplane("XY").box(60, 200, 100).translate((x0 + dx + 20, y0, z0 + 110))) for dx in (-380, 330)]
        ex, ey, ez = (v * 1000 for v in RACK["extinguisher_pos"])
        bottle = cq.Workplane("XY").circle(80).extrude(520).translate((ex, ey, ez)).edges(">Z").fillet(40)
        valve = cq.Workplane("XY").box(60, 40, 90).translate((ex, ey, ez + 560))
        bracket = cq.Workplane("XY").circle(88).circle(82).extrude(60).translate((ex, ey, ez + 200))
        out = helve.union(head).union(bar).union(adze).union(fork).union(bottle).union(valve).union(bracket)
        for c in cradles:
            out = out.union(c)
        return out

    def suction(self, p):
        """The 3 m suction hose coiled flat on the left (where the flak rode), its Storz coupling on top."""
        x, y, z = -100.0, 640.0, 760.0
        coil = cq.Workplane("XZ").circle(260).circle(170).extrude(60, both=True).translate((x, y, z))
        coupling = cq.Workplane("XZ").circle(45).extrude(40, both=True).translate((x, y + 70, z + 230))
        return coil.union(coupling)

    def piping(self, p):
        px, py, pz = (v * 1000 for v in PUMP["pos"])
        mx, my, mz = (v * 1000 for v in MONITOR["pos"])
        tx, ty, tz = (v * 1000 for v in TANK["pos"])
        rx, rz = tx, (TANK["pos"][2] + TANK["size_m"][2]) * 1000 + 40
        pts = [_tube((px, py, pz + 100), (mx - 60, my, pz + 100), 16), _tube((mx - 60, my, pz + 100), (mx, my, mz - 90), 16),     # the pump to the cannon
               _tube((tx + 310, -200, tz + 40), (0.2 * 1000, -200, 330), 26), _tube((0.2 * 1000, -200, 330), (px - 380, -200, pz - 60), 26),
               _tube((px - 380, -200, pz - 60), (px + 60, py, pz - 160), 26),                                               # the tank to the pump, under the frame
               _tube((px - 40, 120, pz + 120), (rx + 200, 120, rz), 14)]                                                    # the pump to the reel
        out = pts[0]
        for s_ in pts[1:]:
            out = out.union(s_)
        return out

    def helmet(self, p):
        """His helmet, in the head's frame (mm): a brimmed shell with a comb, a visor and a lamp."""
        shell = cq.Workplane("XY").sphere(150).cut(cq.Workplane("XY").box(400, 400, 200).translate((0, 0, -100)))
        brim = cq.Workplane("XY").ellipse(200, 170).extrude(12).translate((-20, 0, 0))
        comb = cq.Workplane("XY").box(250, 24, 40).translate((0, 0, 150))
        lamp = cq.Workplane("YZ").circle(22).extrude(40).translate((140, 0, 60))
        return shell.union(brim).union(comb).union(lamp)

    def build(self, p):
        if p["part"] != "water_system":
            return getattr(self, p["part"])(p)
        parts = [self.tank(p), self.pump(p), self.monitor(p), self.reel(p), self.suction(p), self.piping(p), self.tools(p)]
        return cq.Compound.makeCompound([w.val() for w in parts])


COLORS = {"tank": (0.80, 0.10, 0.10, 1.0), "pump": (0.25, 0.27, 0.30, 1.0), "monitor": (0.85, 0.75, 0.15, 1.0),
          "reel": (0.85, 0.15, 0.12, 1.0), "suction": (0.20, 0.25, 0.70, 1.0), "piping": (0.70, 0.70, 0.72, 1.0),
          "tools": (0.85, 0.12, 0.10, 1.0)}
HELMET_RGBA = (0.95, 0.80, 0.10, 1.0)


def chuck_mesh():
    """The water system's pieces for ``kerchak_render`` on the quad's sprung body."""
    d = ChuckDesign()
    p = d.resolve()
    return [("quad_chassis", gq._tess(getattr(d, n)(p)), c) for n, c in COLORS.items()]


if __name__ == "__main__":
    print(summary())
