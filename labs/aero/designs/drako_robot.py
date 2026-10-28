"""Drako (``drako.Drako``) as a Chiron robot, for the Yax export (``tools/export.py drako``, ``exports/drako.yx2``).

The root link, ``drako``: the two hexagonal plates, six arm tubes, the landing gear (the two skids are the feet),
the cargo cradle, and, as visual geoms that weigh, the motors and propeller discs. The battery, the avionics and the
latch servos are point masses. Four latch links hang on hinges from the cradle, each driven by a
self-locking worm servo (``LATCH_ACTUATOR``): closed (q = 0) they hold the tote by its lip, opened they drop it. The masses are ``Drako.MASSES_KG`` and ``Drako.battery_mass``, split over the parts
by the share stated here; the package is not part of the machine (a payload, set by the scenario).

    import drako_robot as dk
    robot = dk.drako()                    # ~22.4 kg empty
"""
from __future__ import annotations

import math

from ogma import actuators as act
from vegeta.chiron import FootSpec, Geom, Joint, Link, PointMass, Robot, Servo

from drako import Drako

FRAME_SPLIT = {"plates": 0.30, "arms": 0.35, "gear": 0.25, "cradle": 0.10}      # of the frame mass (assumed)
LATCH_ACTUATOR = "standard 60 g worm"                           # self-locking: a latch holds the package unpowered
LATCH_OPEN_RAD = 1.2
LATCH_GAINS = (8.0, 0.2, 0.001)                                 # kp [N m/rad], kd, armature (assumed)
LATCHES = ("latch_FL", "latch_FR", "latch_RL", "latch_RR")
CARBON, DARK, RED = (0.12, 0.12, 0.13, 1.0), (0.05, 0.05, 0.05, 1.0), (0.70, 0.15, 0.15, 1.0)


def design_params(overrides: dict | None = None) -> dict:
    p = {q.name: q.default for q in Drako.parameters}
    p.update(overrides or {})
    return p


def mass_budget(p: dict | None = None) -> dict:
    """Every part [kg]: the design's catalogue masses and the battery."""
    p = design_params(p)
    return {**Drako.MASSES_KG, "battery (12S semi-solid)": Drako.battery_mass(p)}


def rotor_layout(p: dict | None = None):
    """Rotor centres [m] (flat six, rotor 0 forward) and spin senses (alternating, +1 counter-clockwise from above)."""
    p = design_params(p)
    R = p["wheelbase"] / 2000
    z = (p["plate_thickness"] + p["plate_gap"] / 2 + p["arm_diameter"] / 2 + p["mount_thickness"] + 55) / 1000
    return [(R * math.cos(math.radians(60 * k)), R * math.sin(math.radians(60 * k)), z) for k in range(6)], \
           [1 if k % 2 == 0 else -1 for k in range(6)]


#: The bought drive: a T-Motor U12 II KV120 on 12S with a T-Motor 30x10.5 carbon propeller. What the listing gives
#: (``DATASHEET``) is a retailer's, not T-Motor's own test table (store.tmotor.com was not reachable): 20.4 kgf at full
#: throttle on 48 V with the 30x10.5. The propeller's planform is generic; its chord is scaled (``CHORD_FIT``) so that
#: Boreas (BEMT + motor model) meets that point. The motor's R and I0 are assumed; fit them to the official table.
DATASHEET = {"motor": "T-Motor U12 II KV120", "propeller": "T-Motor 30x10.5 carbon (G30*10.5 / HEP-S class)",
             "max_thrust_kgf": 20.4, "voltage_V": 48.0, "throttle": 1.0,
             "source": "retailer listing (robotshop.com, U12 II KV120: 30x10.5, 48 V, max thrust 20.4 kg), seen in a web "
                       "search 2026-10-28; NOT the manufacturer's test table: verify"}
CHORD_FIT = 0.896                                               # chord scale on the generic planform, fitted to DATASHEET
ESC_MAX_A = 100.0                                               # 12S 100 A FOC ESC class (full throttle draws ~81 A)
SPIN_UP_S = 0.12                                                # 30-inch rotor, 0 -> full thrust time constant (assumed)
PROP_KG, ROTOR_KG = 0.15, 0.25                                  # propeller; propeller + motor bell (spinning)


def boreas_system(p: dict | None = None, *, chord_scale: float = CHORD_FIT) -> tuple:
    from vegeta import boreas
    p = design_params(p)
    d, pitch = boreas.inches(p["prop_diameter"], 10.5)
    k = chord_scale
    prop = boreas.Propeller.from_pitch("T-Motor 30x10.5 carbon", d, pitch, blades=2, chord_root_m=0.045 * k, chord_max_m=0.065 * k,
                                       chord_tip_m=0.025 * k, mass_kg=PROP_KG, rotor_mass_kg=ROTOR_KG,
                                       notes=f"generic planform, chord x {k:.3f} fitted to {DATASHEET['max_thrust_kgf']} kgf at "
                                             f"{DATASHEET['voltage_V']:g} V full throttle ({DATASHEET['source']})")
    airfoil = boreas.Airfoil(name="carbon prop section", cl_alpha=2 * math.pi * 0.9, alpha0_deg=-4.0, cl_max=1.3, cd0=0.012, k=0.03,
                             source="assumed for a moulded carbon blade at Re ~ 4e5")
    motor = boreas.Motor("T-Motor U12 II KV120", kv_rpm_per_volt=120, resistance_ohm=0.045, no_load_current_a=1.6,
                         max_current_a=ESC_MAX_A, mass_kg=0.80,
                         source="KV from the listing; R 45 mOhm, I0 1.6 A, 0.8 kg assumed (class values); current limit: the ESC")
    battery = boreas.Battery("12S 64 Ah semi-solid", cells=p["battery_cells"], capacity_ah=p["battery_capacity"],
                             usable_fraction=0.85, mass_kg=Drako.battery_mass(p))
    return prop, airfoil, motor, battery


def system(p: dict | None = None, *, cells: int | None = None):
    """One rotor as a Boreas ``Propulsion`` (propeller, motor, pack); ``cells`` replaces the pack's series count."""
    from vegeta import boreas
    from vegeta.boreas.system import Propulsion
    prop, airfoil, motor, battery = boreas_system(p)
    if cells:
        battery = boreas.Battery(f"{cells}S", cells=cells, capacity_ah=battery.capacity_ah, usable_fraction=battery.usable_fraction)
    return Propulsion(prop, airfoil, motor, battery)


def datasheet_check() -> dict:
    """The fitted model at the listing's point (13S nominal = 48.1 V stands in for 48 V): thrust against 20.4 kgf."""
    q = system(cells=13).at_throttle(1.0)
    return {"thrust_kgf": q.thrust / 9.81, "rpm": q.rpm, "current_A": q.current, "listed_kgf": DATASHEET["max_thrust_kgf"]}


def full_throttle(p: dict | None = None) -> dict:
    """The rotor at full throttle on the pack (Boreas: BEMT + motor): rpm, thrust [N], torque [N m], shaft and
    electrical power [W], current [A], kM = Q / T [m]."""
    q = system(p).at_throttle(1.0)
    a = q.aero
    return {"rpm": q.rpm, "thrust_N": a.thrust, "torque_Nm": a.torque, "power_W": a.power, "electrical_W": q.electrical_power,
            "current_A": q.current, "kM": a.torque / a.thrust, "w_max": q.rpm * 2 * math.pi / 60,
            "figure_of_merit": a.figure_of_merit}


def latch_servo() -> Servo:
    a = act.get(LATCH_ACTUATOR)
    kp, kd, arm = LATCH_GAINS
    return Servo(a.stall_Nm, a.rated_Nm, a.no_load_rpm * 2 * math.pi / 60, a.stall_A, a.voltage_V, kp, kd, arm,
                 source=f"{a.key}: {a.source}")


def drako(overrides: dict | None = None, *, name: str = "Drako") -> Robot:
    p = design_params(overrides)
    mm = 1e-3
    m = mass_budget(p)
    frame = m["frame (plates, folding arms, gear, cradle)"]
    t, gap = p["plate_thickness"] * mm, p["plate_gap"] * mm
    rh, R = p["plate_diameter"] / 2 * mm, p["wheelbase"] / 2 * mm
    arm_z, ra = t + gap / 2, p["arm_diameter"] / 2 * mm
    geoms = [Geom(f"plate_{k}", "cylinder", (rh * math.cos(math.pi / 6), t / 2), pos=(0, 0, z), mass=frame * FRAME_SPLIT["plates"] / 2,
                  role="body", rgba=CARBON) for k, z in (("lower", t / 2), ("upper", 1.5 * t + gap))]
    pos, _ = rotor_layout(p)
    motor = next(v for k, v in m.items() if k.startswith("6 x motor")) / 6
    prop = next(v for k, v in m.items() if k.startswith("6 x T-Motor 30x10.5")) / 6
    rp = p["prop_diameter"] * 25.4 / 2 * mm
    for k in range(6):
        c, s = math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))
        geoms.append(Geom(f"arm{k}", "capsule", (ra,), fromto=(0.55 * rh * c, 0.55 * rh * s, arm_z, R * c, R * s, arm_z),
                          mass=frame * FRAME_SPLIT["arms"] / 6, role="body", rgba=CARBON))
        mz = arm_z + ra + p["mount_thickness"] * mm
        geoms.append(Geom(f"motor{k}", "cylinder", (0.050, 0.0225), pos=(R * c, R * s, mz + 0.0225), mass=motor,
                          role="visual", rgba=RED))
        geoms.append(Geom(f"prop{k}", "cylinder", (rp, 0.001), pos=pos[k], mass=prop, role="visual",
                          rgba=(0.55, 0.75, 0.95, 0.30)))
    # landing gear: skids (the feet) and four legs
    h, sy, L, rl = p["leg_height"] * mm, p["leg_spread"] / 2 * mm, p["skid_length"] * mm, p["leg_diameter"] / 2 * mm
    m_gear = frame * FRAME_SPLIT["gear"]
    feet = []
    for side, sg in (("L", 1), ("R", -1)):
        z = -h + rl
        geoms.append(Geom(f"skid_{side}", "capsule", (rl,), fromto=(-L / 2 + rl, sg * sy, z, L / 2 - rl, sg * sy, z),
                          mass=m_gear * 0.25, role="foot", friction=(0.8, 0.005, 0.0001), rgba=DARK))
        for sx in (0.35 * L, -0.35 * L):
            geoms.append(Geom(f"leg_{side}{'f' if sx > 0 else 'r'}", "capsule", (rl * 0.8,),
                              fromto=(sx * 0.45, sg * rh * 0.6, 0.0, sx, sg * sy, z), mass=m_gear * 0.125,
                              role="visual", rgba=DARK))
        feet.append(FootSpec(f"skid_{side}", f"skid_{side}", [], "drako"))
    # cargo cradle: the frame as four bars
    X, Y, b = p["cradle_x"] * mm, p["cradle_y"] * mm, p["cradle_bar"] * mm
    zc = -h + (p["package_z"] + p["cradle_bar"]) * mm - b / 2
    mc = frame * FRAME_SPLIT["cradle"] / 4
    geoms += [Geom("cradle_front", "box", (b / 2, Y / 2 + b, b / 2), pos=(X / 2 + b / 2, 0, zc), mass=mc, role="link", rgba=CARBON),
              Geom("cradle_rear", "box", (b / 2, Y / 2 + b, b / 2), pos=(-X / 2 - b / 2, 0, zc), mass=mc, role="link", rgba=CARBON),
              Geom("cradle_left", "box", (X / 2, b / 2, b / 2), pos=(0, Y / 2 + b / 2, zc), mass=mc, role="link", rgba=CARBON),
              Geom("cradle_right", "box", (X / 2, b / 2, b / 2), pos=(0, -Y / 2 - b / 2, zc), mass=mc, role="link", rgba=CARBON)]
    top = 2 * t + gap
    masses = [PointMass("battery", m["battery (12S semi-solid)"], (0, 0, top + 0.055))]
    masses += [PointMass(k.split(",")[0].split(" (")[0].replace(" ", "_"), v, (0, 0, top + 0.01))
               for k, v in m.items() if k.startswith("FC")]
    # the four hook latches: a hinge along x on the long cradle bars; q = 0 closed under the tote's lip, + opens
    servo_kg = act.get(LATCH_ACTUATOR).mass_g / 1000
    hook_kg = max(next(v for k, v in m.items() if k.startswith("latches")) / 4 - servo_kg, 0.01)
    latches = []
    for name_, sx, sy in zip(LATCHES, (1, 1, -1, -1), (1, -1, 1, -1)):
        x, y = sx * 0.25 * X, sy * (Y / 2 - 0.003)
        masses.append(PointMass(f"{name_}_servo", servo_kg, (x, sy * (Y / 2 + b + 0.010), zc)))
        latches.append(Link(name_, joints=[Joint(f"{name_}_hinge", "hinge", axis=(-sy, 0.0, 0.0), pos=(x, y, zc - b / 2),
                                                 range=(-0.05, LATCH_OPEN_RAD + 0.05), tag="latch", servo=latch_servo())],
                            geoms=[Geom(f"{name_}_arm", "box", (0.015, 0.003, 0.020), pos=(x, y, zc - b / 2 - 0.020),
                                        mass=hook_kg * 0.7, role="link", rgba=RED),
                                   Geom(f"{name_}_tip", "box", (0.015, 0.006, 0.003), pos=(x, y - sy * 0.006, zc - b / 2 - 0.037),
                                        mass=hook_kg * 0.3, role="link", rgba=RED)]))
    robot = Robot(name, Link("drako", log=True, geoms=geoms, masses=masses, children=latches), feet=feet,
                  nominal_qpos={f"{n}_hinge": 0.0 for n in LATCHES},
                  notes="Drako: heavy-lift cargo hexacopter, 1650 mm, 6 x 30-inch, 12S 60 Ah; payload up to 15 kg (not a link)",
                  sources={"geometry": "designs/drako.py (Drako parameters)", "masses": "Drako.MASSES_KG, Drako.battery_mass",
                           "split": "drako_robot.FRAME_SPLIT (assumed)",
                           "latch servo": f"ogma.actuators: {LATCH_ACTUATOR}", "looks": "designs/drako.py:Drako"})
    robot.validate()
    robot.params = p
    return robot
