"""How to drive a spider robot's joints — electric or fluid — sized for the three spiders of ``docs/spiders.md``:
ARACHNE-2 (palm-sized wall climber), TARANTULA (a bigger walker and climber) and SALTY (a jumping spider). Pure
Python: the numbers behind the recommendation, each part's mass from a datasheet or a stated assumption.

**The question.** Real spiders extend two leg joints by blood pressure (4–8 kPa walking, 60–130 kPa in jumps and
runs) and flex every joint with muscles. Is a fluid-pressure joint worth it for each robot — and can it be light?

**Architectures** (per actuated joint, plus what the whole robot shares):

* ``servo`` — an electric servo at each joint (today's ARACHNE; ROBOTIS XL430 class at TARANTULA's size);
* ``tendon`` — motors in the body pull tendons or pushrods; springs extend (ARACHNE-2's choice); high-ratio gearing
  holds a pose without current;
* ``spider`` — the spider's own split: one shared low-pressure circuit extends every joint through a bellows (no
  valves: one pump, one accumulator, a regulator), electric flexor tendons control each joint against it;
* ``valved`` — full hydraulics: a pump and accumulator at system pressure, a cylinder and two on/off valves per
  joint, a joint-angle sensor;
* ``hydrostatic`` — a motor-driven master cylinder in the body per joint, a rolling-diaphragm slave at the joint
  (Disney Research style): no valves, fluid as the tendon.

For each robot and architecture: the actuation mass (and its share of the robot), the power to hold a stance, the
power to walk, the number of parts and of fluid seals — the measures that decide a robot built for a job.
"""
from __future__ import annotations

import math

__all__ = ["ROBOTS", "PARTS", "ARCHITECTURES", "ROUTING", "CUT_LOADS", "joint_demand", "size", "compare",
           "tendon_routing", "cutter", "tarantula_winch", "jump", "salty_catapult",
           "SOURCES", "G", "extension_support", "recommend", "EFFICIENCY", "SERVOS", "MOTORS",
           "pick_motor"]

G = 9.81

#: the robots: mass with payload [kg], leg reach hip→foot [m], actuated joints, legs in stance (worst case),
#: surfaces it climbs (adhesion loads joints in flexion, ground walking loads them in extension)
ROBOTS = {
    "ARACHNE-2": {"mass": 0.18, "reach": 0.075, "joints": 24, "stance": 6, "climbs": True,
                  "role": "palm-sized wall climber: glass, acrylic, varnished wood; bark kit"},
    "TARANTULA": {"mass": 1.5, "reach": 0.25, "joints": 24, "stance": 6, "climbs": True,
                  "role": "bigger walker and climber: rough ground, slopes, trees and walls; 0.3 kg payload"},
    "SALTY": {"mass": 0.25, "reach": 0.07, "joints": 20, "stance": 6, "climbs": False,
              "role": "jumping spider: walks, jumps ~1 m onto targets, sees with big front cameras"},
}

#: parts: mass [g], capability, source ("datasheet" values cited in SOURCES; "assumed" = a class estimate to be
#: replaced by a sourced part)
PARTS = {
    "micro servo 4 g": {"g": 4.1, "stall_nm": 0.062, "cont_nm": 0.025, "hold_a_per_nm": 0.35 / 0.062, "volt": 3.8,
                        "src": "EMAX ES9051 class at 3.8 V (arachne_specs)"},
    "XL430-W250": {"g": 57.2, "stall_nm": 1.4, "cont_nm": 0.45, "hold_a_per_nm": 1.3 / 1.4, "volt": 11.1,
                   "src": "ROBOTIS e-Manual: 57.2 g, 1.4 N·m stall at 11.1 V, 1.3 A stall; continuous: assumed 1/3"},
    "gearmotor 1000:1 (Pololu HPCB 6V)": {"g": 10.5, "stall_nm": 1.08, "cont_nm": 0.3, "hold_a_per_nm": 0.0,
                                          "volt": 6.0, "src": "Pololu #3070: 10.5 g, 11 kg·cm stall at 6 V, 1.5 A; "
                                          "holds by gear friction at ~986:1 (assumed — test it; else a worm stage)"},
    "gearmotor 6 mm + worm (micro)": {"g": 2.5, "stall_nm": 0.07, "cont_nm": 0.025, "hold_a_per_nm": 0.0, "volt": 3.7,
                                      "src": "assumed class: 6 mm planetary micro gearmotor + worm stage, "
                                             "self-locking; to be sourced"},
    "magnetic joint encoder": {"g": 0.4, "src": "assumed: diametral magnet + AS5600-class sensor on a flex"},
    "tendon, pulley, spring": {"g": 1.0, "src": "assumed: Dyneema line, printed pulley, steel spring (small robot)"},
    "tendon, pulley, spring (large)": {"g": 4.0, "src": "assumed: TARANTULA scale"},
    "TPU bellows extensor": {"g": 0.3, "kpa_max": 150.0, "src": "printed TPU sealed with latex (Göttler/Spröwitz, "
                                                                "ETH/MPI: water extends, cables flex, 125° travel)"},
    "TPU bellows extensor (large)": {"g": 2.5, "kpa_max": 300.0, "src": "assumed: TARANTULA-scale bellows"},
    "piezo micropump (Bartels mp6)": {"g": 2.0, "kpa_max": 60.0, "ml_min": 8.0,
                                      "src": "Bartels datasheet: 2 g, water ~500–600 mbar typical, 6–12 ml/min"},
    "micro plunger pump + motor": {"g": 18.0, "kpa_max": 2000.0, "ml_min": 60.0,
                                   "src": "assumed: 4 mm plunger on a micro gearmotor + check valves"},
    "small gear pump + BLDC": {"g": 180.0, "kpa_max": 3000.0, "ml_min": 400.0,
                               "src": "assumed: RC-hydraulics class (1:14 machines run 25–40 bar, ~400 ml/min)"},
    "accumulator (bladder, small)": {"g": 3.0, "src": "assumed: latex bladder in a printed shell, low pressure"},
    "accumulator (gas, 5 cm³, 20 bar)": {"g": 12.0, "src": "assumed: aluminium/carbon shell, nitrogen pre-charge"},
    "accumulator (gas, 20 cm³, 30 bar)": {"g": 45.0, "src": "assumed"},
    "pressure sensor + regulator": {"g": 2.0, "src": "assumed: MEMS gauge + relief valve"},
    "on/off valve (Lee HDI class)": {"g": 2.5, "src": "Lee Company: HDI 2-port valve under 2.5 g"},
    "hydraulic cylinder 6 mm bore": {"g": 9.0, "src": "assumed: aluminium body, 30 mm stroke"},
    "rolling-diaphragm pair (master+slave)": {"g": 14.0, "src": "assumed: TARANTULA scale, printed + moulded"},
    "rolling-diaphragm pair (micro)": {"g": 2.5, "src": "assumed: ARACHNE scale"},
    "fluid line per joint": {"g": 0.6, "src": "assumed: 30 cm of 1 mm PTFE tube with water-glycol"},
}

SOURCES = {
    "spider pressure": "Göttler et al., Soft Matter 2021 (and refs.): 4–8 kPa walking; 60 and up to 130 kPa reported "
                       "in jumps and runs; femur–patella and tibia–metatarsus joints have no extensor muscles",
    "spider-like joint": "Spröwitz, Göttler et al., ICRA 2017; Göttler, ETH Diss. 27370: TPU joint sealed with latex, "
                        "water extends, wires flex, 125° working angle",
    "hydrostatic transmission": "Whitney et al., Disney Research, IROS 2014 / ICRA 2016: rolling-diaphragm "
                                "transmissions; N hydraulic lines + one shared pneumatic preload line",
    "micropump": "Bartels Mikrotechnik mp6: 2 g, 30 × 15 × 3.8 mm",
    "valves": "The Lee Company HDI solenoid valves: under 2.5 g",
    "gearmotor": "Pololu #3070 1000:1 HPCB 6V: 10.5 g, 11 kg·cm stall",
    "XL430": "ROBOTIS e-Manual XL430-W250: 57.2 g, 1.4 N·m at 11.1 V",
    "Salto": "Haldane et al., Science Robotics 2016: 100 g, 1 m vertical jumps, series-elastic power modulation",
    "peacock spider": "Macquarie University 2025: the third leg pair powers the jump",
}


def joint_demand(robot):
    """Per actuated joint: holding torque [N·m] (one stance leg's share of the weight at the full reach), peak
    (×2: climbing transitions, a slipping neighbour), swing speed [rad/s]: 60° in a swing time that grows with the
    square root of the leg (dynamic similarity, Froude): 0.3 s at ARACHNE's 75 mm, 0.55 s at TARANTULA's 250 mm."""
    r = ROBOTS[robot]
    hold = r["mass"] * G * r["reach"] / r["stance"]
    swing_s = 0.3 * math.sqrt(r["reach"] / 0.075)
    return {"hold_nm": hold, "peak_nm": 2 * hold, "speed_rad_s": math.radians(60) / swing_s, "swing_s": swing_s}


def _p(name):
    return PARTS[name]


#: drive efficiency, electrical → joint work (assumed: servos ~35 %, high-ratio spur gearmotors ~25 %, a hydrostatic
#: pair ~90 % of its motor's)
EFFICIENCY = {"servo": 0.35, "tendon": 0.25, "spider": 0.25, "hydrostatic": 0.22}

#: joint servos, lightest first: (name, g, stall N·m, continuous N·m, A per N·m held, V, no-load rad/s)
SERVOS = [("micro servo 4 g", 4.1, 0.062, 0.025, 0.35 / 0.062, 3.8, 9.2),
          ("XL330-M288", 18.0, 0.52, 0.10, 1.47 / 0.52, 5.0, 10.9),
          ("XL430-W250", 57.2, 1.4, 0.45, 1.3 / 1.4, 11.1, 6.2),
          ("smart servo 25 Nm (ogma)", 250.0, 25.0, 8.0, 6.0 / 25.0, 24.0, 4.2)]
#: body motors for tendons / pushrods / master cylinders, lightest first: (name, g, stall N·m, no-load rad/s, source)
#: — they are geared further by the pulley ratio r (0.5–4); they hold without current (non-backdrivable)
MOTORS = [("6 mm planetary + worm (class)", 2.5, 0.07, 6.0, "assumed class"),
          ("micro metal gearmotor 250:1 (class)", 9.5, 0.29, 13.0, "assumed: scaled within Pololu's HPCB 6V family"),
          ("micro metal gearmotor 1000:1 (Pololu #3070)", 10.5, 1.08, 3.35, "datasheet: 11 kg·cm, 32 rpm, 10.5 g"),
          ("2 × 1000:1 on one spool", 21.0, 2.16, 3.35, "two #3070 in parallel"),
          ("25 mm metal gearmotor 499:1 (class)", 100.0, 4.5, 2.0, "assumed class")]


def pick_motor(peak_nm, speed_rad_s, ratios=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0)):
    """The lightest body motor and pulley ratio that give ``peak_nm`` at the joint and swing at ``speed_rad_s``; None
    when nothing in ``MOTORS`` does."""
    for name, g, stall, w0, src in MOTORS:
        for r in ratios:
            if stall * r >= peak_nm and w0 / r >= speed_rad_s:
                return {"name": name, "g": g, "ratio": r, "stall_nm": stall * r, "src": src}
    return None


def _support(robot, hold):
    """The spider circuit: bellows Ø 8 % of the reach on a lever 4 % of it; the pressure it needs to carry the whole
    hold, capped by a printed TPU bellows (300 kPa); the pump that reaches it (piezo ≤ 60 kPa, plunger ≤ 2 MPa)."""
    reach = ROBOTS[robot]["reach"]
    A, lever = math.pi * (0.04 * reach) ** 2, 0.04 * reach
    need = hold / (A * lever)
    p = min(need, 300e3)
    pump = "piezo micropump (Bartels mp6)" if p <= 60e3 else "micro plunger pump + motor"
    return {"pressure_kpa": p / 1e3, "torque_nm": p * A * lever, "fraction_of_hold": p * A * lever / hold, "pump": pump,
            "bellows_g": 0.2 + 0.02 * (0.08 * reach * 1e3) ** 2 * 0.25}


def extension_support(robot):
    """The spider architecture's shared pressure: the extension torque one bellows gives [N·m] and its fraction of
    the holding torque (``_support``)."""
    return _support(robot, joint_demand(robot)["hold_nm"])


def size(robot, arch):
    """{"mass_g", "hold_w", "walk_w", "parts", "seals", "ok", "note"} of ``arch`` on ``robot``: each architecture picks
    the lightest parts that meet the joint's torque and speed (``ok`` False when nothing does). Walking power: half
    the joints working at the holding torque at a mean 0.25 rad/s, over the drive's efficiency (valved: throttling —
    every displaced volume costs the full system pressure), plus any holding current."""
    r = ROBOTS[robot]
    d = joint_demand(robot)
    n = r["joints"]
    work_w = 0.5 * n * d["hold_nm"] * 0.25
    enc = _p("magnetic joint encoder")["g"]
    tendon_g = 0.6 + 0.004 * d["peak_nm"] * 1000                         # line, pulley, spring: grows with the load
    if arch == "servo":
        for name, g, stall, cont, apn, volt, w0 in SERVOS:
            if cont >= d["hold_nm"] and stall >= d["peak_nm"] and w0 >= d["speed_rad_s"]:
                hold_w = n * 0.5 * d["hold_nm"] * apn * volt
                return dict(mass_g=n * g, hold_w=hold_w, walk_w=hold_w + work_w / EFFICIENCY["servo"], parts=n,
                            seals=0, ok=True, note=f"{name} at every joint; holding costs current")
        return dict(mass_g=n * SERVOS[-1][1], hold_w=float("nan"), walk_w=float("nan"), parts=n, seals=0, ok=False,
                    note="no servo in the catalogue holds this joint continuously")
    if arch in ("tendon", "hydrostatic"):
        m = pick_motor(d["peak_nm"], d["speed_rad_s"])
        if m is None:
            return dict(mass_g=float("nan"), hold_w=0.0, walk_w=float("nan"), parts=0, seals=0, ok=False,
                        note="no body motor in the catalogue drives this joint")
        if arch == "tendon":
            return dict(mass_g=n * (m["g"] + tendon_g + enc), hold_w=0.0, walk_w=work_w / EFFICIENCY["tendon"],
                        parts=3 * n, seals=0, ok=True, note=f"{m['name']} ×{m['ratio']:g} pulleys; tendons, springs")
        force = d["peak_nm"] / (0.04 * r["reach"])                       # the diaphragm's force on a 4 %-reach lever
        pair = 1.5 + 0.08 * force
        return dict(mass_g=n * (m["g"] + pair + _p("fluid line per joint")["g"] + enc), hold_w=0.0,
                    walk_w=work_w / EFFICIENCY["hydrostatic"], parts=4 * n, seals=2 * n, ok=True,
                    note=f"{m['name']} on a master cylinder; a diaphragm slave at the joint")
    if arch == "spider":
        sup = _support(robot, d["hold_nm"])
        s = min(sup["fraction_of_hold"], 1.0)
        # climbers: the flexors carry the wall's pull in full; walkers: the pressure carries the weight
        flex_peak = d["peak_nm"] if r["climbs"] else max(0.3 * d["peak_nm"], d["peak_nm"] - s * d["hold_nm"])
        m = pick_motor(flex_peak, d["speed_rad_s"])
        if m is None:
            return dict(mass_g=float("nan"), hold_w=0.0, walk_w=float("nan"), parts=0, seals=0, ok=False,
                        note="no flexor motor drives this joint")
        pump = _p(sup["pump"])
        acc = _p("accumulator (bladder, small)" if sup["pressure_kpa"] <= 60 else "accumulator (gas, 5 cm³, 20 bar)")
        per_joint = sup["bellows_g"] + _p("fluid line per joint")["g"] + m["g"] + 0.7 * tendon_g + enc
        shared = pump["g"] + acc["g"] + _p("pressure sensor + regulator")["g"] + 0.01 * n * r["reach"] * 1e3
        walk = work_w * (1 - 0.5 * s) / EFFICIENCY["spider"] + (0.6 if sup["pressure_kpa"] > 60 else 0.05)
        return dict(mass_g=n * per_joint + shared, hold_w=0.0, walk_w=walk, parts=4 * n + 4, seals=n + 3, ok=True,
                    note=f"shared extension pressure {sup['pressure_kpa']:.0f} kPa carries {s * 100:.0f} % of a "
                         f"joint's hold; {m['name']} flexors; adjustable leg stiffness")
    if arch == "valved":
        p_sys = 2.0e6 if r["mass"] < 1.0 else 3.0e6
        lever = 0.04 * r["reach"]
        force = d["peak_nm"] / lever
        area = force / p_sys
        cyl = 1.5 + 0.06 * force                                          # aluminium cylinder: g per N of capacity
        pump = _p("micro plunger pump + motor") if r["mass"] < 1.0 else _p("small gear pump + BLDC")
        acc = _p("accumulator (gas, 5 cm³, 20 bar)") if r["mass"] < 1.0 else _p("accumulator (gas, 20 cm³, 30 bar)")
        per_joint = cyl + 2 * _p("on/off valve (Lee HDI class)")["g"] + _p("fluid line per joint")["g"] + enc
        shared = pump["g"] + acc["g"] + _p("pressure sensor + regulator")["g"] + 0.02 * n * r["reach"] * 1e3
        stroke = lever * 1.0                                              # ~1 rad of joint travel
        v_cycle = n * 0.5 * 2 * area * stroke                              # m³ per 2 s gait cycle
        walk_w = p_sys * v_cycle / 2.0 / 0.5                              # 50 % pump efficiency
        return dict(mass_g=n * per_joint + shared, hold_w=0.0, walk_w=walk_w, parts=5 * n + 6, seals=4 * n + 6,
                    ok=True, note=f"{p_sys / 1e5:.0f} bar, cylinders for {force:.0f} N; valves throttle at system "
                                  f"pressure: holding free, walking costly")
    raise KeyError(arch)


def recommend(robot):
    """The lightest architecture that works, and the runner-up: (best, second, table)."""
    t = compare(robot)
    ok = sorted((a for a in t if t[a]["ok"]), key=lambda a: t[a]["mass_g"])
    return ok[0], (ok[1] if len(ok) > 1 else None), t


ARCHITECTURES = ("servo", "tendon", "spider", "valved", "hydrostatic")


def compare(robot):
    """{arch: size(robot, arch) + share of the robot's mass}."""
    out = {}
    for a in ARCHITECTURES:
        s = size(robot, a)
        s["share"] = s["mass_g"] / (ROBOTS[robot]["mass"] * 1000)
        s["ok"] = s["ok"] and s["share"] < 0.6                           # actuation above 60 % of the robot: no
        out[a] = s
    return out


# --------------------------------------------------------------------------------------------- tendon routing
#: how a tendon reaches its joint — the loss ``size`` leaves out: over idler pulleys on the joint axes (the tendon's
#: length does not change with the joints it crosses; assumed 0.97 per pulley, three pulleys) or in a sleeve (Bowden:
#: capstan friction e^(−μθ) over the sleeve's total bend θ as it crosses the hip yaw, the hip lift and the knee)
ROUTING = {"pulleys on the joint axes": None,
           "sleeve, μ 0.1, 150°": (0.1, 150.0), "sleeve, μ 0.1, 270°": (0.1, 270.0),
           "sleeve, μ 0.2, 150°": (0.2, 150.0), "sleeve, μ 0.2, 270°": (0.2, 270.0)}


def tendon_routing(robot):
    """Tendon against hydrostatic once the tendon's routing loss η counts: per ``ROUTING``, the tendon leg resized (its
    motor picked for the peak torque / η) and its walking power (/ η); the hydrostatic leg as ``size`` has it (no
    routing loss: the fluid line bends freely). Hydrostatic walks on less power whenever η is below
    ``EFFICIENCY['hydrostatic'] / EFFICIENCY['tendon']`` (``break_even_eta``)."""
    r, d = ROBOTS[robot], joint_demand(robot)
    n = r["joints"]
    tendon_g = 0.6 + 0.004 * d["peak_nm"] * 1000
    enc = _p("magnetic joint encoder")["g"]
    work_w = 0.5 * n * d["hold_nm"] * 0.25
    h = size(robot, "hydrostatic")
    h["share"] = h["mass_g"] / (r["mass"] * 1000)
    out = {"robot": robot, "break_even_eta": EFFICIENCY["hydrostatic"] / EFFICIENCY["tendon"], "hydrostatic": h,
           "routes": {}}
    for name, mu_wrap in ROUTING.items():
        eta = 0.97 ** 3 if mu_wrap is None else math.exp(-mu_wrap[0] * math.radians(mu_wrap[1]))
        m = pick_motor(d["peak_nm"] / eta, d["speed_rad_s"])
        mass = n * (m["g"] + tendon_g + enc) if m else float("nan")
        share = mass / (r["mass"] * 1000)
        out["routes"][name] = {"eta": eta, "mass_g": mass, "share": share, "walk_w": work_w / (EFFICIENCY["tendon"] * eta),
                               "motor": f"{m['name']} ×{m['ratio']:g}" if m else None, "ok": bool(m) and share < 0.6}
    return out


# --------------------------------------------------------------------------------------------- TARANTULA's jaws
#: what the jaws should cut: (shear strength [MPa], cut area [mm²], size between the jaws [mm], source) — a sharp
#: bypass shear needs ≈ strength × area (a blunt or loose blade needs more)
CUT_LOADS = {
    "cable tie 4.8 × 1.3 mm (nylon 6/6)": (60.0, 4.8 * 1.3, 1.3, "assumed: nylon 6/6 shear ~60 MPa"),
    "polypropylene cord Ø3 mm": (25.0, math.pi * 1.5 ** 2, 3.0, "assumed: PP shear ~25 MPa"),
    "copper wire Ø1.5 mm": (150.0, math.pi * 0.75 ** 2, 1.5, "assumed: annealed copper shear ~150 MPa"),
    "mild steel wire Ø1.0 mm": (280.0, math.pi * 0.5 ** 2, 1.0, "assumed: low-carbon steel shear ~280 MPa"),
    "green twig Ø5 mm": (20.0, math.pi * 2.5 ** 2, 5.0, "assumed: ~20 MPa effective across the grain — measure"),
    "galvanised fence wire Ø2.5 mm": (280.0, math.pi * 1.25 ** 2, 2.5, "assumed: as mild steel"),
}
#: the jaws' parts [g] (assumed class estimates)
CUTTER_PARTS = {"micro metal gearmotor 250:1": 9.5, "M4 × 0.7 leadscrew + bronze nut": 4.0, "crossbar yoke + 2 links": 4.0,
                "2 jaws (aluminium, hardened steel blade inserts)": 18.0, "cheliceral housing": 14.0,
                "pivot pins + tongues": 3.0, "limit switch + current sense": 1.0}


def cutter(spec=None, motor="micro metal gearmotor 250:1 (class)", screw_eff=0.35, load=0.5, throat_mm=10.0):
    """TARANTULA's jaws (``spider_specs.TARANTULA['cutter']``). A ``motor`` (``MOTORS``) turns the leadscrew: the nut
    pushes with F = 2π T η / lead at ``load`` × stall (and at stall). By virtual work over the linkage
    (``spider_specs.cutter_linkage``) the force on the work ``a`` mm in front of the pivots is F |dx/dθ| / (2 a) — one
    screw, two jaws. A part of size s at the throat (``throat_mm``) is met when the gap there closes to s
    (``cutter_gap``) and is cut by the time the edges cross; the force available is the least over that stretch.
    Returns the screw force, stroke, closing time, mass, the force/gap table and per ``CUT_LOADS`` whether it cuts
    at the running load and at stall."""
    import spider_specs as ss
    spec = spec or ss.TARANTULA
    c = spec["cutter"]
    name, _, stall, w0, _ = next(m for m in MOTORS if m[0] == motor)
    lead = c["lead_mm"] * 1e-3
    f_run = 2 * math.pi * stall * load * screw_eff / lead

    def dxdt(deg, h=0.25):
        lo, hi = max(deg - h, 0.0), deg + h
        return abs(ss.cutter_linkage(spec, hi)["yoke_x"] - ss.cutter_linkage(spec, lo)["yoke_x"]) / math.radians(hi - lo)

    def blade(deg, a):
        return f_run * dxdt(deg) / (2 * a)

    def crossed(a):                                                       # the opening where the edges meet at a
        lo, hi = 0.0, c["open_deg"]
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ss.cutter_gap(spec, mid, a) < 0 else (lo, mid)
        return lo

    def met(s, a):                                                        # the opening where the gap equals s
        lo, hi = crossed(a), c["open_deg"]
        if ss.cutter_gap(spec, hi, a) < s:
            return None
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ss.cutter_gap(spec, mid, a) < s else (lo, mid)
        return hi

    th0 = crossed(throat_mm)
    rows = [{"open_deg": d, "gap_throat_mm": ss.cutter_gap(spec, d, throat_mm),
             "blade_n_throat": blade(d, throat_mm), "blade_n_far": blade(d, c["cut_zone"][1])}
            for d in (30.0, 25.0, 20.0, 15.0, 10.0, 5.0)]
    cuts = {}
    for k, (tau, area, s, _) in CUT_LOADS.items():
        need = tau * area
        th1 = met(s, throat_mm)
        if th1 is None:
            cuts[k] = {"need_n": need, "fits": False, "cuts": False, "cuts_at_stall": False}
            continue
        avail = min(blade(th0 + (th1 - th0) * i / 20, throat_mm) for i in range(21))
        cuts[k] = {"need_n": need, "fits": True, "met_at_deg": th1, "avail_n": avail, "cuts": need <= avail,
                   "cuts_at_stall": need <= avail / load}
    o, z = ss.cutter_linkage(spec, c["open_deg"]), ss.cutter_linkage(spec, 0.0)
    stroke = abs(z["yoke_x"] - o["yoke_x"])
    speed = w0 * (1 - load) / (2 * math.pi) * c["lead_mm"]              # mm/s on the motor's linear torque-speed line
    return {"screw_n": f_run, "stroke_mm": stroke, "close_s": stroke / speed, "edges_cross_deg": th0,
            "max_gap_mm": ss.cutter_gap(spec, c["open_deg"], throat_mm), "table": rows, "cuts": cuts,
            "mass_g": sum(CUTTER_PARTS.values()), "motor": name}


# --------------------------------------------------------------------------------------------- TARANTULA's dragline
#: the dragline pod's parts [g] (class estimates)
WINCH_PARTS = {"spool + 10 m of 0.8 mm UHMWPE line": 8.0, "gearmotor 1000:1 (Pololu #3070 class)": 10.5,
               "clutch, band brake + its micro servo": 9.0, "launcher tube, spring, latch": 14.0, "harpoon": 12.0,
               "bracket, fairlead": 8.0}


def tarantula_winch(spec=None, mass_kg=1.5, line_break_n=400.0, harpoon_g=12.0, spring_n_m=None, launch_eff=0.6,
                    pullout_n=(80.0, 150.0), descent_m_s=0.25, share=0.6):
    """TARANTULA's dragline pod (``spider_specs.TARANTULA['winch']``): the line, the spool, the winch, the brake and
    the harpoon. The winch is a 1000:1 micro gearmotor straight on the spool (``MOTORS``): its pull at the full and
    the empty drum, its reel-in speed carrying ``share`` of the robot's weight and all of it; the brake's heat at the
    descent speed; the harpoon's spring energy, speed, range and holding against the robot's weight (``pullout_n``:
    a toggle-barbed dart in softwood, assumed). Returns a dict."""
    import spider_specs as ss
    spec = spec or ss.TARANTULA
    w = spec["winch"]
    W = mass_kg * G
    name, _, stall, w0, _ = next(m for m in MOTORS if m[0].startswith("micro metal gearmotor 1000:1"))
    r_empty, r_full = w["drum"] * 1e-3, (w["flange"] - 1.0) * 1e-3
    vol = math.pi * ((w["flange"] - 1.0) ** 2 - w["drum"] ** 2) * w["width"] * 0.65          # mm³, 65 % packed
    capacity_m = vol / (math.pi * (w["line_mm"] / 2) ** 2) / 1000

    def speed(force, r):                                                  # the motor's linear torque–speed line
        frac = force * r / stall
        return max(0.0, 1 - frac) * w0 * r, frac

    v_share, f_share = speed(share * W, r_full)
    v_all, f_all = speed(W, r_full)
    x = w["spring_mm"] * 1e-3
    E = None
    if spring_n_m is None:                                                # sized for 0.72 J at the muzzle
        E = 0.72 / launch_eff
        spring_n_m = 2 * E / x ** 2
    E = 0.5 * spring_n_m * x ** 2
    ek = E * launch_eff
    v = math.sqrt(2 * ek / (harpoon_g * 1e-3))
    return {"weight_n": W, "line_capacity_m": capacity_m, "line_safety": line_break_n / W,
            "pull_stall_n": {"full drum": stall / r_full, "empty drum": stall / r_empty},
            "reel_in_mm_s": {f"{share:.0%} of the weight": v_share * 1000, "all the weight": v_all * 1000},
            "motor_load_fraction": {f"{share:.0%} of the weight": f_share, "all the weight": f_all},
            "brake_heat_w": (1 - share) * 0 + W * descent_m_s, "descent_m_s": descent_m_s,
            "spring_n_m": spring_n_m, "spring_j": E, "cocking_force_n": spring_n_m * x,
            "can_recock": stall / r_full >= spring_n_m * x, "harpoon_m_s": v, "harpoon_j": ek,
            "range_45deg_m": v * v / G, "anchor_safety": (pullout_n[0] / W, pullout_n[1] / W),
            "mass_g": sum(WINCH_PARTS.values()), "motor": name}


# --------------------------------------------------------------------------------------------- jumping (SALTY)
def jump(mass_kg, distance_m, stroke_m, efficiency=0.5, angle_deg=45.0):
    """A ballistic jump of ``distance_m`` on level ground: take-off speed, kinetic and stored energy [J], take-off
    time [s], mean push force [N] and power [W] over a leg stroke of ``stroke_m`` (``efficiency``: stored → kinetic)."""
    a = math.radians(angle_deg)
    v = math.sqrt(G * distance_m / math.sin(2 * a))
    ek = 0.5 * mass_kg * v ** 2
    t = 2 * stroke_m / v
    f = mass_kg * (v ** 2 / (2 * stroke_m) + G * math.sin(a))
    return {"v_m_s": v, "kinetic_j": ek, "stored_j": ek / efficiency, "takeoff_s": t, "force_n": f,
            "power_w": ek / t, "apex_m": (v * math.sin(a)) ** 2 / (2 * G)}


def salty_catapult(mass_kg=0.25, distance_m=1.0, stroke_m=0.05, lever_ratio=7.5, bore_mm=10.0, p_hi_bar=16.0,
                   p_lo_bar=11.0, recharge_s=2.5):
    """SALTY's hydraulic catapult, spider-style: the pump charges a gas accumulator permanently connected to the
    third-leg pair's extensor cylinders while a flexor latch holds the legs folded; a trigger releases the latch and
    the pressure throws the legs open — no fast valve. Returns the sizing [SI and bar, cm³, g]."""
    j = jump(mass_kg, distance_m, stroke_m)
    per_leg_force = j["force_n"] / 2 * lever_ratio                        # at the cylinder, two jumping legs
    area = math.pi * (bore_mm / 2e3) ** 2
    p_need = per_leg_force / area / 1e5
    v_need = j["stored_j"] / (0.5 * (p_hi_bar + p_lo_bar) * 1e5) * 1e6        # fluid delivered [cm³]
    gas = v_need * p_lo_bar / (p_hi_bar - p_lo_bar)                       # isothermal gas pre-volume [cm³]
    q = v_need / recharge_s                                               # recharge flow [cm³/s]
    pump_w = p_hi_bar * 1e5 * q * 1e-6 / 0.4                              # 40 % pump + motor efficiency
    flow_peak = v_need / j["takeoff_s"]                                   # through the release line [cm³/s]
    line_v = flow_peak * 1e-6 / (math.pi * (1.5e-3) ** 2)                 # 3 mm bore line [m/s]
    mass = {"plunger pump + motor": 18.0, "gas accumulator": 12.0, "two cylinders": 16.0, "fluid + lines": 6.0,
            "latch + trigger": 4.0, "pressure sensor + relief": 2.0}
    spring_alt = {"spring (latex/steel) storing the same energy": j["stored_j"] / 0.25, "winch motor": 10.0,
                  "latch + trigger": 4.0}
    return {"jump": j, "cylinder_force_n": per_leg_force, "pressure_needed_bar": p_need, "fluid_cm3": v_need,
            "gas_cm3": gas, "recharge_flow_cm3_s": q, "pump_w": pump_w, "release_flow_cm3_s": flow_peak,
            "release_line_m_s": line_v, "mass_g": mass, "mass_total_g": sum(mass.values()),
            "spring_alternative_g": spring_alt, "spring_total_g": sum(spring_alt.values())}
