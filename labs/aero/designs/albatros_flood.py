"""ALBATROS over a flooded town (notebook 50, ``sims/albatros_flood.py``): medicine to people on the roofs.

A river town on the plain of eastern Poland or Lithuania — a main street of painted wooden houses with sheet-metal
gable roofs, a church, a school, a block of flats, poplars — stands in 2.4 m of water. People wait on the roofs.
ALBATROS does not stop: it flies one pass over the town and releases its bay load of ÉMERILLONs (notebook 37's
electric speed quadcopter, here ``Carrier``: an ER2 with a medicine pod in its tail). Each ÉMERILLON catches itself,
flies to its roof and delivers:

- on a **pitched roof** it hovers tail-down a little above the ridge beside the people and lets go a **saddle bag**:
  two pouches on a strap, dropped across the ridge so that one hangs on each slope (a single box slides off a wet
  metal roof into the water; ``payload="box"`` shows it);
- on a **flat roof** it lands on its fins with the medicine on board and stays.

After a drop it flies to the school's flat roof and lands there to be collected.

**What MuJoCo simulates.** The town is static geometry (walls, roof slabs, ridge caps, the church, trees; the water
is a level, not a fluid). Every ÉMERILLON is a free rigid body: gravity, contacts with roofs, and each step the
forces this module computes — thrust along the body axis (limited by ``emerillon_flight.drive``'s thrust against
airspeed, with a first-order lag), a torque that turns the axis to where the wanted force points, and the air's
force from the airspeed through wind and gusts (``emerillon_flight``'s axial drag area and normal-force area;
``ogma.wind``). The saddle bags are two free pouches joined by a chain of capsule links on ball joints; they fall,
hit the metal, slide, catch on the ridge or do not — that is contact physics, not a rule. ALBATROS is a moving
marker (its flight is ``albatros_flight``'s); the release gives each ÉMERILLON its velocity.

**What is not simulated.** After its drop a carrier keeps its inertia and gets the bag's weight back as an upward
force (it hovers at the right thrust; it accelerates a little lazily). The strap is taken as paid out at the release (the bag appears spread under the tail);
no propeller aerodynamics at an angle, no rotor wash on the bag or the people, no attitude sensor noise, no radio;
the people do not move. A bag in the water is counted lost. The guidance knows the roofs' positions (a survey by
the first drone on scene, or a map) — it does not see.

``Judge``: a bag is **delivered** when it has come to rest above the water on its roof, within reach of the people
(``reach_m``); its ``shock_g`` is the largest deceleration a pouch saw, against ``SHOCK_LIMIT_G`` for what is in it
(vaccine vials in foam, or solids: tablets, dressings). ``cold_chain`` follows a vaccine box's temperature from the
depot through an hour in an unheated bay at altitude to the hours on a roof.

    import albatros_flood as fl
    scene = fl.Scene()
    run = fl.run(scene)
    fl.table(run), fl.render_movie(run, "flood.mp4")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ogma.units import G
from ogma.units import RHO0 as RHO

FLOOD_M = 2.4
SHOCK_LIMIT_G = {"vaccine": 60.0, "solids": 300.0}       # assumed: glass vials in 30 mm of foam; blister packs and dressings
HOUSE_COLOURS = ((0.80, 0.66, 0.30), (0.45, 0.58, 0.42), (0.55, 0.38, 0.26), (0.42, 0.55, 0.68), (0.78, 0.74, 0.62), (0.62, 0.30, 0.25))
ROOF_COLOURS = ((0.55, 0.20, 0.16), (0.42, 0.44, 0.46), (0.30, 0.34, 0.30), (0.50, 0.30, 0.22))


# =========================================================================================================== the carrier
@dataclass(frozen=True)
class Carrier:
    """ÉMERILLON with a medicine pod: the numbers the simulation needs (``carrier()`` reads them from notebook 37)."""
    name: str
    mass_kg: float                    # without the medicine
    payload_kg: float
    length_m: float
    body_d_m: float
    arm_r_m: float
    prop_d_m: float
    thrust_n: tuple                   # ((airspeed, total thrust), ...) at full throttle
    drag_area_m2: float               # axial Cd.A
    normal_area_m2: float             # Cd.A broadside
    pack_wh: float
    disk_area_m2: float               # the four disks
    pod_l_m: float = 0.14
    pod_d_m: float = 0.095
    drive_efficiency: float = 0.55    # motor, controller and propeller figure of merit together (assumed)
    usable: float = 0.8

    @property
    def gross_kg(self) -> float:
        return self.mass_kg + self.payload_kg

    def thrust_max(self, v_axial: float) -> float:
        v, t = zip(*self.thrust_n)
        return float(np.interp(abs(v_axial), v, t))

    def hover_power(self, mass_kg: float | None = None) -> float:
        T = (mass_kg or self.gross_kg) * G
        return T * math.sqrt(T / (2 * RHO * self.disk_area_m2)) / self.drive_efficiency

    @property
    def envelope_m(self) -> tuple:
        """(length, width, height) it takes in the bay, the two-blade propellers parked along its sides."""
        s = 2 * (self.arm_r_m / math.sqrt(2) + 0.03)
        return (self.length_m + 0.4 * self.pod_l_m, s, s)


def carrier(variant: str = "ER2", payload_kg: float = 0.5) -> Carrier:
    """The ``Carrier`` of an ÉMERILLON variant: its mass, thrust, drag and pack from ``emerillon_flight``, a pod of
    0.09 kg (printed shell, latch, servo) in place of the tail cone's end, the pod's base drag on the axial area."""
    import emerillon_flight as ef
    import emerillon_market as mk

    c = ef.VARIANTS[variant]
    g = ef.geometry(c)
    bat = mk.part(c.battery)
    thrust = tuple((float(v), float(ef.drive(c, float(v))["thrust"])) for v in (0.0, 15.0, 30.0, 45.0, 60.0))
    pod_d = 0.095
    base = 0.25 * math.pi * pod_d ** 2 / 4                              # a blunt base behind a boattail: Cd ~ 0.25 on its area
    return Carrier(f"Émerillon {variant} with a medicine pod", ef.mass_budget(c).mass + 0.09, payload_kg, g["length"], g["d"], g["r_arm"],
                   g["prop_d"], thrust, ef.drag_buildup(c, 40.0)["CdA"] + base, ef.normal_force_area(c, math.pi / 2) + 0.6 * pod_d * 0.14,
                   bat.cells * 3.7 * bat.ah * c.packs,
                   4 * math.pi * (g["prop_d"] / 2) ** 2)


def bay_option(car: Carrier, count: int = 6, abreast: int = 2):
    """``albatros_bay.BayOption`` of ``count`` carriers lying along the fuselage, ``abreast`` side by side."""
    import albatros_bay as B

    L, W, H = car.envelope_m
    v = B.Multirotor(car.name, 4, car.payload_kg, car.mass_kg, 0.25, car.prop_d_m, car.hover_power(), 300.0, (L, W, H),
                     car.arm_r_m, False, {"cargo": car.payload_kg, "the carrier": car.mass_kg})
    return B.BayOption(f"{count} Émerillons with medicine pods", (v,) * count, f"{abreast} abreast", abreast=abreast)


# =========================================================================================================== the town
@dataclass
class Building:
    kind: str                         # house | school | block | church | tower
    x: float
    y: float
    yaw_deg: float
    length: float                     # along the ridge (local x)
    width: float
    wall_h: float
    pitch_deg: float = 0.0            # 0: a flat roof
    colour: tuple = (0.7, 0.7, 0.7)
    roof_colour: tuple = (0.45, 0.45, 0.45)
    people: int = 0

    @property
    def ridge_z(self) -> float:
        return self.wall_h + 0.5 * self.width * math.tan(math.radians(self.pitch_deg))

    def world(self, lx: float, ly: float) -> tuple:
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        return self.x + c * lx - s * ly, self.y + s * lx + c * ly

    def roof_z(self, lx: float, ly: float) -> float:
        """Height of the roof surface over the local point (inside the footprint)."""
        if self.pitch_deg == 0:
            return self.wall_h
        return self.ridge_z - abs(ly) * math.tan(math.radians(self.pitch_deg))


@dataclass
class Scene:
    """The town, the wind, the pass and who gets what. ``payload``: "saddle" (two pouches on a strap, across the
    ridge) or "box" (one box on the ridge: what does not work). ``contents`` cycles over the recipients."""
    seed: int = 7
    flood_m: float = FLOOD_M
    wind: tuple = (5.0, 3.0, 0.0)                 # steady [m/s]
    gust_sigma: float = 1.5
    pass_height: float = 100.0
    pass_speed: float | None = None               # None: the sized Albatros's release speed (``pass_speed_default``)
    pass_speed_default: float = 36.0
    pass_y: float = -15.0
    release_x: float = -190.0                     # where the first Émerillon leaves the bay
    release_gap_s: float = 0.6
    drones: int = 6
    payload: str = "saddle"
    contents: tuple = ("vaccine", "solids")
    hover_above_ridge: float = 1.1                # the tail above the ridge at the drop [m]
    reach_m: float = 3.0                          # a bag this close to the people along the roof is in reach
    cruise_speed: float = 32.0
    transit_height: float = 34.0                  # over the church's tower and the poplars
    carrier: Carrier | None = None
    buildings: list = field(default_factory=list)
    trees: list = field(default_factory=list)
    time_limit: float = 150.0

    def __post_init__(self):
        if self.payload not in ("saddle", "box"):
            raise ValueError("payload is 'saddle' or 'box'")
        self.carrier = self.carrier or carrier()
        self.pass_speed = self.pass_speed or self.pass_speed_default
        if not self.buildings:
            self._build_town()

    # ------------------------------------------------------------------------------------------------ layout
    def _build_town(self):
        rng = np.random.default_rng(self.seed)
        b = []
        # the main street runs along x; a row of houses on each side, gables to the street (ridge across it) or along it
        for side in (-1, 1):
            x = -105.0
            while x < 95.0:
                L, W = rng.uniform(9.0, 12.5), rng.uniform(6.5, 8.5)
                along = rng.random() < 0.6
                b.append(Building("house", x + L / 2, side * (13.0 + rng.uniform(0, 3.0) + (W if along else L) / 2), 0.0 if along else 90.0, L, W,
                                  rng.uniform(2.9, 3.4), rng.uniform(33.0, 42.0), HOUSE_COLOURS[rng.integers(len(HOUSE_COLOURS))],
                                  ROOF_COLOURS[rng.integers(len(ROOF_COLOURS))]))
                x += (L if along else W) + rng.uniform(7.0, 13.0)
        # a second street of houses behind, to the north
        x = -90.0
        while x < 40.0:
            L, W = rng.uniform(9.0, 12.0), rng.uniform(6.5, 8.0)
            b.append(Building("house", x + L / 2, 52.0 + rng.uniform(-2, 2), rng.uniform(-8, 8), L, W, rng.uniform(2.9, 3.3), rng.uniform(34.0, 42.0),
                              HOUSE_COLOURS[rng.integers(len(HOUSE_COLOURS))], ROOF_COLOURS[rng.integers(len(ROOF_COLOURS))]))
            x += L + rng.uniform(9.0, 16.0)
        b.append(Building("school", 75.0, 55.0, 0.0, 32.0, 13.0, 7.6, 0.0, (0.80, 0.76, 0.66), (0.33, 0.33, 0.35)))
        b.append(Building("block", -70.0, -55.0, 0.0, 26.0, 11.0, 11.6, 0.0, (0.72, 0.72, 0.70), (0.30, 0.30, 0.32)))
        b.append(Building("church", 118.0, -4.0, 0.0, 22.0, 10.0, 8.0, 52.0, (0.88, 0.86, 0.80), (0.28, 0.36, 0.30)))
        b.append(Building("tower", 103.5, -4.0, 0.0, 5.5, 5.5, 20.0, 0.0, (0.88, 0.86, 0.80), (0.28, 0.36, 0.30)))
        self.buildings = b
        houses = [i for i, q in enumerate(b) if q.kind == "house"]
        chosen = list(rng.choice(houses, size=max(self.drones - 1, 1), replace=False))
        for i in chosen:
            b[i].people = int(rng.integers(2, 5))
        b[[q.kind for q in b].index("school")].people = 6
        self.trees = [(float(rng.uniform(-125, 135)), float(rng.choice((-1, 1)) * rng.uniform(30, 44)), float(rng.uniform(9, 15))) for _ in range(16)]
        self.trees += [(float(rng.uniform(-110, 60)), float(rng.uniform(64, 74)), float(rng.uniform(9, 14))) for _ in range(8)]

    # ------------------------------------------------------------------------------------------------ who gets what
    def recipients(self) -> list:
        """One entry per Émerillon: the building, where the people are, the hover or landing point, what is carried."""
        out = []
        sites = [q for q in self.buildings if q.people and q.kind == "house"] + [q for q in self.buildings if q.people and q.kind != "house"]
        for k, q in enumerate(sites[:self.drones]):
            if q.pitch_deg > 0:
                lx_people = -0.18 * q.length
                lx_drop = lx_people + 0.5 * 0.75 * (q.people - 1) + 1.6      # an arm's length and a bit past the last person
                px, py = q.world(lx_people, 0.0)
                dx, dy = q.world(lx_drop, 0.0)
                out.append({"building": q, "mode": "drop", "people": (px, py, q.ridge_z), "target": (dx, dy, q.ridge_z + self.hover_above_ridge),
                            "contents": self.contents[k % len(self.contents)], "ridge_yaw": math.radians(q.yaw_deg)})
            else:
                px, py = q.world(-0.25 * q.length, 0.0)
                lx, ly = q.world(-0.25 * q.length + 3.5, 1.5)
                out.append({"building": q, "mode": "land", "people": (px, py, q.wall_h), "target": (lx, ly, q.wall_h), "contents": self.contents[k % len(self.contents)],
                            "ridge_yaw": 0.0})
        return out

    def collection(self, k: int) -> tuple:
        """Where the k-th Émerillon lands after its drop: a row on the school's flat roof, away from the people."""
        s = next(q for q in self.buildings if q.kind == "school")
        x, y = s.world(2.0 + 2.2 * k, -2.5)
        return x, y, s.wall_h


# =========================================================================================================== the model
def _rgba(c, a=1.0):
    return f"{c[0]:.2f} {c[1]:.2f} {c[2]:.2f} {a:.2f}"


def mjcf(scene: Scene) -> str:
    """The MuJoCo model of the scene as XML."""
    car = scene.carrier
    F = scene.flood_m
    w = []
    add = w.append
    add('<geom name="ground" type="box" size="900 900 1" pos="0 0 -1" rgba="0.32 0.30 0.24 1" friction="0.8"/>')
    add(f'<geom name="water" type="box" size="900 900 {F / 2:.3f}" pos="0 0 {F / 2:.3f}" rgba="0.40 0.45 0.42 0.85" contype="0" conaffinity="0"/>')
    for i, q in enumerate(scene.buildings):
        add(f'<body name="b{i}" pos="{q.x:.2f} {q.y:.2f} 0" euler="0 0 {q.yaw_deg:.2f}">')
        add(f'<geom type="box" size="{q.length / 2:.2f} {q.width / 2:.2f} {q.wall_h / 2:.2f}" pos="0 0 {q.wall_h / 2:.2f}" rgba="{_rgba(q.colour)}"/>')
        if q.pitch_deg > 0:
            p = math.radians(q.pitch_deg)
            slope = 0.5 * q.width / math.cos(p) + 0.35                      # with the eaves' overhang
            for sgn in (-1, 1):
                cy = sgn * 0.5 * (slope - 0.0) * math.cos(p) - sgn * 0.0
                cz = q.ridge_z - 0.5 * slope * math.sin(p) - 0.04
                add(f'<geom name="roof{i}_{0 if sgn < 0 else 1}" type="box" size="{q.length / 2 + 0.3:.2f} {slope / 2:.3f} 0.04" pos="0 {cy:.3f} {cz:.3f}" '
                    f'euler="{-sgn * q.pitch_deg:.2f} 0 0" rgba="{_rgba(q.roof_colour)}" friction="0.35 0.01 0.001"/>')
            add(f'<geom name="ridge{i}" type="capsule" size="0.05 {q.length / 2 + 0.3:.2f}" pos="0 0 {q.ridge_z + 0.01:.3f}" euler="0 90 0" '
                f'rgba="{_rgba(q.roof_colour)}" friction="0.5 0.01 0.001"/>')
            gable = q.ridge_z - q.wall_h                                    # the gable walls: a stack of narrowing boxes
            for k in range(4):
                hw = 0.5 * q.width * (1 - (k + 0.5) / 4)
                for sgn in (-1, 1):
                    add(f'<geom type="box" size="0.12 {hw:.2f} {gable / 8:.3f}" pos="{sgn * (q.length / 2 - 0.12):.2f} 0 {q.wall_h + (k + 0.5) * gable / 4:.3f}" '
                        f'rgba="{_rgba(q.colour)}" contype="0" conaffinity="0"/>')
            if q.kind == "house":
                add(f'<geom type="box" size="0.3 0.3 0.6" pos="{0.3 * q.length:.2f} {0.12 * q.width:.2f} {q.ridge_z + 0.2:.2f}" rgba="0.45 0.25 0.20 1"/>')
        else:
            add(f'<geom name="roof{i}_0" type="box" size="{q.length / 2:.2f} {q.width / 2:.2f} 0.05" pos="0 0 {q.wall_h - 0.05:.2f}" rgba="{_rgba(q.roof_colour)}" friction="0.7"/>')
            if q.kind == "tower":
                for k in range(5):
                    s = 0.5 * q.width * (1 - k / 5) + 0.2
                    add(f'<geom type="box" size="{s:.2f} {s:.2f} 0.7" pos="0 0 {q.wall_h + 0.7 + 1.4 * k:.2f}" rgba="{_rgba(q.roof_colour)}"/>')
            else:
                for sx, sy, hx, hy in ((0, 1, q.length / 2, 0.1), (0, -1, q.length / 2, 0.1), (1, 0, 0.1, q.width / 2), (-1, 0, 0.1, q.width / 2)):
                    add(f'<geom type="box" size="{hx:.2f} {hy:.2f} 0.25" pos="{sx * (q.length / 2 - 0.1):.2f} {sy * (q.width / 2 - 0.1):.2f} {q.wall_h + 0.25:.2f}" '
                        f'rgba="{_rgba(q.colour)}"/>')
        add('</body>')
    for j, (x, y, h) in enumerate(scene.trees):
        add(f'<geom type="cylinder" size="0.25 {h / 2:.2f}" pos="{x:.1f} {y:.1f} {h / 2:.2f}" rgba="0.35 0.27 0.18 1"/>')
        add(f'<geom type="ellipsoid" size="2.0 2.0 {0.38 * h:.2f}" pos="{x:.1f} {y:.1f} {0.72 * h:.2f}" rgba="0.24 0.42 0.20 1" contype="0" conaffinity="0"/>')
    # the people: capsules on the ridge or the flat roof (they do not collide: nobody is hit in this model)
    shirts = ((0.85, 0.2, 0.2), (0.2, 0.3, 0.8), (0.9, 0.8, 0.2), (0.9, 0.9, 0.9), (0.2, 0.6, 0.3), (0.6, 0.3, 0.6))
    n = 0
    for r in scene.recipients():
        q, (px, py, pz) = r["building"], r["people"]
        c, s = math.cos(math.radians(q.yaw_deg)), math.sin(math.radians(q.yaw_deg))
        for k in range(q.people):
            off = (k - (q.people - 1) / 2) * 0.75
            x, y = px + c * off - (s * 1.2 * (k % 2) if q.pitch_deg == 0 else 0.0), py + s * off + (c * 1.2 * (k % 2) if q.pitch_deg == 0 else 0.0)
            sit = q.pitch_deg > 0
            hgt = 0.45 if sit else 0.62
            add(f'<geom type="capsule" size="0.17 {hgt:.2f}" pos="{x:.2f} {y:.2f} {pz + hgt + 0.17:.2f}" rgba="{_rgba(shirts[n % len(shirts)])}" contype="0" conaffinity="0"/>')
            add(f'<geom type="sphere" size="0.12" pos="{x:.2f} {y:.2f} {pz + 2 * hgt + 0.42:.2f}" rgba="0.87 0.72 0.60 1" contype="0" conaffinity="0"/>')
            n += 1
    # Albatros: a marker that flies the pass
    add('<body name="albatros" mocap="true" pos="-900 0 100">')
    add('<geom type="capsule" size="0.3 1.5" euler="0 90 0" rgba="0.92 0.92 0.94 1" contype="0" conaffinity="0"/>')
    add('<geom type="box" size="0.25 1.6 0.03" pos="0.1 0 0.25" rgba="0.92 0.92 0.94 1" contype="0" conaffinity="0"/>')
    add('<geom type="box" size="0.18 0.5 0.02" pos="-1.5 0 0.4" rgba="0.9 0.35 0.1 1" contype="0" conaffinity="0"/>')
    add('<geom type="capsule" size="0.08 0.2" pos="-1.0 0.42 0.25" euler="0 90 0" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>')
    add('<geom type="capsule" size="0.08 0.2" pos="-1.0 -0.42 0.25" euler="0 90 0" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>')
    add('</body>')
    # the Émerillons: body axis = local z (nose up when it hovers), the tail's four feet at the fin tips
    Lb, r_arm = car.length_m, car.arm_r_m
    zt = -0.52 * Lb - 0.4 * car.pod_l_m                                     # the feet's plane below the centre of mass
    I_t, I_a = car.gross_kg * (Lb ** 2 / 12 + r_arm ** 2 / 4) * 0.9, car.gross_kg * r_arm ** 2 * 0.45
    for k in range(scene.drones):
        add(f'<body name="em{k}" pos="{-900 + 3 * k} 40 300"><freejoint name="em{k}_j"/>')
        add(f'<inertial pos="0 0 0" mass="{car.gross_kg:.4f}" diaginertia="{I_t:.5f} {I_t:.5f} {I_a:.5f}"/>')
        add(f'<geom type="capsule" size="{car.body_d_m / 2:.3f} {0.36 * Lb:.3f}" pos="0 0 {0.08 * Lb:.3f}" rgba="0.95 0.45 0.10 1" contype="0" conaffinity="0"/>')
        add(f'<geom type="cylinder" size="{car.pod_d_m / 2:.3f} {car.pod_l_m / 2:.3f}" pos="0 0 {-0.38 * Lb:.3f}" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>')
        for a in (45, 135, 225, 315):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            add(f'<geom type="capsule" size="0.012 {r_arm / 2:.3f}" pos="{ca * r_arm / 2:.3f} {sa * r_arm / 2:.3f} 0.02" euler="{-math.degrees(math.atan2(ca, 0.0)) if False else 0} 90 {a}" '
                f'rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>')
            add(f'<geom type="cylinder" size="{car.prop_d_m / 2:.3f} 0.004" pos="{ca * r_arm:.3f} {sa * r_arm:.3f} 0.07" rgba="0.6 0.6 0.6 0.35" contype="0" conaffinity="0"/>')
        for a in (0, 90, 180, 270):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            add(f'<geom name="em{k}_foot{a}" type="sphere" size="0.02" pos="{ca * 0.115:.3f} {sa * 0.115:.3f} {zt:.3f}" rgba="0.1 0.1 0.1 1" friction="1.0 0.02 0.001" '
                f'solref="0.01 1"/>')
        add('</body>')
    # the bags, parked far away until their release
    for k in range(scene.drones):
        x0 = 4000.0 + 3.0 * k
        pouch = 'type="box" size="0.055 0.04 0.03" mass="{m:.3f}" friction="0.8 0.02 0.002" solref="0.02 1" rgba="{c}"'
        col = "0.95 0.15 0.15 1" if k % 2 == 0 else "0.15 0.35 0.9 1"
        half = 0.5 * scene.carrier.payload_kg - 0.012
        if scene.payload == "saddle":
            add(f'<body name="bag{k}a" pos="{x0} 0 0.05"><freejoint name="bag{k}_j"/><geom name="bag{k}a_g" {pouch.format(m=half, c=col)}/>')
            links, ll = 4, 0.13
            for j in range(links):
                add(f'<body name="bag{k}l{j}" pos="{0.055 if j == 0 else ll:.3f} 0 0"><joint type="ball" damping="0.002"/>'
                    f'<geom type="capsule" size="0.008 {ll / 2 - 0.008:.3f}" pos="{ll / 2:.3f} 0 0" euler="0 90 0" mass="0.006" friction="0.8 0.02 0.002" rgba="0.95 0.85 0.1 1"/>')
            add(f'<body name="bag{k}b" pos="{ll + 0.055:.3f} 0 0"><joint type="ball" damping="0.002"/><geom name="bag{k}b_g" {pouch.format(m=half, c=col)}/></body>')
            add('</body>' * links)
            add('</body>')
        else:
            add(f'<body name="bag{k}a" pos="{x0} 0 0.05"><freejoint name="bag{k}_j"/>'
                f'<geom name="bag{k}a_g" type="box" size="0.07 0.055 0.045" mass="{scene.carrier.payload_kg:.3f}" friction="0.8 0.02 0.002" solref="0.02 1" rgba="{col}"/></body>')
    world = "\n".join(w)
    return f"""<mujoco model="albatros_flood">
  <compiler angle="degree"/>
  <option timestep="0.002" gravity="0 0 -9.80665" integrator="implicitfast"/>
  <statistic extent="300" center="0 0 10"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/><map znear="0.001" zfar="12"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5"/><rgba haze="0.75 0.78 0.80 1"/></visual>
  <asset><texture type="skybox" builtin="gradient" rgb1="0.62 0.66 0.70" rgb2="0.82 0.84 0.85" width="64" height="256"/></asset>
  <worldbody>
    <light directional="true" pos="0 0 200" dir="0.3 0.4 -1" diffuse="0.6 0.6 0.58" castshadow="true"/>
{world}
  </worldbody>
</mujoco>"""


# =========================================================================================================== the flight
def _axis(R):
    return R[:, 2]


class Pilot:
    """One Émerillon's guidance and control: a mode, a wanted velocity, the wanted force, the axis and the thrust."""

    def __init__(self, k: int, scene: Scene, recipient: dict):
        self.k, self.scene, self.r = k, scene, recipient
        self.car = scene.carrier
        self.mode = "bay"
        self.t_mode = 0.0
        self.thrust = 0.0
        self.integral = np.zeros(3)
        self.energy_wh = 0.0
        self.mass = self.car.gross_kg
        self.events = {}
        self.dropped = False
        self.peak_tilt = 0.0
        self.lightened = 0.0              # N: the weight that left with the bag (the body keeps its mass in the model:
                                          # MuJoCo's constants are compiled; the weight is given back as a force)

    def go(self, mode, t):
        self.mode, self.t_mode = mode, t
        self.events[mode] = t
        self.integral[:] = 0.0

    def goal(self):
        """(point, speed cap, height to hold until close) for the mode."""
        sc = self.scene
        tail = 0.52 * self.car.length_m + 0.4 * self.car.pod_l_m       # the feet below the centre of mass
        if self.mode in ("transit", "approach", "hover"):
            return np.array(self.r["target"]) + (0, 0, tail if self.r["mode"] == "drop" else 0.9), sc.cruise_speed
        x, y, z = sc.collection(self.k) if self.r["mode"] == "drop" else self.r["target"]
        return np.array([x, y, z + 0.9]), sc.cruise_speed

    def command(self, t, p, v, R, omega, wind, dt, feet_force):
        """The force and torque to apply this step (world frame), and the mode changes."""
        car, sc = self.car, self.scene
        n = _axis(R)
        if self.mode in ("bay", "done"):
            return np.zeros(3), np.zeros(3)
        va = v - wind
        v_ax = float(va @ n)
        v_perp = va - v_ax * n
        aero = -0.5 * RHO * (car.drag_area_m2 * abs(v_ax) * v_ax * n + car.normal_area_m2 * np.linalg.norm(v_perp) * v_perp)
        aero = aero + (0.0, 0.0, self.lightened)
        if self.mode == "fall":                                         # clear of the aircraft, rotors starting
            if t - self.t_mode > 0.5:
                self.go("transit", t)
            return aero, -0.02 * omega
        target, vcap = self.goal()
        e = target - p
        d_h = float(np.hypot(e[0], e[1]))
        landing = self.mode in ("land", "descend")
        if self.mode == "transit":                                      # high until nearly over it, then down
            hold = max(sc.transit_height, target[2] + 6.0)
            aim = np.array([target[0], target[1], hold if d_h > 18.0 else target[2] + 4.0])
            if d_h < 4.0 and abs(p[2] - (target[2] + 4.0)) < 2.0 and np.linalg.norm(v) < 4.0:
                self.go("approach", t)
        elif self.mode == "egress":
            hold = max(sc.transit_height, target[2] + 6.0)
            aim = np.array([target[0], target[1], hold if d_h > 14.0 else target[2] + 2.0])
            if d_h < 3.0 and abs(p[2] - (target[2] + 2.0)) < 2.0:
                self.go("descend", t)
        elif self.mode == "climb":                                      # straight up from the roof after the drop
            aim = np.array([p[0], p[1], self.r["target"][2] + 9.0])
            if p[2] > self.r["target"][2] + 6.0:
                self.go("egress", t)
        else:
            aim = target.copy()
        ea = aim - p
        dist = float(np.linalg.norm(ea))
        slow = self.mode in ("approach", "hover", "descend", "climb")
        near = self.mode in ("transit", "egress") and d_h <= 18.0       # coming down over the roof: gently
        speed = min(vcap, math.sqrt(2 * (1.5 if near else 5.0) * dist)) if not slow else min(2.0, 1.2 * dist)
        v_cmd = ea / max(dist, 1e-6) * speed
        if self.mode == "descend":                                      # settle onto the flat roof
            v_cmd = np.array([1.5 * e[0], 1.5 * e[1], -0.7])
            v_cmd[:2] = np.clip(v_cmd[:2], -1.5, 1.5)
        a = (v_cmd - v) / 0.45
        if slow:
            if dist < 1.0 and self.mode in ("approach", "hover"):     # the last metre: hold the point against the wind
                self.integral += 1.2 * ea * dt
            self.integral = np.clip(self.integral, -3.0, 3.0)
            a += self.integral
        a_lim = 14.0 if not slow else 6.0
        na = float(np.linalg.norm(a))
        if na > a_lim:
            a *= a_lim / na
        f_des = self.mass * (a + np.array([0, 0, G])) - aero
        if f_des[2] < 0.25 * self.mass * G:                             # never ask the axis below the horizon
            f_des[2] = 0.25 * self.mass * G
        fn = float(np.linalg.norm(f_des))
        n_des = f_des / fn
        # the events of the delivery
        if self.mode == "approach" and dist < 0.35 and np.linalg.norm(v) < 0.5:
            self.go("hover", t)
        if self.mode == "hover" and t - self.t_mode > 1.0 and dist < 0.3 and np.linalg.norm(v) < 0.35:
            if self.r["mode"] == "drop":
                self.dropped = True                                     # the simulation spawns the bag and lightens the carrier
                self.go("climb", t)
            else:
                self.go("descend", t)
        if self.mode == "descend" and feet_force > 0.5 * self.mass * G * 0.3 and t - self.t_mode > 0.5:
            self.go("landed", t)
        if self.mode == "landed":
            if t - self.t_mode > 1.5:
                self.go("done", t)
            self.thrust += (0.0 - self.thrust) * dt / 0.15
            return aero + self.thrust * n, -0.05 * omega
        want = float(np.clip(f_des @ n, 0.0, car.thrust_max(v_ax)))
        self.thrust += (want - self.thrust) * dt / 0.06
        # the axis follows the wanted force: a PD on the angle between them, the spin about the axis damped
        err = np.cross(n, n_des)
        if float(n @ n_des) < 0:                                        # more than a right angle away: turn at the full rate
            ne = float(np.linalg.norm(err))
            err = err / ne if ne > 1e-6 else np.cross(n, (1.0, 0.0, 0.0))
        w_ax = float(omega @ n)
        w_t = omega - w_ax * n
        I_t = car.gross_kg * (car.length_m ** 2 / 12 + car.arm_r_m ** 2 / 4) * 0.9
        I_a = car.gross_kg * car.arm_r_m ** 2 * 0.45
        tau = I_t * (160.0 * err - 22.0 * w_t) - I_a * 8.0 * w_ax * n
        tmax = 0.35 * car.thrust_max(0.0) * car.arm_r_m                 # what differential thrust gives
        nt = float(np.linalg.norm(tau))
        if nt > tmax:
            tau *= tmax / nt
        self.debug = {"n_des": n_des, "tau": tau, "f_des": f_des, "a": a, "v_cmd": v_cmd, "omega": omega}
        self.peak_tilt = max(self.peak_tilt, math.degrees(math.acos(max(min(n[2], 1.0), -1.0))))
        v_i = math.sqrt(max(self.thrust, 0.0) / (2 * RHO * car.disk_area_m2))
        self.energy_wh += self.thrust * (max(v_ax, 0.0) + v_i) / car.drive_efficiency * dt / 3600
        return aero + self.thrust * n, tau


def run(scene: Scene | None = None, *, log_dt: float = 0.02, progress=None) -> dict:
    """Fly the pass and the deliveries. Returns the model, the logged states (for the movie), the pilots and the
    judgement of every bag (``Judge``)."""
    import mujoco
    from ogma.wind import Wind

    scene = scene or Scene()
    car = scene.carrier
    model = mujoco.MjModel.from_xml_string(mjcf(scene))
    data = mujoco.MjData(model)
    recips = scene.recipients()
    n = min(scene.drones, len(recips))
    pilots = [Pilot(k, scene, recips[k]) for k in range(n)]
    wind = Wind(steady=scene.wind, sigma=scene.gust_sigma, tau=4.0, seed=scene.seed)
    wind.reset()
    bid = [model.body(f"em{k}").id for k in range(n)]
    jad = [model.joint(f"em{k}_j").qposadr[0] for k in range(n)]
    jvd = [model.joint(f"em{k}_j").dofadr[0] for k in range(n)]
    bag_q = [model.joint(f"bag{k}_j").qposadr[0] for k in range(n)]
    bag_v = [model.joint(f"bag{k}_j").dofadr[0] for k in range(n)]
    bag_a = [model.body(f"bag{k}a").id for k in range(n)]
    bag_b = [model.body(f"bag{k}b").id if scene.payload == "saddle" else model.body(f"bag{k}a").id for k in range(n)]
    foot_geoms = [{model.geom(f"em{k}_foot{a}").id for a in (0, 90, 180, 270)} for k in range(n)]
    mocap = model.body("albatros").mocapid[0]
    dt = model.opt.timestep
    x0 = scene.release_x - scene.pass_speed * 8.0                        # Albatros arrives eight seconds before the first release
    t_rel = [8.0 + k * scene.release_gap_s for k in range(n)]
    v_pass = np.array([scene.pass_speed, 0.0, 0.0])
    judge = Judge(scene, recips[:n])
    log = {"t": [], "qpos": [], "mocap": [], "thrust": [], "mode": []}
    track = {k: {"t": [], "p": [], "v": [], "thrust": [], "energy": []} for k in range(n)}
    released = [False] * n
    spawned = [False] * n
    t, next_log = 0.0, 0.0
    up = np.array([1.0, 0, 0, 0])
    while t < scene.time_limit:
        pa = np.array([x0 + scene.pass_speed * t, scene.pass_y, scene.pass_height])
        data.mocap_pos[mocap] = pa
        wind.step(dt)
        w = wind.at(t)
        for k, pl in enumerate(pilots):
            q0, v0 = jad[k], jvd[k]
            if not released[k]:
                if t >= t_rel[k]:                                        # out of the bay: nose forward along the pass, pushed down
                    released[k] = True
                    data.qpos[q0:q0 + 3] = pa + (0.0, 0.0, -0.6)
                    data.qpos[q0 + 3:q0 + 7] = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)      # body z (the nose) along +x
                    data.qvel[v0:v0 + 3] = v_pass + (0.0, 0.0, -3.0)
                    data.qvel[v0 + 3:v0 + 6] = 0.0
                    pl.go("fall", t)
                    pl.events["released"] = t
                else:                                                    # parked, weightless, out of sight
                    data.qpos[q0:q0 + 3] = (-900.0 + 3 * k, 40.0, 300.0)
                    data.qpos[q0 + 3:q0 + 7] = up
                    data.qvel[v0:v0 + 6] = 0.0
                    continue
            b = bid[k]
            p, R = data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy()
            vel = data.qvel[v0:v0 + 3].copy()
            omega = R @ data.qvel[v0 + 3:v0 + 6]                         # the free joint's angular velocity is in the body frame
            ff = 0.0
            for i in range(data.ncon):
                c = data.contact[i]
                if c.geom1 in foot_geoms[k] or c.geom2 in foot_geoms[k]:
                    f6 = np.zeros(6)
                    mujoco.mj_contactForce(model, data, i, f6)
                    ff += f6[0]
            force, torque = pl.command(t, p, vel, R, omega, w, dt, ff)
            data.xfrc_applied[b, :3], data.xfrc_applied[b, 3:] = force, torque
            if pl.dropped and not spawned[k]:
                spawned[k] = True
                _spawn_bag(model, data, scene, pl, p, vel, bag_q[k], bag_v[k])
                pl.lightened = car.payload_kg * G                        # the bag's weight is off it from here on
                pl.events["bag released"] = t
                judge.released(k, t)
            if pl.mode == "landed" and "landed" in pl.events and pl.r["mode"] == "land":
                judge.landed(k, t, p, R)
        mujoco.mj_step(model, data)
        t += dt
        if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            raise FloatingPointError(f"the simulation went unstable at t = {t:.2f} s: " + "; ".join(
                f"em{k} {pl.mode} thrust {pl.thrust:.1f} p {np.round(data.xpos[bid[k]], 1)} xfrc {np.round(data.xfrc_applied[bid[k]], 1)}" for k, pl in enumerate(pilots)))
        judge.step(t, model, data, bag_a, bag_b, spawned, dt)
        if t >= next_log:
            next_log += log_dt
            log["t"].append(t)
            log["qpos"].append(data.qpos.copy())
            log["mocap"].append(pa.copy())
            log["mode"].append([pl.mode for pl in pilots])
            for k, pl in enumerate(pilots):
                if released[k]:
                    tr = track[k]
                    tr["t"].append(t)
                    tr["p"].append(data.xpos[bid[k]].copy())
                    tr["v"].append(float(np.linalg.norm(data.qvel[jvd[k]:jvd[k] + 3])))
                    tr["thrust"].append(pl.thrust)
                    tr["energy"].append(pl.energy_wh)
        if progress and int(t / 10) != int((t - dt) / 10):
            progress(t, [pl.mode for pl in pilots])
        if all(pl.mode == "done" for pl in pilots) and judge.settled(t):
            break
    for tr in track.values():
        for key in tr:
            tr[key] = np.array(tr[key])
    return {"scene": scene, "model": model, "log": {k: (np.array(v) if k != "mode" else v) for k, v in log.items()}, "pilots": pilots,
            "track": track, "judge": judge, "deliveries": judge.results(pilots), "time_s": t, "release_times": t_rel}


def _spawn_bag(model, data, scene, pilot, p, vel, qadr, vadr):
    """The bag appears under the tail at the drop: the saddle bag spread across the ridge (its strap paid out), the
    box under the pod. It starts with the carrier's velocity."""
    car = scene.carrier
    z = p[2] - 0.52 * car.length_m - 0.4 * car.pod_l_m - 0.10
    if scene.payload == "saddle":
        yaw = pilot.r["ridge_yaw"] + math.pi / 2                         # the chain lies across the ridge
        span = 0.055 + 4 * 0.13 + 0.055
        c, s = math.cos(yaw), math.sin(yaw)
        data.qpos[qadr:qadr + 3] = (p[0] - c * span / 2, p[1] - s * span / 2, z)
        data.qpos[qadr + 3:qadr + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        nq_chain = 5 * 4                                                 # four links and the second pouch: ball joints
        data.qpos[qadr + 7:qadr + 7 + nq_chain] = np.tile((1.0, 0.0, 0.0, 0.0), 5)
        data.qvel[vadr:vadr + 6 + 15] = 0.0
    else:
        data.qpos[qadr:qadr + 3] = (p[0], p[1], z)
        data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[vadr:vadr + 6] = 0.0
    data.qvel[vadr:vadr + 3] = vel


class Judge:
    """What became of every bag, from the physics alone (see the module text)."""

    def __init__(self, scene: Scene, recipients: list):
        self.scene, self.recipients = scene, recipients
        n = len(recipients)
        self.t_release = [None] * n
        self.rest_since = [None] * n
        self.done = [None] * n
        self.shock = [0.0] * n
        self.prev_v = [None] * n
        self.window = [[] for _ in range(n)]
        self.land = [None] * n

    def released(self, k, t):
        self.t_release[k] = t

    def landed(self, k, t, p, R):
        if self.land[k] is None:
            self.land[k] = {"t": t, "p": p.copy(), "tilt_deg": math.degrees(math.acos(max(min(R[2, 2], 1.0), -1.0)))}

    def step(self, t, model, data, bag_a, bag_b, spawned, dt):
        import mujoco

        sc = self.scene
        for k, r in enumerate(self.recipients):
            if not spawned[k] or self.done[k] is not None:
                continue
            pa, pb = data.xpos[bag_a[k]], data.xpos[bag_b[k]]
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bag_a[k], vel6, 0)
            va = vel6[3:].copy()
            if self.prev_v[k] is not None:                               # deceleration over a 6 ms window
                self.window[k].append((va - self.prev_v[k]) / dt)
                self.window[k] = self.window[k][-3:]
                if len(self.window[k]) == 3 and t - self.t_release[k] > 0.05:
                    self.shock[k] = max(self.shock[k], float(np.linalg.norm(np.mean(self.window[k], axis=0) + (0, 0, G))) / G)
            self.prev_v[k] = va
            low = min(pa[2], pb[2])
            if low < sc.flood_m + 0.05:
                self.done[k] = {"t": t, "where": "in the water", "pa": pa.copy(), "pb": pb.copy()}
                continue
            if np.linalg.norm(va) < 0.04:
                self.rest_since[k] = self.rest_since[k] if self.rest_since[k] is not None else t
                if t - self.rest_since[k] > 1.5:
                    self.done[k] = {"t": self.rest_since[k], "where": "at rest", "pa": pa.copy(), "pb": pb.copy()}
            else:
                self.rest_since[k] = None

    def settled(self, t) -> bool:
        return all(self.done[k] is not None or self.t_release[k] is None for k in range(len(self.recipients)))

    def results(self, pilots) -> list:
        sc = self.scene
        out = []
        for k, r in enumerate(self.recipients):
            q, pl = r["building"], pilots[k]
            row = {"drone": k, "roof": f"{q.kind} at ({q.x:.0f}, {q.y:.0f})", "pitch [deg]": round(q.pitch_deg), "people": q.people,
                   "contents": r["contents"], "mode": "saddle-bag drop" if (r["mode"] == "drop" and sc.payload == "saddle") else
                   ("box drop" if r["mode"] == "drop" else "lands with it"), "released [s]": pl.events.get("released")}
            limit = SHOCK_LIMIT_G[r["contents"]]
            if r["mode"] == "drop":
                d = self.done[k]
                if d is None:
                    row.update({"outcome": "not dropped" if self.t_release[k] is None else "still moving", "delivered": False})
                else:
                    mid = 0.5 * (d["pa"] + d["pb"])
                    dist = float(np.hypot(mid[0] - r["people"][0], mid[1] - r["people"][1]))
                    ok = d["where"] == "at rest" and dist <= sc.reach_m and mid[2] > q.wall_h - 0.3
                    astride = sc.payload == "saddle" and d["where"] == "at rest" and _local_y(q, d["pa"]) * _local_y(q, d["pb"]) < 0
                    row.update({"outcome": ("astride the ridge" if astride else "on the roof") if d["where"] == "at rest" and mid[2] > q.wall_h - 0.3
                                else ("on the ground" if d["where"] == "at rest" else d["where"]),
                                "from the people [m]": round(dist, 2), "after the release from Albatros [s]": round(d["t"] - pl.events["released"], 1),
                                "shock [g]": round(self.shock[k], 1), "within the shock limit": self.shock[k] <= limit, "delivered": bool(ok)})
            else:
                ld = self.land[k]
                if ld is None:
                    row.update({"outcome": "did not land", "delivered": False})
                else:
                    dist = float(np.hypot(ld["p"][0] - r["people"][0], ld["p"][1] - r["people"][1]))
                    row.update({"outcome": "standing on the roof" if ld["tilt_deg"] < 20 else "fell over on the roof", "from the people [m]": round(dist, 2),
                                "after the release from Albatros [s]": round(ld["t"] - pl.events["released"], 1), "shock [g]": None,
                                "within the shock limit": True, "delivered": bool(dist <= 6.0)})
            row["carrier ends"] = {"done": "landed", "landed": "landed"}.get(pl.mode, pl.mode)
            row["energy used [Wh]"] = round(pl.energy_wh, 1)
            row["pack left [%]"] = round(100 * (1 - pl.energy_wh / (sc.carrier.pack_wh * sc.carrier.usable)))
            out.append(row)
        return out


def _local_y(q: Building, p) -> float:
    c, s = math.cos(math.radians(q.yaw_deg)), math.sin(math.radians(q.yaw_deg))
    return -s * (p[0] - q.x) + c * (p[1] - q.y)


def table(result: dict):
    import pandas as pd
    return pd.DataFrame(result["deliveries"]).set_index("drone")


def summary(result: dict) -> dict:
    d = result["deliveries"]
    ok = [r for r in d if r["delivered"] and r.get("within the shock limit", True)]
    times = [r["after the release from Albatros [s]"] for r in d if r.get("after the release from Albatros [s]") is not None]
    return {"roofs": len(d), "delivered": len(ok), "people reached": sum(r["people"] for r in ok), "people waiting": sum(r["people"] for r in d),
            "first delivery [s]": min(times) if times else None, "last delivery [s]": max(times) if times else None,
            "carriers landed": sum(r["carrier ends"] == "landed" for r in d), "most energy used [Wh]": max(r["energy used [Wh]"] for r in d),
            "payload": result["scene"].payload, "simulated [s]": round(result["time_s"], 1)}


# =========================================================================================================== pictures
def _camera(mujoco, lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def frame(result: dict, t: float, *, lookat=(0.0, 0.0, 6.0), distance=230.0, azimuth=115.0, elevation=-28.0, size=(1280, 720), renderer=None):
    """One picture of the logged run at time ``t`` (needs an OpenGL context: ``MUJOCO_GL=glfw`` under ``xvfb-run``)."""
    import mujoco

    model, log = result["model"], result["log"]
    i = min(int(np.searchsorted(log["t"], t)), len(log["t"]) - 1)
    data = mujoco.MjData(model)
    data.qpos[:] = log["qpos"][i]
    data.mocap_pos[model.body("albatros").mocapid[0]] = log["mocap"][i]
    mujoco.mj_forward(model, data)
    r = renderer or mujoco.Renderer(model, size[1], size[0])
    r.update_scene(data, _camera(mujoco, lookat, distance, azimuth, elevation))
    img = r.render().copy()
    if renderer is None:
        r.close()
    return img


def render_movie(result: dict, path, *, speed: float = 2.0, fps: int = 25, size=(1280, 720), follow: int | None = 0) -> str:
    """The run as an MP4 (through ``ffmpeg``): the pass seen from over the town, then — ``follow`` — one Émerillon
    from its release to its roof, then the town again while the others finish."""
    import subprocess

    import mujoco

    model, log, scene = result["model"], result["log"], result["scene"]
    t_end = float(log["t"][-1])
    t_rel = result["release_times"][0]
    tr = result["track"].get(follow) if follow is not None else None
    t_drop = result["pilots"][follow].events.get("bag released", result["pilots"][follow].events.get("landed", t_rel + 25)) if tr is not None else t_rel
    r = mujoco.Renderer(model, size[1], size[0])
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
                             "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "24", str(path)], stdin=subprocess.PIPE)
    t = max(t_rel - 6.0, 0.0)
    while t < t_end:
        if tr is not None and t_rel + 1.5 < t < t_drop + 7.0:
            i = min(int(np.searchsorted(tr["t"], t)), len(tr["t"]) - 1)
            p = tr["p"][i]
            near = t > t_drop - 6.0
            img = frame(result, t, lookat=p, distance=7.0 if near else 28.0, azimuth=60.0 + 12.0 * (t - t_rel), elevation=-18.0 if near else -25.0, size=size, renderer=r)
            step = 1.0 if near else speed
        else:
            img = frame(result, t, lookat=(0.0, 5.0, 8.0), distance=250.0, azimuth=110.0 + 2.0 * t, elevation=-27.0, size=size, renderer=r)
            step = speed
        proc.stdin.write(img.tobytes())
        t += step / fps
    proc.stdin.close()
    proc.wait()
    r.close()
    return str(path)


# =========================================================================================================== cold chain
def cold_chain(*, vial_kg: float = 0.25, pcm_kg: float = 0.10, pcm_c: float = 5.0, pcm_kj_kg: float = 200.0, wall_mm: float = 15.0,
               conductivity: float = 0.035, area_m2: float = 0.05, start_c: float = 5.0, phases=(("depot and loading", 20.0, 22.0),
               ("Albatros's bay at 6000 m", 63.0, -24.0), ("Émerillon to the roof", 2.0, 18.0), ("on the roof", 300.0, 24.0)), dt_s: float = 10.0) -> dict:
    """A vaccine box's temperature through the sortie: vials (as water) and a phase-change pack that freezes and melts
    at ``pcm_c``, behind ``wall_mm`` of foam, in the air of each phase (name, minutes, °C). Returns the history, the
    lowest and highest vial temperature and the time on the roof until it leaves 2-8 °C."""
    ua = conductivity / (wall_mm / 1000) * area_m2
    cap = vial_kg * 4186.0 + pcm_kg * 2000.0
    latent = pcm_kg * pcm_kj_kg * 1000.0
    T, melted = start_c, 0.5 * latent                                     # the pack starts half melted: room both ways
    t, hist = 0.0, {"t_min": [], "T": [], "air": [], "phase": []}
    left = None
    for name, minutes, air in phases:
        for _ in range(int(minutes * 60 / dt_s)):
            q = ua * (air - T) * dt_s
            if abs(T - pcm_c) < 0.02 or (T - pcm_c) * (T + q / cap - pcm_c) < 0:
                m2 = melted + q
                if 0.0 <= m2 <= latent:
                    melted, T, q = m2, pcm_c, 0.0
                else:
                    q = m2 - latent if m2 > latent else m2
                    melted = latent if m2 > latent else 0.0
            T += q / cap
            t += dt_s
            hist["t_min"].append(t / 60); hist["T"].append(T); hist["air"].append(air); hist["phase"].append(name)
            if left is None and name == phases[-1][0] and not 2.0 <= T <= 8.0:
                left = t / 60 - sum(p[1] for p in phases[:-1])
    Ts = np.array(hist["T"])
    return {**{k: np.array(v) if k != "phase" else v for k, v in hist.items()}, "ua_w_k": ua, "min_c": float(Ts.min()), "max_c": float(Ts.max()),
            "in_range_until_roof": bool(np.all((Ts[:int(sum(p[1] for p in phases[:-1]) * 60 / dt_s)] >= 2.0) & (Ts[:int(sum(p[1] for p in phases[:-1]) * 60 / dt_s)] <= 8.0))),
            "hours_on_roof_in_range": (left / 60) if left is not None else phases[-1][1] / 60, "box_kg": vial_kg + pcm_kg + 0.15}


__all__ = ["Carrier", "carrier", "bay_option", "Building", "Scene", "mjcf", "Pilot", "run", "Judge", "table", "summary", "frame", "render_movie",
           "cold_chain", "FLOOD_M", "SHOCK_LIMIT_G"]
