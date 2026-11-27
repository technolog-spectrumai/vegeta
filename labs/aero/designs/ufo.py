"""The alien pirates' two UFOs for the sea game - **fictional game craft**, flying the owner's alien Jolly Roger:

* the **flying saucer**, which *behaves like a helicopter*: it hovers, flies sideways and backwards, turns on the spot,
  and banks into a coordinated turn in forward flight;
* the **flying cigar**, which *behaves like a jet*: it cannot hover or fly below a minimum speed, turns by banking
  (its turn rate bounded by its load factor), climbs and dives within a pitch limit.

**No flight physics** (by the owner's direction): no lift, rotor or engine model. What is real is what the game needs
for these craft to look and play right:

* **speed and turn**: each craft's speed band, accelerations, climb and descent rates, bank and roll rate, yaw rate and
  the turn rate and radius those give at each speed - the class values of the aircraft each stands in for (a medium
  twin helicopter of a ~14 m rotor for the saucer; a light-to-midsize business jet or fast jet drone for the cigar),
  with their basis in ``BASIS``. ``step`` is the kinematic model that applies them: commanded velocities, heading and
  height, approached within the limits - a helicopter's or an aeroplane's way of moving, not its physics;
* **collisions**: each craft's collision shape as simple primitives (ellipsoids, capsule, box) that hug the CAD
  (``collision``), the same ones as the Yax rigid body's geoms, and the CAD meshes for exact checks;
* the **signatures** a sensor sees (radar cross-section by aspect, IR hot spots and skin heating, lights) and the
  figures for hunting them with flak (``flak``): how long each stays inside a gun's envelope on a typical pass, the
  angular rate the gun must track and the lead it must lay.

No weapons on either craft; no AIS (they are not ships, and they would not carry it). x forward, y left, z up; the
saucer's z = 0 its landing pads' feet, the cigar's z = 0 its axis; both centred on x = 0, y = 0.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None

G = 9.81
KN = 0.5144
CP_AIR = 1005.0
RECOVERY = 0.89                       # turbulent recovery factor: skin over ambient = r v^2 / (2 cp)

#: The two craft (class values of the aircraft each stands in for; basis in ``BASIS``).
CRAFT = {
    "saucer": {
        "what": "a 12 m flying saucer: a lens-shaped disc with a glass dome, a mast and flag on the dome, three landing pads; "
                "flies like a helicopter",
        "model": "helicopter",
        "diameter_m": 12.0, "rim_z_m": 1.15, "belly_z_m": 0.40, "deck_z_m": 1.90, "dome_r_m": 2.3, "dome_h_m": 1.5,
        "mast_h_m": 2.6, "flag_m": (1.5, 1.0), "pad_r_m": 0.35, "pad_h_m": 0.40, "pad_ring_m": 3.6,
        "mass_kg": 6400.0,
        "v_max_m_s": 78.0, "v_cruise_m_s": 68.0, "v_side_m_s": 18.0, "v_back_m_s": 15.0,
        "accel_m_s2": 3.0, "decel_m_s2": 4.0, "climb_m_s": 10.0, "descent_m_s": 10.0,
        "yaw_rate_deg_s": 40.0, "bank_max_deg": 45.0, "roll_rate_deg_s": 40.0, "pitch_max_deg": 20.0,
        "ceiling_m": 6000.0, "alt_band_m": {"nap_of_the_earth": [10.0, 30.0], "patrol": [100.0, 500.0], "transit": [500.0, 1500.0]},
        "rcs_m2": {"front": 4.0, "side": 6.0, "rear": 4.0, "top": 30.0, "bottom": 40.0},
        "ir": {"hot_spots": [{"what": "the belly's drive ring (where a helicopter's engine exhausts would be)", "T_K": 520.0, "area_m2": 0.8}],
               "skin_dT_K": 8.0},
        "lights": "a ring of 24 cyan lights round the rim, a white light on the mast; can go dark",
        "sound": "a low hum, about a medium helicopter's level at a distance (no rotor slap)",
        "crew": [2, 4],
    },
    "cigar": {
        "what": "an 18 m winged cigar (after the 1952 'winged cigar-shape' sketch): a smooth 2 m spindle, swept wings "
                "of 10 m span from mid-body, a dorsal fin with the flag on both sides, one glowing nozzle; flies like a jet",
        "model": "airplane",
        "length_m": 18.0, "diameter_m": 2.0, "nose_m": 4.5, "tail_m": 5.0, "nozzle_r_m": 0.45,
        "fin_root_m": 3.4, "fin_tip_m": 1.8, "fin_h_m": 2.2, "fin_t_m": 0.14, "fin_sweep_m": 1.4, "flag_m": (1.5, 1.0),
        "wing_span_m": 10.0, "wing_root_m": 5.0, "wing_tip_m": 1.2, "wing_sweep_deg": 50.0, "wing_t_m": 0.25,
        "wing_le_x_m": 1.0, "wing_z_m": -0.2,
        "mass_kg": 8500.0,
        "v_min_m_s": 70.0, "v_max_m_s": {"low": 215.0, "high": 245.0}, "v_cruise_m_s": {"low": 200.0, "high": 230.0},
        "v_max_alt_split_m": 3000.0,
        "accel_m_s2": 2.5, "decel_m_s2": 4.0, "climb_m_s": 25.0, "descent_m_s": 40.0,
        "n_max": {"low": 2.5, "high": 1.5}, "bank_max_deg": 67.0, "roll_rate_deg_s": 90.0, "pitch_max_deg": 25.0,
        "ceiling_m": 13000.0, "alt_band_m": {"low_pass": [30.0, 300.0], "high": [7000.0, 12000.0]},
        "rcs_m2": {"front": 1.5, "side": 10.0, "rear": 3.0, "top": 30.0, "bottom": 25.0},
        "ir": {"hot_spots": [{"what": "the tail nozzle (a small turbofan's exhaust)", "T_K": 700.0, "area_m2": 0.6}],
               "skin_dT_K": None},
        "lights": "a red glow at the nozzle, no navigation lights",
        "sound": "a jet's roar on a low pass, heard after it has gone by",
        "crew": [1, 3],
    },
}

#: Where each class value comes from - the real aircraft each craft stands in for (no vendor's aircraft; the class).
BASIS = {
    "saucer": "sized as a medium twin helicopter of a 13-14 m rotor (the AW139 / Bell 412 class): MTOW ~6.4 t; max "
              "speed ~78 m/s (150 kn) and cruise ~68 m/s (130 kn); sideward and rearward flight limits 35 and 30 kn; "
              "climb ~10 m/s; a hover yaw rate of ~40 deg/s; 45 deg bank, 20 deg pitch in manoeuvres; accelerating at "
              "~0.3 g (nose ~17 deg down), flaring at ~0.4 g. RCS of a helicopter that size (a few m2 nose-on); IR of its "
              "exhausts (~500 K) put at the belly. The disc, dome and its flight are fiction.",
    "cigar": "sized as a light-to-midsize business jet or a fast jet drone: 15-20 m long, MTOW ~8.5 t; Mach 0.6-0.8 "
             "(215 m/s low, 245 m/s high); a minimum speed of ~70 m/s (its stall, clean); load factor 2.5 at low level "
             "(transport-category limit), 1.5 sustainable near its ceiling; roll ~90 deg/s; climb ~25 m/s; thrust/weight "
             "~0.35 (accelerating ~2.5 m/s2 at low speed). RCS of a bizjet (~1 m2 nose-on, ~10 beam); IR of a small "
             "turbofan's nozzle (~700 K). The winged spindle and its flight are fiction.",
}

#: Flak the game may give the guards (class envelopes; **game assumptions**, not any vessel's fit in this repo).
FLAK = {
    "hmg_12_7": {"what": "a 12.7 mm heavy machine gun", "range_m": 1500.0, "muzzle_m_s": 890.0, "mean_m_s": 650.0, "rate_rpm": 600},
    "ciws_30": {"what": "a 30 mm cannon (CIWS class)", "range_m": 2500.0, "muzzle_m_s": 1080.0, "mean_m_s": 850.0, "rate_rpm": 600},
    "aa_35": {"what": "a twin 35 mm anti-aircraft gun", "range_m": 4000.0, "muzzle_m_s": 1175.0, "mean_m_s": 900.0, "rate_rpm": 1100},
}
#: The passes the flak table is worked for: speed, height, horizontal miss distance from the gun.
PASSES = {
    "saucer": [{"what": "patrol pass", "v_m_s": 68.0, "alt_m": 150.0},
               {"what": "nap-of-the-earth dash", "v_m_s": 78.0, "alt_m": 20.0}],
    "cigar": [{"what": "low pass", "v_m_s": 200.0, "alt_m": 100.0},
              {"what": "high transit", "v_m_s": 230.0, "alt_m": 9000.0}],
}
OFFSETS_M = (0.0, 500.0, 1000.0, 2000.0)


# ----------------------------------------------------------------------------------------------------- kinematics
def _limits(name: str, alt_m: float = 0.0) -> dict:
    c = CRAFT[name]
    if c["model"] == "helicopter":
        return {"v_min": 0.0, "v_max": c["v_max_m_s"], "n_max": 1 / math.cos(math.radians(c["bank_max_deg"]))}
    band = "low" if alt_m < c["v_max_alt_split_m"] else "high"
    return {"v_min": c["v_min_m_s"], "v_max": c["v_max_m_s"][band], "n_max": c["n_max"][band]}


def turn(name: str, v: float, alt_m: float = 0.0) -> dict:
    """The tightest steady turn at ``v`` (m/s): the bank, turn rate (deg/s), radius (m) and time for 180 deg.
    A helicopter: banked to its limit, or - slow - turning on the spot at its yaw rate. An aeroplane: at its load factor
    (bank = acos(1/n)), never below its minimum speed."""
    c, lim = CRAFT[name], _limits(name, alt_m)
    v = max(v, lim["v_min"])
    bank_n = math.degrees(math.acos(1 / lim["n_max"]))
    bank = min(c["bank_max_deg"], bank_n)
    w_bank = G * math.tan(math.radians(bank)) / v if v > 0 else math.inf
    w = min(math.radians(c.get("yaw_rate_deg_s", 1e9)), w_bank)
    if c["model"] == "helicopter" and w < w_bank:                  # slow: yaw-rate limited, banked only as the rate needs
        bank = math.degrees(math.atan(v * w / G))
    return {"v_m_s": round(v, 1), "bank_deg": round(bank, 1), "rate_deg_s": round(math.degrees(w), 2),
            "radius_m": round(v / w, 1) if v > 0 else 0.0, "t_180_s": round(math.pi / w, 1)}


def kinematics(name: str) -> dict:
    """The card's kinematic block: the limits ``step`` applies, and the turn by speed."""
    c = CRAFT[name]
    keys = ("model", "accel_m_s2", "decel_m_s2", "climb_m_s", "descent_m_s", "bank_max_deg", "roll_rate_deg_s", "pitch_max_deg",
            "ceiling_m", "alt_band_m")
    k = {q: c[q] for q in keys}
    if c["model"] == "helicopter":
        k |= {"hover": True, "v_max_m_s": c["v_max_m_s"], "v_cruise_m_s": c["v_cruise_m_s"], "v_side_m_s": c["v_side_m_s"],
              "v_back_m_s": c["v_back_m_s"], "yaw_rate_deg_s": c["yaw_rate_deg_s"]}
        speeds = (0.0, 10.0, 20.0, 30.0, 50.0, c["v_cruise_m_s"], c["v_max_m_s"])
        k["turn_by_speed"] = [turn(name, v) for v in speeds]
    else:
        k |= {"hover": False, "v_min_m_s": c["v_min_m_s"], "v_max_m_s": c["v_max_m_s"], "v_cruise_m_s": c["v_cruise_m_s"],
              "v_max_alt_split_m": c["v_max_alt_split_m"], "n_max": c["n_max"]}
        k["turn_by_speed"] = [turn(name, v) | {"alt": "low"} for v in (c["v_min_m_s"], 100.0, 150.0, c["v_cruise_m_s"]["low"], c["v_max_m_s"]["low"])] + \
                             [turn(name, v, 9000.0) | {"alt": "high"} for v in (150.0, c["v_cruise_m_s"]["high"], c["v_max_m_s"]["high"])]
    return k


def _approach(x, target, up, down, dt):
    return min(x + up * dt, target) if target > x else max(x - down * dt, target)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def new_state(name: str, x=0.0, y=0.0, z=100.0, heading_deg=0.0, speed=None) -> dict:
    """A craft's kinematic state: position (m), heading (rad, from +x toward +y), body velocities u forward, v left,
    w up (m/s), yaw rate r (rad/s, + to the left), and the attitude the game draws - derived, not dynamics - in the
    aviation convention: roll + right wing down, pitch + nose up (rad)."""
    c = CRAFT[name]
    if speed is None:
        speed = 0.0 if c["model"] == "helicopter" else c["v_cruise_m_s"]["low"]
    return {"x": x, "y": y, "z": z, "psi": math.radians(heading_deg), "u": speed, "v": 0.0, "w": 0.0, "roll": 0.0, "pitch": 0.0, "r": 0.0}


def step(name: str, s: dict, cmd: dict, dt: float) -> dict:
    """One kinematic step (``dt`` s) toward a command - the way a helicopter or an aeroplane moves, within the craft's
    limits, without its physics. Returns the new state (``s`` is not changed).

    Helicopter (saucer) command: ``u``, ``v`` (body forward/left m/s), ``w`` (climb m/s), and ``heading_deg`` or
    ``yaw_rate_deg_s``. Speeds clip to the forward/side/back limits; the heading turns at the yaw rate when slow, at the
    banked turn's rate (g tan bank / speed) when fast; the attitude follows the acceleration (pitch) and the turn (roll).

    Aeroplane (cigar) command: ``speed`` (m/s), ``heading_deg``, ``alt_m``. The speed stays in [v_min, v_max(alt)]; the
    bank rolls at the roll rate toward what the heading error asks, within the load factor's bank; the turn rate is
    g tan(bank) / speed; the climb is the height error's, within the climb/descent rates and the pitch limit."""
    c = CRAFT[name]
    n = dict(s)
    if c["model"] == "helicopter":
        u_t = min(max(cmd.get("u", 0.0), -c["v_back_m_s"]), c["v_max_m_s"])
        v_t = min(max(cmd.get("v", 0.0), -c["v_side_m_s"]), c["v_side_m_s"])
        w_t = min(max(cmd.get("w", 0.0), -c["descent_m_s"]), c["climb_m_s"])
        acc = lambda x, t: _approach(x, t, c["accel_m_s2"] if abs(t) > abs(x) else c["decel_m_s2"],
                                     c["accel_m_s2"] if abs(t) > abs(x) else c["decel_m_s2"], dt)
        n["u"], n["v"], n["w"] = acc(s["u"], u_t), acc(s["v"], v_t), acc(s["w"], w_t)
        n["w"] = min(n["w"], 0.0) if s["z"] >= c["ceiling_m"] else n["w"]
        sp = math.hypot(n["u"], n["v"])
        r_max = math.radians(turn(name, sp)["rate_deg_s"])
        if "heading_deg" in cmd:
            err = _wrap(math.radians(cmd["heading_deg"]) - s["psi"])
            r_t = max(-r_max, min(r_max, err / max(dt, 1e-3), 2.0 * err))
        else:
            r_t = max(-r_max, min(r_max, math.radians(cmd.get("yaw_rate_deg_s", 0.0))))
        n["r"] = r_t
        ax = (n["u"] - s["u"]) / dt if dt else 0.0
        n["pitch"] = -math.atan(ax / G) if sp > 0.5 or abs(ax) > 0 else 0.0
        roll_t = -(math.atan(sp * r_t / G) + math.atan(((n["v"] - s["v"]) / dt if dt else 0.0) / G))   # right wing down +
        n["roll"] = _approach(s["roll"], roll_t, math.radians(c["roll_rate_deg_s"]), math.radians(c["roll_rate_deg_s"]), dt)
    else:
        lim = _limits(name, s["z"])
        sp_t = min(max(cmd.get("speed", s["u"]), lim["v_min"]), lim["v_max"])
        n["u"] = max(min(_approach(s["u"], sp_t, c["accel_m_s2"], c["decel_m_s2"], dt), lim["v_max"]), lim["v_min"])
        n["v"] = 0.0
        bank_max = math.radians(min(c["bank_max_deg"], math.degrees(math.acos(1 / lim["n_max"]))))
        err = _wrap(math.radians(cmd.get("heading_deg", math.degrees(s["psi"]))) - s["psi"])
        r_want = 0.5 * err                                             # a turn of 0.5 rad/s per rad of error, then bank for it
        bank_t = max(-bank_max, min(bank_max, -math.atan(n["u"] * r_want / G)))    # a right turn (r < 0): right wing down
        rr = math.radians(c["roll_rate_deg_s"])
        n["roll"] = _approach(s["roll"], bank_t, rr, rr, dt)
        n["r"] = -G * math.tan(n["roll"]) / n["u"]
        dz = cmd.get("alt_m", s["z"]) - s["z"]
        w_lim_up = min(c["climb_m_s"], n["u"] * math.sin(math.radians(c["pitch_max_deg"])))
        w_lim_dn = min(c["descent_m_s"], n["u"] * math.sin(math.radians(c["pitch_max_deg"])))
        w_t = max(-w_lim_dn, min(w_lim_up, 0.2 * dz))
        n["w"] = _approach(s["w"], w_t, G * 0.5, G * 0.5, dt)       # at most a 0.5 g pull to change the flight path
        n["w"] = min(n["w"], 0.0) if s["z"] >= c["ceiling_m"] else n["w"]
        n["pitch"] = math.asin(max(-1.0, min(1.0, n["w"] / n["u"])))
    n["psi"] = _wrap(s["psi"] + n["r"] * dt)
    ch, sh = math.cos(n["psi"]), math.sin(n["psi"])
    n["x"] = s["x"] + (n["u"] * ch - n["v"] * sh) * dt
    n["y"] = s["y"] + (n["u"] * sh + n["v"] * ch) * dt
    n["z"] = max(0.0, s["z"] + n["w"] * dt)
    return n


# ----------------------------------------------------------------------------------------------------- signatures
def skin_dT_K(v: float) -> float:
    """The skin's rise over the ambient air from flying at ``v`` (adiabatic wall, turbulent recovery)."""
    return RECOVERY * v * v / (2 * CP_AIR)


def signatures(name: str) -> dict:
    c = CRAFT[name]
    v_fast = c["v_max_m_s"] if c["model"] == "helicopter" else c["v_max_m_s"]["low"]
    ir = dict(c["ir"])
    ir["skin_dT_K"] = c["ir"]["skin_dT_K"] if c["ir"]["skin_dT_K"] is not None else round(skin_dT_K(v_fast), 1)
    ir["skin_dT_note"] = ("a warm hull from the drive inside it (class)" if c["ir"]["skin_dT_K"] is not None
                          else f"aerodynamic heating at {v_fast:g} m/s: r v^2 / 2cp")
    return {"rcs_m2": c["rcs_m2"], "ir": ir, "lights": c["lights"], "sound": c["sound"], "ais": None}


# ----------------------------------------------------------------------------------------------------- flak
def pass_through(v: float, alt_m: float, offset_m: float, gun: dict) -> dict:
    """A straight, level pass at ``v`` and ``alt_m`` whose ground track misses the gun by ``offset_m``: the time inside
    the gun's slant range, the closest slant range, the tracking rate there (deg/s) and the lead the gun lays there (m
    ahead of the craft, and as an angle: the shell's mean speed over the range)."""
    r_min = math.hypot(alt_m, offset_m)
    R = gun["range_m"]
    if r_min >= R:
        return {"in_range_s": 0.0, "closest_m": round(r_min), "track_deg_s": None, "lead_m": None, "lead_deg": None}
    half = math.sqrt(R * R - r_min * r_min)
    tof = r_min / gun["mean_m_s"]
    return {"in_range_s": round(2 * half / v, 1), "closest_m": round(r_min), "track_deg_s": round(math.degrees(v / r_min), 1),
            "lead_m": round(v * tof, 1), "lead_deg": round(math.degrees(math.atan(v * tof / r_min)), 1),
            "rounds_in_range": round(2 * half / v * gun["rate_rpm"] / 60)}


def flak(name: str) -> dict:
    """The flak table: for each typical pass and each gun, by miss distance, how long the craft is in range and what
    the gun must do; and the hover (the saucer only: as long as it stays)."""
    out = {"guns": FLAK, "passes": []}
    for p in PASSES[name]:
        rows = {g: {f"{o:g}m": pass_through(p["v_m_s"], p["alt_m"], o, gun) for o in OFFSETS_M} for g, gun in FLAK.items()}
        out["passes"].append(p | {"by_gun": rows})
    if CRAFT[name]["model"] == "helicopter":
        out["hover"] = "a hovering saucer is in range as long as it stays: tracking ~0 deg/s, no lead - its danger is the hover"
    else:
        lim = _limits(name, 9000.0)
        out["high_band"] = (f"above {PASSES[name][1]['alt_m']:g} m it is out of every gun's reach "
                            f"(the longest, {max(g['range_m'] for g in FLAK.values()):g} m)")
        out["min_speed_m_s"] = lim["v_min"]
    return out


# ----------------------------------------------------------------------------------------------------- collisions
def collision(name: str) -> list:
    """The collision shape: primitives in the craft's frame (m; MJCF sizes: an ellipsoid's three radii, a capsule's
    radius and half-length along its axis, a box's half-sizes; ``quat`` (w, x, y, z) turns a box, as the cigar's swept
wings'). They hug the CAD's surfaces; the flag is visual only."""
    c = CRAFT[name]
    if name == "saucer":
        R = c["diameter_m"] / 2
        disc_h = (c["deck_z_m"] - c["belly_z_m"]) / 2
        out = [{"name": "disc", "type": "ellipsoid", "size": [R, R, disc_h], "pos": [0.0, 0.0, c["belly_z_m"] + disc_h]},
               {"name": "dome", "type": "ellipsoid", "size": [c["dome_r_m"], c["dome_r_m"], c["dome_h_m"]], "pos": [0.0, 0.0, c["deck_z_m"]]},
               {"name": "mast", "type": "capsule", "size": [0.06, c["mast_h_m"] / 2], "pos": [0.0, 0.0, c["deck_z_m"] + c["dome_h_m"] + c["mast_h_m"] / 2],
                "axis": [0, 0, 1]}]
        for k in range(3):
            a = math.radians(90 + 120 * k)
            out.append({"name": f"pad_{k}", "type": "cylinder", "size": [c["pad_r_m"], c["pad_h_m"] / 2],
                        "pos": [round(c["pad_ring_m"] * math.cos(a), 3), round(c["pad_ring_m"] * math.sin(a), 3), c["pad_h_m"] / 2], "axis": [0, 0, 1]})
        return out
    L, r = c["length_m"], c["diameter_m"] / 2
    chord = max(c["fin_root_m"], c["fin_sweep_m"] + c["fin_tip_m"])          # the swept fin's whole x extent
    tail_r = c["nozzle_r_m"] + 0.35                                   # the boat-tail's last 2 m and the nozzle
    return [{"name": "body", "type": "capsule", "size": [r, L / 2 - r], "pos": [0.0, 0.0, 0.0], "axis": [1, 0, 0]},
            {"name": "tail", "type": "cylinder", "size": [tail_r, 1.05], "pos": [-L / 2 + 1.0, 0.0, 0.0], "axis": [1, 0, 0]},
            {"name": "fin", "type": "box", "size": [chord / 2, c["fin_t_m"] / 2, c["fin_h_m"] / 2],
             "pos": [round(_fin_le_x(c) - chord / 2, 3), 0.0, round(_fin_base_z(c) + c["fin_h_m"] / 2, 3)]},
            _wing_box(c, 1), _wing_box(c, -1)]


def wing_planform(c, side: int = 1, y_root: float = 0.0) -> list:
    """One wing's planform (x, y) in m, corners LE root, LE tip, TE tip, TE root; ``side`` +1 port (y > 0), -1
    starboard. The leading edge swept ``wing_sweep_deg`` from ``wing_le_x_m`` at the axis."""
    t = math.tan(math.radians(c["wing_sweep_deg"]))
    b = c["wing_span_m"] / 2
    xr = c["wing_le_x_m"] - y_root * t
    xt = c["wing_le_x_m"] - b * t
    return [(xr, side * y_root), (xt, side * b), (xt - c["wing_tip_m"], side * b), (xr - c["wing_root_m"] + y_root * (c["wing_root_m"] - c["wing_tip_m"]) / b, side * y_root)]


def _wing_box(c, side: int) -> dict:
    """The box that holds one wing outside the body: along the mid-chord line, turned about z (``quat``)."""
    pts = wing_planform(c, side, y_root=c["diameter_m"] / 2 * 0.7)
    (x0, y0), (x1, y1) = ((pts[0][0] + pts[3][0]) / 2, pts[0][1]), ((pts[1][0] + pts[2][0]) / 2, pts[1][1])
    d = (x1 - x0, y1 - y0)
    L = math.hypot(*d)
    ux, uy = d[0] / L, d[1] / L
    along = [(px - x0) * ux + (py - y0) * uy for px, py in pts]
    across = [-(px - x0) * uy + (py - y0) * ux for px, py in pts]
    a0, a1, c0, c1 = min(along), max(along), min(across), max(across)
    cx = x0 + ux * (a0 + a1) / 2 - uy * (c0 + c1) / 2
    cy = y0 + uy * (a0 + a1) / 2 + ux * (c0 + c1) / 2
    yaw = math.atan2(uy, ux)
    return {"name": f"wing_{'port' if side > 0 else 'starboard'}", "type": "box",
            "size": [round((a1 - a0) / 2, 3), round((c1 - c0) / 2, 3), round(c["wing_t_m"] / 2, 3)],
            "pos": [round(cx, 3), round(cy, 3), c["wing_z_m"]], "quat": [round(math.cos(yaw / 2), 6), 0.0, 0.0, round(math.sin(yaw / 2), 6)]}


def _fin_base_z(c):
    """The fin's foot: where it meets the body's top (a little below the top, inside the hull)."""
    return c["diameter_m"] / 2 * 0.75


def _fin_le_x(c):
    """The fin's root leading edge (m): it stands over the start of the boat-tail."""
    return -c["length_m"] / 2 + c["tail_m"] * 0.2 + c["fin_root_m"]


def bounding_radius(name: str) -> float:
    """The radius of the sphere round the craft's centre that holds it: the broad-phase test."""
    c = CRAFT[name]
    if name == "saucer":
        h = c["deck_z_m"] + c["dome_h_m"] + c["mast_h_m"]
        return round(math.hypot(c["diameter_m"] / 2, h / 2), 2)
    return round(max(math.hypot(c["length_m"] / 2, _fin_base_z(c) + c["fin_h_m"]),
                     max(math.hypot(x, y) for x, y in wing_planform(c)) + 0.1), 2)


# ----------------------------------------------------------------------------------------------------- flag
def flag_frame(name: str) -> dict:
    """Where the flag panel sits (m): its hoist-and-foot corner, its fly direction (u: hoist -> fly) and its up
    direction (v), its size; the texture's (0, 0) at the foot of the hoist, (1, 1) at the top of the fly."""
    c = CRAFT[name]
    w, h = c["flag_m"]
    if name == "saucer":
        top = c["deck_z_m"] + c["dome_h_m"] + c["mast_h_m"]
        return {"origin": [-0.06, 0.0, top - h - 0.05], "u": [-1.0, 0.0, 0.0], "v": [0.0, 0.0, 1.0], "size": [w, h],
                "faces": "both", "where": "on the mast atop the dome, flying aft"}
    return {"origin": [round(_fin_le_x(c) - 1.0, 3), 0.0, round(_fin_base_z(c) + 0.45, 3)], "u": [-1.0, 0.0, 0.0], "v": [0.0, 0.0, 1.0],
            "size": [w, h], "faces": "both", "offset_m": c["fin_t_m"] / 2 + 0.005, "where": "painted on both sides of the dorsal fin"}


def flag_mesh(name: str, texture: str = "alien_jolly_roger.png") -> dict:
    """The flag as a textured model: two quads (the faces), each with UVs so the image reads correctly from its side
    (hoist toward the nose on both: a painted or printed flag, not a cloth one seen through). Metres; ``uv`` per vertex."""
    f = flag_frame(name)
    (ox, oy, oz), (w, h) = f["origin"], f["size"]
    off = f.get("offset_m", 0.01)
    verts, uv, tris = [], [], []
    for side in (1, -1):                                             # +y (port) face, -y (starboard) face
        y = oy + side * off
        base = len(verts) // 3
        for (a, b) in ((0, 0), (1, 0), (1, 1), (0, 1)):
            verts += [round(ox - a * w, 4), round(y, 4), round(oz + b * h, 4)]
            uv += [a, b]
        # outward normals: +y face wound (0,1,2)/(0,2,3) seen from +y with u running -x; flip for -y
        q = [(0, 2, 1), (0, 3, 2)] if side == 1 else [(0, 1, 2), (0, 2, 3)]
        for t in q:
            tris += [base + i for i in t]
    return {"name": f"{name}_flag", "source": "vegeta labs/aero/designs/ufo.py (flag panel)", "closed": False,
            "color": [1.0, 1.0, 1.0], "size": [w, round(2 * off, 3), h], "vertices": verts, "triangles": tris,
            "uv": uv, "texture": texture}


# ----------------------------------------------------------------------------------------------------- CAD
def _revolve_z(prof):
    """A solid of revolution about z from a closed (r, z) profile in mm."""
    return cq.Workplane("XZ").polyline(prof).close().revolve(360, (0, 0, 0), (0, 1, 0))


def saucer_profile(n: int = 24) -> list:
    """The saucer's lens (r, z) in m, from the belly's centre out to the rim and back over the top to the deck."""
    c = CRAFT["saucer"]
    R = c["diameter_m"] / 2
    pts = []
    for i in range(n + 1):                                            # belly: a shallow curve from the centre to the rim
        r = R * i / n
        pts.append((r, c["belly_z_m"] + (c["rim_z_m"] - c["belly_z_m"]) * (r / R) ** 2.2))
    for i in range(n - 1, -1, -1):                                    # top: up from the rim to the deck under the dome
        r = R * i / n
        pts.append((r, c["deck_z_m"] - (c["deck_z_m"] - c["rim_z_m"]) * (r / R) ** 1.8))
    return pts


def cigar_profile(n: int = 20) -> list:
    """The cigar's (x, r) in m from the nose (x = +L/2) to the nozzle (x = -L/2): an ogive nose, a straight body,
    a boat-tail to the nozzle."""
    c = CRAFT["cigar"]
    L, R, ln, lt, rn = c["length_m"], c["diameter_m"] / 2, c["nose_m"], c["tail_m"], c["nozzle_r_m"]
    pts = []
    for i in range(n + 1):
        s = i / n
        pts.append((L / 2 - ln * s, R * math.sqrt(1 - (1 - s) ** 2)))                  # elliptic nose, tip to shoulder
    for i in range(1, n + 1):
        s = i / n
        pts.append((-L / 2 + lt * (1 - s), R - (R - rn) * s ** 1.6))                    # boat-tail
    return pts


def build(name: str, part: str = "craft"):
    """The craft's CAD (mm): ``part`` "hull" (saucer: disc + pads; cigar: body), "dome" / "fin", "mast" (saucer),
    "nozzle" (cigar), or "craft" (everything but the flag, which is a textured panel: ``flag_mesh``)."""
    if cq is None:
        raise ImportError("the UFOs' CAD needs CadQuery")
    c = CRAFT[name]
    mm = 1000.0
    if name == "saucer":
        def hull():
            prof = [(r * mm, z * mm) for r, z in saucer_profile()]
            prof[0] = (0.0, prof[0][1]); prof[-1] = (0.0, prof[-1][1])
            disc = _revolve_z(prof)
            for k in range(3):
                a = math.radians(90 + 120 * k)
                x, y = c["pad_ring_m"] * math.cos(a) * mm, c["pad_ring_m"] * math.sin(a) * mm
                leg = cq.Workplane("XY").add(cq.Solid.makeCylinder(c["pad_r_m"] * mm, (c["belly_z_m"] + 0.25) * mm, cq.Vector(x, y, 0)))
                disc = disc.union(leg)
            return disc

        def dome():
            rr, hh = c["dome_r_m"] * mm, c["dome_h_m"] * mm
            prof = [(rr * math.cos(t), c["deck_z_m"] * mm + hh * math.sin(t)) for t in [math.pi / 2 * i / 16 for i in range(17)]]
            prof = [(0.0, c["deck_z_m"] * mm - 40)] + [(rr, c["deck_z_m"] * mm - 40)] + prof
            prof[-1] = (0.0, prof[-1][1])
            return _revolve_z(prof)

        def mast():
            z0 = (c["deck_z_m"] + c["dome_h_m"] - 0.05) * mm
            return cq.Workplane("XY").add(cq.Solid.makeCylinder(60, (c["mast_h_m"] + 0.05) * mm, cq.Vector(0, 0, z0)))
        parts = {"hull": hull, "dome": dome, "mast": mast}
    else:
        def hull():
            prof = [(x * mm, r * mm) for x, r in cigar_profile()]
            prof = [(prof[0][0], 0.0)] + prof[1:] + [(prof[-1][0], 0.0)]
            # revolve the (x, r) profile about the x axis
            return cq.Workplane("XY").polyline(prof).close().revolve(360, (0, 0, 0), (1, 0, 0))

        def fin():
            x_le = _fin_le_x(c) * mm
            zb, h, sw = _fin_base_z(c) * mm - 150, c["fin_h_m"] * mm + 150, c["fin_sweep_m"] * mm
            pts = [(x_le, zb), (x_le - sw, zb + h), (x_le - sw - c["fin_tip_m"] * mm, zb + h), (x_le - c["fin_root_m"] * mm, zb)]
            return cq.Workplane("XZ").polyline(pts).close().extrude(c["fin_t_m"] * mm / 2, both=True)

        def nozzle():
            L = c["length_m"]
            return cq.Workplane("YZ").workplane(offset=(-L / 2 - 0.05) * 1000).circle(c["nozzle_r_m"] * 0.85 * mm).extrude(80)
        def wings():
            out = None
            for side in (1, -1):
                pts = [(x * mm, y * mm) for x, y in wing_planform(c, side)]
                w = (cq.Workplane("XY").workplane(offset=(c["wing_z_m"] - c["wing_t_m"] / 2) * mm)
                     .polyline(pts).close().extrude(c["wing_t_m"] * mm))
                out = w if out is None else out.union(w)
            return out
        parts = {"hull": hull, "wings": wings, "fin": fin, "nozzle": nozzle}
    if part in parts:
        return parts[part]()
    if part != "craft":
        raise KeyError(part)
    out = None
    for f in parts.values():
        out = f() if out is None else out.union(f())
    return out


PART_COLOURS = {"saucer": {"hull": "#8e9aa6", "dome": "#5dade2", "mast": "#2c3e50"},
                "cigar": {"hull": "#7b7d7d", "wings": "#5d6d7e", "fin": "#1b2631", "nozzle": "#e74c3c"}}


def sizing(name: str) -> dict:
    """Everything the card carries, for one craft."""
    c = CRAFT[name]
    if name == "saucer":
        size = [c["diameter_m"], c["diameter_m"], round(c["deck_z_m"] + c["dome_h_m"] + c["mast_h_m"], 2)]
    else:
        size = [c["length_m"], c["wing_span_m"], round(c["diameter_m"] / 2 + _fin_base_z(c) + c["fin_h_m"], 2)]
    return {"name": name, "what": c["what"], "fictional": True, "size": size, "mass_kg": c["mass_kg"],
            "kinematics": kinematics(name), "collision": collision(name), "bounding_radius_m": bounding_radius(name),
            "signature": signatures(name), "flak": flak(name), "flag": flag_frame(name) | {"texture": "alien_jolly_roger.png"},
            "crew": c["crew"], "basis": BASIS[name]}


if __name__ == "__main__":
    import json
    for n in CRAFT:
        s = sizing(n)
        print(n, json.dumps(s["kinematics"]["turn_by_speed"]))
        for p in s["flak"]["passes"]:
            print("  ", p["what"], {g: {o: r["in_range_s"] for o, r in rows.items()} for g, rows in p["by_gun"].items()})
