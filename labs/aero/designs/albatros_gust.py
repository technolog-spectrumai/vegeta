"""ALBATROS against the wind (notebook 50): the wing in gusts, the hover over a moving deck, the stand on that deck
and the float on calm water — the numbers that choose the layout beside the fuel.

**The wing in the cruise.** A sharp-edged vertical gust ``U`` changes the angle of attack by ``U / V`` and the lift by
``rho V U a S / 2``: the same force on any aircraft of that wing, so the load factor it gives is inversely
proportional to the wing loading, ``dn = K_g rho U V a / (2 W/S)`` (Pratt's formula, ``K_g = 0.88 mu / (5.3 + mu)``
with the mass ratio ``mu = 2 (W/S) / (rho c a g)``; ``gust_load``). The lift slope ``a`` falls with sweep and with a
lower aspect ratio (``lift_slope``, the DATCOM formula). A vertical take-off removes the stall-speed limit that makes
a runway or catapult aircraft carry a big wing — that is the "special wing": small and heavily loaded.
``gust_alleviation`` then flies a 1-cos gust in heave with full-span flaperons as direct lift control, driven by the
angle of attack measured at a nose probe (which meets the gust ``probe_lead_m / V`` before the wing does) through
an actuator lag: what an active wing takes out of what is left.

**Thrust vectoring** gives a moment that does not depend on the airspeed, ``T sin(delta) l``; the tail's falls with
the dynamic pressure. ``vectoring_authority`` gives both against speed and the speed below which the nozzles are
the stronger control — the transition and the hover, where the tail and the flaperons have nothing to work with.

**The hover over the deck** (``hover_controls``, ``hover_gust``, ``deck_heave``). Each layout has a way to make a
horizontal force and a way to change its vertical force, each with a size and a lag:

- ``lift_cruise``: it tilts on its rotors (differential thrust; attitude follows in ~0.3 s), and the rotors change
  their thrust in ~0.1 s with 40 % in hand;
- ``vectored``: the nozzles swing (fast) and the body tilts (slower); the thrust follows the spool (a lag of
  ``THRUST_LAG`` x the idle-to-full time, near full speed) — but the nozzles, held a little splayed, trade splay for vertical force at once;
- ``tailsitter``: the nozzles tilt the whole aircraft (a force at the tail first pushes it the wrong way); its wing
  stands broadside to the wind; the thrust follows the spool, with the same splay trick.

``hover_gust`` holds position (one axis, a PD loop no faster than the actuator allows) in a steady wind over the deck
plus a 1-cos gust on the layout's own drag area; ``deck_heave`` follows a heaving deck with the vertical force.

**On the deck** (``deck_stand``): the wind and the ship's roll against the footprint — the tail-sitter stands as tall
as it is long. **On the water** (``afloat``): draft, freeboard to the intakes and the rotors, and the metacentric
height of the sealed rounded-box hull.

Limits: one axis at a time, linear actuators with a lag and a limit, quasi-steady aerodynamics (Pratt's factor for
the unsteady lift), no ground effect, no hot-gas re-ingestion, no ship airwake model beyond a gust on a wind.
"""
from __future__ import annotations

import math

import numpy as np

from ogma.units import G
from ogma.units import RHO0 as RHO

SEA_WATER = 1025.0
THRUST_LAG = 0.125                 # a jet's thrust lag near full speed / its idle-to-full time (most of that time is spent low)


# ------------------------------------------------------------------------------------------------ the wing
def lift_slope(aspect_ratio: float, sweep_deg: float = 0.0, mach: float = 0.0, eta: float = 0.95) -> float:
    """Lift-curve slope [1/rad] of a wing (DATCOM): ``2 pi AR / (2 + sqrt(4 + AR² beta² / eta² (1 + tan² sweep / beta²)))``."""
    b2 = 1 - mach ** 2
    t = math.tan(math.radians(sweep_deg))
    return 2 * math.pi * aspect_ratio / (2 + math.sqrt(4 + aspect_ratio ** 2 * b2 / eta ** 2 * (1 + t * t / b2)))


def gust_load(wing_loading: float, V: float, *, chord: float, aspect_ratio: float, sweep_deg: float = 0.0, U: float = 10.0,
              rho: float = RHO, mach: float = 0.0) -> dict:
    """Pratt's gust load factor increment for a wing loading [kg/m²] at true airspeed V in a true gust U [m/s]."""
    a = lift_slope(aspect_ratio, sweep_deg, mach)
    ws = wing_loading * G
    mu = 2 * ws / (rho * chord * a * G)
    kg = 0.88 * mu / (5.3 + mu)
    return {"lift_slope": a, "mass_ratio": mu, "K_g": kg, "dn": kg * rho * U * V * a / (2 * ws),
            "dn_sharp": rho * U * V * a / (2 * ws)}


def gust_alleviation(wing_loading: float, V: float, *, chord: float, aspect_ratio: float, sweep_deg: float = 0.0,
                     U: float = 10.0, rho: float = RHO, gradient_chords: float = 12.5, flap_effect: float = 0.5,
                     flap_max_deg: float = 15.0, actuator_s: float = 0.04, probe_lead_m: float = 1.2, gain: float = 1.0,
                     dt: float = 0.002) -> dict:
    """A 1-cos gust (``gradient_chords`` to its peak) flown in heave, with and without the flaperons.

    Per unit wing area: ``m'' dv/dt = q a K_g (w_g - v) / V + q a flap_effect delta`` (``flap_effect`` = d alpha /
    d delta of a full-span 25 % flaperon). The law: ``delta = -gain (w_probe / V) / flap_effect`` — cancel the gust's
    angle of attack as the probe measured it ``probe_lead_m / V`` earlier — limited to ``flap_max_deg`` and lagged by
    the actuator. Returns the peak load factor increments and the time histories."""
    g = gust_load(wing_loading, V, chord=chord, aspect_ratio=aspect_ratio, sweep_deg=sweep_deg, U=U, rho=rho)
    a, kg = g["lift_slope"], g["K_g"]
    q = 0.5 * rho * V * V
    H = gradient_chords * chord
    T = 2 * H / V
    n = int((T * 2.5 + 0.5) / dt)
    t = np.arange(n) * dt
    lead = probe_lead_m / V

    def wg(tt):
        return np.where((tt >= 0) & (tt <= T), 0.5 * U * (1 - np.cos(2 * math.pi * tt / T)), 0.0)

    out = {}
    for active in (False, True):
        v, d = 0.0, 0.0
        dn, dd = np.zeros(n), np.zeros(n)
        lim = math.radians(flap_max_deg)
        for i in range(n):
            w = float(wg(t[i]))
            cmd = float(np.clip(-gain * float(wg(t[i] + lead)) / V / flap_effect, -lim, lim)) if active else 0.0
            d += (cmd - d) * dt / actuator_s
            acc = q * a * (kg * (w - v) / V + flap_effect * d) / wing_loading          # m/s² (per kg/m² of wing)
            v += acc * dt
            dn[i], dd[i] = acc / G, d
        out["active" if active else "passive"] = {"dn": dn, "flap_deg": np.degrees(dd)}
    p, c = float(np.max(np.abs(out["passive"]["dn"]))), float(np.max(np.abs(out["active"]["dn"])))
    return {"t": t, "gust": wg(t), "passive": out["passive"], "active": out["active"], "dn_passive": p, "dn_active": c,
            "alleviation": 1 - c / p, "flap_peak_deg": float(np.max(np.abs(out["active"]["flap_deg"]))), "pratt_dn": g["dn"]}


def vectoring_authority(thrust_n: float, arm_m: float, tail_area_m2: float, tail_arm_m: float, *, deflection_deg: float = 15.0,
                        tail_cl: float = 0.8, speeds=None, rho: float = RHO) -> dict:
    """Pitching moment of the nozzles (``thrust_n`` deflected by ``deflection_deg`` at ``arm_m`` behind the centre of
    gravity) and of the tail at full deflection, against airspeed; and the speed where they are equal."""
    speeds = np.linspace(0.0, 120.0, 61) if speeds is None else np.asarray(speeds, float)
    m_vec = thrust_n * math.sin(math.radians(deflection_deg)) * arm_m
    m_tail = 0.5 * rho * speeds ** 2 * tail_area_m2 * tail_cl * tail_arm_m
    v_eq = math.sqrt(2 * m_vec / (rho * tail_area_m2 * tail_cl * tail_arm_m))
    return {"V": speeds, "nozzles_nm": np.full_like(speeds, m_vec), "tail_nm": m_tail, "crossover_speed": v_eq}


# ------------------------------------------------------------------------------------------------ the hover
def hover_controls(design) -> dict:
    """The layout's force actuators in the hover as fractions of the weight (see the module text) and its drag areas
    [m²] in a horizontal gust: ``horizontal`` = [(limit, lag s), ...] fastest first, ``vertical`` = the same for the
    upward force change (``down`` the limit downwards)."""
    kind = design.concept.kind
    p, S = design.params, design.wing_area_m2
    L, W, H = p["fuselage_length"] / 1000, p["fuselage_width"] / 1000, p["fuselage_height"] / 1000
    side = 0.85 * L * H * 0.9 + 2 * p["vtail_span"] * p["vtail_chord"] * 1e-6 * 0.7       # beam wind: the fuselage's side, the tail
    front = 0.895 * W * H * 0.35 + 0.02 * S
    spool = design.engine.spool_time * THRUST_LAG
    margin = design.engines * _static_thrust(design) / (design.mtow_kg * G)
    if kind == "lift_cruise":
        return {"horizontal": [(math.tan(math.radians(20.0)), 0.30)], "vertical": [(design.concept.rotor_margin - 1.0, 0.10)],
                "down": 0.9, "area_head": front, "area_beam": side, "broadside": False,
                "how": "tilts on its rotors; rotor thrust in 0.1 s"}
    splay = math.radians(15.0)
    fast_v = (1 - math.cos(splay), 0.08)                    # nozzles from the splay to straight down
    if kind == "vectored":
        return {"horizontal": [(0.5 * math.tan(math.radians(15.0)), 0.08), (math.tan(math.radians(10.0)), 0.5)],
                "vertical": [fast_v, (margin * math.cos(splay) - 1.0, spool)], "down": 0.6, "area_head": front, "area_beam": side,
                "broadside": False, "how": "nozzles swing, then the body tilts; thrust by splay, then by the spool"}
    return {"horizontal": [(math.tan(math.radians(15.0)), 0.40)], "vertical": [fast_v, (margin * math.cos(splay) - 1.0, spool)],
            "down": 0.6, "area_head": 1.2 * S + 0.9 * L * W, "area_beam": 1.2 * S + 0.9 * L * W, "broadside": True,
            "how": "nozzles tilt the whole aircraft; thrust by splay, then by the spool"}


def _static_thrust(design) -> float:
    """Static thrust per podded engine, the lift jets' share added (so engines x this = all the installed thrust)."""
    from vegeta.boreas import microjet as mj
    each = mj.CATALOGUE[design.engine_name]["thrust_N"] * design.engine_scale
    lift = design.lift_engine
    return each + (design.concept.lift_engines * mj.CATALOGUE[lift[0]]["thrust_N"] * lift[1] / design.engines if lift else 0.0)


def _hold(mass, actuators, disturbance, dt, bias=0.0, down=None, feedforward=False):
    """One axis, position held at zero: ``mass x'' = disturbance(t) + the actuators' forces``. The last actuator is
    the large one: a PD loop with a natural frequency of 0.35 / its lag (at most 2.5 rad/s) commands it, between
    ``-down`` (default: its limit) and its limit less ``bias`` (the steady force already asked of it). The faster,
    smaller actuators before it give what the command asks and the large one has not delivered yet. ``feedforward``:
    the disturbance is measured and added to the command (a deck's motion is; a gust is not). Returns x and the force."""
    W = mass * G
    big_lim, big_lag = actuators[-1]
    wn = min(2.5, 0.35 / big_lag)
    kp, kd = mass * wn * wn, 2 * 0.8 * mass * wn
    lo, hi = -(big_lim if down is None else down) * W, big_lim * W - bias
    x = v = 0.0
    f = [0.0] * len(actuators)
    xs, fs = np.zeros(len(disturbance)), np.zeros(len(disturbance))
    for i, dist in enumerate(disturbance):
        u = -kp * x - kd * v - (dist if feedforward else 0.0)
        f[-1] += (min(max(u, lo), hi) - f[-1]) * dt / big_lag
        got = f[-1]
        for j in range(len(actuators) - 2, -1, -1):
            lim, lag = actuators[j]
            f[j] += (min(max(u - got, -lim * W), lim * W) - f[j]) * dt / lag
            got += f[j]
        v += (dist + got) / mass * dt
        x += v * dt
        xs[i], fs[i] = x, got
    return xs, fs


def hover_gust(design, *, wind: float = 12.0, gust: float = 8.0, gust_s: float = 3.0, beam: bool = True, mass_kg: float | None = None,
               dt: float = 0.01, seconds: float = 20.0) -> dict:
    """Hold position in a steady wind over the deck plus a 1-cos gust of ``gust`` m/s lasting ``gust_s`` (``beam``: the
    gust comes from the side, on the beam area; else from ahead)."""
    hc = hover_controls(design)
    m = mass_kg or design.mtow_kg
    area = hc["area_beam"] if beam else hc["area_head"]
    t = np.arange(int(seconds / dt)) * dt
    u = np.where((t >= 2.0) & (t <= 2.0 + gust_s), 0.5 * gust * (1 - np.cos(2 * math.pi * (t - 2.0) / gust_s)), 0.0)
    steady = 0.5 * RHO * hc["area_head"] * wind ** 2
    if beam and not hc["broadside"]:
        dist = 0.5 * RHO * area * u ** 2                             # a side gust on an aircraft that points into the wind
    else:
        dist = 0.5 * RHO * area * ((wind + u) ** 2 - wind ** 2)
    x, f = _hold(m, hc["horizontal"], dist, dt, bias=steady)
    auth = sum(a for a, _ in hc["horizontal"]) * m * G
    return {"t": t, "gust": u, "x": x, "force": f, "disturbance": dist, "peak_excursion_m": float(np.max(np.abs(x))),
            "steady_force_n": steady, "steady_tilt_deg": math.degrees(math.atan(steady / (m * G))),
            "gust_force_n": float(np.max(dist)), "authority_n": auth, "authority_used": float((steady + np.max(np.abs(f))) / auth),
            "wind_limit": math.sqrt(0.6 * auth / (0.5 * RHO * hc["area_head"]))}


def deck_heave(design, *, amplitude: float = 1.0, period: float = 8.0, mass_kg: float | None = None, dt: float = 0.01,
               seconds: float = 48.0) -> dict:
    """Follow a deck heaving ``amplitude`` m at ``period`` s, a fixed height above it, with the vertical force: the
    error in the height above the deck (the last two periods) and the force it took."""
    hc = hover_controls(design)
    m = mass_kg or design.mtow_kg
    t = np.arange(int(seconds / dt)) * dt
    w = 2 * math.pi / period
    deck = amplitude * np.sin(w * t)
    # in the deck's frame the aircraft feels the deck's acceleration as a disturbance force
    dist = m * amplitude * w * w * np.sin(w * t)
    x, f = _hold(m, hc["vertical"], dist, dt, down=hc["down"], feedforward=True)
    tail = t > seconds - 2 * period
    return {"t": t, "deck": deck, "error": x, "force": f, "rms_error_m": float(np.sqrt(np.mean(x[tail] ** 2))),
            "peak_error_m": float(np.max(np.abs(x[tail]))), "force_needed": amplitude * w * w / G,
            "peak_force": float(np.max(np.abs(f[tail])) / (m * G))}


def deck_stand(design, *, roll_deg: float = 8.0, roll_period: float = 10.0, wind: float = 15.0, mass_kg: float | None = None) -> dict:
    """Standing on the deck after landing, engines off, not yet lashed: the roll angle at which it tips (static), and
    the wind that tips it at ``roll_deg`` of ship roll (the roll's lateral acceleration at the deck included)."""
    kind = design.concept.kind
    p = design.params
    m = mass_kg or design.dry_kg
    L, W, H = p["fuselage_length"] / 1000, p["fuselage_width"] / 1000, p["fuselage_height"] / 1000
    S = design.wing_area_m2
    if kind == "tailsitter":
        r = p["vtail_span"] / 1000 * 1.15                           # the four fin-tip feet, on a circle
        half, h_cg, h_cp, area = r / math.sqrt(2), 0.42 * L, 0.5 * L, 1.2 * S + 0.9 * L * W
    else:
        half, h_cg, h_cp, area = 0.45 * W, 0.5 * H + 0.12, 0.5 * H + 0.12, 0.9 * 0.85 * L * H   # two skids under the hull's chines
    static = math.degrees(math.atan(half / h_cg))
    th = math.radians(roll_deg)
    a_lat = 0.0                                                     # at the deck's roll centre; off it, r w² th adds
    restoring = m * G * (half * math.cos(th) - h_cg * math.sin(th)) - m * a_lat * h_cg
    v_tip = math.sqrt(max(restoring, 0.0) / (0.5 * RHO * area * h_cp)) if restoring > 0 else 0.0
    drag = 0.5 * RHO * area * wind ** 2
    return {"footprint_half_m": half, "cg_height_m": h_cg, "static_tip_deg": static, "tip_wind_at_roll": v_tip,
            "margin_at_wind": restoring / (drag * h_cp) if drag > 0 else float("inf"), "wind_force_n": drag}


# ------------------------------------------------------------------------------------------------ the water
def afloat(design, mass_kg: float | None = None, *, kg_frac: float = 0.42) -> dict:
    """The sealed hull at rest on calm water: draft, freeboard to the intake lips and to the lift rotors, the
    metacentric height (the box-like middle of the fuselage carries the buoyancy; ``kg_frac`` = centre of gravity
    above the keel / hull height: the bay, the tanks and the battery are low)."""
    import albatros as al

    p = design.params
    m = mass_kg or design.dry_kg
    L = al.Albatros.layout(p)
    W, H = p["fuselage_width"] / 1000, p["fuselage_height"] / 1000
    length = (L["x_cone"] - (L["x_nose"] + 0.5 * p["nose_shape"])) / 1000 + 0.3 * p["tail_cone"] / 1000
    z = np.linspace(0, H, 400)
    half = W / 2 * np.clip(1 - np.abs((z - H / 2) / (H / 2)) ** al.SUPER_N, 0, 1) ** (1 / al.SUPER_N)
    area = np.concatenate([[0.0], np.cumsum(0.5 * (half[1:] + half[:-1]) * 2 * np.diff(z))])
    need = m / SEA_WATER / length
    total = float(area[-1]) * length
    if need >= area[-1]:
        return {"floats": False, "displacement_m3": m / SEA_WATER, "hull_volume_m3": total}
    draft = float(np.interp(need, area, z))
    i = int(np.searchsorted(z, draft))
    wl = 2 * float(np.interp(draft, z, half))
    kb = float(np.sum(0.5 * (z[1:i + 1] + z[:i]) * (half[1:i + 1] + half[:i]) * np.diff(z[:i + 1])) / max(need, 1e-9))
    bm = (length * wl ** 3 / 12) / (m / SEA_WATER)
    gm = kb + bm - kg_frac * H
    intake = H - draft + (L["nacelle_z"] - L["r_intake"]) / 1000
    rotor = (H - draft + (L["boom_z"] + p["boom_diameter"] / 2) / 1000 + 0.07 * p["rotor_diameter"] / 1000) if p["lift_rotors"] else None
    return {"floats": True, "draft_m": draft, "freeboard_m": H - draft, "reserve_buoyancy": total / (m / SEA_WATER) - 1,
            "waterline_width_m": wl, "GM_m": gm, "intake_above_water_m": intake, "rotor_above_water_m": rotor,
            "wave_limit_m": 0.5 * min(H - draft, intake if rotor is None else min(intake, rotor)),
            "displacement_m3": m / SEA_WATER, "hull_volume_m3": total}


def scorecard(designs: dict, *, wind: float = 12.0, gust: float = 8.0, deck_amplitude: float = 1.0, deck_period: float = 8.0,
              roll_deg: float = 8.0, cruise_gust: float = 10.0, altitude_m: float = 6000.0):
    """The gust, launch, deck and water numbers of every layout, side by side ("-" where a row does not apply: a
    catapult aircraft does not hover, a thrust-borne one needs no catapult)."""
    import pandas as pd
    import albatros_flight as F

    rows = {}
    for d in designs.values():
        m = d.flight.landing_mass_kg
        p = d.params
        chord = p["root_chord"] * (1 + p["taper"]) / 2000
        V = d.flight.cruise_speed_out
        kw = dict(chord=chord, aspect_ratio=d.airframe.aspect_ratio, sweep_deg=p["sweep_deg"], U=cruise_gust, rho=F.density(altitude_m))
        gl = gust_load(d.mtow_kg / d.wing_area_m2, V, **kw)
        ga = gust_alleviation(d.mtow_kg / d.wing_area_m2, V, **kw)
        fl = afloat(d, m)
        row = {"wing loading [kg/m²]": round(d.mtow_kg / d.wing_area_m2),
               f"cruise gust {cruise_gust:g} m/s: load factor step [g]": round(gl["dn"], 2),
               "the same with the flaperons working [g]": round(ga["dn_active"], 2)}
        if d.concept.kind == "catapult":
            la = F.launch(d)
            row.update({"release over the vessel": f"a pass at {la['release_speed']:.0f} m/s",
                        "stall speed at take-off [m/s]": round(la["stall_speed"], 1), "catapult end speed [m/s]": round(la["launch_speed"], 1),
                        "catapult stroke at 8 g [m]": round(la["catapult_stroke_m"], 1),
                        "the same with 10 m/s of wind over the deck [m]": round(la["stroke_with_10ms_wind_m"], 1),
                        "catapult energy [kJ]": round(la["catapult_energy_kj"]),
                        "runway take-off roll [m]": round(la["runway_roll_m"]), "approach speed [m/s]": round(la["approach_speed"], 1),
                        "landing stop at 0.35 g [m]": round(la["landing_roll_m"])})
        else:
            hg = hover_gust(d, wind=wind, gust=gust, mass_kg=m)
            dh = deck_heave(d, amplitude=deck_amplitude, period=deck_period, mass_kg=m)
            ds = deck_stand(d, roll_deg=roll_deg, mass_kg=m)
            row.update({"release over the vessel": "from a hover", f"hover, side gust {gust:g} m/s: drift [m]": round(hg["peak_excursion_m"], 2),
                        "hover wind limit [m/s]": round(hg["wind_limit"]),
                        f"deck heave ±{deck_amplitude:g} m / {deck_period:g} s: height error rms [m]": round(dh["rms_error_m"], 2),
                        "tips on the deck at roll [deg]": round(ds["static_tip_deg"]),
                        f"wind that tips it at {roll_deg:g} deg roll [m/s]": round(ds["tip_wind_at_roll"])})
        row.update({"afloat (ditched, calm sea): draft [m]": round(fl["draft_m"], 2) if fl["floats"] else "sinks",
                    "metacentric height [m]": round(fl["GM_m"], 2) if fl["floats"] else "-"})
        rows[d.concept.name] = row
    t = pd.DataFrame(rows).astype(object)
    return t.where(t.notna(), "-")


__all__ = ["lift_slope", "gust_load", "gust_alleviation", "vectoring_authority", "hover_controls", "hover_gust", "deck_heave",
           "deck_stand", "afloat", "scorecard", "SEA_WATER"]
