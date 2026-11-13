"""ALBATROS lets its multirotor go (notebook 50): the folded hexacopter leaves the belly bay in a pass at 1.3 x the
stall speed, falls clear, is slowed and set upright by a drogue, swings its arms out, starts its rotors, drops the
drogue and flies. This module follows that sequence and gives what it needs: the clearance from the aircraft, the
loads, the time, the ground it covers and — the number the pass is flown by — the height it uses up.

**The package** (``Package``, from ``albatros_bay``'s multirotor): the folded vehicle with its cargo box, a flat box
``length x width x height``. Its drag area is ``flat_cd`` x the large face when it falls flat, ``end_cd`` x the small
face end-on, and something between while it tumbles (``tumble`` in 0..1).

**The sequence** (``simulate``; a point mass in the vertical plane of the pass, 5 ms steps, wind along the pass):

1. *Ejection* at ``eject_speed`` downwards from the bay. ``separation`` follows the first second in the aircraft's
   frame: the package falls and drops behind (its drag), the aircraft — lighter by the package, its lift unchanged —
   rises; the smallest gap to the belly's upswept tail line and to the nozzles is the clearance.
2. *Free fall*, tumbling, until the static line (``Drogue.line_m``, tied to the bay) is taut: it pulls the drogue out.
3. *Drogue*: it fills in ``Drogue.fill_s``; the opening load is ``opening_factor`` x the steady drag at that moment
   (the peak deceleration of the package). Under it the package hangs hub-up, flat side to the flow, slows and
   steepens its path.
4. *Unfolding* starts when the airspeed is below ``unfold_speed`` and the path is within ``unfold_path_deg`` of the
   vertical. Each arm swings in the hub's plane on a sprung hinge against a damper and the air's side load
   (``unfold_arm``: the time to the lock and the energy the lock must take).
5. *Rotor start*: the thrust rises over ``spin_up_s`` to ``thrust_to_weight`` x the weight; the drogue is cut when
   the rotors carry ``cut_fraction`` of the weight.
6. *Recovery*: the thrust opposes the velocity and holds the weight; in the vortex-ring band (a descent of 0.5-1.5 x
   the hover's induced velocity with little sideways speed) the thrust is derated by ``vrs_factor``. The end is a
   hover.

``min_release_height`` runs the sequence in still air; ``monte_carlo`` scatters the release speed, the tumbling, the
drogue's size, delay and opening factor, the wind and gusts, the rotors' spin-up and thrust, and a rotor that does not
start — and gives the height that covers them, which is the height of the pass. ``without_drogue`` is the same
package unfolding straight into the airstream of the pass: what the drogue is for.

Limits: a point mass with attitude taken as given by the drogue (no tumbling dynamics, no riser pendulum, no line
snatch), quasi-steady drag, one arm at a time against a steady side wind, a first-order rotor, the aircraft's wake
and the bay's flow left out of the separation (``eject_speed`` is the margin against them). The numbers are a first
sizing of the release, to be replaced by drop tests of a dummy package from the bay.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ogma.units import G
from ogma.units import RHO0 as RHO


@dataclass(frozen=True)
class Package:
    """The folded multirotor with its cargo."""
    mass_kg: float
    length_m: float
    width_m: float
    height_m: float
    rotors: int
    arm_m: float                      # hub centre to motor axis (unfolded)
    prop_m: float
    thrust_to_weight: float = 2.0
    disk_loading: float = 110.0       # N/m² in the hover
    flat_cd: float = 1.1
    end_cd: float = 0.9
    arm_tip_kg: float = 0.33          # motor, controller and folding propeller at an arm's end
    arm_tube_kg: float = 0.12

    @property
    def weight(self) -> float:
        return self.mass_kg * G

    def drag_area(self, tumble: float = 0.5) -> float:
        """Cd.A [m²] of the folded box: end-on (tumble = 0) to flat (1)."""
        end, flat = self.end_cd * self.width_m * self.height_m, self.flat_cd * self.length_m * self.width_m
        return end + (flat - end) * min(max(tumble, 0.0), 1.0)

    @property
    def induced_velocity(self) -> float:
        return math.sqrt(self.disk_loading / (2 * RHO))

    @property
    def hinge_radius(self) -> float:
        return self.arm_m - max(0.28 * self.arm_m, 0.06)

    @property
    def arm_inertia(self) -> float:
        r = self.hinge_radius
        return self.arm_tip_kg * r * r + self.arm_tube_kg * r * r / 3


def package(multirotor, **kw) -> Package:
    """The ``Package`` of an ``albatros_bay.Multirotor``."""
    L, W, H = multirotor.folded_m
    drive = multirotor.budget.get("motors, controllers, propellers", 0.33 * multirotor.rotors)
    return Package(multirotor.mass_kg, L, W, H, multirotor.rotors, multirotor.arm_m, multirotor.prop_diameter_m,
                   **{"arm_tip_kg": drive / multirotor.rotors, **kw})


@dataclass(frozen=True)
class Drogue:
    drag_area_m2: float = 0.60        # Cd.A, filled (a 1.1 m cruciform or ribbon canopy)
    line_m: float = 3.0               # static line from the bay: taut this far from the aircraft, it pulls the drogue out
    fill_s: float = 0.25
    opening_factor: float = 1.4       # peak / steady drag at the opening
    mass_kg: float = 0.25

    def filled(self, t_since: float) -> float:
        return 0.0 if t_since < 0 else min(t_since / self.fill_s, 1.0)


@dataclass(frozen=True)
class Sequence:
    eject_speed: float = 3.0          # m/s down, from the bay's ejector (``separation``: what clears the bay's rear sill)
    unfold_speed: float = 22.0        # m/s: the arms are let go below this airspeed ...
    unfold_path_deg: float = 30.0     # ... with the path this close to the vertical
    spin_up_s: float = 0.5            # rotors from rest to full thrust
    cut_fraction: float = 0.6         # thrust / weight at which the drogue is cut
    vrs_factor: float = 0.75          # thrust left in the vortex-ring band
    load_limit_g: float = 8.0         # what the folded airframe and the cargo take at the opening
    spring_start_nm: float = 1.6      # hinge spring torque folded ...
    spring_end_nm: float = 0.5        # ... and at the lock
    damper: float = 0.06              # N m s/rad at the hinge
    swing_deg: float = 90.0           # the arm that swings furthest
    lock_energy_j: float = 3.0        # what a lock takes without a buffer


# ------------------------------------------------------------------------------------------------ separation
def separation(pkg: Package, params: dict, speed: float, aircraft_mass_kg: float, *, eject_speed: float = 3.0, tumble: float = 0.5,
               seconds: float = 1.0, dt: float = 0.002) -> dict:
    """The first second in the aircraft's frame (x aft from the bay's middle, z down from the package's start).

    Inside the bay the package is out of the airstream: its drag grows with the part of it that has come out below
    the belly. Until all of it is out (z = its height), its rear edge must not reach the bay's rear sill: ``sill_gap_m``
    is what is left of the clearance there (negative: it strikes; a faster ejector or guide rails). Once out, the
    airstream can pitch it: at worst with the moment of its whole flat face's drag at a quarter of its length, so its
    corner rises as it turns; ``tail_gap_m`` is the least distance from that corner to the belly's upswept line
    behind the bay, from 30 ms after it has left the sill. ``aircraft_mass_kg`` is the mass before the release; the aircraft holds its speed and its
    lift, so it rises as the package leaves."""
    import albatros as al

    L = al.Albatros.layout(params)
    Hf = params["fuselage_height"] / 1000
    a_full = 0.5 * RHO * speed ** 2 * pkg.drag_area(tumble) / pkg.mass_kg
    a_up = G * pkg.mass_kg / (aircraft_mass_kg - pkg.mass_kg)
    n = int(seconds / dt) + 1
    t = np.arange(n) * dt
    x, z, gap = np.zeros(n), np.zeros(n), np.full(n, np.nan)
    vx, vz = 0.0, eject_speed
    clear = (params["bay_length"] / 1000 - pkg.length_m) / 2          # fore-and-aft room in the bay, each end
    half = 0.5 * math.hypot(pkg.length_m, pkg.height_m)
    pitch_acc = 0.5 * RHO * speed ** 2 * pkg.drag_area(1.0) * pkg.length_m / 4 / (pkg.mass_kg * pkg.length_m ** 2 / 12)
    th_max = math.pi / 2 - math.atan2(pkg.height_m, pkg.length_m)
    sill_gap, t_out = clear, None
    for i in range(1, n):
        out = min(z[i - 1] / pkg.height_m, 1.0)
        vx += a_full * out * dt
        vz += (G + a_up) * dt
        x[i], z[i] = x[i - 1] + vx * dt, z[i - 1] + vz * dt
        if z[i] < pkg.height_m:
            sill_gap = min(sill_gap, clear - x[i])
        else:
            t_out = t_out if t_out is not None else t[i]
            th = min(0.5 * pitch_acc * (t[i] - t_out) ** 2, th_max)
            reach = 0.5 * pkg.length_m * math.sin(th) + 0.5 * pkg.height_m * math.cos(th)
            xa = min(params["bay_x"] + (x[i] + half) * 1000, L["x_end"])
            s, zc = al.Albatros.section_scale(params, xa)
            belly_up = Hf + (zc - s * params["fuselage_height"] / 2) / 1000      # how far the belly has risen at that station
            gap[i] = (z[i] - pkg.height_m / 2) + belly_up - reach
    x_tail = (L["x_end"] - params["bay_x"]) / 1000
    i_tail = min(int(np.searchsorted(x, x_tail)), n - 1)
    i_line = min(int(np.searchsorted(np.hypot(x, z), 3.0)), n - 1)
    return {"t": t, "x": x, "z": z, "gap": gap, "sill_gap_m": float(sill_gap), "time_out_of_bay_s": t_out,
            "tail_gap_m": float(np.nanmin(gap[t >= (t_out or 0.0) + 0.03])), "aircraft_dn": a_up / G, "drag_decel_g": a_full / G,
            "time_past_tail_s": float(t[i_tail]), "drop_at_tail_m": float(z[i_tail]), "tail_x_m": x_tail, "line_taut_s": float(t[i_line]),
            "eject_speed": eject_speed}


def eject_speed_needed(pkg: Package, params: dict, speed: float, aircraft_mass_kg: float, *, margin_m: float = 0.015, **kw) -> float:
    """The ejector speed that leaves ``margin_m`` at the bay's rear sill (flat-side drag: the worst tumble)."""
    lo, hi = 0.0, 20.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ok = separation(pkg, params, speed, aircraft_mass_kg, eject_speed=mid, **{"tumble": 1.0, **kw})["sill_gap_m"] >= margin_m
        lo, hi = (lo, mid) if ok else (mid, hi)
    return hi


# ------------------------------------------------------------------------------------------------ the arms
def unfold_arm(pkg: Package, seq: Sequence | None = None, *, side_speed: float = 0.0, arm_cd_area: float | None = None,
               dt: float = 0.0005, limit_s: float = 3.0) -> dict:
    """One arm from folded to its lock: ``I th'' = spring(th) - damper th' - the air's moment``. The air's moment is
    the drag of the arm, its motor and its folded propeller in a wind of ``side_speed`` across the hub's plane, at
    half the hinge radius, taken as opposing the swing all the way (the worst arm)."""
    seq = seq or Sequence()
    r, I = pkg.hinge_radius, pkg.arm_inertia
    area = arm_cd_area if arm_cd_area is not None else 1.2 * (0.045 * pkg.arm_m * r + 0.012)
    m_air = 0.5 * RHO * side_speed ** 2 * area * 0.55 * r
    end = math.radians(seq.swing_deg)
    th = w = t = 0.0
    while th < end and t < limit_s:
        spring = seq.spring_start_nm + (seq.spring_end_nm - seq.spring_start_nm) * th / end
        acc = (spring - seq.damper * w - m_air) / I
        w = max(w + acc * dt, 0.0) if th <= 0 and acc < 0 else w + acc * dt
        th = max(th + w * dt, 0.0)
        t += dt
        if w <= 0 and acc <= 0 and th > 0:
            break
    locked = th >= end
    energy = 0.5 * I * w * w if locked else 0.0
    return {"locked": locked, "time_s": t if locked else float("inf"), "lock_energy_j": energy, "tip_speed": w * r if locked else 0.0,
            "air_moment_nm": m_air, "spring_margin": seq.spring_end_nm / m_air if m_air > 0 else float("inf"),
            "stalled_at_deg": None if locked else math.degrees(th)}


def unfold_speed_limit(pkg: Package, seq: Sequence | None = None) -> float:
    """The side wind in which the worst arm still reaches its lock."""
    lo, hi = 0.0, 80.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if unfold_arm(pkg, seq, side_speed=mid)["locked"] else (lo, mid)
    return lo


# ------------------------------------------------------------------------------------------------ the fall
def simulate(pkg: Package, speed: float, *, drogue: Drogue | None = Drogue(), seq: Sequence | None = None, wind: float = 0.0,
             updraft: float = 0.0, tumble: float = 0.5, rotors_out: int = 0, thrust_scale: float = 1.0, drogue_delay: float = 0.0,
             height_m: float = 1e4, dt: float = 0.005, limit_s: float = 40.0) -> dict:
    """The release flown (see the module text). ``speed`` is the aircraft's airspeed in the pass; ``wind`` blows
    along the pass (+ = a tailwind for the aircraft), ``updraft`` upwards. ``drogue=None`` unfolds straight into the
    airstream once clear of the aircraft. x is the ground distance from the release point, h the height lost."""
    seq = seq or Sequence()
    m = pkg.mass_kg
    vx, vz = speed + wind, -seq.eject_speed                          # ground speed; z up
    x = z = t = 0.0
    xa = 0.0                                                         # the aircraft, for the static line
    phase, events = "free fall", {}
    t_drogue = t_unfold = t_locked = t_cut = None
    arm = None
    peak_g, peak_g_recovery, thrust = 0.0, 0.0, 0.0
    t_max = (pkg.rotors - 2 * rotors_out if pkg.rotors >= 6 else pkg.rotors) / pkg.rotors * pkg.thrust_to_weight * pkg.weight * thrust_scale
    if pkg.rotors < 6 and rotors_out:
        t_max = 0.0                                                  # a quadcopter with a rotor out does not fly
    log = {k: [] for k in ("t", "x", "h", "speed", "path_deg", "g", "thrust", "drag_area")}
    vi = pkg.induced_velocity
    while t < limit_s:
        ax_, az_ = vx - wind, vz - updraft                           # velocity through the air
        va = math.hypot(ax_, az_)
        path = math.degrees(math.atan2(abs(ax_), max(-az_, 1e-9)))  # from the vertical
        # --- events
        if phase == "free fall" and drogue is not None and math.hypot(xa - x, z) >= drogue.line_m and t >= drogue_delay:
            phase, t_drogue = "drogue", t
            events["drogue out"] = t
        if drogue is None and phase == "free fall" and t >= 0.6 + drogue_delay:
            phase, t_unfold = "unfolding", t
            arm = unfold_arm(pkg, seq, side_speed=va)
            events["arms released"] = t
        if phase == "drogue" and t - t_drogue >= drogue.fill_s and va < seq.unfold_speed and path < seq.unfold_path_deg:
            phase, t_unfold = "unfolding", t
            arm = unfold_arm(pkg, seq, side_speed=abs(ax_))
            events["arms released"] = t
        if phase == "unfolding" and arm["locked"] and t - t_unfold >= arm["time_s"]:
            phase, t_locked = "rotor start", t
            events["arms locked"] = t
        if phase == "unfolding" and not arm["locked"] and t - t_unfold > 1.0:
            events["failed"] = "an arm does not reach its lock in this airstream"
            break
        if phase == "rotor start" and thrust >= seq.cut_fraction * pkg.weight:
            phase, t_cut = "recovery", t
            events["drogue cut" if drogue is not None else "rotors carrying"] = t
        # --- drag area
        fill = drogue.filled(t - t_drogue) if (drogue is not None and t_drogue is not None and t_cut is None) else 0.0
        hang = min((t - t_drogue) / 0.8, 1.0) if t_drogue is not None else 0.0     # hub-up under the drogue: flat to the flow
        cda = pkg.drag_area(tumble + (1.0 - tumble) * hang)
        if t_locked is not None:
            cda += 0.10                                                              # arms and stopped propellers out
        shock = 1.0
        if drogue is not None and fill > 0 and t - t_drogue < 2 * drogue.fill_s:
            shock = 1.0 + (drogue.opening_factor - 1.0) * math.sin(math.pi * (t - t_drogue) / (2 * drogue.fill_s))
        cda_total = cda + (drogue.drag_area_m2 * fill * shock if drogue is not None else 0.0)
        d = 0.5 * RHO * va * cda_total                                               # drag / speed
        fx, fz = -d * ax_, -d * az_ - pkg.weight
        # --- thrust
        if t_locked is not None and t_max > 0:
            ramp = min((t - t_locked) / seq.spin_up_s, 1.0)
            avail = t_max * ramp
            if 0.5 * vi < -az_ < 1.5 * vi and abs(ax_) < 0.5 * vi:
                avail *= seq.vrs_factor
            # command: hold the weight and stop the motion (1.5 /s on the velocity), within what is available
            # (the flight controller's integral term is taken as having found the air's force: it is fed forward)
            cx, cz = -1.5 * m * vx - fx, -1.5 * m * vz - fz
            if phase == "rotor start":
                cx = 0.0                                                             # under the drogue: straight up only
            c = math.hypot(cx, cz)
            thrust = min(c, avail)
            if c > 0:
                fx, fz = fx + thrust * cx / c, fz + thrust * cz / c
        elif t_locked is not None:
            events["failed"] = "the rotors cannot carry it"
            break
        g_load = math.hypot(fx, fz + pkg.weight) / pkg.weight
        if t_cut is None:
            peak_g = max(peak_g, g_load)
        else:
            peak_g_recovery = max(peak_g_recovery, g_load)
        vx += fx / m * dt
        vz += fz / m * dt
        x += vx * dt
        z += vz * dt
        xa += (speed + wind) * dt
        t += dt
        for k, v in (("t", t), ("x", x), ("h", -z), ("speed", va), ("path_deg", path), ("g", g_load), ("thrust", thrust), ("drag_area", cda_total)):
            log[k].append(v)
        if -z >= height_m:
            events["failed"] = events.get("failed", "it reaches the sea")
            break
        if phase == "recovery" and math.hypot(vx, vz) < 0.7:
            phase = "hover"
            events["hover"] = t
            break
    ok = phase == "hover"
    if not ok and "failed" not in events:
        events["failed"] = "the airspeed never falls to the unfolding speed" if t_unfold is None else "no hover within the time limit"
    h = np.array(log["h"])
    return {"ok": ok, "events": events, "height_lost_m": float(h.max()) if len(h) else 0.0, "downrange_m": x, "time_s": t,
            "peak_g_opening": peak_g, "peak_g_recovery": peak_g_recovery, "arm": arm,
            "speed_at_unfold": float(np.interp(t_unfold, log["t"], log["speed"])) if t_unfold is not None else None,
            "descent_at_cut": float(-np.interp(t_cut, log["t"], np.gradient(-h, log["t"]))) if t_cut is not None else None,
            "within_load_limit": peak_g <= seq.load_limit_g,
            "locks_without_buffer": bool(arm and arm["locked"] and arm["lock_energy_j"] <= seq.lock_energy_j),
            "track": {k: np.array(v) for k, v in log.items()}}


def without_drogue(pkg: Package, speed: float, **kw) -> dict:
    """The same release with no drogue: the arms are let go in the airstream of the pass."""
    return simulate(pkg, speed, drogue=None, **kw)


def drogue_trade(pkg: Package, speed: float, areas=(0.2, 0.4, 0.6, 0.9, 1.3, 1.8), seq: Sequence | None = None):
    """Drogue size against what it costs and buys: the opening load, the height lost, the descent rate at the cut."""
    import pandas as pd

    rows = {}
    for a in areas:
        r = simulate(pkg, speed, drogue=Drogue(drag_area_m2=a), seq=seq)
        rows[a] = {"canopy Ø [m]": round(math.sqrt(4 * a / 0.65 / math.pi), 2), "opening load [g]": round(r["peak_g_opening"], 1),
                   "airspeed at unfolding [m/s]": round(r["speed_at_unfold"], 1) if r["speed_at_unfold"] else "-",
                   "height lost [m]": round(r["height_lost_m"]) if r["ok"] else "-", "time to the hover [s]": round(r["time_s"], 1) if r["ok"] else "-",
                   "downrange [m]": round(r["downrange_m"]) if r["ok"] else "-",
                   "result": "flies" if r["ok"] and r["within_load_limit"] else ("over the load limit" if r["ok"] else r["events"].get("failed"))}
    t = pd.DataFrame(rows).T
    t.index.name = "drogue Cd·A [m²]"
    return t


def min_release_height(pkg: Package, speed: float, *, drogue: Drogue | None = Drogue(), seq: Sequence | None = None,
                       margin: float = 1.5, floor_m: float = 30.0) -> dict:
    """Still air, everything nominal: the height used to the hover, and the pass height with ``margin`` on it plus
    ``floor_m`` above the sea for the hover."""
    r = simulate(pkg, speed, drogue=drogue, seq=seq)
    return {"height_lost_m": r["height_lost_m"], "pass_height_m": margin * r["height_lost_m"] + floor_m, "ok": r["ok"], "run": r}


def monte_carlo(pkg: Package, speed: float, n: int = 400, *, drogue: Drogue = Drogue(), seq: Sequence | None = None, seed: int = 0,
                speed_sigma: float = 2.0, wind_sigma: float = 5.0, updraft_sigma: float = 2.0, p_rotor_out: float = 0.02,
                dt: float = 0.01) -> dict:
    """The release ``n`` times with what is not known scattered (see the module text). Returns the arrays, the
    height that covers 99 % of the flights, and the share that ends in a hover within the load limit."""
    rng = np.random.default_rng(seed)
    seq = seq or Sequence()
    out = {k: np.zeros(n) for k in ("height_lost_m", "peak_g_opening", "downrange_m", "time_s", "lock_energy_j")}
    ok = np.zeros(n, bool)
    why = {}
    for i in range(n):
        d = replace(drogue, drag_area_m2=drogue.drag_area_m2 * rng.uniform(0.85, 1.15), opening_factor=rng.uniform(1.2, 1.8),
                    fill_s=drogue.fill_s * rng.uniform(0.7, 1.5))
        s = replace(seq, spin_up_s=rng.uniform(0.3, 0.8))
        r = simulate(pkg, speed + rng.normal(0, speed_sigma), drogue=d, seq=s, wind=rng.normal(0, wind_sigma),
                     updraft=rng.normal(0, updraft_sigma), tumble=rng.uniform(0.0, 1.0), rotors_out=int(rng.random() < p_rotor_out),
                     thrust_scale=rng.uniform(0.85, 1.0), drogue_delay=rng.uniform(0.0, 0.4), dt=dt)
        ok[i] = r["ok"] and r["within_load_limit"]
        if not ok[i]:
            k = r["events"].get("failed", "over the load limit")
            why[k] = why.get(k, 0) + 1
        for k in ("height_lost_m", "peak_g_opening", "downrange_m", "time_s"):
            out[k][i] = r[k]
        out["lock_energy_j"][i] = r["arm"]["lock_energy_j"] if r["arm"] else np.nan
    good = out["height_lost_m"][ok]
    p99 = float(np.percentile(good, 99)) if len(good) else float("nan")
    return {**out, "ok": ok, "success": float(ok.mean()), "failures": why, "height_p50_m": float(np.median(good)) if len(good) else float("nan"),
            "height_p99_m": p99, "pass_height_m": 1.25 * p99 + 30.0, "g_p99": float(np.percentile(out["peak_g_opening"], 99)),
            "downrange_p05_m": float(np.percentile(out["downrange_m"][ok], 5)) if ok.any() else float("nan"),
            "downrange_p95_m": float(np.percentile(out["downrange_m"][ok], 95)) if ok.any() else float("nan")}


__all__ = ["Package", "package", "Drogue", "Sequence", "separation", "eject_speed_needed", "unfold_arm", "unfold_speed_limit", "simulate", "without_drogue",
           "drogue_trade", "min_release_height", "monte_carlo"]
