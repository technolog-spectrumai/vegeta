"""Track mechanics of a two-track skid-steer vehicle (notebook 30, todo 5j.1 and 5j.3). SI: N, m, Pa, rad.

* **resistance** — ``resistance``: soil compaction (``terramechanics``) or the rubber's rolling resistance on hard
  ground, the track's internal loss (belt bending, road-wheel and sprocket bearings, link hinges:
  f_in = f₀ + f₁ v, Wong's form), the grade and the air drag;
* **traction** — ``tractive_limit``: what the ground gives (μW on hard ground, the soil's shear strength with the
  slip–thrust law otherwise);
* **road-wheel loads** — ``road_wheel_loads``: a rigid frame (a linear distribution over the wheels, the elastic
  foundation solution) or wheels in pairs on bogies (the bogie pivots carry the load statically determinate, each
  wheel half its bogie's);
* **tension** — the belt's sag between rollers (a parabola: s = w L² / (8 T)), the tension that keeps the guide horns
  engaged against a lateral load (``derail_tension``), the tension from the drive (tight side T₀ + F);
* **skid steering** — ``skid_steer``: Wong's turning resistance M_r = μ_t W L / 4 (uniform pressure), the lateral
  coefficient falling with the turn radius μ_t = μ_t,max / (0.925 + 0.15 R / B) (Wong's empirical fit), the outer
  and inner track forces F_o,i = R_x / 2 ± M_r / B, the power of a clutch-brake (inner track braked, its power lost)
  and a regenerative steer (power flows from the inner track to the outer);
* **stability** — ``stability``: static tip-over angles, the largest step and trench.
"""
from __future__ import annotations

import math

import numpy as np

import terramechanics as tm

__all__ = ["RHO_AIR", "G", "ground_pressure", "internal_coefficient", "resistance", "tractive_limit",
           "road_wheel_loads", "sag", "derail_tension", "drive_tension", "lateral_coefficient", "skid_steer",
           "min_turn_radius", "stability"]

RHO_AIR = 1.225
G = 9.81


def ground_pressure(W: float, b: float, L: float, n_tracks: int = 2) -> float:
    """Nominal ground pressure [Pa]: W / (n b L)."""
    return W / (n_tracks * b * L)


def internal_coefficient(v: float, f0: float = 0.035, f1: float = 0.002) -> float:
    """Internal motion-resistance coefficient f_in = f₀ + f₁ v (v [m/s]); Wong gives 0.0222 + 0.0003 V (V [km/h])
    for steel-link tracks of heavy vehicles; a small rubber/plastic track with small road wheels loses more
    (``ASSUMPTIONS`` of pekari_rover_robot)."""
    return f0 + f1 * abs(v)


def resistance(W: float, *, b: float, L: float, soil, grade_deg: float = 0.0, v: float = 0.0, CdA: float = 0.0,
               f0: float = 0.035, f1: float = 0.002) -> dict:
    """Motion resistance [N] of the vehicle (two tracks, each W/2 on b × L) on ``soil`` at ``grade_deg`` and speed
    ``v``: compaction (soft soil; 0 on hard ground), rolling (the rubber's C_rr on hard ground), internal, grade,
    air drag; ``total``."""
    s = tm.soil(soil)
    th = math.radians(grade_deg)
    Wn = W * math.cos(th)
    comp = 2 * tm.compaction_resistance_track(Wn / 2, b, L, s)
    roll = s.c_rr * Wn if s.rigid else 0.0
    internal = internal_coefficient(v, f0, f1) * Wn
    grade = W * math.sin(th)
    drag = 0.5 * RHO_AIR * CdA * v * v
    return {"compaction": comp, "rolling": roll, "internal": internal, "grade": grade, "drag": drag,
            "total": comp + roll + internal + grade + drag}


def tractive_limit(W: float, *, b: float, L: float, soil, grade_deg: float = 0.0, slip: float = 1.0) -> float:
    """Thrust [N] of both tracks at ``slip`` (1 = the ground's limit) on ``grade_deg``."""
    Wn = W * math.cos(math.radians(grade_deg))
    return 2 * float(tm.thrust_track(Wn / 2, b, L, slip, soil))


def road_wheel_loads(W_track: float, wheel_x, x_cg: float, *, bogies=None) -> np.ndarray:
    """Load [N] on each road wheel of one track carrying ``W_track`` with its CG at ``x_cg`` [m] (on flat ground).

    ``bogies`` None: the wheels on a rigid frame on an elastic ground — a linear distribution
    F_i = W/n + W (x_cg − x̄)(x_i − x̄) / Σ(x_j − x̄)². ``bogies`` = [(i, j), …] pairs of wheel indices on bogies
    pivoting midway: two bogies share W by their pivots' lever arms (statically determinate), each wheel half of
    its bogie's (more bogies: the linear distribution over the pivots)."""
    x = np.asarray(wheel_x, dtype=float)
    if bogies is None:
        xm = x.mean()
        return W_track / len(x) + W_track * (x_cg - xm) * (x - xm) / np.sum((x - xm) ** 2)
    piv = np.array([(x[i] + x[j]) / 2 for i, j in bogies])
    pm = piv.mean()
    fp = W_track / len(piv) + W_track * (x_cg - pm) * (piv - pm) / np.sum((piv - pm) ** 2)
    out = np.zeros_like(x)
    for (i, j), f in zip(bogies, fp):
        out[i] += f / 2
        out[j] += f / 2
    return out


def sag(T: float, w: float, span: float) -> float:
    """Mid-span sag [m] of a belt of weight ``w`` [N/m] under tension ``T`` [N] over ``span`` [m]: w L² / (8 T)."""
    return w * span * span / (8.0 * T)


def derail_tension(F_lat: float, span: float, guide_height: float, engagement: float = 0.5) -> float:
    """Belt tension [N] that keeps the guide horns in the road wheels when a lateral force ``F_lat`` [N] acts on the
    free span ``span`` [m] between two wheels: the belt deflects sideways like a string, δ = F L / (4 T); it must
    stay below ``engagement`` × the horn height ``guide_height`` [m]."""
    return F_lat * span / (4.0 * engagement * guide_height)


def drive_tension(T0: float, F_drive: float) -> dict:
    """Tight-side and slack-side tension [N] of a sprocket-driven belt with pre-tension ``T0`` delivering
    ``F_drive`` (the slack side keeps T₀, the tight side adds the drive force: the rear-sprocket layout puts the
    ground run on the tight side when driving forward)."""
    return {"tight": T0 + F_drive, "slack": T0}


def lateral_coefficient(R: float, B: float, mu_t_max: float) -> float:
    """Wong's lateral resistance coefficient for a turn of radius ``R`` [m] (track centre line spacing ``B``):
    μ_t = μ_t,max / (0.925 + 0.15 R / B)."""
    return mu_t_max / (0.925 + 0.15 * R / B)


def skid_steer(W: float, *, L: float, B: float, R: float, mu_t_max: float, rolling: float, v: float = 0.0) -> dict:
    """A steady skid-steer turn of radius ``R`` [m] at speed ``v`` [m/s] (centre), uniform pressure: turning
    resistance M_r = μ_t W L / 4, outer/inner track forces F_o,i = R_x / 2 ± M_r / B (``rolling`` = R_x, the
    straight-line resistance [N]); track speeds v (R ± B/2) / R; the power of a clutch-brake steer (outer power, the
    inner track's braking power lost) and a regenerative steer (net of both)."""
    mu_t = lateral_coefficient(R, B, mu_t_max)
    Mr = mu_t * W * L / 4.0
    Fo, Fi = rolling / 2 + Mr / B, rolling / 2 - Mr / B
    vo = v * (R + B / 2) / R if R > 0 else v
    vi = v * (R - B / 2) / R if R > 0 else -v
    p_o, p_i = Fo * vo, Fi * vi
    return {"R_m": R, "mu_t": mu_t, "M_r_Nm": Mr, "F_outer_N": Fo, "F_inner_N": Fi, "v_outer": vo, "v_inner": vi,
            "P_clutch_brake_W": p_o + max(0.0, p_i), "P_braked_lost_W": max(0.0, -p_i),
            "P_regenerative_W": p_o + p_i}


def min_turn_radius(W: float, *, L: float, B: float, mu_t_max: float, rolling: float, F_max: float,
                    R_grid=None) -> float:
    """The smallest radius [m] whose outer-track force stays within ``F_max`` [N] (the lower of the ground's and the
    drive's limit); 0 when a pivot turn is possible."""
    R_grid = np.linspace(0.0, 20 * B, 2001) if R_grid is None else R_grid
    for R in R_grid:
        if skid_steer(W, L=L, B=B, R=R, mu_t_max=mu_t_max, rolling=rolling)["F_outer_N"] <= F_max:
            return float(R)
    return math.inf


def stability(*, x_cg: float, z_cg: float, x_front: float, x_rear: float, half_gauge: float, half_width: float,
              step_height: float, contact_front: float, contact_rear: float) -> dict:
    """Static stability of the vehicle on its tracks [deg, m] from the CG (x forward from the contact centre, z
    above the ground): uphill and downhill tip-over angles about the rear/front ends of the contact patch, the
    side-slope tip angle about the downhill track's outer edge (``half_width``: the track centre plus half the
    track width), the side-slope limit to skidding sideways is the soil's (μ); the step the front idler climbs
    (``step_height``: the idler axle height — the belt wraps it) and the trench the tracks bridge with the CG
    still over the far edge (the distance from the CG to the nearer end of the track envelope ``x_front`` /
    ``x_rear``)."""
    up = math.degrees(math.atan2(x_cg - contact_rear, z_cg))
    down = math.degrees(math.atan2(contact_front - x_cg, z_cg))
    side = math.degrees(math.atan2(half_width, z_cg))
    trench = min(x_front - x_cg, x_cg - x_rear)
    return {"tip_uphill_deg": up, "tip_downhill_deg": down, "tip_side_deg": side, "step_m": step_height,
            "trench_m": trench}
