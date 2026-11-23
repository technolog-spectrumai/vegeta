"""Manoeuvrability statistics of a run and of the aircraft, with the literature beside them.

From a recorded trajectory (``history``: t, position, velocity, thrust, power, speed — and, from a MuJoCo run, the
attitude quaternion and the body rates): the linear envelope (peak acceleration and deceleration along the flight
path, peak lateral acceleration, the g-load the thrust vector carried), the turning (least turn radius from v² / a_lat
at speed, peak turn rate), the rotational envelope (peak pitch / roll / yaw rates and angular accelerations — from
the body rates when recorded, else from the thrust vector's tilt, which is what a quad must rotate to accelerate),
the time to speed and the braking, and the control quality (tracking error to the racing line, the share of the lap
at the thrust limit, the tilt used). ``envelope`` gives what the aircraft could do from its drive table and inertia
(the ceiling the run is measured against). ``LITERATURE`` holds published reference values.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import refquad

G = 9.81

#: published reference values for racing quads (see the notebook's sources)
LITERATURE = [
    {"source": "Foehn, Romero & Scaramuzza, Science Robotics 2021 (time-optimal waypoint flight)", "quantity": "modelled acceleration range", "value": "1–20 m/s² (≈ 2 g)"},
    {"source": "Foehn et al. 2021", "quantity": "maximal body rate used by the planner", "value": "10 rad/s (573 °/s)"},
    {"source": "Foehn et al. 2021", "quantity": "thrust-to-weight of the racing platform", "value": "3.3 planned, ≈ 4 measured"},
    {"source": "UZH / Kaufmann et al., Nature 2023 (Swift vs. champions)", "quantity": "speeds on the race track", "value": "over 100 km/h (≈ 28 m/s)"},
    {"source": "typical FPV racer settings (Betaflight rates; not a measurement)", "quantity": "commanded roll / pitch rate at full stick", "value": "800–1200 °/s; yaw 500–900 °/s"},
    {"source": "this model (refquad, Boreas table)", "quantity": "static thrust-to-weight", "value": f"{4 * refquad.propulsion().max_thrust_N / (refquad.MASS_KG * G):.1f}"},
]


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _smooth(x, n=5):
    if len(x) < n:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def from_history(H: np.ndarray, *, path=None, dt_min: float = 1e-3) -> dict:
    """The statistics of one run (see the module). ``H`` columns: 0 t, 1–3 p, 4–6 v, 7 T, 8 P, 9 |v| [, 10–13 quat w x y z, 14–16 body rates]."""
    t = H[:, 0]
    dt = np.gradient(t)
    dt = np.where(dt > dt_min, dt, dt_min)
    v = H[:, 4:7]
    a = np.gradient(v, axis=0) / dt[:, None]
    a = np.column_stack([_smooth(a[:, i]) for i in range(3)])
    speed = H[:, 9]
    with np.errstate(invalid="ignore", divide="ignore"):
        e_t = np.where(speed[:, None] > 0.5, v / np.maximum(speed, 1e-6)[:, None], 0.0)
    a_along = np.einsum("ij,ij->i", a, e_t)
    a_lat_vec = a - a_along[:, None] * e_t
    a_lat = np.linalg.norm(a_lat_vec, axis=1)
    load = np.linalg.norm(a + np.array([0.0, 0.0, G]), axis=1) / G                # the specific force the thrust carried
    fast = speed > 3.0
    pk = lambda x: float(np.percentile(x, 99)) if len(x) else np.nan             # "peak" = the 99th percentile: a sampled, differentiated trace has spikes at the zone hand-overs
    r_turn = np.where(a_lat > 2.0, speed ** 2 / np.maximum(a_lat, 2.0), np.inf)
    turn_rate = np.where(speed > 1.0, a_lat / np.maximum(speed, 1.0), 0.0)
    out = {"peak acceleration along the path [g]": pk(a_along[fast]) / G, "peak deceleration along the path [g]": pk(-a_along[fast]) / G,
           "peak lateral acceleration [g]": pk(a_lat[fast]) / G, "peak load factor (thrust / weight carried)": pk(load),
           "mean load factor": float(load.mean()), "least turn radius above 3 m/s [m]": float(np.percentile(r_turn[fast & np.isfinite(r_turn)], 1)) if (fast & np.isfinite(r_turn)).any() else np.nan,
           "peak turn rate [deg/s]": float(np.degrees(pk(turn_rate[fast])))}
    v_max = speed.max()
    i_fast = np.argmax(speed >= min(20.0, 0.9 * v_max))
    out["time to 20 m/s (or 90 % of the peak) [s]"] = float(t[i_fast] - t[0]) if speed.max() > 3 else np.nan
    # the rotational envelope: from the body rates when recorded, else from the thrust vector's tilt
    if H.shape[1] >= 17:
        w = H[:, 14:17]
        rates = np.degrees(np.abs(w))
        alpha = np.gradient(w, axis=0) / dt[:, None]
        out.update({"peak roll rate [deg/s]": pk(rates[:, 0]), "peak pitch rate [deg/s]": pk(rates[:, 1]), "peak yaw rate [deg/s]": pk(rates[:, 2]),
                    "peak roll / pitch angular acceleration [rad/s²]": pk(np.abs(alpha[:, :2]).max(axis=1)), "peak yaw angular acceleration [rad/s²]": pk(np.abs(alpha[:, 2])),
                    "rates from": "the recorded body rates (MuJoCo)"})
        tilt = np.degrees(np.arccos(np.clip([_quat_to_R(q)[2, 2] for q in H[:, 10:14]], -1, 1)))
    else:
        f = a + np.array([0.0, 0.0, G])
        n = f / np.maximum(np.linalg.norm(f, axis=1), 1e-6)[:, None]
        tilt = np.where(load > 0.5, np.degrees(np.arccos(np.clip(n[:, 2], -1, 1))), np.nan)      # a tilt means something only while the thrust carries weight
        dn = np.gradient(n, axis=0) / dt[:, None]
        rate = np.degrees(np.linalg.norm(dn, axis=1))                      # the tilt vector's slew: a roll / pitch rate the frame must follow
        rate = _smooth(rate, 7)
        heading = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        yaw_rate = np.degrees(np.abs(np.gradient(heading) / dt))
        yaw_rate = np.where(speed > 2.0, _smooth(yaw_rate, 7), 0.0)
        ang_acc = np.gradient(np.radians(rate)) / dt
        out.update({"peak roll / pitch rate (thrust-vector slew) [deg/s]": pk(rate[fast]), "peak yaw rate (heading) [deg/s]": pk(yaw_rate[fast]),
                    "peak roll / pitch angular acceleration (thrust-vector slew) [rad/s²]": pk(np.abs(_smooth(ang_acc, 7))[fast]), "rates from": "the thrust vector's slew and the heading (no attitude recorded: the mock)"})
    out["peak tilt [deg]"] = float(np.nanpercentile(tilt, 99)); out["mean tilt above 3 m/s [deg]"] = float(np.nanmean(tilt[fast])) if fast.any() else np.nan
    T = H[:, 7]
    out["share of the lap at ≥ 90 % thrust"] = float(np.mean(T >= 0.9 * 4 * refquad.propulsion().max_thrust_N))
    out["share of the lap at ≥ 50 % thrust"] = float(np.mean(T >= 0.5 * 4 * refquad.propulsion().max_thrust_N))
    if path is not None:
        err = []
        s = 0.0
        for p in H[:, 1:4]:
            s = path.project(p, s)
            q, _, _, _ = path.at(s)
            err.append(np.linalg.norm(p - q))
        err = np.array(err)
        out["tracking error to the racing line, RMS [m]"] = float(np.sqrt(np.mean(err ** 2)))
        out["tracking error, peak [m]"] = float(err.max())
    return out


def envelope() -> dict:
    """What the reference quad could do, from its drive table and a frame inertia estimate: the ceiling of the run."""
    p = refquad.propulsion()
    m, W = refquad.MASS_KG, refquad.MASS_KG * G
    T0 = 4 * p.max_thrust_N
    arm = refquad.FRAME["arm_r_m"] * math.cos(math.radians(45))                 # the X frame's arm about the roll / pitch axes
    I_rp = 0.6 * m * arm ** 2 + 4 * 0.0405 * arm ** 2                           # the frame spread, the motors at the arms
    I_y = 2 * I_rp
    dT = p.max_thrust_N - W / 4
    return {"static thrust / weight": T0 / W, "peak linear acceleration, any direction, from the hover [g]": T0 / W - 1,
            "horizontal acceleration at 70° tilt, height held [g]": math.tan(math.radians(70)), "lateral acceleration at 70° tilt [g]": math.tan(math.radians(70)),
            "least turn radius at 20 m/s, 70° tilt [m]": 400 / (G * math.tan(math.radians(70))), "least turn radius at 10 m/s [m]": 100 / (G * math.tan(math.radians(70))),
            "roll / pitch angular acceleration (two motors up, two at idle) [rad/s²]": 2 * dT * arm / I_rp, "yaw angular acceleration (torque, ± hover thrust) [rad/s²]": 2 * p.torque_per_thrust_m * (W / 4) / I_y,
            "time for a 90° roll, bang-bang at half authority [s]": 2 * math.sqrt(math.pi / 2 / (dT * arm / I_rp)), "sustainable power [W]": refquad.P_SUSTAIN_W,
            "braking from 20 m/s at 70° back-tilt, with the frame's drag [m]": 400 / (2 * (G * math.tan(math.radians(70)) + 0.5 * 1.225 * 400 * refquad.CDA_FORWARD_M2 / m)),
            "inertia estimate roll / pitch [kg·m²]": I_rp, "note": "a frame inertia estimate; the body rates themselves are limited by the controller's rate setting, not by the motors"}


def table(runs: dict, *, with_envelope: bool = True) -> pd.DataFrame:
    """One column per run (``{name: result}``) and the aircraft's envelope beside them."""
    cols = {name: from_history(r["history"], path=r.get("path")) for name, r in runs.items()}
    df = pd.DataFrame(cols)
    if with_envelope:
        env = envelope()
        df["aircraft envelope"] = pd.Series({"peak lateral acceleration [g]": env["lateral acceleration at 70° tilt [g]"], "peak acceleration along the path [g]": env["horizontal acceleration at 70° tilt, height held [g]"],
                                             "peak load factor (thrust / weight carried)": env["static thrust / weight"], "least turn radius above 3 m/s [m]": env["least turn radius at 10 m/s [m]"],
                                             "peak roll / pitch angular acceleration [rad/s²]": env["roll / pitch angular acceleration (two motors up, two at idle) [rad/s²]"]})
    return df


def literature() -> pd.DataFrame:
    return pd.DataFrame(LITERATURE).set_index("quantity")
