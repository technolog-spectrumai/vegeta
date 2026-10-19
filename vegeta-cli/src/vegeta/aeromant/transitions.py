"""transitions — a tailsitter's transitions, hover → wing-borne flight and back, through the stall.

A rigid body in the vertical plane (x, z, pitch): lifting **panels** (``Vehicle.panels``; ``biplane_panels`` builds a
biplane's four from a layout dict) blown by their rotors (momentum theory, Glauert's induced velocity), the rotors
through a first-order motor lag between the idle and the full-throttle thrust of a drive table, a scheduled pitch PID
with a height and airspeed loop, optional elevons. ``simulate`` flies one transition and measures it against
``CRITERIA``; ``matrix`` / ``verdict`` sweep vehicles × directions × schedules × headwinds; ``remedy_matrix`` /
``remedy_verdict`` sweep remedies (any ``simulate`` keyword: reverse thrust, a crow brake, an overshoot, a descending
entry) for one vehicle. The vehicle supplies its polar, its drive table and, optionally, its propellers' normal force
(``prop_cn_alpha``); nothing here knows a particular aircraft. Frame: the earth's x forward, z up; θ = the nose's
elevation (90° = hovering nose-up).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from typing import Callable

G = 9.80665
RHO = 1.225
WING_DROP_DCL = 0.30             # the lift one side's free-stream panels lose when they stall first
WN = 12.0                        # the pitch loop's natural frequency on the inertia [rad/s]
K_AERO = 0.6                     # the pitch gain's airspeed share: K_AERO × q S c per radian (scheduled on airspeed)
KI_RATIO = 2.0                   # the integral gain as a multiple of the proportional [1/s]
CRITERIA = {"height loss [m]": 3.0, "height gain, forward [m]": 3.0, "height gain, back [m]": 10.0, "pitch error [deg]": 10.0,
            "moment shortfall [s]": 0.2, "roll demand / authority": 1.0}
REF_RATE = math.radians(90.0)    # the hover's lean reference changes at most this fast [rad/s]
V_CLIMB, V_DESCENT = 3.0, 2.0    # the height loop's vertical-speed limits [m/s] (no tail-first fall at idle thrust)
V_DIVERGED = 150.0               # m/s: an integration past this has diverged (a 2 kg tailsitter never flies it): a result, not a crash


def induced_velocity(thrust, V, alpha_rad, A, rho=RHO, iterations=40):
    """Glauert's induced velocity at a disk [m/s] in an inflow V at α to its axis (axial component V cos α)."""
    if thrust <= 0:
        return 0.0
    va, vn = V * math.cos(alpha_rad), V * math.sin(alpha_rad)
    vh = math.sqrt(thrust / (2 * rho * A))
    vi = vh
    for _ in range(iterations):
        new = thrust / (2 * rho * A * math.sqrt(max(va + vi, 0.0) ** 2 + vn ** 2 + 1e-9))
        vi = 0.5 * vi + 0.5 * new
    if va < -2 * vh:                                # windmill-brake state: momentum theory fails; keep the hover value
        vi = vh
    return vi


# ================================================================================================= the vehicle
@dataclass
class Vehicle:
    """What the transition model needs. ``airframe``: None (the analytic ``Polar`` on each wing part) or a
    ``kestrel_cfd.CFDAirframe`` whose ``correction(alpha, mu)`` → (ΔCL, ΔCD, ΔCm) — the CFD's coefficients minus the
    model's, μ = V / (V + Δv) the slipstream ratio — is added to the model's airframe forces."""
    name: str
    p: dict
    L: dict
    mass: float
    x_cg: float
    z_cg: float
    Iyy: float
    Ixx: float
    pol: object                     # the polar: cl_alpha, alpha0, alpha_stall, cl_max, cd0, k and its coefficient methods
    drive: object                   # the drive table: T_min and the full-throttle thrust at an axial inflow
    elevons: bool = True
    elevon_max_deg: float = 25.0
    elevon_tau: float = 0.45
    motor_tau: float = 0.06
    cd_pod_cross: float = 1.0
    airframe: object = None
    panels: list = None             # the lifting panels (see ``biplane_panels``); None: the biplane's from the layout
    z_rotor: float = None           # the upper rotors' height above the pod's axis (the differential thrust's lever); None: the gap's half
    elevon_y_in: float = None       # the elevons' inner end along the panel; None: the slipstream's inner edge at each instant
    lift_slope_vertical: float = None   # the aircraft's vertical lift slope on S_ref [1/rad] the autopilot trims with; None: the polar's
    prop_cn_alpha: Callable | None = None   # (thrust each [N], V [m/s]) -> one propeller's normal-force slope on q·A [1/rad]; None: no normal force
    V_cruise: float = 16.0          # the cruise airspeed the transitions start from / end at [m/s]

    def __post_init__(self):
        if self.panels is None:
            self.panels = biplane_panels(self.L)
        if self.z_rotor is None:
            self.z_rotor = self.L["h"]
        if self.elevon_y_in is None and self.L.get("elevons"):
            self.elevon_y_in = self.L["elevon_y_in"]
        if self.lift_slope_vertical is None:
            self.lift_slope_vertical = self.pol.cl_alpha

    @property
    def weight(self):
        return self.mass * G


def biplane_panels(L: dict) -> list:
    """The biplane as four flat panels: each wing's two halves from the pod to the tip, at z = ±h, the upper two blown
    by the upper rotors and the lower two by the lower. A panel: ``phi`` its dihedral about x (the lift normal tilted
    away from the vertical), ``z`` its mean height above the pod's axis, ``y0``–``y1`` its extent along its own span
    axis from the pod's axis, ``y_motor`` its rotor's station along that axis, ``rotor`` ``up`` or ``lo``, ``S`` its area,
    ``sy`` its side."""
    out = []
    for z, rotor in ((L["h"], "up"), (-L["h"], "lo")):
        for sy in (-1, 1):
            out.append({"phi": 0.0, "sy": sy, "z": z, "y0": L["pod_width"] / 2, "y1": L["span"] / 2, "y_motor": L["motor_y"],
                        "rotor": rotor, "S": L["S_wing"] / 2})
    return out


# ================================================================================================= the aerodynamics
def slipstream(veh: Vehicle, T: float, u: float, w: float, vi0: float | None = None, panel: dict | None = None) -> dict:
    """One rotor's slipstream at its panel: the induced velocity at the disk (Glauert), the velocity added at the
    wing's quarter chord, the slipstream's radius there and the blown length of the panel (``y_in``–``y_out`` along the
    panel; ``blown_span`` is that length — for the biplane's half-wing, one side)."""
    L = veh.L
    panel = panel or veh.panels[1]
    A, R = L["A_disc"], L["R_prop"]
    V = math.hypot(u, w)
    alpha = math.atan2(w, u) if V > 1e-6 else 0.0
    vi = induced_velocity(T, V, alpha, A) if vi0 is None else vi0
    if vi0 is not None and T > 0:                                    # a few fixed-point steps from the last value
        va, vn = u, w
        for _ in range(4):
            vi = 0.5 * vi + 0.5 * T / (2 * RHO * A * math.sqrt(max(va + vi, 0.0) ** 2 + vn ** 2 + 1e-9))
    d = L["x_ac"] - L["prop_x"]
    dv = vi * (1 + d / math.sqrt(R ** 2 + d ** 2))
    ua = max(u, 0.0)
    r_s = R * math.sqrt((ua + vi) / max(ua + dv, 1e-6)) if dv > 0 else R
    y_in = max(panel["y_motor"] - r_s, panel["y0"])
    y_out = min(panel["y_motor"] + r_s, panel["y1"])
    return {"vi": vi, "dv": dv, "r_s": r_s, "blown_span": max(y_out - y_in, 0.0), "y_in": y_in, "y_out": y_out}


def _part(veh, u_l, w_l, S, z_w, mu=1.0, phi=0.0):
    """Force (body x, z) and moment about the CG of a wing part of area S at height z_w in the local flow (u_l, w_l).
    A panel at the dihedral ``phi`` sees the normal component w cos φ of the vertical flow and gives cos φ of its
    normal force back as vertical force (the side components of the mirrored panels cancel)."""
    L = veh.L
    cph = math.cos(phi)
    w_l = w_l * cph
    V2 = u_l * u_l + w_l * w_l
    if V2 < 1e-8 or S <= 0:
        return 0.0, 0.0, 0.0, 0.0
    a_l = math.atan2(w_l, u_l)
    cl, cd, cm = veh.pol.coefficients(a_l)
    qS = 0.5 * RHO * V2 * S
    Vn = math.sqrt(V2)
    dx, dz = u_l / Vn, w_l / Vn                     # drag along the local flow (in the panel's plane)
    lx, lz = -dz, dx                                # lift perpendicular, "up" at α = 0
    Fx = qS * (float(cl) * lx + float(cd) * dx)
    Fz = qS * (float(cl) * lz + float(cd) * dz) * cph
    rx, rz = L["x_ac"] - veh.x_cg, z_w - veh.z_cg
    M = rz * Fx - rx * Fz + qS * L["chord"] * float(cm) * cph
    return Fx, Fz, M, a_l


def airframe_forces(veh: Vehicle, u: float, w: float, T_up: float, T_lo: float, vi=(None, None), *, props_normal: bool = True) -> dict:
    """The aerodynamic forces in body axes and the moment about the CG [N, N·m] of both wings (blown and free-stream
    parts), the pod's cross-flow and the four propellers' normal force (``props_normal``), for the relative air (u, w)
    and the rotor thrusts (each rotor of the upper pair T_up, of the lower T_lo); with the CFD's correction when the
    vehicle carries one."""
    L = veh.L
    V = math.hypot(u, w)
    Fx = Fz = M = 0.0
    out = {"alpha": math.atan2(w, u) if V > 1e-6 else math.pi / 2, "V": V, "panels": []}
    thrust = {"up": T_up, "lo": T_lo}
    vi_by = {"up": vi[0], "lo": vi[1]}
    for pn in veh.panels:
        T = thrust[pn["rotor"]]
        s = slipstream(veh, max(T, 0.0), u, w, vi_by[pn["rotor"]], pn)
        Sb = min(s["blown_span"], pn["y1"] - pn["y0"]) * L["chord"]
        Sf = pn["S"] - Sb
        mu = V / max(V + s["dv"], 1e-6)
        fb = _part(veh, u + s["dv"], w, Sb, pn["z"], mu, pn["phi"])
        ff = _part(veh, u, w, Sf, pn["z"], 1.0, pn["phi"])
        Fx += fb[0] + ff[0]; Fz += fb[1] + ff[1]; M += fb[2] + ff[2]
        out["panels"].append({"rotor": pn["rotor"], "slip": s, "S_blown": Sb, "alpha_blown": fb[3], "panel": pn})
        if f"slip_{pn['rotor']}" not in out:
            out[f"slip_{pn['rotor']}"] = s
            out[f"alpha_blown_{pn['rotor']}"] = fb[3]
            out[f"S_blown_{pn['rotor']}"] = 0.0
        out[f"S_blown_{pn['rotor']}"] += Sb
        # the panel's rotor's normal force, at the propeller plane
        if V > 0.2 and props_normal:
            a = math.atan2(w, u)
            N = (veh.prop_cn_alpha(T, V) if veh.prop_cn_alpha else 0.0) * 0.5 * RHO * V * V * L["A_disc"] * math.sin(a)
            Fz += N
            M += -(L["prop_x"] - veh.x_cg) * N
    # the pod in cross-flow, at its plan area's centre
    A_pod = L["pod_length"] * L["pod_width"]
    Fp = 0.5 * RHO * veh.cd_pod_cross * A_pod * w * abs(w)
    Fz += Fp
    M += -(0.45 * L["pod_length"] - veh.x_cg) * Fp
    if veh.airframe is not None and V > 0.5:                     # the CFD's correction, in the free stream's axes
        mu = V / (V + 0.5 * (out["slip_up"]["dv"] + out["slip_lo"]["dv"]))
        dCL, dCD, dCm = veh.airframe.correction(out["alpha"], mu)
        qS = 0.5 * RHO * V * V * L["S_ref"]
        Fx += qS * (dCL * (-w / V) + dCD * (u / V))
        Fz += qS * (dCL * (u / V) + dCD * (w / V))
        M += qS * L["chord"] * dCm
    out.update(Fx=Fx, Fz=Fz, M=M)
    return out


def elevon_moment_per_rad(veh: Vehicle, u: float, w: float, panels: list) -> tuple[float, float]:
    """The pitch moment [N·m] per radian of elevon (every panel's outer surface together, trailing edge down = negative
    moment: nose down) and the roll moment per radian used differentially, in the slipstream; the surfaces lose their
    effect as the blown panel's own angle of attack passes the stall (to nothing 30° beyond it). ``panels`` is
    ``airframe_forces``' per-panel list; the elevon runs from ``veh.elevon_y_in`` (or the slipstream's inner edge) to
    the panel's tip."""
    if not veh.elevons:
        return 0.0, 0.0
    L = veh.L
    pitch = roll = 0.0
    a_s = veh.pol.alpha_stall
    for pp in panels:
        s, pn = pp["slip"], pp["panel"]
        y_e = s["y_in"] if veh.elevon_y_in is None else max(veh.elevon_y_in, pn["y0"])
        span = max(pn["y1"] - y_e, 0.0)
        S_e = span * L["chord"]
        cph = math.cos(pn["phi"])
        ub = u + s["dv"]
        wn = w * cph
        V2 = ub * ub + wn * wn
        a_b = abs(math.atan2(wn, ub)) if V2 > 1e-8 else 0.0
        eff = float(np.clip(1.0 - (a_b - a_s) / math.radians(30.0), 0.0, 1.0)) if a_b > a_s else 1.0
        dL = 0.5 * RHO * V2 * S_e * veh.pol.cl_alpha * veh.elevon_tau * eff     # per radian, normal to the panel
        arm = L["x_le"] + 0.6 * L["chord"] - veh.x_cg
        pitch += dL * arm * cph
        roll += dL * 0.5 * (y_e + pn["y1"])
    return pitch, roll


# ================================================================================================= the simulation
@dataclass
class Episode:
    kind: str
    t_tr: float
    headwind: float
    vehicle: str
    elevons: bool
    log: dict = field(repr=False)
    metrics: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)


def _sched(tau):
    tau = min(max(tau, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * tau)


def _alpha_trim(veh, V):
    """The autopilot's trim angle of attack at V: a linear lift on the vehicle's vertical slope, capped short of the stall."""
    CL = veh.weight / (0.5 * RHO * V * V * veh.L["S_ref"])
    return min(CL / veh.lift_slope_vertical, veh.pol.alpha_stall * 0.85) + veh.pol.alpha0


def simulate(veh: Vehicle, kind: str = "forward", *, t_tr: float = 3.0, headwind: float = 0.0, gust_up: float = 0.0,
             V_cruise: float | None = None, V_entry: float | None = None, floor: float = 0.0, Z0: float = 30.0, dt: float = 0.002, t_hold: float = 1.5, t_after: float = 6.0,
             wn: float = WN, zeta: float = 0.8, airbrake_deg: float = 0.0, overshoot_deg: float = 0.0, reverse_N: float = 0.0, descent_entry: float = 0.0) -> Episode:
    """One transition. ``kind``: ``forward`` (hover → cruise) or ``back`` (cruise → hover). ``headwind`` [m/s] blows
    against the direction of flight; ``gust_up`` [m/s] is a 1-cos vertical gust of 1 s at the middle of the pitch
    schedule. The aircraft holds the hover (or the cruise) for ``t_hold`` s first and is followed ``t_after`` s after
    the schedule ends. ``V_entry``: the back transition starts from level flight at this airspeed (default the cruise);
    ``floor``: during the pitch schedule the collective never drops below this fraction of the hover thrust (keeps the
    slipstream on the wings and room for the rotors' torque).

    The back transition's remedies for a clean aircraft that balloons (notebook 42): ``airbrake_deg`` — through the
    schedule the four elevons are split as a crow brake (the upper wing's up, the lower's down: the pitch cancels, the
    drag of all four stays, the pitch travel left is what remains of ±``elevon_max_deg``); ``overshoot_deg`` — the pitch
    reference bulges past the vertical by this much at the schedule's middle (the thrust vector tilted aft brakes);
    ``reverse_N`` — bidirectional drives: through the schedule the least thrust is −``reverse_N`` per rotor instead of
    the idle, so the height loop can brake and pull down; ``descent_entry`` — the height reference descends at this
    rate [m/s] from the start to the schedule's middle (the balloon lands back on the datum; the height criteria are
    read against the reference)."""
    L = veh.L
    V_cruise = veh.V_cruise if V_cruise is None else V_cruise
    V_entry = V_cruise if V_entry is None else V_entry
    m, W, I = veh.mass, veh.weight, veh.Iyy
    h, zc = veh.z_rotor, veh.z_cg
    S = L["S_ref"]
    d = veh.drive
    wind = np.array([-headwind, 0.0])
    pol = veh.pol
    Vs = math.sqrt(2 * W / (RHO * S * pol.cl_max))

    # initial state
    if kind == "forward":
        X, Z, vx, vz, th, q = 0.0, Z0, 0.0, 0.0, math.pi / 2, 0.0
        mode = "hover"
        c0 = W / 4
    elif kind == "back":
        a0 = _alpha_trim(veh, V_entry)
        X, Z, vx, vz, th, q = 0.0, Z0, V_entry - headwind, -descent_entry, a0, 0.0
        mode = "cruise"
        CD = pol.cd0 + pol.k * (W / (0.5 * RHO * V_entry ** 2 * S)) ** 2
        c0 = max(0.5 * RHO * V_entry ** 2 * S * CD / 4, d.T_min)
        V_cruise = V_entry
    else:
        raise ValueError("kind is 'forward' or 'back'")
    T_act = np.array([c0, c0])                      # (upper rotor, lower rotor)
    vi = [None, None]
    th_start = None                                 # the reference at the schedule's start (no step in the reference)
    th_ref_prev = th
    th_end = _alpha_trim(veh, V_cruise) if kind == "forward" else math.pi / 2
    e_int = z_int = v_int = 0.0
    t_sched0, t_sched1 = t_hold, t_hold + t_tr
    t_end = t_sched1 + t_after
    n = int(round(t_end / dt)) + 1
    keys = ("t", "X", "Z", "Z_ref", "vx", "vz", "theta", "q", "theta_ref", "alpha", "alpha_blown_up", "alpha_blown_lo", "V", "T_up", "T_lo",
            "T_max", "delta", "M_cmd", "M_got", "M_aero", "collective_cmd", "sat_moment", "sat_collective", "margin", "P",
            "roll_demand", "roll_authority", "roll_authority_rotors", "mode", "lift_share")
    log = {k: [] for k in keys}
    E = 0.0
    done_at = None
    diverged = None
    for i in range(n):
        t = i * dt
        # ---- a diverged integration is a result (an airframe correction gone wrong, a polar that is not one), not a crash
        if not all(math.isfinite(v) for v in (X, Z, vx, vz, th, q)) or math.hypot(vx, vz) > V_DIVERGED or abs(q) > 200.0:
            diverged = t
            break
        # ---- the air
        g_up = gust_up * 0.5 * (1 - math.cos(2 * math.pi * (t - (t_sched0 + 0.5 * t_tr - 0.5)))) if abs(t - (t_sched0 + 0.5 * t_tr)) < 0.5 else 0.0
        rel = np.array([wind[0] - vx, g_up - vz])            # the air's velocity relative to the aircraft (earth)
        bx = np.array([-math.cos(th), -math.sin(th)])          # body x (aft) in the earth frame
        bz = np.array([-math.sin(th), math.cos(th)])           # body z
        u, w = float(rel @ bx), float(rel @ bz)
        V = math.hypot(u, w)
        aero = airframe_forces(veh, u, w, T_act[0], T_act[1], vi)
        vi = [aero["slip_up"]["vi"], aero["slip_lo"]["vi"]]
        Tmax = float(d.t_max(max(u, 0.0)))
        Tmin = d.T_min
        Zr = Z0 - descent_entry * min(t, t_sched0 + 0.5 * t_tr) if kind == "back" else Z0        # the height reference (a descent into the transition)
        # ---- guidance
        tau = (t - t_sched0) / t_tr
        v_air_fwd = -rel[0]                                      # airspeed along the flight direction (earth x)
        gamma = math.atan2(vz, max(vx - wind[0], 1e-3)) if V > 1.0 else 0.0
        # the autopilot's lift estimate: linear, no blowing, no stall
        a_est = th - gamma
        L_est = 0.5 * RHO * V * V * S * min(max(veh.lift_slope_vertical * math.sin(a_est) * math.cos(a_est), -pol.cl_max), pol.cl_max) if V > 1.0 else 0.0
        if kind == "forward":
            if t < t_sched0:
                mode = "hover"
            elif tau < 1.0 or v_air_fwd < 1.15 * Vs:
                mode = "transition"
            else:
                mode = "cruise"
        else:
            if t < t_sched0:
                mode = "cruise"
            elif tau < 1.0:
                mode = "transition"
            else:
                mode = "hover"
        if mode == "hover":
            ax_des = -1.2 * vx + (0.4 * (0.0 - X) if kind == "forward" else 0.0)
            th_ref = math.pi / 2 - math.atan2(float(np.clip(ax_des, -0.45 * G, 0.45 * G)), G)
            th_ref = th_ref_prev + float(np.clip(th_ref - th_ref_prev, -REF_RATE * dt, REF_RATE * dt))
        elif mode == "transition":
            if th_start is None:
                th_start = th_ref_prev
            th_ref = th_start + (th_end - th_start) * _sched(tau)
            if kind == "back" and overshoot_deg and 0.0 <= tau <= 1.0:                    # the bulge past the vertical: the thrust braking
                th_ref += math.radians(overshoot_deg) * math.sin(math.pi * tau)
            if kind == "back" and tau >= 1.0:
                th_ref = math.pi / 2
        else:                                                    # cruise: height by pitch
            th_ref = _alpha_trim(veh, max(v_air_fwd, Vs)) + 0.04 * (Zr - Z) - 0.05 * vz + 0.01 * z_int
            th_ref = float(np.clip(th_ref, -0.3, 0.6))
        th_ref_prev = th_ref
        # ---- collective thrust
        vz_cmd = float(np.clip(1.0 * (Zr - Z) + 0.1 * z_int - (descent_entry if (kind == "back" and t < t_sched0 + 0.5 * t_tr) else 0.0), -max(V_DESCENT, descent_entry), V_CLIMB))
        az_des = 2.5 * (vz_cmd - vz)
        if mode == "cruise":
            CL_c = W / (0.5 * RHO * max(V, 1.0) ** 2 * S)
            D_est = 0.5 * RHO * V * V * S * (pol.cd0 + pol.k * CL_c ** 2)
            T_cmd = D_est + m * (1.0 * (V_cruise - v_air_fwd) + 0.3 * v_int) + m * G * math.sin(gamma)
            v_int += (V_cruise - v_air_fwd) * dt
        else:
            Fz_req = m * (G + az_des) - L_est * math.cos(gamma)
            T_cmd = Fz_req / max(math.sin(th), 0.3)
            if mode == "hover" and kind == "back":
                T_cmd = (m * (G + az_des)) / max(math.sin(th), 0.5)
        z_int = float(np.clip(z_int + (Zr - Z) * dt, -20.0, 20.0))
        if reverse_N > 0 and mode == "transition" and kind == "back":                      # bidirectional drives: the rotors may brake
            Tmin = -reverse_N
        c_cmd = float(np.clip(T_cmd / 4, Tmin, Tmax))
        if mode == "transition" and floor > 0:
            c_cmd = max(c_cmd, min(floor * W / 4, Tmax))
        # ---- pitch: PID with the gains scheduled on the measured airspeed, no aerodynamic feed-forward
        e = math.atan2(math.sin(th_ref - th), math.cos(th_ref - th))
        e_int = float(np.clip(e_int + e * dt, -0.4, 0.4))
        qSc = 0.5 * RHO * V * V * S * L["chord"]
        Kp = I * wn * wn + K_AERO * qSc
        M_cmd = Kp * e - 2 * zeta * wn * I * q + KI_RATIO * Kp * e_int
        # ---- allocation: differential thrust first, elevons for the rest, the collective gives way
        Mp_rad, Mr_rad = elevon_moment_per_rad(veh, u, w, aero["panels"])
        dmax = math.radians(veh.elevon_max_deg)
        brake = math.radians(airbrake_deg) if (airbrake_deg and mode == "transition" and kind == "back" and veh.elevons) else 0.0
        dmax = max(dmax - brake, math.radians(2.0))                                         # the crow brake uses up travel
        D_lim = 0.5 * (Tmax - Tmin)
        delta = 0.0
        if Mp_rad > 1e-9:                                     # the elevons take their share of the moment by authority
            share = Mp_rad * dmax / (Mp_rad * dmax + 4 * h * D_lim)
            delta = float(np.clip(-share * M_cmd / Mp_rad, -dmax, dmax))
        Delta = (M_cmd + Mp_rad * delta - 4 * zc * c_cmd) / (4 * h)
        if abs(Delta) > D_lim:
            Delta_c = math.copysign(D_lim, Delta)
            c = 0.5 * (Tmax + Tmin)
            M_left = M_cmd - (4 * zc * c + 4 * h * Delta_c)
            if Mp_rad > 1e-9:
                delta = float(np.clip(-M_left / Mp_rad, -dmax, dmax))           # trailing edge down: nose down
            else:
                delta = 0.0
            Delta = Delta_c
        else:
            c = float(np.clip(c_cmd, Tmin + abs(Delta), Tmax - abs(Delta)))
        T_up_c, T_lo_c = c - Delta, c + Delta
        M_got_cmd = 4 * zc * c + 4 * h * Delta - Mp_rad * delta
        sat_m = abs(M_cmd - M_got_cmd) > 0.05 * abs(M_cmd) + 0.02
        sat_c = abs(c - c_cmd) > 0.05 * c_cmd + 0.05
        margin = 1.0 - abs(Delta) / D_lim if D_lim > 0 else 0.0
        # ---- roll check (not simulated): one side's free-stream panels stall first
        a_free = abs(aero["alpha"])
        demand = 0.0
        if V > 2.0 and pol.alpha_stall - math.radians(3) < a_free < math.radians(50):
            for pp in aero["panels"]:
                if pp["panel"]["sy"] > 0:                                   # one side's free-stream inner panels stall first
                    y0, y1 = pp["panel"]["y0"], pp["slip"]["y_in"]
                    S_side = max(y1 - y0, 0.0) * L["chord"]
                    demand += 0.5 * RHO * V * V * S_side * WING_DROP_DCL * 0.5 * (y0 + y1)
        kap = float(d.kap(max(u, 0.0)))
        # one diagonal pair (same sense, pitch- and yaw-neutral) up to T_max, the other down to T_min
        auth_rot = 2 * kap * (max(Tmax - max(T_up_c, T_lo_c), 0.0) + max(min(T_up_c, T_lo_c) - Tmin, 0.0))
        auth = auth_rot + Mr_rad * max(dmax - abs(delta), 0.0)
        # ---- the rotors' lag, the forces, the integration (semi-implicit Euler)
        T_act += (np.array([T_up_c, T_lo_c]) - T_act) * (dt / veh.motor_tau)
        T_act = np.clip(T_act, min(Tmin, 0.0), Tmax)
        T_tot = 2 * (T_act[0] + T_act[1])
        Fx_b = aero["Fx"] - T_tot
        if brake > 0 and V > 1e-3:                                                           # the crow brake's drag, along the relative air
            y_e = veh.elevon_y_in if veh.elevon_y_in is not None else L["motor_y"] - L["R_prop"]
            S_e4 = 4 * max(L["span"] / 2 - y_e, 0.0) * L["chord"]
            D_b = 0.5 * RHO * V * V * S_e4 * 1.2 * math.sin(brake) ** 2
            Fx_b += D_b * u / V
            Fz_brk = D_b * w / V
        else:
            Fz_brk = 0.0
        arm_e = L["x_le"] + 0.6 * L["chord"] - veh.x_cg
        Fz_b = aero["Fz"] + (Mp_rad * delta / arm_e if Mp_rad > 0 else 0.0) + Fz_brk     # the elevons' lift (trailing edge down: up)
        M_thr = 2 * (-(h - zc) * T_act[0] + (h + zc) * T_act[1])
        M_tot = aero["M"] + M_thr - Mp_rad * delta
        F = Fx_b * bx + Fz_b * bz
        ax_, az_ = F[0] / m, F[1] / m - G
        vx += ax_ * dt; vz += az_ * dt
        X += vx * dt; Z += vz * dt
        q += M_tot / I * dt
        th += q * dt
        P = 2 * d.power(abs(T_act[0]), max(u, 0.0)) + 2 * d.power(abs(T_act[1]), max(u, 0.0))
        E += P * dt
        lift_share = aero["Fz"] * math.cos(th) / W if V > 1 else 0.0
        for k, v in (("t", t), ("X", X), ("Z", Z), ("Z_ref", Zr), ("vx", vx), ("vz", vz), ("theta", th), ("q", q), ("theta_ref", th_ref),
                     ("alpha", aero["alpha"]), ("alpha_blown_up", aero["alpha_blown_up"]), ("alpha_blown_lo", aero["alpha_blown_lo"]),
                     ("V", V), ("T_up", T_act[0]), ("T_lo", T_act[1]), ("T_max", Tmax), ("delta", delta), ("M_cmd", M_cmd),
                     ("M_got", M_got_cmd), ("M_aero", aero["M"]), ("collective_cmd", c_cmd), ("sat_moment", sat_m),
                     ("sat_collective", sat_c), ("margin", margin), ("P", P), ("roll_demand", demand), ("roll_authority", auth),
                     ("roll_authority_rotors", auth_rot), ("mode", mode), ("lift_share", lift_share)):
            log[k].append(v)
        if kind == "back" and mode == "hover" and V < 0.5 and done_at is None:
            done_at = t
    log = {k: np.array(v) for k, v in log.items()}
    ep = Episode(kind, t_tr, headwind, veh.name, veh.elevons, log, options={"V_entry": V_entry, "floor": floor, "gust_up": gust_up})
    ep.metrics = metrics(ep, veh, dt, Z0, E, done_at, diverged=diverged)
    return ep


def metrics(ep: Episode, veh: Vehicle, dt: float, Z0: float, E: float, done_at, diverged=None) -> dict:
    g = ep.log
    if len(g["t"]) < 2:                                     # diverged before the second step: nothing to measure
        g = {k: (np.array([0.0, 0.0]) if k != "mode" else np.array(["hover", "hover"])) for k in g}
        g["Z"] = np.array([Z0, Z0]); g["Z_ref"] = np.array([Z0, Z0]); g["T_max"] = np.array([1.0, 1.0]); g["margin"] = np.array([0.0, 0.0])
    tr = (g["mode"] == "transition")
    a_deg = np.degrees(np.abs(g["alpha"]))
    # stalled in the transition (not the hover leaning into a headwind, where the free-stream wing is stalled as well)
    stall = (a_deg > math.degrees(veh.pol.alpha_stall)) & (a_deg < 80.0) & (g["V"] > 2.0) & tr
    blown = np.degrees(np.maximum(np.abs(g["alpha_blown_up"]), np.abs(g["alpha_blown_lo"])))
    ratio = np.where(g["roll_authority"] > 1e-6, g["roll_demand"] / np.maximum(g["roll_authority"], 1e-6), np.where(g["roll_demand"] > 0, np.inf, 0.0))
    ratio_rot = np.where(g["roll_authority_rotors"] > 1e-6, g["roll_demand"] / np.maximum(g["roll_authority_rotors"], 1e-6),
                         np.where(g["roll_demand"] > 0, np.inf, 0.0))
    perr = np.degrees(np.abs(np.arctan2(np.sin(g["theta_ref"] - g["theta"]), np.cos(g["theta_ref"] - g["theta"]))))
    out = {
        "height loss [m]": float((g["Z_ref"] - g["Z"]).max()) if "Z_ref" in g else float(Z0 - g["Z"].min()),
        "height gain [m]": float(g["Z"].max() - Z0),                      # above the start: a descent into the transition must land the balloon back on it
        "pitch error max [deg]": float(perr[tr].max()) if tr.any() else float(perr.max()),
        "α free-stream max in transition [deg]": float(a_deg[tr & (g["V"] > 2.0)].max()) if (tr & (g["V"] > 2.0)).any() else math.nan,
        "time stalled, free stream [s]": float(stall.sum() * dt),
        "α blown max while stalled [deg]": float(blown[stall].max()) if stall.any() else 0.0,
        "moment shortfall [s]": float(g["sat_moment"].sum() * dt),
        "collective given way [s]": float(g["sat_collective"].sum() * dt),
        "least pitch margin": float(g["margin"].min()),
        "peak rotor thrust / T_max": float(np.max(np.maximum(g["T_up"], g["T_lo"]) / g["T_max"])),
        "elevon max [deg]": float(np.degrees(np.abs(g["delta"]).max())),
        "roll demand / authority": float(ratio.max()),
        "roll demand / rotors' authority": float(ratio_rot.max()),
        "speed at the end [m/s]": float(g["V"][-1]), "distance [m]": float(g["X"][-1] - g["X"][0]),
        "energy [Wh]": E / 3600.0, "duration [s]": float(g["t"][-1]),
        "hover reached at [s]": done_at if done_at is not None else math.nan,
    }
    c = CRITERIA
    gain = c["height gain, forward [m]"] if ep.kind == "forward" else c["height gain, back [m]"]
    out["holds height"] = out["height loss [m]"] <= c["height loss [m]"] and out["height gain [m]"] <= gain
    out["holds attitude"] = out["pitch error max [deg]"] <= c["pitch error [deg]"] and out["moment shortfall [s]"] <= c["moment shortfall [s]"]
    out["roll holds"] = out["roll demand / authority"] <= c["roll demand / authority"]
    out["diverged at [s]"] = math.nan if diverged is None else float(diverged)
    out["pass"] = out["holds height"] and out["holds attitude"] and out["roll holds"] and diverged is None
    return out


def matrix(vehicles: dict, *, kinds=("forward", "back"), durations=(1.5, 3.0, 5.0), headwinds=(0.0, 6.0), gust_up=0.0,
           progress=False, **kw) -> tuple["pd.DataFrame", dict]:
    """Every vehicle × direction × schedule length × headwind: the metrics table and the episodes."""
    import pandas as pd
    jobs = [(vn, k, tt, hw) for vn in vehicles for k in kinds for tt in durations for hw in headwinds]
    it = jobs
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(jobs, desc="transitions")
    rows, eps = {}, {}
    for vn, k, tt, hw in it:
        ep = simulate(vehicles[vn], k, t_tr=tt, headwind=hw, gust_up=gust_up, **kw)
        key = (vn, k, tt, hw)
        rows[key] = ep.metrics
        eps[key] = ep
    df = pd.DataFrame(rows).T
    df.index.names = ["vehicle", "direction", "t_tr [s]", "headwind [m/s]"]
    return df, eps


def verdict(df) -> dict:
    """The answer per vehicle: does every transition pass, which fail and on what."""
    out = {}
    for vn, sub in df.groupby(level=0):
        fails = sub[~sub["pass"].astype(bool)]
        why = []
        for idx, r in fails.iterrows():
            reasons = [k for k in ("holds height", "holds attitude", "roll holds") if not bool(r[k])]
            if "diverged at [s]" in r and np.isfinite(float(r["diverged at [s]"])):
                reasons.append(f"diverged at {float(r['diverged at [s]']):.2f} s")
            why.append(f"{idx[1]} {idx[2]:g} s, headwind {idx[3]:g} m/s: fails {', '.join(reasons)}")
        out[vn] = {"passes": int(sub["pass"].astype(bool).sum()), "of": len(sub), "failures": why}
    return out


# ================================================================================================= remedy studies
def remedy_matrix(veh, remedies: dict, *, kind="back", durations=(1.5, 3.0, 5.0), headwinds=(0.0, 6.0), gust_up=0.0, progress=False):
    """A transition for every remedy × schedule × headwind: the metrics table and the episodes. ``remedies``:
    ``{name: simulate keyword arguments}`` (e.g. ``{"as is": {}, "reverse": {"reverse_N": 3.0}}``)."""
    import pandas as pd
    jobs = [(r, tt, hw) for r in remedies for tt in durations for hw in headwinds]
    it = jobs
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(jobs, desc=f"{veh.name}: {kind}-transition remedies")
    rows, eps = {}, {}
    for r, tt, hw in it:
        ep = simulate(veh, kind, t_tr=tt, headwind=hw, gust_up=gust_up, **remedies[r])
        rows[(r, tt, hw)] = ep.metrics
        eps[(r, tt, hw)] = ep
    df = pd.DataFrame(rows).T
    df.index.names = ["remedy", "t_tr [s]", "headwind [m/s]"]
    return df, eps


def remedy_verdict(df) -> "pd.DataFrame":
    """Per remedy: passes, the worst balloon, the worst pitch error, the energy, and the gusts' fate when given."""
    import pandas as pd
    rows = {}
    for r, sub in df.groupby(level=0, sort=False):
        ref = (3.0, 0.0) in [(a, b) for _, a, b in sub.index]
        rows[r] = {"passes": f"{int(sub['pass'].astype(bool).sum())} of {len(sub)}", "worst height gain [m]": float(sub["height gain [m]"].astype(float).max()),
                   "worst height loss [m]": float(sub["height loss [m]"].astype(float).max()), "worst pitch error [deg]": float(sub["pitch error max [deg]"].astype(float).max()),
                   "worst roll demand / authority": float(sub["roll demand / authority"].astype(float).max()),
                   "energy, 3 s still air [Wh]": float(sub.xs((3.0, 0.0), level=(1, 2))["energy [Wh]"].iloc[0]) if ref else np.nan,
                   "hover reached, 3 s [s]": float(sub.xs((3.0, 0.0), level=(1, 2))["hover reached at [s]"].iloc[0]) if ref else np.nan}
    return pd.DataFrame(rows).T
