"""STERNA's flight model and MuJoCo twin (notebook 51) — the lab's fixed-wing model, carried with STERNA.

The force model (aero: lift, drag and moments from a coefficient set, with the stall blend), the rigid-body
derivative, trim and modes, and the MuJoCo side (mjcf, Sim, Autopilot) are the ARIGO study's
(arigo1_model / arigo1_mujoco on its branch), copied here so STERNA stands on its own where that study is not
merged. Nothing is changed in the physics; Sim takes a coefficient set instead of an ARIGO variant name.
Body frame: x forward, y right, z down; controls in degrees. sterna_model.coefficients builds the set.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import mujoco
except ImportError:                      # the force model, trim and modes need no MuJoCo
    mujoco = None

RHO, G = 1.225, 9.80665


TAU_E, TAU_A, TAU_R = 0.55, 0.50, 0.55   # HAND: flap effectiveness for 35/25/40 % chord (Raymer fig. 16.7, +-0.1)


DE_MAX, DA_MAX, DR_MAX = 25.0, 20.0, 25.0  # deg, servo travel


# ----------------------------------------------------------------------------------------------------------- forces
def max_thrust(C, V) -> float:
    """Full-throttle net thrust [N] at airspeed V: the propulsor map's table (``sterna_model.thrust_table``) when the
    coefficients carry one, else the old hand line T0 (1 - V/V0)."""
    th = C["thrust"]
    if "table_V" in th:
        return float(max(np.interp(V, th["table_V"], th["table_T"]), 0.0))
    return max(th["T0_N"] * (1 - V / th["V0_m_s"]), 0.0)


def _cfd_static(t, al):
    """CL, CD, Cm at alpha [rad] from the CFD table; outside it (beyond -4..24 deg) a blend onto the flat plate over
    6 degrees, Cm carried on with the table's last slope (bounded) — HAND outside the table."""
    a = math.degrees(al)
    A, CLt, CDt, Cmt = t["alpha_deg"], t["CL"], t["CD"], t["Cm"]
    if A[0] <= a <= A[-1]:
        return float(np.interp(a, A, CLt)), float(np.interp(a, A, CDt)), float(np.interp(a, A, Cmt))
    lo = a < A[0]
    i, j = (0, 1) if lo else (-1, -2)
    d = a - A[i]
    sl = lambda y: (y[i] - y[j]) / (A[i] - A[j])
    w = min(1.0, abs(d) / 6.0)
    fp_cl, fp_cd = 2 * math.sin(al) * math.cos(al) * 0.9, 2 * math.sin(al) ** 2 * 0.9 + float(CDt.min())
    cl = (1 - w) * (CLt[i] + sl(CLt) * d) + w * fp_cl
    cd = (1 - w) * (CDt[i] + sl(CDt) * d) + w * fp_cd
    cm = float(np.clip(Cmt[i] + sl(Cmt) * d, -1.2, 1.2))
    return float(cl), float(cd), cm


def aero(C: dict, vel_b, omega_b, de=0.0, da=0.0, dr=0.0, throttle=0.5):
    """Force and moment in the body frame (x fwd, y right, z down) about the CG [N, N m]; controls in deg."""
    u, v, w = vel_b
    p, q, r = omega_b
    V = max(math.sqrt(u * u + v * v + w * w), 0.5)
    al = math.atan2(w, u)
    be = math.asin(max(-1.0, min(1.0, v / V)))
    qd = 0.5 * RHO * V * V
    S, c, b = C["S"], C["c"], C["b"]
    s = 1 / (1 + math.exp(-(abs(al) - C["alpha_stall"]) / math.radians(2.0)))
    if "cfd" in C:
        CL, CD, Cm_static = _cfd_static(C["cfd"], al)
        Cm_static += C.get("Cm_rig", 0.0)
    else:
        # lift: linear to the stall, then a smooth fall onto the flat plate (2 sin a cos a) — HAND
        CL_lin = C["CL0"] + C["CLa"] * al
        CL_fp = 2 * math.sin(al) * math.cos(al)
        CL = (1 - s) * CL_lin + s * CL_fp * 0.9
        CD = C["CD0"] + C["k"] * CL_lin ** 2 * (1 - s) + s * (2 * math.sin(al) ** 2 * 0.9 + C["CD0"])
        k_ps = C.get("post_stall_cm_factor", 1.0)       # HAND: 1 = the stability slope kept past the stall; 0.35 = mostly lost
        Cm_static = C["Cm0"] + C["Cma"] * ((1 - s) * al + s * (C["alpha_stall"] + k_ps * (abs(al) - C["alpha_stall"])) * math.copysign(1, al))
    CL += C["CLde"] * de + C["CLq"] * q * c / (2 * V)
    CY = C["CYb"] * be + C["CYdr"] * dr
    Cm = Cm_static + C["Cmq"] * q * c / (2 * V) + C["Cmde"] * de
    CLu = max(CL, 0.0)
    Cl = (C["Clb"] + C["Clb_sweep_per_CL"] * CLu) * be + C["Clp"] * p * b / (2 * V) + C["Clr_per_CL"] * CLu * r * b / (2 * V) \
        + C["Clda"] * da + C["Cldr"] * dr
    Cl *= (1 - 0.6 * s)                                             # HAND: the ailerons and roll damping fade in the stall
    Cn = C["Cnb"] * be + C["Cnp_per_CL"] * CLu * p * b / (2 * V) + C["Cnr"] * r * b / (2 * V) + C["Cnda"] * da + C["Cndr"] * dr
    ca, sa, cb, sb = math.cos(al), math.sin(al), math.cos(be), math.sin(be)
    # wind -> body: drag along -V, lift perpendicular in the x-z plane, side force along y
    Fx = qd * S * (-CD * ca * cb + CL * sa)
    Fy = qd * S * (-CD * sb + CY)
    Fz = qd * S * (-CD * sa * cb - CL * ca)
    T = max_thrust(C, u) * max(min(throttle, 1.0), 0.0)
    M_thrust = -C.get("thrust_z_m", 0.0) * T                       # a thrust line above the CG pitches the nose down
    return np.array([Fx + T, Fy, Fz]), np.array([qd * S * b * Cl, qd * S * c * Cm + M_thrust, qd * S * b * Cn]), {"alpha": al, "beta": be, "V": V, "CL": CL, "stall": s}


# ----------------------------------------------------------------------------------------- rigid body (numpy twin)
def _R(phi, th, psi):
    cf, sf, ct, st, cp, sp = math.cos(phi), math.sin(phi), math.cos(th), math.sin(th), math.cos(psi), math.sin(psi)
    return np.array([[ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp],
                     [ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp],
                     [-st, sf * ct, cf * ct]])                       # body -> NED


def deriv(C, x, u):
    """x = [u v w p q r phi theta psi N E D]; u = (de, da, dr, throttle)."""
    vb, wb, (phi, th, psi) = x[0:3], x[3:6], x[6:9]
    F, M, _ = aero(C, vb, wb, *u)
    m, I = C["mass"], np.array(C["inertia"])
    gb = _R(phi, th, psi).T @ np.array([0, 0, G])
    dv = F / m + gb - np.cross(wb, vb)
    dw = (M - np.cross(wb, I * wb)) / I
    p, q, r = wb
    dang = np.array([p + (q * math.sin(phi) + r * math.cos(phi)) * math.tan(th), q * math.cos(phi) - r * math.sin(phi),
                     (q * math.sin(phi) + r * math.cos(phi)) / math.cos(th)])
    return np.concatenate([dv, dw, dang, _R(phi, th, psi) @ vb])


def trim(C, V: float = 18.0) -> dict:
    """Level flight at V: alpha, elevator, throttle. Bracketing, robust on the CFD's piecewise-linear tables: at each
    alpha the elevator that zeroes the pitch acceleration and the throttle that zeroes the axial one, then the alpha
    where the vertical acceleration crosses zero (scan, then Brent). Raises if there is none."""
    from scipy.optimize import brentq

    def d(al, de, th):
        return deriv(C, np.array([V * math.cos(al), 0, V * math.sin(al), 0, 0, 0, 0, al, 0, 0, 0, 0]), (de, 0, 0, th))

    def inner(al):
        de, th = 0.0, 0.4
        for _ in range(3):
            fq = lambda x: d(al, x, th)[4]
            try:
                de = brentq(fq, -60.0, 60.0, xtol=1e-9)
            except ValueError:
                return None
            u0, u1 = d(al, de, 0.0)[0], d(al, de, 1.0)[0]
            th = -u0 / (u1 - u0) if u1 != u0 else 0.0
        return de, th

    def fz(al):
        r = inner(al)
        return math.nan if r is None else d(al, *r)[2]
    grid = np.radians(np.arange(-4.0, 30.01, 0.5))
    vals = [fz(a) for a in grid]
    root = None
    for k in range(len(grid) - 1):
        if vals[k] == vals[k] and vals[k + 1] == vals[k + 1] and vals[k] * vals[k + 1] <= 0:
            root = brentq(fz, grid[k], grid[k + 1], xtol=1e-10)
            break
    if root is None:
        raise np.linalg.LinAlgError(f"no trim at {V} m/s")
    de, th = inner(root)
    return {"V": V, "alpha_deg": math.degrees(root), "elevator_deg": float(de), "throttle": float(th),
            "ok": bool(0 <= th <= 1 and abs(de) < DE_MAX)}


def modes(C, V: float = 18.0) -> dict:
    """The linear modes at the level trim (finite differences of ``deriv``): period, damping ratio, time to half
    (or double) amplitude for each."""
    t = trim(C, V)
    al = math.radians(t["alpha_deg"])
    x0 = np.array([V * math.cos(al), 0, V * math.sin(al), 0, 0, 0, 0, al, 0, 0, 0, 0])
    u0 = (t["elevator_deg"], 0, 0, t["throttle"])
    n = 9
    A = np.zeros((n, n))
    f0 = deriv(C, x0, u0)[:n]
    for j in range(n):
        dx = np.zeros(12); dx[j] = 1e-5
        A[:, j] = (deriv(C, x0 + dx, u0)[:n] - f0) / 1e-5
    lon, lat = [0, 2, 4, 7], [1, 3, 5, 6]
    out = {"trim": t}
    for name, idx in (("longitudinal", lon), ("lateral", lat)):
        ev = np.linalg.eigvals(A[np.ix_(idx, idx)])
        rows = []
        for e in sorted(ev, key=lambda z: -abs(z)):
            if e.imag < -1e-9:
                continue
            wn = abs(e)
            rows.append({"real": float(e.real), "imag": float(e.imag), "omega_n": float(wn), "zeta": float(-e.real / wn) if wn else None,
                         "period_s": float(2 * math.pi / e.imag) if e.imag > 1e-9 else None,
                         "t_half_or_double_s": float(math.log(2) / abs(e.real)) if abs(e.real) > 1e-9 else None, "stable": bool(e.real < 0)})
        out[name] = rows
    return out


def name_modes(md: dict) -> dict:
    """Short period / phugoid; roll subsidence / Dutch roll / spiral — by the eigenvalues' shape."""
    lon = md["longitudinal"]; lat = md["lateral"]
    osc = sorted([r for r in lon if r["period_s"]], key=lambda r: -r["omega_n"])
    named = {}
    if osc:
        named["short_period"] = osc[0]
        if len(osc) > 1:
            named["phugoid"] = osc[-1]
    real = sorted([r for r in lat if not r["period_s"]], key=lambda r: r["real"])
    if real:
        named["roll_subsidence"] = real[0]
        if len(real) > 1:
            named["spiral"] = real[-1]
    o = [r for r in lat if r["period_s"]]
    if o:
        named["dutch_roll"] = o[0]
    return named


# ------------------------------------------------------------------------------------------------- MuJoCo
DT = 0.002


FLIP = np.array([1.0, -1.0, -1.0])               # FLU <-> FRD


def mjcf(C: dict, mesh: str | None = None, bandit: bool = False) -> str:
    """The model as MJCF: ground, the aircraft (a free body with the model's mass and inertia, a belly box for contact,
    the CAD as a visual mesh when given), optionally a bandit (a mocap box)."""
    Ixx, Iyy, Izz = C["inertia"]
    b, L = C["b"], 0.72
    vis = ""
    if mesh:
        # the CAD frame: x aft from the nose, z up, mm -> the body: x forward, origin at the CG
        vis = (f'<geom type="mesh" mesh="sterna" contype="0" conaffinity="0" rgba="0.75 0.72 0.62 1" '
               f'pos="{C["x_cg_m"]:.4f} 0 {-C["z_cg_m"]:.4f}" euler="0 0 180"/>')
    else:
        vis = (f'<geom type="box" size="{L / 2:.3f} 0.035 0.035" contype="0" conaffinity="0" rgba="0.7 0.7 0.6 1"/>'
               f'<geom type="box" size="0.12 {b / 2:.3f} 0.008" pos="0 0 0" contype="0" conaffinity="0" rgba="0.75 0.72 0.62 1"/>')
    asset = f'<mesh name="sterna" file="{mesh}" scale="0.001 0.001 0.001"/>' if mesh else ""
    band = ('<body name="bandit" mocap="true" pos="40 0 30"><geom type="box" size="0.18 0.18 0.05" rgba="0.8 0.15 0.1 1" '
            'contype="0" conaffinity="0"/></body>') if bandit else ""
    return f"""<mujoco model="sterna">
  <option timestep="{DT}" gravity="0 0 -9.80665" density="0" viscosity="0" integrator="RK4"/>
  <statistic extent="10" center="0 0 30"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/><map znear="0.005" zfar="300"/></visual>
  <asset>{asset}
    <texture name="grid" type="2d" builtin="checker" rgb1="0.32 0.42 0.30" rgb2="0.28 0.37 0.26" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="200 200"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.7 0.9" rgb2="0.9 0.93 0.97" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="0 0 300" dir="0.2 0.3 -1" directional="true"/>
    <geom type="plane" size="2000 2000 1" material="grid"/>
    <body name="sterna" pos="0 0 30">
      <freejoint/>
      <inertial pos="0 0 0" mass="{C['mass']:.4f}" diaginertia="{Ixx:.5f} {Iyy:.5f} {Izz:.5f}"/>
      <geom name="belly" type="box" size="0.2 0.04 0.02" pos="0 0 -0.045" rgba="0 0 0 0" friction="0.5 0.01 0.001"/>
      {vis}
    </body>
    {band}
  </worldbody>
</mujoco>"""


class Sim:
    """One aircraft in MuJoCo. ``controls`` (de, da, dr [deg], throttle 0-1) are held between steps; ``wind`` [m/s,
    world]."""

    def __init__(self, C: dict, mesh=None):
        self.C = C
        self.m = mujoco.MjModel.from_xml_string(mjcf(self.C, mesh))
        self.d = mujoco.MjData(self.m)
        self.bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "sterna")
        self.controls = np.array([0.0, 0.0, 0.0, 0.5])
        self.wind = np.zeros(3)
        self.info = {}

    def start(self, V=18.0, h=40.0, heading_deg=0.0):
        """Level trimmed flight at V, altitude h, heading (0 = +x). Above the top level speed (no trim: the thrust
        is short) the controls of the 26 m/s trim, at V — as if diving into the manoeuvre."""
        try:
            t = trim(self.C, V)
            assert t["ok"]
        except (np.linalg.LinAlgError, AssertionError):
            t = {**trim(self.C, 26.0), "V": V, "ok": False, "note": "beyond the top level speed: entered from a dive"}
        al = math.radians(t["alpha_deg"])
        mujoco.mj_resetData(self.m, self.d)
        ps = math.radians(heading_deg)
        q_yaw = np.array([math.cos(ps / 2), 0, 0, math.sin(ps / 2)])
        q_pitch = np.array([math.cos(-al / 2), 0, math.sin(-al / 2), 0])     # nose up = -rotation about y_left? (y left: + is nose down)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, q_yaw, q_pitch)
        self.d.qpos[:3] = (0, 0, h)
        self.d.qpos[3:7] = q
        self.d.qvel[:3] = (V * math.cos(ps), V * math.sin(ps), 0.0)
        self.controls[:] = (t["elevator_deg"], 0, 0, t["throttle"])
        self.trim = t
        mujoco.mj_forward(self.m, self.d)
        return t

    # ----------------------------------------------------------------------------------------------------- state
    def R(self):
        return self.d.xmat[self.bid].reshape(3, 3)

    def state(self) -> dict:
        R = self.R()
        v_w = self.d.qvel[:3].copy()
        v_air = R.T @ (v_w - self.wind) * FLIP
        w = self.d.qvel[3:6] * FLIP
        x_b, y_b, z_b = R[:, 0], R[:, 1], R[:, 2]
        theta = math.asin(max(-1, min(1, x_b[2])))
        phi = math.atan2(y_b[2], z_b[2])
        psi = math.atan2(x_b[1], x_b[0])
        return {"pos": self.d.qpos[:3].copy(), "v_world": v_w, "v_air_frd": v_air, "omega_frd": w, "phi": phi, "theta": theta, "psi": psi,
                "V": float(np.linalg.norm(v_air)), "t": self.d.time}

    def step(self, n=1):
        for _ in range(n):
            s = self.state()
            F, M, info = aero(self.C, s["v_air_frd"], s["omega_frd"], *self.controls)
            R = self.R()
            self.d.xfrc_applied[self.bid, :3] = R @ (F * FLIP)
            self.d.xfrc_applied[self.bid, 3:] = R @ (M * FLIP)
            self.info = {**info, "load_factor": float(-F[2] / (self.C["mass"] * G))}
            mujoco.mj_step(self.m, self.d)


class Autopilot:
    """Bank and load factor (or pitch angle) commands -> elevator, aileron; a turn coordinator on the rudder; speed
    on the throttle. Gains by hand for the model's modes (short period ~7 rad/s, roll time constant ~0.2 s)."""

    def __init__(self, sim: Sim):
        self.s = sim
        self.i_q = 0.0

    def __call__(self, phi_c=0.0, n_c=None, theta_c=None, V_c=18.0, throttle=None, protect=True):
        s, C = self.s.state(), self.s.C
        p, q, r = s["omega_frd"]
        V = max(s["V"], 3.0)
        if n_c is None:
            th_c = 0.0 if theta_c is None else theta_c
            n_c = math.cos(s["theta"]) / max(math.cos(s["phi"]), 0.3) + 2.0 * (th_c - s["theta"]) * V / G
        n_av = 0.5 * RHO * V * V * C["S"] * C["CL_MAX"] / (C["mass"] * G)     # what the wing can give at this speed
        n_c = max(-0.5 * n_av, min(n_c, 0.9 * n_av))                                  # the alpha limiter
        q_c = G * (n_c - math.cos(s["theta"]) * math.cos(s["phi"])) / V
        e = q_c - q
        self.i_q = max(-1.0, min(1.0, self.i_q + e * DT))
        de = self.s.trim["elevator_deg"] - (180 / math.pi) * (0.4 * e + 1.2 * self.i_q) * (18.0 / V)
        vb = s["v_air_frd"]
        al = math.atan2(vb[2], vb[0])
        a_lim = C["alpha_stall"] - math.radians(2.0)                  # alpha protection: nose down past the limit
        if protect and al > a_lim:
            de += 4.0 * math.degrees(al - a_lim)
            self.i_q = min(self.i_q, 0.0) if e > 0 else self.i_q
        err_phi = (phi_c - s["phi"] + math.pi) % (2 * math.pi) - math.pi
        p_c = max(-6.0, min(6.0, 5.0 * err_phi))
        da = (180 / math.pi) * 0.12 * (p_c - p) * (18.0 / V) ** 2
        r_c = G * math.sin(s["phi"]) * math.cos(s["theta"]) / V
        dr = -(180 / math.pi) * 0.3 * (r_c - r)
        thr = throttle if throttle is not None else max(0.0, min(1.0, self.s.trim["throttle"] + 0.15 * (V_c - s["V"])))
        self.s.controls[:] = (max(-DE_MAX, min(DE_MAX, de)), max(-DA_MAX, min(DA_MAX, da)),
                              max(-DR_MAX, min(DR_MAX, dr)), thr)
