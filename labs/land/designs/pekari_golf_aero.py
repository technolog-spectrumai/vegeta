"""The Pekari's golf-ball launcher: the aerodynamics and the machine worked out by hand (no CFD, no CAD kernel).

Everything here is a closed-form estimate or a short integration whose inputs are written down, so the CFD
(``pekari_golf_cfd``) and the CAD (``pekari_golf``) can be checked against it, and Armageddon's game model
(``armageddon/launcher.py``, ``balls.py``) can take its numbers from here.

Sections
--------
1. The ball and the air: Reynolds number, drag of a smooth sphere (Achenbach 1972, the drag crisis) and of a
   *dimpled* golf ball (Bearman & Harvey 1976; Smits & Smith 1994 for spin), lift from backspin (Magnus).
2. The flight: point-mass integration with drag, Magnus lift and gravity; range tables.
3. The flywheels: exit speed and spin from two rims, the energy a shot takes, the speed drop, spin-up power.
4. The tube: bore and clearance, the air the ball pushes ahead of it, friction, recoil.

The headline correction to the game model
-----------------------------------------
Armageddon flies the ball as a **smooth** sphere ("the tube is a smooth bore"). The bore does not take the dimples off:
a golf ball is dimpled whatever it is fired from, and its dimples trip the boundary layer at Re ~ 4e4..6e4, so between
~15 and ~100 m/s it flies with Cd ~ 0.25, *half* a smooth sphere's 0.47. At 45 m/s and 10 degrees that is the
difference between ~53 m and ~70 m. And two top/bottom flywheels at different speeds give backspin, which lifts.
``compare_game_model()`` puts the two side by side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# --- 1. the ball and the air ------------------------------------------------------------------

D_BALL = 0.04267          # m, the Rules of Golf minimum diameter
R_BALL = D_BALL / 2
M_BALL = 0.04593          # kg, the maximum mass
A_BALL = math.pi * R_BALL ** 2
I_BALL = 0.4 * M_BALL * R_BALL ** 2 * 0.93     # a two-piece ball is a little denser outside-in than uniform: k ~0.93
RHO = 1.225               # kg/m3, ISA sea level
NU = 1.46e-5              # m2/s
G = 9.81


def reynolds(v, d=D_BALL, nu=NU):
    return abs(v) * d / nu


def cd_smooth(re):
    """A smooth sphere (Achenbach 1972, Schlichting): ~0.47 subcritical, the crisis Re 2.5e5..4e5 down to ~0.10,
    recovering to ~0.18 by 1e6. Below Re 1e3 it rises (not needed here: 1e3 is 0.34 m/s)."""
    if re < 1e3:
        return 24 / max(re, 1e-9) * (1 + 0.15 * re ** 0.687) + 0.42 / (1 + 4.25e4 / re ** 1.16)
    if re <= 2.5e5:
        return 0.47
    if re <= 3.8e5:
        return 0.47 - 0.37 * (re - 2.5e5) / 1.3e5
    return min(0.18, 0.10 + 0.08 * (re - 3.8e5) / 6.2e5)


def cd_dimpled(re, spin_ratio=0.0):
    """A dimpled golf ball. Without spin (Bearman & Harvey 1976; Choi, Jeon & Choi 2006): subcritical ~0.50 below
    Re 4e4, the dimples' early crisis between 4e4 and 7e4, then a nearly flat ~0.25 up to 2.5e5+ (slowly rising,
    ~0.26 at 2e5). With spin ``S = omega r / V`` drag rises: Smits & Smith (1994), Cd = 0.24 + 0.18 S over Re 0.9e5..
    2.4e5, S 0.04..0.3; their 0.24 is the same plateau, so the spin term is added on top of the curve."""
    if re <= 4e4:
        base = 0.50
    elif re <= 7e4:
        base = 0.50 - 0.25 * (re - 4e4) / 3e4
    else:
        base = 0.25 + 0.01 * min(1.0, (re - 7e4) / 1.3e5)
    return base + 0.18 * max(spin_ratio, 0.0)


def cl_dimpled(re, spin_ratio):
    """Magnus lift of a spinning dimpled ball, Smits & Smith (1994) fit Cl = 0.54 S^0.4 (S 0.04..0.3); below the
    dimples' crisis the lift is weaker and can even reverse (Kim, Choi et al. 2014): taken as halved under Re 5e4."""
    if spin_ratio <= 0:
        return 0.0
    cl = 0.54 * min(spin_ratio, 0.4) ** 0.4
    return cl * (0.5 if re < 5e4 else 1.0)


def spin_decay_rate(v, omega):
    """Spin's own decay, d omega/dt = -k omega: Smits & Smith give ~4 %/s at driver speeds. Taken as the aerodynamic
    moment C_M = 0.012 S referred to the ball (k = C_M q A r / (I omega))."""
    if omega == 0 or v <= 0:
        return 0.0
    s = omega * R_BALL / v
    torque = 0.012 * s * 0.5 * RHO * v * v * A_BALL * D_BALL
    return torque / (I_BALL * abs(omega))


# --- 2. the flight ------------------------------------------------------------------------------

@dataclass
class Flight:
    range_m: float          # where it lands on flat ground (z back to the start height minus h0)
    apex_m: float
    time_s: float
    v_land: float
    path: list              # [(t, x, z, v)]


def fly(v0, elev_deg, *, h0=0.40, spin_rpm=0.0, ball="dimpled", dt=1e-3, wind=0.0):
    """Point-mass flight in the vertical plane: drag along -v, Magnus lift normal to v (backspin: up), gravity.
    ``h0``: the muzzle above the ground (the Pekari's roof plus the mount: ~0.4 m). ``ball``: 'dimpled', 'smooth' or
    'vacuum'. ``wind``: a head wind (+) along -x, m/s. RK4."""
    a = math.radians(elev_deg)
    omega = spin_rpm * 2 * math.pi / 60

    def deriv(s):
        x, z, vx, vz, w = s
        ux, uz = vx + wind, vz            # velocity relative to the air
        u = math.hypot(ux, uz)
        if ball == "vacuum" or u == 0:
            return (vx, vz, 0.0, -G, 0.0)
        re = reynolds(u)
        S = abs(w) * R_BALL / u
        if ball == "smooth":
            cd, cl = cd_smooth(re), (0.0 if w == 0 else 0.15 * min(S, 0.4) ** 0.4)   # a smooth ball's weak Magnus
        else:
            cd, cl = cd_dimpled(re, S), cl_dimpled(re, S)
        k = 0.5 * RHO * A_BALL * u / M_BALL
        sign = 1.0 if w >= 0 else -1.0
        ax = -k * cd * ux - sign * k * cl * uz
        az = -k * cd * uz + sign * k * cl * ux - G
        return (vx, vz, ax, az, -spin_decay_rate(u, w) * w)

    s = (0.0, h0, v0 * math.cos(a), v0 * math.sin(a), omega)
    t, apex, path = 0.0, h0, [(0.0, 0.0, h0, v0)]
    while True:
        k1 = deriv(s)
        k2 = deriv(tuple(s[i] + 0.5 * dt * k1[i] for i in range(5)))
        k3 = deriv(tuple(s[i] + 0.5 * dt * k2[i] for i in range(5)))
        k4 = deriv(tuple(s[i] + dt * k3[i] for i in range(5)))
        n = tuple(s[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(5))
        t += dt
        if n[1] < 0:                      # interpolate to z = 0
            f = s[1] / (s[1] - n[1])
            x = s[0] + f * (n[0] - s[0])
            v = math.hypot(s[2], s[3])
            path.append((t, x, 0.0, v))
            return Flight(x, apex, t - (1 - f) * dt, v, path)
        s = n
        apex = max(apex, s[1])
        if len(path) < 4000 and int(t / dt) % 10 == 0:
            path.append((t, s[0], s[1], math.hypot(s[2], s[3])))
        if t > 60:
            raise RuntimeError("no landing in 60 s")


def best_elevation(v0, **kw):
    """The elevation of the longest carry (golden-section on 5..50 degrees)."""
    lo, hi = 5.0, 50.0
    gr = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        a, b = hi - gr * (hi - lo), lo + gr * (hi - lo)
        if fly(v0, a, **kw).range_m < fly(v0, b, **kw).range_m:
            lo = a
        else:
            hi = b
    e = (lo + hi) / 2
    return e, fly(v0, e, **kw).range_m


def terminal_speed(cd):
    return math.sqrt(2 * M_BALL * G / (RHO * cd * A_BALL))


# --- 3. the flywheels ---------------------------------------------------------------------------

@dataclass
class Flywheels:
    """Two wheels one above the other (axes across the tube), the ball squeezed between their rims. A tennis-ball
    machine's layout: the top wheel faster than the bottom one gives topspin, slower gives backspin."""
    diameter: float = 0.100       # m
    width: float = 0.030          # m, rim width
    mass: float = 0.45            # kg each (an aluminium disc with a polyurethane tyre)
    radius_of_gyration: float = 0.80   # k / r: a disc is 0.707, a rim-heavy wheel more
    squeeze: float = 0.003        # m: the gap is the ball's diameter minus this (a hard golf ball: ~2..4 mm)
    grip: float = 0.97            # exit speed / mean rim speed (slip, the ball's compression)
    loss: float = 1.0             # energy lost in the nip per shot, in the ball's kinetic energy (hysteresis, slip)

    @property
    def inertia(self):
        return self.mass * (self.radius_of_gyration * self.diameter / 2) ** 2

    def gap(self):
        return D_BALL - self.squeeze

    def rpm_for(self, v_rim):
        return v_rim / (math.pi * self.diameter) * 60

    def exit(self, v_top, v_bottom):
        """(exit speed m/s, spin rpm; + = backspin) from the rims' surface speeds. The ball rolls between the two rims:
        its centre moves at the mean, it turns at the difference over its diameter. Bottom faster = backspin."""
        v = self.grip * (v_top + v_bottom) / 2
        omega = self.grip * (v_bottom - v_top) / D_BALL
        return v, omega * 60 / (2 * math.pi)

    def rims_for(self, v_exit, spin_rpm=0.0):
        """The rim speeds (top, bottom) for an exit speed and a backspin."""
        mean = v_exit / self.grip
        diff = spin_rpm * 2 * math.pi / 60 * D_BALL / self.grip
        return mean - diff / 2, mean + diff / 2

    def stored(self, v_top, v_bottom):
        r = self.diameter / 2
        return 0.5 * self.inertia * ((v_top / r) ** 2 + (v_bottom / r) ** 2)

    def shot(self, v_exit, spin_rpm=0.0):
        """What one shot costs: the ball's kinetic energy (+ spin) and the nip loss, taken from both wheels in
        proportion to what they store; returns the energies and the rims' speed drop (fraction)."""
        top, bot = self.rims_for(v_exit, spin_rpm)
        e0 = self.stored(top, bot)
        w = spin_rpm * 2 * math.pi / 60
        e_ball = 0.5 * M_BALL * v_exit ** 2 + 0.5 * I_BALL * w ** 2
        e_taken = e_ball * (1 + self.loss)
        drop = 1 - math.sqrt(max(0.0, 1 - e_taken / e0))
        return {"rim_top": top, "rim_bottom": bot, "rpm_top": self.rpm_for(top), "rpm_bottom": self.rpm_for(bot),
                "stored_J": e0, "ball_J": e_ball, "taken_J": e_taken, "speed_drop": drop}

    def power(self, v_exit, rate_hz, spin_rpm=0.0, spin_up_s=2.0):
        """Motor power: steady (shots per second times the energy each takes) and to spin up from rest in
        ``spin_up_s`` (the mean power; the peak at a constant torque is twice that)."""
        s = self.shot(v_exit, spin_rpm)
        return {"steady_W": s["taken_J"] * rate_hz, "spin_up_mean_W": s["stored_J"] / spin_up_s}

    def squeeze_force(self, stiffness=3.5e5):
        """Normal force in the nip for the squeeze. A golf ball is stiff: a "compression 90" ball deflects ~2.5 mm under
        200 lbf (890 N), so ~350 kN/m; the tyres' own compliance is taken as inside ``squeeze``."""
        return stiffness * self.squeeze

    def max_tangential(self, mu=0.8):
        """The most a rubber rim can push along the ball before it slips: mu N, per wheel; both wheels."""
        return 2 * mu * self.squeeze_force()

    def nip_check(self, v_exit, mu=0.8):
        """Does the grip hold? The ball is accelerated from ~0 to v over the contact arc; the arc length is about
        the rims' angle in contact (from the squeeze) times r. Mean force needed = m v^2 / (2 s)."""
        r = self.diameter / 2
        half_angle = math.acos(1 - (self.squeeze / 2) / r) + math.asin(min(1.0, R_BALL / (r + R_BALL)) * 0.25)
        s = 2 * half_angle * r + 0.010       # contact arc plus the ball's own flattening (~10 mm)
        f_need = M_BALL * v_exit ** 2 / (2 * s)
        return {"contact_m": s, "force_needed_N": f_need, "force_available_N": self.max_tangential(mu),
                "slips": f_need > self.max_tangential(mu)}


# --- 4. the tube --------------------------------------------------------------------------------

@dataclass
class Tube:
    bore: float = 0.0445          # m: 42.67 ball + 1.8 mm diametral clearance
    length: float = 0.40          # m, from the wheels' nip to the muzzle
    vents: bool = True            # slots along the first 2/3 let the air ahead escape sideways

    def clearance(self):
        return (self.bore - D_BALL) / 2

    def blockage(self):
        return (D_BALL / self.bore) ** 2

    def piston(self, v):
        """The ball pushing the tube's air out of the muzzle. Unvented: the air column ahead (up to the full length)
        leaves at about the ball's speed; ahead of the ball the pressure is the column's dynamic pressure plus
        the annulus' jet: the annulus passes the leak flow U_leak with Δp = ½ρ (v_gap)²/Cd_orifice². A quasi-steady
        estimate, upper bound: the pressure on the ball and the work it costs over the tube."""
        a_ann = math.pi / 4 * (self.bore ** 2 - D_BALL ** 2)
        a_bore = math.pi / 4 * self.bore ** 2
        # all the displaced air has to go: ahead out of the muzzle (the column moves with the ball) ⇒ the front sees
        # roughly the stagnation of that column; behind, the vacuum is fed back through the annulus
        q = 0.5 * RHO * v ** 2
        dp = q * (1.0 if not self.vents else 0.25)
        f = dp * A_BALL
        work = f * self.length * (2 / 3)
        return {"annulus_mm2": a_ann * 1e6, "blockage": (a_bore - a_ann) / a_bore, "dp_Pa": dp, "force_N": f,
                "work_J": work, "speed_loss_frac": work / (0.5 * M_BALL * v * v) / 2}

    def bounce_time(self, v):
        return self.length / v

    def recoil(self, v):
        """Momentum the launcher takes per shot (the ball's, the air's is ~1 %), and the mean force over the nip."""
        return {"impulse_Ns": M_BALL * v, "per_100_balls_Ns": 100 * M_BALL * v}


# --- the summary tables -------------------------------------------------------------------------

def range_table(speeds=(20, 30, 45, 60, 70), elevs=(5, 10, 20, 30), h0=0.40):
    """Rows: speed, elevation, carry for vacuum, smooth (Armageddon's model), dimpled, dimpled with 3000 rpm backspin."""
    rows = []
    for v in speeds:
        for e in elevs:
            rows.append({"v0": v, "elev": e,
                         "vacuum": fly(v, e, ball="vacuum", h0=h0).range_m,
                         "smooth": fly(v, e, ball="smooth", h0=h0).range_m,
                         "dimpled": fly(v, e, ball="dimpled", h0=h0).range_m,
                         "dimpled_3000rpm": fly(v, e, ball="dimpled", spin_rpm=3000, h0=h0).range_m})
    return rows


def compare_game_model(v=45.0, elev=10.0):
    """Armageddon's numbers (docs/golf.md: 53 m at 45 m/s and 10 degrees, 86 m without air, terminal 33 m/s)
    against this module's."""
    return {
        "smooth": fly(v, elev, ball="smooth", h0=0.0).range_m,
        "vacuum": fly(v, elev, ball="vacuum", h0=0.0).range_m,
        "dimpled": fly(v, elev, ball="dimpled", h0=0.0).range_m,
        "dimpled_backspin_3000": fly(v, elev, ball="dimpled", spin_rpm=3000, h0=0.0).range_m,
        "terminal_smooth": terminal_speed(0.47),
        "terminal_dimpled": terminal_speed(0.25),
        "deceleration_at_60_smooth": 0.5 * RHO * 0.47 * A_BALL * 60 ** 2 / M_BALL,
        "deceleration_at_60_dimpled": 0.5 * RHO * cd_dimpled(reynolds(60)) * A_BALL * 60 ** 2 / M_BALL,
    }


def report(v=45.0, spin=0.0):
    fw, tb = Flywheels(), Tube()
    re = reynolds(v)
    out = {
        "Re": re, "Cd_smooth": cd_smooth(re), "Cd_dimpled": cd_dimpled(re),
        "drag_N_dimpled": 0.5 * RHO * v * v * A_BALL * cd_dimpled(re),
        "wheel": fw.shot(v, spin), "nip": fw.nip_check(v), "power_2.5Hz": fw.power(v, 2.5, spin),
        "tube": tb.piston(v), "clearance_mm": tb.clearance() * 1e3, "recoil": tb.recoil(v),
        "game": compare_game_model(v),
    }
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(report(), indent=1, default=float))
    for r in range_table(speeds=(45,), elevs=(10, 20)):
        print({k: round(x, 1) for k, x in r.items()})
