"""A grappling harpoon for the Pekari: an air cannon throws a four-claw hook trailing a climbing line over a parapet,
the claws bite, and a person (Batman, 100 kg with his gear) climbs the line.

Built on the golf air cannon (``golf_gun``: tank -> real valve -> barrel, 0-D interior ballistics) with what a hook
on a line adds:

1. **The hook.** A hub (shank) with four claws on hinge pins. Folded, the claws lie back along the shank and the
   bore holds them shut; out of the muzzle a torsion spring swings each to 50 degrees, where a stop holds it. Sized
   for the climb: a claw must take the whole design load alone (the worst case: one claw on an edge).
2. **The line.** 6 mm Dyneema (SK78) core with a polyester cover: ~28 g/m, MBS ~ 20 kN; a static line, so a
   climber's bounce is the dynamic factor below. It pays out from a flaked bag on the Pekari's basket.
3. **The flight with the line.** The hook drags the line: the payout (line taken from rest to the hook's speed:
   tension mu v^2), the airborne line's weight (half of it on the hook), skin friction on the moving line; plus the
   hook's own drag and gravity. It must clear the parapet and drop behind it.
4. **The launch.** ``gun``: ``golf_gun``'s 0-D gun generalised to any bore and mass. The tank and valve are sized
   for the speed the flight needs.
5. **The rover.** The shot's recoil on a 24 kg Pekari: what the tracks' friction holds back, and how far it slides.
6. **The climb.** Design load, claw bending, the line's margin, the parapet edge.

Units SI unless named (``_mm``, ``_bar``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

G = 9.81
RHO = 1.204
P_AMB, T_AMB = 101325.0, 293.15


# --- 1. the hook -----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Material:
    name: str
    density: float          # kg/m3
    yield_mpa: float


STEEL_17_4 = Material("17-4PH H900 stainless", 7800.0, 1170.0)
AL_7075 = Material("7075-T6 aluminium", 2810.0, 500.0)
TI_6AL4V = Material("Ti-6Al-4V", 4430.0, 880.0)


@dataclass(frozen=True)
class Hook:
    shank_od_mm: float = 30.0
    shank_len_mm: float = 160.0
    shank_wall_mm: float = 4.0
    claw_len_mm: float = 95.0            # hinge pin to tip
    claw_width_mm: float = 12.0          # b, across the bending plane
    claw_depth_mm: float = 18.0          # h, in the bending plane (the load bends it about its width)
    claw_open_deg: float = 50.0          # from the shank axis, held by the stop
    tip_radius_mm: float = 6.0           # the bite: the tip's hook-back
    n_claws: int = 4
    claw_mat: Material = STEEL_17_4
    shank_mat: Material = TI_6AL4V
    nose_mm: float = 40.0                # a rubber nose cone on the shank's front (a softer landing)

    @property
    def folded_od_mm(self):
        """The bore it must fit: claws folded flat along the shank."""
        return self.shank_od_mm + 2 * self.claw_depth_mm + 2.0

    def mass(self):
        shank = math.pi / 4 * (self.shank_od_mm ** 2 - (self.shank_od_mm - 2 * self.shank_wall_mm) ** 2) * self.shank_len_mm * 1e-9 * self.shank_mat.density
        claws = self.n_claws * self.claw_width_mm * self.claw_depth_mm * self.claw_len_mm * 0.85e-9 * self.claw_mat.density
        hardware = 0.060 + 0.030 + 0.040          # pins + springs, the eye and the swivel, the nose cone
        return {"shank": shank, "claws": claws, "hardware": hardware, "total": shank + claws + hardware}

    def claw_check(self, load_n):
        """One claw takes ``load_n`` at its tip, the line pulling along the shank. The claw is a cantilever from its
        pin: the moment is load x the tip's distance from the shank axis line (claw length x sin(open angle))."""
        arm = self.claw_len_mm * 1e-3 * math.sin(math.radians(self.claw_open_deg))
        m = load_n * arm
        z = self.claw_width_mm * self.claw_depth_mm ** 2 / 6 * 1e-9
        sigma = m / z / 1e6
        return {"moment_Nm": m, "stress_MPa": sigma, "yield_MPa": self.claw_mat.yield_mpa,
                "safety_on_yield": self.claw_mat.yield_mpa / sigma}

    def pin_check(self, load_n, pin_mm=6.0, pin_mpa=1100.0):
        """The hinge pin in double shear under the claw's reaction (the stop takes the moment; the pin the force
        ~ load / cos(open))."""
        f = load_n / math.cos(math.radians(self.claw_open_deg))
        tau = f / (2 * math.pi / 4 * (pin_mm * 1e-3) ** 2) / 1e6
        return {"pin_force_N": f, "shear_MPa": tau, "safety": 0.6 * pin_mpa / tau}

    def frontal_area(self):
        return math.pi / 4 * (self.folded_od_mm * 1e-3) ** 2

    def cd(self):
        """In flight, claws open: a bluff, cross-shaped body nose first; ~1.0 on the folded diameter's area
        (Hoerner: open frames 1.0..1.2, a blunt body with fins 0.8). Rough, +-30 %."""
        return 1.0


# --- 2. the line -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Line:
    name: str = "6 mm Dyneema SK78 core, polyester cover"
    diameter_mm: float = 6.0
    mass_g_per_m: float = 28.0
    mbs_kn: float = 20.0
    cf: float = 0.012                    # skin friction on its wetted area, moving along itself
    cd_normal: float = 1.2               # across it

    @property
    def mu(self):
        return self.mass_g_per_m * 1e-3


# --- 3. the flight -------------------------------------------------------------------------------------------------

@dataclass
class Flight:
    path: list                           # (t, x, z, v, tension)
    cleared: bool
    clearance_m: float                   # above the parapet's top when over its x
    landed_x: float                      # where it comes down behind (on the roof at the parapet's height)
    line_out_m: float
    max_tension_n: float


def fly(v0, elev_deg, hook: Hook, line: Line, *, wall_x=10.0, wall_h=15.0, h0=0.45, roof_depth=8.0, dt=5e-4):
    """The hook from the muzzle (height ``h0``) at ``v0`` and ``elev_deg`` toward a wall ``wall_x`` metres away whose
    parapet's top is ``wall_h`` high; the roof behind it is ``roof_depth`` deep, at the parapet's height.

    Forces on the hook: gravity, its drag, and the line's pull back along the line (taken along -v):
      payout          mu v^2                (the bag is at rest: each metre is taken from 0 to v)
      airborne weight 0.5 mu g s            (s: the line out; half its weight hangs on the hook, half on the rover)
      skin friction   0.5 rho v^2 cf pi d s / 3   (the line's speed falls ~linearly from v at the hook to 0)
    The first two dominate; the result is checked against the parapet's x and height."""
    m = hook.mass()["total"]
    a = math.radians(elev_deg)
    x, z, vx, vz, s, t = 0.0, h0, v0 * math.cos(a), v0 * math.sin(a), 0.0, 0.0
    k_hook = 0.5 * RHO * hook.cd() * hook.frontal_area()
    path, cleared, clearance, tmax = [], None, None, 0.0
    while t < 20:
        v = math.hypot(vx, vz)
        tension = line.mu * v * v + 0.5 * line.mu * G * s + 0.5 * RHO * v * v * line.cf * math.pi * line.diameter_mm * 1e-3 * s / 3
        tmax = max(tmax, tension)
        f = k_hook * v * v + tension
        ax, az = -f * vx / (v * m), -f * vz / (v * m) - G
        nx, nz = x + vx * dt, z + vz * dt
        if clearance is None and x < wall_x <= nx:
            zw = z + (nz - z) * (wall_x - x) / (nx - x)
            clearance = zw - wall_h
            cleared = clearance > 0.3                 # 0.3 m over the coping: the claws are ~0.15 m across
            if not cleared:
                path.append((t, wall_x, zw, v, tension))
                return Flight(path, False, clearance, float("nan"), s, tmax)
        if clearance is not None and nz < wall_h:     # down on the roof
            path.append((t, nx, wall_h, v, tension))
            return Flight(path, True, clearance, nx - wall_x, s, tmax)
        if nz < 0:
            return Flight(path, False, -wall_h, float("nan"), s, tmax)
        s += v * dt
        vx, vz = vx + ax * dt, vz + az * dt
        x, z, t = nx, nz, t + dt
        if int(t / dt) % 20 == 0:
            path.append((t, x, z, v, tension))
    return Flight(path, False, clearance or -1, float("nan"), s, tmax)


def min_speed(elev_deg, hook, line, *, land_behind=(0.5, 6.0), **kw):
    """The slowest muzzle speed that clears the parapet and lands 0.5..6 m behind it (on the roof, short of its far
    edge): bisection on the clearance, then the landing checked."""
    lo, hi = 5.0, 120.0
    for _ in range(40):
        mid = (lo + hi) / 2
        f = fly(mid, elev_deg, hook, line, **kw)
        if f.cleared and f.landed_x >= land_behind[0]:
            hi = mid
        else:
            lo = mid
    f = fly(hi, elev_deg, hook, line, **kw)
    return hi, f


def best_elevation(hook, line, **kw):
    rows = []
    for e in range(40, 81, 2):
        v, f = min_speed(e, hook, line, **kw)
        rows.append((v, e, f))
    return min(rows, key=lambda r: r[0])


# --- 4. the launch ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Gun:
    bore_mm: float
    barrel_m: float
    tank_l: float
    tank_rated_bar: float
    valve_name: str
    valve_orifice_mm: float
    valve_cd: float
    valve_open_s: float
    friction_n: float = 15.0             # the sabot cup's seal plus the claws' tips on the bore


def gun(p_gauge, g: Gun, mass, dt=2e-6, gamma=1.4, R=287.05, cp=1005.0):
    """``golf_gun.gun`` for any bore and projectile mass (the same lumped model: isentropic tank, a valve of area
    cd*A ramped open, choked/subsonic flow, adiabatic gas behind the projectile, a piston-driven shock ahead).
    Returns the muzzle speed, the time in the barrel, the air used and the peak force on the projectile."""
    A = math.pi / 4 * (g.bore_mm * 1e-3) ** 2
    av_full = g.valve_cd * math.pi / 4 * (g.valve_orifice_mm * 1e-3) ** 2
    p0i = P_AMB + p_gauge * 1e5
    V0 = g.tank_l * 1e-3
    m = m0 = p0i * V0 / (R * T_AMB)
    x0 = 0.03
    Vb0 = A * x0
    Eb = P_AMB * Vb0 / (gamma - 1)
    x = u = t = 0.0
    fmax = 0.0
    crit = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    a_snd = math.sqrt(gamma * R * T_AMB)
    travel = g.barrel_m - x0
    while x < travel and t < 0.5:
        rho0 = m / V0
        p0 = p0i * (rho0 / (p0i / (R * T_AMB))) ** gamma
        T0 = p0 / (rho0 * R)
        pb = (gamma - 1) * Eb / (Vb0 + A * x)
        av = av_full * min(1.0, t / g.valve_open_s)
        r = pb / p0
        if r <= crit:
            mdot = av * p0 / math.sqrt(R * T0) * math.sqrt(gamma) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
        elif r < 1:
            mdot = av * p0 / math.sqrt(R * T0) * math.sqrt(2 * gamma / (gamma - 1) * (r ** (2 / gamma) - r ** ((gamma + 1) / gamma)))
        else:
            mdot = 0.0
        M = u / a_snd
        pf = P_AMB * (1 + gamma * (gamma + 1) / 4 * M * M + gamma * M * math.sqrt(1 + ((gamma + 1) / 4) ** 2 * M * M))
        F = A * (pb - pf)
        fmax = max(fmax, F)
        Fn = F - g.friction_n if (u > 0 or F > g.friction_n) else 0.0
        m -= mdot * dt
        Eb += (mdot * cp * T0 - pb * A * u) * dt
        u = max(0.0, u + Fn / mass * dt)
        x += u * dt
        t += dt
    return {"muzzle": u if x >= travel else 0.0, "time_s": t, "air_kg": m0 - m, "peak_force_n": fmax,
            "p_end_bar_g": (p0 - P_AMB) / 1e5}


def size(g: Gun, mass, target):
    lo, hi = 0.1, 300.0
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if gun(mid, g, mass, dt=5e-6)["muzzle"] < target:
            lo = mid
        else:
            hi = mid
    return hi


# --- 5. the rover ------------------------------------------------------------------------------------------------

def recoil(mass_proj, v, elev_deg, air_kg, *, rover_kg=24.0, mu_track=0.7, jet_speed=300.0, shot_s=0.02):
    """Momentum to the rover: the projectile's and the air's (leaving at ~``jet_speed``, the gas's exit speed), along
    the barrel. Its downward part presses the tracks into the ground, its horizontal part pushes the rover back. The
    tracks hold back mu * (m g + F_down) for the shot's duration; what is left moves the rover."""
    J = mass_proj * v + air_kg * jet_speed
    a = math.radians(elev_deg)
    jh, jv = J * math.cos(a), J * math.sin(a)
    hold = mu_track * (rover_kg * G * shot_s + jv)
    slip = max(0.0, jh - hold)
    v_back = slip / rover_kg
    slide = v_back ** 2 / (2 * mu_track * G)
    return {"impulse_Ns": J, "horizontal_Ns": jh, "down_Ns": jv, "tracks_hold_Ns": hold, "rover_kick_mps": v_back,
            "slide_m": slide, "mean_force_N": J / shot_s}


# --- 6. the climb ------------------------------------------------------------------------------------------------

CLIMBER_KG = 100.0          # Batman with his suit, belt and boots
DYNAMIC = 3.0               # a static line: a climber's bounce, a slip and catch on the hands, the hook settling
SAFETY = 2.0


def climb_check(hook: Hook, line: Line):
    static = CLIMBER_KG * G
    design = static * DYNAMIC * SAFETY
    return {"static_N": static, "design_N": design, "claw": hook.claw_check(design), "pin": hook.pin_check(design),
            "line_margin": line.mbs_kn * 1e3 / (static * DYNAMIC), "line_mbs_N": line.mbs_kn * 1e3}


# --- the design point ---------------------------------------------------------------------------------------------

HOOK = Hook()
LINE = Line()
GUN = Gun(bore_mm=72.0, barrel_m=0.70, tank_l=3.0, tank_rated_bar=16.0, valve_name='1.5" diaphragm pulse valve',
          valve_orifice_mm=38.0, valve_cd=0.70, valve_open_s=8e-3)
SABOT_KG = 0.025            # a PE cup behind the hook that seals the bore and drops off at the muzzle


def report(wall_x=10.0, wall_h=15.0, margin=1.10):
    hook, line, g = HOOK, LINE, GUN
    m = hook.mass()["total"]
    v_min, e, f = best_elevation(hook, line, wall_x=wall_x, wall_h=wall_h)
    v_design = v_min * margin
    # take the elevation that gives the most clearance with margin at v_design
    fd = fly(v_design, e, hook, line, wall_x=wall_x, wall_h=wall_h)
    p = size(g, m + SABOT_KG, v_design)
    shot = gun(p, g, m + SABOT_KG)
    return {
        "hook_mass": hook.mass(), "folded_od_mm": hook.folded_od_mm, "bore_mm": g.bore_mm,
        "target": {"wall_x": wall_x, "wall_h": wall_h},
        "elevation_deg": e, "v_min": v_min, "v_design": v_design,
        "flight": {"clearance_m": fd.clearance_m, "landed_behind_m": fd.landed_x, "line_out_m": fd.line_out_m,
                   "max_tension_N": fd.max_tension_n, "time_s": fd.path[-1][0]},
        "no_line_v_min": min_speed(e, hook, replace(line, mass_g_per_m=1e-6, cf=0.0), wall_x=wall_x, wall_h=wall_h)[0],
        "tank_p_bar_g": p, "shot": shot, "energy_J": 0.5 * m * v_design ** 2,
        "recoil": recoil(m + SABOT_KG, shot["muzzle"], e, shot["air_kg"]),
        "climb": climb_check(hook, line),
        "line_needed_m": math.hypot(wall_x, wall_h) + 3 + 2,       # to the parapet, over it to the hook, to the ground
    }


def target_table(targets=((5, 8), (10, 10), (10, 15), (10, 20), (15, 25), (20, 30))):
    rows = []
    m = HOOK.mass()["total"] + SABOT_KG
    for wx, wh in targets:
        v, e, f = best_elevation(HOOK, LINE, wall_x=wx, wall_h=wh)
        vd = 1.1 * v
        p = size(GUN, m, vd)
        rows.append({"wall_x": wx, "wall_h": wh, "elev": e, "v_min": v, "v_design": vd, "tank_bar_g": p,
                     "within_rating": p <= GUN.tank_rated_bar, "line_out": f.line_out_m})
    return rows


if __name__ == "__main__":
    import json
    print(json.dumps(report(), indent=1, default=float))
    for r in target_table():
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
