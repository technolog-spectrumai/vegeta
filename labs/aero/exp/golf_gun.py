"""An air cannon for golf balls (the Pekari's launcher, pneumatic version): tank -> valve -> barrel -> flight.

Built on the rubber-ball blowdown study (``sonic_ball``, notebook 51; ``sonic_flight`` for the flight): the same 0-D
interior ballistics extended with a *real valve* (its own flow area, discharge coefficient and opening time), the
same transient CFD (``rhoCentralFoam`` on a moving layered mesh, the ball a six-DoF piston) on the golf rig, and a
flight with Mach-dependent drag for a dimpled ball.

Three cases, each with hardware one can buy:

=============  ==========================================  ==========================================
case           tank                                        valve
=============  ==========================================  ==========================================
slow           1.0 L aluminium, shop air, ~2 bar g          1/2" direct solenoid (12 mm orifice,
                                                            Cd 0.6, opens in 25 ms)
subsonic fast  3.0 L aluminium (16 bar rated), ~7 bar g     1.5" diaphragm pulse valve (38 mm, Cd 0.7,
                (under a compressor's 8 bar)                opens in 8 ms)
near sonic     300 bar HPA bottle through a regulator       1.5" hammer (piston) valve, 38 mm, Cd 0.75,
               into a 2.0 L steel plenum (100 bar) ~50 bar  2 ms; the plenum is what is fired
=============  ==========================================  ==========================================

The barrel: 600 mm, bore 42.7 mm (the ball's 42.67 + a 3 g PE cup behind it that seals: a golf ball alone in a
bore leaks round its dimples). The pressure in each case is *sized* (``size``) for the target muzzle speed.

Modelling choices stated: ideal gas (CoolProp's real-gas factor at 40 bar is 0.99), adiabatic tank and barrel,
friction of the cup a constant 4 N, the air ahead pushed by a piston-driven shock (as in ``sonic_ball.gun``). The CFD's
valve is ideal (opens fully at t = 0, full bore); the 0-D runs both the ideal and the real valve, which is the
valve's own cost.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import sonic_ball as sb
import sonic_flow as sf

D_BALL = 0.04267
M_BALL = 0.04593 + 0.003           # ball + sealing cup
BORE = 0.0427
A_BORE = math.pi / 4 * BORE ** 2
BARREL = 0.600                      # m
P_AMB, T_AMB = 101325.0, 293.15
A_SOUND = math.sqrt(1.4 * 287.05 * T_AMB)


@dataclass(frozen=True)
class Tank:
    name: str
    volume_l: float
    rated_bar: float                # what the vessel is rated to (gauge)
    material: str
    bore_mm: float                  # inner diameter (for the CAD / the CFD container)


@dataclass(frozen=True)
class Valve:
    name: str
    orifice_mm: float
    cd: float
    open_s: float                   # time from shut to fully open (area ramps linearly)

    @property
    def area(self):
        return math.pi / 4 * (self.orifice_mm * 1e-3) ** 2


@dataclass(frozen=True)
class Case:
    name: str
    target: float                   # m/s at the muzzle
    tank: Tank
    valve: Valve
    p_gauge: float = 0.0            # bar, set by ``size``


CASES = (
    Case("slow", 25.0, Tank("1 L aluminium, shop air", 1.0, 10.0, "6061-T6", 80.0),
         Valve('1/2" direct solenoid', 12.0, 0.60, 25e-3)),
    Case("subsonic fast", 120.0, Tank("3 L aluminium, compressor air", 3.0, 16.0, "6061-T6", 110.0),
         Valve('1.5" diaphragm (pulse-jet / QEV)', 38.0, 0.70, 8e-3)),
    Case("near sonic", 0.85 * A_SOUND, Tank("2 L steel plenum off a 300 bar HPA bottle + regulator", 2.0, 100.0, "steel 4130", 90.0),
         Valve('1.5" hammer (piston) valve', 38.0, 0.75, 2e-3)),
)


def gun(p_gauge, tank: Tank, valve: Valve | None, friction=4.0, dt=1e-6, t_max=0.2, gamma=1.4, R=287.05):
    """0-D interior ballistics, ``sonic_ball.gun`` with a real valve: the tank (isentropic) feeds the barrel behind the
    ball through the valve's area cd*A(t) (ramped open over ``open_s``; ``valve=None``: full bore at once). Choked or
    subsonic isentropic flow; the gas behind is well mixed and adiabatic; the air ahead a piston-driven shock."""
    g = gamma
    p0i = P_AMB + p_gauge * 1e5
    V0 = tank.volume_l * 1e-3
    m = p0i * V0 / (R * T_AMB)
    x0 = 0.02                                       # dead volume behind the ball (valve to cup): 20 mm of bore
    Vb0 = A_BORE * x0
    Eb = P_AMB * Vb0 / (g - 1)
    x = u = t = 0.0
    travel = BARREL - x0 - D_BALL
    crit = (2 / (g + 1)) ** (g / (g - 1))
    rows = []
    k = 0
    while x < travel and t < t_max:
        rho0 = m / V0
        p0 = p0i * (rho0 / (p0i / (R * T_AMB))) ** g
        T0 = p0 / (rho0 * R)
        pb = (g - 1) * Eb / (Vb0 + A_BORE * x)
        if valve is None:
            a_v = A_BORE
        else:
            a_v = valve.cd * valve.area * min(1.0, t / valve.open_s if valve.open_s > 0 else 1.0)
        r = pb / p0
        if r <= crit:
            mdot = a_v * p0 / math.sqrt(R * T0) * math.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
        elif r < 1.0:
            mdot = a_v * p0 / math.sqrt(R * T0) * math.sqrt(2 * g / (g - 1) * (r ** (2 / g) - r ** ((g + 1) / g)))
        else:
            mdot = 0.0
        M = u / A_SOUND
        pf = P_AMB * (1 + g * (g + 1) / 4 * M * M + g * M * math.sqrt(1 + ((g + 1) / 4) ** 2 * M * M))
        F = A_BORE * (pb - pf)
        Fn = F - friction if (u > 0 or F > friction) else 0.0
        a = Fn / M_BALL
        if k % 20 == 0:
            rows.append((t, x, u, p0, pb, pf))
        k += 1
        m -= mdot * dt
        Eb += (mdot * 1005.0 * T0 - pb * A_BORE * u) * dt
        u = max(0.0, u + a * dt)
        x += u * dt
        t += dt
    rows.append((t, x, u, p0, pb, pf))
    out = dict(zip(("t", "x", "u", "p_tank", "p_behind", "p_ahead"), np.array(rows).T))
    out["muzzle"] = u if x >= travel else 0.0
    out["time"] = t
    # air used: the tank's mass drop (what a shot costs the bottle)
    out["air_g"] = (p0i * V0 / (R * T_AMB) - m) * 1e3
    return out


def size(case: Case, lo=0.05, hi=200.0, real_valve=True):
    """The tank's gauge pressure (bar) that gives the target muzzle speed, by bisection."""
    v = case.valve if real_valve else None
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if gun(mid, case.tank, v, dt=2e-6)["muzzle"] < case.target:
            lo = mid
        else:
            hi = mid
    return replace(case, p_gauge=round(hi, 3))


# --- the flight --------------------------------------------------------------------------------------------

def cd_golf(M, re):
    """A dimpled golf ball against Mach: the subsonic curve of ``pekari_golf``'s hand calculation (0.50 below Re 4e4,
    the dimples' crisis to 0.25 by 7e4, ~0.26 plateau), rising through the transonic range like ``sonic_flight``'s
    sphere (its wave drag: +0.5 by Mach 1, from Mach 0.6; Hoerner ch. 16). A ±20 % value at Mach 0.8+."""
    if re <= 4e4:
        sub = 0.50
    elif re <= 7e4:
        sub = 0.50 - 0.25 * (re - 4e4) / 3e4
    else:
        sub = 0.25 + 0.01 * min(1.0, (re - 7e4) / 1.3e5)
    return float(sf_ramp(M, sub, sub + 0.50, 0.6, 1.0))


def sf_ramp(M, sub, sonic, m0, m1):
    s = min(max((M - m0) / (m1 - m0), 0.0), 1.0)
    return sub + (sonic - sub) * 0.5 * (1 - math.cos(math.pi * s))


def fly(v0, elev_deg, h0=0.4, drag=True, dt=2e-4):
    rho, nu = 1.204, 1.51e-5
    A = math.pi / 4 * D_BALL ** 2
    m = 0.04593
    a = math.radians(elev_deg)
    x, z, vx, vz, t = 0.0, h0, v0 * math.cos(a), v0 * math.sin(a), 0.0
    apex = h0
    while z >= 0:
        v = math.hypot(vx, vz)
        k = 0.5 * rho * cd_golf(v / A_SOUND, v * D_BALL / nu) * A * v / m if drag else 0.0
        vx += -k * vx * dt
        vz += (-k * vz - 9.81) * dt
        x += vx * dt
        z += vz * dt
        t += dt
        apex = max(apex, z)
    return dict(range=x, apex=apex, time=t, v_land=math.hypot(vx, vz))


def best(v0):
    e = max(np.arange(5, 46, 1.0), key=lambda e: fly(v0, e, dt=1e-3)["range"])
    return e, fly(v0, e)


# --- the transient CFD (sonic_ball's case on the golf rig) ------------------------------------------------

def cfd_rig(case: Case):
    t = case.tank
    return sf.Rig(pipe_bore=BORE * 1e3, pipe_length=BARREL * 1e3, pipe_wall=3.0, valve_x=20.0,
                  container_volume_l=t.volume_l, container_bore=t.bore_mm * 2.5, container_wall=4.0,
                  ambient_length=300.0, ambient_radius=150.0)


def cfd_ball():
    # sonic_ball's slug: a sealing piston one diameter long with the ball + cup's mass (density from the mass)
    vol = math.pi / 6 * (BORE) ** 3
    return sb.Ball(diameter=BORE * 1e3, density=M_BALL / vol, rear_x=20.5, stop_front_mm=BARREL * 1e3 - 8.0,
                   contact_pressure=4.0 / (0.7 * math.pi * BORE * 3e-3))


def cfd(case: Case, root=Path("output/golf_gun"), mesh=sf.Mesh(pipe=1.5, container_far=12.0, ambient_far=12.0),
        run=True, max_time=0.08):
    rig, ball = cfd_rig(case), cfd_ball()
    # the CFD's valve is ideal (full bore at t = 0), so it is fired at the pressure that gives the case's muzzle
    # speed through an ideal valve: the barrel's physics at the case's speed, against gun(valve=None)
    p_ideal = size(case, real_valve=False).p_gauge
    cond = replace(sb.BALL_CONDITIONS, p0=P_AMB + p_ideal * 1e5)
    path = Path(root) / case.name.replace(" ", "_")
    sb.solve(path, sf.AIR, rig, cond, mesh, ball, run=run, max_time=max_time, chunk=max(25e-6, 1e-4 * 25 / case.target))
    track = sb.ball_track(path)
    return dict(track=track, muzzle=sb.muzzle(track, ball, rig), p_ideal=p_ideal,
                gun_ideal=gun(p_ideal, case.tank, None)["muzzle"])


def summary(cases=CASES):
    rows = []
    for c in cases:
        c = size(c)
        real = gun(c.p_gauge, c.tank, c.valve)
        ideal = gun(c.p_gauge, c.tank, None)
        e, fl = best(real["muzzle"])
        rows.append(dict(case=c.name, tank=c.tank.name, valve=c.valve.name, p_bar_g=c.p_gauge, rated=c.tank.rated_bar,
                         muzzle=real["muzzle"], mach=real["muzzle"] / A_SOUND, ideal_valve_muzzle=ideal["muzzle"],
                         barrel_ms=real["time"] * 1e3, air_g=real["air_g"], energy_J=0.5 * 0.04593 * real["muzzle"] ** 2,
                         r10=fly(real["muzzle"], 10)["range"], r30=fly(real["muzzle"], 30)["range"],
                         best_elev=e, best_range=fl["range"], vacuum_best=real["muzzle"] ** 2 / 9.81,
                         flight_s=fl["time"]))
    return rows


if __name__ == "__main__":
    for r in summary():
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})


# --- CAD (Dedalus / CadQuery): tank, valve, barrel, ball, per case ------------------------------------------

def wall_mm(tank: Tank, safety=3.0):
    """Thin-wall hoop stress t = p r SF / sigma_y (6061-T6 240 MPa, 4130 460 MPa), at the rated pressure."""
    sy = 240e6 if tank.material.startswith("6061") else 460e6
    return max(1.5, tank.rated_bar * 1e5 * tank.bore_mm / 2 * safety / sy)


def cad(case: Case):
    """The gun along +x: tank (cylinder with hemispherical heads, the wall from ``wall_mm``), the valve body (a
    block with the orifice), the barrel (600 mm, bore 42.7, 3 mm wall), the cup and the ball at the breech."""
    import cadquery as cq
    t, v = case.tank, case.valve
    w = wall_mm(t)
    r = t.bore_mm / 2
    l_cyl = max(0.0, (t.volume_l * 1e6 - 4 / 3 * math.pi * r ** 3) / (math.pi * r * r))
    outer = (cq.Workplane("YZ").circle(r + w).extrude(l_cyl).union(cq.Workplane("XY").sphere(r + w))
             .union(cq.Workplane("XY").sphere(r + w).translate((l_cyl, 0, 0))))
    inner = (cq.Workplane("YZ").circle(r).extrude(l_cyl).union(cq.Workplane("XY").sphere(r))
             .union(cq.Workplane("XY").sphere(r).translate((l_cyl, 0, 0))))
    tank = outer.cut(inner).translate((-(l_cyl + r + w), 0, 0))
    vb = v.orifice_mm + 30
    valve = (cq.Workplane("XY").box(vb, vb, vb).translate((vb / 2, 0, 0))
             .cut(cq.Workplane("YZ").circle(v.orifice_mm / 2).extrude(vb)))
    actuator = cq.Workplane("XY").circle(vb * 0.35).extrude(vb * 0.8).translate((vb / 2, 0, vb / 2))
    bore = BORE * 1e3
    barrel = cq.Workplane("YZ").circle(bore / 2 + 3).circle(bore / 2).extrude(BARREL * 1e3).translate((vb, 0, 0))
    x_ball = vb + 20 + D_BALL * 500
    ball = cq.Workplane("XY").sphere(D_BALL * 500).translate((x_ball, 0, 0))
    cup = (cq.Workplane("YZ").circle(bore / 2).extrude(8).cut(cq.Workplane("XY").sphere(D_BALL * 500).translate((8 + D_BALL * 500 - 4, 0, 0)))
           .translate((x_ball - D_BALL * 500 - 4, 0, 0)))
    return {"tank": tank, "valve": valve.union(actuator), "barrel": barrel, "ball": ball, "cup": cup,
            "wall_mm": w, "tank_length_mm": l_cyl + 2 * (r + w)}


def export_cad(root=Path("output/golf_gun/cad")):
    import cadquery as cq
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for c in CASES:
        parts = cad(c)
        asm = cq.Assembly()
        for k in ("tank", "valve", "barrel", "cup", "ball"):
            asm.add(parts[k], name=k)
        name = c.name.replace(" ", "_")
        asm.save(str(root / f"{name}.step"))
        whole = parts["tank"].union(parts["valve"]).union(parts["barrel"])
        cq.exporters.export(whole, str(root / f"{name}.stl"))
        out[c.name] = dict(wall_mm=parts["wall_mm"], tank_length_mm=parts["tank_length_mm"])
    return out


# --- the long CFD, run by the engineer (golf_cfd.sh) and exported as one JSON to paste back ------------------

MESHES = {   # bore cell size mm: the near-sonic case's run here was 1.5 mm (7.8k cells, ~12 s per 25 us chunk)
    "smoke": sf.Mesh(pipe=4.0, container_far=25.0, ambient_far=25.0),
    "medium": sf.Mesh(pipe=1.5, container_far=12.0, ambient_far=12.0),
    "fine": sf.Mesh(pipe=0.75, container_far=8.0, ambient_far=8.0),
}


def run_cfd(names=None, mesh="medium", root=Path("output/golf_gun"), smoke=False):
    """Run the transient CFD of the chosen cases and return a compact, pasteable summary: the CFD muzzle speed
    against the 0-D gun with the same (ideal) valve, the real valve's 0-D speed corrected by the CFD/0-D ratio,
    the ranges, and the ball's track decimated to ~60 points. ``smoke``: 0.3 ms only (does the case run?)."""
    import time
    out = {"mesh": mesh, "cases": {}}
    for c in CASES:
        if names and c.name.replace(" ", "_") not in names and c.name not in names:
            continue
        c = size(c)
        t0 = time.time()
        r = cfd(c, root=Path(root) / mesh, mesh=MESHES[mesh], max_time=3e-4 if smoke else 0.1)
        tr = r["track"]
        k = max(1, len(tr["t"]) // 60)
        real = gun(c.p_gauge, c.tank, c.valve)["muzzle"]
        ratio = r["muzzle"]["speed"] / r["gun_ideal"] if r["gun_ideal"] and not smoke else None
        v_corr = real * ratio if ratio else None
        out["cases"][c.name] = {
            "p_real_valve_bar_g": c.p_gauge, "p_cfd_ideal_valve_bar_g": r["p_ideal"],
            "cfd_muzzle": None if smoke else r["muzzle"]["speed"],      # a smoke run's 0.3 ms says nothing of it
            "cfd_speed_at_stop": r["muzzle"]["speed_at_stop"], "cfd_front_at_stop_mm": r["muzzle"]["front_at_stop_mm"],
            "cfd_time_ms": r["muzzle"]["time"] * 1e3, "gun0d_ideal_muzzle": r["gun_ideal"],
            "cfd_over_0d": ratio, "real_valve_0d": real, "real_valve_corrected": v_corr,
            "range_best_m": best(v_corr)[1]["range"] if v_corr else None,
            "range_10deg_m": fly(v_corr, 10)["range"] if v_corr else None,
            "wall_s": round(time.time() - t0, 1),
            "track": {key: [round(float(x), 6) for x in tr[key][::k]] for key in ("t", "x", "u")},
        }
    return out
