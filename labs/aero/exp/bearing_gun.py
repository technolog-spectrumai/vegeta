"""The golf air cannon re-barrelled for 8 mm stainless steel balls (G100, AISI 420/440C, Ø 8.000 mm).

Same device, same pressures (``golf_gun.CASES``: the tank, the valve and the gauge pressure each case was sized to);
only the tube changes:

- **the tube**: 600 mm (as the golf barrel), a precision-bore stainless hydraulic tube, bore 8.00 mm H8 (8.000..8.022),
  OD 12 mm, honed. A G100 ball is 8.000 ± 0.0025 mm round: it seals by its fit (a diametral clearance of 0..22 µm
  leaks a few percent of the gas; taken as a 3 % loss of the pressure force), and a 0.05 g felt wad behind it stops it
  rolling back.
- **the adapter**: a stepped reducer from the valve outlet (12 / 25 / 38 mm orifice) into the 8 mm bore, with a 30°
  cone; its volume adds to the dead volume behind the ball.
- **the feed**: a gravity magazine over a breech port with a spring detent, as the golf hopper.

The ball: 8 mm, density 7750 kg/m3 (440C) -> 2.08 g. Interior ballistics: ``harpoon.gun`` (the golf cannon's 0-D gun:
isentropic tank, the real valve ramped open, choked/subsonic flow, adiabatic gas behind the ball, a piston-driven
shock ahead) with the bore, mass, dead volume and friction of the steel ball.

The flight: a smooth sphere's drag against Mach (Hoerner ch. 16; Bailey & Hiatt 1972, AIAA J. 10): 0.47 subsonic,
0.55 at Mach 0.8, 0.80 at Mach 1.0, ~0.95..1.0 through Mach 1.2..1.5, 0.95 at 2. Its Reynolds number stays between
1e4 and 2e5, below a smooth ball's drag crisis. No spin, still air, from a 0.4 m muzzle height.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

import golf_gun as gg
import harpoon as hp

D = 0.008
RHO_STEEL = 7750.0
MASS = RHO_STEEL * math.pi / 6 * D ** 3 + 0.05e-3          # ball + felt wad
BORE_MM = 8.0
BARREL_M = 0.600
LEAK = 0.97                                                  # pressure force kept past the ball's clearance
RHO_AIR, A_SOUND, NU = 1.204, 343.2, 1.51e-5

MACH = np.array([0.0, 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0])
CD = np.array([0.47, 0.47, 0.49, 0.55, 0.65, 0.80, 0.90, 0.95, 1.00, 0.95, 0.92])


def cd_sphere(mach, re):
    cd = float(np.interp(mach, MACH, CD))
    if re < 1e4:                                             # slow: the low-Re rise (Schiller-Naumann region)
        cd = max(cd, 24 / re * (1 + 0.15 * re ** 0.687) + 0.42 / (1 + 4.25e4 / re ** 1.16))
    return cd


def device(case: gg.Case) -> hp.Gun:
    """The golf case's tank and valve, the new tube. The reducer's dead volume is folded into the gun's 30 mm start
    (harpoon.gun's x0) by a longer equivalent start; friction 2 N (wad + ball on the honed bore)."""
    return hp.Gun(bore_mm=BORE_MM, barrel_m=BARREL_M, tank_l=case.tank.volume_l, tank_rated_bar=case.tank.rated_bar,
                  valve_name=case.valve.name, valve_orifice_mm=case.valve.orifice_mm, valve_cd=case.valve.cd,
                  valve_open_s=case.valve.open_s, friction_n=2.0)


def shot(case: gg.Case):
    g = device(case)
    # the leak: the same as a 3 % lower driving pressure on the ball (applied to the gauge pressure)
    r = hp.gun(case.p_gauge * LEAK, g, MASS, dt=2e-7)
    return r


def fly(v0, elev_deg, h0=0.4, dt=5e-5):
    A = math.pi / 4 * D ** 2
    m = MASS - 0.05e-3
    a = math.radians(elev_deg)
    x, z, vx, vz, t = 0.0, h0, v0 * math.cos(a), v0 * math.sin(a), 0.0
    apex = h0
    while z >= 0:
        v = math.hypot(vx, vz)
        k = 0.5 * RHO_AIR * cd_sphere(v / A_SOUND, v * D / NU) * A * v / m
        vx -= k * vx * dt
        vz -= (k * vz + 9.81) * dt
        x += vx * dt
        z += vz * dt
        t += dt
        apex = max(apex, z)
        if dt < 2e-4 and v < 150:
            dt = 2e-4
    return {"range": x, "apex": apex, "time": t, "v_land": math.hypot(vx, vz)}


def best(v0):
    e = max(np.arange(10, 46, 1.0), key=lambda e: fly(v0, e, dt=2e-4)["range"])
    return float(e), fly(v0, e)


def summary():
    rows = []
    for c in gg.CASES:
        c = gg.size(c)                                       # the golf cannon's pressure for this case
        r = shot(c)
        v = r["muzzle"]
        e, fb = best(v)
        golf = gg.gun(c.p_gauge, c.tank, c.valve)["muzzle"]
        rows.append({
            "case": c.name, "tank": c.tank.name, "valve": c.valve.name, "p_bar_g": c.p_gauge,
            "golf_muzzle": golf, "golf_best_range": gg.best(golf)[1]["range"],
            "muzzle": v, "mach": v / A_SOUND, "energy_J": 0.5 * (MASS - 0.05e-3) * v * v,
            "barrel_ms": r["time_s"] * 1e3, "air_g": r["air_kg"] * 1e3, "peak_force_N": r["peak_force_n"],
            "range_10deg": fly(v, 10)["range"], "range_30deg": fly(v, 30)["range"],
            "best_elev": e, "best_range": fb["range"], "apex_best": fb["apex"], "flight_s": fb["time"],
            "v_land_best": fb["v_land"], "vacuum_best": v * v / 9.81,
        })
    return rows


if __name__ == "__main__":
    print(f"ball {MASS*1e3:.2f} g (with wad)")
    for r in summary():
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
