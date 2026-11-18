"""The light recoilless net interceptor: small, single-purpose, for bandit quadcopters and FPV drones in mid-flight.

Trades range and reserve shots for weight, against the golf-cannon net launcher (``net_launcher_cad``):

=====================  ===========================================  ===========================================
                       golf-cannon net launcher                     light interceptor
=====================  ===========================================  ===========================================
cartridge              150 g, 42.5 x 80 mm, 2.4 m net, 4 x 20 g     70 g, 31.8 x 70 mm, 1.5 m net (0.15 m mesh,
                                                                    1 mm Dyneema, 30 g), 4 x 8 g weights, 8 g shell
tube                   0.6 m, bore 42.7 mm, steel                   0.35 m, bore 32 mm, 6061 1.5 mm wall
air                    3 L aluminium tank, 8.5 bar                  0.2 L carbon HPA bottle (300 bar) + regulator
                                                                    -> 0.25 L steel plenum at 32.7 bar
valve                  1.5" diaphragm valve                         1" hammer valve (25.4 mm, opens in 2 ms)
rear venturi           throat 36.3 mm                               throat 27.2 mm
shot                   40 m/s, 22 g air, back-blast ~7 m            45 m/s, 8.8 g air, back-blast ~6.4 m
shots per fill         ~10 (3 L at 16 bar)                          ~6 (0.2 L at 300 bar)
capture window         ~3-10 m                                      ~2-6 m
mass (loaded)          ~5 kg                                        ~1.5 kg
=====================  ===========================================  ===========================================

The recoilless sizing is ``golf_recoilless.recoilless_design`` with ``proj=``. A 3/4" valve could not feed a throat
big enough to null the recoil (25 mm of throat drained the chamber: 15 m/s), hence the 1" hammer valve and the higher
plenum pressure (a plain gun would need only 6.4 bar). The mass budget is by hand (catalogue-like parts).

    python net_light_design.py
"""
from __future__ import annotations

import math

import golf_gun as gg
import golf_recoilless as g

TARGET_V = 45.0
CASE = gg.Case("light net interceptor", TARGET_V, gg.Tank("0.25 L steel plenum", 0.25, 100.0, "steel 4130", 50.0),
               gg.Valve('1" hammer valve', 25.4, 0.75, 2e-3), 0.0)
PROJ = dict(bore_m=0.032, mass=0.070, barrel_m=0.35, d_ball=0.07)
FRICTION = 2.0

MASS_KG = {   # by hand
    "tube 32 mm bore, 0.35 m, 6061, 1.5 mm wall": 0.15,
    "chamber + rear venturi (6061)": 0.14,
    "1\" hammer valve + pilot solenoid": 0.30,
    "0.25 L plenum, steel, 1.6 mm wall (100 bar, SF 3)": 0.26,
    "0.2 L carbon HPA bottle (300 bar) + regulator": 0.42,
    "breech door, trigger electronics, wiring": 0.08,
    "mount bracket (to the Pekari's pan-tilt)": 0.10,
    "net cartridge (loaded)": 0.07,
}


def design(verbose=True):
    th, p = g.recoilless_design(CASE, TARGET_V, proj=PROJ, dt=2e-6, iters=5, friction=FRICTION)
    r = g.shot(CASE, th, 2.0, 1.0, p, proj=PROJ, dt=2e-6, friction=FRICTION)
    p_plain = g.restore_pressure(CASE, 0.0, 2.0, 1.0, TARGET_V, proj=PROJ, dt=2e-6, friction=FRICTION)
    bottle_air = 0.2e-3 * ((300 - p - 20) * 1e5) / (287.05 * 293.15)
    out = dict(throat_mm=th, p_bar_g=p, v=r["v"], impulse=r["impulse_Ns"], air_g=r["air_g"], jet=r["jet_speed"],
               backblast_m=g.backblast(r["jet_speed"], th, 2.0), plain_p=p_plain,
               shots_per_fill=int(bottle_air * 1e3 / r["air_g"]), mass_kg=sum(MASS_KG.values()))
    if verbose:
        print(f"light interceptor: throat {th:.1f} mm, {p:.1f} bar g -> {r['v']:.1f} m/s, recoil {r['impulse_Ns']:+.3f} N s, "
              f"air {r['air_g']:.1f} g/shot, {out['shots_per_fill']} shots per 0.2 L fill, back-blast {out['backblast_m']:.1f} m "
              f"(a plain gun: {p_plain:.1f} bar)")
        for k, m in MASS_KG.items():
            print(f"  {m:5.2f} kg  {k}")
        print(f"  {out['mass_kg']:5.2f} kg  total, loaded")
    return out


if __name__ == "__main__":
    design()
