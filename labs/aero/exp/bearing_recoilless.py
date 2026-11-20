"""A recoilless pneumatic emitter made for 8 mm steel balls: 100 m effective range.

The golf cannon's rear venturi (``golf_recoilless``), scaled to an 8 mm bore:

    300 bar HPA bottle -> regulator -> 0.5 L steel plenum (100 bar rated) -> 1/2" hammer valve (12 mm, 1.5 ms)
      -> side inlet into a small chamber -> the ball (2.08 g + felt wad) at the start of a 1.0 m honed 8 mm tube
      -> muzzle;  chamber's rear -> throat -> 2:1 cone -> the jet out of the back.

Why these numbers:
- **speed 280 m/s** (Mach 0.82): fast enough that the ball still has ~100 m/s and ~10 J at 100 m and the drop is
  small, and below the transonic drag rise (Mach 0.85+), where a smooth sphere's drag doubles and its flight wobbles;
- **a longer tube (1.0 m)**: the same speed from a lower pressure, and a straighter start;
- **the throat** is sized with the pressure so the shot's net impulse is zero (``golf_recoilless.recoilless_design``):
  in an 8 mm bore the throat is ~6-7 mm, small next to a 12 mm valve, so the valve feeds the nozzle easily (the golf
  cannon's problem was the reverse);
- **scatter** (for the target tests): a honed tube with a tight ball fit and a rigid mount, 0.06 deg (about 3.5 MOA)
  in direction and 0.5 % in speed, against the game launcher's 0.25 deg and 1 %.

    python bearing_recoilless.py          # sizes it, prints the design
"""
from __future__ import annotations

import math

import bearing_gun as bg
import golf_gun as gg
import golf_recoilless as g

TARGET_V = 280.0
BARREL_M = 1.0
EPS = 2.0
FRICTION = 2.0
DT = 1e-6
SCATTER_DEG = 0.06
SCATTER_SPEED = 0.005

CASE = gg.Case("bearing recoilless", TARGET_V,
               gg.Tank("0.5 L steel plenum, regulated from a 300 bar HPA bottle", 0.5, 100.0, "steel 4130", 50.0),
               gg.Valve('1/2" hammer valve', 12.0, 0.75, 1.5e-3), 0.0)
PROJ = dict(bore_m=0.008, mass=bg.MASS, barrel_m=BARREL_M, d_ball=0.008)


def design(verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    log(f"sizing: 8 mm ball {bg.MASS * 1e3:.2f} g, tube {BARREL_M} m, target {TARGET_V:.0f} m/s, zero recoil")
    p_plain = g.restore_pressure(CASE, 0.0, EPS, 1.0, TARGET_V, proj=PROJ, dt=DT, friction=FRICTION)
    plain = g.shot(CASE, 0.0, EPS, 1.0, p_plain, proj=PROJ, dt=DT, friction=FRICTION)
    log(f"  plain gun (no nozzle): {p_plain:.1f} bar g, recoil {plain['impulse_Ns']:.3f} N s, air {plain['air_g']:.2f} g")
    th, p = 0.0, p_plain
    for it in range(2):
        th = g.null_throat(CASE, EPS, 1.0, p, proj=PROJ, dt=DT, friction=FRICTION)
        p = g.restore_pressure(CASE, th, EPS, 1.0, TARGET_V, proj=PROJ, dt=DT, friction=FRICTION)
        log(f"  pass {it + 1}: throat {th:.2f} mm, {p:.1f} bar g")
    r = g.shot(CASE, th, EPS, 1.0, p, proj=PROJ, dt=DT, friction=FRICTION)
    fine = g.shot(CASE, th, EPS, 1.0, p, proj=PROJ, dt=DT / 5, friction=FRICTION)
    out = dict(p_bar_g=p, throat_mm=th, exit_mm=th * math.sqrt(EPS), v=r["v"], v_fine=fine["v"], impulse=r["impulse_Ns"],
               impulse_fine=fine["impulse_Ns"], air_g=r["air_g"], pc_max_bar=r["pc_max_bar"], jet_speed=r["jet_speed"],
               barrel_ms=r["barrel_ms"], backblast_m=g.backblast(r["jet_speed"], th, EPS), energy_J=0.5 * PROJ["mass"] * r["v"] ** 2,
               plain_p=p_plain, plain_impulse=plain["impulse_Ns"], plain_air=plain["air_g"],
               shots_per_fill=None)
    # shots from a 0.8 L, 300 bar bottle down to the regulator's set pressure (+20 bar margin)
    bottle_air = 0.8e-3 * ((300 - p - 20) * 1e5) / (287.05 * 293.15)
    out["shots_per_fill"] = int(bottle_air * 1e3 / r["air_g"])
    log(f"  design: {p:.1f} bar g, throat {th:.2f} mm (exit {out['exit_mm']:.2f} mm) -> {r['v']:.1f} m/s "
        f"(5x finer step: {fine['v']:.1f}), recoil {r['impulse_Ns']:+.4f} N s (fine {fine['impulse_Ns']:+.4f}), "
        f"air {r['air_g']:.2f} g/shot, {out['shots_per_fill']} shots per 0.8 L bottle, jet {r['jet_speed']:.0f} m/s, "
        f"back-blast {out['backblast_m']:.1f} m, muzzle energy {out['energy_J']:.0f} J")
    return out


if __name__ == "__main__":
    design()
