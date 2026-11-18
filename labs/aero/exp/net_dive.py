"""Net fired from a diving aircraft (the X-wing X-A, or GAVILAN-H) with less pressure: what the dive buys.

The net (``net_drone``'s light 1.5 m net, 70 g cartridge) leaves the muzzle at the aircraft's airspeed V plus the
muzzle speed v0; the corner weights splay only with v0 (their sideways speed is v0 tan 12 deg). For each (V, v0):
the net's opening against distance, its speed, and the gap to the aircraft flying on straight at V (the aircraft
must not catch its own net: it breaks away right after the shot); and the tank pressure the recoilless interceptor
needs for v0 (``golf_recoilless`` with the light design's valve and plenum).

    python net_dive.py
"""
from __future__ import annotations

import math

import numpy as np

import golf_recoilless as g
import net_drone as nd
import net_light_design as nl


def fly(V, v0, t_max=1.0):
    nd.configure(light=True, drone="fpv")
    sim = nd.Sim()
    up = np.array([0.0, 0.0, 80.0])
    sim.reset(0.0, 0.0, v0, None, np.zeros(3), with_drone=False, origin=up)
    d = sim.d
    for dv in sim.kdof:                               # the aircraft's speed added to every knot (level here: the dive's
        d.qvel[dv] += V                               # direction; gravity's sideways part is small over a second)
    rows = []
    while d.time < t_max:
        for _ in range(50):
            sim.step()
        c = sim.centre()
        v = d.qvel[sim.kdof].mean()
        rows.append((d.time, c[0], sim.span(), v, c[0] - V * d.time))
    return np.array(rows)


def pressure(v0):
    th, p = g.recoilless_design(nl.CASE, v0, proj=nl.PROJ, dt=2e-6, iters=4, friction=nl.FRICTION)
    r = g.shot(nl.CASE, th, 2.0, 1.0, p, proj=nl.PROJ, dt=2e-6, friction=nl.FRICTION)
    return p, th, r["air_g"]


if __name__ == "__main__":
    import sys
    for v0 in (() if "--no-pressure" in sys.argv else (15.0, 25.0, 45.0)):
        p, th, air = pressure(v0)
        print(f"muzzle {v0:.0f} m/s: {p:.1f} bar, throat {th:.1f} mm, air {air:.1f} g", flush=True)
    for V in (0.0, 20.0, 30.0):
        for v0 in (15.0, 25.0, 45.0):
            r = fly(V, v0)
            t_open = r[np.argmax(r[:, 2] >= 1.2), 0] if (r[:, 2] >= 1.2).any() else float("nan")
            x_open = float(np.interp(t_open, r[:, 0], r[:, 1])) if t_open == t_open else float("nan")
            closes = r[(r[:, 0] > t_open) & (r[:, 2] < 0.6)] if t_open == t_open else r[:0]
            x_close = closes[0, 1] if len(closes) else float("nan")
            gap = r[:, 4]
            k = int(np.argmax(gap))                   # the net's lead peaks, then the aircraft closes on it
            back = r[k:][gap[k:] < 0.5]
            t_catch = back[0, 0] if len(back) else float("nan")
            x_catch = float(np.interp(t_catch, r[:, 0], r[:, 1])) if t_catch == t_catch else float("nan")
            print(f"dive {V:3.0f} m/s, muzzle {v0:3.0f}: max lead {gap.max():4.1f} m; open (1.2 m) at {x_open:5.1f} m ({t_open:.2f} s), closes at {x_close:5.1f} m; "
                  f"net speed at 0.2 s {np.interp(0.2, r[:, 0], r[:, 3]):5.1f} m/s; gap to the aircraft at 0.3 s "
                  f"{np.interp(0.3, r[:, 0], gap):5.1f} m, it catches its own net (gap back under 0.5 m) at "
                  f"{'never (1 s)' if t_catch != t_catch else f'{t_catch:.2f} s, {x_catch:.1f} m out'}", flush=True)
