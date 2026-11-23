"""Crude flat-plate propeller STL for the rotor tests (Aeromant itself does not generate geometry)."""
import math

import numpy as np

from vegeta.aeromant.stl import Surface


def flat_plate_propeller(diameter=0.127, blades=2, chord=0.016, thickness=0.0016, pitch_deg=15.0, n=12):
    """Blades are flat plates at ``pitch_deg`` to the rotation plane, axis +x, blade 1 along +y.
    With rotation=1 (right-hand about +x) the leading edge is upstream: the rotor pushes fluid to +x."""
    tris = []
    r = np.linspace(0.12 * diameter / 2, diameter / 2, n + 1)
    beta = math.radians(pitch_deg)
    quad = lambda a, b, c, d: [(a, b, c), (a, c, d)]
    for k in range(blades):
        ang = 2 * math.pi * k / blades
        er = np.array([0.0, math.cos(ang), math.sin(ang)])
        et = np.array([0.0, -math.sin(ang), math.cos(ang)])

        def pt(rr, s, t):   # s along the chord [-0.5, 0.5], t across the thickness [-0.5, 0.5]
            x = -s * chord * math.sin(beta) + t * thickness * math.cos(beta)
            y = s * chord * math.cos(beta) + t * thickness * math.sin(beta)
            return np.array([x, 0.0, 0.0]) + rr * er + y * et

        for i in range(n):
            r0, r1 = r[i], r[i + 1]
            for t in (-0.5, 0.5):
                a, b, c, d = pt(r0, -0.5, t), pt(r1, -0.5, t), pt(r1, 0.5, t), pt(r0, 0.5, t)
                tris += quad(a, d, c, b) if t > 0 else quad(a, b, c, d)
            for s in (-0.5, 0.5):
                a, b, c, d = pt(r0, s, -0.5), pt(r1, s, -0.5), pt(r1, s, 0.5), pt(r0, s, 0.5)
                tris += quad(a, d, c, b) if s > 0 else quad(a, b, c, d)
            if i == 0:
                a, b, c, d = pt(r0, -0.5, -0.5), pt(r0, 0.5, -0.5), pt(r0, 0.5, 0.5), pt(r0, -0.5, 0.5)
                tris += quad(a, d, c, b)
            if i == n - 1:
                a, b, c, d = pt(r1, -0.5, -0.5), pt(r1, 0.5, -0.5), pt(r1, 0.5, 0.5), pt(r1, -0.5, 0.5)
                tris += quad(a, b, c, d)
    surf = Surface(np.array(tris), "prop")
    if surf.volume < 0:  # keep the winding outward whatever the parameters
        surf = Surface(surf.triangles[:, ::-1, :], "prop")
    return surf
