"""Gannet and Fulmar (the carrier's drones) as Chiron robots, for the Yax export (``machines.py``): one rigid link each,
empty with its pack (the payload - crate or surveillance pod - rides on the Ketos card), its mass that of the sizing.

* **Gannet**: the body box (frame, flight stack, the pack's sled) and the four motors with their propellers as point
  masses at the rotor positions, the skids as a light collision box. Root frame: the skids' bottom, the CAD's.
* **Fulmar**: the fuselage as a cylinder, the wing as a box, the two motors as point masses at the nacelles. Root
  frame: the CAD's (the wing root's leading edge, x aft).

Units SI. The flight model is on the Ketos card (power, endurance, limits), not here."""
from __future__ import annotations

import math

from vegeta.chiron import Geom, Link, PointMass, Robot


def gannet() -> Robot:
    import gannet as gm
    p = gm.Gannet.p()
    a = p["arm"] / math.sqrt(2) / 1000
    motor_kg = (gm.MASS["motors 4x"] + gm.MASS["propellers 4x 28 in"] + gm.MASS["ESCs 4x"]) / 4
    body_kg = gm.Gannet.empty_kg(p) - 4 * motor_kg - 0.4
    geoms = [Geom("body", "box", (p["body"] / 2000, p["body"] * 0.4 / 1000, 0.15), pos=(0.0, 0.0, 0.38), mass=body_kg, role="body",
                  rgba=(0.12, 0.38, 0.55, 1.0)),
             Geom("skids", "box", (0.35, p["skid_span"] / 2000, 0.015), pos=(0.0, 0.0, 0.015), mass=0.4, role="body", rgba=(0.34, 0.4, 0.45, 1.0))]
    masses = [PointMass(f"motor_{i}", motor_kg, (x, y, 0.47)) for i, (x, y) in enumerate(((a, a), (a, -a), (-a, a), (-a, -a)))]
    r = Robot("Gannet", Link("body", log=True, geoms=geoms, masses=masses),
              notes="Gannet, the carrier's VTOL quad, empty with its pack: a rigid body (payloads on the Ketos card)",
              sources={"masses": "gannet.MASS + pack (class values)", "looks": "designs/gannet.py:Gannet"})
    r.validate()
    return r


def fulmar() -> Robot:
    import fulmar as fm
    p = fm.Fulmar.p()
    wing = fm.MASS["wing (foam-core glass, carbon spar)"]
    motors = fm.MASS["motors + props 2x"]
    body = fm.Fulmar.empty_kg(p) - wing - motors
    L, r = p["fuselage_length"] / 1000, p["fuselage_diameter"] / 2000
    x0 = -p["nose_length"] / 1000
    chord = p["root_chord"] / 1000 * (1 + p["taper"]) / 2
    geoms = [Geom("fuselage", "cylinder", (r, L / 2), pos=(x0 + L / 2, 0.0, 0.0), quat=(0.7071068, 0.0, 0.7071068, 0.0), mass=body,
                  role="body", rgba=(0.95, 0.61, 0.07, 1.0)),
             Geom("wing", "box", (chord / 2, p["span"] / 2000, 0.025), pos=(chord / 2, 0.0, 0.0), mass=wing, role="body", rgba=(0.95, 0.61, 0.07, 1.0))]
    masses = [PointMass(f"motor_{s}", motors / 2, (-p["nacelle_forward"] / 1000, sy * p["nacelle_y"] / 1000, 0.0)) for s, sy in (("l", 1), ("r", -1))]
    rb = Robot("Fulmar", Link("fuselage", log=True, geoms=geoms, masses=masses),
               notes="Fulmar, the carrier's fixed-wing, empty with its pack: a rigid body (payloads on the Ketos card)",
               sources={"masses": "fulmar.MASS + pack (class values)", "looks": "designs/fulmar.py:Fulmar"})
    rb.validate()
    return rb
