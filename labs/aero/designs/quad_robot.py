"""Notebook 08's quadcopter as a Chiron robot: the ``QuadFrame`` X-frame with its bought parts, nothing added —
Drongo (``drongo_robot``) without the skids and the pincer. The frame is the root link and lands on its arms
(``role='body'``); motors and propeller discs are drawn and weigh but do not collide. Geometry, rotor layout and the
parts list are Drongo's (notebook 08's numbers, ``drongo_robot.PARTS_G``, ``CAD['frame']``).

    import quad_robot as qr
    robot = qr.quad()                         # ~0.41 kg
"""
from __future__ import annotations

import math

import drongo_robot as dr
from vegeta.chiron import Geom, PointMass, Link, Robot


def mass_budget(cad: dict | None = None) -> dict:
    """Every part [g]: notebook 08's bought parts and the printed frame (CAD volume × PETG-CF density)."""
    cad = dict(dr.CAD) if cad is None else cad
    return {**dr.PARTS_G, "frame (QuadFrame, PETG-CF, CAD)": cad["frame"] * dr.MATERIALS["PETG-CF"] * dr.FILL["frame"]}


def quad(overrides: dict | None = None, *, name: str = "Quadcopter") -> Robot:
    """The quad as a Chiron ``Robot``: one rigid link (the frame), no joints."""
    p = dr.design_params(overrides)
    g = dr.geometry(p)
    m = {k: v / 1000.0 for k, v in mass_budget().items()}
    frame_rgba, dark = (0.20, 0.22, 0.25, 1.0), (0.08, 0.08, 0.09, 1.0)
    v_plate = math.pi * g["plate_r"] ** 2 * g["plate_t"]
    v_arm = g["R"] * g["arm_w"] * g["arm_h"]
    m_frame = m["frame (QuadFrame, PETG-CF, CAD)"]
    m_plate = m_frame * v_plate / (v_plate + 4 * v_arm)
    m_arm = (m_frame - m_plate) / 4
    geoms = [Geom("plate", "cylinder", (g["plate_r"], g["plate_t"] / 2), pos=(0, 0, g["plate_t"] / 2), mass=m_plate,
                  role="body", rgba=frame_rgba)]
    pos, _ = dr.rotor_layout(p)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        geoms.append(Geom(f"arm{k}", "box", (g["R"] / 2, g["arm_w"] / 2, g["arm_h"] / 2),
                          pos=(g["R"] / 2 * math.cos(a), g["R"] / 2 * math.sin(a), g["arm_h"] / 2), quat=dr._quat_z(a),
                          mass=m_arm, role="body", rgba=frame_rgba))
        geoms.append(Geom(f"motor{k}", "cylinder", (g["motor_r"], g["motor_h"] / 2),
                          pos=(pos[k, 0], pos[k, 1], g["arm_h"] + g["motor_h"] / 2), mass=m["motors 2306 (4x)"] / 4,
                          role="visual", rgba=dark))
        geoms.append(Geom(f"prop{k}", "cylinder", (g["prop_r"], 0.0008), pos=tuple(pos[k]),
                          mass=m["propellers 5x4.3 (4x)"] / 4, role="visual", rgba=(0.55, 0.75, 0.95, 0.35)))
    top = g["plate_t"]
    stack = [k for k in m if k not in ("frame (QuadFrame, PETG-CF, CAD)", "motors 2306 (4x)", "propellers 5x4.3 (4x)")]
    masses = [PointMass(k, m[k], (0.0, 0.0, top + 0.015)) for k in stack]      # battery and stack on the centre plate
    robot = Robot(name, Link("frame", log=True, geoms=geoms, masses=masses),
                  notes="notebook 08's quadcopter: the QuadFrame, 2306 motors, 5x4.3 props, 4S 1500 mAh",
                  sources={"geometry": "drongo_robot.geometry (designs/quad_frame.py)", "masses": "notebook 08 §1 "
                           "(drongo_robot.PARTS_G) and the frame's CAD volume (drongo_robot.CAD)",
                           "looks": "designs/quadcopter_assembly.py:Quadcopter"})
    robot.validate()
    return robot
