"""The Onager Sweeper's mission controller for ChironLab: sweep the street, stop for what the broom cannot take,
pick it up with a pincer and drop it into the basket, sweep on.

``Mission`` is the Manus's (``onager_manus_controller.Mission``: the chassis drive, the arms' joint-space and
straight-line moves, the jaws' angle / grip / close modes) plus the **broom**: its velocity servo gets the speed
target ``broom_rpm`` (0 = off). The suction is not the controller's: the scene's ``Vacuum`` hook pulls at the
litter under the hood whenever the fan runs (``lab.fan_on``, which the mission switches).

``sweep_and_clear`` builds the phases for a scene (``onager_sweeper_scenario.Scene``): broom and fan on, sweep at
``v_sweep`` to each listed pick-up (a brick, a box), stop with it ahead of the shoulders, take it from above with
the hooked jaws (the object's pose read from the simulation when the arm reaches: ideal perception), lift, swing
the arm back over the side to the basket (the pedestals turn through 180°: a slip ring), release, stow, sweep on.
Whether the object stays in the jaws and lands in the basket is physics.

    from . import onager_sweeper_controller as osc
    mission = osc.Mission(osc.sweep_and_clear(scene))

Promoted from ``notebooks/designs/onager_sweeper_controller.py`` as it was proven there; the notebook copy may move on.
"""
from __future__ import annotations

import math

import numpy as np

from . import onager_controller as oc
from . import onager_manus_controller as omc
from . import onager_manus_robot as omr
from . import onager_sweeper_robot as osr

__all__ = ["Mission", "sweep_and_clear", "PICK"]

smooth, Phase = oc.smooth, oc.Phase

#: How each kind of pick-up is taken (inputs): a hook grasp — the jaws close until the hooked tips (40 mm inward,
#: 220 mm from the pin) meet the object's sides, so for a width w the jaws stop at θ with 0.22 sin θ − 0.04 cos θ =
#: w/2 (brick 25°, box 37°) and the hooks sit 0.22 cos θ below the pin: ``grip_depth`` (the pin above the object's
#: top) puts them low on its sides, above the road; ``grip_torque`` the jaw drive's squeeze [N·m].
#: The pin must also clear the object's top corners of the jaws' inner faces: pin − top ≥ (w/2) / tan θ (brick 0.12 m,
#: box 0.13 m), so the box is taken by the hooks on its upper sides.
PICK = {"brick": {"grip_depth": 0.14, "grip_torque": 50.0, "jaw_open_deg": 50.0},
        "box": {"grip_depth": 0.14, "grip_torque": 25.0, "jaw_open_deg": 55.0}}


class Mission(omc.Mission):
    """``onager_manus_controller.Mission`` with the broom (a velocity servo) and the fan switch."""

    def __init__(self, phases: list, *, accel: float = 0.5, name: str = "sweeper mission", broom_rpm: float = 0.0):
        super().__init__(phases, accel=accel, name=name)
        self.broom_rpm0 = float(broom_rpm)

    def tool_reset(self, lab):
        super().tool_reset(lab)
        self.broom_rpm = self.broom_rpm0
        lab.fan_on = False

    def tool_command(self, obs, q_t, qd_t, ff):
        super().tool_command(obs, q_t, qd_t, ff)
        q_t["broom"] = float(obs.q[self.idx["broom"]])          # kp = 0: only the speed target matters
        qd_t["broom"] = self.broom_rpm * 2 * math.pi / 60.0
        ff["broom"] = 0.0


def sweep_and_clear(scene, *, v_sweep: float = 1.0, broom_rpm: float = osr.BROOM_RPM, r_pick: float = 0.75,
                    drop_x: float = -0.45, drop_dy: float = 0.02, drop_clear: float = 0.36) -> list:
    """The phases of notebook 23's mission on ``scene``: broom and fan on, sweep to each pick-up in ``scene.pickups``
    (name, kind, arm side), take it to the basket, sweep on to ``scene.x_end``. ``r_pick``: the object ahead of the
    shoulders when the robot stops; ``drop_x``: the drop point along the hull (in the basket's front half);
    ``drop_clear``: the jaw pin above the basket rim at the drop (the jaws hang 0.32 m below the pin, the object
    below them). The lift goes straight up to the drop height before the swing, so the object clears the basket's
    wall on its way (a lower swing carried the brick into it)."""
    g = omr.arm_geometry(osr.design_params())
    sg = osr.sweeper_geometry()
    rad = math.radians
    down = -math.pi / 2
    bx0, by0, bz0, bx1, by1, bz1 = sg["basket"]

    def wait(T):
        return lambda m, obs, tau: tau >= T

    def jaw(side, mode, val=None):
        def start(m, obs):
            m.jaw[side] = (mode, val)
        return start

    def both(*starts):
        fs = tuple(starts)

        def start(m, obs):
            for f in fs:
                f(m, obs)
        return start

    def switch(broom, fan):
        def start(m, obs):
            m.broom_rpm = broom
            m.lab.fan_on = fan
            m.log.append([float(obs.t), m.phase, f"broom {broom:g} rpm, fan {'on' if fan else 'off'}"])
        return start

    def broom_up(m, obs, tau):
        w = abs(float(obs.qd[m.idx["broom"]])) * 60 / (2 * math.pi)
        return w >= 0.9 * m.broom_rpm or tau >= 4.0

    def set_v(x_stop):
        def update(m, obs, tau):
            return m.drive_to(obs, x_stop, v_sweep)
        return update

    def obj_now(name):
        def f(m, obs):
            if m.memory.get(name) is None:
                m.memory[name] = np.asarray(obs.prop_pos[m.lab.prop_bodies.index(name)], dtype=float).copy()
            return m.memory[name]
        return f

    def stow_q(m, obs):
        return (omr.STOW["yaw"], omr.STOW["shoulder"], omr.STOW["elbow"], omr.STOW["wrist"])

    def ik_to(side, p_fn, elev):
        return lambda m, obs: m.arm_target(obs, side, p_fn(m, obs), elev)

    def hull_to_world(m, obs, p_hull):
        w, x, y, z = obs.base_quat
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        return R @ np.asarray(p_hull, dtype=float) + np.asarray(obs.base_pos)

    phases = []
    add = lambda name, upd, start=None, timeout=30.0: phases.append(Phase(name, upd, start, timeout))  # noqa: E731
    add("broom and fan on", broom_up, switch(broom_rpm, True), timeout=5.0)
    for name, kind, side in scene.pickups:
        s = omr.SIDES[side]
        pk = PICK[kind]
        top = scene.top_of(name)                                          # the object's top over its centre
        x_obj = scene.position(name)[0]
        x_stop = x_obj - r_pick - g["x"]
        above = lambda m, obs, f=obj_now(name), t=top, d=pk["grip_depth"]: f(m, obs) + np.array([0.0, 0.0, t + d + 0.30])  # noqa: E731
        grip = lambda m, obs, f=obj_now(name), t=top, d=pk["grip_depth"]: f(m, obs) + np.array([0.0, 0.0, t + d])         # noqa: E731
        drop_hull = (drop_x, s * (g["y"] + drop_dy), bz1 + drop_clear)
        drop = lambda m, obs, p=drop_hull: hull_to_world(m, obs, p)                                                   # noqa: E731
        pin = lambda m, obs, sd=side: omc._pin_world(m, obs, sd)                                                      # noqa: E731
        add(f"sweep to the {name}", set_v(x_stop), timeout=60.0)
        add("settle", wait(0.5))
        st, up = omc._joint_move(side, ik_to(side, above, down), 3.0)
        add(f"{side} arm: over the {name}", up, both(jaw(side, "angle", rad(pk["jaw_open_deg"])), st))
        st, up = omc._line_move(side, above, grip, down, 2.0)
        add(f"{side} arm: jaws around the {name}", up, st)
        add("grip", wait(1.2), jaw(side, "grip", pk["grip_torque"]))
        lifted = lambda m, obs, gr=grip: np.array([gr(m, obs)[0], gr(m, obs)[1], drop(m, obs)[2]])                   # noqa: E731
        st, up = omc._line_move(side, grip, lifted, down, 3.0)
        add("lift", up, st)
        st, up = omc._joint_move(side, ik_to(side, drop, down), 4.0)
        add(f"{side} arm: swing back to the basket", up, st)
        add("over the basket", wait(0.5))
        add("release", wait(1.0), jaw(side, "angle", rad(pk["jaw_open_deg"])))
        st, up = omc._line_move(side, pin, lambda m, obs, pn=pin: pn(m, obs) + np.array([0.0, 0.0, 0.05]), down, 1.0)
        add(f"{side} arm: up", up, st)
        st, up = omc._joint_move(side, stow_q, 4.0)
        add(f"{side} arm: stow", up, both(jaw(side, "angle", 0.0), st))
    add("sweep on", set_v(scene.x_end), timeout=60.0)
    add("broom and fan off", wait(1.0), switch(0.0, False))
    return phases
