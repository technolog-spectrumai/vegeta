"""FALCO's flight controller: NISUS+'s autopilot (``nisus_plus_controller``: TECS-like energy control, the mission
phases, the energy manager, the crow + brake descent, the steep crow final, the flare on the lidar height) with the
**propeller parked for the landing**:

* on the final, below ``park_agl`` (20 m) the throttle goes to idle, the ESC's brake stops the blades and the park
  flag is set (the seventh command channel): the parking routine ends with the blades horizontal
  (``FalcoAero.parked``); the flare then controls the sink with the pitch alone (NISUS+'s flare used a little
  throttle), crow and the brake stay available as airbrakes;
* a go-around (too high over the aim point) clears the park flag: the throttle restores and the motor spins up;
* the touchdown event reports **"propeller parked"** or **"propeller NOT parked"** (the outcome counts the latter as
  a failure: a blade strike).
"""
from __future__ import annotations

import math

import numpy as np

import nisus_plus_controller as fc_
from nisus_plus_controller import Gains, EnergyManager, AutoMission, slope_survey, PHASES  # noqa: F401

G = 9.81


class FalcoController(fc_.NisusPlusController):
    def __init__(self, *args, park_agl: float = 20.0, name: str = "Falco flight controller", **kw):
        super().__init__(*args, name=name, **kw)
        self.park_agl = park_agl

    def reset(self, lab, seed=None):
        super().reset(lab, seed)
        self.park_cmd = 0.0
        self.parked_at_touchdown = None
        self.t_park_cmd = None

    def _attitude(self, roll_cmd, pitch_cmd, roll, pitch, omega_b, V, eas, throttle, crow, brake):
        if self.park_cmd > 0.5:
            throttle = 0.0
            brake = 1.0 if not getattr(self.aero, "parked", False) else brake
        super()._attitude(roll_cmd, pitch_cmd, roll, pitch, omega_b, V, eas, throttle, crow, brake)
        self.aero.command = np.append(self.aero.command[:6], self.park_cmd)

    def __call__(self, obs):
        a = self.aero.last
        t = float(obs.t)
        if a:
            on_final = self.phase == "approach" and self.target == "final"
            want = on_final and a["agl"] < self.park_agl
            if want and self.park_cmd < 0.5:
                self.park_cmd, self.t_park_cmd = 1.0, t
                self.events.append([t, "landing", f"{a['agl']:.0f} m: throttle idle, brake on, park the propeller"])
            elif not want and self.park_cmd > 0.5 and self.phase == "approach" and self.target != "final":
                self.park_cmd = 0.0
                self.events.append([t, "landing", "go-around: park cleared, the motor spins up"])
        td_before = self.touchdown
        out = super().__call__(obs)
        if td_before is None and self.touchdown is not None:
            parked = bool(getattr(self.aero, "parked", False))
            self.parked_at_touchdown = parked
            rpm = float(self.aero.last.get("rpm", 0.0)) if hasattr(self.aero, "last") else 0.0
            self.events.append([self.touchdown, "landing", "propeller parked horizontal at touchdown" if parked else f"propeller NOT parked at touchdown — a blade strike"])
        if self.phase == "landed" and len(self.aero.command) == 7:
            self.aero.command[6] = self.park_cmd
        for ev in getattr(self.aero, "park_events", []):
            self.events.append([ev[0], "propeller", ev[1]])
        if hasattr(self.aero, "park_events"):
            self.aero.park_events = []
        return out


__all__ = ["FalcoController", "Gains", "EnergyManager", "AutoMission", "slope_survey", "PHASES"]
