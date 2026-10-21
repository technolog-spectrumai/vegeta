"""og.propulsion — propellers, fans and their electric drives as tables: what thrust and power a unit gives at an
airspeed, and what electrical power a thrust costs.

* PropulsionMap — thrust, shaft power, torque, current, voltage on a grid of airspeed x rpm at one battery
  voltage (at(V, throttle), throttle_for_thrust, electrical_for_thrust, JSON in and out).
* ElectricDrive — the signed thrust and torque of a propeller on (V, rpm) at sea-level density, scaled by σ =
  ρ/ρ0, driven by a DC motor (Kv, R, I0, I_max) from a pack: throttle, freewheel, locked drag, braking and
  regeneration with charge and current limits; map_at(ρ) gives a PropulsionMap.
* Unit — a flight-envelope view: full thrust and the electrical power of a thrust against airspeed.
* DriveTable — a multirotor's view indexed by thrust: T_max(V), κ(V) = Q/T, P(V, T).

The tables are made by design tools (Vegeta's Boreas BEMT); Ogma evaluates them. Moved from Vegeta's
nisus_systems.PropulsionMap, nisus_plus_systems.NisusPlusDrive, merlin_flight.Unit and
kestrel_systems.DriveTable (the races' refquad.Drive is the same table); reproduced exactly
(tests/test_propulsion.py).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

RHO0 = 1.225
RHO = RHO0


@dataclass
class PropulsionMap:
    """The drive as tables over airspeed ``V`` [m/s] and ``rpm``: thrust [N] (corrected), shaft power [W], torque,
    the motor's current [A] and the voltage it needs [V] (so throttle = V_need / V_battery), electrical power [W].
    Static thrust, in-flight thrust, shaft power and electrical input are separate columns of one table."""
    V: np.ndarray
    rpm: np.ndarray
    thrust: np.ndarray
    shaft_power: np.ndarray
    torque: np.ndarray
    current: np.ndarray
    voltage: np.ndarray
    battery_v: float
    esc_efficiency: float = 0.96
    motor_name: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def electrical(self):
        return self.voltage * self.current / self.esc_efficiency

    def _interp_rows(self, V):
        V = float(np.clip(V, self.V[0], self.V[-1]))
        i = min(int(np.searchsorted(self.V, V, side="right")) - 1, len(self.V) - 2)
        t = (V - self.V[i]) / (self.V[i + 1] - self.V[i])
        return i, t

    def row(self, V, table):
        i, t = self._interp_rows(V)
        return (1 - t) * table[i] + t * table[i + 1]

    def rpm_for_throttle(self, V, throttle):
        """The rpm the motor settles at for a throttle (fraction of the battery voltage) at airspeed V: the rpm
        where the voltage it needs equals the applied voltage (monotone in rpm)."""
        volts = float(np.clip(throttle, 0.0, 1.0)) * self.battery_v
        vn = self.row(V, self.voltage)
        if volts <= vn[0]:
            return float(self.rpm[0]) * volts / max(vn[0], 1e-9)
        return float(np.interp(volts, vn, self.rpm))

    def at(self, V, throttle) -> dict:
        n = self.rpm_for_throttle(V, throttle)
        T = float(np.interp(n, self.rpm, self.row(V, self.thrust)))
        I = float(np.interp(n, self.rpm, self.row(V, self.current)))
        Vn = float(np.interp(n, self.rpm, self.row(V, self.voltage)))
        Ps = float(np.interp(n, self.rpm, self.row(V, self.shaft_power)))
        if throttle <= 0.0:
            return {"rpm": 0.0, "thrust": 0.0, "current": 0.0, "electrical": 0.0, "shaft_power": 0.0, "voltage": 0.0}
        return {"rpm": n, "thrust": T, "current": I, "electrical": Vn * I / self.esc_efficiency, "shaft_power": Ps, "voltage": Vn}

    def max_thrust(self, V):
        return self.at(V, 1.0)["thrust"]

    def throttle_for_thrust(self, V, T):
        """Throttle for a thrust at V (clipped to full throttle)."""
        Tr = self.row(V, self.thrust)
        vn = self.row(V, self.voltage)
        if T <= Tr[0]:
            return float(vn[0] / self.battery_v * max(T, 0.0) / max(Tr[0], 1e-9))
        if T >= Tr[-1]:
            return 1.0
        n = float(np.interp(T, Tr, self.rpm))
        return float(min(np.interp(n, self.rpm, vn) / self.battery_v, 1.0))

    def electrical_for_thrust(self, V, T):
        th = self.throttle_for_thrust(V, T)
        return self.at(V, th)["electrical"] if T <= self.max_thrust(V) + 1e-9 else math.inf

    def table(self, V) -> pd.DataFrame:
        i, t = self._interp_rows(V)
        return pd.DataFrame({"rpm": self.rpm, "thrust [N]": self.row(V, self.thrust), "shaft [W]": self.row(V, self.shaft_power),
                             "current [A]": self.row(V, self.current), "voltage [V]": self.row(V, self.voltage),
                             "electrical [W]": self.row(V, self.electrical), "throttle": self.row(V, self.voltage) / self.battery_v}).set_index("rpm")

    def to_json(self) -> dict:
        return {"V": self.V.tolist(), "rpm": self.rpm.tolist(), "thrust": self.thrust.tolist(), "shaft_power": self.shaft_power.tolist(),
                "torque": self.torque.tolist(), "current": self.current.tolist(), "voltage": self.voltage.tolist(), "battery_v": self.battery_v,
                "esc_efficiency": self.esc_efficiency, "motor_name": self.motor_name, "notes": self.notes}

    @classmethod
    def from_json(cls, d: dict) -> "PropulsionMap":
        return cls(np.asarray(d["V"]), np.asarray(d["rpm"]), np.asarray(d["thrust"]), np.asarray(d["shaft_power"]), np.asarray(d["torque"]),
                   np.asarray(d["current"]), np.asarray(d["voltage"]), d["battery_v"], d.get("esc_efficiency", 0.96), d.get("motor_name", ""), d.get("notes", {}))


@dataclass
class ElectricDrive:
    """The drive over (airspeed V, rpm) at sea level — signed thrust [N] and shaft torque [N m] from Boreas BEMT on the
    15x8 (calibrated on the positive region) — and the fitted motor; at any density the propeller scales with σ and the
    motor's current and voltage follow (``rows``). The throttle is the ESC's applied voltage over the pack's loaded
    voltage; at low throttle and speed the voltage is below the back-EMF and the current reverses (``at``)."""
    V: np.ndarray
    rpm: np.ndarray
    thrust0: np.ndarray
    torque0: np.ndarray
    kv: float
    R: float
    I0: float
    max_current_a: float
    battery_v: float = 21.6
    esc_efficiency: float = 0.96
    motor_name: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def kt(self):
        return 60 / (2 * math.pi * self.kv)

    def _row(self, V, table):
        V = float(np.clip(V, self.V[0], self.V[-1]))
        i = min(int(np.searchsorted(self.V, V, side="right")) - 1, len(self.V) - 2)
        t = (V - self.V[i]) / (self.V[i + 1] - self.V[i])
        return (1 - t) * table[i] + t * table[i + 1]

    def rows(self, V, rho=RHO0) -> dict:
        """Over the rpm grid at airspeed V and density rho: thrust, torque, the motor's current and the voltage it needs."""
        s = rho / RHO0
        T = s * self._row(V, self.thrust0)
        Q = s * self._row(V, self.torque0)
        I = self.I0 + Q / self.kt
        Vn = self.rpm / self.kv + I * self.R
        return {"rpm": self.rpm, "thrust": T, "torque": Q, "current": I, "voltage": Vn, "shaft_power": Q * self.rpm * math.pi / 30}

    def _battery_w(self, Vn, I):
        P = Vn * I
        return P / self.esc_efficiency if P >= 0 else P * self.esc_efficiency

    def _point(self, r, n, mode):
        T = float(np.interp(n, r["rpm"], r["thrust"]))
        I = float(np.interp(n, r["rpm"], r["current"]))
        Vn = float(np.interp(n, r["rpm"], r["voltage"]))
        Ps = float(np.interp(n, r["rpm"], r["shaft_power"]))
        return {"rpm": float(n), "thrust": T, "current": I, "voltage": Vn, "shaft_power": Ps, "electrical": self._battery_w(max(Vn, 0.0), I), "mode": mode}

    def freewheel(self, V, rho=RHO0) -> dict:
        """The ESC off (or its motor output floating): the propeller windmills where the motor carries no current — its
        friction torque −kt I0 only (on the stable branch: above the rpm of the most negative torque); when even the
        lowest rpm's torque cannot overcome the friction, the propeller stops (``locked_drag``)."""
        r = self.rows(V, rho)
        Q_need = -self.kt * self.I0
        i0 = int(np.argmin(r["torque"]))
        if r["torque"][i0] < Q_need:
            n = float(np.interp(Q_need, r["torque"][i0:], r["rpm"][i0:]))
            p = self._point(r, n, "freewheel")
            p.update(current=0.0, voltage=n / self.kv, electrical=0.0)
            return p
        return {"rpm": 0.0, "thrust": -self.locked_drag(V, rho), "current": 0.0, "voltage": 0.0, "shaft_power": 0.0, "electrical": 0.0, "mode": "stopped"}

    def locked_drag(self, V, rho=RHO0) -> float:
        """A stopped two-blade propeller: the blades' area (~9 % of the disc for a 15-inch thin-electric) nearly flat to the
        flow at a flat plate's CD of 1.2 (assumed; Boreas cannot solve rpm = 0)."""
        R = self.notes.get("prop_radius_m", 0.1905)
        return 0.5 * rho * V * V * 1.2 * 0.09 * math.pi * R * R

    def at(self, V, throttle, rho=RHO0, v_batt=None, *, i_charge_max=math.inf, i_motor_max=None) -> dict:
        """The drive at airspeed V (axial), ``throttle`` 0..1 of the pack's loaded voltage ``v_batt`` (default nominal),
        density rho; ``throttle`` None: freewheel. Below the windmill's back-EMF the motor generates: the battery-side
        power is negative (``electrical``: charging, mode 'regen'). The motor current is held to ``i_motor_max`` and the
        charge current (battery side) to ``i_charge_max`` by moving the ESC's voltage back toward the free windmill (less
        braking: the ESC cannot dump the energy anywhere else)."""
        if throttle is None:
            return self.freewheel(V, rho)
        vb = self.battery_v if v_batt is None else float(v_batt)
        i_motor_max = self.max_current_a if i_motor_max is None else i_motor_max
        r = self.rows(V, rho)
        vn = np.maximum.accumulate(r["voltage"])
        volts = float(np.clip(throttle, 0.0, 1.0)) * vb
        if volts <= vn[0]:
            pt = self._point(r, float(r["rpm"][0]), "drive")
            if vn[0] > 0:
                f = volts / vn[0]
                pt.update(thrust=pt["thrust"] * f, current=pt["current"] * f, shaft_power=pt["shaft_power"] * f, electrical=pt["electrical"] * f, rpm=pt["rpm"] * f)
            return pt
        n = float(np.interp(volts, vn, r["rpm"]))
        pt = self._point(r, n, "regen" if np.interp(n, r["rpm"], r["current"]) < 0 else "drive")
        if pt["current"] > i_motor_max:
            pt = self._point(r, float(np.interp(i_motor_max, r["current"], r["rpm"])), "drive (current limited)")
        if pt["mode"] == "regen":
            fw = self.freewheel(V, rho)
            lim_motor = pt["current"] < -i_motor_max
            lim_charge = -pt["electrical"] / vb > i_charge_max
            if lim_motor or lim_charge:
                ns_ = np.linspace(n, max(fw["rpm"], n), 41)
                ok = np.array([(lambda q: q["current"] >= -i_motor_max and -q["electrical"] / vb <= i_charge_max + 1e-9)(self._point(r, k, "")) for k in ns_])
                k = ns_[int(np.argmax(ok))] if ok.any() else ns_[-1]
                pt = self._point(r, float(k), "regen (current limited)" if not lim_charge else "regen (charge limited)")
                if i_charge_max <= 0:                   # no charging at all: only the short circuit brakes (the windings take it all)
                    pt = self._point(r, self.short_rpm(V, rho), "short-circuit brake (no charging)")
                    pt["electrical"] = 0.0
        return pt

    def short_rpm(self, V, rho=RHO0) -> float:
        """The rpm where the motor's terminal voltage is zero: the short-circuit brake."""
        r = self.rows(V, rho)
        vn = np.maximum.accumulate(r["voltage"])
        if vn[0] >= 0:
            return float(r["rpm"][0])
        return float(np.interp(0.0, vn, r["rpm"]))

    def max_drag_rpm(self, V, rho=RHO0) -> float:
        """The rpm (between the short circuit and the free windmill) where the braked propeller's drag is largest: below
        it the blades stall and the drag falls again."""
        r = self.rows(V, rho)
        fw = self.freewheel(V, rho)
        if fw["mode"] == "stopped":
            return 0.0
        n_s = self.short_rpm(V, rho)
        ns_ = np.linspace(n_s, fw["rpm"], 60)
        T = np.interp(ns_, r["rpm"], r["thrust"])
        return float(ns_[int(np.argmin(T))])

    def brake(self, V, b, rho=RHO0, v_batt=None, *, i_charge_max=math.inf, i_motor_max=None) -> dict:
        """The brake command b in 0..1 at airspeed V: 0 = the free windmill (no current), 1 = the rpm of the largest drag
        (``max_drag_rpm``); between, the ESC holds a voltage below the back-EMF, the propeller slows, its drag and the
        charging current grow. Returns ``at``'s dict (with the limits)."""
        vb = self.battery_v if v_batt is None else float(v_batt)
        fw = self.freewheel(V, rho)
        if fw["mode"] == "stopped" or b <= 0:
            return fw
        r = self.rows(V, rho)
        n = fw["rpm"] - float(np.clip(b, 0, 1)) * (fw["rpm"] - self.max_drag_rpm(V, rho))
        th = max(float(np.interp(n, r["rpm"], np.maximum.accumulate(r["voltage"]))), 0.0) / vb
        return self.at(V, th, rho, vb, i_charge_max=i_charge_max, i_motor_max=i_motor_max)

    def max_thrust(self, V, rho=RHO0, v_batt=None):
        return self.at(V, 1.0, rho, v_batt)["thrust"]

    def throttle_for_thrust(self, V, T, rho=RHO0, v_batt=None):
        vb = self.battery_v if v_batt is None else float(v_batt)
        r = self.rows(V, rho)
        Tr = np.maximum.accumulate(r["thrust"])
        if T >= self.at(V, 1.0, rho, vb)["thrust"]:
            return 1.0
        n = float(np.interp(T, Tr, r["rpm"]))
        return float(np.clip(np.interp(n, r["rpm"], r["voltage"]) / vb, 0.0, 1.0))

    def electrical_for_thrust(self, V, T, rho=RHO0, v_batt=None):
        if T > self.max_thrust(V, rho, v_batt) + 1e-9:
            return math.inf
        return self.at(V, self.throttle_for_thrust(V, T, rho, v_batt), rho, v_batt)["electrical"]

    def map_at(self, rho=RHO0, battery_v=None) -> PropulsionMap:
        """NISUS's ``PropulsionMap`` at density rho (the positive, driving region: for the envelope arithmetic of
        ``nisus_flight.MapUnit`` / ``merlin_flight.performance``)."""
        bv = self.battery_v if battery_v is None else battery_v
        T = np.zeros((len(self.V), len(self.rpm))); Q = np.zeros_like(T); I = np.zeros_like(T); Vn = np.zeros_like(T); P = np.zeros_like(T)
        for i, v in enumerate(self.V):
            r = self.rows(v, rho)
            T[i], Q[i], I[i], Vn[i], P[i] = r["thrust"], r["torque"], r["current"], r["voltage"], r["shaft_power"]
        keep = np.all(Vn > 0, axis=0)
        T = np.maximum.accumulate(np.maximum(T[:, keep], 0.0), axis=1) + np.arange(keep.sum()) * 1e-9
        return PropulsionMap(self.V.copy(), self.rpm[keep].copy(), T, np.maximum(P[:, keep], 0), np.maximum(Q[:, keep], 0), np.maximum(I[:, keep], 0),
                                Vn[:, keep], bv, self.esc_efficiency, self.motor_name, dict(self.notes, rho=rho))

    def to_json(self) -> dict:
        return {"V": self.V.tolist(), "rpm": self.rpm.tolist(), "thrust0": self.thrust0.tolist(), "torque0": self.torque0.tolist(), "kv": self.kv,
                "R": self.R, "I0": self.I0, "max_current_a": self.max_current_a, "battery_v": self.battery_v, "esc_efficiency": self.esc_efficiency,
                "motor_name": self.motor_name, "rho0": RHO0, "notes": self.notes}

    @classmethod
    def from_json(cls, d: dict) -> "NisusPlusDrive":
        return cls(np.asarray(d["V"]), np.asarray(d["rpm"]), np.asarray(d["thrust0"]), np.asarray(d["torque0"]), d["kv"], d["R"], d["I0"],
                   d["max_current_a"], d["battery_v"], d["esc_efficiency"], d["motor_name"], d.get("notes", {}))


@dataclass
class Unit:
    """A propulsor tabulated over airspeed ``V`` (m/s) and ``rpm``: ``thrust[i, j]`` net thrust [N] on the aircraft,
    ``power[i, j]`` shaft power [W]. Full throttle is the rpm at which the battery's electrical limit
    ``max_electrical_w`` times ``drive_efficiency`` is reached, or ``rpm_max``, whichever comes first."""
    name: str
    V: np.ndarray
    rpm: np.ndarray
    thrust: np.ndarray
    power: np.ndarray
    mass_kg: float
    max_electrical_w: float
    drive_efficiency: float = 0.85
    notes: str = ""

    def _row(self, V):
        V = float(np.clip(V, self.V[0], self.V[-1]))
        i = min(int(np.searchsorted(self.V, V, side="right")) - 1, len(self.V) - 2)
        t = (V - self.V[i]) / (self.V[i + 1] - self.V[i])
        return (1 - t) * self.thrust[i] + t * self.thrust[i + 1], (1 - t) * self.power[i] + t * self.power[i + 1]

    def full(self, V):
        """(net thrust [N], electrical power [W], rpm) at full throttle and airspeed V."""
        T, P = self._row(V)
        p_max = self.max_electrical_w * self.drive_efficiency
        if P[-1] <= p_max:
            return float(T[-1]), float(P[-1] / self.drive_efficiency), float(self.rpm[-1])
        n = float(np.interp(p_max, P, self.rpm))
        return float(np.interp(n, self.rpm, T)), self.max_electrical_w, n

    def electrical_power(self, V, T_net):
        """Electrical power [W] for a net thrust ``T_net`` at V; ``inf`` if more than full throttle gives."""
        T, P = self._row(V)
        if T_net > self.full(V)[0] + 1e-9:
            return math.inf
        n = float(np.interp(T_net, T, self.rpm))           # T rises with rpm on the working branch
        return float(np.interp(n, self.rpm, P)) / self.drive_efficiency


@dataclass(frozen=True)
class DriveTable:
    """One drive tabulated against the axial inflow ``V`` [m/s] (negative inflow is read as 0, beyond the table as its
    last row): full-throttle thrust ``T_max`` [N] (current-limited rows held at the motor's current limit), the
    torque-to-thrust ratio ``kappa`` [m] (reaction torque per newton of thrust, near the hover thrust), and the
    electrical power over a thrust grid ``P[i, j]`` at ``V[i]``, ``T[j]``. ``T_min`` is the least thrust the controller
    may command (the motor kept spinning: an ESC's idle)."""
    V: tuple
    T_max: tuple
    kappa: tuple
    T: tuple
    P: tuple
    T_min: float = 0.25
    diameter: float = 0.2286
    rpm_hover: float = 0.0
    source: str = ""
    P_cap: float = 370.0            # one drive's electrical ceiling [W]: the motor's current limit at the pack voltage (25 A × 14.8 V)

    def t_max(self, v_axial):
        return np.interp(np.maximum(v_axial, 0.0), self.V, self.T_max)

    def kap(self, v_axial):
        return np.interp(np.maximum(v_axial, 0.0), self.V, self.kappa)

    def power(self, thrust, v_axial):
        """Electrical power [W] of one drive at ``thrust`` [N] and axial inflow (bilinear in the table)."""
        V, T, P = np.asarray(self.V), np.asarray(self.T), np.asarray(self.P)
        v = float(np.clip(v_axial, V[0], V[-1]))
        i = min(int(np.searchsorted(V, v, side="right")) - 1, len(V) - 2)
        a = (v - V[i]) / (V[i + 1] - V[i])
        row = np.minimum((1 - a) * P[i] + a * P[i + 1], self.P_cap)      # beyond the table's full thrust the rows extrapolate: cap them
        return float(min(np.interp(max(float(thrust), 0.0), T, row), self.P_cap))
