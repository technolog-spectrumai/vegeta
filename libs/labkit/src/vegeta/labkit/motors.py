"""Motors as Yax part data (``vendor/yax/docs/engineering.md``, the ``00motor`` blocks): what a Chiron servo and an Ogma
actuator state, in the contract's fields — units in the names, nothing converted, nothing derived here.

A Chiron ``Servo`` is the design's joint model: stall and rated torque at the output, no-load speed, stall current at a
voltage, the PD gains the design ran, the reflected rotor inertia ("armature"). Its torque constant and resistance are
the servo's own *estimates* from the stall point (``Servo.torque_constant``, ``Servo.resistance``): they leave as such,
marked in ``source``. An Ogma ``Actuator`` is the bought unit: datasheet-class values, plus the gear ratio and efficiency
when the entry states them (``gear_ratio``, ``gear_efficiency``), never parsed out of its text.
"""
from __future__ import annotations


def from_servo(servo, *, derived_from: str | None = None) -> dict:
    """The motor block of a joint from its Chiron ``Servo``."""
    out = {"kind": "servo model", "stall_torque_nm": float(servo.stall_torque), "rated_torque_nm": float(servo.rated_torque),
           "no_load_speed_rad_s": float(servo.no_load_speed), "stall_current_a": float(servo.stall_current),
           "voltage_v": float(servo.voltage),
           "kt_nm_per_a": float(servo.torque_constant), "resistance_ohm": float(servo.resistance),
           "control": {"kp": float(servo.kp), "kd": float(servo.kd), "law": "PD clipped to the torque–speed line",
                       "four_quadrant": bool(getattr(servo, "four_quadrant", False))},
           "source": (servo.source or "the design's servo model") +
                     "; kt and resistance are the servo model's estimates from the stall point (kt = τ_stall / I_stall, R = V / I_stall)"}
    if servo.armature:
        out["transmission"] = {"reflected_inertia_kgm2": float(servo.armature), "source": "the servo model's armature (rotor inertia × gear ratio²)"}
    if derived_from:
        out["derived_from"] = derived_from
    return out


def from_actuator(act) -> dict:
    """The motor block of an Ogma catalogue ``Actuator`` (the bought unit, at its output)."""
    out = {"name": act.key, "kind": act.kind, "mass_kg": float(act.mass_g) / 1000.0, "stall_torque_nm": float(act.stall_Nm),
           "rated_torque_nm": float(act.rated_Nm), "stall_current_a": float(act.stall_A), "voltage_v": float(act.voltage_V),
           "no_load_speed_rpm": float(act.no_load_rpm), "source": act.source or "ogma.actuators"}
    tr = {}
    if getattr(act, "gear_ratio", None) is not None:
        tr["gear_ratio"] = float(act.gear_ratio)
    if getattr(act, "gear_efficiency", None) is not None:
        tr["efficiency"] = float(act.gear_efficiency)
    if act.self_locking:
        tr["self_locking"] = True
    if tr:
        tr["source"] = "ogma.actuators: as the catalogue entry states it"
        out["transmission"] = tr
    return out


def catalogue_key(servo) -> str | None:
    """The Ogma catalogue entry a servo was made from (``Servo.from_actuator`` writes ``"<key>: <source>"``; a design may
    write ``"<key>, …"``), or None."""
    from ogma import actuators as act
    src = servo.source or ""
    for a in act.CATALOG:
        if src == a.key or src.startswith(a.key + ":") or src.startswith(a.key + ","):
            return a.key
    return None


def attach(m, robot) -> list:
    """Put the robot's servos on the machine's joints as motor blocks and the catalogue units they came from in the
    machine's ``motors``; the joint names that got one."""
    from ogma import actuators as act
    done = []
    joints = {j.name: j for j in m.joints}
    for cj in robot.joints():
        if cj.servo is None or cj.name not in joints:
            continue
        key = catalogue_key(cj.servo)
        joints[cj.name].data["motor"] = from_servo(cj.servo, derived_from=key)
        if key:
            m.data.setdefault("motors", {})[key] = from_actuator(act.get(key))
        done.append(cj.name)
    return done
