"""Kestrel-4Q — the transitions: hover → wing-borne flight and back, through the stall (notebook 40).

A tailsitter crosses the whole range of angle of attack twice per flight: pitching over from the hover (α ≈ 90°) to
cruise (α ≈ 5°), and pitching back up to the hover. Between α ≈ 11° and 60° the free-stream wing is stalled. The
question this module answers: **can the four motors, their propellers and the controller carry the aircraft through
that stall — hold the attitude, hold the height, keep the rotors inside their limits — in both directions?**

The model, honest about what it is: a rigid body in the vertical plane (x, z, pitch), three degrees of freedom.

- Each wing is split into the span **blown** by its two propellers and the rest. The slipstream at the wing (momentum
  theory, Glauert's induced velocity in inclined inflow, developed to the wing's quarter chord: Δv = vᵢ (1 + d/√(R²+d²)),
  contracted by continuity) adds Δv along the body axis, so the blown span sees a much smaller angle of attack and a
  higher dynamic pressure than the free stream — the reason a tailsitter's wing is not as stalled as its flight path
  says. Each part uses ``kestrel_flight.Polar`` (or the CFD's table, ``airframe=``) at its own local flow.
- The pod's cross-flow drag; the propellers' normal force (``kestrel_flight.prop_cn_alpha``) at the propeller plane.
- The rotors: the controller's thrusts reach the air through a first-order motor lag; each rotor between ``T_min``
  (idle) and the full-throttle thrust at its axial inflow (``kestrel_systems.DriveTable``).
- The controller: a pitch schedule (90° → the cruise attitude over ``t_tr``, or back), an inner PID on pitch whose
  gains are scheduled on the measured airspeed (as a tailsitter autopilot scales its gains) but with no knowledge of
  the aerodynamic moment, an outer loop that holds the height with the collective thrust (with a crude
  linear lift estimate, no blowing and no stall — the real wing differs from what the autopilot believes) and, in
  cruise, the airspeed with the thrust and the height with the pitch; in the hover it holds the ground speed at zero by
  leaning. The moment goes to the differential thrust (upper pair against lower pair: the gap is the lever)
  and to the elevons (optional: trailing-edge surfaces on the blown outer panels) in proportion to their authority; the
  collective gives way when the rotors saturate (attitude before height). ``floor`` keeps a least collective through
  the pitch schedule.
- The airframe is a list of **panels** (``Vehicle.panels``, ``biplane_panels``): each at its dihedral φ about the
  pod's axis, its height, its extent along its own span axis and its rotor. The biplane is four flat panels; an
  X-wing (``kestrel_xwing``) four panels at ±45°, whose lift normal is tilted: a panel sees w cos φ and gives back cos φ
  of its normal force vertically. The same model flies both.
- Roll and yaw are not simulated. ``wing_drop`` checks the roll axis along the trajectory: the rolling moment of one
  side's free-stream panels stalling before the other's (ΔCL = ``WING_DROP_DCL``) against what is left to resist it — the
  rotors' reaction torque (one diagonal pair up, the other down: the only roll lever without surfaces) and the elevons
  used differentially.

The height loop commands a vertical speed limited to +3 / −2 m/s (``V_CLIMB``, ``V_DESCENT``): without the limit, the
balloon of a back transition is followed by a fall at idle thrust, tail first, into reversed flow — a tail slide.

Pass criteria (``CRITERIA``): no more than 3 m of height lost, no more than 3 m gained going forward (10 m going back:
a back transition is flown as a climbing pitch-up), pitch tracking within 10°, the moment commanded delivered (shortfall
under 0.2 s in all), the roll check's demand below the authority. Frame: the earth's x forward, z up; θ = the nose's
elevation (90° = hovering nose-up); the body axes are ``kestrel``'s.

The model itself is generic and lives in ``vegeta.aeromant.transitions``; this module builds Kestrel's vehicles for it
(``make_vehicle``) and re-exports the model, so ``kestrel_transition.simulate`` and the rest read as before.
"""
from __future__ import annotations

import kestrel
import kestrel_flight as kf
import kestrel_systems as ks
from vegeta.aeromant.transitions import *  # noqa: F401,F403
from vegeta.aeromant.transitions import (CRITERIA, G, RHO, Episode, Vehicle, _alpha_trim, _sched, airframe_forces,  # noqa: F401
                                         biplane_panels, elevon_moment_per_rad, induced_velocity, matrix, metrics,
                                         remedy_matrix, remedy_verdict, simulate, slipstream, verdict)


def kestrel_rotors(p) -> dict:
    """The Kestrel-specific parts of a transition vehicle: its propellers' normal force (De Young, calibrated by
    prop_lab when on file) and the cruise airspeed."""
    return {"prop_cn_alpha": lambda T, V, _p=p: kf.prop_cn_alpha(T, V, _p), "V_cruise": kf.V_CRUISE}


def make_vehicle(p=None, *, name="Kestrel-4Q", payload=True, x_cg=None, elevons=True, faired=False, extra_kg=0.0,
                 airframe=None, drive=None, pol=None, aero=None) -> Vehicle:
    """The vehicle from the CAD's layout, the mass table (the CG at ``x_cg`` if given), the polar and the drive."""
    p = kestrel.resolve(p)
    L = kestrel.Kestrel4Q.layout(p)
    table = ks.mass_table(p, x_cg_target=x_cg, payload=payload)
    if extra_kg:
        table.append({"item": "extra payload", "mass_kg": extra_kg, "x": L["x_ac"], "y": 0.0, "z": 0.0, "group": "payload", "rod": None})
    ci = ks.cg_inertia(table)
    pol = pol or kf.polar(p, a=aero, faired=faired)
    return Vehicle(name, p, L, ci["mass_kg"], ci["x_cg"], ci["z_cg"], ci["Iyy"], ci["Ixx"], pol, drive or ks.drive_table(),
                   elevons=elevons, airframe=airframe, **kestrel_rotors(p))
