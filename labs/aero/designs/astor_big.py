"""What if ASTOR were twice the size, and what if it had four rotors on an X (2026-11-24).

A first look, on GAVILAN's scorecard (``gavilan_flight.scorecard``), of four changes to ASTOR-TWIN (PROTOHAWK):

* ``twin``       ASTOR-TWIN as it is (``hawk_compare.astor_machine``).
* ``twin_2x_w``  twice every length, the mass x 4: the wing loading kept. This is the LIGHTEST a doubled aircraft can
                 be argued to be (skins and equipment per square metre unchanged); HAND.
* ``twin_2x_c``  twice every length, the mass x 8: the square-cube law, the same aircraft drawn at twice the scale. HAND.
* ``x``          ASTOR-TWIN at its own size with the two motors replaced by four on an X of four panels (GAVILAN-X's
                 arrangement) IN ADDITION to the wing and the tail: thrust on four corners holds the attitude at any
                 airspeed, so the aircraft may hover, stop and fly beyond the stall (a 'tailsitter' to the scorecard).
* ``x_2x_w``     the X at twice the size, wing loading kept.

Scaling rules (all HAND, stated so they can be argued with): areas x k^2; thrust, pack and power x the mass ratio
(thrust/weight and pack share kept — what that asks of the drive is printed); roll rate / k (the same helix angle);
the polar unchanged (twice the Reynolds number would help a little; not credited).

The X (HAND): four panels of ``X_RADIUS`` x ``X_CHORD``, at 45 degrees; they add lift at cos 45 of their area's worth
(``gavilan_flight.gavilan_x``'s treatment) and skin drag; the same total power and disc area on four propellers, so
the thrust table is the twin's. Hover power is momentum theory over a figure of merit. The twin's blown flaps are
lost (no propeller in front of the wing): the scorecard never credited them, so nothing changes here, but
``astor_v_flight``'s low-speed lift would.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

import gavilan_flight as GFm
import hawk_compare as HC

X_RADIUS, X_CHORD = 0.45, 0.15         # each of the four panels [m] (HAND: the propellers clear the wing and each other)
X_SPAR_KG_M = 0.06                     # a panel's carbon tube with a 2 kW motor on its tip [kg/m] (HAND)
X_EXTRA_KG = 0.15                      # two more speed controllers, pods and leads [kg] (HAND; the motors' total mass taken as unchanged)
FOM, DRIVE = 0.60, 0.85                # hover figure of merit; motor and controller (HAND)
DISC_M2 = 2 * math.pi * (13 * 0.0254 / 2) ** 2      # the twin's two 13-inch discs; the X keeps this area on four


def power(T, v):
    """Electrical power for a thrust at an airspeed, the same for every ASTOR here: thrust x speed over a 75 % propeller,
    never less than the hover power of momentum theory over the figure of merit, both over the drive."""
    T = max(T, 0.0)
    return max(T * max(v, 0.0) / (0.75 * DRIVE), T ** 1.5 / math.sqrt(2 * GFm.RHO * DISC_M2) / (FOM * DRIVE)) + 15.0


def scaled(m: GFm.Machine, k: float, mr: float, label: str) -> GFm.Machine:
    return dataclasses.replace(m, name=label, mass=m.mass * mr, span=m.span * k, length=m.length * k, S=m.S * k * k, cda0=m.cda0 * k * k,
                               plan_body=m.plan_body * k * k, t_max=lambda v: mr * m.t_max(v), power=lambda T, v: mr * m.power(T / mr, v),
                               battery_wh=m.battery_wh * mr, roll_rate=m.roll_rate / k, payload_kg=m.payload_kg * mr,
                               notes={**m.notes, "scale": k, "mass_ratio": mr})


def astor_x(base: GFm.Machine) -> GFm.Machine:
    sx = 4 * X_RADIUS * X_CHORD
    c45 = math.cos(math.radians(45.0))
    S = base.S + sx
    p, w = base.polar, base.S / S
    pol = GFm.Polar(cl_alpha=p.cl_alpha * w + 0.5 * p.cl_alpha * (1 - w), cd0=p.cd0 * w, k=p.k, cl_max=p.cl_max * w + GFm.CL_MAX_3D * c45 * (1 - w),
                    cd90=p.cd90, source="ASTOR-TWIN's polar and four X panels at half the slope and cos 45 of the maximum lift, by area (HAND)")
    cf = GFm.cf_turbulent(25.0 * X_CHORD / GFm.NU)
    cda = cf * (1 + 2 * 0.10) * 2.04 * sx * 1.1 + 0.0010                    # panels' skin with interference, and four pods (HAND)
    mass = base.mass + 2.04 * sx * GFm.AREAL_DENSITY + 4 * X_RADIUS * X_SPAR_KG_M + X_EXTRA_KG

    return dataclasses.replace(base, name="ASTOR-X: wing, tail and four rotors on an X", kind="tailsitter", mass=mass, S=S, polar=pol, cda0=cda,
                               roll_rate=math.radians(600.0) * 0.48 / (2 * X_RADIUS * c45 + 0.3), plan_body=0.11 * 1.09 * 0.8,
                               notes={**base.notes, "x_area_m2": sx, "added_kg": mass - base.mass, "added_cda_m2": cda})


def machines() -> dict:
    twin = dataclasses.replace(HC.machines()["astor_twin"], power=power)
    x = astor_x(twin)
    return {"twin": twin,
            "twin_2x_w": scaled(twin, 2.0, 4.0, "ASTOR-TWIN x 2, mass x 4 (wing loading kept)"),
            "twin_2x_c": scaled(twin, 2.0, 8.0, "ASTOR-TWIN x 2, mass x 8 (square-cube)"),
            "x": x,
            "x_2x_w": scaled(x, 2.0, 4.0, "ASTOR-X x 2, mass x 4"),
            "gavilan_x": GFm.gavilan_x()}


def study() -> dict:
    ms = machines()
    rows = {r["key"]: r for r in GFm.scorecard(ms)}
    out = {}
    for key, m in ms.items():
        r = rows[key]
        row = {k: r.get(k) for k in HC.KEEP}
        row.update({"wing area [m2]": m.S, "wing loading [kg/m2]": m.mass / m.S, "full power asked of the drive [kW]": m.power(m.t_max(0.0), 0.0) / 1e3 if key != "gavilan_x" else None,
                    "pack [Wh]": m.battery_wh, "roll rate [deg/s]": math.degrees(m.roll_rate), "span / radius inst @15": m.span / r["radius inst @15 [m]"]})
        out[key] = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else None if isinstance(v, (float, np.floating)) and not math.isfinite(v)
                        else round(float(v), 2) if isinstance(v, (float, np.floating, int, np.integer)) else v) for k, v in row.items()}
    return out


if __name__ == "__main__":
    import yaml
    import arigo1_model as am
    out = study()
    (am.DATA / "astor_twin" / "astor_big.yaml").write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=160))
    keys = list(out)
    for k in list(out["twin"])[:-7] + list(out["twin"])[-6:]:
        print(f"{k[:36]:36s} " + " ".join(f"{str(out[m][k])[:15]:>15s}" for m in keys))
