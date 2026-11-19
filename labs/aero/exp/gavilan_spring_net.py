"""A spring-thrown net for GAVILAN-X, against the one-shot air launcher: how heavy, how fast, with or without recoil.

The opposed spring ejector: one spring between two masses pushes them apart — the net cartridge forward, a counter-
mass (or a second net) aft — so their momenta cancel and the aircraft feels no recoil and no back-blast. What it costs
is the energy: two masses at speed ``v`` take ``m v^2`` (each mass ``m``), against ``m v^2 / 2`` for a net thrown
alone. And a spring cannot throw faster than its own material lets it: a stretched spring's own parts move too, and
moving them takes energy, so the spring's mass grows without bound as ``v^2 / 6`` approaches the energy it stores
per kilogram (``spring_mass``).

Material energy per kilogram that a spring gives back (HAND ESTIMATES, from the textbook limits, with the margin stated):
- **steel coil (music wire)**: tau^2 / (4 G rho) = 326 J/kg at 900 MPa shear, halved for a round wire's stress
  gradient -> 160 J/kg;
- **latex rubber (slingshot bands)**: about 2000 J/kg stored at a working stretch, 80 % given back -> 1600 J/kg;
- **glass-fibre limbs (a bow)**: sigma^2 / (2 E rho) = 4200 J/kg at 800 MPa, a third for a bending limb, halved
  again for its hardware -> 700 J/kg.

The net's speed: the light net (net_drone ``--light``: 1.5 m, four 8 g weights splayed 12 degrees) opens to the same
1.75-1.95 m at 3-4 m whether it leaves at 15 or 45 m/s (``slow_net`` in the experiment's log): the splay sets the
opening, the speed sets the flight time and the drop. So a slower throw is worth looking at.

    python gavilan_spring_net.py        # the table
"""
from __future__ import annotations

import json
from pathlib import Path

CARTRIDGE_KG = 0.070                 # the light net's cartridge (net_light_design)
U_RETURNED = {"steel coil": 160.0, "latex bands": 1600.0, "glass-fibre limbs": 700.0}       # J/kg, hand estimates (above)
MARGIN = 1.25
# the rest of a spring launcher (hand estimates, g): a carbon tube open at both ends, two carriages, a latch released
# by a servo, the pod fairing; a counter-mass of the cartridge's mass when the ejector is opposed
HARDWARE_G = {"carbon tube 0.30 m, 32 mm bore, open at both ends": 75.0, "two carriages and the latch": 35.0,
              "release servo": 12.0, "pod fairing and strut (printed foam)": 40.0, "trigger electronics and wiring": 20.0}
GAS_ONE_SHOT_KG = 0.685              # gavilan_flight.net_pod_kg(): the one-shot air launcher in GAVILAN-X's pod, with its margin
HUNTER_KG = 2.03 - GAS_ONE_SHOT_KG   # GAVILAN-X with no catcher (gavilan_flight.gavilan_x_net() less its pod)


def spring_mass(v: float, moving_kg: float, u: float) -> float:
    """The spring that throws ``moving_kg`` (all the masses together) to ``v``: its own kinetic energy at the end is
    1/6 of its mass x v^2 (a spring between two masses, its middle still) or 1/3 (fixed at one end) — the 1/6 case
    here, so  m_s u = moving v^2 / 2 + m_s v^2 / 6. ``inf`` past the material's limit."""
    room = u - v * v / 6.0
    return float("inf") if room <= 0 else 0.5 * moving_kg * v * v / room


def launcher(v: float, material: str, opposed: bool, countermass: str = "dead mass") -> dict:
    moving = CARTRIDGE_KG * (2 if opposed else 1)
    ms = spring_mass(v, moving, U_RETURNED[material])
    hw = sum(HARDWARE_G.values()) / 1e3
    kg = (ms + hw + moving) * MARGIN
    recoil = 0.0 if opposed else CARTRIDGE_KG * v
    return {"speed_m_s": v, "spring": material, "opposed": opposed, "aft": countermass if opposed else "nothing",
            "energy_J": 0.5 * moving * v * v, "spring_kg": ms, "launcher_kg": kg, "lighter_than_gas_by_kg": GAS_ONE_SHOT_KG - kg,
            "recoil_N_s": recoil, "hunter_kick_m_s": recoil / (HUNTER_KG + kg - CARTRIDGE_KG)}


def table(speeds=(15.0, 20.0, 30.0, 45.0)) -> list:
    rows = []
    for v in speeds:
        for mat in U_RETURNED:
            for opposed in (True, False):
                rows.append(launcher(v, mat, opposed, "a second net" if opposed else "nothing"))
    return rows


if __name__ == "__main__":
    rows = table()
    for r in rows:
        print(f"{r['speed_m_s']:4.0f} m/s  {r['spring']:18s} {'opposed' if r['opposed'] else 'single ':8s} energy {r['energy_J']:6.1f} J  spring {r['spring_kg'] * 1e3:7.0f} g"
              f"  launcher {r['launcher_kg'] * 1e3:7.0f} g  (gas one-shot {GAS_ONE_SHOT_KG * 1e3:.0f} g)  kick {r['hunter_kick_m_s']:.2f} m/s")
    Path(__file__).with_name("gavilan_spring_net_results.json").write_text(json.dumps(
        {"source": "gavilan_spring_net.table()", "note": "hand estimates of material energy and hardware masses, x 1.25", "u_returned_J_per_kg": U_RETURNED,
         "hardware_g": HARDWARE_G, "gas_one_shot_kg": GAS_ONE_SHOT_KG, "rows": rows}, indent=1, default=lambda o: None))
