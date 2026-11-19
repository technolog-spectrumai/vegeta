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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "designs"))

from gavilan_slingshot import (CARTRIDGE_KG, GAS_ONE_SHOT_KG, HARDWARE_G, HUNTER_KG, MARGIN, U_RETURNED,  # noqa: F401  (moved to the design)
                               launcher, spring_mass, table)


if __name__ == "__main__":
    rows = table()
    for r in rows:
        print(f"{r['speed_m_s']:4.0f} m/s  {r['spring']:18s} {'opposed' if r['opposed'] else 'single ':8s} energy {r['energy_J']:6.1f} J  spring {r['spring_kg'] * 1e3:7.0f} g"
              f"  launcher {r['launcher_kg'] * 1e3:7.0f} g  (gas one-shot {GAS_ONE_SHOT_KG * 1e3:.0f} g)  kick {r['hunter_kick_m_s']:.2f} m/s")
    Path(__file__).with_name("gavilan_spring_net_results.json").write_text(json.dumps(
        {"source": "gavilan_spring_net.table()", "note": "hand estimates of material energy and hardware masses, x 1.25", "u_returned_J_per_kg": U_RETURNED,
         "hardware_g": HARDWARE_G, "gas_one_shot_kg": GAS_ONE_SHOT_KG, "rows": rows}, indent=1, default=lambda o: None))
