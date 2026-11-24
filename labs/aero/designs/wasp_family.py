"""The WASP family on one basis, and which builds are kept.

Four builds came out of ``wasp`` and ``wasp_micro``: WASP-1 and WASP-1F (ÉMERILLON small, for speed) and WASP-M and
WASP-ML (smaller still, with a payload bay). They were worked out for different questions and quoted on different
terms. ``compare`` puts them on one: each with the same payload aboard (none, and 10 g), its top speed, its static
thrust, and the delivery range at 100 km/h with 20 % kept and half a minute of hover at each end
(``wasp_micro.delivery``; each build on its own airframe constants).

``KEPT`` is the family from here on; ``RETIRED`` says what each dropped build was and why. A retired build stays
computable (its configuration is still in its module): nothing is deleted.
"""
from __future__ import annotations

import wasp
import wasp_micro as wm

BUILDS = {  # name -> (configuration, micro airframe constants?)
    "WASP-1": (wasp.WASP1, False), "WASP-1F": (wasp.WASP1_FAST, False),
    "WASP-M": (wm.WASP_M, True), "WASP-ML": (wm.WASP_M_LONG, True),
}
KEPT = {
    "WASP-1F": "the fast one: the highest top speed and thrust margin of the four, and the longest range of them with a payload",
    "WASP-ML": "the small carrier: under 250 g with its payload, 20 g of load, 200 km/h, the cheapest way to move a few grams a few kilometres",
}
RETIRED = {
    "WASP-1": "between the two kept builds: slower and shorter-ranged than WASP-1F; against WASP-ML it is 65 g heavier and goes two thirds "
              "as far for 20 km/h more top speed; its one reason was the 250 g line, which WASP-ML is also under",
    "WASP-M": "WASP-ML with a smaller pack: 40 g lighter, the same speed, less than half the range; nothing it does that WASP-ML does not",
}


def compare(payloads=(0.0, 10.0)) -> list:
    rows = []
    for name, (c, micro) in BUILDS.items():
        for pay in payloads:
            r = wm.delivery(c, pay, constants=micro)
            cr = r["cruise"]
            rows.append({"build": name, "kept": name in KEPT, "payload_g": pay, "mass_g": round(r["mass_g"], 1), "top_kmh": round(r["kmh"], 0),
                         "static_tw": round(float(r["static_tw"]), 2), "length_mm": round(r["length"] * 1e3), "across_tips_mm": round(r["across_tips"] * 1e3),
                         "battery_wh": round(r["wh"], 2), "cruise_w": None if cr is None else round(cr["electric_w"], 1), "hover_w": round(float(r["hover_w"]), 1),
                         "range_km": None if r["range_km"] is None else round(r["range_km"], 1), "minutes": None if r["minutes"] is None else round(r["minutes"], 1),
                         "g_km_payload": None if not pay or not r["range_km"] else round(pay * r["range_km"], 0), "price_usd": round(r["price"], 0),
                         "motor_dT": round(float(r["motor_dT"])), "fails": r["fails"], "label": r["label"]})
    return rows
