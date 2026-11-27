"""The surveillance payload for the guard carrier's drones (Esmeralda, ``labs/marine/designs/esmeralda.py``): a swap-in
pod that replaces Gannet's 5 kg crate (``gannet.py``) and Fulmar's cargo pod (``fulmar.py``) - a stabilised day/IR
gimbal with a laser rangefinder, an AIS receiver and a video datalink to the carrier. The carrier's payload loader
swaps a pod for a crate in the time of a battery swap.

Two pods, one per drone, because the two aircraft carry different weights and stay up for different times:

* **Gannet's pod** (``POD["gannet"]``): a light gimbal (HD day 30x, uncooled LWIR 640, 1.5 km rangefinder), 2.2 kg -
  a quad's few minutes on station near the carrier, where the boat and the ship are;
* **Fulmar's pod** (``POD["fulmar"]``): a cooled MWIR gimbal (HD day 30x, cooled MWIR 640, 5 km rangefinder), 3.8 kg -
  the long patrol, the search for what the radar cannot see.

A small maritime radar for Fulmar (``RADAR_OPTION``) is weighed and does not fit with the gimbal in 5 kg.

The ranges are ``labs/marine/designs/sensing.py``'s (Johnson criteria, the atmosphere, the horizon) taken from the
drone's altitude: a slant range to the target, given as the **ground range** (the distance along the sea). **Sea fog
is a layer** (150 m thick, assumed): from above it, a drone's line of sight crosses only the layer's part of the
slant path, so it sees down through fog farther than a ship sees across it. Class values (assumed) throughout; no
vendor's equipment.
"""
from __future__ import annotations

import math
import os
import sys

_MARINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "marine", "designs")
if _MARINE not in sys.path:
    sys.path.insert(0, _MARINE)
import sensing  # noqa: E402

#: The pods (class values, assumed): parts (kg, W, EUR), the gimbal's channels, its rangefinder and jitter, the pod's
#: drag area, and its datalink.
POD = {
    "gannet": {
        "parts": {"gimbal: HD day 30x zoom + uncooled LWIR 640 + laser rangefinder 1.5 km": (1.30, 25.0, 18000.0),
                  "AIS receiver (dual channel) + antenna": (0.17, 2.0, 600.0),
                  "video datalink 2 W (COFDM/MANET) + 2 antennas": (0.35, 12.0, 4000.0),
                  "pod: the crate hook's quick-release mount, dampers, connector": (0.35, 0.0, 1200.0)},
        "channels": [{"band": "visible", "pixels_h": 1920, "hfov_narrow_deg": 2.3, "hfov_wide_deg": 63.0, "zoom": 30},
                     {"band": "ir", "pixels_h": 640, "hfov_narrow_deg": 12.4, "hfov_wide_deg": 12.4, "zoom": 1, "netd_mK": 50.0}],
        "lrf_km": 1.5, "jitter_urad": 50.0, "cda_m2": 0.02,
        "link": {"tx_W": 2.0, "carrier_antenna": "omni on the mast", "class_range_km": 15.0, "video_mbps": 4.0, "capacity_mbps": 8.0},
        "altitudes_m": (120.0, 300.0)},
    "fulmar": {
        "parts": {"gimbal: HD day 30x zoom + cooled MWIR 640 (2.5 deg narrow) + laser rangefinder 5 km": (2.70, 45.0, 75000.0),
                  "AIS receiver (dual channel) + antenna": (0.17, 2.0, 600.0),
                  "video datalink 5 W + antennas (the carrier's tracking antenna)": (0.40, 25.0, 8000.0),
                  "pod: the cargo bay's mount, dampers, a retracting gimbal well": (0.50, 0.0, 1500.0)},
        "channels": [{"band": "visible", "pixels_h": 1920, "hfov_narrow_deg": 2.3, "hfov_wide_deg": 63.0, "zoom": 30},
                     {"band": "ir", "pixels_h": 640, "hfov_narrow_deg": 2.5, "hfov_wide_deg": 25.0, "zoom": 10, "netd_mK": 25.0}],
        "lrf_km": 5.0, "jitter_urad": 30.0, "cd0_delta": -0.002,
        "link": {"tx_W": 5.0, "carrier_antenna": "tracking directional on the mast", "class_range_km": 60.0, "video_mbps": 3.0, "capacity_mbps": 6.0},
        "altitudes_m": (300.0, 900.0)},
}
#: A small maritime surveillance radar for Fulmar, weighed (class values, assumed).
RADAR_OPTION = {"what": "lightweight maritime surveillance radar (X/Ku-band, 360 deg scan, small-target mode)", "mass_kg": 3.6,
                "power_W": 110.0, "price_eur": 90000.0, "skiff_km_from_900m": 12.0}
#: Gannet's **ladder pod** (class values, assumed): the escalation ladder's lights and voice from the air, for a carrier
#: whose boat cannot reach a fast skiff early - a searchlight, a white strobe (non-laser) and a directional
#: loudspeaker on a pan-tilt under the body, with the 2 W link carrying the operator's live voice; no gimbal camera
#: (the flight stack's deck camera stays). It replaces the crate or the EO/IR pod.
LADDER_POD = {
    "parts": {"LED searchlight 6000 lm, 6 deg, on the pan-tilt": (0.90, 60.0, 2500.0),
              "white LED strobe 0.5 Mcd peak, 5 ms at 6 Hz (non-laser)": (0.30, 30.0, 1500.0),
              "directional loudspeaker, 130 dB at 1 m (2 kHz), 30 deg": (1.20, 60.0, 4000.0),
              "pan-tilt head and the crate hook's quick-release mount": (0.45, 8.0, 2500.0),
              "video and voice datalink 2 W + antennas": (0.35, 12.0, 4000.0)},
    "light_lm": 6000.0, "light_deg": 6.0, "strobe_cd": 5.0e5, "speaker_db": 130.0, "cda_m2": 0.025}
FOG_LAYER_M = 150.0
TARGETS = ("skiff_6m", "fishing_20m")


def pod_totals(drone: str) -> dict:
    parts = POD[drone]["parts"]
    return {"mass_kg": round(sum(m for m, _, _ in parts.values()), 2), "power_W": sum(w for _, w, _ in parts.values()),
            "price_eur": sum(e for _, _, e in parts.values())}


def ground_dri(drone: str, target: str, altitude_m: float) -> dict:
    """The pod's DRI at ``altitude_m``, as ground ranges (km), day (the better channel) and night (IR), in clear air,
    haze and fog (the fog a layer ``FOG_LAYER_M`` thick)."""
    pod = POD[drone]
    t = sensing.TARGETS[target]
    jit = pod["jitter_urad"] * 1e-6
    rows = {}
    for light in ("day", "night"):
        def slant(weather):
            best = {}
            for ch in pod["channels"]:
                r = sensing.camera_dri(ch, t, altitude_m, weather, light == "night", jit)["range_m"]
                for task, v in r.items():
                    best[task] = max(best.get(task, 0), v)
            return best
        clear = slant("clear")
        for weather in ("clear", "haze", "fog"):
            s = slant(weather)
            if weather == "fog" and altitude_m > FOG_LAYER_M:          # only the layer's share of the path is in fog
                s = {k: min(clear[k], v * altitude_m / FOG_LAYER_M) for k, v in s.items()}
            rows[f"{light}_{weather}"] = {k: round(math.sqrt(max(v * v - altitude_m ** 2, 0.0)) / 1000, 2) for k, v in s.items()}
    return rows


def fov(drone: str) -> dict:
    return {ch["band"]: {"narrow_deg": ch["hfov_narrow_deg"], "wide_deg": ch["hfov_wide_deg"], "zoom": ch["zoom"],
                         "pixels_h": ch["pixels_h"], **({"netd_mK": ch["netd_mK"]} if "netd_mK" in ch else {})} for ch in POD[drone]["channels"]}


def link(drone: str, altitude_m: float, carrier_mast_m: float = 10.0) -> dict:
    """The datalink's range: the radio horizon between the drone and the carrier's mast, or the link budget's class
    range, whichever is shorter; the video it carries."""
    L = POD[drone]["link"]
    hz = sensing.horizon_km(altitude_m, carrier_mast_m, "radar")
    return L | {"altitude_m": altitude_m, "horizon_km": round(hz, 1), "range_km": round(min(hz, L["class_range_km"]), 1),
                "limited_by": "horizon" if hz < L["class_range_km"] else "link budget"}


def radar_fits_fulmar(payload_max_kg: float = 5.0) -> dict:
    """Whether the radar fits Fulmar's payload with the EO/IR pod, and alone (with AIS, link and mount)."""
    with_eoir = pod_totals("fulmar")["mass_kg"] + RADAR_OPTION["mass_kg"]
    alone = RADAR_OPTION["mass_kg"] + sum(m for k, (m, _, _) in POD["fulmar"]["parts"].items() if not k.startswith("gimbal"))
    return {"with_eoir_kg": round(with_eoir, 2), "radar_only_pod_kg": round(alone, 2), "payload_max_kg": payload_max_kg,
            "fits_with_eoir": with_eoir <= payload_max_kg, "fits_alone": alone <= payload_max_kg,
            "verdict": ("does not fit with the gimbal (%.1f kg against %.0f kg); a radar-only pod (%.1f kg) %s - a choice between seeing "
                        "and finding, or a bigger aircraft" % (with_eoir, payload_max_kg, alone, "would fit" if alone <= payload_max_kg else "does not fit either"))}


def ladder_totals() -> dict:
    parts = LADDER_POD["parts"]
    return {"mass_kg": round(sum(m for m, _, _ in parts.values()), 2), "power_W": sum(w for _, w, _ in parts.values()),
            "price_eur": sum(e for _, _, e in parts.values())}


def ladder_effectors() -> dict:
    """What the ladder pod does from the air (``sensing.py``): the light's reach, the strobe's dazzle, the speaker's
    intelligible range - over a skiff's running outboards it is a few metres, so the voice step needs the skiff slowed
    or stopped (or the VHF hail from the carrier); the light and the strobe work at speed."""
    L = LADDER_POD
    return {"searchlight": sensing.searchlight(L["light_lm"], L["light_deg"]) | {"haze_range_m": sensing.searchlight(L["light_lm"], L["light_deg"], 5.0)["range_m"]},
            "strobe": sensing.strobe(L["strobe_cd"], 0.005, 6.0), "loudspeaker": sensing.hailer(L["speaker_db"]),
            "verdict": "lights and strobe: yes, at speed, from 600 m; voice: only to a skiff idling (150 m) or stopped - "
                       "against outboards at speed it is heard at ~17 m, too close for a drone; the hail (step 1) stays the carrier's VHF"}
