"""The ARIGO family's names (notebook 39). One airframe family, four propulsion designs:

- **ARIGO-0** — one tractor propeller in the nose; the camera under the nose looks through the disc (synchronised
  global-shutter capture, ``arigo1_camera``). Built by ``arigo1.Arigo1`` with ``propulsion="tractor_lowcam"``.
  Its earlier form with the camera on a 150 mm chin strut is kept for comparison as ``propulsion="tractor"``.
- **ARIGO-1** — one pusher propeller behind the tail; the nose camera sees clear. ``propulsion="pusher"``.
- **ARIGO-2** — two folding tractors on wing nacelles (``arigo2.Arigo2``). ARIGO-1's ``propulsion="twin"`` is its
  first, bare sketch (nacelles without propellers), kept only for the earlier comparisons.
- **ARIGO-20** — two ducted fans over the rear fuselage, H tail (``arigo20.Arigo20``).

Each of ARIGO-0/1/2 takes one of the three wings (``goshawk``, ``zero``, ``hybrid``); ARIGO-20 has its own.
The code keeps its internal keys (``pusher``, ``tractor_lowcam``...) so the saved CFD cases stay valid; the names
are for people, pictures and result files.
"""
from __future__ import annotations

NAMES = {"tractor_lowcam": "ARIGO-0", "tractor": "ARIGO-0 (chin-strut camera)", "pusher": "ARIGO-1",
         "twin": "ARIGO-2 sketch (bare nacelles)"}
DESIGNS = {"ARIGO-0": ("arigo1", "tractor_lowcam"), "ARIGO-1": ("arigo1", "pusher"), "ARIGO-2": ("arigo2", None), "ARIGO-20": ("arigo20", None)}


def name(propulsion: str, wing: str | None = None) -> str:
    n = NAMES.get(propulsion, propulsion)
    return f"{n} {wing}" if wing else n


def design(aircraft: str, wing: str = "hybrid"):
    """(Design instance, parameters) for ``aircraft`` (e.g. "ARIGO-0") on ``wing``."""
    mod, prop = DESIGNS[aircraft.upper()]
    if mod == "arigo1":
        import arigo1 as a1
        return a1.Arigo1(), a1.variant(wing, prop)
    if mod == "arigo2":
        import arigo2 as a2
        return a2.Arigo2(), a2.variant(wing)
    import arigo20 as a20
    return a20.Arigo20(), a20.variant()
