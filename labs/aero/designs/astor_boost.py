"""ASTOR-BOOST — ASTOR-V with a ducted fan in the tail for a boost (a variant, 2026-11-21).

**What the fan is for.** Not top speed for its own sake (a stronger nose motor gives that for less mass): a second
source of thrust that works when the propeller does not. (1) A burst: propeller and fan together for the climb-out
and the overtake (estimate: +50 % acceleration at 20 m/s, top speed ~68 -> ~76 m/s). (2) The clear-eyed approach:
the nose propeller stopped and folded so the chase camera sees through no blades, the fan alone holding ~50 m/s for
the last seconds before the nets go.

**How it gets its air.** The fan sits in the tail, exhausting straight aft. Two **cheek scoops** low on the rear
fuselage's sides (each 38 x 68 mm outside, ~18 cm2 of mouth: together a little over the fan's swept area) feed the annulus around the tail
cone ahead of the fan. Not on the back: the tucked wing panels lie there. The duct's inside is not designed here
(a layout: where things are, what they weigh); an open scoop costs drag when the fan is off (doors: later).

**What moves.** The tail's centre line is the fan's now, so the two nets go back to a **belly bay** (pushed out
downward; no blade turns in the open behind them: the fan is in its duct). The drop camera moves ahead of the bay.
The pack grows to 6S 5000 mAh: propeller and fan together draw ~124 A, over the 4000 pack's 120 A.
Everything else is ASTOR-V's (``astor_v``): the sliding pivot carriage, the fanning tail, the folding tractor
propeller (``astor_prop``). Masses are HAND estimates. Frame: x aft from the nose tip, y right, z up; mm.
"""
from __future__ import annotations

import dataclasses

import cadquery as cq

import arigo20 as a20
import astor_v as V
from vegeta.dedalus import Parameter

FAN_W = 1200.0
_OVERRIDE = {**V._OVERRIDE, "fuselage_length": 705.0, "tail_cone": 230.0, "tail_end_diameter": 26.0}
_NEW = [
    Parameter("edf_diameter", 70.0, "mm", min=40.0, max=100.0, description="the boost fan's rotor"),
    Parameter("edf_length", 150.0, "mm", min=80.0, description="the fan duct: lip to nozzle"),
    Parameter("edf_wall", 2.5, "mm", min=1.0),
    Parameter("nozzle_ratio", 0.88, "", min=0.6, max=1.0),
    Parameter("edf_x", 690.0, "mm", min=400.0, description="the duct's lip from the nose tip"),
    Parameter("intake_x", 545.0, "mm", min=300.0, description="the cheek scoops' mouths from the nose tip"),
    Parameter("intake_width", 38.0, "mm", min=10.0), Parameter("intake_height", 68.0, "mm", min=20.0),
]
_DEFAULTS = {"net_bay_x": 255.0, "belly_camera_x": 215.0, "tail_pivot_y": 30.0, "nets_station_mm": 420.0}

EXTRA_ITEMS = [x for x in V.EXTRA_ITEMS if "net tube" not in x[0]] + [
    ("belly net bay fairing, 2 ejector springs and latches", 45.0, "HAND (+-15)"),
    ("boost fan unit 70 mm, motor included", 175.0, "HAND (+-30); arigo_mass's value"),
    ("boost fan ESC 80 A", 60.0, "HAND (+-15)"),
    ("cheek scoops, duct, tail fairing", 60.0, "HAND (+-25)"),
]
DEFAULT_PACK = "6S 5000 mAh LiPo"


class AstorBoost(V.AstorV):
    """ASTOR-BOOST: ASTOR-V with a ducted fan in the tail fed by two cheek scoops; the nets in a belly bay."""

    parameters = [dataclasses.replace(q, default={**_OVERRIDE, **_DEFAULTS}[q.name]) if q.name in {**_OVERRIDE, **_DEFAULTS} else q
                  for q in V.AstorV.parameters] + _NEW

    def _fan(self, p):
        """The fan in its duct at the tail (ARIGO-20's nacelle generator), and the fairing from the fuselage onto it."""
        nac = a20.Arigo20._nacelle(self, p).translate((p["edf_x"], 0, 0))
        Ro = p["edf_diameter"] / 2 + p["edf_wall"] + 2.0
        R = p["fuselage_diameter"] / 2
        x0, x1 = p["intake_x"] - 40.0, p["edf_x"] + 12.0
        prof = [(x0, 0.0), (x0, R * 0.98), (x0 + 0.45 * (x1 - x0), Ro + 3.0), (x1, Ro + 1.0), (x1, 0.0)]
        fair = cq.Workplane("XY").polyline(prof).close().revolve(360.0, (0, 0, 0), (1, 0, 0))
        fair = fair.cut(cq.Workplane("YZ").circle(p["edf_diameter"] / 2).extrude(40.0).translate((x1 - 39.0, 0, 0)))
        return nac.union(fair)

    def _scoops(self, p):
        """Two cheek scoops low on the rear fuselage's sides, their mouths forward."""
        R = p["fuselage_diameter"] / 2
        w, h = p["intake_width"], p["intake_height"]
        L = p["edf_x"] - p["intake_x"] + 10.0
        out = None
        for s in (1, -1):
            body = (cq.Workplane("XY").box(L, w, h).edges("|X").fillet(min(w, h) * 0.3)
                    .translate((p["intake_x"] + L / 2, s * (R + w / 2 - 9.0), -10.0)))
            out = body if out is None else out.union(body)
        return out

    def _scoop_mouths(self, p):
        R = p["fuselage_diameter"] / 2
        w, h = p["intake_width"], p["intake_height"]
        out = None
        for s in (1, -1):
            m = (cq.Workplane("YZ").rect(w - 7.0, h - 9.0).extrude(3.0).translate((p["intake_x"] - 2.0, s * (R + w / 2 - 9.0), -10.0)))
            out = m if out is None else out.union(m)
        return out

    # the nets: a belly bay (the tail's centre line is the fan's)
    def _net_bay(self, p):
        R = p["fuselage_diameter"] / 2
        n = int(p["net_count"])
        L = n * (p["net_length"] + 12.0) + 30.0
        w, h = p["net_diameter"] + 10.0, p["net_diameter"] * 0.75
        return (cq.Workplane("XY").box(L, w, h).edges("|Y").fillet(h * 0.42).translate((p["net_bay_x"] + L / 2 - 15.0, 0, -R - h / 2 + 8.0)))

    def _nets(self, p):
        R = p["fuselage_diameter"] / 2
        out = None
        for k in range(int(p["net_count"])):
            c = (cq.Workplane("YZ").circle(p["net_diameter"] / 2).extrude(p["net_length"])
                 .translate((p["net_bay_x"] + k * (p["net_length"] + 12.0), 0, -R - p["net_diameter"] * 0.30)))
            out = c if out is None else out.union(c)
        return out

    def _tail(self, p):
        return super()._tail({**p, "fuselage_length": p["edf_x"] + p["edf_length"] - 5.0})      # the fin under the duct's end

    def parts(self, p) -> dict:
        out = super().parts(p)
        out["fan"] = self._fan(p)
        out["scoops"] = self._scoops(p)
        out["scoop_mouths"] = self._scoop_mouths(p)
        return out


def variant(**over) -> dict:
    q = {**_OVERRIDE, **_DEFAULTS}
    q["tail_le_x"] = q["fuselage_length"] - 10.0 - q["htail_root_chord"]
    return {**q, **over}


def resolve(q=None) -> dict:
    return AstorBoost().resolve(**dict(q or variant()))


def budget(p=None, pack=None) -> dict:
    return V.budget(p or resolve(), pack, extra=EXTRA_ITEMS, default_pack=DEFAULT_PACK)


def checks() -> dict:
    p = resolve()
    mb = budget(p)
    bal = V.balance(p, mb=mb, stations={"boost fan unit": 765.0, "boost fan ESC": 600.0, "cheek scoops": 640.0, "belly net bay": 420.0})
    fsa = 3.14159 * (p["edf_diameter"] / 2) ** 2 * (1 - 0.36 ** 2)            # swept area less the hub
    scoop = 2 * (p["intake_width"] - 7.0) * (p["intake_height"] - 9.0)
    amps = (V.A.MOTOR_W + FAN_W) / (6 * 3.5)
    sh = AstorBoost().generate(**variant(wing_sweep_deg=V.STOWED_DEG, tail_fan=0.0)).shape
    bb = (sh.val() if hasattr(sh, "val") else sh).BoundingBox()
    return {"mass": {k: mb[k] for k in ("empty_g", "total_g", "without_nets_g")}, "pack": mb["pack"]["name"],
            "added_over_astor_v_g": round(mb["total_g"] - V.budget()["total_g"], 0),
            "intake": {"fan_swept_area_cm2": round(fsa / 100, 1), "scoop_mouths_cm2": round(scoop / 100, 1), "ratio": round(scoop / fsa, 2)},
            "current": {"propeller_and_fan_A": round(amps), "pack_limit_A": mb["pack"]["max_A"], "ok": bool(amps <= mb["pack"]["max_A"])},
            "balance_by_mass": bal, "length_mm": round(p["edf_x"] + p["edf_length"]),
            "stowed_envelope_mm": {"length": round(bb.xlen), "width": round(bb.ylen), "height": round(bb.zlen)}}
