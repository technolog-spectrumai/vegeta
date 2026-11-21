"""ASTOR-BOOST — ASTOR-V with a ducted fan in a pod under the tail for a boost (a variant, 2026-11-21).

**What the fan is for.** Not top speed for its own sake (a stronger nose motor gives that for less mass): a second
source of thrust that works when the propeller does not. (1) A burst: propeller and fan together for the climb-out
and the overtake (estimate: +50 % acceleration at 20 m/s, top speed ~68 -> ~76 m/s). (2) The clear-eyed approach:
the nose propeller stopped and folded so the chase camera sees through no blades, the fan alone holding ~50 m/s for
the last seconds before the nets go.

**Where the fan is, and why.** In a pod on a short pylon under the rear fuselage. The first layout had it on the
tail's centre line with cheek scoops: the folded tail panels then swung straight through the duct and its jet, and no
in-plane fold avoids that without a much wider stowed aircraft. Under the tail:

- the tail fan folds exactly as ASTOR-V's, over the fuselage's end, with nothing in its way;
- **the fan breathes through its own mouth**: a straight (pitot) intake at the front of the pod, the whole lip in the
  free stream — no scoops, no bent duct, no losses to design around;
- the tail's centre line is free again, so **the nets stay in ASTOR-V's tail tube** (a belly bay would drop them in
  front of the pod's mouth). The tube's mouth is ~75 mm above the fan's jet.
- the fin stands under the pod.

The thrust line is ~75 mm below the fuselage axis: the fan's thrust pitches the nose up a little (0.5-1.2 N m; the
tail fan trims it). The pack grows to 6S 5000 mAh: propeller and fan together draw ~124 A, over the 4000 pack's
120 A. Everything else is ASTOR-V's (``astor_v``). Masses are HAND estimates; a layout, not an aerodynamic design
of the pod. Frame: x aft from the nose tip, y right, z up; mm.
"""
from __future__ import annotations

import dataclasses

import cadquery as cq

import arigo20 as a20
import astor_v as V
from vegeta.dedalus import Parameter

FAN_W = 1200.0
_DEFAULTS = {"belly_camera_x": 455.0, "fin_height": 70.0, "fin_root_chord": 150.0, "fin_taper": 0.5}
_NEW = [
    Parameter("edf_diameter", 70.0, "mm", min=40.0, max=100.0, description="the boost fan's rotor"),
    Parameter("edf_length", 150.0, "mm", min=80.0, description="the pod: lip to nozzle"),
    Parameter("edf_wall", 2.5, "mm", min=1.0),
    Parameter("nozzle_ratio", 0.88, "", min=0.6, max=1.0),
    Parameter("edf_x", 575.0, "mm", min=300.0, description="the pod's lip from the nose tip"),
    Parameter("pod_gap", 6.0, "mm", min=2.0, description="between the pod and the fuselage's underside (the pylon's height)"),
]

EXTRA_ITEMS = list(V.EXTRA_ITEMS) + [
    ("boost fan unit 70 mm, motor included", 175.0, "HAND (+-30); arigo_mass's value"),
    ("boost fan ESC 80 A", 60.0, "HAND (+-15)"),
    ("fan pod shell and pylon", 45.0, "HAND (+-20)"),
]
DEFAULT_PACK = "6S 5000 mAh LiPo"


class AstorBoost(V.AstorV):
    """ASTOR-BOOST: ASTOR-V with a ducted fan in a pod under the rear fuselage; the fin under the pod."""

    parameters = [dataclasses.replace(q, default=_DEFAULTS[q.name]) if q.name in _DEFAULTS else q for q in V.AstorV.parameters] + _NEW

    @staticmethod
    def pod(p) -> dict:
        Ro = p["edf_diameter"] / 2 + p["edf_wall"] + 2.0
        R = p["fuselage_diameter"] / 2
        return {"Ro": Ro, "z": -(R + p["pod_gap"] + Ro), "x0": p["edf_x"], "x1": p["edf_x"] + p["edf_length"]}

    def _fan(self, p):
        d = self.pod(p)
        nac = a20.Arigo20._nacelle(self, p).translate((d["x0"], 0, d["z"]))
        R = p["fuselage_diameter"] / 2
        pyl = cq.Workplane("XY").box(0.6 * p["edf_length"], 12.0, p["pod_gap"] + 14.0).translate(
            (d["x0"] + 0.45 * p["edf_length"], 0, -R - p["pod_gap"] / 2 + 2.0))
        return nac.union(pyl)

    def _fin(self, p):
        d = self.pod(p)
        return self._surface(d["x1"] - 8.0 - p["fin_root_chord"], p["fin_root_chord"], p["fin_taper"], p["fin_height"], p["fin_sweep_deg"],
                             p["tail_thickness"], "z", z0=-(d["z"]) + d["Ro"] - 4.0).mirror("XY")

    def parts(self, p) -> dict:
        out = super().parts(p)
        out["fan"] = self._fan(p)
        return out


def variant(**over) -> dict:
    q = {**V._OVERRIDE, **_DEFAULTS}
    q["tail_le_x"] = q["fuselage_length"] - 10.0 - q["htail_root_chord"]
    return {**q, **over}


def resolve(q=None) -> dict:
    return AstorBoost().resolve(**dict(q or variant()))


def budget(p=None, pack=None) -> dict:
    return V.budget(p or resolve(), pack, extra=EXTRA_ITEMS, default_pack=DEFAULT_PACK)


def _bbox(shape):
    return (shape.val() if hasattr(shape, "val") else shape).BoundingBox()


def checks() -> dict:
    p = resolve()
    d = AstorBoost.pod(p)
    mb = budget(p)
    bal = V.balance(p, mb=mb, stations={"boost fan unit": d["x0"] + 0.5 * p["edf_length"], "boost fan ESC": 520.0, "fan pod shell": d["x0"] + 0.5 * p["edf_length"]})
    A = AstorBoost()
    ps = A.resolve(**variant(wing_sweep_deg=V.STOWED_DEG, tail_fan=0.0, prop_fold_deg=82.0))
    tail, fan = A._tail_panels(ps), A._fan(ps)
    overlap = tail.intersect(fan)
    vol = sum(s.Volume() for s in overlap.solids().vals()) if overlap.solids().vals() else 0.0
    bt, bf = _bbox(tail), _bbox(fan)
    amps = (V.A.MOTOR_W + FAN_W) / (6 * 3.5)
    bb = _bbox(A.generate(**variant(wing_sweep_deg=V.STOWED_DEG, tail_fan=0.0, prop_fold_deg=82.0)).shape)
    return {"mass": {k: mb[k] for k in ("empty_g", "total_g", "without_nets_g")}, "pack": mb["pack"]["name"],
            "added_over_astor_v_g": round(mb["total_g"] - V.budget()["total_g"], 0),
            "intake": {"kind": "straight (pitot) at the pod's mouth", "fan_swept_area_cm2": round(3.14159 * (p["edf_diameter"] / 2) ** 2 * (1 - 0.36 ** 2) / 100, 1),
                       "scoops": "none"},
            "folded_tail_and_fan": {"overlap_volume_mm3": round(vol, 1), "clear": bool(vol < 1.0),
                                    "tail_panels_lowest_z_mm": round(bt.zmin, 1), "pod_highest_z_mm": round(bf.zmax, 1),
                                    "gap_mm": round(bt.zmin - (d["z"] + d["Ro"]), 1)},
            "thrust_line_below_axis_mm": round(-d["z"]), "net_tube_above_jet_axis_mm": round(-d["z"]),
            "current": {"propeller_and_fan_A": round(amps), "pack_limit_A": mb["pack"]["max_A"], "ok": bool(amps <= mb["pack"]["max_A"])},
            "balance_by_mass": bal,
            "stowed_with_propeller_folded_mm": {"length": round(bb.xlen), "width": round(bb.ylen), "height": round(bb.zlen)}}
