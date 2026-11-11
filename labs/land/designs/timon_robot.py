"""Timon — the fleet's servicing rover (``docs/timon.md``): a Pekari-family tracked machine sized for one job, keeping
the other robots working in the field — swapping their battery modules, changing the tools on their quick couplers,
and recovering (towing or winching) one that has failed.

**How it services.** Timon drives up nose-on to the side of the robot to service and docks ``DOCK_GAP`` from its
tracks. The main arm (Pumba's 6-DOF arm with its grab, front left) reaches across to the battery hatch on the
target's centre line; the second hand (Pekari-2M's light 5-DOF arm, front right) works the near half of the target
— the hatch lid, the connector, steadying a module — where the Perseus investigation (``docs/perseus_options.md``)
found a second hand pays: close in. Spare modules and the tool kit ride on the deck behind the arms; a tow hitch and
an electric winch at the rear recover a failed robot.

**How it is sized** (``REQUIREMENTS``, ``check``, ``sweep``): the hull length and width are not chosen but swept;
``design(length, width)`` builds a candidate (Pekari parameters, arms placed by rule, deck behind them) and
``check`` judges it against the requirements —

* reach: the main arm to the hatch of the widest fleet robot (Calydon / Pumba) docked ahead, the second hand to the
  near half of it;
* lift: the heaviest swappable module in the fleet (Pumba's 26 kg) at that reach ahead, and swung round to the
  deck, within the arm's rating and 75 % of tipping, **without stabilizers**;
* deck: two modules (0.25 x 0.40 m each) side by side and a tool rack (0.20 m), 0.70 x 0.40 m at least; the
  nominal load (two Pumba modules, the grab, the saw, a gripper) on it;
* recovery: tow a failed Pumba with its deck loaded (386 kg, tracks freewheeling: internal loss 0.08) on flat
  gravel with the drives, and hold it on a 15 deg slope on the winch with Timon's tracks braked;
* access: no wider than Pumba (it goes where Pumba goes), no taller than 0.8 m folded, the 20 deg climb loaded;

and the lightest candidate that passes everything is ``TIMON`` (``sweep`` reproduces the choice).

    import timon_robot as tr
    tr.check(tr.TIMON)            # requirement -> (value, needed, ok)
    robot = tr.timon()
"""
from __future__ import annotations

import math
from dataclasses import replace

from ogma import actuators as act
from ogma import tracks

import pekari_arms as pa
import pekari_family as pf
import pekari_rover_robot as prr

__all__ = ["G", "DOCK_GAP", "MODULE", "REQUIREMENTS", "design", "chassis", "check", "sweep", "CHASSIS", "CAD", "TIMON",
           "FLEET_MODULES", "timon", "timon_lab", "tow_force", "drawbar", "winch_hold"]

G = 9.81
DOCK_GAP = 0.15                   # Timon's track nose to the target's track edge when docked [m]
MODULE = {"kg": 26.0, "size": (0.25, 0.40, 0.22)}            # Pumba's 48 V 60 Ah module, the heaviest single one
FLEET_MODULES = {"Pekari-2M": 9.0, "Pumba": 26.0, "Calydon (one of two)": 22.0, "Perseus": 16.0}
HATCH_ABOVE_ROOF = 0.25           # the module's grip point above the target's roof as it comes out of the hatch [m]
NEAR_HALF = 0.30                  # how far into the target the second hand must reach [m]
TOW_F0 = 0.08                     # a failed tracked robot towed with its drives declutched: internal loss
WINCH_N = 4500.0                  # electric winch, line pull [N] (assumed: a 450 kg-class utility winch)
SPADE_N = 3000.0                  # the rear earth anchor's hold in field soil / gravel [N] (assumed, to be tested)
DECK_MIN = (0.70, 0.40)           # two modules side by side plus the tool rack [m]
REQUIREMENTS = {
    "main arm reaches the hatch": "the target's centre line, HATCH_ABOVE_ROOF above its roof",
    "second hand reaches the near half": "NEAR_HALF into the target, at its roof height + 0.15 m",
    "lifts a module at the hatch": "MODULE kg at that reach ahead: min(arm, 75 % tipping)",
    "swings a module to the deck": "MODULE kg sideways at 0.8 m: min(arm, 75 % tipping)",
    "deck fits 2 modules + tool rack": "DECK_MIN",
    "tows a failed Pumba on flat gravel": "drawbar pull > the loaded Pumba's resistance (drives declutched)",
    "holds it on the winch on a 15 deg slope": "braked tracks hold the winch's pull",
    "20 deg climb loaded": "peak drive and traction",
    "no wider than Pumba": "track outer width",
    "folded height <= 0.8 m": "",
}


# ----------------------------------------------------------------------------------------------- the candidates
def chassis(length_mm: float, width_mm: float) -> dict:
    """Pekari parameters [mm] for a Timon hull ``length_mm`` x ``width_mm`` (280 mm deep... 260 mm), 160 mm band
    tracks on six road wheels; the deck (the basket) behind the arm pedestals, as long as the roof allows."""
    L, W = float(length_mm), float(width_mm)
    half = L / 2
    spacing = round((half + 5.0 - 80.0) * 2 / 5, 1)              # six wheels: the outer ones 10 mm inside the axles
    deck_front = half - 300.0 - 130.0                            # behind the pedestals (and their 0.13 m radius)
    deck_len = deck_front - (-half + 40.0)
    return {
        "hull_length": L, "hull_width": W, "hull_height": 260.0, "hull_chamfer": 30.0, "ground_clearance": 150.0,
        "sensor_box": 140.0, "basket_length": deck_len, "basket_width": W - 220.0, "basket_height": 100.0,
        "basket_wall": 1.5, "basket_x": (deck_front + (-half + 40.0)) / 2, "frame_offset": 120.0,
        "frame_thickness": 8.0, "track_width": 160.0, "track_pitch": 72.0, "link_thickness": 14.0,
        "grouser_height": 12.0, "guide_height": 22.0, "guide_width": 26.0, "pin_diameter": 10.0, "sprocket_teeth": 13,
        "sprocket_x": -(half + 5.0), "sprocket_z": 270.0, "idler_diameter": 230.0, "idler_x": half + 5.0,
        "idler_z": 195.0, "road_wheel_diameter": 140.0, "road_wheels": 6, "road_wheel_spacing": spacing,
        "return_roller_diameter": 80.0, "return_roller_z": 300.0, "axle_diameter": 20.0, "sprocket_bore": 25.0,
        "pinion_module": 2.0, "pinion_teeth": 18, "pinion_width": 20.0, "pinion_bore": 10.0,
    }


_CAD_CACHE: dict = {}


def _cad(p: dict) -> dict:
    k = tuple(sorted(p.items()))
    if k not in _CAD_CACHE:
        _CAD_CACHE[k] = prr.cad_numbers(prr.design_params(p), recompute=True)
    return _CAD_CACHE[k]


PARTS_KG = {
    "battery 48 V 60 Ah LiFePO4 (2.9 kWh): the fleet's swappable module": 26.0,
    "computer, motor and arm controllers, radio mesh node, IMU, GNSS": 4.0,
    "sensors: 2 RGB-D (front, wrist), 3D lidar, rear camera for the hitch, docking fiducial reader": 4.5,
    "wiring, connectors, e-stop, switches, contactors, module charger": 6.0,
    "hull frame (welded Al), deck with module cradle and tool rack, tow hitch, winch 4.5 kN with 15 m line": 34.0,
    "track frames, bogies, axles, bearings 2x": 16.0,
}


def design(length_mm: float, width_mm: float, cad: dict | None = None, spade: bool = True) -> pf.Platform:
    """A Timon candidate: the chassis, the main arm front left and the second hand front right (0.30 m behind the
    hull's nose, inside the roof's edge), the rear earth anchor (``spade``: a leg pressed into the ground behind the
    tracks — the winch's anchor and a rear support), the parts, Pumba's drives and band track."""
    p = chassis(length_mm, width_mm)
    L, W = length_mm / 1000, width_mm / 1000
    x_arm = L / 2 - 0.30
    mounts = (pf.Mount("main", pa.PUMBA_MAIN_ARM, x=x_arm, y=W / 2 - 0.12),
              pf.Mount("aux", pa.PEKARI_ARM, x=x_arm, y=-(W / 2 - 0.08)))
    return pf.Platform(
        name="Timon", chassis=p, cad=cad if cad is not None else _cad(p), mounts=mounts,
        stabilizers=((pf.Stabilizer("spade", x=-(L / 2 + 0.12), y=0.0, pad_radius=0.10, drive="stick cylinder 8 kN",
                                    stroke=0.22, mass_kg=11.0),) if spade else ()),
        parts_kg=dict(PARTS_KG), drive="pumba track drive 2.5 kW, 48 V", materials=dict(prr.MATERIALS),
        shell_t_mm=2.0, wheel_fill=0.25, band_track_kg_m=4.5, bus_V=48.0, battery_Wh=2880.0, electronics_W=90.0,
        payload_kg=2 * MODULE["kg"] + 5.0 + pa.TOOLS["chainsaw 2 kW"].mass_kg + 0.55, payload_range_kg=(0.0, 100.0),
        v_design=8 / 3.6, sensors=("rgbd_front", "rgbd_wrist", "lidar_3d", "rear_camera", "gnss", "imu"),
        colors={"arm": (0.10, 0.45, 0.85, 1.0), "stabilizer": (0.10, 0.45, 0.85, 1.0)},
        bom_extra={
            "chassis: welded Al frame, Al shell, deck, service hatches": (1, 3000.0, 7000.0),
            "tracks: 160 mm rubber band tracks, wheels, sprockets, idlers, tensioners, 2 sides": (1, 1800.0, 4500.0),
            "battery: the fleet's 48 V 60 Ah module with BMS, charger": (1, 1600.0, 3000.0),
            "motor and arm controllers, contactors, DC/DC, module charger on the deck": (1, 1500.0, 3500.0),
            "computer (embedded GPU module class), IMU, RTK GNSS": (1, 1500.0, 3500.0),
            "mesh radio node and safety (e-stop) radio receiver": (1, 900.0, 2700.0),
            "RGB-D cameras (front, wrist), rear hitch camera": (3, 300.0, 500.0),
            "3D lidar": (1, 2000.0, 6000.0),
            "two-jaw grab with quick coupler (main arm)": (1, 1000.0, 3000.0),
            "parallel gripper with quick-change flange (second hand)": (1, 400.0, 1500.0),
            "module cradle (2 slots) and tool rack on the deck": (1, 300.0, 800.0),
            "tow hitch and 4.5 kN electric winch with 15 m line": (1, 600.0, 1500.0),
            "rear earth anchor: leg, hinge, spade (fabrication)": (1, 300.0, 700.0),
            "wiring, connectors, e-stop, lights, covers": (1, 1000.0, 2500.0),
        },
        assumptions={
            "job": "field service of the fleet: battery module swaps, tool changes on quick couplers, recovery",
            "docking": f"nose-on to the target's side, {DOCK_GAP} m from its tracks, guided by fiducials on the target",
            "arms": "Pumba's 6-DOF arm with the grab (main) and Pekari-2M's 5-DOF arm (second hand): the family's "
                    "modules, no new arm",
            "battery": "Timon runs on the fleet's 60 Ah module it carries spares of",
            "track": "180-class band track as Pumba's, 160 mm at 4.5 kg/m (assumed)",
            "winch": f"{WINCH_N / 1000:g} kN electric winch, 15 m line, in the frame row with the hitch",
            "spade": f"rear earth anchor on an 8 kN cylinder, 11 kg; its hold in soil {SPADE_N / 1000:g} kN is assumed "
                     "(braked tracks alone hold a failed Pumba on 15 deg only if Timon weighed ~440 kg)",
        })


# ----------------------------------------------------------------------------------------------- the requirements
def _target():
    import pumba_robot as pu
    import calydon_robot as cr

    return {"Pumba": pu.PUMBA, "Calydon": cr.CALYDON}


def tow_force(pl_target: pf.Platform, grade_deg: float = 0.0) -> float:
    """The pull [N] to tow ``pl_target`` loaded with its drives declutched (``TOW_F0``) on gravel at ``grade_deg``."""
    g = pf.geometry(pl_target)
    W = pf.total_mass(pl_target, pl_target.payload_kg) * G
    return tracks.resistance(W, b=g["b"], L=g["L"], soil="gravel", grade_deg=grade_deg, v=0.3, f0=TOW_F0)["total"]


def drawbar(pl: pf.Platform, grade_deg: float = 0.0) -> float:
    """Timon's drawbar pull [N] at ``grade_deg``: the smaller of its drives' rated force and traction, less its own
    resistance (loaded)."""
    g = pf.geometry(pl)
    W = pf.total_mass(pl, pl.payload_kg) * G
    d = act.get(pl.drive)
    F = min(2 * d.rated_Nm / g["r_sprocket"], tracks.tractive_limit(W, b=g["b"], L=g["L"], soil="gravel", grade_deg=grade_deg))
    return F - tracks.resistance(W, b=g["b"], L=g["L"], soil="gravel", grade_deg=grade_deg, v=0.3, f0=pl.internal_f0)["total"]


def winch_hold(pl: pf.Platform, grade_deg: float = 15.0, mu: float = 0.6) -> float:
    """What Timon holds [N] against a winch line downhill at ``grade_deg``: its braked tracks (friction
    μ W cos − W sin) plus the rear earth anchor (``SPADE_N``) when it has one."""
    W = pf.total_mass(pl, pl.payload_kg) * G
    t = math.radians(grade_deg)
    spade = SPADE_N if any(s.name == "spade" for s in pl.stabilizers) else 0.0
    return mu * W * math.cos(t) - W * math.sin(t) + spade


def _ik_any(arm, target, shoulder):
    """IK with the tool pointing down, 45 deg down, level or 30 deg up — whichever reaches."""
    for e in (-math.pi / 2, -math.pi / 4, 0.0, math.pi / 6):
        q = pa.ik(arm, target, e, shoulder)
        if q is not None:
            return q
    return None


def _hatch_reach(pl: pf.Platform, target: pf.Platform, mount: str, into: float | None = None) -> tuple:
    """The horizontal reach [m] from ``mount``'s shoulder to the service point on ``target`` docked ahead, and its
    height: the target's centre line (``into`` None) or ``into`` m inside its near edge."""
    g, gt = pf.geometry(pl), pf.geometry(target)
    x_edge = g["x_front"] + DOCK_GAP                              # the target's near track edge
    x = x_edge + (gt["half_width"] if into is None else into)
    sh = pf.shoulder(pl, mount)
    z = pf.roof_z(target) + (HATCH_ABOVE_ROOF if into is None else 0.15)
    return math.hypot(x - sh[0], 0.0), x, z


def check(pl: pf.Platform) -> dict:
    """requirement -> (value, needed, ok) for a Timon candidate (see ``REQUIREMENTS``)."""
    import pumba_robot as pu

    out = {}
    worst_main, worst_aux = 0.0, 0.0
    reach_ok = aux_ok = True
    for name, t in _target().items():
        r, x, z = _hatch_reach(pl, t, "main")
        sh = pf.shoulder(pl, "main")
        q = _ik_any(pl.mount("main").arm, (x, sh[1], z), sh)
        reach_ok &= q is not None
        worst_main = max(worst_main, r)
        ra, xa, za = _hatch_reach(pl, t, "aux", NEAR_HALF)
        sha = pf.shoulder(pl, "aux")
        qa = _ik_any(pl.mount("aux").arm, (xa, sha[1], za), sha)
        aux_ok &= qa is not None
        worst_aux = max(worst_aux, ra)
    out["main arm reaches the hatch"] = (worst_main, pa.reach(pl.mount("main").arm), reach_ok)
    out["second hand reaches the near half"] = (worst_aux, pa.reach(pl.mount("aux").arm), aux_ok)
    cap = pf.lift_capacity(pl, "main", worst_main, 0.0)
    out["lifts a module at the hatch"] = (cap["capacity_kg"], MODULE["kg"], cap["capacity_kg"] >= MODULE["kg"])
    side = pf.lift_capacity(pl, "main", 0.8, math.pi / 2)
    out["swings a module to the deck"] = (side["capacity_kg"], MODULE["kg"], side["capacity_kg"] >= MODULE["kg"])
    p = pl.chassis
    deck = (p["basket_length"] / 1000, p["basket_width"] / 1000)
    out["deck fits 2 modules + tool rack"] = (deck, DECK_MIN, deck[0] >= DECK_MIN[0] and deck[1] >= DECK_MIN[1])
    need = tow_force(pu.PUMBA)
    have = drawbar(pl)
    out["tows a failed Pumba on flat gravel"] = (have, need, have >= need)
    need15 = min(tow_force(pu.PUMBA, 15.0), WINCH_N)
    hold = winch_hold(pl, 15.0)
    out["holds it on the winch on a 15 deg slope"] = (hold, need15, hold >= need15)
    dc = pf.drive_check(pl, grade_deg=20.0, v_ms=0.3)
    out["20 deg climb loaded"] = (dc["drive_stall_N"], dc["resistance_N"], dc["ok_peak"] and dc["ok_traction"])
    w = pf.geometry(pl)["half_width"] * 2
    wp = pf.geometry(pu.PUMBA)["half_width"] * 2
    out["no wider than Pumba"] = (w, wp, w <= wp + 1e-9)
    h = pf.capabilities(pl)["height_folded_m"]
    out["folded height <= 0.8 m"] = (h, 0.8, h <= 0.8)
    return out


def sweep(lengths=(1100.0, 1150.0, 1200.0, 1250.0, 1300.0), widths=(560.0, 600.0, 620.0, 640.0, 680.0)) -> list:
    """Every candidate: (length, width, mass empty, passes all, the failed requirements), lightest first."""
    rows = []
    for L in lengths:
        for W in widths:
            pl = design(L, W)
            c = check(pl)
            fails = [k for k, v in c.items() if not v[2]]
            rows.append((L, W, pf.total_mass(pl), not fails, fails))
    return sorted(rows, key=lambda r: (not r[3], r[2]))


# ----------------------------------------------------------------------------------------------- the chosen Timon
#: The lightest candidate of ``sweep`` that meets every requirement: 1200 x 620 mm (the deck sets the length and
#: the width; stability does not bind). ``check(design(1200, 620))`` holds it.
CHASSIS = chassis(1200.0, 620.0)

#: CAD measurements of ``CHASSIS`` (``pekari_rover.PekariRover``; mm², mm³, mm).
CAD = {
    "hull_surface_area": 2436217.5, "hull_com": (5.29, 0.0, 281.75), "link_volume": 207758.2,
    "sprocket_volume": 828622.8, "idler_volume": 6082123.4, "road_wheel_volume": 2123716.6, "pinion_volume": 18188.2,
    "basket_volume": 817028.5, "basket_com": (0.0, 0.0, 26.66),
}

TIMON = replace(design(1200.0, 620.0, cad=CAD), roles=("service", "tow"))


def timon(**kw):
    """Timon as a Chiron ``Robot`` (``pekari_family.robot`` keywords)."""
    return pf.robot(TIMON, **kw)


def timon_lab(terrain=None, robot=None, **kwargs):
    return pf.lab(TIMON, terrain, robot or timon(), **kwargs)
