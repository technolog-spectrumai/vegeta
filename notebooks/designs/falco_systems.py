"""FALCO's systems: the components, the pack, the masses, the drive.

NISUS+'s module does the arithmetic (``nisus_plus_systems``: the cells, the packs, the mass table through its
``design``/``items``/``components``/``battery`` seams, the loads, the mission energy); this one holds what FALCO
changes:

* the propulsion — a **T-Motor AT5220-A KV220** (52 mm stator, 450 g; its published full-throttle point on APC 18x8 at
  12S fits the motor model) turning an **APC 20x15E** in the nose (the frame study's 20-inch disc; the pitch from
  ``falco_flight.propeller_trade``: the 20x13 cruises near its zero-thrust advance ratio), on an **8S3P** pack of the same 24 Molicel P45B
  cells as NISUS+'s 6S4P (the same box, the same energy: the series count is the only change). ``drive_options``
  shows why: the KV220 on 6S reaches ~3900 rpm and ~500 W shaft (no climb margin), the KV380 on 6S runs the 20x13 at
  its 100 A limit (a copper heater); the KV220 on 8S gives the frame study's 20-inch point (~4800 rpm, ~1.1 kW shaft)
  at ~50 A;
* the components — the motor, the propeller, the spinner, the propeller's **position sensor** (a Hall sensor on the
  firewall and a magnet in the hub: the ESC's active brake stops the blades, the sensor says where, the parking
  routine nudges them horizontal before the flare), an 8S-rated ESC, BEC and Orin buck, the elevator and rudder servos
  at the tail's end fitting (short linkages; 44 g at the tail trims the motor in the nose);
* the structure items — the tractor fuselage shell, the firewall, the single roll-wrapped tail tube in its keel
  socket, the end fitting, the stabiliser's 12/10 spar, no booms and no boom fittings.

Every sourced figure carries its page and date in ``data/falco_component_sources.md``; **availability is not
confirmed** (search-index snippets only); what was not researched is marked **assumed**.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import falco
import nisus_plus_systems as fs
import frame_study as fst
from nisus_systems import Component, MATERIALS
from nisus_plus_systems import (G, atmosphere, density_altitude, atmosphere_table, LiIonCell, CELLS, CHARGE_MIN_C, LiIonPack,  # noqa: F401
                                SNIPPET, ZERO, JETSON, JETSON_INSTALLATION_W, JETSON_INSTALLATION_RANGE_W, ALLOWANCE_FRACTION,
                                PHASES, phase_power, electrical_loads, servo_check, requirements, MissionPlan, mission_energy, level_power,
                                max_roc, climb_profile, return_energy, NisusPlusDrive, V_GRID, drive_table)

try:
    from vegeta.boreas import RHO0
except Exception:                                                        # pragma: no cover
    RHO0 = 1.225

DATA = Path(__file__).resolve().parent / "data"
ACCESS = "2026-10-10"

# ================================================================================================= CAD numbers
#: CAD volumes [mm³] of the default FALCO parts (``falco.Falco().generate(part=...).measure()['volume']``);
#: ``cad_numbers(recompute=True)`` rebuilds them (about 30 s).
CAD = {"aircraft": 23021511.0, "wing": 12783897.0, "spar": 174536.0, "rear_spar": 15215.0, "spar_joiner": 18871.0, "pod": 401111.0,
       "nose": 34478.0, "tail_tube": 79262.0, "tail_socket": 145537.0, "tail_fitting": 77055.0, "tail": 1898937.0, "motor_mount": 46435.0,
       "tray": 35216.0, "battery_tray": 74020.0, "skid": 116960.0, "flap": 336993.0, "aileron": 277388.0}


def cad_numbers(p=None, *, recompute: bool = False) -> dict:
    return fs.cad_numbers(p, recompute=recompute, design=falco.Falco(), table=CAD)


# ================================================================================================= the pack
#: FALCO's pack: NISUS+'s 24 P45B cells in series-parallel 8S3P — the same 264 x 76 x 47 mm box and 388 Wh, 28.8 V
#: nominal (33.6 V full): the KV220 motor needs the volts (``drive_options``). The 6S4P stays as the comparison.
PACKS = {"8s3p-p45b": LiIonPack("8s3p-p45b", CELLS["P45B"], 8, 3), **fs.PACKS}
DEFAULT_PACK = "8s3p-p45b"
BATTERY_V = PACKS[DEFAULT_PACK].voltage


def pack(key: str = DEFAULT_PACK) -> LiIonPack:
    return PACKS[key]


def pack_table(T_C=(25.0, 5.0, -10.0)) -> pd.DataFrame:
    rows = {}
    for k, b in PACKS.items():
        r = {"pack": b.name, "cells": b.series * b.parallel, "V nom": b.voltage, "Ah": b.capacity_ah, "Wh nominal": b.energy_wh,
             "mass [g]": b.mass_g, "Wh/kg": b.energy_wh / b.mass_g * 1000, "max A (cells)": b.max_current_a, "max charge A": b.max_charge_a,
             "R at 25 °C [mΩ]": 1000 * b.resistance(25.0), "size [mm]": "x".join(f"{v:.0f}" for v in b.size_mm)}
        for T in T_C:
            r[f"usable Wh at {T:g} °C, 20 A"] = b.usable_wh(T, 20.0)
            r[f"sag at 50 A, {T:g} °C [V]"] = 50.0 * b.resistance(T)
        rows[k] = r
    return pd.DataFrame(rows).T


def battery_fits(b: LiIonPack, p=None) -> dict:
    return fs.battery_fits(b, p, design=falco.Falco())


# ================================================================================================= components
# positions: FALCO's frame (x aft from the wing root LE, z up from the pod top; the fuselage's axis at z = -65), mm
_L = falco.Falco.layout(falco.resolve())
_X_MOTOR = 0.5 * (_L["x_motor0"] + _L["x_motor1"])
_Z_AXIS = _L["pod_axis_z"]
_X_FW = _L["x_firewall"]
COMPONENTS = [
    # --- propulsion
    Component("motor T-Motor AT5220-A KV220 (long shaft)", "propulsion/ESC/wiring", 450.0, ZERO, _X_MOTOR, _Z_AXIS, supply="battery", supply_v=BATTERY_V, price_eur=160.0,
              url="https://store.tmotor.com/product/at5220-a-fixed-wing-motor.html",
              source=f"450 g with cable, Φ60 x 112.3 mm, 36 mΩ, 24N28P, 6-12S, idle 1.8 A at 10 V, 70 A / 2700 W for 180 s (T-Motor page and the "
                     f"ligpower datasheet via robotshop PDF); APC 18x8 at 100 %: 43.33 V, 48.86 A, 2116.8 W, 8141 rpm, 2.037 N m, 8185 gf, 3.87 g/W "
                     f"(the German page of the same table); {SNIPPET}",
              note="the published full-throttle point fits the motor model (falco_systems.motor_model); the 20x13 point is the model's; price assumed"),
    Component("propeller APC 20x15E thin electric (fixed: it brakes, regenerates and parks)", "propulsion/ESC/wiring", 118.0, ZERO, _L["prop_x"], _Z_AXIS, price_eur=25.0,
              url="https://motionrc.com/collections/2-blade-propellers/products/apc-20x15-thin-electric-propeller-lp20015e",
              source=f"117.93 g (Motion RC, LP20015E); the 20x13E is 117.08 g (the reverse LP20013EP listing); 118 g taken; 6.35 mm bore; {SNIPPET}",
              note="the 15-inch pitch: the propeller trade (falco_flight.propeller_trade) — the 20x13 of the frame study cruises near its zero-thrust advance "
                   "ratio (profile drag), the 20x15 cruises at 73 % and climbs 15 % faster; two blades: parked horizontal they lie along the wing"),
    Component("spinner 44 mm (aluminium backplate, printed cone) with the hub's parking magnet", "propulsion/ESC/wiring", 30.0, ZERO, _L["x_nose"] + 16.0, _Z_AXIS,
              kind="assumed", price_eur=20.0, source="estimate: a 44 mm two-blade spinner with an aluminium backplate and the 8 mm shaft adapter, 25 g, plus a 5 g magnet carrier"),
    Component("propeller position sensor (Hall sensor on the firewall, magnet in the hub)", "propulsion/ESC/wiring", 5.0, ZERO, _X_FW - 5.0, _Z_AXIS + 28.0, supply="5V", supply_v=5.3,
              p_idle_w=0.05, p_typ_w=0.05, p_peak_w=0.05, kind="assumed", price_eur=5.0,
              source="assumed: a latching Hall sensor and a 5 mm magnet on the spinner's backplate: one pulse per turn says where the blades are; "
                     "ArduPilot has no propeller-stowing function (forum thread 'using magnetic position sensor on motor to stow the propeller', 2022: "
                     "a position-sensing ESC or a VESC in HFI mode are the suggestions), so the parking routine is a companion-computer script on "
                     "the Orin: brake to a stop with the ESC's active brake, read the sensor, nudge in reverse or forward to the horizontal mark",
              note="to be proven on the bench: the brake's stopping angle and the nudge's repeatability (todo.md)"),
    Component("ESC 80 A 3-8S, AM32 firmware (active freewheeling, brake, bidirectional)", "propulsion/ESC/wiring", 50.0, ZERO, _X_FW + 30.0, _Z_AXIS + 25.0, supply="battery", supply_v=BATTERY_V,
              p_idle_w=0.5, p_typ_w=0.5, p_peak_w=0.5, price_eur=60.0, kind="assumed",
              url="https://www.rotorama.com/product/pilotix-esc-4in1-pilotix-80a-am32-8s-cz",
              source=f"80 A continuous / 100 A peak, 3-8S AM32 class (as NISUS+'s: a single-motor 80 A AM32 ESC assumed, 50 g with wires and cap); AM32's "
                     f"'damped light' mode does regenerative braking (Hobbywing 65A AM32 listing); the brake-on-stop and the bidirectional mode are "
                     f"AM32 settings; {SNIPPET}",
              note="on 8S the full pack is 33.6 V: the 8S rating is now used, the regen spikes above it are the ESC's limit (to measure); quiescent 0.5 W assumed"),
    Component("wiring 10 AWG, XT90, bullets, leads, capacitor", "propulsion/ESC/wiring", 65.0, ZERO, -200.0, _Z_AXIS - 20.0, kind="assumed",
              source="estimate: 10 AWG battery lead 2 x 350 mm (the pack under the wing, the ESC behind the firewall), XT90 pair, 3 x 12 AWG motor leads "
                     "100 mm (the motor is next door), bullets, a 1000 µF low-ESR bank rated 50 V"),
    Component("battery straps, velcro, foam, insulating sleeve", "battery/retention", 30.0, ZERO, -10.0, -110.0, kind="assumed",
              source="estimate: two 25 mm straps with buckles, velcro pads, 5 mm closed-cell foam sleeve (keeps the pack warm at altitude)"),
    # --- actuation: six servos of the 20-25 g metal-gear class; the tail's two at the tube's end fitting
    *[Component(f"servo 22 g metal gear ({name})", "servos/linkages", 22.0, ZERO, x, z, y_mm=y, supply="servo", supply_v=7.4,
                p_idle_w=0.2, p_typ_w=1.0, p_peak_w=12.0, price_eur=25.0, kind="assumed",
                source="assumed: a 20-25 g metal-gear wing servo, 6-8 kg·cm at 7.4 V, 0.10 s/60° (KST / Emax class); not researched",
                note="typical 1.0 W = 20 % of the time moving; peak = stall 1.6 A at 7.4 V")
      for name, x, y, z in (("flap L", 230.0, -405.0, 5.0), ("flap R", 230.0, 405.0, 5.0), ("aileron L", 150.0, -880.0, 5.0), ("aileron R", 150.0, 880.0, 5.0),
                            ("elevator, at the tail fitting", _L["tail_le"] + 20.0, -22.0, _Z_AXIS), ("rudder, at the tail fitting", _L["tail_le"] + 20.0, 22.0, _Z_AXIS))],
    Component("horns, clevises, pushrods, hinges", "servos/linkages", 30.0, ZERO, 350.0, -15.0, kind="assumed",
              source="estimate: 6 horns, 2 mm carbon pushrods (short: the tail's servos sit on the end fitting), hinge tape"),
    Component("servo lead 3 x 0.25 mm² through the tail tube (2 x 900 mm) and the wing", "servos/linkages", 25.0, ZERO, 450.0, -50.0, kind="assumed",
              source="estimate: two servo leads inside the 28 mm tube to the end fitting, the wing's four through the saddle; connectors"),
    Component("servo BEC 8 A, 7.4 V (8S input)", "servos/linkages", 20.0, ZERO, 60.0, -80.0, kind="assumed",
              source="estimate: a separate 8 A switching BEC rated to 36 V for six servos, 85 %"),
    # --- flight control
    Component("flight controller Matek H743-WING (ArduPilot)", "flight controller/receiver/GNSS/wiring", 28.0, ZERO, 35.0, -55.0,
              supply="battery", supply_v=BATTERY_V, p_idle_w=1.5, p_typ_w=1.8, p_peak_w=2.5, price_eur=80.0, kind="assumed",
              url="https://www.mateksys.com/?portfolio=h743-wing-v3",
              source="assumed: the H743-WING class (6-36 V input: 8S fits, 5 V/9 V BECs, current sensor, ArduPilot Plane with TECS and terrain following); not researched"),
    Component("GNSS + compass Matek M10Q-5883 (on the tail cone)", "flight controller/receiver/GNSS/wiring", 8.0, ZERO, 230.0, -8.0, supply="5V", supply_v=5.3,
              p_idle_w=0.25, p_typ_w=0.25, p_peak_w=0.35, price_eur=25.0, url="https://www.unmannedtechshop.co.uk/products/matek-m10q-5883-gps-module",
              source="as NISUS's (nisus_component_sources.md); on the tail cone's top, away from the motor's field and the pack's leads"),
    Component("airspeed sensor (pitot-static, I2C differential pressure) in the chin, outside the propeller's hub", "flight controller/receiver/GNSS/wiring", 12.0, ZERO, _X_FW + 40.0, -125.0,
              supply="5V", supply_v=5.3, p_idle_w=0.05, p_typ_w=0.05, p_peak_w=0.05, price_eur=40.0, kind="assumed",
              source="assumed: an MS4525-class sensor; the pitot on a 60 mm stalk under the chin, ahead of the wing and clear of the slipstream's core — the "
                     "propeller's wash raises its reading at power (to calibrate in the flight tests)"),
    Component("lidar rangefinder (40 m) for the landing, in the chin", "flight controller/receiver/GNSS/wiring", 50.0, ZERO, _X_FW + 60.0, -122.0, supply="5V", supply_v=5.3,
              p_idle_w=1.0, p_typ_w=1.0, p_peak_w=1.0, price_eur=80.0, kind="assumed",
              source="assumed: a Benewake TF02-Pro-class lidar (40 m, ~1 W) looking down through a chin window: the height over the meadow in the flare"),
    Component("RC receiver ELRS 900 MHz", "flight controller/receiver/GNSS/wiring", 5.0, ZERO, 50.0, -40.0, supply="5V", supply_v=5.3,
              p_idle_w=0.5, p_typ_w=0.5, p_peak_w=0.8, price_eur=25.0, kind="assumed", source="assumed: 900 MHz diffracts over ridges better than 2.4 GHz"),
    Component("telemetry RFD900x (900 MHz, 1 W)", "flight controller/receiver/GNSS/wiring", 15.0, ZERO, 60.0, -40.0, supply="5V", supply_v=5.3,
              p_idle_w=0.4, p_typ_w=2.0, p_peak_w=5.0, price_eur=120.0, kind="assumed", source="assumed: RFD900x class, 14.5 g, up to 1 W output; typical 2 W at a 30 % transmit duty"),
    Component("signal wiring, connectors, antenna mounts", "flight controller/receiver/GNSS/wiring", 25.0, ZERO, 0.0, -40.0, kind="assumed", source="estimate"),
    # --- the computer (the Orin Nano Super installation: NISUS's JETSON_OPTIONS entry) in the bay behind the ESC
    Component("Jetson Orin Nano Super: module + carrier + heatsink/fan", "computer", 230.0, ZERO, _X_FW + 115.0, -60.0, supply="jetson", supply_v=12.0,
              p_idle_w=7.0, p_typ_w=15.0, p_peak_w=25.0, price_eur=None, url="https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit",
              source=f"NISUS's JETSON_OPTIONS['orin_nano_super']: 7/15/25 W modes, 9-19 V input, JetPack 6.2; mass not published "
                     f"(the kit with its heatsink ~250 g, 230 g taken without the case); {SNIPPET}",
              note="the installation's battery-side power is JETSON_INSTALLATION_W (15 W mode + camera + buck); it also runs the propeller's parking script"),
    Component("buck 8S → 12 V / 5 A for the Orin (36 V input)", "computer", 32.0, ZERO, _X_FW + 180.0, -100.0, kind="assumed",
              source="assumed: the Orin kit takes 9-19 V and a full 8S pack is 33.6 V: a 12 V / 5 A synchronous buck rated to 36 V, 92 %"),
    Component("vision camera IMX477 (12 MP, CSI) + 6 mm lens + cable, in the chin", "computer", 35.0, ZERO, _X_FW + 20.0, -112.0, supply="jetson", supply_v=5.0,
              p_idle_w=0.4, p_typ_w=0.7, p_peak_w=0.8, price_eur=60.0, kind="assumed",
              source="assumed: Raspberry Pi HQ camera class, looking forward-down under the propeller's hub: the blades cross the top of the frame (computer vision "
                     "masks them by the rpm-synchronous pattern; accepted)"),
    Component("data link (LTE stick + antenna)", "computer", 25.0, ZERO, 120.0, -10.0, supply="jetson", supply_v=5.0,
              p_idle_w=1.0, p_typ_w=2.0, p_peak_w=3.0, price_eur=40.0, kind="assumed", source="as NISUS's (the computer's results and a low-rate video)"),
    Component("storage (NVMe 256 GB) and the onboard DEM", "computer", 10.0, ZERO, _X_FW + 115.0, -60.0, kind="assumed",
              source="assumed: the terrain model of the mission area (a 30 m DEM clip) lives on the Orin; ArduPilot's terrain database on its SD card"),
    Component("Orin mounting frame (printed), cooling duct from the cowl's intake, vibration dampers", "joints/mounts/adhesive", 25.0, ZERO, _X_FW + 115.0, -95.0, kind="assumed",
              source="estimate: PETG rails behind the firewall, the cooling air from the cowl's intake over the motor and the ESC to the fan"),
]


def components() -> list:
    return list(COMPONENTS)


def component_table() -> pd.DataFrame:
    rows = [{"item": c.key, "group": c.group, "mass [g]": c.mass_g, "x [mm]": c.x("Zero"), "z [mm]": c.z_mm, "supply": c.supply, "V": c.supply_v,
             "idle [W]": c.p_idle_w, "typ [W]": c.p_typ_w, "peak [W]": c.p_peak_w, "price [EUR]": c.price_eur, "kind": c.kind, "availability": c.availability,
             "URL": c.url, "source": c.source} for c in COMPONENTS]
    return pd.DataFrame(rows).set_index("item")


# ================================================================================================= mass table
STAB_SPAR = dict(od=12.0, id_=10.0)
FIN_ROD = dict(od=10.0, id_=8.0, length=300.0)
PRINT_FILL = dict(fs.PRINT_FILL, tail_socket=0.40, tail_fitting=0.60)


def structure_items(p=None, cad: dict | None = None) -> list:
    """The airframe's parts (item, group, mass_g, x, y, z, basis): NISUS+'s wing recipe unchanged; the tractor fuselage
    as a LW-PLA shell with the printed nose cowl, the PETG firewall, the keel's socket; one roll-wrapped tail tube, its
    end fitting, the stabiliser's 12/10 spar, the fin's rod; no booms, no boom fittings."""
    p = falco.resolve(p)
    cad = cad or CAD
    L = falco.Falco.layout(p)
    m = MATERIALS
    x_wing = L["x_mac_le"] + 0.42 * L["mac"]
    prof = falco.Falco.pod_profile(p)
    per = math.pi * (prof[:, 1] + prof[:, 2])
    x_pod = float(np.trapezoid(prof[:, 0] * per, prof[:, 0]) / np.trapezoid(per, prof[:, 0]))
    w = falco.wetted_areas(p)
    foam = cad["wing"] * 1e-3 * m["XPS foam 30"]["rho"] * (1 - fs.ns.FOAM_LIGHTENING)
    balsa = 2 * 0.30 * 0.5 * (p["root_chord"] + p["tip_chord"]) * p["span"] * 2.0 * 1e-3 * m["balsa"]["rho"] * 1.05 + 25.0
    carbon = m["carbon tube"]["rho"]
    rods = fs.TIP_RODS["count"] * math.pi / 4 * (fs.TIP_RODS["od"] ** 2 - fs.TIP_RODS["id_"] ** 2) * fs.TIP_RODS["length"] * 1e-3 * carbon
    film_wing = m["covering film"]["areal_g_m2"] * w["wing"]
    x_tail = L["tail_le"] + 0.4 * p["tail_chord"]
    tail_foam = cad["tail"] * 1e-3 * m["XPS foam 30"]["rho"]
    stab_spar = math.pi / 4 * (STAB_SPAR["od"] ** 2 - STAB_SPAR["id_"] ** 2) * p["tail_span"] * 1e-3 * carbon
    fin_rod = math.pi / 4 * (FIN_ROD["od"] ** 2 - FIN_ROD["id_"] ** 2) * FIN_ROD["length"] * 1e-3 * carbon
    film_tail = m["covering film"]["areal_g_m2"] * (w["tail"] + w["fins"])
    petg = m["PETG printed"]["rho"]
    lw = m["LW-PLA printed"]["rho"]
    x_rear = p["rear_spar_x_frac"] * p["root_chord"]
    x_tube = 0.5 * (L["x_tube0"] + p["tail_x_end"])
    zt = p["boom_z"]
    items = [
        ("wing: XPS core (25 % lightened)", "wing/reinforcement/covering", foam, x_wing, 0.0, 8.0, "CAD volume x 30 kg/m³ x 0.75"),
        ("wing: carbon spar tube 20/17 x 2000 (three pieces)", "wing/reinforcement/covering", cad["spar"] * 1e-3 * carbon, L["spar_x_root"], 0.0, 6.0, "CAD volume x 1.55 g/cm³"),
        ("wing: spar joiners 16.8/13.5 x 240 (carbon), 2 x", "wing/reinforcement/covering", 2 * cad["spar_joiner"] * 1e-3 * carbon, L["spar_x_root"] + 15, 0.0, 12.0, "CAD volume x 1.55"),
        ("wing: tip rods 8/6 carbon, 2 x 200", "wing/reinforcement/covering", rods, L["spar_x_root"] + 30, 0.0, 50.0, "tube section x length x 1.55"),
        ("wing: balsa D-box 2 mm (LE to spar, full span), TE strips, servo bays, joint ribs", "wing/reinforcement/covering", balsa, 0.15 * L["mac"] + 10, 0.0, 8.0,
         "D-box area x 2 mm x 0.16 g/cm³ + 25 g strips and ribs"),
        (f"wing: rear carry-through tube {p['rear_spar_od']:g}/{p['rear_spar_id']:g} x {2 * p['rear_spar_half_length']:g} (carbon)", "wing/reinforcement/covering",
         cad["rear_spar"] * 1e-3 * carbon, x_rear, 0.0, 10.0, "CAD volume x 1.55"),
        ("wing: covering film", "wing/reinforcement/covering", film_wing, x_wing, 0.0, 8.0, "32 g/m² x wetted area"),
        ("wing: flap and aileron hinges, horns' hard points, tape", "wing/reinforcement/covering", 20.0, 0.82 * p["root_chord"], 0.0, 4.0, "assumed"),
        ("wing: panel joint pins, latches, root ribs (plywood)", "wing/reinforcement/covering", 30.0, 0.35 * p["root_chord"], 0.0, 10.0, "assumed"),
        ("fuselage: LW-PLA shell (1.5 mm) from the firewall joint to the tail cone", "pod", cad["pod"] * 1e-3 * lw, x_pod, 0.0, zt, "CAD shell volume x 0.55 g/cm³"),
        ("fuselage: nose cowl and spinner shell (LW-PLA, 1.5 mm), cooling intake", "pod", cad["nose"] * 1e-3 * lw + 8.0, L["x_nose"] + 0.6 * p["nose_cone_length"], 0.0, zt, "CAD x 0.55 + 8 g intake lips"),
        ("fuselage: firewall (PETG, solid) with the 4 x M4 inserts", "pod", cad["motor_mount"] * 1e-3 * petg * PRINT_FILL["motor_mount"] + 6.0, L["x_firewall"], 0.0, zt, "CAD x 1.25 + inserts"),
        ("fuselage: PETG keel frame and wing saddle (bolts, spar bridge)", "pod", 70.0, 100.0, 0.0, -30.0, "assumed (NISUS+'s)"),
        ("fuselage: hatch, latches, chin windows (camera, lidar)", "pod", 35.0, -180.0, 0.0, -20.0, "assumed"),
        ("fuselage: electronics tray (PETG)", "pod", cad["tray"] * 1e-3 * petg * PRINT_FILL["tray"], 35.0, 0.0, -82.0, "CAD x 1.25 x fill 0.5"),
        ("fuselage: battery tray (PETG)", "pod", cad["battery_tray"] * 1e-3 * petg * PRINT_FILL["battery_tray"], -30.0, 0.0, -125.0, "CAD x 1.25 x fill 0.6"),
        ("fuselage: vents, grommets, foam liner", "pod", 15.0, 50.0, 0.0, -70.0, "assumed"),
        (f"tail: roll-wrapped carbon tube {p['tail_tube_od']:g}/{p['tail_tube_id']:g} x {L['tube_length']:g}", "booms/tail", cad["tail_tube"] * 1e-3 * carbon, x_tube, 0.0, zt,
         "CAD volume x 1.55 (roll-wrapped: the frame study's torsion finding)"),
        ("tail: keel socket (PETG, ribbed) around the tube from the saddle to the tail cone", "booms/tail", cad["tail_socket"] * 1e-3 * petg * PRINT_FILL["tail_socket"],
         0.5 * (L["x_tube0"] + L["x_pod_end"]), 0.0, zt, "CAD x 1.25 x fill 0.4"),
        ("tail: end fitting (PETG) carrying the stabiliser's spar, the fin's post and the bumper", "booms/tail", cad["tail_fitting"] * 1e-3 * petg * PRINT_FILL["tail_fitting"],
         L["tail_le"] + 0.5 * p["tail_chord"], 0.0, zt, "CAD x 1.25 x fill 0.6"),
        ("tail: foam stabiliser and fin (11 mm plates)", "booms/tail", tail_foam, x_tail, 0.0, zt + 0.4 * p["fin_height"] * 0.3, "CAD volume x 30 kg/m³"),
        (f"tail: stabiliser spar {STAB_SPAR['od']:g}/{STAB_SPAR['id_']:g} carbon x {p['tail_span']:g}", "booms/tail", stab_spar, L["tail_le"] + 0.3 * p["tail_chord"], 0.0, L["tail_z"], "tube section x length x 1.55"),
        (f"tail: fin rod {FIN_ROD['od']:g}/{FIN_ROD['id_']:g} carbon x {FIN_ROD['length']:g}", "booms/tail", fin_rod, L["fin_le"] + 0.45 * p["fin_chord"], 0.0, L["tail_z"] + 150.0, "tube section x length x 1.55"),
        ("tail: covering film", "booms/tail", film_tail, x_tail, 0.0, zt + 60.0, "32 g/m² x wetted area"),
        ("tail: elevator/rudder hinges, servo covers", "booms/tail", 15.0, L["tail_le"] + 130, 0.0, zt, "assumed"),
        ("belly keel skid (TPU)", "joints/mounts/adhesive", cad["skid"] * 1e-3 * m["TPU 95A"]["rho"] * PRINT_FILL["skid"], 0.5 * (p["skid_x0"] + p["skid_x1"]), 0.0,
         -p["pod_height"] - 0.4 * p["skid_depth"], "CAD x 1.21 x fill 0.5"),
        ("tail bumper (TPU) under the end fitting", "joints/mounts/adhesive", 8.0, p["tail_x_end"] - 25.0, 0.0, zt - 25.0, "assumed"),
        ("GNSS mount, pitot stalk, chin windows' frames, antenna feed-throughs", "joints/mounts/adhesive", 20.0, -150.0, 0.0, -40.0, "assumed"),
        ("adhesive (epoxy, CA, hot glue), screws, inserts", "joints/mounts/adhesive", 50.0, 50.0, 0.0, -30.0, "assumed"),
    ]
    struct_mass = sum(i[2] for i in items)
    x_s = sum(i[2] * i[3] for i in items) / struct_mass
    z_s = sum(i[2] * i[5] for i in items) / struct_mass
    items.append(("construction allowance (8 % of structure and joints: paint, tape, filler)", "construction allowance",
                  ALLOWANCE_FRACTION * struct_mass, x_s, 0.0, z_s, f"{ALLOWANCE_FRACTION:.0%} of the structure, assumed"))
    return items


def mass_table(battery_key: str = DEFAULT_PACK, *, p=None, cad=None, cg_frac_mac: float = 0.28, battery_x: float | None = None) -> pd.DataFrame:
    """FALCO's mass table [g, mm] through NISUS+'s seams: the structure from the CAD, the components, the pack placed
    along its bay for the CG at ``cg_frac_mac`` of the MAC (the motor in the nose pulls it forward: the pack slides aft)."""
    return fs.mass_table(battery_key, p=falco.resolve(p), cad=cad or CAD, cg_frac_mac=cg_frac_mac, battery_x=battery_x, design=falco.Falco(),
                         items=structure_items, components=COMPONENTS, variant="Zero", battery=pack(battery_key))


def cg_inertia(table: pd.DataFrame, p=None) -> dict:
    return fs.cg_inertia(table, falco.resolve(p), design=falco.Falco())


def servo_check(p=None, **kw) -> pd.DataFrame:
    """NISUS+'s hinge-moment check on FALCO's surfaces (one rudder)."""
    return fs.servo_check(falco.resolve(p), design=falco.Falco(), **kw)


# ================================================================================================= the drive
#: T-Motor's bench table for the AT5220-A KV220 on APC 18x8 at 12S (the German page of the datasheet carries the rows to
#: 100 %); the 100 % row fits the motor model, the 70 % row checks the propeller's torque calibration.
PUBLISHED_STATIC = dict(voltage=43.33, current=48.86, power_w=2116.80, thrust_n=8.185 * G, rpm=8141.0, torque_nm=2.037, kv=220.0,
                        part_load=dict(voltage=43.93, current=20.49, power_w=900.27, thrust_n=4.668 * G, rpm=6232.0, torque_nm=1.131, throttle=0.70),
                        rows=[(0.40, 44.30, 4.27, 189.20, 3585, 0.383, 1469), (0.45, 44.27, 5.57, 246.66, 3980, 0.460, 1818), (0.50, 44.20, 8.59, 379.74, 4592, 0.600, 2446),
                              (0.55, 44.13, 11.17, 492.86, 5055, 0.745, 3009), (0.60, 44.08, 13.86, 611.06, 5473, 0.857, 3502), (0.65, 44.01, 16.93, 745.0, 5839, 0.991, 4080),
                              (0.70, 43.93, 20.49, 900.27, 6232, 1.131, 4668), (0.75, 43.84, 24.92, 1092.62, 6656, 1.326, 5323), (0.80, 43.75, 29.00, 1268.8, 6969, 1.466, 5924),
                              (0.90, 43.54, 38.65, 1682.85, 7614, 1.757, 7142), (1.00, 43.33, 48.86, 2116.80, 8141, 2.037, 8185)],
                        source="T-Motor AT5220-A KV220 + APC 18x8 at 12S, 40-100 % throttle (store.tmotor.com, the German page; the ligpower datasheet via "
                               "robotshop PDF to 70 %; snippets of 2026-10-10). The 65 % and 80 % powers read 611.06 and 1092.62 W in the source (copied from "
                               "the rows above: typos) and are replaced by V x I here",
                        datasheet=dict(mass_g=450.0, resistance_ohm=0.036, no_load_a=1.8, max_a=70.0, max_w=2700.0, cells="6-12S", dims="Φ60 x 112.3 mm"))
#: the alternatives weighed in ``drive_options`` (the KV380 has no published table in the snippets: its R is the datasheet's
#: 11 mΩ scaled as the KV220's fitted R is to its 36 mΩ — an assumption)
MOTORS = {
    "AT5220-A KV220": dict(kv=220.0, mass_g=450.0, r_sheet=0.036, no_load_a=1.8, max_a=70.0, max_w=2700.0, cells="6-12S", published=True,
                           url="https://store.tmotor.com/product/at5220-a-fixed-wing-motor.html"),
    "AT5220-A KV380": dict(kv=380.0, mass_g=465.0, r_sheet=0.011, no_load_a=4.0, max_a=100.0, max_w=2200.0, cells="6S", published=False,
                           url="https://www.ligpower.com/product/at5220-a--kv380-fixed-wing-motor.html"),
}
PROP = (20.0, 15.0)
ESC_MAX_A = 80.0


def propeller(d_in: float = PROP[0], pitch_in: float = PROP[1]):
    """The APC 20x15E as the frame study's scaled generic planform (NISUS+'s 9x6 shape x 20/15)."""
    prop = fst.propeller(d_in, pitch_in)
    return prop


def blade_section():
    from vegeta import boreas
    return boreas.Airfoil(name="thin cambered section", cl_alpha=2 * math.pi * 0.9, alpha0_deg=-2.5, cl_max=1.1, cd0=0.016, k=0.04,
                          source="assumed for a 20-inch blade at Re ~ 3.5e5 (NISUS+'s values, cd0 a little lower for the larger chord)")


def motor_model(no_load_a: float | None = None, max_current_a: float | None = None):
    """The AT5220-A KV220 fitted to its published full-throttle point on the 18x8 at 12S: R = (V − rpm/KV)/I (the ESC
    and the wires included, as NISUS+'s fit), the shaft torque kt (I − I0) there against the published 2.037 N m (the
    check of kt and I0: the datasheet's I0 is kept) and against the BEMT's 18x8 torque (the torque calibration), the
    published thrust against the BEMT's (the thrust calibration). The calibration ratios are carried to the 20x13 —
    the planform's, assumed to carry over (the frame study did the same). Returns (motor, fit)."""
    from vegeta import boreas
    P, D = PUBLISHED_STATIC, PUBLISHED_STATIC["datasheet"]
    no_load_a = D["no_load_a"] if no_load_a is None else no_load_a
    max_current_a = D["max_a"] if max_current_a is None else max_current_a
    prop18, af = fst.propeller(18.0, 8.0), blade_section()
    rpm, I, V = P["rpm"], P["current"], P["voltage"]
    R = (V - rpm / P["kv"]) / I
    motor = boreas.Motor("T-Motor AT5220-A KV220 (fitted)", kv_rpm_per_volt=P["kv"], resistance_ohm=R, no_load_current_a=no_load_a,
                         max_current_a=max_current_a, mass_kg=D["mass_g"] / 1000,
                         source=f"KV, I0, the current limit and the full-throttle point published ({P['source']}); R fitted at the point (ESC and wires included)")
    op = boreas.solve(prop18, af, rpm, 0.0, RHO0)
    Q_motor = motor.kt * (I - no_load_a)
    fit = {"rpm_at_point": rpm, "resistance_ohm": R, "no_load_a": no_load_a, "kt": motor.kt, "bemt_thrust_n": op.thrust, "published_thrust_n": P["thrust_n"],
           "thrust_correction": P["thrust_n"] / op.thrust, "bemt_torque_nm": op.torque, "motor_torque_nm": Q_motor, "published_torque_nm": P["torque_nm"],
           "torque_correction": P["torque_nm"] / op.torque, "kt_check": Q_motor / P["torque_nm"],
           "shaft_power_w": P["torque_nm"] * rpm * math.pi / 30, "motor_efficiency_at_point": P["torque_nm"] * rpm * math.pi / 30 / (V * I),
           "fitted_on": "APC 18x8 at 12S", "note": "the corrections say how far the generic blade is from the real 18x8 at the static point; carried to the 20x13 (assumed)"}
    pl = P["part_load"]
    op2 = boreas.solve(prop18, af, pl["rpm"], 0.0, RHO0)
    fit["check_part_load_torque_nm"] = op2.torque * fit["torque_correction"]
    fit["check_part_load_thrust_n"] = op2.thrust * fit["thrust_correction"]
    fit["published_part_load"] = dict(pl)
    return motor, fit


def _static_point(prop, af, kv, R, I0, V, kQ):
    """The static full-throttle point of a motor (kv, R, I0) on ``prop`` at the motor voltage ``V``: the rpm where the
    motor's torque meets the propeller's (calibrated by kQ); returns (rpm, current, torque)."""
    from vegeta import boreas
    kt = 60 / (2 * math.pi * kv)
    lo, hi = 300.0, kv * V
    for _ in range(60):
        n = 0.5 * (lo + hi)
        Im = (V - n / kv) / R
        Qm = kt * (Im - I0)
        Qp = boreas.solve(prop, af, n, 0.0, RHO0).torque * kQ
        lo, hi = (n, hi) if Qm > Qp else (lo, n)
    return n, (V - n / kv) / R, boreas.solve(prop, af, n, 0.0, RHO0).torque * kQ


def drive_options(p=None, battery_key_6s: str = "6s4p-p45b", battery_key_8s: str = DEFAULT_PACK, progress: bool = False) -> pd.DataFrame:
    """The motor / pack / propeller combinations weighed, at the nominal pack voltage, static, sea level: the full-throttle
    rpm, the motor current, the shaft power, the static thrust, the motor's efficiency there, and what the limits say
    (the motor's 180 s current, the ESC's 80 A). The requirement column is NISUS+'s static T/W 0.7 on FALCO's mass."""
    from vegeta import boreas
    m0, fit = motor_model()
    af = blade_section()
    kQ, kT = fit["torque_correction"], fit["thrust_correction"]
    r_scale = fit["resistance_ohm"] / MOTORS["AT5220-A KV220"]["r_sheet"]
    mass = cg_inertia(mass_table(), p)["mass_kg"]
    T_req = 0.70 * mass * G
    combos = [("AT5220-A KV220", battery_key_6s, (20.0, 13.0)), ("AT5220-A KV220", battery_key_6s, (20.0, 15.0)), ("AT5220-A KV220", battery_key_6s, (22.0, 12.0)),
              ("AT5220-A KV220", battery_key_8s, (20.0, 13.0)), ("AT5220-A KV220", battery_key_8s, (20.0, 15.0)), ("AT5220-A KV220", battery_key_8s, (22.0, 12.0)),
              ("AT5220-A KV380", battery_key_6s, (20.0, 13.0)), ("AT5220-A KV380", battery_key_6s, (18.0, 10.0))]
    rows = {}
    if progress:
        from tqdm.auto import tqdm
        combos = tqdm(combos, desc="drive options (motor x pack x propeller: BEMT static points)")
    for name, bk, (d, pi) in combos:
        M, b = MOTORS[name], pack(bk)
        R = M["r_sheet"] * r_scale
        prop = fst.propeller(d, pi)
        n, I, Q = _static_point(prop, af, M["kv"], R, M["no_load_a"], b.voltage, kQ)
        limited = ""
        I_lim = min(M["max_a"], ESC_MAX_A * b.voltage / b.voltage)
        if I > I_lim:                                                   # the throttle held back to the current limit
            kt = 60 / (2 * math.pi * M["kv"])
            Q_lim = kt * (I_lim - M["no_load_a"])
            n = n * math.sqrt(Q_lim / Q)
            lo, hi = 300.0, n * 1.2
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if boreas.solve(prop, af, mid, 0.0, RHO0).torque * kQ < Q_lim else (lo, mid)
            n, I, Q = lo, I_lim, Q_lim
            limited = f"throttle held to {I_lim:.0f} A (the motor's {M['max_a']:.0f} A / the ESC's {ESC_MAX_A:.0f} A)"
        T = boreas.solve(prop, af, n, 0.0, RHO0).thrust * kT
        P_shaft = Q * n * math.pi / 30
        V_m = n / M["kv"] + I * R
        P_in = V_m * I
        rows[f"{name} / {b.name.split(' ')[0]} / {d:g}x{pi:g}"] = {
            "motor": name, "pack": b.name, "V nominal": b.voltage, "propeller": f"APC {d:g}x{pi:g}E", "rpm (static, full throttle)": n, "motor current [A]": I,
            "shaft power [W]": P_shaft, "motor efficiency": P_shaft / P_in, "static thrust [N]": T, f"T/W ≥ 0.7 ({T_req:.0f} N)": T >= T_req,
            "pitch speed [m/s]": n / 60 * pi * 0.0254, "limit": limited or "within the motor's and the ESC's limits",
            "published": "the KV220's 18x8 table" if M["published"] else "no table: R scaled from the datasheet's 11 mΩ (assumed)"}
    df = pd.DataFrame(rows).T
    df.attrs["selected"] = f"AT5220-A KV220 / {pack(battery_key_8s).name.split(' ')[0]} / 20x15"
    df.attrs["T_req_N"] = T_req
    df.attrs["why"] = ("the KV220 on 6S has the volts for ~3900 rpm only (~0.5 kW shaft, T/W < 0.7: no climb margin at 4500 m); the KV380 on 6S "
                       "would run the 20x13 past the ESC's 80 A and sits at that limit when held (3/4 of its power, the ESC at its edge); the KV220 "
                       "on 8S3P — the same 24 cells re-connected, the same box — gives the frame study's 20-inch point at ~50 A within every limit; the pitch "
                       "is the propeller trade's (falco_flight.propeller_trade: the 20x15 over the 20x13 for the cruise). "
                       "The fallback that keeps the 6S4P is the KV380 with an 18x10 (full throttle at the ESC's limit, the smaller disc: the "
                       "frame study's 18-inch gain, not the 20-inch one)")
    return df


PROPULSION_JSON = DATA / "falco_propulsion.json"
RPM_GRID = np.r_[150.0, 300.0, 500.0, 750.0, np.arange(1000.0, 2000.0, 250.0), np.arange(2000.0, 8001.0, 500.0)]


def build_drive(*, V=V_GRID, rpm=RPM_GRID, battery_v=BATTERY_V, esc_efficiency=0.96, correct: bool = True) -> NisusPlusDrive:
    """NISUS+'s map builder on the 20x15 and the fitted AT5220 at the 8S3P's nominal voltage (~400 BEMT solves)."""
    motor, fit = motor_model()
    dr = fs.build_drive(V=V, rpm=rpm, battery_v=battery_v, esc_efficiency=esc_efficiency, correct=correct, prop=propeller(), motor=motor, fit=fit)
    dr.notes["limits"] = ("no installation effect in the map (falco_flight.installation gives the tractor's wake fraction and thrust deduction at the "
                          "flight points); the pack's sag enters through v_batt; no motor heating model")
    dr.notes["blade section"] = blade_section().source
    return dr


def drive(path: Path = PROPULSION_JSON, rebuild: bool = False) -> NisusPlusDrive:
    """The cached drive (``data/falco_propulsion.json``), rebuilt when missing or ``rebuild``."""
    path = Path(path)
    if path.exists() and not rebuild:
        return NisusPlusDrive.from_json(json.loads(path.read_text()))
    d = build_drive()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d.to_json(), indent=1))
    return d


def mission_table(dr, airframe_fn, plan: MissionPlan = MissionPlan(), packs=None) -> pd.DataFrame:
    """NISUS+'s battery iteration on FALCO's packs (the 8S3P and, for the record, the 6S ones)."""
    rows = {}
    for k in (packs or ("8s3p-p45b", "6s4p-p45b")):
        b = pack(k)
        af = airframe_fn(k)
        r = mission_energy(b, dr, af, plan)
        rows[b.name] = {"pack mass [g]": b.mass_g, "aircraft mass [kg]": af["mass_kg"], "nominal [Wh]": b.energy_wh, "available [Wh]": r["E_available_wh"],
                        "climb [Wh]": r["E_climb_wh"], "climb time [s]": r["t_climb_s"], "survey power [W]": r["P_survey_w"], "used [Wh]": r["E_used_wh"],
                        "left [% avail.]": r["E_left_pct_available"], "reserve kept": r["reserve_kept"], "feasible survey [min]": r["t_survey_feasible_s"] / 60,
                        f"{plan.survey_s / 60:.0f}-min survey ok": r["survey_time_ok"] and r["reserve_kept"], "fits the bay": battery_fits(b)["fits"]}
    return pd.DataFrame(rows).T


__all__ = ["CAD", "cad_numbers", "PACKS", "DEFAULT_PACK", "BATTERY_V", "pack", "pack_table", "battery_fits", "COMPONENTS", "components", "component_table",
           "structure_items", "mass_table", "cg_inertia", "PUBLISHED_STATIC", "MOTORS", "PROP", "ESC_MAX_A", "propeller", "blade_section", "motor_model",
           "drive_options", "build_drive", "drive", "drive_table", "mission_table", "requirements", "MissionPlan", "mission_energy", "phase_power",
           "electrical_loads", "servo_check", "atmosphere", "JETSON_INSTALLATION_W"]
