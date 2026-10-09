"""NISUS systems — the two variants, their components with sources, mass budgets, centre of gravity and inertia,
electrical loads, propulsion sizing and selection, batteries and the mission energy (notebook 31).

Everything here is a number with a stated origin. Three kinds appear, and every table says which:

- **calculated** — from the CAD (``nisus.CAD`` volumes x material densities), from the polar, from Boreas BEMT;
- **sourced** — a published figure: the component table ``COMPONENTS`` carries the URL, the access date and whether
  the page itself was opened (it was not: the research of 2026-10-09 could only read search-index snippets of the
  cited pages, so **no availability is confirmed** — ``data/nisus_component_sources.md`` has the whole table);
- **assumed** — an engineer's first estimate, labelled so, to be replaced by a measurement.

The variants: **Nisus-OBS** flies with a pilot on the ground (camera, 5.8 GHz video link, flight controller with
stabilisation and return-to-home); **Nisus-Zero** adds a Jetson computer for onboard vision with its own camera,
power branch and a data link. The reference budget of the earlier camera-free design (1100 g) is kept only as the
column to compare against.

    import nisus_systems as ns
    t = ns.mass_table("Zero")                 # the mass table with positions, the battery placed for the target CG
    ns.cg_inertia(t)                          # mass, CG, inertia tensor about the CG
    ns.electrical_loads("Zero")               # every load: supply, idle/typical/peak, duty, regulator, battery side
    pm = ns.propulsion_map()                  # thrust, power, current over (airspeed, rpm) from Boreas, cached as JSON
    ns.mission_energy("Zero", ns.battery("gens-ace-3s-2200"), pm, af)   # the ten-minute survey with its return reserve
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import nisus

G = 9.81
RHO = 1.225
DATA = Path(__file__).resolve().parent / "data"
ACCESS = "2026-10-09"
SNIPPET = "search-index snippet of the cited page (2026-10-09); page not opened; availability not confirmed"

# ================================================================================================= materials
MATERIALS = {
    # density g/cm³, E MPa, strength MPa (yield or flexural), source
    "XPS foam 30": dict(rho=0.030, E=15.0, strength=0.4, source="extruded polystyrene 30 kg/m³ (Styrodur-class), catalogue values; assumption"),
    "carbon tube": dict(rho=1.55, E=120e3, strength=500.0, source=nisus.CARBON_TUBE["source"]),
    "PETG printed": dict(rho=1.25, E=2000.0, strength=45.0, strength_z=27.0,
                         source="PETG filament datasheet class (E 2.0 GPa, yield 45 MPa in the layer plane); the Z (layer-adhesion) "
                                "strength is taken as 60 % of that (assumption to check with coupons)"),
    "LW-PLA printed": dict(rho=0.55, E=900.0, strength=10.0,
                           source="colorFabb LW-PLA foamed at ~230 °C: 0.5-0.6 g/cm³ (vendor); E and strength of the foamed material assumed"),
    "TPU 95A": dict(rho=1.21, E=30.0, strength=8.0, source="TPU 95A filament datasheet class"),
    "balsa": dict(rho=0.16, E=3000.0, strength=15.0, source="medium balsa, 160 kg/m³"),
    "covering film": dict(areal_g_m2=32.0, source="iron-on laminating film, 30-35 g/m² (vendor class)"),
}
#: printed parts' mass as a share of their solid CAD volume: perimeters + gyroid infill on a 14-30 mm wide part
#: (assumption; the slicer's estimate replaces it); the motor mount is printed solid
PRINT_FILL = {"boom_fitting": 0.60, "tail_fitting": 0.60, "tray": 0.50, "battery_tray": 0.60, "motor_mount": 1.0, "skid": 0.50}
FOAM_LIGHTENING = 0.25      # share of the wing core's volume removed (lightening holes aft of the spar); assumption
TAIL_SPAR = dict(od=4.0, id_=3.0, length=400.0)   # mm, carbon tube in the stabiliser
TIP_RODS = dict(od=6.0, id_=4.0, length=200.0, count=2)   # mm, carbon rods from the spar's end to the tips

# ================================================================================================= CAD numbers
#: CAD volumes [mm³] of the default Nisus parts (``nisus.Nisus().generate(part=...).measure()['volume']``);
#: ``cad_numbers(recompute=True)`` rebuilds them (about 30 s).
CAD = {"aircraft": 8792303.0, "wing": 4745391.0, "spar": 28302.0, "rear_spar": 9904.0, "pod": 175564.0, "nose": 15556.0,
       "boom": 15551.0, "boom_fitting": 44290.0, "tail": 528186.0, "tail_fitting": 10713.0, "motor_mount": 4951.0, "tray": 14556.0,
       "skid": 38671.0, "battery_tray": 26984.0}


def cad_numbers(p: dict | None = None, *, recompute: bool = False) -> dict:
    if not recompute and not p:
        return dict(CAD)
    d = nisus.Nisus()
    out = {}
    for part in CAD:
        out[part] = float(d.generate(**nisus.overrides(p), part=part).measure()["volume"])
    return out


# ================================================================================================= components
@dataclass(frozen=True)
class Component:
    """One bought part: mass, place, power. ``variants``: which aircraft carry it ('OBS', 'Zero'); ``supply``: where
    its power comes from ('battery' direct, '5V' the flight controller's 5.3 V BEC, 'servo' its servo rail, 'jetson'
    the computer's own 5 V buck, 'vtx5v' the video transmitter's 5 V output, 'none'); the powers are at the device
    (the regulator's loss is added in ``electrical_loads``). ``kind``: 'sourced' / 'assumed' / 'calculated'."""
    key: str
    group: str
    mass_g: float
    variants: tuple
    x_mm: object                     # a position [mm], or {variant: position} when the two aircraft place it differently
    z_mm: float = -55.0
    y_mm: float = 0.0
    supply: str = "none"
    supply_v: float = 0.0
    p_idle_w: float = 0.0
    p_typ_w: float = 0.0
    p_peak_w: float = 0.0
    price_eur: float | None = None
    url: str = ""
    kind: str = "sourced"
    source: str = ""
    availability: str = "not confirmed"
    note: str = ""

    def x(self, variant: str) -> float:
        return float(self.x_mm[variant]) if isinstance(self.x_mm, dict) else float(self.x_mm)


BOTH = ("OBS", "Zero")
ZERO = ("Zero",)
# positions: the aircraft frame of ``nisus`` (x aft from the wing root LE, z up from the pod top), mm
COMPONENTS = [
    # --- propulsion
    Component("motor SunnySky X2216 V3 KV1250", "propulsion/ESC/wiring", 69.0, BOTH, 271.0, 0.0, supply="battery", supply_v=11.1,
              price_eur=23.0, url="https://team-legit.com/sunnysky-x2216-v3-brushless-motor-1250kv.html",
              source=f"mass 69 g, 40 A/30 s (retailer copies of SunnySky's table); {SNIPPET}",
              note="static 11.1 V APC 9x6: 20.8 A, 231 W, 1040 gf (retailer copy of the SunnySky table; official file not opened)"),
    Component("propeller APC 9x6E", "propulsion/ESC/wiring", 12.0, BOTH, 296.0, 0.0, price_eur=3.4,
              url="https://abc-rc.pl/product-pol-1259-Smiglo-APC-9x6E-LP09060E.html", kind="assumed",
              source=f"14.50 zł at abc-rc.pl; mass not stated (APC 10x5E is 20.1 g): 12 g assumed; {SNIPPET}"),
    Component("ESC AM32 40 A 2-4S (5 V/2 A BEC, unused)", "propulsion/ESC/wiring", 14.5, BOTH, 215.0, -30.0, supply="battery", supply_v=11.1,
              p_idle_w=0.3, p_typ_w=0.3, p_peak_w=0.3, price_eur=15.0,
              url="https://totsrc.com/products/40amp-esc-40a-esc-4s-esc-for-rc-airplane-fpv-wing-rc-helicopter-w-bec-5v2a",
              source=f"13.7-14.5 g with BEC (listing); {SNIPPET}; the Hobbywing Skywalker 40A V2 (36 g, €19.90) is the proven fallback",
              note="quiescent 0.3 W assumed (gate drivers, BEC idle); the motor's electrical power is in the propulsion map"),
    Component("wiring, XT60, bullets, leads", "propulsion/ESC/wiring", 25.0, BOTH, 60.0, -60.0, kind="assumed",
              source="estimate: 14 AWG battery lead 2 x 150 mm, XT60 pair, 3 x 16 AWG motor leads 350 mm, bullets, heat shrink"),
    # --- battery (the pack itself comes from BATTERIES; this is its retention)
    Component("battery strap, velcro pad, foam", "battery/retention", 8.0, BOTH, -190.0, -80.0, kind="assumed",
              source="estimate: 20 mm hook-and-loop strap with buckle, two velcro pads, 3 mm foam"),
    # --- actuation
    Component("servo Emax ES08MA II (aileron L)", "servos/linkages", 12.0, BOTH, 160.0, 5.0, y_mm=-480.0, supply="servo", supply_v=6.0,
              p_idle_w=0.15, p_typ_w=0.6, p_peak_w=6.0, price_eur=9.0,
              url="https://www.getfpv.com/emax-es08ma-ii-12g-mini-metal-gear-analog-servo-for-rc-model.html",
              source=f"12 g, 2.0 kg·cm / 0.10 s per 60° at 6 V (listings); stall current not stated: 1.0 A at 6 V assumed; {SNIPPET}",
              note="typical 0.6 W = 20 % of the time moving at 0.5 A; peak = stall"),
    Component("servo Emax ES08MA II (aileron R)", "servos/linkages", 12.0, BOTH, 160.0, 5.0, y_mm=480.0, supply="servo", supply_v=6.0,
              p_idle_w=0.15, p_typ_w=0.6, p_peak_w=6.0, price_eur=9.0, source="as the left aileron servo"),
    Component("servo Emax ES08MA II (elevator)", "servos/linkages", 12.0, BOTH, 185.0, 0.0, y_mm=-140.0, supply="servo", supply_v=6.0,
              p_idle_w=0.15, p_typ_w=0.6, p_peak_w=6.0, price_eur=9.0, source="as the aileron servos; in the left wing root behind the spar, a 1.2 mm pushrod inside the left boom tube to the elevator"),
    Component("servo Emax ES08MA II (both rudders, torque rod)", "servos/linkages", 12.0, BOTH, 185.0, 0.0, y_mm=140.0, supply="servo", supply_v=6.0,
              p_idle_w=0.15, p_typ_w=0.4, p_peak_w=6.0, price_eur=9.0, source="as the aileron servos; in the right wing root behind the spar, a pushrod inside the right boom to its rudder, a cross rod along the stabiliser to the left rudder"),
    Component("horns, clevises, pushrods, hinges", "servos/linkages", 12.0, BOTH, 400.0, -15.0, kind="assumed",
              source="estimate: 4 control horns, 1.2 mm carbon pushrods (two of 150 mm, two of 480 mm inside the boom tubes), the rudder cross rod, CA hinges"),
    # --- flight control
    Component("flight controller Holybro Kakute F405-Wing Mini", "flight controller/receiver/GNSS/wiring", 17.0, BOTH, {"OBS": -65.0, "Zero": 35.0}, -60.0,
              supply="battery", supply_v=11.1, p_idle_w=1.0, p_typ_w=1.2, p_peak_w=1.8, price_eur=84.1,
              url="https://holybro.com/products/kakute-f405-wing-mini",
              source=f"17 g with the USB board, 2-8S input, BEC 5.3 V 3 A (4.8 A peak) + servo rail 5.3/7.2 V 3 A, 110 A current sensor, "
                     f"ArduPilot target KakuteF405-Wing (Plane 4.5.2+, Holybro's page); €84.10 at openelab.io ('in stock' snippet); {SNIPPET}",
              note="board power 1.0-1.8 W assumed (F405 + IMU + baro + blackbox + LEDs; not stated); it feeds the 5.3 V loads below"),
    Component("GNSS + compass Matek M10Q-5883", "flight controller/receiver/GNSS/wiring", 8.0, BOTH, -230.0, -5.0, supply="5V", supply_v=5.3,
              p_idle_w=0.25, p_typ_w=0.25, p_peak_w=0.35, price_eur=25.0,
              url="https://www.unmannedtechshop.co.uk/products/matek-m10q-5883-gps-module",
              source=f"~8 g, 4-9 V (listing); current: the snippet's 13 mA looks too low for an M10 + compass, 50 mA (0.25 W) assumed; {SNIPPET}"),
    Component("RC receiver RadioMaster RP3 V2 ELRS 2.4 GHz", "flight controller/receiver/GNSS/wiring", 4.6, BOTH, {"OBS": -50.0, "Zero": 50.0}, -40.0, supply="5V", supply_v=5.3,
              p_idle_w=0.5, p_typ_w=0.5, p_peak_w=0.8, price_eur=20.0,
              url="https://radiomasterrc.com/products/rp3-expresslrs-2-4ghz-nano-receiver",
              source=f"4.6 g with two antennas (product page); current not stated: 100 mA at 5 V assumed (ELRS RX with 100 mW telemetry); {SNIPPET}"),
    Component("telemetry Holybro SiK V3 433 MHz", "flight controller/receiver/GNSS/wiring", 7.4, BOTH, {"OBS": -40.0, "Zero": 60.0}, -40.0, supply="5V", supply_v=5.3,
              p_idle_w=0.15, p_typ_w=0.4, p_peak_w=0.55, price_eur=30.0, url="https://holybro.com/products/sik-telemetry-radio-v3",
              source=f"3.6 g (7.4 g with antenna); TX 100 mA at 20 dBm, RX 25 mA at 5 V (Holybro docs); typical = 50 % TX duty; {SNIPPET}"),
    Component("signal wiring, connectors, antenna mounts", "flight controller/receiver/GNSS/wiring", 15.0, BOTH, {"OBS": -60.0, "Zero": 40.0}, -40.0, kind="assumed",
              source="estimate: JST-GH / servo leads, 433 MHz and 2.4 GHz antenna mounts, cable ties"),
    # --- camera and video link (both variants: the pilot's view; Zero's vision camera is on the Jetson)
    Component("camera RunCam Phoenix 2 (pilot's view)", "camera/video", 9.0, BOTH, -288.0, -45.0, supply="vtx5v", supply_v=5.0,
              p_idle_w=1.0, p_typ_w=1.0, p_peak_w=1.0, price_eur=28.0, url="https://shop.runcam.com/runcam-phoenix-2/",
              source=f"9 g, 5-36 V, 200 mA at 5 V (RunCam page); {SNIPPET}"),
    Component("video transmitter RushFPV Tank Solo 5.8 GHz", "camera/video", 12.0, BOTH, {"OBS": -140.0, "Zero": 100.0}, -8.0, supply="battery", supply_v=11.1,
              p_idle_w=2.2, p_typ_w=3.0, p_peak_w=4.6, price_eur=40.0, url="https://rushfpv.net/products/tank-solo-vtx",
              kind="assumed",
              source=f"12 g, 7-36 V, 25/400/800 mW steps, 5 V/1 A aux output (product page); its current is not stated: the Rush Tank Ultimate "
                     f"Mini's 185 mA at 200 mW and 380 mA at 800 mW (~12 V) give 2.2 W idle (25 mW), ~3.0 W at 400 mW, 4.6 W at 800 mW — educated estimate; {SNIPPET}"),
    Component("5.8 GHz antenna (RHCP) + camera mount", "camera/video", 15.0, BOTH, {"OBS": -130.0, "Zero": 110.0}, -20.0, kind="assumed",
              source="estimate: 7 g antenna with SMA, 8 g printed camera cradle and lens window"),
    Component("video wiring, 470 µF capacitor", "camera/video", 6.0, BOTH, {"OBS": -100.0, "Zero": 60.0}, -40.0, kind="assumed", source="estimate"),
    # --- the Jetson installation (Nisus-Zero): the prompt's first-approximation installation of an original Nano 4 GB
    Component("Jetson Nano 4 GB B01 dev kit: computer + carrier + heatsink", "computer", 150.0, ZERO, -65.0, -50.0, y_mm=28.0, supply="jetson", supply_v=5.0,
              p_idle_w=4.0, p_typ_w=10.0, p_peak_w=12.0, price_eur=None, url="https://developer.nvidia.com/embedded/lifecycle",
              source=f"mass 150 g from the earlier estimate (module mass not published); 5 W / 10 W modes (NVIDIA; nvpmodel); the dev kit is "
                     f"end-of-life, the Nano module is available to January 2027, final software JetPack 4.6.6 (NVIDIA forum/lifecycle); {SNIPPET}",
              note="the installation's 18 W battery-side figure (ns.JETSON_INSTALLATION_W) covers this row, the camera, the fan and the conversion loss"),
    Component("vision camera IMX219 (CSI) + cable", "computer", 15.0, ZERO, -282.0, -30.0, supply="jetson", supply_v=5.0,
              p_idle_w=0.3, p_typ_w=0.5, p_peak_w=0.6, price_eur=25.0, url="https://www.elektor.com/raspberry-pi-camera-module-v2",
              source=f"camera 3 g + cable/holder (15 g in the earlier estimate); current not stated, 100 mA assumed; {SNIPPET}"),
    Component("5 V / 5 A buck converter for the Jetson", "computer", 20.0, ZERO, -20.0, -75.0, kind="assumed",
              source="estimate: a 5 V 5 A synchronous buck module (e.g. Matek UBEC class), 90 % efficiency assumed"),
    Component("Jetson mounts, fan, wiring", "computer", 35.0, ZERO, -65.0, -75.0, y_mm=28.0, supply="jetson", supply_v=5.0,
              p_idle_w=0.5, p_typ_w=1.0, p_peak_w=1.5, kind="assumed", source="estimate (the earlier 35 g); the 40 mm fan 1 W"),
    Component("storage (microSD) and allowance", "computer", 10.0, ZERO, -65.0, -50.0, y_mm=28.0, kind="assumed", source="the earlier 10 g"),
    Component("data link for the Jetson (2.4 GHz Wi-Fi / LTE USB stick + antenna)", "computer", 25.0, ZERO, 120.0, -10.0, supply="jetson", supply_v=5.0,
              p_idle_w=1.0, p_typ_w=2.0, p_peak_w=3.0, price_eur=40.0, kind="assumed",
              source="assumption: the earlier estimate had no link for the computer's results; a USB LTE stick or a Wi-Fi radio with antenna, 25 g, 2 W"),
    Component("Jetson mounting frame (printed) and cooling duct", "joints/mounts/adhesive", 12.0, ZERO, -65.0, -80.0, y_mm=28.0, kind="assumed",
              source="estimate: PETG rails in the pod and an inlet duct to the heatsink fan"),
]
JETSON_INSTALLATION_W = 18.0          # battery-side, the complete original-Nano installation (prompt's estimate); 15-20 W sensitivity
JETSON_INSTALLATION_RANGE_W = (15.0, 20.0)

#: the computer options for Nisus-Zero: the reference (the prompt's original Nano installation) and the substitution the
#: market offers today — a substitution needs the mass, power and packaging below re-checked (the Orin kit is 9-19 V
#: input: it needs a boost/buck-boost from 3S, and its mass is not published)
JETSON_OPTIONS = {
    "nano_b01": dict(name="Jetson Nano 4 GB developer kit (B01) installation", mass_g=230.0, battery_w=18.0, modes_w=(5.0, 10.0),
                     software="JetPack 4.6.6 (L4T 32.7, Ubuntu 18.04) — final release", status="dev kit EOL; module available to Jan 2027",
                     size_mm=(100, 80, 30), input="5 V (barrel jack 4 A)", kind="prompt's estimate; status sourced (NVIDIA forum/lifecycle snippets)"),
    "orin_nano_super": dict(name="Jetson Orin Nano Super developer kit installation", mass_g=300.0, battery_w=24.0, modes_w=(7.0, 15.0, 25.0),
                            software="JetPack 6.2 (Ubuntu 22.04, CUDA 12): JetPack 4 code must be retargeted", status="current product; EU €355-519, one EU shop out of stock (snippet)",
                            size_mm=(100, 79, 21), input="9-19 V DC (needs a boost from 3S) or 5 V on the module's carrier rails",
                            kind="mass not published: 300 g assumed (kit with heatsink/fan ~250 g + camera, mounts, converter 50 g); battery-side power at the 15 W mode + camera + converter 85 %: ~24 W (assumed)"),
}

REFERENCE_BUDGET_G = {"wing/reinforcement/covering": 220.0, "pod": 150.0, "booms/tail": 120.0, "joints/mounts/adhesive": 60.0,
                      "propulsion/ESC/wiring": 140.0, "battery/retention": 200.0, "servos/linkages": 40.0,
                      "flight controller/receiver/GNSS/wiring": 60.0, "construction allowance": 110.0}
ALLOWANCE_FRACTION = 0.08             # of the structure and joints: glue soak, paint, tape, filler (assumption)


def components(variant: str) -> list:
    if variant not in nisus.VARIANTS:
        raise ValueError(f"variant must be one of {nisus.VARIANTS}")
    return [c for c in COMPONENTS if variant in c.variants]


def component_table(variant: str) -> pd.DataFrame:
    rows = [{"item": c.key, "group": c.group, "mass [g]": c.mass_g, "x [mm]": c.x(variant), "supply": c.supply, "P typ [W]": c.p_typ_w,
             "price [EUR]": c.price_eur, "kind": c.kind, "availability": c.availability, "url": c.url, "source": c.source}
            for c in components(variant)]
    return pd.DataFrame(rows).set_index("item")


# ================================================================================================= batteries
@dataclass(frozen=True)
class Battery:
    key: str
    name: str
    cells: int
    chemistry: str
    capacity_mah: float
    mass_g: float
    size_mm: tuple
    c_rating: float
    price: str
    url: str
    availability: str = "not confirmed"
    cell_v_nominal: float = 3.7
    note: str = ""

    @property
    def voltage(self):
        return self.cells * self.cell_v_nominal

    @property
    def energy_wh(self):
        return self.voltage * self.capacity_mah / 1000

    @property
    def max_current_a(self):
        return self.c_rating * self.capacity_mah / 1000


BATTERIES = [
    Battery("gens-ace-3s-2200", "Gens Ace 3S 2200 mAh 45C G-Tech", 3, "LiPo", 2200, 190.0, (105, 34, 23), 45, "€24.99 (gensace.de)",
            "https://gensace.de/collections/3s-lipo-battery", note=f"the reference pack; {SNIPPET}"),
    Battery("tattu-3s-2200", "Tattu 3S 2200 mAh 45C", 3, "LiPo", 2200, 190.0, (107, 36, 24), 45, "not stated",
            "https://www.motionew.com/shop/power-solution/battery-lipo-3s-2200mah-45c-tattu/", note=SNIPPET),
    Battery("gnb-3s-3000-lihv", "GNB 3S 3000 mAh 120C LiHV", 3, "LiHV", 3000, 213.0, (109, 34, 30), 120, "not stated",
            "https://www.gaoneng.shop/products/gaoneng-gnb-lihv-3s-11.4v-3000mah-120c-xt60-lipo-battery", cell_v_nominal=3.8,
            note=f"LiHV: 3.8 V nominal, 4.35 V charged — the ESC and the regulators must accept 13.05 V (they do: 2-8S); {SNIPPET}"),
    Battery("turnigy-3s-3000-nano", "Turnigy nano-tech 3S 3000 mAh 30C", 3, "LiPo", 3000, 215.0, (135, 44, 17), 30, "not stated",
            "https://hobbyking.com/turnigy-nano-tech-3000mah-3s-30c-lipo-pack-wxt60.html", note=SNIPPET),
    Battery("tattu-3s-3300", "Tattu 3S 3300 mAh 30C", 3, "LiPo", 3300, 248.0, (141, 44, 21), 30, "US$47.00",
            "https://www.motionew.com/shop/power-solution/battery-lipo-3s-3300mah-30c-tattu/", note=SNIPPET),
    Battery("gens-ace-3s-3300-60c", "Gens Ace 3S 3300 mAh 60C G-Tech", 3, "LiPo", 3300, 267.0, (136.5, 42.6, 21.6), 60, "not stated",
            "https://genstattu.com/gens-ace-g-tech-3300mah-3s-11-1v-60c-lipo-battery-pack-with-xt60-plug/", note=SNIPPET),
    Battery("turnigy-4s-2200", "Turnigy 4S 2200 mAh 40C", 4, "LiPo", 2200, 216.0, (106, 35, 29), 40, "not stated",
            "https://hobbyking.com/turnigy-2200mah-4s-40c-lipo-pack-w-xt60.html", note=f"4S: the motor needs a lower KV or a smaller propeller (the X2216 KV1250 on 9x6 at 14.8 V exceeds 40 A static); {SNIPPET}"),
    Battery("liion-3s2p-p42a", "Li-ion 3S2P Molicel P42A (assembled, US shop)", 3, "Li-ion", 8400, 450.0, (70, 48, 67), 10.7, "US$102",
            "https://nexusbatterysystems.com/products/3s2p-42a", cell_v_nominal=3.6,
            note=f"90 A pack claim (45 A per cell); 10.8 V nominal, 9.0 V at the end: the propulsion map at 11.1 V flatters it by ~5 %; no EU seller with a stated mass found; {SNIPPET}"),
]


def battery(key: str) -> Battery:
    return next(b for b in BATTERIES if b.key == key)


def battery_table() -> pd.DataFrame:
    rows = [{"pack": b.name, "cells": b.cells, "chemistry": b.chemistry, "V nom": b.voltage, "mAh": b.capacity_mah, "Wh": b.energy_wh,
             "mass [g]": b.mass_g, "Wh/kg": b.energy_wh / b.mass_g * 1000, "L x W x H [mm]": "x".join(f"{v:g}" for v in b.size_mm),
             "C": b.c_rating, "max A": b.max_current_a, "price": b.price, "availability": b.availability, "url": b.url}
            for b in BATTERIES]
    return pd.DataFrame(rows).set_index("pack")


def battery_fits(b: Battery, p=None) -> dict:
    """Does the pack fit the battery bay (length along x, width, height inside the pod with 4 mm of foam around)?"""
    p = nisus.resolve(p)
    x0, x1, w, h = nisus.Nisus.bays(p)["battery bay"]
    L_bay = x1 - x0
    dims = sorted(b.size_mm, reverse=True)
    fits = dims[0] <= L_bay - 6 and dims[1] <= w - 8 and dims[2] <= h - 20
    return {"bay [mm]": (L_bay, w, h), "pack [mm]": tuple(dims), "fits": bool(fits)}


# ================================================================================================= mass table
def structure_items(p=None, cad: dict | None = None, variant: str = "Zero") -> list:
    """The airframe's parts as (item, group, mass_g, x, y, z, basis): CAD volumes x densities, with the stated
    lightening, print fill and covering. Positions are the parts' centroids in the aircraft frame [mm]."""
    p = nisus.resolve(p)
    cad = cad or CAD
    L = nisus.Nisus.layout(p)
    m = MATERIALS
    x_wing = L["x_mac_le"] + 0.42 * L["mac"]
    prof = nisus.Nisus.pod_profile(p)
    per = math.pi * (prof[:, 1] + prof[:, 2])
    x_pod = float(np.trapezoid(prof[:, 0] * per, prof[:, 0]) / np.trapezoid(per, prof[:, 0]))
    z_pod = float(np.trapezoid(prof[:, 3] * per, prof[:, 0]) / np.trapezoid(per, prof[:, 0]))
    w = nisus.wetted_areas(p)
    foam = cad["wing"] * 1e-3 * m["XPS foam 30"]["rho"] * (1 - FOAM_LIGHTENING)
    # the D-box: both skins from the leading edge to the spar (30 % chord) over the inner 60 % of the span, 1.5 mm balsa
    balsa = 2 * 0.30 * 0.5 * (p["root_chord"] + p["tip_chord"]) * 0.6 * p["span"] * 1.5 * 1e-3 * m["balsa"]["rho"] * 1.05 + 8.0
    spar = cad["spar"] * 1e-3 * m["carbon tube"]["rho"]
    rods = TIP_RODS["count"] * math.pi / 4 * (TIP_RODS["od"] ** 2 - TIP_RODS["id_"] ** 2) * TIP_RODS["length"] * 1e-3 * m["carbon tube"]["rho"]
    film_wing = m["covering film"]["areal_g_m2"] * w["wing"]
    x_tail = L["tail_le"] + 0.4 * p["tail_chord"]
    tail_foam = cad["tail"] * 1e-3 * m["XPS foam 30"]["rho"]
    tail_spar = math.pi / 4 * (TAIL_SPAR["od"] ** 2 - TAIL_SPAR["id_"] ** 2) * TAIL_SPAR["length"] * 1e-3 * m["carbon tube"]["rho"]
    film_tail = m["covering film"]["areal_g_m2"] * (w["tail"] + w["fins"])
    booms = cad["boom"] * 1e-3 * m["carbon tube"]["rho"]
    petg = m["PETG printed"]["rho"]
    items = [
        ("wing: XPS core (25 % lightened)", "wing/reinforcement/covering", foam, x_wing, 0.0, 6.0, "CAD volume x 30 kg/m³ x 0.75"),
        ("wing: carbon spar tube 10/8 x 1000", "wing/reinforcement/covering", spar, L["spar_x_root"], 0.0, 4.0, "CAD volume x 1.55 g/cm³"),
        ("wing: tip rods 6/4 carbon, 2 x 200", "wing/reinforcement/covering", rods, L["spar_x_root"] + 10, 0.0, 25.0, "tube section x length x 1.55"),
        ("wing: balsa D-box 1.5 mm (LE to spar, inner 60 %), TE strips, servo bays", "wing/reinforcement/covering", balsa, 70.0, 0.0, 6.0,
         "D-box area x 1.5 mm x 0.16 g/cm³ + 8 g strips"),
        ("wing: rear carry-through tube 10/8 x 350 (carbon)", "wing/reinforcement/covering", cad["rear_spar"] * 1e-3 * m["carbon tube"]["rho"],
         148.0, 0.0, 9.0, "CAD volume x 1.55 g/cm³"),
        ("wing: covering film", "wing/reinforcement/covering", film_wing, x_wing, 0.0, 6.0, "32 g/m² x wetted area"),
        ("wing: aileron hinges and tape", "wing/reinforcement/covering", 6.0, 190.0, 0.0, 2.0, "assumed"),
        ("pod: LW-PLA shell with nose cone (1.2 mm)", "pod", cad["pod"] * 1e-3 * m["LW-PLA printed"]["rho"], x_pod, 0.0, z_pod, "CAD shell volume x 0.55 g/cm³"),
        ("pod: PETG keel frame and wing saddle (bolts, spar bridge)", "pod", 25.0, 90.0, 0.0, -20.0, "assumed"),
        ("pod: hatch, latches, nose bayonet ring", "pod", 14.0, -150.0, 0.0, -10.0, "assumed"),
        ("pod: electronics tray (PETG)", "pod", cad["tray"] * 1e-3 * petg * PRINT_FILL["tray"], 35.0 if variant == "Zero" else -65.0, 0.0, -65.0, "CAD x 1.25 x fill 0.5"),
        ("pod: battery tray (PETG)", "pod", cad["battery_tray"] * 1e-3 * petg * PRINT_FILL["battery_tray"], -190.0, 0.0, -97.0, "CAD x 1.25 x fill 0.6"),
        ("pod: vents, grommets, foam liner", "pod", 6.0, 150.0, 0.0, -60.0, "assumed"),
        ("booms: carbon tube 10/8 x 580, 2 x", "booms/tail", 2 * booms, 0.5 * (p["boom_x0"] + L["boom_x1"]), 0.0, p["boom_z"], "CAD volume x 1.55 g/cm³"),
        ("tail: foam stabiliser and fins (7 mm plates)", "booms/tail", tail_foam, x_tail, 0.0, 10.0, "CAD volume x 30 kg/m³"),
        ("tail: carbon spar 4/3 x 400", "booms/tail", tail_spar, L["tail_le"] + 0.3 * p["tail_chord"], 0.0, p["boom_z"], "tube section x length x 1.55"),
        ("tail: covering film", "booms/tail", film_tail, x_tail, 0.0, 10.0, "32 g/m² x wetted area"),
        ("tail: boom-end fittings (PETG), 2 x", "booms/tail", 2 * cad["tail_fitting"] * 1e-3 * petg * PRINT_FILL["tail_fitting"], L["tail_le"] + 70, 0.0, p["boom_z"], "CAD x 1.25 x fill 0.6"),
        ("tail: elevator/rudder hinges, torque rod", "booms/tail", 8.0, L["tail_le"] + 100, 0.0, 0.0, "assumed"),
        ("boom root fittings (PETG, clamp rings on both tubes), 2 x", "joints/mounts/adhesive", 2 * cad["boom_fitting"] * 1e-3 * petg * PRINT_FILL["boom_fitting"], 140.0, 0.0, -5.0, "CAD x 1.25 x fill 0.6"),
        ("motor mount (PETG, solid)", "joints/mounts/adhesive", cad["motor_mount"] * 1e-3 * petg * PRINT_FILL["motor_mount"], L["x_pod_end"], 0.0, p["motor_z"], "CAD x 1.25"),
        ("belly keel skid (TPU)", "joints/mounts/adhesive", cad["skid"] * 1e-3 * m["TPU 95A"]["rho"] * PRINT_FILL["skid"], 0.5 * (p["skid_x0"] + p["skid_x1"]), 0.0, -p["pod_height"] - 0.4 * p["skid_depth"], "CAD x 1.21 x fill 0.5"),
        ("GNSS mast, camera cradle, antenna feed-throughs", "joints/mounts/adhesive", 10.0, -150.0, 0.0, -10.0, "assumed"),
        ("adhesive (epoxy, CA, hot glue), screws, inserts", "joints/mounts/adhesive", 20.0, 100.0, 0.0, -20.0, "assumed"),
    ]
    struct_mass = sum(i[2] for i in items)
    x_s = sum(i[2] * i[3] for i in items) / struct_mass
    z_s = sum(i[2] * i[5] for i in items) / struct_mass
    items.append(("construction allowance (8 % of structure and joints: paint, tape, filler)", "construction allowance",
                  ALLOWANCE_FRACTION * struct_mass, x_s, 0.0, z_s, f"{ALLOWANCE_FRACTION:.0%} of the structure, assumed"))
    return items


def mass_table(variant: str, battery_key: str = "gens-ace-3s-2200", *, p=None, cad=None, jetson: str = "nano_b01",
               cg_frac_mac: float = 0.30, battery_x: float | None = None) -> pd.DataFrame:
    """The mass table of one variant [g, mm]: the structure from the CAD, the components of the variant, the pack.
    The battery is placed along its bay so the centre of gravity sits at ``cg_frac_mac`` of the MAC (as builders
    trim a model: the battery sets the CG); if the bay's ends do not allow it, the battery sits at the end and the
    table says so (``attrs['ballast_note']``). ``jetson``: ``JETSON_OPTIONS`` key for Nisus-Zero (``nano_b01`` is the
    reference installation; another key rescales the computer rows' mass to that option)."""
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
    rows = [{"item": i[0], "group": i[1], "mass [g]": i[2], "x [mm]": i[3], "y [mm]": i[4], "z [mm]": i[5], "basis": i[6]}
            for i in structure_items(p, cad, variant)]
    comps = components(variant)
    scale = 1.0
    if variant == "Zero" and jetson != "nano_b01":
        ref = sum(c.mass_g for c in comps if c.group == "computer")
        scale = JETSON_OPTIONS[jetson]["mass_g"] / ref
    for c in comps:
        mg = c.mass_g * (scale if c.group == "computer" else 1.0)
        rows.append({"item": c.key, "group": c.group, "mass [g]": mg, "x [mm]": c.x(variant), "y [mm]": c.y_mm, "z [mm]": c.z_mm,
                     "basis": f"{c.kind}: {c.source[:60]}"})
    df = pd.DataFrame(rows).set_index("item")
    b = battery(battery_key)
    x0, x1, _, _ = nisus.Nisus.bays(p)["battery bay"]
    half = sorted(b.size_mm, reverse=True)[0] / 2
    lo, hi = x0 + 4 + half, x1 - 4 - half
    z_b = -p["pod_height"] + p["pod_wall"] + 14.0 + sorted(b.size_mm)[0] / 2
    note = ""
    if battery_x is None:
        target = L["x_mac_le"] + cg_frac_mac * L["mac"]
        m_rest, mx_rest = df["mass [g]"].sum(), (df["mass [g]"] * df["x [mm]"]).sum()
        battery_x = (target * (m_rest + b.mass_g) - mx_rest) / b.mass_g
        if battery_x < lo:
            note = f"battery at the bay's front ({lo:.0f} mm): the CG is {((mx_rest + b.mass_g * lo) / (m_rest + b.mass_g) - L['x_mac_le']) / L['mac']:.1%} MAC, aft of the target — add nose ballast or move the tray"
            battery_x = lo
        elif battery_x > hi:
            note = f"battery at the bay's rear ({hi:.0f} mm): the CG is {((mx_rest + b.mass_g * hi) / (m_rest + b.mass_g) - L['x_mac_le']) / L['mac']:.1%} MAC, ahead of the target"
            battery_x = hi
    df.loc[f"battery {b.name}"] = {"group": "battery/retention", "mass [g]": b.mass_g, "x [mm]": battery_x, "y [mm]": -22.0 if variant == "Zero" else 0.0, "z [mm]": z_b,
                                   "basis": f"sourced: {b.mass_g:g} g ({SNIPPET})"}
    df.attrs.update(variant=variant, battery=b.key, battery_item=f"battery {b.name}", jetson=jetson if variant == "Zero" else None, ballast_note=note,
                    battery_x_range=(lo, hi), cg_target_frac=cg_frac_mac)
    return df


def budget_comparison(variant: str, **kw) -> pd.DataFrame:
    """Groups of the mass table against the earlier camera-free reference budget; the variant's additions (camera/video,
    computer) have no reference line."""
    t = mass_table(variant, **kw)
    g = t.groupby("group")["mass [g]"].sum()
    rows = {}
    for key in list(REFERENCE_BUDGET_G) + ["camera/video", "computer"]:
        rows[key] = {"reference [g]": REFERENCE_BUDGET_G.get(key, np.nan), f"{variant} [g]": float(g.get(key, 0.0))}
    df = pd.DataFrame(rows).T
    df.loc["total"] = [sum(REFERENCE_BUDGET_G.values()), t["mass [g]"].sum()]
    df["difference [g]"] = df[f"{variant} [g]"] - df["reference [g]"]
    return df


def cg_inertia(table: pd.DataFrame, p=None) -> dict:
    """Mass [kg], centre of gravity [m] and the inertia tensor about it [kg m²] in the aircraft frame (x aft, y
    right, z up). Every row is a point mass at its position; the extended parts (wing core and skins, pod shell,
    booms, tail) add their own inertia about their centroid as simple shapes (thin plates and rods, the pod an
    ellipsoidal shell). ``Ixz`` is the only off-diagonal term (y symmetry)."""
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
    m = table["mass [g]"].to_numpy() / 1000
    r = table[["x [mm]", "y [mm]", "z [mm]"]].to_numpy() / 1000
    M = m.sum()
    cg = (m[:, None] * r).sum(0) / M
    d = r - cg
    I = np.zeros((3, 3))
    for mi, di in zip(m, d):
        I += mi * (np.dot(di, di) * np.eye(3) - np.outer(di, di))
    own = np.zeros((3, 3))
    b, c = p["span"] / 1000, L["mac"] / 1000
    for item, mi in zip(table.index, m):
        if item.startswith("wing:"):
            own += mi * np.diag([b ** 2 / 12, c ** 2 / 12, (b ** 2 + c ** 2) / 12])
        elif item.startswith("pod:") and "shell" in item:
            a_, b_, c_ = p["pod_length"] / 2000, p["pod_width"] / 2000, p["pod_height"] / 2000
            own += mi / 3 * np.diag([b_ ** 2 + c_ ** 2, a_ ** 2 + c_ ** 2, a_ ** 2 + b_ ** 2])    # thin ellipsoidal shell
        elif item.startswith("booms:"):
            Lb = p["boom_length"] / 1000
            own += mi * np.diag([0.0, Lb ** 2 / 12, Lb ** 2 / 12])
        elif item.startswith("tail:"):
            s, ct = p["tail_span"] / 1000, p["tail_chord"] / 1000
            own += mi * np.diag([s ** 2 / 12, ct ** 2 / 12, (s ** 2 + ct ** 2) / 12])
    I += own
    return {"mass_kg": float(M), "x_cg_m": float(cg[0]), "y_cg_m": float(cg[1]), "z_cg_m": float(cg[2]),
            "x_cg_frac_mac": float((cg[0] * 1000 - L["x_mac_le"]) / L["mac"]),
            "Ixx": float(I[0, 0]), "Iyy": float(I[1, 1]), "Izz": float(I[2, 2]), "Ixz": float(-I[0, 2]),
            "inertia": I, "wing_loading_N_m2": float(M * G / L["S_ref"]), "wing_loading_g_dm2": float(M * 1000 / (L["S_ref"] * 100))}


# ================================================================================================= electrical loads
PHASES = ("prelaunch", "launch+climb", "outbound", "survey", "return", "approach+landing")
#: regulator efficiencies by supply branch (assumptions: a switching BEC on 3S to 5.3 V ~ 85 %, the servo rail the same,
#: the Jetson's own 5 V buck 90 %; direct battery loads 100 %)
REGULATOR_EFF = {"battery": 1.0, "5V": 0.85, "servo": 0.85, "jetson": 0.90, "vtx5v": 0.85, "none": 1.0}
#: how much of each phase a load spends at its typical power (the rest at idle): servos move little in cruise, a lot in
#: the launch, the survey turns and the landing; the telemetry and video link transmit all the time; the computer works
#: from the prelaunch checks on. Assumptions.
DUTY = {
    "servo": {"prelaunch": 0.2, "launch+climb": 0.8, "outbound": 0.3, "survey": 0.6, "return": 0.3, "approach+landing": 0.8},
    "default": {ph: 1.0 for ph in PHASES},
}


def electrical_loads(variant: str, jetson_w: float = JETSON_INSTALLATION_W) -> pd.DataFrame:
    """Every electrical load of the variant with its supply voltage, idle / typical / peak power at the device, the
    survey-phase duty cycle, the regulator efficiency and the battery-side typical and peak power. Servo peaks
    (stall) are listed separately from their average. The Jetson installation is one line at the battery side
    (``jetson_w``: the complete original-Nano installation, camera, cooling and conversion included — not added twice);
    the propulsion motor is not here (the propulsion map carries it); the prelaunch camera/computer operation is in
    ``phase_power``."""
    rows = []
    for c in components(variant):
        if c.supply == "none" or c.p_peak_w <= 0:
            continue
        if c.supply == "jetson":
            continue                                     # covered by the installation line below
        eff = REGULATOR_EFF[c.supply]
        duty = DUTY["servo" if c.supply == "servo" else "default"]["survey"]
        p_avg = duty * c.p_typ_w + (1 - duty) * c.p_idle_w
        rows.append({"load": c.key, "supply": c.supply, "V": c.supply_v, "idle [W]": c.p_idle_w, "typical [W]": c.p_typ_w, "peak [W]": c.p_peak_w,
                     "duty (survey)": duty, "regulator eff.": eff, "battery-side average [W]": p_avg / eff, "battery-side peak [W]": c.p_peak_w / eff,
                     "source": ("assumed" if c.kind == "assumed" else "sourced") + ": " + c.source[:90]})
    if variant == "Zero":
        rows.append({"load": "Jetson installation (computer, vision camera, fan, 5 V buck losses)", "supply": "battery (own 5 V/5 A buck)", "V": 11.1,
                     "idle [W]": 0.6 * jetson_w, "typical [W]": jetson_w, "peak [W]": jetson_w * 20 / 18, "duty (survey)": 1.0, "regulator eff.": 1.0,
                     "battery-side average [W]": jetson_w, "battery-side peak [W]": jetson_w * 20 / 18,
                     "source": f"prompt's estimate for the original-Nano installation: {jetson_w:g} W battery side, {JETSON_INSTALLATION_RANGE_W[0]:g}-{JETSON_INSTALLATION_RANGE_W[1]:g} W sensitivity"})
        link = next(c for c in components(variant) if c.key.startswith("data link"))
        rows.append({"load": link.key, "supply": "jetson 5 V", "V": 5.0, "idle [W]": link.p_idle_w, "typical [W]": link.p_typ_w, "peak [W]": link.p_peak_w,
                     "duty (survey)": 1.0, "regulator eff.": REGULATOR_EFF["jetson"], "battery-side average [W]": link.p_typ_w / 0.9,
                     "battery-side peak [W]": link.p_peak_w / 0.9, "source": "assumed: " + link.source[:90]})
    df = pd.DataFrame(rows).set_index("load")
    df.loc["TOTAL (battery side)"] = {"supply": "", "V": np.nan, "idle [W]": df["idle [W]"].sum(), "typical [W]": df["typical [W]"].sum(),
                                      "peak [W]": df["peak [W]"].sum(), "duty (survey)": np.nan, "regulator eff.": np.nan,
                                      "battery-side average [W]": df["battery-side average [W]"].sum(),
                                      "battery-side peak [W]": df["battery-side peak [W]"].sum(), "source": "sum; servo peaks would not all coincide"}
    return df


def phase_power(variant: str, jetson_w: float = JETSON_INSTALLATION_W) -> pd.DataFrame:
    """The electronics' battery-side average power [W] in each mission phase (the duty cycles of ``DUTY``, the Jetson
    on from the prelaunch checks, the video link transmitting throughout, servo averages not peaks)."""
    out = {}
    for ph in PHASES:
        total = 0.0
        for c in components(variant):
            if c.supply in ("none", "jetson") or c.p_peak_w <= 0:
                continue
            duty = DUTY["servo" if c.supply == "servo" else "default"][ph]
            total += (duty * c.p_typ_w + (1 - duty) * c.p_idle_w) / REGULATOR_EFF[c.supply]
        if variant == "Zero":
            total += jetson_w + 2.0 / 0.9 if ph != "prelaunch" else 0.8 * jetson_w + 2.0 / 0.9
        out[ph] = total
    return pd.DataFrame({"electronics battery-side [W]": out})


def regulator_check(variant: str) -> pd.DataFrame:
    """The regulated branches against their ratings: the flight controller's 5.3 V BEC (3 A continuous / 4.8 A peak),
    its servo rail (3 A / 4.8 A), the video transmitter's 5 V output (1 A), the Jetson's buck (5 A). Continuous =
    the sum of typical powers, transient = the sum of peaks (every servo stalled at once is the worst case)."""
    branches = {"5V": ("FC BEC 5.3 V", 5.3, 3.0, 4.8), "servo": ("FC servo rail 5.3/7.2 V", 6.0, 3.0, 4.8), "vtx5v": ("VTX 5 V aux", 5.0, 1.0, 1.0),
                "jetson": ("Jetson 5 V buck", 5.0, 5.0, 6.0)}
    rows = {}
    for key, (name, v, i_cont, i_peak) in branches.items():
        cs = [c for c in components(variant) if c.supply == key]
        if not cs:
            continue
        p_cont = sum(c.p_typ_w for c in cs)
        p_peak = sum(c.p_peak_w for c in cs)
        if key == "jetson":
            p_cont, p_peak = 0.9 * JETSON_INSTALLATION_W, 0.9 * JETSON_INSTALLATION_RANGE_W[1]
        rows[name] = {"loads": len(cs), "continuous [A]": p_cont / v, "rating cont. [A]": i_cont, "margin cont.": i_cont / (p_cont / v),
                      "transient [A]": p_peak / v, "rating peak [A]": i_peak, "margin peak": i_peak / (p_peak / v)}
    return pd.DataFrame(rows).T


def servo_check(p=None, V_max: float = 28.0, deflection_deg: float = 25.0, ch_delta: float = 0.40, linkage_ratio: float = 1.0,
                servo_kgcm: float = 2.0, servo_speed_s60: float = 0.10) -> pd.DataFrame:
    """Hinge moments at the never-exceed speed with full deflection against the servo's torque: ``H = q S_s c_s Ch_δ δ``
    with ``Ch_δ`` ~ 0.4 per rad for a plain flap of 25-35 % chord (a textbook-level estimate, no aerodynamic balance);
    the rudder servo carries both rudders. Speed: 0.10 s per 60° is 600°/s, well above the ~300°/s a fixed-wing
    autopilot's attitude loop wants."""
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
    q = 0.5 * RHO * V_max ** 2
    d = math.radians(deflection_deg)
    c_ail = p["aileron_chord_frac"] * 0.5 * (p["root_chord"] + (p["tip_chord"] - p["root_chord"]) * (L["y_aileron0"] / L["b2"]) + p["tip_chord"]) / 1000
    surfaces = {"aileron (each)": (L["S_aileron_each"], c_ail), "elevator": (L["S_elevator"], p["elevator_frac"] * p["tail_chord"] / 1000),
                "rudders (both on one servo)": (2 * L["S_rudder_each"], p["rudder_frac"] * p["fin_chord"] / 1000)}
    rows = {}
    for name, (S, c) in surfaces.items():
        H = q * S * c * ch_delta * d
        rows[name] = {"area [cm²]": S * 1e4, "hinge moment [N·cm]": H * 100, "servo torque needed [kg·cm]": H / G * 100 / linkage_ratio,
                      "servo [kg·cm]": servo_kgcm, "safety factor": servo_kgcm / (H / G * 100 / linkage_ratio),
                      "rate [deg/s]": 60 / servo_speed_s60}
    return pd.DataFrame(rows).T


# ================================================================================================= propulsion
#: the motor shortlist: published data as found on 2026-10-09 (see data/nisus_component_sources.md); g/W from the
#: published static point where one exists
MOTOR_SHORTLIST = [
    dict(motor="SunnySky X2216 V3 KV1250", kv=1250, mass_g=69.0, max_a=40.0, max_w=400.0, burst="40 A / 30 s", cells="3S",
         point="APC 9x6, 11.1 V: 20.8 A, 231 W, 1040 gf (retailer copy of the maker's table)", thrust_g=1040.0, power_w=231.0,
         fit="28 mm can, 34 mm long; 16/19 mm cross mount; fits the mount cup", url="https://sunnyskyusa.com/products/sunnysky-x2216"),
    dict(motor="T-Motor AT2312 KV1150", kv=1150, mass_g=60.0, max_a=25.0, max_w=350.0, burst="25 A / 180 s", cells="2-4S",
         point="no 3S test row found; recommended AUW 400-900 g (maker)", thrust_g=np.nan, power_w=np.nan,
         fit="28.4 mm can, 44.5 mm long (with shaft); 16/19 mm mount", url="https://store.tmotor.com/product/at2312-long-shaft-fixed-wing-motor.html"),
    dict(motor="T-Motor AT2814 KV1050", kv=1050, mass_g=107.0, max_a=50.0, max_w=700.0, burst="50 A / 180 s", cells="3-4S",
         point="APC 10x5.5, 15.0 V, 65 %: 17.8 A, 267 W, 1347 gf, 8876 rpm (maker's table)", thrust_g=1347.0, power_w=267.2,
         fit="35 mm can: needs a larger mount cup (+38 g over the X2216)", url="https://store.tmotor.com/product/at2814-long-shaft-fixed-wing-motor.html"),
    dict(motor="Emax GT2215/09 KV1180", kv=1180, mass_g=70.0, max_a=26.0, max_w=288.0, burst="26 A / 60 s", cells="3S",
         point="10x6, 3S: ~8850 rpm, 24 A, 1140 gf (old listing; power not stated)", thrust_g=1140.0, power_w=np.nan,
         fit="28.5 mm can; 16/19 mm mount", url="https://www.amainhobbies.com/emax-gt2215-09-1180kv-brushless-motor-emx-mt-0409/p694266"),
    dict(motor="Cobra C-2217/16 KV1180", kv=1180, mass_g=73.0, max_a=24.0, max_w=265.0, burst="24 A continuous", cells="3S",
         point="no static point in the snippets", thrust_g=np.nan, power_w=np.nan, fit="28 mm class", url="https://www.cobramotorsusa.com/motors-2217-16.html"),
    dict(motor="HobbyKing FC 28-22 KV1200", kv=1200, mass_g=39.0, max_a=14.5, max_w=160.0, burst="not stated", cells="3S",
         point="10x5, 3S: 7100 rpm, 14.5 A, 710 gf", thrust_g=710.0, power_w=np.nan, fit="3 mm shaft; light but at its limit", url="https://hobbyking.com/en_us/fc-28-22-brushless-outrunner-1200kv.html"),
]


def motor_shortlist_table(t_max_required_n: float) -> pd.DataFrame:
    rows = []
    for m in MOTOR_SHORTLIST:
        g_w = m["thrust_g"] / m["power_w"] if np.isfinite(m.get("power_w", np.nan)) else np.nan
        rows.append({"motor": m["motor"], "KV": m["kv"], "mass [g]": m["mass_g"], "max A": m["max_a"], "max W": m["max_w"], "burst": m["burst"],
                     "published point": m["point"], "static g/W": g_w, "static T/W (thrust/motor mass)": m["thrust_g"] / m["mass_g"] if np.isfinite(m["thrust_g"]) else np.nan,
                     f"meets {t_max_required_n:.1f} N static": (m["thrust_g"] * G / 1000 >= t_max_required_n) if np.isfinite(m["thrust_g"]) else "no data",
                     "fit": m["fit"], "url": m["url"], "availability": "not confirmed (snippets)"})
    return pd.DataFrame(rows).set_index("motor")


def requirements(mass_kg: float, cd0: float, AR: float, oswald: float, S: float, *, V_cruise=16.0, V_climb=13.0, roc=3.0, V_max=25.0,
                 launch_tw=0.45, gust_margin=0.30, eta_prop_cruise=0.60, eta_prop_climb=0.55, eta_prop_static=0.0, eta_motor=0.80, eta_esc=0.96,
                 rho=RHO) -> pd.DataFrame:
    """What the propulsion must deliver, from the aircraft (mass, polar) and the targets: cruise thrust (= drag),
    the climb (``roc`` at ``V_climb``), the dash at ``V_max``, the launch (static thrust ≥ ``launch_tw`` x weight: a hand
    launch at ~8 m/s accelerates through the stall region only with that much), and the maximum with a margin for
    gusts and turns. Each row: thrust, aerodynamic power (T V), shaft power (/ η_prop), electrical input (/ η_motor η_esc).
    Static thrust has no T V power: its shaft power comes from the propeller model (``propulsion_map``)."""
    W = mass_kg * G
    k = 1 / (math.pi * AR * oswald)

    def drag(V, n=1.0):
        q = 0.5 * rho * V ** 2
        cl = n * W / (q * S)
        return q * S * (cd0 + k * cl ** 2)

    rows = {}
    D_c = drag(V_cruise)
    rows["cruise (level)"] = dict(V=V_cruise, thrust=D_c, eta_prop=eta_prop_cruise)
    D_cl = drag(V_climb)
    rows[f"climb {roc:g} m/s at {V_climb:g} m/s"] = dict(V=V_climb, thrust=D_cl + W * roc / V_climb, eta_prop=eta_prop_climb)
    rows[f"dash {V_max:g} m/s (level)"] = dict(V=V_max, thrust=drag(V_max), eta_prop=eta_prop_cruise)
    rows["climbing turn (n = 1.4) with gust margin"] = dict(V=V_climb, thrust=(drag(V_climb, 1.4) + W * roc / V_climb) * (1 + gust_margin), eta_prop=eta_prop_climb)
    rows[f"launch: static T/W ≥ {launch_tw:g}"] = dict(V=0.0, thrust=launch_tw * W, eta_prop=np.nan)
    out = {}
    for name, r in rows.items():
        P_aero = r["thrust"] * r["V"]
        P_shaft = P_aero / r["eta_prop"] if r["V"] > 0 else np.nan
        out[name] = {"airspeed [m/s]": r["V"], "thrust [N]": r["thrust"], "T/W": r["thrust"] / W, "aero power T·V [W]": P_aero,
                     "η_prop": r["eta_prop"], "shaft power [W]": P_shaft, "electrical [W]": P_shaft / (eta_motor * eta_esc) if r["V"] > 0 else np.nan}
    df = pd.DataFrame(out).T
    df.attrs["T_max_required_N"] = float(max(df["thrust [N]"]))
    df.attrs["T_cruise_N"] = float(D_c)
    return df


def propeller_9x6():
    """The APC 9x6E as Boreas' generic planform (as ``scenarios/run_scenario.air``): constant geometric pitch 152 mm,
    chord 14 → 22 → 6 mm."""
    from vegeta import boreas
    d, pitch = boreas.inches(9, 6)
    return boreas.Propeller.from_pitch("APC 9x6E (generic planform)", d, pitch, blades=2, chord_root_m=0.014, chord_max_m=0.022,
                                       chord_tip_m=0.006, mass_kg=0.012, rotor_mass_kg=0.045,
                                       notes="generic planform for a 9x6 thin-electric; the real blade's chord and twist are not in the model")


def blade_section():
    from vegeta import boreas
    return boreas.Airfoil(name="thin cambered section", cl_alpha=2 * math.pi * 0.9, alpha0_deg=-2.5, cl_max=1.1, cd0=0.02, k=0.04,
                          source="assumed for a 9-inch blade at Re ~ 1.5e5 (notebook 09a's values)")


PUBLISHED_STATIC = dict(voltage=11.1, current=20.8, power_w=231.0, thrust_n=1.040 * G, kv=1250.0,
                        source="SunnySky X2216 KV1250 + APC 9x6 at 11.1 V, retailer copies of the maker's table (2026-10-09 snippets)")


def x2216_model(no_load_a: float = 0.6, eta_motor_guess: float = 0.80, max_current_a: float = 40.0):
    """A first-order motor model of the X2216 KV1250 fitted to the published static point: the propeller model
    gives the rpm at which the shaft power is ``η_motor x 231 W``; the winding resistance then follows from
    ``V = rpm/KV + I R``; ``I0`` is assumed. Returns (motor, fit dict). The thrust the BEMT predicts at that rpm
    against the published 10.2 N is the model's calibration ratio (``thrust_correction``), applied to the map."""
    from vegeta import boreas
    prop, af = propeller_9x6(), blade_section()
    P_shaft = eta_motor_guess * PUBLISHED_STATIC["power_w"]
    lo, hi = 3000.0, 16000.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if boreas.solve(prop, af, mid, 0.0, RHO).power < P_shaft:
            lo = mid
        else:
            hi = mid
    rpm = 0.5 * (lo + hi)
    op = boreas.solve(prop, af, rpm, 0.0, RHO)
    I = PUBLISHED_STATIC["current"]
    R = (PUBLISHED_STATIC["voltage"] - rpm / PUBLISHED_STATIC["kv"]) / I
    R = max(R, 0.02)
    motor = boreas.Motor("SunnySky X2216 V3 KV1250 (fitted)", kv_rpm_per_volt=PUBLISHED_STATIC["kv"], resistance_ohm=R, no_load_current_a=no_load_a,
                         max_current_a=max_current_a, mass_kg=0.069,
                         source=f"KV and the static point published ({PUBLISHED_STATIC['source']}); R fitted, I0 assumed")
    fit = {"rpm_at_point": rpm, "shaft_power_w": op.power, "bemt_thrust_n": op.thrust, "published_thrust_n": PUBLISHED_STATIC["thrust_n"],
           "thrust_correction": PUBLISHED_STATIC["thrust_n"] / op.thrust, "resistance_ohm": R, "no_load_a": no_load_a,
           "motor_efficiency_at_point": motor.efficiency(rpm, op.torque), "ct": op.ct, "cp": op.cp,
           "note": "the correction says how far the generic blade is from the real one at the static point; applied uniformly to the thrust table (assumption)"}
    return motor, fit


PROPULSION_JSON = DATA / "nisus_propulsion.json"


@dataclass
class PropulsionMap:
    """The drive as tables over airspeed ``V`` [m/s] and ``rpm``: thrust [N] (corrected), shaft power [W], torque,
    the motor's current [A] and the voltage it needs [V] (so throttle = V_need / V_battery), electrical power [W].
    Static thrust, in-flight thrust, shaft power and electrical input are separate columns of one table."""
    V: np.ndarray
    rpm: np.ndarray
    thrust: np.ndarray
    shaft_power: np.ndarray
    torque: np.ndarray
    current: np.ndarray
    voltage: np.ndarray
    battery_v: float
    esc_efficiency: float = 0.96
    motor_name: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def electrical(self):
        return self.voltage * self.current / self.esc_efficiency

    def _interp_rows(self, V):
        V = float(np.clip(V, self.V[0], self.V[-1]))
        i = min(int(np.searchsorted(self.V, V, side="right")) - 1, len(self.V) - 2)
        t = (V - self.V[i]) / (self.V[i + 1] - self.V[i])
        return i, t

    def row(self, V, table):
        i, t = self._interp_rows(V)
        return (1 - t) * table[i] + t * table[i + 1]

    def rpm_for_throttle(self, V, throttle):
        """The rpm the motor settles at for a throttle (fraction of the battery voltage) at airspeed V: the rpm
        where the voltage it needs equals the applied voltage (monotone in rpm)."""
        volts = float(np.clip(throttle, 0.0, 1.0)) * self.battery_v
        vn = self.row(V, self.voltage)
        if volts <= vn[0]:
            return float(self.rpm[0]) * volts / max(vn[0], 1e-9)
        return float(np.interp(volts, vn, self.rpm))

    def at(self, V, throttle) -> dict:
        n = self.rpm_for_throttle(V, throttle)
        T = float(np.interp(n, self.rpm, self.row(V, self.thrust)))
        I = float(np.interp(n, self.rpm, self.row(V, self.current)))
        Vn = float(np.interp(n, self.rpm, self.row(V, self.voltage)))
        Ps = float(np.interp(n, self.rpm, self.row(V, self.shaft_power)))
        if throttle <= 0.0:
            return {"rpm": 0.0, "thrust": 0.0, "current": 0.0, "electrical": 0.0, "shaft_power": 0.0, "voltage": 0.0}
        return {"rpm": n, "thrust": T, "current": I, "electrical": Vn * I / self.esc_efficiency, "shaft_power": Ps, "voltage": Vn}

    def max_thrust(self, V):
        return self.at(V, 1.0)["thrust"]

    def throttle_for_thrust(self, V, T):
        """Throttle for a thrust at V (clipped to full throttle)."""
        Tr = self.row(V, self.thrust)
        vn = self.row(V, self.voltage)
        if T <= Tr[0]:
            return float(vn[0] / self.battery_v * max(T, 0.0) / max(Tr[0], 1e-9))
        if T >= Tr[-1]:
            return 1.0
        n = float(np.interp(T, Tr, self.rpm))
        return float(min(np.interp(n, self.rpm, vn) / self.battery_v, 1.0))

    def electrical_for_thrust(self, V, T):
        th = self.throttle_for_thrust(V, T)
        return self.at(V, th)["electrical"] if T <= self.max_thrust(V) + 1e-9 else math.inf

    def table(self, V) -> pd.DataFrame:
        i, t = self._interp_rows(V)
        return pd.DataFrame({"rpm": self.rpm, "thrust [N]": self.row(V, self.thrust), "shaft [W]": self.row(V, self.shaft_power),
                             "current [A]": self.row(V, self.current), "voltage [V]": self.row(V, self.voltage),
                             "electrical [W]": self.row(V, self.electrical), "throttle": self.row(V, self.voltage) / self.battery_v}).set_index("rpm")

    def to_json(self) -> dict:
        return {"V": self.V.tolist(), "rpm": self.rpm.tolist(), "thrust": self.thrust.tolist(), "shaft_power": self.shaft_power.tolist(),
                "torque": self.torque.tolist(), "current": self.current.tolist(), "voltage": self.voltage.tolist(), "battery_v": self.battery_v,
                "esc_efficiency": self.esc_efficiency, "motor_name": self.motor_name, "notes": self.notes}

    @classmethod
    def from_json(cls, d: dict) -> "PropulsionMap":
        return cls(np.asarray(d["V"]), np.asarray(d["rpm"]), np.asarray(d["thrust"]), np.asarray(d["shaft_power"]), np.asarray(d["torque"]),
                   np.asarray(d["current"]), np.asarray(d["voltage"]), d["battery_v"], d.get("esc_efficiency", 0.96), d.get("motor_name", ""), d.get("notes", {}))


def build_propulsion_map(battery_v: float = 11.1, *, V=np.linspace(0.0, 30.0, 13), rpm=np.linspace(2000.0, 14000.0, 13), esc_efficiency=0.96,
                         thrust_correction: bool = True) -> PropulsionMap:
    """Boreas BEMT over (V, rpm) for the 9x6 on the fitted X2216: thrust (times the static calibration ratio when
    ``thrust_correction``), shaft power and torque; the motor model's current and voltage. ~170 BEMT solves."""
    from vegeta import boreas
    prop, af = propeller_9x6(), blade_section()
    motor, fit = x2216_model()
    k = fit["thrust_correction"] if thrust_correction else 1.0
    T = np.zeros((len(V), len(rpm))); P = np.zeros_like(T); Q = np.zeros_like(T); I = np.zeros_like(T); Vn = np.zeros_like(T)
    for i, v in enumerate(V):
        for j, n in enumerate(rpm):
            op = boreas.solve(prop, af, n, v, RHO)
            T[i, j], P[i, j], Q[i, j] = max(op.thrust, 0.0) * k, max(op.power, 0.0), max(op.torque, 0.0)
            Vn[i, j], I[i, j] = motor.voltage_for(n, Q[i, j])
    T = np.maximum.accumulate(T, axis=1) + np.arange(len(rpm)) * 1e-9
    notes = {"propeller": prop.describe()["notes"], "section": af.source, "motor": motor.source, "fit": {k_: (float(v_) if isinstance(v_, (int, float)) else v_) for k_, v_ in fit.items()},
             "battery_v": battery_v, "esc_efficiency": esc_efficiency, "thrust_correction_applied": bool(thrust_correction),
             "limits": "BEMT at axial inflow; no installation (pusher behind the pod: the wake raises the thrust a few % and costs efficiency, notebook 25's finding); "
                       "no battery sag (nominal voltage); no motor heating; the current limit is checked, not enforced"}
    return PropulsionMap(np.asarray(V, float), np.asarray(rpm, float), T, P, Q, I, Vn, battery_v, esc_efficiency, motor.name, notes)


def propulsion_map(path: Path = PROPULSION_JSON, rebuild: bool = False) -> PropulsionMap:
    """The cached map (``data/nisus_propulsion.json``), rebuilt when missing or ``rebuild``."""
    path = Path(path)
    if path.exists() and not rebuild:
        return PropulsionMap.from_json(json.loads(path.read_text()))
    pm = build_propulsion_map()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pm.to_json(), indent=1))
    return pm


def propulsion_points(pm: PropulsionMap, mass_kg: float, cd0: float, AR: float, oswald: float, S: float, *, V_cruise=16.0, V_climb=13.0, roc=3.0, rho=RHO) -> pd.DataFrame:
    """The selected drive at the design points: static full throttle, cruise, climb, full throttle at cruise speed —
    thrust, rpm, current, shaft and electrical power, propeller efficiency, the motor-current check."""
    W = mass_kg * G
    k = 1 / (math.pi * AR * oswald)

    def drag(V):
        q = 0.5 * rho * V ** 2
        return q * S * (cd0 + k * (W / (q * S)) ** 2)

    rows = {}
    st = pm.at(0.0, 1.0)
    rows["static, full throttle"] = dict(st, airspeed=0.0, eta_prop=np.nan)
    for name, V, T in (("cruise (T = D)", V_cruise, drag(V_cruise)), (f"climb {roc:g} m/s", V_climb, drag(V_climb) + W * roc / V_climb)):
        th = pm.throttle_for_thrust(V, T)
        a = pm.at(V, th)
        rows[name] = dict(a, airspeed=V, throttle=th, eta_prop=a["thrust"] * V / a["shaft_power"] if a["shaft_power"] > 0 else np.nan, required=T)
    full = pm.at(V_cruise, 1.0)
    rows[f"full throttle at {V_cruise:g} m/s"] = dict(full, airspeed=V_cruise, throttle=1.0, eta_prop=full["thrust"] * V_cruise / full["shaft_power"])
    df = pd.DataFrame(rows).T
    df["motor current ok (≤ 25 A continuous)"] = df["current"] <= 25.0
    return df


# ================================================================================================= mission energy
@dataclass
class MissionPlan:
    """The sizing mission: ground operation, a hand launch, a climb, the outbound leg to the survey area, a survey
    pattern, the return, approach and landing. ``airborne_s`` is the target (ten minutes including the return);
    the survey time is what is left of it after the climb, the legs and the approach."""
    ground_s: float = 180.0          # electronics on, camera/computer running, GNSS fix, checks
    climb_alt_m: float = 80.0        # survey height above the launch point
    radius_m: float = 600.0          # launch point to the survey area's centre
    airborne_s: float = 600.0        # the ten minutes, launch to touchdown
    V_cruise: float = 16.0
    V_survey: float = 15.0
    V_climb: float = 13.0
    roc: float = 3.0
    approach_s: float = 60.0
    wind_m_s: float = 0.0            # steady wind; the outbound leg flies into it (worst case: the return has it behind? no —
                                     # the return is sized against the wind: see ``return_energy``)
    derating: float = 0.90           # usable share of the nominal capacity: temperature, age, the ESC's cut-off voltage (assumption)
    reserve_frac: float = 0.20       # of the available (derated) energy left after landing
    uncertainty_frac: float = 0.15   # on the predicted return and landing energy
    go_arounds: int = 1              # missed approaches budgeted in the return energy (a gust or an updraft on the final)
    circuit_m: float = 900.0         # one go-around: back to the approach fix and round again at the cruise speed


def _level_power(pm: PropulsionMap, mass_kg, cd0, AR, oswald, S, V, roc=0.0, rho=RHO):
    W = mass_kg * G
    k = 1 / (math.pi * AR * oswald)
    q = 0.5 * rho * V ** 2
    D = q * S * (cd0 + k * (W / (q * S)) ** 2)
    T = D + W * roc / V
    return pm.electrical_for_thrust(V, T), T


def return_energy(pm: PropulsionMap, mass_kg, cd0, AR, oswald, S, distance_m, electronics_w, plan: MissionPlan) -> dict:
    """The energy to come home from ``distance_m`` against the wind at the cruise airspeed (ground speed V - wind),
    descend and land (``approach_s`` at the approach power), ``go_arounds`` missed-approach circuits, plus the
    electronics all along: this is what the return trigger compares against — not a fixed share of the pack."""
    P_cruise, _ = _level_power(pm, mass_kg, cd0, AR, oswald, S, plan.V_cruise)
    gs = max(plan.V_cruise - plan.wind_m_s, 3.0)
    t_back = distance_m / gs
    P_app, _ = _level_power(pm, mass_kg, cd0, AR, oswald, S, 1.3 * 10.0)      # approach near 13 m/s, idle-ish (descent): 60 % of level
    E_back = (P_cruise + electronics_w) * t_back / 3600
    E_land = (0.6 * P_app + electronics_w) * plan.approach_s / 3600
    E_ga = plan.go_arounds * (P_cruise + electronics_w) * plan.circuit_m / plan.V_cruise / 3600
    E = E_back + E_land + E_ga
    return {"t_back_s": t_back, "ground_speed": gs, "E_back_wh": E_back, "E_land_wh": E_land, "E_go_around_wh": E_ga, "E_return_wh": E,
            "E_uncertainty_wh": plan.uncertainty_frac * E}


def mission_energy(variant: str, bat: Battery, pm: PropulsionMap, airframe: dict, plan: MissionPlan = MissionPlan(), *,
                   jetson_w: float = JETSON_INSTALLATION_W, mass_kg: float | None = None) -> dict:
    """The sizing mission's energy, phase by phase, for one variant on one pack. ``airframe``: cd0, AR, oswald, S
    (and mass_kg unless given). Each electrical load is counted once: the propulsion from the map, the electronics
    from ``phase_power``. Returns the phase table [Wh, kWh, % of nominal], the totals, the reserve test, the
    return-trigger energy, the electronics' share, the feasible survey time and the mission radius."""
    m = mass_kg if mass_kg is not None else airframe["mass_kg"]
    cd0, AR, e, S = airframe["cd0"], airframe["AR"], airframe["oswald"], airframe["S"]
    el = phase_power(variant, jetson_w)["electronics battery-side [W]"]
    E_nom = bat.energy_wh
    E_avail = E_nom * plan.derating
    E_reserve = plan.reserve_frac * E_avail
    # phases
    t_climb = plan.climb_alt_m / plan.roc
    P_climb, T_climb = _level_power(pm, m, cd0, AR, e, S, plan.V_climb, plan.roc)
    gs_out = max(plan.V_cruise - plan.wind_m_s, 3.0)
    t_out = max(plan.radius_m - 0.5 * plan.V_climb * t_climb, 0.0) / gs_out
    P_cruise, _ = _level_power(pm, m, cd0, AR, e, S, plan.V_cruise)
    P_survey, _ = _level_power(pm, m, cd0, AR, e, S, plan.V_survey)
    P_survey *= 1.10                                    # turns in the pattern: +10 % (assumption)
    ret = return_energy(pm, m, cd0, AR, e, S, plan.radius_m, el["return"], plan)
    t_survey = plan.airborne_s - t_climb - t_out - ret["t_back_s"] - plan.approach_s
    phases = [
        ("ground operation (prelaunch)", plan.ground_s, 0.0, el["prelaunch"]),
        ("launch + climb", t_climb, P_climb, el["launch+climb"]),
        ("outbound", t_out, P_cruise, el["outbound"]),
        ("survey pattern", max(t_survey, 0.0), P_survey, el["survey"]),
        ("return", ret["t_back_s"], P_cruise, el["return"]),
        ("approach + landing", plan.approach_s, ret["E_land_wh"] * 3600 / plan.approach_s - el["return"], el["approach+landing"]),
    ]
    rows = []
    for name, t, P_prop, P_el in phases:
        E_prop, E_el = P_prop * t / 3600, P_el * t / 3600
        rows.append({"phase": name, "time [s]": t, "propulsion [W]": P_prop, "electronics [W]": P_el, "propulsion [Wh]": E_prop,
                     "electronics [Wh]": E_el, "energy [Wh]": E_prop + E_el, "energy [kWh]": (E_prop + E_el) / 1000,
                     "% of nominal": 100 * (E_prop + E_el) / E_nom})
    df = pd.DataFrame(rows).set_index("phase")
    E_used = df["energy [Wh]"].sum()
    E_el = df["electronics [Wh]"].sum()
    left = E_avail - E_used
    # the return trigger: energy that must remain at the far point of the survey
    trigger = ret["E_return_wh"] + ret["E_uncertainty_wh"] + E_reserve
    # feasible survey time at this radius: everything else fixed, the survey eats what is left above the trigger
    E_fixed = df.loc[["ground operation (prelaunch)", "launch + climb", "outbound"], "energy [Wh]"].sum()
    E_for_survey = E_avail - E_fixed - trigger
    P_survey_total = P_survey + el["survey"]
    t_survey_feasible = max(E_for_survey, 0.0) * 3600 / P_survey_total
    return {"variant": variant, "battery": bat.key, "mass_kg": m, "phases": df, "E_nominal_wh": E_nom, "E_available_wh": E_avail,
            "E_reserve_wh": E_reserve, "E_used_wh": E_used, "E_used_kwh": E_used / 1000, "E_left_wh": left,
            "E_left_pct_available": 100 * left / E_avail, "reserve_kept": bool(left >= E_reserve - 1e-9),
            "electronics_wh": E_el, "electronics_share": E_el / E_used if E_used > 0 else np.nan,
            "return": ret, "trigger_wh": trigger, "trigger_pct_nominal": 100 * trigger / E_nom,
            "comparison_30pct_wh": 0.30 * E_nom, "t_survey_s": t_survey, "t_survey_feasible_s": t_survey_feasible,
            "survey_time_ok": bool(t_survey <= t_survey_feasible + 1e-9 and t_survey > 0), "P_cruise_w": P_cruise, "P_climb_w": P_climb,
            "P_survey_w": P_survey, "electronics_w": dict(el), "plan": asdict(plan)}


def mission_radius(variant: str, bat: Battery, pm: PropulsionMap, airframe: dict, plan: MissionPlan = MissionPlan(), *,
                   min_survey_s: float = 120.0, jetson_w: float = JETSON_INSTALLATION_W, mass_kg=None, limit: str = "time") -> float:
    """The farthest survey-area centre [m] (bisection on the radius). ``limit='time'``: within the plan's airborne time
    (ten minutes) with at least ``min_survey_s`` of survey, landing with the reserve — at short ranges the clock binds
    first. ``limit='energy'``: the airborne time is free (climb, legs, ``min_survey_s`` of survey, the approach), the
    pack's available energy and the reserve bind."""
    def ok(R):
        q = MissionPlan(**{**asdict(plan), "radius_m": float(R)})
        if limit == "energy":
            t_climb = q.climb_alt_m / q.roc
            gs = max(q.V_cruise - q.wind_m_s, 3.0)
            t_out = max(R - 0.5 * q.V_climb * t_climb, 0.0) / gs
            t_back = R / max(q.V_cruise - q.wind_m_s, 3.0)
            q = MissionPlan(**{**asdict(q), "airborne_s": t_climb + t_out + min_survey_s + t_back + q.approach_s})
        r = mission_energy(variant, bat, pm, airframe, q, jetson_w=jetson_w, mass_kg=mass_kg)
        return r["t_survey_s"] >= min_survey_s - 1e-6 and r["survey_time_ok"] and r["reserve_kept"]
    lo, hi = 0.0, 8000.0 if limit == "time" else 40000.0
    if not ok(lo + 50.0):
        return 0.0
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def mission_table(variant: str, pm: PropulsionMap, airframe_fn, plan: MissionPlan = MissionPlan(), batteries=None, *, jetson_w=JETSON_INSTALLATION_W) -> pd.DataFrame:
    """One row per pack: the aircraft's mass with that pack (``airframe_fn(variant, battery_key) -> airframe dict``),
    cruise power, energy used, the reserve test, the electronics' share, the feasible survey time and the mission
    radius — the battery iteration (mass → power → endurance)."""
    rows = {}
    for b in (batteries or BATTERIES):
        af = airframe_fn(variant, b.key)
        r = mission_energy(variant, b, pm, af, plan, jetson_w=jetson_w)
        rows[b.name] = {"pack mass [g]": b.mass_g, "aircraft mass [kg]": af["mass_kg"], "nominal [Wh]": b.energy_wh, "available [Wh]": r["E_available_wh"],
                        "cruise power [W]": r["P_cruise_w"], "electronics [W]": r["electronics_w"]["survey"],
                        "used [Wh]": r["E_used_wh"], "left [% avail.]": r["E_left_pct_available"], "reserve kept": r["reserve_kept"],
                        "electronics share": r["electronics_share"], "survey in 10 min [s]": r["t_survey_s"], "feasible survey [s]": r["t_survey_feasible_s"],
                        "10-min mission ok": r["survey_time_ok"] and r["reserve_kept"],
                        "radius, 10-min mission [km]": mission_radius(variant, b, pm, af, plan, jetson_w=jetson_w) / 1000,
                        "radius, energy-limited (2 min survey) [km]": mission_radius(variant, b, pm, af, plan, jetson_w=jetson_w, limit="energy") / 1000,
                        "fits the bay": battery_fits(b)["fits"]}
    return pd.DataFrame(rows).T


__all__ = ["G", "RHO", "MATERIALS", "PRINT_FILL", "CAD", "cad_numbers", "Component", "COMPONENTS", "components", "component_table",
           "JETSON_INSTALLATION_W", "JETSON_INSTALLATION_RANGE_W", "JETSON_OPTIONS", "REFERENCE_BUDGET_G", "Battery", "BATTERIES", "battery",
           "battery_table", "battery_fits", "structure_items", "mass_table", "budget_comparison", "cg_inertia", "PHASES", "REGULATOR_EFF", "DUTY",
           "electrical_loads", "phase_power", "regulator_check", "servo_check", "MOTOR_SHORTLIST", "motor_shortlist_table", "requirements",
           "propeller_9x6", "blade_section", "PUBLISHED_STATIC", "x2216_model", "PropulsionMap", "build_propulsion_map", "propulsion_map",
           "propulsion_points", "MissionPlan", "return_energy", "mission_energy", "mission_radius", "mission_table"]
