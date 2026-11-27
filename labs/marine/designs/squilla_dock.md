# Squilla's resident dock on the jacket (task 5)

**Squilla** is the resident AUV (`squilla.md`). It lives in **`resident_dock`** (`oil_platform.ResidentDock`), a 3.0 x
1.6 x 1.4 m garage with an inductive charger, on a leg of the 110 m **jacket** (`oil_platform.Jacket`). The carrier
does not launch Squilla.

The check is in `squilla_dock.py` (`report()`), with the render `squilla_dock_3d.png` (`squilla_dock_draw3d.py`). The
`resident_dock` card and mesh were re-exported: the mounting changed. These are **class values, assumed**; the
jacket's geometry is the design's.

## Does it fit?

The dock's envelope is the garage, the charger and bracket, the two collars, and a **12 m approach corridor** in front
of the mouth for Squilla. The mouth faces out from the jacket. I placed it on each of the four legs at every 5 m from
−30 to −80 m, and measured it against every other member (braces, horizontals, conductors, risers, the caisson) and
every anode (`Jacket.members()`).

| leg | fits at | does not fit at |
|---|---|---|
| **A1** (chosen), B1, B2 | −30, −40, −45, −55, −60, −70, −75, **−80 m** | −35 and −65 m (the horizontal frames' levels), −50 m (the leg's own anode) |
| A2 | nowhere | the two risers run down this leg: the garage meets riser R-1 at every depth |

- **At −80 m on leg A1** the least clearance is **8.95 m**, to the X-brace below. The leg's own anode at −87.5 m is
  6.2 m clear of the lower collar.
- **−80 m stays the best depth**:
  - A storm's wave orbital velocity there is 0.06 m/s (Hs 5 m, Tp 10 s), against 0.47 m/s at −30 m.
  - It is clear of the frames and anodes.
  - It is above Squilla's working depth at the mudline (−110 m).

## The clamp-on mounting

No welding. **Two split collars** go round the 2.0 m leg:
- 0.45 m high, 30 mm wall, 1.5 m apart;
- an elastomer liner (2.024 m bore) on a leg cleaned of growth;
- each collar has two bolted half shells with **4 x M30 10.9** studs (280 kN preload each);
- a bracket takes out the leg's 1:10 batter so the garage lies level.

| | |
|---|---|
| loads | dock 15.4 kN in water, 3.2 kN drag at 1 m/s of current (plus the storm's orbital), 3.25 m lever from the leg's axis |
| clamp force | 418 kN normal force a collar (friction 0.2, safety factor 2) against 1120 kN of preload |
| mass | collars, bolts and flanges **1644 kg**, on top of the garage's 1800 kg |
| install | the halves lowered from the DP vessel and closed by the work-class ROV (Homarus), the studs tensioned hydraulically |
| cathodic protection | bonded to the jacket's cathodic protection through the collars' straps, plus two anodes on the garage |

## Power and data

- **Charging**: 2 kW, inductive, with no wet-mate pins. Squilla's 4.5 kWh pack (3.8 kWh usable) fills in **2.2 h**.
  - A guard patrol round the jacket (236 W) lasts **16.2 h** on a charge, so Squilla is out of the dock **87 %** of
    the time.
  - Intervention work (650 W) empties the pack about 3x faster.
- **To the topside**: a 160 m armoured hybrid cable up the leg in clamps, with 2 single-mode fibres and 2 x 6 mm²
  copper.
  - Power is 400 V DC, 2.5 kW, with ~40 W lost in the cable.
  - Data is **1 Gbit/s Ethernet** over the fibre.
- **Docked**: an optical free-space modem at the coupler, ~100 Mbit/s over ~1 m, carries video and logs off and
  mission plans on.
- **Away**: an acoustic modem and a USBL head on the dock reach ~5–10 kbit/s to ~2 km and give Squilla's position to
  the carrier and the topside.

## Docking cycle

| | |
|---|---|
| dock | **157 s**: home on the dock's USBL to 30 m off; hover and line up 10 m off, tail to the mouth, into the current (60 s); back in 10 m at 0.15 m/s along the guide lights (67 s); latch, seat on the coupler, optical link (30 s) |
| undock | **23 s** |
| current limits | head-on 1.0 m/s (Squilla's surge); **across the mouth 0.42 m/s** (70 % of its 200 N of lateral thrust against its broadside drag) |

A rotating tide crosses a fixed mouth twice a tide. Docking waits for the cross component to drop under 0.42 m/s;
North Sea tides at 80 m are ~0.3–0.6 m/s (assumed).

## Assumed or not modelled

- **Assumed**:
  - growth of 50 mm on the leg below −40 m;
  - friction 0.2 on the liner;
  - the bolt preload;
  - the charger's efficiency (0.9);
  - the docking times and the 70 % of lateral thrust usable;
  - the tidal currents.
- **Not modelled**:
  - the collars' fatigue under the leg's own wave-induced bending;
  - scour and debris at the mouth;
  - a fixed diver-detection sonar on the dock (Esmeralda's brief suggests it; the dock has room on its bracket).
