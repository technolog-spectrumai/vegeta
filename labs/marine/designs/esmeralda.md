# Esmeralda — design brief (v0)

Esmeralda is the **guard-duty variant of Zarco** (`zarco.md`, the unmanned drone-carrier catamaran, formerly
Pinna). It is the carrier of the sea game: on anti-piracy guard at an offshore platform, at night, with air drones,
the interceptor boat Barracuda (`barracuda.md`) and, at the platform, the resident AUV Squilla. The design, sizing
and CAD are in `esmeralda.py` (a subclass of `Zarco`); the render is `esmeralda_3d.png` (`esmeralda_draw3d.py`);
the card is `exports/ketos/esmeralda.card.json`, next to Zarco's, which is unchanged.

All numbers below come from `python labs/marine/designs/esmeralda.py`. They are **class values and assumptions**,
as in Zarco.

## What is kept from Zarco, unchanged

- the hulls: 24 x 10 m, 108 t design displacement;
- the propulsion: 2 x 250 kW gensets, 2 x 280 kW pods, a 350 kWh battery, 10 kn service and 14.6 kn max;
- the autonomy mast, with its X- and S-band radars, AIS, lidar ring and day/IR camera ring;
- the flight deck: 4 VTOL pads, the catapult and the recovery net, closed above Hs 2.0 m;
- the swap robot, the deck tug and the cleaner. The crate loader stays as the **payload loader**, which swaps an
  EO/IR pod or a cargo crate.

## What changes

| | Zarco | Esmeralda |
|---|---|---|
| aft deck | 4 AHC Walrus davits, tether reels, basket gantry, USBL | **the boat bay**: a 3.6 m slot down the centreline from the transom 14 m forward, under the flight deck, with a **13.5 m stern ramp at 8 deg**, a capture hook, a 15 kN winch and a cradle |
| amidships | the processing house, packer, chiller, 2 t crate store | gone; the hangar's centre is the boat bay |
| hangar | 8 Gannet, 3 Fulmar, 24 + 9 packs, 12 chargers | two 2.5 m side bays. Port: **4 Gannets** with the EO/IR payload, plus the payload workshop. Starboard: **2 Fulmars**, wings along the bay, plus the battery room. 12 + 6 packs, **5 chargers** |
| effectors | horn, lights | a **160 dB hailer** and **2 x 35 000 lm searchlights** on the mast, a **3000 L/min bow water cannon** |
| sensors | radars, AIS, lidar, ring cameras | the same, plus a **long-range EO/IR turret** on the mast at 9 m |
| mass | 96.75 t | **94.0 t** |
| draught | 1.39 m | **1.36 m** |
| trim | level (by assumption) | **0.3 deg by the stern** (0.12 m); moving the fuel's centre 2.1 m forward levels it |
| GM transverse | 12.5 m | **13.0 m** (the boat sits low and on the centreline) |
| station load | 82 kW, 15.6 days | **60.2 kW, 21.2 days**; peak 171 kW with the cannon, lights, hailer and ramp running (gensets 500 kW) |
| price | EUR 4.15 M | **EUR 4.83 M**; EUR 5.77 M with Barracuda |

## The boat bay

- **Barracuda's stowed position**: the bay starts at the transom (x = −12 m) and the boat sits on the ramp's top with
  its centre at x = −3.4 m.
  - Its keel is 0.27 m above the waterline at the transom and 1.54 m at the bow, so the ramp is longer than
    Barracuda's 11 m minimum to stow the boat clear of the water.
  - There is **4.2 m of clear height** under the flight deck, against the 2.9 m it needs with its mast folded. With
    the mast up it would reach 6.8 m and hit the flight deck's underside at 5.7 m: **the mast folds in the bay**.
- **Fit between the hulls**: the ramp is 3.4 m wide in a 4.4 m tunnel.
- **Structure**: the cross deck is cut for the bay. The hangar's walls and the flight deck bridge the slot as a box
  girder; its **3.0 t** is counted (assumed).
- **Ramp system**: **3.6 t**. Launch takes **90 s**, recovery **240 s** with refuelling at 100 L/min and shore power,
  limited to **Hs 1.95 m** (Barracuda's ramp numbers). The fuel comes from the ship's 9 t.
- **Operating note**: the net's Fulmar recovery and the boat's ramp recovery both want particular headings (into the
  wind, into the sea). They are not run together.

## Effectors

| effector | numbers |
|---|---|
| long-range acoustic hailer | 160 dB at 1 m, 15 deg beam: understood at **370 m** over a fast skiff's outboards, **1.3 km** by an idling skiff, 2.3 km on a quiet deck; audible to 3 km |
| 2 searchlights | 35 000 lm in a 2.5 deg beam each, 14 Mcd: 1 lux at **2.8 km** in clear air, 1.8 km in haze |
| bow water cannon | 3000 L/min at 12 bar, sea water, from its own sea chest; 48 m/s at the nozzle, **79 m throw**, 2.5 kN reaction; 84 kW electric pump off the gensets' bus |
| flare / marker launcher | **not fitted. A decision for the owner.** Pyrotechnics from a civilian unmanned vessel are a rules-of-engagement question for the flag state and the platform's operator, and distress signals are reserved by COLREGs Annex IV. A 60 kg foundation is reserved on the mast house. A non-pyrotechnic alternative is self-locating datum marker buoys (a light and an AIS/GNSS beacon) to mark a contact's track; it is also not fitted until agreed. |

## Sensors

**The long-range EO/IR turret**, at 9 m: HD day 1920 px at 0.5 deg narrow, cooled MWIR 1280 px at 0.9 deg narrow,
a 20 km laser rangefinder, 10 µrad stabilisation.

| target | condition | detect | recognise | identify |
|---|---|---|---|---|
| 6 m skiff | clear, day or night | 15.7 km (horizon) | 15.7 km | **13.2 km** |
| 6 m skiff | haze | 15.5 km | 11.7 km | 9.7 km |
| 6 m skiff | fog (0.5 km visibility) | **0.58 km** | 0.43 km | 0.36 km |
| 20 m fishing boat | clear | 20.9 km | 20.9 km | 19.4 km |
| 20 m fishing boat | haze | 17.3 km | 13.4 km | 11.4 km |
| 20 m fishing boat | fog | 0.64 km | 0.50 km | 0.42 km |

Fog takes all the optics down to about 0.6 km. Only the radar sees past that.

**Zarco's camera ring** (kept; day 4K and uncooled LWIR, 60 deg each), for a skiff:
- by day: detect 4.9 km, identify 0.8 km;
- at night: detect 0.8 km, identify 0.13 km (an uncooled LWIR core); in fog, detect 0.47 km.

It is for the deck, collision avoidance and close-in work.

**Radars**, at 8.5 m:

| radar | skiff, Hs 0.5 m | skiff, Hs 2.5 m | fishing boat |
|---|---|---|---|
| X-band | 7.5 km | 4.3 km | 22 km to 14 km |
| S-band | 6.4 km | 3.7 km | 20 km to 12 km |

The S-band is for rain, which is not modelled.

**AIS**: class A ships at 29 km, class B craft at 15 km.

**Lidar ring**: a skiff at 200 m, a swimmer at 60 m (class).

**Diver-detection sonar (considered, not fitted)**:
- Its class numbers are an open-circuit diver at 700 m, a closed-circuit diver at 350 m and a small UUV at 400 m. On
  DP beside a steel jacket, roughly half that.
- Fitting it would cost EUR 380k, 0.6 t and 1.5 kW.
- **Not worth it on the carrier**: the carrier moves with the guard picture and works on noisy DP next to a structure
  that clutters it, and the underwater layer is Squilla's. If the platform's owner wants diver detection, it belongs
  fixed on the jacket (task 5).

## The hangar and the deck

- **Chargers**: 3 for the Gannets and 2 for the Fulmars, 7.2 kW, sized to keep one Gannet and one Fulmar with the
  surveillance pod on station continuously (42 and 150 min a sortie, task 3): the packs a sortie uses an hour times a
  1.1 h charge, plus one spare.
- **Fit**: a Gannet (1.79 m square) fits the 2.5 m side bay, four of them in 8.4 m of its 12 m. A Fulmar stows with
  its 3.6 m wing along the bay and its 2.1 m fuselage across it.
- **Rates**: unchanged from Zarco's pads, catapult and net: 240 VTOL launches and 84.7 recoveries an hour, 15
  fixed-wing launches and 12 recoveries an hour. The six surveillance pods (task 3, `labs/aero/designs/eoir_payload.py`:
  4 Gannet pods at EUR 23.8k, 2 Fulmar pods at EUR 85.1k) cost EUR 265k and weigh 16 kg.

## Assumed or not modelled

- **Weight centres**: each weight's centre is a class estimate from the CAD layout.
- **Trim**: the hull's LCB is taken where Zarco's loaded LCG is (Zarco level). The waterplane coefficients (0.78 area,
  0.055 longitudinal inertia per hull) are class values.
- **Not modelled**:
  - the stability of the box girder that bridges the bay's slot (its mass only);
  - seakeeping beyond Zarco's `SEA_RESPONSE`;
  - waves running up the ramp into the bay above Hs ~1.5 m;
  - the water cannon's reaction on DP (2.5 kN, small against the pods');
  - rain on the radars and cameras.

## Open for the owner

- the flare or marker launcher;
- the legal basis for blocking and for the water cannon (Barracuda's question too);
- a fixed diver-detection sonar on the jacket.
