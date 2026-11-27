# Barracuda — design brief (v0)

Named after *Sphyraena barracuda*: long, fast, a patient shadower that closes in a burst. Barracuda is the
**unmanned interceptor boat of the guard carrier Esmeralda** (`esmeralda.md`, a guard-duty variant of Zarco). Near an
offshore platform at night it finds, shadows and blocks small fast craft (skiffs). It climbs an escalation ladder
under a remote operator:

1. hail;
2. shadow;
3. lights and loudhailer;
4. block;
5. water cannon.

It does not fight. The design, sizing and CAD are in `barracuda.py`; the render is `barracuda_3d.png`
(`barracuda_draw3d.py`); the Ketos card is `exports/ketos/barracuda.card.json`. Sensor and effector physics are in
`sensing.py`, which the carrier and the drones share.

All numbers below come from `python labs/marine/designs/barracuda.py`. They are **class values and assumptions**:
no vendor's boat or equipment.

## Decisions

| question | decision |
|---|---|
| hull | an 8.4 m aluminium deep V (24 deg deadrise at the transom, a 2.0 m chine beam), the survey boat's lofted hull at full size, with a **closed-cell foam collar** (Ø 0.48 m). A RIB's collar that cannot be punctured: it is the fender it blocks with. 8.98 m x 2.88 m overall. |
| propulsion | **twin high-speed diesels (2 x 220 kW) on twin waterjets** with reversing buckets: no propeller near people or lines when it blocks, the buckets for the crash stop, one engine and one jet for the loiter. The diesel ratings are steps; the sizing picks the smallest pair that drives the full-load boat at the design sprint (38 kn). Hybrid was not taken: the loiter on one diesel already gives 41 h. |
| water cannon | driven by the starboard engine's front PTO through a clutch, drawing sea water from its own sea chest |
| sensors | a folding mast (4.0 m above the waterline): X-band solid-state radar; a gimbal with an HD day zoom, a cooled MWIR 640 and a laser rangefinder; AIS; VHF with DSC; an 8-mic array |
| effectors | a 150 dB hailer, a 14 000 lm searchlight and a white LED strobe, all on a pan-tilt head slaved to the gimbal; the water-cannon monitor on the foredeck |
| launch and recovery | a **stern ramp between the carrier's demihulls**: the bow eye goes into a capture hook, a winch hauls the boat up 8 deg onto a cradle with keel guides, then it is refuelled and put on shore power. No crane: no-one is aboard to hook on. |
| g limits | impacts averaged over the 1/10 highest at the CG: **2.0 g sustained, 3.0 g in a sprint**. There is no crew to limit it (crewed practice is 1.0–1.5 g); the equipment is shock-mounted and the hull's fatigue sets the limit. These are assumptions, and the speed with a crew's 1.5 g is shown too. |

## v0 numbers

### Mass

| | |
|---|---|
| lightship | 3.12 t: hull 0.89 t, collar 0.17 t, engines 0.77 t, jets 0.15 t, cannon 0.18 t, console and mast 0.19 t, the rest; 8 % margin |
| fuel | 320 L, the tank sized by the 150 nm cruise range + 10 % |
| full load | **3.48 t**, draught 0.63 m, freeboard 0.77 m |

### Speed

Calm-water resistance is Savitsky's planing method, with a class curve below the hump. On top of it: a 1.20 service
margin, added resistance in waves (+0.35 Hs/b) and air drag. Jet efficiency is 0.43 at 8 kn and 0.62 at 35 kn
(class curve).

| Hs (m) | sustained (kn) | sprint (kn) | with a crew, 1.5 g (kn) | what limits it |
|---|---|---|---|---|
| 0 | 46.9 | 46.9 | 46.9 | propulsion |
| 0.5 | 45.3 | 45.3 | 40.0 | propulsion |
| 1.0 | **29.3** | 43.9 | 22.3 | slamming (sustained), propulsion (sprint) |
| 1.5 | 21.1 | 30.9 | 17.5 | slamming |
| 2.0 | 17.8 | 23.7 | 15.3 | slamming |
| 2.5 | 15.9 | 20.1 | 13.8 | slamming |

- **Sprint**: 46.6 kn at full load in calm water. The brief asks for ≥ 35 kn.
- **Cruise**: 25 kn in Hs 1.0 m is met at the 2.0 g sustained limit. With a crew's 1.5 g it would not be (22.3 kn).
- **What limits it**: from Hs 1 m up it is slamming, never the power or the hull's strength. The engines alone would
  still drive it at 40 kn in Hs 2.5 m.
- **Where the model is extrapolated**: Savitsky and Brown is fitted to H/b ≤ 0.7 and V/√L ≤ 6 (knots, feet). Hs
  1.5–2.5 m and the high speeds at 0.5 m are extrapolated, and the card flags each row.

### Fuel and endurance

At half load.

| band | kW | L/h | range or endurance |
|---|---|---|---|
| drift (one engine idling to charge) | 2.3 | 2.0 | 143 h |
| loiter, 8 kn, one engine | 21 | 7.0 | **41.6 h**, 333 nm |
| below the hump, 12 kn | 79 | 24.3 | 144 nm (avoid this band) |
| cruise, 25 kn, Hs 1 m | 178 | 47.7 | **152 nm** |
| sprint, 38 kn | 273 | 68.4 | 4.3 h |
| water cannon | 35.7 | +9.2 | on top of the speed's fuel |

### Manoeuvres

- **Turning circle**: 20 m below 10 kn (2.2 lengths: the jets vector), 43 m at 20 kn, 97 m at 30 kn. The limit is a
  0.5 g sustained turn: a deep V banks, and above that it chine-walks and risks tripping.
- **0–30 kn**: 5.2 s over 44 m.
- **Crash stop**: from 46.9 kn in 5.8 s and 67 m (7.4 lengths), with the buckets at 45 % of the 22 kN bollard thrust.
- **Station keeping beside a target**: holds a 2 m offset on a relative fix of ±0.5 m (laser range and bearing, radar
  track, our RTK).
- **Pacing a target**: it can pace one up to 2 kn under its sustained speed: 27 kn in Hs 1 m, 14 kn in Hs 2.5 m.
- **Blocking**: it can only get ahead of a target 5 kn slower than its sprint: a 39 kn target in Hs 1 m, 15 kn in
  Hs 2.5 m.

### Sensors

The mast is at 4.0 m. Ranges are 50 % Johnson ranges, cut by the atmosphere and the horizon; the gimbal's jitter is
25 µrad.

**EO/IR gimbal**: day HD 1920 px at 2.0 deg narrow (30x); MWIR 640 px at 2.2 deg narrow (11x).

| target | condition | detect | recognise | identify |
|---|---|---|---|---|
| 6 m skiff | day, clear | 11.9 km (horizon) | 9.2 km | 6.8 km |
| 6 m skiff | night, clear or haze | 11.9 km (horizon) | 5.2 km | 3.2 km |
| 6 m skiff | fog (0.5 km visibility) | 0.58 km | 0.43 km | 0.36 km |
| 20 m fishing boat | clear | 17.0 km (horizon) | 17.0 km | 13.2 km |
| 20 m fishing boat | haze | 17.0 km | 13.4 km | 11.4 km |
| 20 m fishing boat | fog | 0.64 km | 0.50 km | 0.42 km |

**Other sensors**:
- **Laser rangefinder**: 10 km.
- **Radar** (10 m² at 9 km in calm sea, class): a skiff (≈ 1 m²) at 4.8 km in Hs 0.5 m, falling to 2.8 km in Hs 2.5 m
  through sea clutter; a fishing boat at 15 km to 8.8 km. Rain is not modelled.
- **AIS**: class A ships at 25 km, class B craft at 15 km.
- **VHF**: 16 km to a small craft.
- **Microphones**: an outboard at speed at 770 m and one idling at 100 m, but only when Barracuda drifts or loiters;
  its own engines at speed drown them (31 m). They also listen for the reply to the hailer.

### Effectors (the ladder)

| step | effector | numbers |
|---|---|---|
| 1 hail | VHF DSC and voice on ch 16, an AIS safety message | 16 km |
| 2 shadow | the boat itself | paces a target as above |
| 3 lights | searchlight, 14 000 lm, 4 deg | 2.2 Mcd: 1 lux at 1.3 km in clear air, 1.0 km in haze |
| 3 lights | strobe, white LED, non-laser, 2 Mcd peak, 5 ms at 6 Hz | dazzles (≥ 10 lux at the eye) to 430 m; seen at 29 km |
| 3 loudhailer | 150 dB at 1 m, 2 kHz, 30 deg beam | understood at 150 m over outboards at speed, 750 m by a skiff idling, 1.6 km on a quiet deck; audible at 2.3 km |
| 4 block | the foam collar | absorbs 5.6 kJ with 0.22 m of crush: parallel contact with a 1.5 t skiff at under 6.4 kn closing; never bow-on |
| 5 water cannon | 1500 L/min at 10 bar, sea water | 44 m/s at the nozzle, **59 m throw**, 1.1 kN reaction, 25 kW hydraulic / 36 kW at the PTO |

### Autonomy and links

- **COLREGs**: rules 8–17 on fused tracks (radar, EO/IR, AIS), with a 50 m CPA while shadowing. Blocking breaks
  COLREGs on purpose: only the operator can suspend them, inside a geofence around the protected platform. **The
  legal basis is not settled** (flag state, the operator's authority).
- **Links**:
  - to the carrier: MANET at 4.9 GHz for video plus a UHF command link, line of sight, about 20 km (radio horizon 21
    km between the carrier's 10 m mast and its own 4 m mast);
  - anywhere: LEO broadband satcom (~50 ms latency), with an L-band backup (~0.7 Mbit/s, ~1 s).
- **Lost link**:
  - 0–10 s: it holds; the water cannon and the strobe stop at once, the hailer stops, the lights stay on.
  - 10–60 s: it opens to ≥ 100 m from any contact, keeps shadowing, and switches to satcom.
  - after 5 min with no link at all: it returns to the carrier's rendezvous at 15 kn under COLREGs.
  - It never climbs above step 1 without a live operator.

### Carrier integration

These are the requirements on Esmeralda:

- a stern ramp 11.0 m long and 3.4 m wide at 8 deg between the demihulls. Zarco's tunnel is 4.4 m wide.
- 2.9 m of clear height over the cradle with the mast folded.
- a 15 kN haul winch: it needs 8.1 kN for this boat and can take up to 6.4 t.
- the system weighs 2.9 t, costs EUR 260k and draws 22 kW.
- **launch: 90 s**: release the lashings, slide down the ramp, jets on.
- **recovery: 240 s**: line up astern at 4 kn, the hook takes the bow eye, haul in, cradle, lash, refuel at
  100 L/min, couple shore power.
- **Sea-state limit: Hs 1.95 m.** The boat's heave against the ramp's end must stay within 1.0 m (assumed). The
  ramp's end moves with the carrier's heave and pitch (`zarco.SEA_RESPONSE`, 11 m lever), taken as 80 % correlated
  with the boat's heave. This is the same as the carrier's 2.0 m deck closure for drones.

### Price

**EUR 0.94 M**: the EO/IR gimbal 185k, hull and build 165k, autonomy computer and software 120k, certification 90k,
engines 84k, jets 48k, the rest.

## What the brief asked for that it misses, or that is not modelled

- **Cruise in Hs 1.0 m**: it reaches 25 kn only because it carries no crew (2.0 g). A crew limit (1.5 g) would give
  22 kn.
- **Sea state 4**: it is not fast there. It does 16–20 kn in Hs 2.5 m, slamming-limited. A skiff slams too (task 4
  will say how much), but **in Hs ≥ 1.5 m Barracuda cannot get ahead of a target faster than ~26 kn to block it**.
- **Launch and recovery above Hs 1.95 m**: it can still run there, but it cannot be recovered.
- **Not modelled**:
  - spray and broaching in following seas;
  - seakeeping beyond the CG's impacts (bow accelerations, which are higher);
  - rain on the radar and the cameras;
  - the strobe's eye safety, to be checked against IEC 62471 at the closest range it is used;
  - the hailer's and the strobe's real beams;
  - the planing hump as anything more than a class curve;
  - the carrier ramp's actual motion: the relative-motion tolerance is assumed, not simulated.
- **Open questions for the owner**:
  - the name (a working name);
  - the legal basis for blocking and for a water cannon from an unmanned civilian vessel;
  - whether it should get a small battery for silent shadowing at night: its diesels are heard (≈ 85 dB at its own
    mast).
