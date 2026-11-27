# The surveillance pod for Gannet and Fulmar — design brief (v0)

Gannet (`gannet.py`, the VTOL cargo quad) and Fulmar (`fulmar.py`, the catapult-and-net fixed-wing) carried crates
for Zarco. On the guard carrier Esmeralda (`labs/marine/designs/esmeralda.md`) they carry a **swap-in surveillance
pod** instead of the 5 kg crate. Each pod holds:

- a stabilised day/IR gimbal;
- a laser rangefinder;
- an AIS receiver;
- a video datalink to the carrier.

The payload loader swaps a pod for a crate in the time of a battery swap.

The design is in `eoir_payload.py`, and each drone's numbers with the pod are in `Gannet.eoir_sizing()` and
`Fulmar.eoir_sizing()`, on the drones' own power models. The CAD is the `gimbal` part, drawn instead of the crate when
`payload="eoir"`. The render is `eoir_payload_3d.png` (`eoir_payload_draw3d.py`). The cards (`exports/ketos/gannet.card.json`,
`fulmar.card.json`) keep every cargo number and add `payloads: {cargo, eoir}`.

All numbers come from the sizing. They are **class values and assumptions**: no vendor's equipment.

## Two pods, one per drone

| | Gannet's pod | Fulmar's pod |
|---|---|---|
| gimbal | HD day 1920 px, 30x (2.3–63 deg); **uncooled** LWIR 640, 12.4 deg, 50 mK; laser rangefinder 1.5 km; 50 µrad jitter | HD day 1920 px, 30x (2.3–63 deg); **cooled** MWIR 640, 2.5–25 deg, 25 mK; laser rangefinder 5 km; 30 µrad jitter |
| AIS | dual-channel receiver | dual-channel receiver |
| datalink | 2 W, omni at the carrier: **15 km** (link budget), 8 Mbit/s for a 4 Mbit/s HD stream | 5 W, the carrier's tracking antenna: **60 km** (link budget), 6 Mbit/s for 3 Mbit/s |
| mass, power, price | **2.17 kg, 39 W, EUR 23.8k** | **3.77 kg, 72 W, EUR 85.1k** |

Why two pods: a quad stays minutes near the carrier, where Barracuda and the contacts are, so a light, cheap gimbal is
enough there. The fixed-wing flies the long patrol and searches for what the radar cannot hold, which is where a cooled
core earns its price.

## The drones with the pod

| | Gannet: cargo → pod | Fulmar: cargo → pod |
|---|---|---|
| take-off mass | 16.4 → **13.6 kg** | 20.2 → **19.0 kg** (stall 14.0 m/s, catapult 17.5 m/s) |
| endurance | hover 22.4 → **29.0 min**; loiter at 11.5 m/s **41.6 min**; cruise 38.5 min | cruise 82.4 → 80.6 min; loiter at 17 m/s **150 min** |
| range | 27 → **34.6 km** at 15 m/s | 129 → **126 km** at 26 m/s |
| time on station | **25 min** 5 km from the carrier | **66 min** 30 km from the carrier |
| limits | wind 14 m/s, deck Hs 2.0 m: unchanged (the pod is lighter than the crate) | wind 15 m/s, net Hs 3.0 m: unchanged |
| rain | light to moderate (≤ 4 mm/h, IP54-class, assumed) | the same |

**What limits the operating radius**: Gannet's datalink, not its battery. 15 km at the link budget against its 35 km
range.

## Detection, recognition and identification

These are 50 % ranges by the Johnson criteria, cut by the atmosphere and the horizon (`labs/marine/designs/sensing.py`).
They are **ground ranges** in km from the drone's altitude, written detection / recognition / identification. By day
the table uses the better of the two cameras; at night, the IR.

Sea fog is a **layer 150 m thick** (assumed). Below its top the drone sees through it like a ship. Above it, only the
layer's part of the slant path is in fog, so a drone sees down through fog farther than anything on the sea sees
across it.

**Gannet** (120 m, 300 m):

| | 6 m skiff | 20 m fishing boat |
|---|---|---|
| day, clear | 13.8 / 6.2 / 3.9 | 16.0 / 10.6 / 8.2 |
| day, haze | 3.9 / 2.3 / 1.7 | 14.3 / 4.0 / 2.5 |
| night, clear or haze | 3.9 / 1.0 / 0.6 | 16.0 (14.3) / 4.0 / 2.5 |
| fog, from 120 m (in the layer) | 0.45 / 0.30 / 0.22 | 0.52 / 0.37 / 0.29 |
| fog, from 300 m (above it) | 0.89 / 0.58 / 0.40 | 1.02 / 0.72 / 0.56 |

**Fulmar** (300 m, 900 m):

| | 6 m skiff | 20 m fishing boat |
|---|---|---|
| day, clear | 18.0 / 9.2 / 5.7 | 29.3 / 18.4 / 11.5 |
| day or night, haze | 15.5 / 4.5 / 2.8 | 17.3 / 13.4 / 11.4 |
| night, clear | 18.0 / 4.5 / 2.8 | 29.3 / 18.4 / 11.5 |
| fog, from 300 m | 1.12 / 0.82 / 0.66 | 1.25 / 0.95 / 0.80 |
| fog, from 900 m | **3.35 / 2.45 / 1.97** | 3.75 / 2.85 / 2.39 |

What the ranges show:

- **Gannet at night is short**. The uncooled LWIR at a fixed 12.4 deg recognises a skiff at only 1 km: it confirms
  what the carrier's turret and radar point it at, and does not search.
- **Fulmar's cooled MWIR recognises a skiff at 4.5 km at night**, about four times Gannet.
- **AIS** from altitude: class B craft to ~40 km from Gannet and ~70 km from Fulmar (radio horizon, capped at a class
  value).

## The radar option, for Fulmar only

A lightweight maritime surveillance radar (class: 3.6 kg, 110 W, EUR 90k, a skiff at ~12 km from 900 m) **does not
fit with the gimbal**: 7.4 kg against the 5 kg payload. A **radar-only pod** with AIS, the link and the mount comes to
4.7 kg and would fit, but then that Fulmar finds and cannot look. The choice is the owner's: mix radar and EO/IR
Fulmars, or a bigger aircraft. Neither is designed here.

## Assumed or not modelled

- **Pod numbers**: masses, powers, prices, the link budgets' class ranges, the pod's drag (Gannet: 0.02 m² instead
  of the crate; Fulmar: cd0 −0.002 with the gimbal well) are all class values.
- **Fog**: the 150 m fog layer with a sharp top. Real sea fog's top is diffuse and often lower.
- **Critical dimension**: the boat's side area. From above, the deck is also seen, so the ranges are a little
  conservative.
- **Not modelled**:
  - the gimbal's own stabilisation in gusts beyond its jitter value;
  - rain on the optics (heavy rain behaves like fog);
  - the operator's search pattern;
  - the effect of the payload loader's swap time on the deck's rates.

## The ladder pod, for Gannet (added for the Ketos ladder)

Ketos's careful captain could not light a fast skiff in time from the 12 m workboat, so Gannet gets a third payload:
the **ladder pod** (`eoir_payload.LADDER_POD`, `Gannet.ladder_sizing()`, the card's `payloads.ladder`). It carries a
6000 lm searchlight (6 deg), a white LED strobe (0.5 Mcd, non-laser) and a 130 dB directional loudspeaker on a
pan-tilt. It weighs **3.2 kg**, draws **170 W** and costs **EUR 14.5k**.

| | |
|---|---|
| Gannet with it | 14.6 kg; hover 24.3 min, loiter 33.7 min; **12.8 min hovering on station 5 km out**; cruise 15 m/s; wind and deck limits unchanged |
| searchlight | 0.42 Mcd: 1 lux at **610 m** (530 m in haze) |
| strobe | dazzles to **220 m**, seen at 24 km |
| loudspeaker | understood at **17 m** over a skiff's outboards at speed, **150 m** by a skiff idling, 540 m on a quiet deck |

**What it does for the ladder**:
- **Lights and strobe: yes**, at speed, from about 600 m. A drone can take step 3's lights well before any boat
  arrives.
- **Voice: only once the skiff slows.** Over running outboards a loudspeaker is heard at 17 m, too close for a drone.
- **The hail (step 1)** stays the carrier's VHF.
- **Blocking (step 4)** needs the fast boat, Barracuda (`labs/marine/designs/barracuda.md`, 46.6 kn), not the
  workboat.
