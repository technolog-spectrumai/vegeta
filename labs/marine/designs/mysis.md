# Mysis — design brief (v0)

Named after the mysid shrimp, which live in swarms. Mysis is a **small portable submarine for maritime maintenance
that works in groups**: one person carries a unit in its case; four to six of them go onto one ship's hull, a pier
or a mooring field under one surface node and one supervisor. Family: Scampi's autonomy stack (`scampi.md`),
Echensis's hull navigation (`echensis.md`) scaled down, Shrimp's "the structure is its own ruler" (`shrimp.md`).

## Decisions (v0)

| question | decision |
|---|---|
| jobs | hull **grooming** (light fouling, early and often, no capture — the heavy cleaning with capture stays with Echensis), hull and appendage **inspection** (photo, anodes, CP by contact probe, thruster tunnels, sea-chest grates), intake grate clearing, mooring chain and pile inspection |
| link | **untethered**: a USBL fix and a 5 kbps acoustic modem from the surface node; dead reckoning on the hull between fixes; the units share the hull map and take lanes from it |
| on the hull | **a magnetic roller pad** under the belly (NdFeB blocks on three rollers, 40 N on steel): the brush's press comes free, the vertical thrusters only land and lift off. The first sizing pressed with the thrusters: 380 W for the press alone, 36 minutes a battery — the fleet would have spent two thirds of its time charging |
| the node | on the workboat or a buoy: the USBL head, the modem, GPS and 4G, the group's computer, **two charging bays** — the group rotates through them |
| compute | a Jetson Orin NX per unit (the hull-following and the lanes are local; the modem is too thin for video) |
| price | under **€20k** a unit, **€15k** the node |

## v0 numbers (`python labs/marine/designs/mysis.py`)

* The unit: 620 × 360 × 240 mm round one Ø130 × 400 mm tube; **18.0 kg** in air (a 25 kg case), 0.6 kg of lead to
  trim; six T200-class thrusters: 85 N forward, **1.65 m/s**; holds on the hull in 0.5 m/s with a ×5 margin.
* Grooming: a 0.35 m brush at 0.4 m/s with 15 % lane overlap and 75 % lane efficiency, **320 m²/h**; 416 W on the
  hull → **1.1 h** a battery (500 Wh); surveying 3.3 h.
* **The group**: with two bays on the node, a unit spends 46 % of its time on the hull (1.1 h on, 1.25 h charging),
  so six units keep about three at work: **890 m²/h**. A 180 m bulker's wetted hull (9 000 m², 169 lanes) takes
  **10 h: one shift** with six, 15 h with four, 60 h alone. The acoustic link carries six units' reports (0.5 Hz,
  100 bytes) with a ×2 margin; a USBL fix every 6 s and 5 cm of dead-reckoning drift between fixes.
* Price (class prices): **€19.1k** a unit; **€14.0k** the node; the fleet of six **€128k**.

## What the group buys

One unit is a 60 h job; six are a shift, with the same supervisor and the same boat. The lanes are the unit of work:
each unit takes the next free lane from the shared map, reports it done, and the map is the record the ship's
superintendent gets. A unit that fails lifts off, surfaces and strobes; its lane goes back into the pool.

## Open questions (next round)

1. Steel only? Aluminium and composite hulls need suction instead of the magnets (Echensis's trade): a second pad?
2. Grooming without capture: which ports allow it (light slime, early) and which don't — the first customers decide
   whether a small capture bag on the brush is needed.
3. The node on the workboat or a free buoy — a buoy lets the group work while the boat is away.
4. Ship inspection as the first product (no brush, no pad: 15 kg, €17k) and grooming as the second?
