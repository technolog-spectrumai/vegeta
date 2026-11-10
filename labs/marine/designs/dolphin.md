# Dolphin — design brief (v0)

An **autonomous electric submarine for a pearl farm** that works with Walrus (`walrus.md`) and flies the pearls out
with **Emerillion** drones (`labs/aero/designs/emerillion.py`: the X-wing tail-sitter — a slender fuselage with a
canopy, a short straight wing with an 80 mm fan pod at each tip, a cruciform X tail it stands on; it takes off and
lands vertically on its four fins; `emerillion_3d.png`). Family: Scampi's autonomy stack, Walrus's harvest, MERLIN's airframe.

## The loop

1. **Walrus** harvests the farm's bottom-cultured pearl oysters along the lanes and releases its baskets on float lines.
2. **Dolphin** swims the lanes, picks each basket's line with its belly hook into the wet-well; its **pearl cell**
   opens and sorts the oysters, dropping each pearl into a 200 g capsule.
3. When a capsule is full Dolphin **surfaces**, opens the hangar hatch, raises an Emerillion nose-up on the deck pad;
   the drone flies the capsule to the farm's small vessel, lands vertically on the vessel's pad, hands over, flies
   back and **lands vertically** on Dolphin's pad; the lift lowers it, the hatch closes, Dolphin dives.
4. Two drones: one flies while the other charges. The pearls never wait for the boat; the boat never has to find
   the submarine.

## Decisions (v0)

| question | decision |
|---|---|
| what flies | **the pearls, not the oysters**: 200 g of pearls is a day's value and a drone's payload; the shells stay in the wet-well and go back to the farm |
| the drone | **Emerillion**: the X-wing — two 80 mm EDFs (22 N each) in the wing-tip pods so it hovers at 1.44× its 3.1 kg, the four X fins as its legs with the latch socket in the tail, a 0.7 m span that fits the hangar without folding, a 10 000 mAh pack; it cruises at 28 m/s (the little wing stalls at 19) |
| launch and recovery | from the **deck pad behind the sail**, surfaced with 250 mm of freeboard, up to sea state 2; a latch cone on the pad catches the tail socket |
| the hull | a Ø600 × 3.6 m torpedo in GRP, 60 m, ballast tanks that blow 376 kg to lift the deck (395 kg of ballast water aboard when neutral) |
| energy | 12 kWh LiFePO4: a day of 40 km of lanes and 12 h with a 70 % margin, the drones' charges included |
| price | under **€250k** |

## v0 numbers (`python labs/marine/designs/dolphin.py`, `python labs/aero/designs/emerillion.py`)

* Dolphin: **357 kg**, 733 L of hull; 71 N of drag at 1.5 m/s → 195 W to the propeller, 245 W hotel; the day
  **6.4 kWh** of the 10.8 usable (+70 %).
* Emerillion: 3.11 kg with the capsule, T/W 1.44, hovers at 1.27 kW; a 10 km pearl run **13 min**, 105 Wh, reserve
  +70 %; the farthest vessel with a 20 % reserve **19 km**.
* The pearl flow at 300 oysters/h: a 200-pearl capsule every 40 min; a flight is 16 min of drone time plus an hour
  of charging, so the two drones alternating keep up — **18 flights and 3 600 pearls a day**, every pearl Walrus
  brings, 12 min on the surface each time. At 1 500 oysters/h the drones become the limit (a third drone, or a
  bigger capsule).
* The launch in MuJoCo (`labs/marine/sims/dolphin_launch.py`, `dolphin_launch.mp4`): hatch, lift, latch, a hover
  climb to 12 m, the pitch-over at full thrust paced by the airspeed — level at ~20 m/s after 17 s.
* The day in Ketos (`products/pearl` on Ketos's `dev_pearl`, the scenario **ketos pearl**): with akoya oysters at 60 %
  a pearl, one Walrus fills a capsule every ~70 min — 6 flights and 1 200 pearls landed in a 10 h day, the drones
  back with 24 % over the reserve from 10 km and 6 % from 15 km; two Walruses make the 400/h pearl cell the limit.
* Price (class prices): **€195.6k** with two drones.

## Open questions (next round)

1. The pearl cell is the risk: an automated opener and sorter at 300 oysters/h is a machine in its own right — a
   first version that only stores the oysters chilled, with the opening on the vessel, is 22 kg and €28k lighter.
2. Sea state 2 for launch and recovery: a pad gyro-stabilised in roll buys sea state 3 for ~8 kg.
3. One Dolphin per Walrus, or one per two (Walrus's 300 oysters/h fills a capsule every 40 min; two fill one every 20)?
4. The vessel's pad: a tail-sitter landing on a small boat needs the boat to hold still or the drone to track it —
   which?
