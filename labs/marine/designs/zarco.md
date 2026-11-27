# Zarco — design brief (v0)

Formerly Pinna (renamed 2026-11).

Zarco is the **unmanned, robot-run drone-carrier catamaran** for the clam bed (Ketos Clams stages 6–8). It replaces
the crewed `clam_boat`. It lies over the bed on dynamic positioning and runs four Walrus crawlers on
motion-compensated davits and tether reels. A robot hauls the baskets; the clams are graded and packed into 5 kg iced
crates, and the crates fly ashore with Gannet cargo quads (four VTOL pads) and Fulmar fixed-wings (a pneumatic
catapult and a recovery net). No crew is aboard at any time; one operator ashore supervises it, and others, over
satcom, 4G and UHF. The design, its sizing and its CAD are in `zarco.py`; the render is `zarco_3d.png`
(`zarco_draw3d.py`); the MuJoCo launch clips are in `labs/marine/sims/zarco_launch`.

## v0 numbers (`python labs/marine/designs/zarco.py`)

All of these are class values for 24 m workcats and USVs (assumed; no vendor's vessel).

| | |
|---|---|
| hulls | two demihulls, 24 m x 10 m overall, 1.5 m draught, 108 t displacement (81.6 t lightship + 15.2 t deadweight) |
| power | diesel-electric: 2 x 250 kW gensets, 2 x 280 kW pods, bow tunnels, a 350 kWh buffer battery |
| speed | 10 kn service (170 kW), 14.6 kn max |
| on the bed | 82 kW, 28.7 L/h of fuel, 15.6 days of endurance |
| Walrus deck | 4 AHC davits (60 kg, launch and recovery to Hs 2.8 m, 2.0 m uncompensated), the basket robot (40 baskets an hour) |
| flight deck | 4 VTOL pads, a 7 m catapult, a 6 x 4 m net; the deck closes above Hs 2.0 m; 240 VTOL launches and 84.7 recoveries an hour, 15 fixed-wing launches and 12 recoveries an hour |
| hangar | 8 Gannets, 3 Fulmars, 12 chargers (16 kW) |
| price | EUR 4.15 M, EUR 1400 a day |

Esmeralda (`esmeralda.md`) is Zarco's guard-duty variant for the sea game.
