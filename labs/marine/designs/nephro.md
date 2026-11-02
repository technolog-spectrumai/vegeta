# Nephro — design brief (v0)

Named after *Nephrops norvegicus*, the Norway lobster, which digs its burrows in the muddy shelf floor that cables
are buried in and feels its way with long antennae (first drafted as Emerita, after the mole crab). Nephro is a hybrid cable AUV in the Scampi family (`scampi.md`). It cruises a
torpedo hull along a buried cable, tracking it with a coil array on the nose (the "antennae"), and hovers or lands to
inspect, mark and lift the cable at a fault. It shares Scampi's housings, Jetson autonomy and water model.

## Decisions

| question | decision |
|---|---|
| cables | wind export (30–100 km) and array cables first, then telecom landings and shelf routes |
| jobs | route survey + depth of burial; exposures and free spans; fault localisation; light intervention (lift and tag, clear debris, markers) |
| depth | 300 m (Scampi's housing class) |
| range | 60–100 km per dive: a whole export cable |

Scampi's open frame cannot do this: 165 Wh/km at 1 m/s gives ~27 km on its pack. A body of revolution needs ~63 Wh/km.

## v0 numbers (`python labs/marine/designs/nephro.py`)

* 3.0 m × Ø0.40 m; **165 kg** in air, +1 kg buoyant. Two Al 6061-T6 housings Ø280 × 10 mm (battery 1.0 m,
  electronics 0.55 m): buckling 7.0 MPa against 4.5 MPa (margin +0.56). They are 51 kg of the mass: a composite
  battery housing is the first weight saving to look at.
* Survey at **1.5 m/s**, 2 m over the seabed: drag 36 N (Cd 0.25 with fins, tunnels, antennae and skids, assumed),
  propulsion 121 W + sensors and computer 220 W = **341 W**.
* Battery 8 kWh Li-ion (32 kg of cells). With 1 h of hover and landed work and a 15 % reserve kept back: **97 km**
  per dive at 1.5 m/s (86 km at 1.0 m/s, 87 km at 2.0 m/s: 1.5 m/s is close to the best speed).
* Sensors: 3-axis coil array (an injected tone, or the passive field of an AC power cable), side-scan, multibeam
  (free spans), sub-bottom profiler (burial check in soft sediment), cameras.
* Hover and work: two stern thrusters behind cruciform fins, four tunnel thrusters (two vertical, two lateral), a
  5-DOF folding arm in a belly bay (cable gripper and tagger), two low skids to land beside the cable.
* Navigation: INS + DVL; the cable itself resets the cross-track drift, and USBL fixes from the vessel reset the
  along-track drift.

## Open questions

* the coil array's reach: burial depth it can measure at 2 m altitude, and on an HVDC export cable (no AC field: needs a tone)
* landing beside a cable in a current: the skids and tunnel thrusters against 1 m/s
* launch and recovery of a 3 m, 165 kg vehicle from a CTV-size boat (Scampi's davit is rated for ~120 kg)
* shallow landings (the cable's landfall): surf, turbid water, a vehicle 3 m long in 3–5 m of water
