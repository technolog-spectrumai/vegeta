# Export — what leaves Vegeta, feature by feature

A machine leaves Vegeta as standard files written through Yax (`tools/export.py`, `python tools/lab.py export`): a
**YX2** file (`.yx2`, the Yax 2 exchange file — a validated subset of USD that states its units and frame, gives every
link, joint and shape a stable id, always says which features it holds and which were withheld, and serializes
canonically: `vendor/yax/docs/yx2.md`; `.usda` is the plain USD stage) and whatever formats a consumer reads beside it
(SDF, URDF, MJCF, STEP, STL, JSBSim). What a machine *is* comes from its lab's `designs/machines.py`: the Chiron robot
and the part data the design computed. Vegeta → YX2 → Yax → YX2 keeps every selected property byte for byte.

## Features

Every physical fact belongs to one of Yax's features (`vendor/yax/docs/interchange.md`): **geometry, mass, inertia,
joints, motors, materials, strength, connections, aerodynamics, sensors, provenance**. The export picks them
independently:

```bash
python tools/export.py drongo --to out/drongo.yx2 --also sdf                         # everything (the default)
python tools/export.py drongo --to out/drongo.yx2 --full                             # everything, said explicitly
python tools/export.py drongo --to out/drongo.yx2 --exclude motors,provenance        # withhold two
python tools/export.py drongo --to out/drongo.yx2 --include geometry,mass,inertia    # the structure alone
python tools/export.py drongo --to out/drongo.yx2 --features export.yaml             # include/exclude lists from a file
```

A withheld feature reaches no file, no metadata and no side file: Yax's `select` withholds and never invents (a
withheld mass is absent, withheld joints become fixed so the tree and the links' names stay, withheld part data is
gone, a withheld provenance is an empty producer stamp), the USD records what was withheld (`yax:features`), and
every other format's report says "withheld by the producer". What Vegeta has **no data for** is *missing*, not
invented — a sensor-less design exports no sensors whatever the flags say — and the inventory printed after each
export tells the two apart:

```
exported:
  geometry: available (33 shapes)
  motors: excluded (withheld by the producer)
  sensors: missing
```

What each feature holds today, per machine, is what the design computed: the Chiron robot's geometry, masses,
inertias and joints; the propeller maps and motor constants (`labs/aero/designs/propulsion_parts.py`, from Boreas,
Ogma and the propulsion files); Drongo's materials catalogue (notebook 08's densities); the provenance (the robot's
`sources` and `notes`, and the export's producer stamp with the git revision). Strength, connections, aerodynamics
and sensors are missing until a design hands them over through its parts function — the export runs no FEA or CFD
to fill them.

## Consumers

A consumer declares what it needs in a Yax requirements file (`tools/consumers/piccolo.json`: required and optional
features). `--consumer FILE` (repeatable) reports, per feature and per written format, whether the export serves it —
available, missing, excluded, or unsupported by that format — and `--strict` exits 1 when it does not. The
requirements never change what is exported: the flags decide, the consumer's file only judges the result.

```bash
python tools/export.py drongo --to out/drongo.yx2 --also sdf --exclude inertia --consumer tools/consumers/piccolo.json
# piccolo (usd): NOT served
#   required inertia: excluded (withheld by the producer)
```
