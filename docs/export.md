# Export — what leaves Vegeta, feature by feature

A machine leaves Vegeta as standard files written through Yax (`tools/export.py`, `python tools/lab.py export`): a
**YX2** file (`.yx2`, the Yax 2 exchange file — a validated subset of USD that states its units and frame, gives every
link, joint and shape a stable id, always says which features it holds and which were withheld, and serializes
canonically: `vendor/yax/docs/yx2.md`; `.usda` is the plain USD stage) and whatever formats a consumer reads beside it
(SDF, URDF, MJCF, STEP, STL, JSBSim). What a machine *is* comes from its lab's `designs/machines.py`: the Chiron robot
and the part data the design computed. Vegeta → YX2 → Yax → YX2 keeps every selected property byte for byte.

## Tags

Every property of a machine has a path (`link:drongo/mass`, `joint:jaw_L/drive`, `data:propulsion/propellers/apc`,
`producer`) and carries **tags**: codes of 1 to 8 alphanumerics (`vendor/yax/docs/interchange.md`). Yax assigns the
**system tags** from what a property is — `00core` (the skeleton: names, poses, the joint tree; never withheld),
`00geom`, `00mass`, `00inert`, `00joint`, `00motor`, `00matl`, `00strgth`, `00conn`, `00aero`, `00sensor`, `00prov`,
`00fmt` — so a consumer can always validate against them. The designs add their own through their parts functions:
`boreas` on the maps a BEMT made, `nb08` on the numbers notebook 08 fixed, `fitted` on a motor fitted to its published
point, `assumed` on what is an assumption. The user adds more on the command line. The export picks by tag:

```bash
python tools/export.py drongo --to out/drongo.yx2 --also sdf                           # everything (the default)
python tools/export.py drongo --to out/drongo.yx2 --full                               # everything, said explicitly
python tools/export.py drongo --to out/drongo.yx2 --drop 00strgth,assumed              # withhold two tags
python tools/export.py drongo --to out/drongo.yx2 --keep 00geom,00mass,00inert,00joint # the structure alone
python tools/export.py drongo --to out/drongo.yx2 --tag 'link:*/mass=weighed'          # a tag of your own
python tools/export.py drongo --to out/drongo.yx2 --tags export.yaml                   # keep, drop and tag from a file
```

A dropped property reaches no file, no metadata and no side file: Yax's `select` withholds and never invents (a
withheld mass is absent, a withheld articulation becomes a fixed joint so the tree and the links' names stay, withheld
part data is gone at its path, a withheld provenance is an empty producer stamp), the file records the tag table and
what was withheld (`yax:tags`), and every other format's report says "withheld by the producer". What Vegeta has **no
data for** is *absent*, not invented — a sensor-less design exports nothing under `00sensor` whatever the flags say —
and the inventory printed after each export says what each tag holds:

```
exported:
  00core: 5 properties — the skeleton: the machine's name, the links' identities and poses, the joint tree
  00geom: 33 properties — shapes (collision and visual), meshes
  00motor: 56 properties — drives, joint motors, the propulsion block, ogma rotor constants
  boreas: 41 properties
  00strgth: withheld by the producer — strength: allowables, safety factors, analysis results
```

What each tag holds today, per machine, is what the design computed: the Chiron robot's geometry, masses, inertias
and joints; the propeller maps and motor constants (`labs/aero/designs/propulsion_parts.py`, from Boreas, Ogma and the
propulsion files); Drongo's materials catalogue (notebook 08's densities); the provenance (the robot's `sources` and
`notes`, and the export's producer stamp with the git revision). Strength, connections, aerodynamics and sensors are
absent until a design hands them over through its parts function — the export runs no FEA or CFD to fill them.

## Engineering blocks

Beside the tags, the export carries the engineering blocks of `vendor/yax/docs/engineering.md` wherever a design
states them: joint motors from the Chiron servos and the Ogma units they came from (`vegeta.labkit.motors`), the
propulsion maps (`labkit.propulsion`), materials and connections (`labkit.materials`), aerodynamics from the design
files (`labkit.aero`), and the assembly from the joint tree (`labkit.detach`). Nothing is computed for the export
and nothing is invented: a design that states no fastener exports none. Progress and limitations: `docs/export2.md`.

## Consumers

A consumer declares the tags it needs in a Yax requirements file (`tools/consumers/piccolo.json`: required and
optional codes — the system tags it counts on, and any producer tag the two sides agreed on, such as `boreas`).
`--consumer FILE` (repeatable) reports, per tag and per written format, whether the export serves it — present,
absent, withheld, or unsupported by that format — and `--strict` exits 1 when it does not. The requirements never
change what is exported: the flags decide, the consumer's file only judges the result.

```bash
python tools/export.py drongo --to out/drongo.yx2 --also sdf --drop 00inert --consumer tools/consumers/piccolo.json
# piccolo (yx2): NOT served
#   required 00inert: withheld (withheld by the producer)
```
