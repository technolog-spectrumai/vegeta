# Yax

**Design interchange through standard files.** Platforms exchange the files that are natural to them — Piccolo
URDF and SDF (ROS, Gazebo), Vegeta MJCF (MuJoCo, Chiron) and JSBSim (flight models), CadQuery STEP — and USD as the
universal file. Yax converts between them; every conversion passes through a USD stage (the hub) and reports what the
target format cannot hold. Yax imports no platform's code; Vegeta and Piccolo vendor it. Design: [docs/converter.md](docs/converter.md).

```python
import yax

report = yax.convert("rover.xml", "rover.urdf")    # MJCF → USD → URDF
print(report)                                      # e.g. "dropped: joint wrist: spherical (no URDF joint) — written as fixed"
m, rep = yax.read("c172x.xml")                     # JSBSim (detected from <fdm_config>) → yax.Machine
yax.save(m, "c172x.usda")                          # the universal file
```

```bash
yax convert rover.xml rover.sdf        # or: python -m yax convert …
yax show rover.sdf                     # links, joints, mass
```

**Tags — what is shared, what is needed.** Every property of a machine carries tags: codes of up to 8 alphanumerics.
The producer tags as it pleases (`boreas` on the maps a BEMT made) and keeps or withholds properties by tag
(`yax select src.yx2 dst.yx2 --tag 'data:propulsion/*=boreas' --drop 00strgth`: withheld, never invented, and the file
says so); the consumer validates the tags it needs (`yax tags dst.yx2 --to sdf --requirements piccolo.json`: present,
absent, withheld or unsupported by the format). Codes starting with `00` are system tags Yax assigns from what a
property is (`00geom`, `00mass`, `00inert`, `00joint`, `00motor`, …), so there is always something to validate against.
The contract: [docs/interchange.md](docs/interchange.md).

**Two teams, one machine — `yax sync`.** The designers' file and the simulation team's model live in their own feedback
loops; the physical numbers the simulation takes from the design are synced explicitly, everything else stays the
team's own:

```bash
yax sync design.usda models/drone/model.sdf --map drone.map.json           # rewrite only the mapped fields, report each
yax sync design.usda models/drone/model.sdf --map drone.map.json --check   # CI: exit 1 when the model has drifted
```

The map names the links (masses, centres, inertias, poses), joints (axes, positions) and single values inside plugins
(e.g. a motor model's `motorConstant` from the design's part data); sensors, controllers, topics, gains and comments are
never touched (`src/yax/sync.py` documents the map).

| Format | Read | Write | Checked by |
|---|---|---|---|
| **YX2** (`.yx2`) | ✓ | ✓ | the exchange file: a validated USD subset — units, frame, stable ids, tag table, canonical text (`write(read(f)) == f`); `yax check f.yx2`; [docs/yx2.md](docs/yx2.md) |
| USD | ✓ | ✓ | exact round trips (double-precision copies of USD's float attributes) |
| MJCF | ✓ (compiled by MuJoCo) | ✓ | MuJoCo compiles it: same bodies, inertias, joints, geoms, actuators |
| URDF | ✓ | ✓ | round trips relative to the root; capsules/ellipsoids as STL meshes, ball joints as fixed (reported) |
| SDF 1.9 | ✓ | ✓ | round trips with absolute poses; drives reported (Gazebo plugins carry them); a propeller map is reported as its reduced constants (kF, kM, where taken) for a motor plugin |
| JSBSim | ✓ | ✓ | the C172X converted loads in JSBSim with the same mass, CG, inertia and aerodynamics (1e-12); a propeller map in the `propulsion` part data becomes a propeller + electric engine whose thrust and power JSBSim reproduces (1e-4) |
| STEP | ✓ (+ exact mass properties given a density) | ✓ | solid volumes equal the shapes' (1e-6) |
| STL | ✓ | ✓ | — |
| Yax 1 package | ✓ | (`yax.write`) | spec tests |

Install: `pip install -e .` (numpy, usd-core) with the extras the formats need: `[mjcf]` MuJoCo, `[step]` CadQuery,
`[jsbsim]` JSBSim, `[all]`. Tests: `./test.sh` (engines not installed → their tests skip), `./user_tests.sh` (verbose, logged).

## Yax 1 packages (still read)

Design interchange. A **package** is a directory with `yax.json` (units, frame, geometry,
mass properties, file hashes) and `body.stl`; a producer writes it, a consumer loads it, and neither imports the
other. Specification: [docs/spec.md](docs/spec.md).

```python
import yax

d = yax.load("drongo.yax")            # verifies hashes and the manifest, raises yax.Invalid on problems
d.mass["kg"], d.mass["inertia"]       # SI, about d.mass["com"]
tris = d.mesh()                       # triangles of body.stl, in d.length_unit
yax.units.LENGTH[d.length_unit]       # 0.001 for mm

yax.write("out/cube.yax", manifest, {"body.stl": "cube.stl"})   # producers: atomic, fills hashes and identity
yax.describe(triangles, "mm", density_kg_m3=2700)              # geometry + mass sections from a closed mesh
```

```bash
python -m yax check out/*.yax      # or the `yax` command once installed
python -m yax info out/cube.yax
```

* **Never** imports a producer or a consumer, numpy, a CAD kernel or a simulator.
* Exact mass properties of a closed mesh (`yax.stl.mass_properties`), so a consumer without CAD can verify a
  producer.
* `examples/cube.yax` is the test vector (`python3 examples/make_cube.py` regenerates it). Tests: `pytest` here.
