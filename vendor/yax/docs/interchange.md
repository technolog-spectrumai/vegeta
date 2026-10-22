# The interchange contract — features

Yax moves engineering data between platforms (Vegeta designs; Piccolo and other simulators consume) as **features**:
independent categories a producer can include or withhold one by one, a consumer can require or use when present,
and a format can or cannot hold. `yax.features` is the module; `yax features FILE` and `yax select SRC DST` the commands.

## The features and where each one lives in the hub

Everything is SI (m, kg, s, N, rad; angles in the hub in radians), every pose is in the machine frame at q = 0, and
**names are identities**: a link, joint, shape, rotor, propeller, motor, material or connection is referred to by its
name everywhere (the sync map, a consumer's own model, a report line). Unknown data keys are kept and ignored.

| Feature | Hub fields | Part data (JSON in `yax:data`) |
|---|---|---|
| `geometry` | `Link.shapes` (collision and visual: box, sphere, ellipsoid, cylinder, capsule, mesh) | — |
| `mass` | `Link.mass`, `Link.com`, `Link.point_masses` | — |
| `inertia` | `Link.inertia` (about the centre of mass, link axes, kg m²) | — |
| `joints` | `Joint.kind`, `pos`, `axis`, `lower`, `upper`; `Joint.data`: `damping`, `effort`, `velocity` | — |
| `motors` | `Joint.drive` (PD: stiffness, damping, max_force) | `Joint.data["motor"]` — the nonlinear motor: `{"stall_nm", "no_load_rad_s", "rated_nm", "kt_nm_per_a", "R_ohm", "I0_a", "voltage_v", "gear_ratio", "torque_speed": {"rad_s": [...], "nm": [...]}, "source"}`; machine `propulsion` (docs/converter.md: rotors, propellers with CT/CP against J and the grid, motors, reduced constants); machine `ogma` (rotor constants) |
| `materials` | — | machine `materials`: `{name: {"density_kg_m3", "E_pa", "nu", "yield_pa", "ultimate_pa", "source"}}`; `Link.data["material"]`: a name in the catalogue, or an entry of the same shape |
| `strength` | — | machine `strength` and `Link.data["strength"]`: `{"allowable_pa", "safety_factor", "analysis": {"tool", "case", "max_stress_pa", "margin", "source"}}` — results of analyses that were run, never computed by Yax |
| `connections` | — | machine `connections`: `[{"name", "links": [a, b], "type": "bolt" \| "weld" \| "bond" \| "pin" \| …, "spec": {…}, "rating": {"shear_n", "tension_n", "torque_nm", …}, "source"}]`; `Joint.data["connection"]` for a joint's own hardware |
| `aerodynamics` | — | machine `aerodynamics`: `{"reference": {"S_m2", "b_m", "c_m", "frame"}, "coefficients": {…}, "derivatives": {…}, "polars": {…}, "source"}`; machine `jsbsim` (an aircraft's sections, verbatim); `Link.data["aero"]` for a lifting link |
| `sensors` | — | `Link.data["sensors"]`: `[{"name", "type", "pos", "quat", "spec": {…}}]` in the link frame; the SDF sensors a Gazebo model carries verbatim (`Link.data["sdf"]["sensor"]`) |
| `provenance` | `Machine.producer` (`tool`, `revision`, `machine`, `identity`, …) | machine `provenance` (anything more: who, when, the design's parameters) |

The schemas in the table are the contract's **recommended** shapes; a producer may put more in each entry (readers
keep what they do not know), never less than a `source` saying where a number comes from. Yax never fills a field a
producer left out.

## Status

For a machine, each feature is

- **available** — the machine holds it (the inventory says how much: "4 shapes", "2 rotors, 1 propellers, 1 motors");
- **missing** — it holds none, and nobody withheld it (the producer had no such data, or a format on the way lost it:
  its report said so);
- **excluded** — the producer withheld it on purpose: `Machine.features["excluded"]` lists it, and the USD carries the
  list (`yax:features` on the root prim), so a consumer can tell "not shared" from "not known";
- **unsupported** — relative to a target format: the format has no place for it (`yax.features.SUPPORT`), whatever the
  machine holds; the consumer reads it from the USD instead.

| Format | Holds | Partial |
|---|---|---|
| USD, YX2 | every feature (YX2: the validated subset, docs/yx2.md) | — |
| MJCF | geometry, mass, inertia, joints, motors | motors: joint drives as position actuators only |
| URDF | geometry, mass, inertia, joints | joints: ball joints as fixed, the drive's max force as the effort limit |
| SDF | geometry, mass, inertia, joints, sensors | joints: no drives (a plugin carries them) |
| JSBSim | mass, inertia, motors, aerodynamics | one rigid body; motors: propellers and electric engines from `propulsion` |
| STEP, STL | geometry | — |
| Yax 1 package | geometry, mass, inertia, provenance | mass: the whole body |

## Selecting: what a producer shares

`yax.features.select(m, include=…, exclude=…)` (CLI: `yax select SRC DST --include a,b --exclude c`) returns a copy
holding only the chosen features and records the rest in `features["excluded"]`. The rule is **withhold, never
invent**: a withheld mass is `None` (writers leave the inertial out), withheld joints become fixed joints (the tree
and the links' names and poses stay, so a consumer's references hold; drives go with them), withheld part data is
gone, a withheld provenance is an empty producer stamp. Nothing a withheld feature contained reaches the file, its
metadata or its side files (meshes, propeller files), and every writer of another format reports "withheld by the
producer" so the loss is on record. A consumer's requirements never change what a producer shares.

## Requirements: what a consumer needs

```python
req = yax.features.Requirements(required=("geometry", "mass", "inertia"), optional=("motors", "sensors"), consumer="piccolo")
verdict = yax.features.check(m, req, fmt="sdf")      # or: yax features drone.usda --to sdf --requirements piccolo.json
verdict.ok                                           # every required feature available (and supported by sdf)
print(verdict)                                       # piccolo (sdf): served / NOT served, then one line per feature
```

A requirements file is JSON: `{"consumer": "piccolo", "required": [...], "optional": [...]}`. The check reads the
file as it is: it enables nothing, converts nothing, and a required feature that is `missing`, `excluded` or
`unsupported` fails it (exit 1 on the command line), each with its reason. Optional features are reported, never
failed.

## Compatibility

Files written before features existed read as machines with an empty `features`: every feature they hold is
available, every other is missing. `Machine.features` is the only new field; `yax:features` the only new attribute.
