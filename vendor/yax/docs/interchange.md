# The interchange contract — tags

Yax moves engineering data between platforms (Vegeta designs; Piccolo and other simulators consume) as **properties
that carry tags**. A tag is a code of one to eight alphanumeric characters. The producer tags properties as it
pleases and keeps or withholds them by tag; the consumer validates the file against the tags it needs. Neither has
to know the other's groups — only the codes they agreed on. Codes beginning with `00` are **system tags**, reserved
to Yax: it assigns them from what a property *is*, so a consumer can always validate against them, old-school.
`yax.tags` is the module; `yax tags FILE` and `yax select SRC DST` the commands. What the blocks hold — their fields, units,
frames and what each format loses of them — is the engineering contract, docs/engineering.md (`yax validate FILE`).

## Properties

Everything is SI (m, kg, s, N; angles in the hub in radians), every pose is in the machine frame at q = 0, and
**names are identities**: a link, joint, shape, point mass, rotor, propeller, motor, material or connection is
referred to by its name everywhere. A property is addressed by its **path**:

| Path | What | System tag |
|---|---|---|
| `link:<name>/pose` | the link's pose (its identity and place) | `00core` |
| `link:<name>/mass` | mass and centre of mass | `00mass` |
| `link:<name>/inertia` | the inertia tensor (about the centre of mass, link axes, kg m²) | `00inert` |
| `link:<name>/collision:<shape>`, `link:<name>/visual:<shape>` | a shape | `00geom` |
| `link:<name>/pmass:<name>` | a point mass | `00mass` |
| `link:<name>/data:<key>[/<key>…]` | the link's part data, any depth | by key (below) |
| `joint:<name>/tree` | parent, child, position, axis (the skeleton) | `00core` |
| `joint:<name>/articulation` | kind, limits, passive damping, effort and velocity limits | `00joint` |
| `joint:<name>/drive` | the PD drive | `00motor` |
| `joint:<name>/data:<key>[/<key>…]` | the joint's part data | by key (below) |
| `data:<key>[/<key>…]` | the machine's part data, any depth | by key (below) |
| `producer` | the producer stamp (tool, revision, machine, identity) | `00prov` |

`data:` paths reach every dictionary key at any depth: a tag on `data:propulsion` covers the block, a tag on
`data:propulsion/propellers/apc` one propeller. Part data keys and the system tags they carry (unknown keys carry
none until a producer tags them):

| Key | Recommended shape | System tag |
|---|---|---|
| machine `propulsion` | docs/engineering.md: rotors, propellers (CT/CP/CQ against J, the grid), motors, reduced constants | `00prop` (its `motors` also `00motor`) |
| machine `ogma` | reduced rotor constants (Gazebo's motor model) | `00prop`, `00motor` |
| machine `motors`, joint `motor` | docs/engineering.md: the motor model (stall, no-load, curve, constants, inertia, limits, transmission); a joint names a catalogue motor or holds one | `00motor` |
| machine `materials`, link `material` | `{name: {"density_kg_m3", "E_pa", "nu", "yield_pa", "ultimate_pa", "source"}}`; a link names one or holds one | `00mat` |
| machine `strength`, link `strength` | `{"allowable_pa", "safety_factor", "analysis": {"tool", "case", "max_stress_pa", "margin", "source"}}` — results of analyses that ran, never computed by Yax | `00str` |
| machine `connections`, joint `connection` | `[{"name", "links": [a, b], "type", "spec": {…}, "rating": {"shear_n", "tension_n", "torque_nm"}, "source"}]` | `00conn` |
| machine `aerodynamics`, `jsbsim`, link `aero` | `{"reference": {"S_m2", "b_m", "c_m", "frame"}, "coefficients", "derivatives", "polars", "source"}`; an aircraft's JSBSim sections verbatim | `00aero` |
| machine `sensors`, link `sensors`; link `sdf/sensor` | `[{"name", "type", "pos", "quat", "spec": {…}}]` in the link frame; SDF sensors carried verbatim | `00sensor` |
| machine `provenance` | anything more than the stamp: who, when, the design's parameters, sources | `00prov` |
| machine `mjcf`, `sdf`; link `sdf/plugin` | another format's own data, verbatim | `00fmt` |

A producer may put more in each entry (readers keep what they do not know), never less than a `source` saying where
a number comes from. Yax never fills a field a producer left out.

## System tags

| Tag | Properties |
|---|---|
| `00core` | the skeleton: the machine's name, the links' identities and poses, the joint tree — always present, never withheld |
| `00geom` | shapes (collision and visual), meshes |
| `00mass` | masses, centres of mass, point masses |
| `00inert` | inertia tensors |
| `00joint` | articulation: joint kinds, axes, limits, passive damping, effort and velocity limits |
| `00motor` | motors and actuators: joint drives, motor models, the propulsion motors, ogma rotor constants |
| `00prop` | propulsion: rotors, propellers (geometry, coefficients, maps), reduced rotor constants |
| `00mat` | materials: the catalogue and each link's |
| `00str` | strength: allowables, safety factors, analysis results |
| `00conn` | connections between links |
| `00aero` | aerodynamics, JSBSim sections, per-link aero |
| `00sensor` | sensors, SDF sensors carried verbatim |
| `00prov` | the producer stamp and provenance data |
| `00fmt` | another format's own data carried verbatim |
| `00detach` | the assembly: components, parent–child relations, attachments, separation limits (docs/engineering.md) |

Yax assigns these; a producer cannot (`ValueError`). Any other code is the producer's: `boreas` on the propeller
maps a BEMT made, `nb08` on the numbers notebook 08 fixed, `weighed` on the masses that were put on a scale.

## The tag table

`Machine.tags` is `{"table": {path: [codes]}, "withheld": [codes]}`. The table lists every tagged property with all
its codes, system tags included, so a file stands on its own (`yax.tags.table(m)` computes it; a YX2 always carries
it as `yax:tags`); `withheld` lists the tags the producer dropped. A machine without a table reads as one with the
system tags implied and nothing withheld — files written before tags existed are unchanged.

```python
tg.tag(m, "data:propulsion/propellers/*", "boreas")      # the producer's tag on a pattern of paths
tg.inventory(m)                                           # tag → paths (None: withheld)
cut = tg.select(m, keep=("00geom", "00mass", "boreas"))   # or drop=(...); 00core always stays
```

## Selecting: withhold, never invent

`select` keeps the properties whose tags meet `keep` (any of) and none of `drop`. A dropped property is absent: a
mass `None` (writers leave the inertial out), an articulation turned into a fixed joint (the tree and the links'
names and poses stay, so a consumer's references hold; the drive goes with it), a data key gone at the path, an
empty producer stamp. Nothing it contained reaches the file, its metadata or its side files, and every writer of
another format reports "withheld by the producer". A consumer's requirements never change what a producer shares.

## Filtering rules

`select(m, keep, drop)` decides property by property, from the full table (system tags Yax assigns, producer tags
given on the path or on any path above it):

| Case | Rule |
|---|---|
| a property with **no tags** (data under a key no schema knows, untagged by the producer) | kept when `keep` is not given; dropped by any `keep` (nothing meets it); never dropped by `drop` |
| **multiple tags** | kept when *any* of its tags is in `keep`; dropped when *any* is in `drop` (`drop` wins) |
| **system and user tags** on one property | the same rule, the two kinds alike: `keep=("boreas",)` keeps a `00prop` map that also carries `boreas` |
| **unknown tags** (a code the reader never heard of) | preserved in the table and the file; a consumer that does not need it ignores it |
| a **required tag missing** from the file | `check` says `absent` (nothing ever carried it) or `withheld` (the producer dropped it, or every property carrying it); the verdict fails |
| a property **dependent on excluded data** (a strength naming a dropped material, a rotor naming a dropped propeller, a component naming a dropped connection) | the referring property stays as it is — never rewritten — and `schema.problems` names the dangling reference, so the consumer sees what it depends on |
| a **container** (a data block) whose children are partly kept | stays, holding the kept children only |
| `00core` | never dropped: names, poses, the joint tree |

An excluded value reaches nothing: the hub's copy no longer holds it, every writer writes from the hub, side files
(JSBSim engine files, meshes) are produced from the hub after selection, and the YX2 table lists only what is there.

## Requirements: what a consumer needs

```python
req = yax.tags.Requirements(required=("00geom", "00mass", "00inert", "00joint"), optional=("00motor", "boreas"), consumer="piccolo")
verdict = yax.tags.check(m, req, fmt="sdf")      # or: yax tags drone.yx2 --to sdf --requirements piccolo.json
verdict.ok                                        # every required tag present (and reaching sdf)
print(verdict)                                    # piccolo (sdf): served / NOT served, then one line per tag
```

A requirements file is JSON: `{"consumer": "piccolo", "required": [...], "optional": [...]}`. Per tag the verdict
is **present** (with how many properties, and how many of them a target format holds), **absent** (nothing carries
it), **withheld** (the producer dropped it), or **unsupported** (the target format has no place for any of its
properties — the consumer reads them from the YX2). The check enables nothing and converts nothing; a required tag
that is not present fails it (exit 1 on the command line).

## Format support

| Format | Holds (property kinds) | Partial |
|---|---|---|
| USD, YX2 | everything | — |
| MJCF | poses, masses, inertias, shapes, the tree, articulation, drives | `00motor`: joint drives as position actuators only |
| URDF | poses, masses, inertias, shapes, the tree, articulation | `00joint`: ball joints as fixed, the drive's max force as the effort limit |
| SDF | the same, plus verbatim SDF data (sensors, plugins) | `00joint`: no drives (a plugin carries them) |
| JSBSim | masses and inertias (one rigid body), the propulsion block, JSBSim sections | `00motor`: propellers and electric engines from `propulsion` |
| STEP, STL | poses, shapes | — |
| Yax 1 package | poses, shapes, mass, inertia, the stamp | mass: the whole body |
