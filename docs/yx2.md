# YX2 — the Yax 2 exchange file (version 0.1)

A `.yx2` file is **one usda text layer** holding one machine: a validated subset of OpenUSD. Anything USD can read
reads it (rename it `.usda`, or import the text into a layer); Yax reads only files that pass the profile below
and writes only files that do. It is the file to exchange between platforms (Vegeta → Piccolo, …); `.usda` stays
plain USD for anyone who wants the hub's stage without the rules.

```python
yax.save(m, "drone.yx2")                 # refuses to write anything outside the profile
m, rep = yax.read("drone.yx2")           # raises yax.Invalid, listing the problems, for a file outside it
yax.formats.yx2.validate("drone.yx2")    # the problems, [] for a YX2 — also: yax check drone.yx2
```

## The profile

1. **One text layer.** usda, UTF-8, self-contained: meshes are inline points and faces, part data is inline JSON.
   No references, payloads, sublayers or side files.
2. **Layer data says the units and the frame.** `customLayerData["yx2"]` holds `version` (`"0.1"`), `units`
   (`{"length": "m", "mass": "kg", "time": "s", "angle": "rad"}`) and `frame` (`{"up": "+z", "forward": "+x",
   "handed": "right", "pose": "machine frame at q = 0"}`). The stage metadata agrees: `upAxis = "Z"`,
   `metersPerUnit = 1`, `kilogramsPerUnit = 1`. Joint limits appear twice, as USD wants them (degrees on the joint
   prim) and as Yax holds them (`yax:limits`, radians); readers take `yax:*`. Every link, shape and point mass carries
   its pose exactly (`yax:pos`, `yax:quat`, double precision) beside USD's xform ops, whose matrix decomposition rounds.
3. **The default prim is the machine**, an `Xform` with `customData` `yax = 2` and `name`, the articulation root.
   It carries `yax:features` always — `{"included": [...], "excluded": [...]}`, what the producer shared and
   withheld (`docs/interchange.md`) — and, when present, `yax:producer` and `yax:data`.
4. **Links** are `Xform` children of the machine with `RigidBodyAPI`, `yax:name` (unique among links) and
   `yx2:id = "link:<name>"`. A link with `MassAPI` also carries the exact `yax:mass`, `yax:com`, `yax:inertia`
   (double precision). Its shapes are `Gprim` children with `yax:name` (unique per role within the link),
   `yax:role` (`collision` or `visual`) and `yx2:id = "shape:<link>/<name>"`; its point masses are mass-only `Xform`
   children with `yax:kind = "point_mass"` and `yx2:id = "mass:<link>/<name>"`.
5. **Joints** live under `<machine>/joints`: `UsdPhysics` joints with `yax:name` (unique among joints),
   `yx2:id = "joint:<name>"`, `yax:kind` (revolute, continuous, prismatic, fixed, spherical), `yax:pos` and
   `yax:axis` (machine frame), whose bodies are links of this machine.
6. **Stable identifiers.** USD mangles a name into a prim path (`MakeValidIdentifier`); the identity of a component
   is its `yax:name`, and `yx2:id` spells it with its kind so two files, or a file and a consumer's model, refer to
   the same part by the same string (what `yax sync` maps on).
7. **Closed vocabulary.** The only attributes in the `yax:` and `yx2:` namespaces are the ones above (`yax:name`,
   `yax:kind`, `yax:pos`, `yax:quat`, `yax:axis`, `yax:limits`, `yax:drive`, `yax:data`, `yax:mass`, `yax:com`,
   `yax:inertia`, `yax:role`, `yax:uri`, `yax:producer`, `yax:features`, `yx2:id`). New physics goes into part
   data under a documented key (`docs/interchange.md`) or into a new profile version — never into an ad-hoc attribute.
8. **Canonical serialization.** Prims in machine order (links as the hub lists them, their shapes and point masses
   in order, then the joints scope), every JSON string with sorted keys and no whitespace, numbers as USD prints
   them (shortest round trip). The same machine gives the same bytes; `write(read(f)) == f`.

## What a version change means

`yx2.version` changes when the profile does. A reader that knows `0.1` refuses other versions rather than guessing;
the hub (`yax.hub`) is what a conversion goes through, so a newer file is converted by a newer Yax, never half-read.

## Why a subset

Plain USD lets everything be anywhere: units in the stage metadata or nowhere, names in paths that mangle them,
part data that may or may not be there. An exchange needs the opposite — one place for each fact, every fact
present or declared absent — and the profile is that: the hub's stage with the ambiguities closed.
