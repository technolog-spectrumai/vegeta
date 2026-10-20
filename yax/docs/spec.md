# Yax design package — specification, version 1

A **Yax package** is a directory that carries one designed object from the tool that designed it (the *producer*,
e.g. Vegeta) to tools that use it (the *consumers*, e.g. Piccolo, Nomios, Ketos). It is plain files: an STL mesh
and a JSON manifest. Nothing is imported across repositories, nothing needs a network or a server.

```
drongo.yax/
  yax.json            manifest (required)
  body.stl            the whole object, binary or ASCII STL (required)
  body.step           exact CAD (optional)
  parts/<name>.stl    parts, in the same frame as body.stl (optional)
```

The directory name is free; the `.yax` suffix is a convention.

## `yax.json`

| Field | Required | Meaning |
|---|---|---|
| `yax` | yes | `1` — this specification |
| `name` | yes | a short name |
| `units` | yes | `{"length": "mm" \| "m", "mass": "kg"}` — every length in the manifest and in the STL files |
| `frame` | yes | `{"up": "+z", "forward": "+x"}` — two perpendicular axes; `left` = `up` × `forward` |
| `files` | yes | `{path: sha256 hex}` for every file of the package, `body.stl` included; paths relative, no `..` |
| `geometry` | yes | `{"bbox": [[xmin, ymin, zmin], [xmax, ymax, zmax]], "volume": V, "com": [x, y, z]}` — of `body.stl`'s solid, in `units.length` |
| `mass` | no | `{"kg": m, "com": [x, y, z], "inertia": 3x3, "source": "density" \| "given"}` — `inertia` in **kg m²**, about `mass.com`, axes of `frame`, products negated (the usual tensor); `com` in `units.length` |
| `parts` | no | `[{"name", "file", "kg"?, "com"?}]` — each `file` listed in `files` |
| `producer` | no | `{"tool", "version", "revision"?, "design_class", "parameters"}` |
| `identity` | with `producer` | SHA-256 hex of the canonical JSON of `{"tool", "design_class", "parameters"}` taken from `producer` |

Other fields are allowed; readers keep and ignore them. A new major version changes `yax`.

**Canonical JSON** (as Porunga's, `porunga/docs/identity.md`): object keys sorted, separators `,` and `:` without
spaces, non-ASCII characters as UTF-8, no NaN or infinity. The identity names *what* was designed — the same tool,
class and parameters give the same identity whatever the tool's version.

## Conventions that matter to consumers

* **Units.** Vegeta writes `mm` (the CAD convention). MuJoCo, ROS and Gazebo want metres: scale meshes by
  `yax.units.LENGTH[units.length]` (0.001 for mm) and use `mass` as given — it is already SI.
* **Frame.** Vegeta writes Z up, X forward. A Y-up consumer (glTF, many game engines) rotates with
  `yax.units.rotation(frame, {"up": "+y", "forward": "+z"})`.
* **Origin.** The design's own origin; nothing is re-centred. `geometry.com` and `mass.com` say where the mass is.
* **Mass.** `source: "density"` — uniform material over the solid; `"given"` — a total mass (e.g. a measured or
  budgeted mass) spread uniformly to obtain the inertia. A consumer that needs a different distribution uses `parts`.

## Validity

`yax.check(dir)` lists the problems; `yax.load(dir)` raises `yax.Invalid` if there are any:
the manifest fields above, every listed file present with its SHA-256, every STL readable, the identity matching
the producer. Test vector: `examples/cube.yax` (10 mm aluminium cube: 0.0027 kg, centre (5, 5, 5) mm,
I = 4.5e-8 kg m² on each axis).
