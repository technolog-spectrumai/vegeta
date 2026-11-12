# Yax 2 — the format converter

Platforms exchange **standard files**; Yax converts between them. Each platform uses the format that is natural to
it, and nobody imports anybody:

| Who | Native format |
|---|---|
| Piccolo (ROS 2, Gazebo) | URDF, SDF |
| Vegeta / Chiron (MuJoCo) | MJCF |
| Vegeta (flight models) | JSBSim aircraft XML |
| CadQuery / Dedalus | STEP |
| everyone (the universal file) | USD |

## The hub is USD

Every conversion goes *format → USD → format*. A USD stage written by Yax follows one **profile**, so any reader
knows where to look:

- Stage metadata: `upAxis = Z`, `metersPerUnit = 1`, `kilogramsPerUnit = 1` (SI throughout; formats in other units
  are converted on the way in and out). The default prim is the machine (an `Xform`); its `customData` holds
  `yax: 2`, `name`, `producer`.
- **Links** are `Xform` children of the machine with `UsdPhysics.RigidBodyAPI` and `UsdPhysics.MassAPI` (mass, centre
  of mass in the link frame, principal inertia + principal axes). Their transforms are the pose at q = 0 in the
  machine frame.
- **Shapes** are children of their link: `Cube` (scaled to a box), `Sphere` (scaled to an ellipsoid), `Cylinder`,
  `Capsule` (with `axis`), `Mesh`. `yax:role` = `collision` (with `UsdPhysics.CollisionAPI`) or `visual`;
  `primvars:displayColor` for colour.
- **Point masses** are mass-only `Xform` children of a link (`MassAPI`, no geometry).
- **Joints** live under `<machine>/joints`: `RevoluteJoint`, `PrismaticJoint`, `FixedJoint`, `SphericalJoint`, with
  `body0` (parent; empty = the world) and `body1` (child), `localPos/localRot` 0 and 1 (the joint frame in each body),
  the joint axis along the joint frame's X (USD's `axis = "X"`), limits (degrees for revolute, as USD), an optional
  `DriveAPI` (stiffness, damping, max force) for actuated joints. The root link is free unless a joint ties it to the
  world.
- **Parts data** USD has no schema for — aero tables, propulsion, flight-model sections, Ogma parts — sit as JSON in
  `yax:data` on the prim they belong to, keyed by kind (`{"jsbsim": …, "ogma": …, "propulsion": …}`), so nothing is
  lost on a round trip through USD. `propulsion` (propeller maps) is the one kind writers interpret: see below.

## Propulsion: propeller maps and properties

USD has no propeller schema and neither has any robot format, so a propeller is **part data**: the ``propulsion``
block in the machine's data (``yax.propulsion``), layered so a consumer takes the deepest layer its engine can use.

```json
{"rotors":     [{"name": "rotor_fl", "link": "rotor_fl", "pos": [0.1, 0.1, 0.0], "axis": [0, 0, 1], "spin": 1,
                 "propeller": "apc_5x4p3", "motor": "2306_2400kv"}],
 "propellers": {"apc_5x4p3": {"blades": 2, "diameter_m": 0.127, "mass_kg": 0.004, "inertia_kgm2": 1e-6,
                              "coefficients": {"J": [0.0, 0.1, …], "CT": [0.10, …], "CP": [0.05, …]},
                              "grid": {"rpm": […], "airspeed_m_s": […], "rho": 1.225,
                                       "thrust_n": [[…]], "torque_nm": [[…]], "power_w": [[…]]},
                              "source": "vegeta.boreas BEMT …"}},
 "motors":     {"2306_2400kv": {"power_w": 300.0, "kv": 2400, "R_ohm": 0.05, "I0_A": 0.8}},
 "reduced":    {"rotor_fl": {"kF": 8.5e-6, "kM": 0.016, "w_max": 1100, "tau_up": 0.0125, "tau_down": 0.025,
                             "taken_at": "hover", "residual": 0.31}}}
```

- ``coefficients`` — thrust and power coefficients against the advance ratio (CT = T / ρ n² D⁴, CP = P / ρ n³ D⁵,
  J = V / n D, n in rev/s): the density-invariant form flight simulators take. ``J`` increasing.
- ``grid`` — the dimensional rpm × airspeed map as the design tool made it (``power_w`` or ``torque_nm``). Without
  ``coefficients``, a writer collapses it onto J (one value per distinct J, averaged over rpm) and reports the collapse.
- ``motors`` — ``power_w`` is what a JSBSim electric engine takes; without it the map's maximum shaft power is assumed
  (reported). Other motor constants ride along for consumers that model the electric chain.
- ``reduced`` — the two-constant motor model (T = kF ω², Q = kM T, spin-up/-down time constants, ω_max) that Gazebo's
  multicopter motor plugin and PX4 take, per rotor name, with where it was taken and its residual against the map.
- ``spin`` is +1 when the propeller's drag torque on the frame is along +``axis``.

| Target | Holds | Report |
|---|---|---|
| USD | the whole block, verbatim (``yax:data``) | nothing |
| JSBSim | the coefficient tables natively: one electric engine per rotor (``Engines/motor_<motor>.xml``) driving a propeller file (``Engines/prop_<propeller>.xml``: ``C_THRUST``, ``C_POWER`` against J, diameter, blades, inertia), placed at ``pos`` along ``axis`` with ``sense = −spin``; written when the aircraft has no verbatim propulsion section | the grid collapse and the assumed motor power, if any |
| SDF | nothing in the file: ``yax sync`` maps ``reduced`` into a motor plugin (``data.propulsion.reduced.<rotor>.kF`` → ``motorConstant``) | per rotor: reduced to kF, kM at …, or that no reduced constants were given |
| MJCF, URDF, STEP, STL | nothing | machine data ``propulsion`` has no place |

Checked by JSBSim: a propeller written from coefficients gives, at the settled rpm, the table's CT ρ n² D⁴ and CP ρ n³ D⁵
(1e-4), the electric motor's power, and a reaction torque on the airframe along +axis for spin +1.

## Conversions are honest

`yax.convert(src, dst)` returns a **report** listing what the target format cannot hold (URDF has no capsule before
ROS 2 Jazzy's extension, no aero tables; MJCF has no aero tables; JSBSim has no joints, …). Nothing is dropped
silently.

## Formats

| Format | Read | Write | Through |
|---|---|---|---|
| USD (`.usda`, `.usdc`, `.usd`) | ✓ | ✓ | `pxr` (usd-core) — the hub |
| MJCF (`.xml` with `<mujoco>`) | ✓ (compiled by MuJoCo: defaults, classes, `fromto` resolved) | ✓ (checked by compiling) | `mujoco` |
| URDF (`.urdf`) | ✓ | ✓ (meshes as STL beside it) | XML |
| SDF (`.sdf`) | ✓ | ✓ | XML |
| JSBSim (`.xml` with `<fdm_config>`) | ✓ (mass and balance → one rigid body + point masses; the rest as `yax:data`) | ✓ (loads in JSBSim) | XML; `jsbsim` to check |
| STEP (`.step`, `.stp`) | ✓ (tessellated `Mesh` + exact mass properties) | ✓ (primitives as solids, meshes as faceted solids) | `cadquery` / OCP |
| STL | ✓ | ✓ | `yax.stl` |
| Yax 1 package (`yax.json` + STL) | ✓ | — | legacy reader |

Engines (`mujoco`, `cadquery`, `jsbsim`) are imported only by the format that needs them.

## Sync: physics across team boundaries

`yax.sync(design, target_sdf, mapping=…)` takes the physical numbers a map names from a design (any format; USD carries
part data too) into an SDF another team owns, and nothing else: link masses, centres and inertias, link poses (relative
to the mapped root links), joint axes and positions, single values inside plugins. The report lists every changed
value (old → new); `check=True` writes nothing — a non-empty report is drift. The file's prolog and comments survive.
