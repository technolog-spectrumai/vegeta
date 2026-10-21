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
  `yax:data` on the prim they belong to, keyed by kind (`{"jsbsim": …, "ogma": …}`), so nothing is lost on a round
  trip through USD.

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
