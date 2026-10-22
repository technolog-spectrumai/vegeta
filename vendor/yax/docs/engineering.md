# The engineering contract — what a machine's blocks hold

Yax transports engineering truth; it does not compute it. Each **block** below is JSON part data under a documented key
on the machine (`data[key]`), a link (`link.data[key]`) or a joint (`joint.data[key]`) — the hub's existing
`yax:data` mechanism, so every block survives USD and YX2 byte for byte (`yax:data` strings are canonical JSON).
`yax.schema` is the oracle: `problems(m)` lists what is wrong with a block, `report_losses` what a target format
cannot carry of it; `yax validate FILE` runs both the schemas and the tag syntax. The expected export of a known
object follows from this page alone, which is what the mock tests compare against.

## Rules every block follows

- **Units are in the field's name**: `stall_torque_nm`, `no_load_speed_rpm`, `density_kg_m3`, `area_m2`. The producer
  keeps the unit it knows (a datasheet's 6000 rpm stays `6000` under `_rpm`; Yax converts nothing). A number whose
  name carries no unit must be in the dimensionless list (`schema.DIMLESS`: `gear_ratio`, `efficiency`, `CT`, `nu`, …).
  Suffixes: `_nm _n _kg _kgm2 _kg_m3 _m _m2 _m3 _m_s _rad _rad_s _rpm _deg _v _a _ohm _w _s _hz _pa _mpa _gpa
  _pa_sqrt_m _c _k _rpm_per_v _nm_per_a _v_s_per_rad _j _wh _ah _per_k _nm_per_rad _n_per_m`.
- **Frames**: positions and axes are in the machine frame at q = 0 unless the block says `"frame": "link:<name>"`.
- **Identities**: links, joints, shapes, materials, motors, propellers, connections and components are referred to by
  name; Yax checks that a reference names something that exists.
- **Provenance and validity**: any block or entry may carry `source` (where the numbers come from) and `validity`
  (`{"<field>": [lo, hi], "note": "…"}`: the ranges they hold in). Neither is ever filled in by Yax.
- **Unknown fields are kept**: a reader keeps what it does not know; a producer may say more than the schema, never
  less than the required fields.
- **Nothing is computed**: Yax does not derive rated from stall, kt from kv, G from E and ν, or a constant from a
  table. If a value is not in the source it is absent, and the inventory says so.
- **Validation is layered and labelled**: tag syntax and the `00` namespace (`yax.tags.check_code`), required tags
  (`yax.tags.check`), required fields, types, units, references and frames (`yax.schema.problems`), the file profile
  (`yax.formats.yx2.validate`). Physical consistency (does the curve match stall and no-load? is the inertia
  positive-definite with the mass?) is **not** validated: a tag or a passing schema is no proof of physics.

## Motors and actuators — `00motor`

A joint's `data["motor"]` is the motor that drives it, or the name of one in the machine's `data["motors"]` catalogue
(several joints, one motor). The joint's `drive` (the PD a simulator runs) stays what it was: it is a control law,
not the motor; the motor block is the physics behind it.

| Field | Meaning |
|---|---|
| `name`, `kind` | identity and class (`bldc`, `brushed`, `servo`, `gearmotor`, `qdd`, …) |
| `stall_torque_nm`, `rated_torque_nm` | peak and continuous (thermal) torque at the output |
| `no_load_speed_rpm` **or** `no_load_speed_rad_s`; `rated_speed_rpm` or `_rad_s` | one of each, as the datasheet gives it |
| `voltage_v`, `stall_current_a`, `no_load_current_a`, `rated_current_a`, `max_current_a` | electrical operating point |
| `resistance_ohm`, `inductance_h`, `kv_rpm_per_v`, `kt_nm_per_a`, `ke_v_s_per_rad` | winding and motor constants |
| `rotor_inertia_kgm2`, `mass_kg` | the rotor's own inertia (not reflected), the unit's mass |
| `electrical_time_constant_s`, `mechanical_time_constant_s`, `response_time_s` | response |
| `torque_speed` | the nonlinear curve: `{"speed_rpm" or "speed_rad_s": [...], "torque_nm": [...], "current_a"?: [...], "power_w"?: [...], "efficiency"?: [...], "voltage_v"?: number, "source"?}` — columns of one length |
| `limits` | `{"max_torque_nm", "continuous_torque_nm", "max_speed_rpm" or `_rad_s`, "max_power_w", "max_current_a", "max_voltage_v", "max_temperature_c", "duty_cycle"}` |
| `transmission` | `{"type": "planetary" …, "gear_ratio", "efficiency" (0, 1], "backlash_rad", "stiffness_nm_per_rad", "max_torque_nm", "max_speed_rpm", "reflected_inertia_kgm2", "self_locking", "stages": [...], "source"}` — the output side's numbers are the motor's × the transmission, stated by the producer, not computed here |
| `control` | what the producer's own controller used (`kp`, `kd`, …): informative |
| `source`, `validity` | provenance and ranges |

What a format keeps: USD and YX2 everything; **MJCF** writes the joint's PD drive and nothing of the motor
("joint X: its motor model (…) — mjcf keeps nothing of the motor: the PD drive written is not the motor model");
**URDF** the drive's max force as the effort limit; **SDF** the joint's limits and damping (a plugin would carry the
motor). Each is a reported loss, never a silent one.

## Propulsion — `00prop` (its motors also `00motor`)

The machine's `data["propulsion"]` (see also docs/converter.md for the JSBSim writer):

- `rotors`: `[{"name", "link"?, "pos" [m, machine frame], "axis" (unit), "spin" ±1 (+1: drag torque on the frame along
  +axis), "propeller": name, "motor": name}]`
- `propellers`: `{name: {"blades", "diameter_m" (required), "hub_radius_m", "pitch_m", "stations": [{"r_m", "chord_m",
  "beta_deg"}], "mass_kg", "rotor_mass_kg", "inertia_kgm2" (about the shaft), "geometry_ref": a link or shape name
  holding the CAD, "coefficients": {"J", "CT", "CP"?, "CQ"?, "eta"? (columns of one length, J increasing),
  "rho_ref_kg_m3"?, "rpm_band"?, "source"?}, "grid": {"rpm", "airspeed_m_s", "rho", "thrust_n", "torque_nm"?,
  "power_w"?, "current_a"?, "voltage_v"?: rpm-major tables}, "source", "validity"}}` — CT = T / ρ n² D⁴,
  CP = P / ρ n³ D⁵, CQ = Q / ρ n² D⁵, J = V / n D, n in rev/s
- `motors`: `{name: motor block as above}` (the electric drive behind each rotor: `power_w` is what a JSBSim engine takes)
- `reduced`: `{rotor: {"kF", "kM", "w_max", "tau_up", "tau_down", "taken_at", "residual"}}` — the two-constant model
  Gazebo and PX4 take, with where it was taken and how far it is from the map
- `battery`, `source`, `validity`

What a format keeps: USD and YX2 everything; **JSBSim** the coefficients against J and the motors' power (a propeller
file and an electric engine per rotor; the grids, stations and validity do not reach it — reported); **SDF** nothing
in the file (the reduced constants reach a motor plugin through `yax sync`; reported); MJCF, URDF, STEP, STL nothing
(reported).

## Materials — `00mat`

The machine's `data["materials"]` is a catalogue `{name: material}`; a link's `data["material"]` is a name in it (the
usual) or a material of its own. A material is **intrinsic**: what the stuff is, whatever part is made of it.

| Field | Meaning |
|---|---|
| `name` (= its key), `class` | identity (`"al6061-t6"`, `"cfrp-t300-epoxy"`), class (`metal`, `polymer`, `composite`, `wood`, …) |
| `density_kg_m3`, `youngs_modulus_pa`, `poisson_ratio` (−1, 0.5), `shear_modulus_pa` | elastic constants — `G` is never derived from `E` and ν here |
| `yield_strength_pa`, `ultimate_strength_pa`, `compressive_strength_pa`, `shear_strength_pa`, `fracture_toughness_pa_sqrt_m`, `elongation_at_break`, `hardness` | strength and toughness, as the datasheet or test gives them |
| `anisotropy` | `isotropic`, `orthotropic`, `transversely isotropic`, `anisotropic` |
| `orthotropic` | `{"E1_pa", "E2_pa", "E3_pa", "G12_pa", "G13_pa", "G23_pa", "nu12", "nu13", "nu23", "Xt_pa", "Xc_pa", "Yt_pa", "Yc_pa", "S_pa"}` in the material's own axes |
| `composite` | `{"fibre", "matrix", "fibre_volume_fraction", "layup": [{"angle_deg", "thickness_m"?, "material"?}, …] (bottom to top), "ply_thickness_m", "reference_axis": [x, y, z], "frame": "link:<name>" — the axis angles count from, "source"}` |
| `failure` | an **optional, calibrated** criterion: `{"criterion": "von_mises" / "max_stress" / "tsai_wu" / "hashin" / …, "parameters": {…}, "calibrated": bool, "source", "validity"}` — reported as given; Yax computes no failure |
| `thermal`, `source`, `validity` | expansion and conductivity if known; provenance; ranges |

## Strength — `00str`

A link's `data["strength"]` (or the machine's) is **component-specific**: what *this part* is allowed, as sized or
as analysed — not what its material can take. `{"material": a catalogue name, "allowable_stress_pa",
"allowable_strain", "safety_factor", "limit_load_n", "ultimate_load_n", "limit_moment_nm", "ultimate_moment_nm",
"fatigue": {…}, "analysis": {"tool", "kind" (static / modal / fatigue / impact), "case", "max_stress_pa",
"max_displacement_m", "margin", "date", "source"}, "source", "validity"}`. A static FEA result is a static result:
it is not a validated impact or fracture threshold, and nothing here turns it into one.

## Connections — `00conn`

The machine's `data["connections"]` is `{name: connection}`: how two links are joined, physically.

| Field | Meaning |
|---|---|
| `name` (= its key), `type` | `bolt`, `screw`, `rivet`, `weld`, `bond`, `pin`, `press`, `snap`, `clamp`, `magnet`, `detachable`, `other` |
| `links` | the two link names it joins (identities) |
| `frame`, `pos`, `quat`, `axis` | where it is: `frame` is `machine` (default) or `link:<name>`; `pos` [m], `quat` (w, x, y, z), `axis` the fastener or weld axis |
| `geometry` | `{"diameter_m", "length_m", "area_m2", "width_m", "thickness_m", "pattern": [[x, y, z], …] (fastener positions in `frame`), "count", "shape"}` |
| `fasteners` | `{"count", "size": "M3", "grade": "8.8", "material": a catalogue name, "preload_n", "torque_nm", "pitch_m", "source"}` |
| `material` | the weld or adhesive, a catalogue name |
| `limits` | `{"tension_n", "compression_n", "shear_n", "moment_nm", "torque_nm", "energy_j", "safety_factor", "basis": "datasheet" / "test" / "analysis" / "assumed"}` — the connection's own strength, stated by the producer |
| `detachable` | `true` when the design means it to come off (stage 4 says how; the `assembly` block refers to it) |
| `source`, `validity` | provenance and ranges |

What a format keeps of materials, strength and connections: USD and YX2 everything; every other format nothing, each
reported ("data:connections: the connections (2 entries) — sdf has no place for it").

## Aerodynamics — `00aero`

*Stage 3 — written when implemented.*

## Assemblies and detachable components — `00detach`

*Stage 4 — written when implemented.*
