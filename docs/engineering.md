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

## Materials, strength, connections — `00mat`, `00str`, `00conn`

*Stage 2 — written when implemented.*

## Aerodynamics — `00aero`

*Stage 3 — written when implemented.*

## Assemblies and detachable components — `00detach`

*Stage 4 — written when implemented.*
