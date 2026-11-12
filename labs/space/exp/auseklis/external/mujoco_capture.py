"""Prepared MuJoCo capture scenario: the servicer approaches a tumbling debris object, the capture point closes (a soft
weld constraint switched on at the moment of contact) and the coupled body rate and constraint force are logged.

    python mujoco_capture.py --variant A --debris 500 --out runs/capture_A_500.xml          # writes the MJCF, runs nothing
    python mujoco_capture.py --variant A --debris 500 --out runs/capture_A_500.xml --run    # needs `pip install mujoco`

PRELIMINARY, needs external verification: a rigid-body weld with tuned softness stands in for the gripper contact;
inertias come from ``cad.mass_properties`` and a uniform cylinder; no flexible appendages, no sloshing, no wheels.
The analytic cross-checks (momentum conservation, 1-D impulse) are in ``auseklis.capture`` and run in the tests."""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402


def build_mjcf(variant: str = "A", debris_kg: int = 500, v_close: float = 0.05, standoff: float = 0.5,
               softness=(0.02, 1.0)) -> str:
    """MJCF text for the scenario (pure string building: no MuJoCo import)."""
    from auseklis import mission as M
    from auseklis.cad import mass_properties
    mp = mass_properties(variant=variant)
    d = M.DEBRIS[debris_kg]
    I_s = np.diag(mp["inertia_kgm2"])
    I_d = np.array([0.5 * d.mass * d.radius ** 2] + [d.mass * (3 * d.radius ** 2 + d.length ** 2) / 12] * 2)
    w0 = math.radians(d.tumble_deg_s)
    reach = 0.5 * 0.9 + (1.0 if variant == "A" else 0.6)                 # capture point ahead of the servicer CoM [m] (rough)
    sep = reach + standoff + d.length / 2
    f = lambda a: " ".join(f"{x:.6g}" for x in a)
    return f"""<mujoco model="auseklis_capture_{variant}_{debris_kg}">
  <compiler angle="radian"/>
  <option gravity="0 0 0" timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <body name="servicer" pos="0 0 0">
      <freejoint name="sat_free"/>
      <inertial pos="0 0 0" mass="{mp['mass_kg']:.4f}" diaginertia="{f(I_s)}"/>
      <geom type="box" size="0.45 0.4 0.425" contype="0" conaffinity="0" rgba="0.8 0.8 0.85 1"/>
      <site name="grip" pos="{reach:.4f} 0 0" size="0.03"/>
    </body>
    <body name="debris" pos="{sep:.4f} 0 0">
      <freejoint name="deb_free"/>
      <inertial pos="0 0 0" mass="{d.mass:.4f}" diaginertia="{f(I_d)}"/>
      <geom type="cylinder" size="{d.radius:.4f} {d.length / 2:.4f}" euler="0 1.5707963 0" contype="0" conaffinity="0" rgba="0.7 0.3 0.2 1"/>
      <site name="fixture" pos="{-d.length / 2:.4f} 0 0" size="0.03"/>
    </body>
  </worldbody>
  <equality>
    <weld name="capture" site1="grip" site2="fixture" active="false" solref="{softness[0]} {softness[1]}"/>
  </equality>
  <!-- initial state (set by run()): debris velocity -{v_close} m/s along x, spin {w0:.5f} rad/s about y, servicer at rest -->
</mujoco>
"""


def run(xml: str, v_close: float, tumble_deg_s: float, t_capture: float = None, t_end: float = 40.0) -> dict:
    """Run the scenario (imports MuJoCo lazily)."""
    if os.environ.get("VEGETA_DEBUG", "").lower() in ("1", "true", "on", "yes") or os.environ.get("VEGETA_NO_SOLVERS"):
        raise RuntimeError("mocks/no-solvers mode: the real MuJoCo run is refused (numbers would mean nothing)")
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    deb = m.joint("deb_free").qveladr[0]
    d.qvel[deb] = -v_close
    d.qvel[deb + 4] = math.radians(tumble_deg_s)
    sep0 = d.qpos[m.joint("deb_free").qposadr[0]] if False else 0.0
    log, captured = [], False
    while d.time < t_end:
        mujoco.mj_step(m, d)
        gap = float(np.linalg.norm(d.site("grip").xpos - d.site("fixture").xpos))
        if not captured and gap < 0.05:
            d.eq_active[0] = 1
            captured = True
        log.append((d.time, gap, float(np.abs(d.efc_force).max()) if d.nefc else 0.0, np.linalg.norm(d.qvel[3:6])))
    log = np.array(log)
    return {"captured": captured, "peak_constraint_force_N": float(log[:, 2].max()), "final_servicer_rate_rad_s": float(log[-1, 3])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", default="A", choices="ABC")
    ap.add_argument("--debris", type=int, default=500, choices=(100, 250, 500))
    ap.add_argument("--v-close", type=float, default=0.05, help="closing speed at capture [m/s] (conservative limit 0.02-0.1)")
    ap.add_argument("--out", default="runs/capture.xml")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args(argv)
    from auseklis import mission as M
    xml = build_mjcf(a.variant, a.debris, a.v_close)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    print(f"wrote {out} ({len(xml)} bytes); not run")
    if a.run:
        print(run(xml, a.v_close, M.DEBRIS[a.debris].tumble_deg_s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
