"""The FEniCSx worker of the fenix book — runs under the fenicsx interpreter only (dolfinx, PETSc, MPI), one case per
process: reads a JSON spec, solves the plate with ``vegeta.hyperion.thermal_fenicsx`` (the steady state, or the
transient with melting and ablation), writes ``result.json`` (the numbers, the temperature field) into the case
directory. The chapter script (``plate_thermal.py``, in the Vegeta venv) launches it and reads the result back — the
process pipe between the two environments is a JSON file each way.

    .fenicsx/env/bin/python workflows/fenix/_worker.py spec.json case_dir
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from vegeta.hyperion import thermal as th
from vegeta.hyperion import thermal_fenicsx as tf


def main(spec_path: str, out_dir: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not tf.available():
        (out / "result.json").write_text(json.dumps({"status": "FAILED", "note": "dolfinx is not importable in this interpreter"}))
        return 1
    s = th.solid(spec["material"])
    t0 = time.monotonic()
    common = dict(thickness=spec["thickness"], power=spec["power"], radius=spec["radius"], shape=spec.get("shape", "tophat"), h=spec.get("h", 10.0),
                  nr=spec["nr"], nz=spec["nz"], r_max=spec.get("r_max"))
    if spec["mode"] == "steady":
        st = tf.steady_plate_fenicsx(s, **common)
        T = st.pop("T_field"); st.pop("mesh", None)
        vals = T.x.array
        res = {"status": "solved" if st["converged"] else "diverged", "process": "ok", "converged": st["converged"], "mode": "steady", **st,
               "T_values": vals.tolist(), "T_dofs": T.function_space.tabulate_dof_coordinates()[:, :2].tolist()}
    else:
        hr = tf.heat_plate_fenicsx(s, t_max=spec["t_max"], stop_at_failure=True, dt=spec.get("dt"), **common)
        d = hr.to_dict()
        res = {"status": "solved", "process": "ok", "converged": True, "mode": "transient", **d,
               "t": hr.t.tolist(), "T_top": hr.T_top.tolist(), "T_bottom": hr.T_bottom.tolist(), "T_max": hr.T_max.tolist(),
               "melt_depth": hr.melt_depth.tolist(), "ablated_depth": hr.ablated_depth.tolist(),
               "T_field": np.where(np.isnan(hr.T_field), None, hr.T_field).tolist(), "r": hr.r.tolist(), "z": hr.z.tolist()}
    res["seconds"] = round(time.monotonic() - t0, 1)
    res["material"] = s.name
    (out / "result.json").write_text(json.dumps(res, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
