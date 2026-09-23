"""Performance maps and the JSON hand-off used by other notebooks and tools."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .airfoil import Airfoil
from .bemt import solve
from .propeller import Propeller
from .result import Result
from .system import excitations


def performance_map(prop: Propeller, airfoil: Airfoil, rpms, airspeeds, rho: float = 1.225) -> dict:
    """Thrust/torque/power/efficiency grids over rpm x airspeed (lists, JSON-ready)."""
    rpms, airspeeds = [float(x) for x in rpms], [float(x) for x in airspeeds]
    grids = {k: np.zeros((len(rpms), len(airspeeds))) for k in ("thrust_n", "torque_nm", "power_w", "efficiency", "ct", "cp")}
    unconverged = 0
    for i, rpm in enumerate(rpms):
        for j, v in enumerate(airspeeds):
            op = solve(prop, airfoil, rpm, v, rho)
            unconverged += not op.converged
            for k, val in (("thrust_n", op.thrust), ("torque_nm", op.torque), ("power_w", op.power),
                           ("efficiency", op.efficiency), ("ct", op.ct), ("cp", op.cp)):
                grids[k][i, j] = val
    return {"rpm": rpms, "airspeed_m_s": airspeeds, "rho": rho, "unconverged_points": unconverged,
            **{k: g.round(6).tolist() for k, g in grids.items()}}


def export(path: str | Path, prop: Propeller, airfoil: Airfoil, *, map: dict | None = None,
           points: dict | None = None, motor=None, battery=None, rho: float = 1.225, notes: str = "") -> Result:
    """Write ``path`` (JSON) with geometry, airfoil model, the map, named operating points and the
    excitation summary at each named point. The file is the hand-off to other tools."""
    t0 = time.monotonic()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    res = Result(kind="boreas.export", metadata={"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    doc = {"tool": "vegeta.boreas", "created_at": res.metadata["created_at"], "notes": notes,
           "propeller": prop.describe(), "airfoil": airfoil.__dict__, "rho": rho,
           "motor": motor.__dict__ if motor is not None else None,
           "battery": battery.__dict__ if battery is not None else None,
           "points": {}, "map": map}
    for name, p in (points or {}).items():
        rec = p.to_dict() if hasattr(p, "to_dict") else dict(p)
        rpm = rec.get("rpm") or rec.get("aero", {}).get("rpm")
        rec["excitation"] = excitations(prop, float(rpm)) if rpm else None
        doc["points"][name] = rec
        if rec.get("aero", rec).get("converged") is False:
            res.messages.append(f"point {name!r} did not converge")
    path.write_text(json.dumps(doc, indent=2, default=float))
    res.artifacts["json"] = path
    res.metrics = {"points": len(doc["points"]), "map_points": (len(map["rpm"]) * len(map["airspeed_m_s"])) if map else 0}
    res.duration_s = time.monotonic() - t0
    return res


def load(path: str | Path) -> dict:
    """Read a Boreas export."""
    return json.loads(Path(path).read_text())
