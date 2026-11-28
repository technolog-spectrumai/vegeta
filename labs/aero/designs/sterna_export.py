"""STERNA and Talassa as Yax exchange files (``.yx2``), each with its description.

One link per aircraft: the mass of its standard loading, the inertia of the flight model, the CAD as its visual mesh
and a box hull for contact. ``data["description"]`` says in words what the aircraft is, what it is for, how it is
launched and recovered, and what the numbers rest on; ``data["sterna"]`` carries the numbers: geometry, mass by
group, the loadings, the aerodynamic coefficients the flight model uses, the drive and the model's results.

Frame: forward +x, up +z, the origin at the CG (the CAD's is x aft from the spinner tip). The propeller is data, not a
link. Nothing in the file is measured: see ``description.basis``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import sterna as st
import sterna_flight as am
import sterna_mass as sm
import sterna_model as mo

DESCRIPTION = {
    "sterna": {
        "title": "STERNA",
        "summary": "A 1.5 m electric fixed wing for coastal survey, environmental monitoring and maritime search support: a shoulder-wing tractor "
                   "with a folding propeller, a payload pallet in the belly just ahead of the centre of gravity, and plug-in wing panels.",
        "role": "Carries one camera pallet of up to 250 g (daylight, stabilised daylight, or daylight with thermal) on one or two 4S battery modules.",
        "operation": "One person carries it with the wings off. Hand launch with one battery module, a bungee line with two. Belly landing on two rails "
                     "in an open field; no water landing.",
        "inspiration": "The tern (a slender body, long lightly loaded outer panels) and the A6M Zero (mass discipline, large control surfaces).",
    },
    "talassa": {
        "title": "Talassa",
        "summary": "STERNA made bigger with a longer wing (first called STERNA-L): the fuselage, fin and rails 1.2 times the size and a 2.3 m wing of "
                   "aspect ratio 11, at STERNA's own wing loading limit.",
        "role": "Carries up to about 800 g of payload, or up to four battery modules for range, or small quadcopters (Cotesia, Vespula) on top of the wing.",
        "operation": "Three-piece wing; too long for STERNA's bag. Bungee line or catapult launch only. Belly landing on two rails in an open field.",
        "inspiration": "As STERNA; the longer, narrower wing is for reach at survey speeds.",
    },
}
BASIS = ("Mass: a budget of class values and CAD areas, not weighed parts. Aerodynamics: a vortex lattice with hand estimates for drag, damping "
         "and the lateral derivatives; STERNA's lift, drag and pitching moment have four coarse CFD cases behind them, Talassa's have none. "
         "Performance: model outputs. Nothing is built or flight tested.")


def export_yx2(name: str, v: dict, path, cfd_rows=None, cl_max_2d: float = 1.33, triangles: int = 20000) -> dict:
    """``v``: the aircraft as ``sims/sterna_run.variants()`` gives it (parameters, standard loading, drive, loadings)."""
    import pyvista as pv
    import yax
    from yax import hub
    p = st.resolve(v["over"])
    L = st.Sterna.layout(p)
    C = mo.coefficients(v["loading"], cl_max_2d=cl_max_2d, **v["over"])
    if cfd_rows:
        C = mo.calibrate(C, cfd_rows)
    b = C["mass_budget"]
    modules = b["battery"]["modules"]
    drive = mo.propulsion(modules=modules, **v["drive"])
    tt = mo.thrust_table(drive)
    sp = mo.speeds(C)
    cr = mo.cruise_point(C, drive, 14.0)
    cg = np.array([C["x_cg_m"], 0.0, C["z_cg_m"]])
    flip = np.diag([-1.0, -1.0, 1.0])
    with tempfile.TemporaryDirectory() as tmp:
        stl = Path(tmp) / "body.stl"
        st.Sterna().generate(**v["over"]).export_stl(str(stl), tolerance=0.2)
        mesh = pv.read(str(stl)).triangulate().clean()
    if mesh.n_cells > triangles:
        mesh = mesh.decimate(1 - triangles / mesh.n_cells)
    pts = (np.asarray(mesh.points) / 1000.0 - cg) @ flip.T
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    k = p["scale"]
    length, half_w, depth = L["length"] / 1e3, 0.050 * k, 0.125 * k
    link = yax.Link(name="airframe", mass=float(C["mass"]), com=(0.0, 0.0, 0.0), inertia=np.diag([float(x) for x in C["inertia"]]), shapes=[
        hub.Shape(name="cad", kind="mesh", role="visual", points=pts, faces=faces, uri=f"vegeta designs:sterna.Sterna ({name})", rgba=(0.93, 0.94, 0.95, 1.0)),
        hub.Shape(name="fuselage", kind="box", role="collision", size=(length / 2, half_w, depth / 2), pos=(float(cg[0] - length / 2), 0.0, -0.0125 * k))])
    items = {}
    for it in b["items"]:
        items[it["group"]] = round(items.get(it["group"], 0.0) + it["g"] / 1e3, 4)
    loadings = []
    for ld in v["loadings"]:
        bb = sm.budget(p, L, ld)
        loadings.append({"name": bb["case"], "payload_kg": bb["pallet"]["g"] / 1e3, "battery_modules": bb["battery"]["modules"], "takeoff_kg": bb["total_g"] / 1e3,
                         "cg_from_nose_m": bb["x_cg_mm"] / 1e3, "cg_fraction_of_mean_chord": bb["cg_mac"], "within_design_mass": bool(bb["within_mtow"])})
    desc = {**DESCRIPTION[name], "basis": BASIS,
            "key_figures": {"span_m": p["span"] / 1e3, "length_m": length, "wing_area_m2": L["S_mm2"] / 1e6, "aspect_ratio": round(L["AR"], 2),
                            "design_mass_limit_kg": p["mtow_g"] / 1e3, "empty_kg": b["empty_g"] / 1e3, "standard_loading": b["case"], "standard_takeoff_kg": float(C["mass"]),
                            "stall_m_s": round(sp["V_stall"], 2), "best_lift_to_drag": round(sp["LD_max"], 1), "cruise_m_s": 14.0,
                            "cruise_minutes_model": round(cr["minutes"], 0), "cruise_range_km_model": round(cr["minutes"] * 60 * 14.0 / 1000, 0)}}
    data = {"description": desc,
            "sterna": {"geometry_m": {"span": p["span"] / 1e3, "length": length, "height": L["height"] / 1e3, "wing_area_m2": L["S_mm2"] / 1e6, "aspect_ratio": L["AR"],
                                      "mean_chord": L["mac"] / 1e3, "root_chord": p["root_chord"] / 1e3, "tip_chord": p["tip_chord"] / 1e3, "centre_span": p["centre_span"] / 1e3,
                                      "dihedral_outer_deg": p["dihedral_deg"], "washout_deg": p["washout_deg"], "tailplane_area_m2": L["S_h_mm2"] / 1e6, "fin_area_m2": L["S_v_mm2"] / 1e6,
                                      "tail_volume_horizontal": L["V_h"], "tail_volume_vertical": L["V_v"], "propeller_diameter": p["prop_diameter"] / 1e3},
                       "mass_kg": {"design_limit": p["mtow_g"] / 1e3, "empty": b["empty_g"] / 1e3, "groups": items, "battery_module": sm.MODULE_G / 1e3,
                                   "battery_module_Wh": sm.MODULE_WH, "payload_station_from_nose_m": sm.PALLET_X * k / 1e3},
                       "loadings": loadings,
                       "coefficients": {kk: float(C[kk]) for kk in ("S", "c", "b", "CL0", "CLa", "CL_MAX", "alpha_stall", "CD0", "k", "Cm0", "Cma", "Cmq", "CLde", "Cmde",
                                                                   "CYb", "Clb", "Cnb", "Clp", "Cnr", "Clda", "Cndr", "x_cg_m", "x_np_m", "static_margin")},
                       "coefficients_from": "CFD fit for lift, drag and pitching moment; lattice and hand for the rest" if cfd_rows else "lattice and hand estimates",
                       "drive": {"motor": drive.motor.name, "propeller": next((getattr(getattr(drive, n), "name") for n in ("propeller", "prop", "rotor") if hasattr(drive, n)), None), "battery_modules": modules,
                                 "static_thrust_N": tt["T0_N"], "static_current_A": tt["table_A"][0], "thrust_table": {"V_m_s": tt["table_V"], "T_N": tt["table_T"]},
                                 "propeller_pos": [float(cg[0] - 0.024 * k), 0.0, 0.0], "propeller_axis": [1.0, 0.0, 0.0]},
                       "model": {"stall_m_s": sp["V_stall"], "best_glide_m_s": sp["V_best_glide"], "LD_max": sp["LD_max"], "cruise_m_s": 14.0, "cruise_power_W": cr["total_W"],
                                 "cruise_minutes": cr["minutes"], "cruise_range_m": cr["minutes"] * 60 * 14.0, "usable_Wh": cr["usable_Wh"]}},
            "provenance": {"design": "labs/aero/designs/sterna.py" + ("" if name == "sterna" else " with labs/aero/designs/talassa.py"), "notebook": "labs/aero/notebooks/51_sterna.ipynb",
                           "notes": BASIS}}
    m = yax.Machine(name=name, links=[link], joints=[], data=data, producer={"tool": "vegeta labs/aero sterna_export", "machine": name, "lab": "aero"})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = yax.save(m, path)
    back, _ = yax.read(path)
    return {"file": str(path), "bytes": path.stat().st_size, "report": str(report), "mass_kg": float(back.links[0].mass),
            "triangles": int(len(back.links[0].shapes[0].faces)), "description": bool(back.data.get("description"))}
