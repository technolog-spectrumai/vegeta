"""Fixed inputs for the part modules moved from Vegeta; golden/parts.json holds what Vegeta's designs/ modules returned (787b988)."""
import dataclasses, json, sys
import numpy as np


def plain(x):
    if dataclasses.is_dataclass(x):
        return plain(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def cases(gears, tracks, soil, cutting, actuators):
    out = {}
    g = out["gears"] = {}
    g["lewis_y"] = [gears.lewis_y(z) for z in (12, 17, 20, 33, 60, 150)]
    g["lewis_bending"] = gears.lewis_bending(1000.0, 2.0, 20.0, 20)
    g["contact_stress"] = [gears.contact_stress(1000.0, 40.0, 20.0, 2.5), gears.contact_stress(800.0, 30.0, 15.0, 3.0, internal=True)]
    g["elastic_coefficient"] = gears.elastic_coefficient(210e3, 0.3, 210e3, 0.3)
    g["involute_profile"] = gears.involute_profile(2.0, 20)
    g["planetary"] = gears.planetary(12, 60, 3)
    g["stage_loads"] = gears.stage_loads(5.0, 1.0, 12, 3)
    g["chain_efficiency"] = gears.chain_efficiency(list(gears.MESH_EFFICIENCY)[:3])
    g["sn_cycles"] = gears.sn_cycles(300.0, 200.0, 1e6, 9.0)
    g["sprocket_chordal"] = gears.sprocket_chordal(11, 12.7)
    g["capstan"] = gears.capstan(0.3, 3.14)
    g["select_ratio"] = gears.select_ratio(stall_Nm=2.0, rated_Nm=0.6, no_load_rpm=4000, tau_climb=40.0, omega_climb=1.0,
                                           tau_flat=8.0, omega_top=6.0, efficiency=0.85)
    s = out["soil"] = {}
    soils = list(soil.SOILS) if isinstance(soil.SOILS, dict) else [x.name for x in soil.SOILS]
    s["soils"] = soils
    for name in soils:
        so = soil.soil(name)
        row = {"rigid_traction": soil.rigid_traction(1000.0, so), "rigid_rolling_resistance": soil.rigid_rolling_resistance(1000.0, so)}
        try:
            row.update(bekker_k=soil.bekker_k(0.15, so), sinkage=soil.sinkage(20e3, 0.15, so),
                       pressure_sinkage=soil.pressure_sinkage(0.03, 0.15, so), track_sinkage=soil.track_sinkage(3000.0, 0.15, 0.8, so),
                       compaction=soil.compaction_resistance_track(3000.0, 0.15, 0.8, so), max_thrust=soil.max_thrust_track(3000.0, 0.15, 0.8, so),
                       thrust=soil.thrust_track(3000.0, 0.15, 0.8, 0.2, so), drawbar=soil.drawbar_pull_track(3000.0, 0.15, 0.8, 0.2, so, 50.0),
                       shear=soil.shear_stress(20e3, 0.05, so), wheel_sinkage=soil.wheel_sinkage(500.0, 0.1, 0.4, so),
                       wheel_compaction=soil.wheel_compaction_resistance(500.0, 0.1, 0.4, so), thrust_wheel=soil.thrust_wheel(500.0, 0.1, 0.4, 0.2, so))
        except Exception as e:                       # rigid ground: the Bekker functions refuse it; record that they do
            row["bekker"] = type(e).__name__
        s[name] = row
    s["mmp_rowland"] = soil.mmp_rowland(3000.0, 6, 0.15, 0.2, 0.1)
    t = out["tracks"] = {}
    so = soil.soil(soils[0])
    t["ground_pressure"] = tracks.ground_pressure(3000.0, 0.15, 0.8)
    t["internal_coefficient"] = tracks.internal_coefficient(1.5)
    t["resistance"] = tracks.resistance(3000.0, b=0.15, L=0.8, soil=so, grade_deg=10.0, v=1.5, CdA=0.5)
    t["tractive_limit"] = tracks.tractive_limit(3000.0, b=0.15, L=0.8, soil=so, grade_deg=10.0)
    t["road_wheel_loads"] = tracks.road_wheel_loads(1500.0, [-0.3, -0.1, 0.1, 0.3], 0.02)
    t["road_wheel_loads_bogie"] = tracks.road_wheel_loads(1500.0, [-0.3, -0.1, 0.1, 0.3], 0.02, bogies=[(0, 1), (2, 3)])
    t["sag"] = tracks.sag(500.0, 2.0, 0.3)
    t["derail_tension"] = tracks.derail_tension(300.0, 0.3, 0.02)
    t["drive_tension"] = tracks.drive_tension(500.0, 800.0)
    t["lateral_coefficient"] = tracks.lateral_coefficient(2.0, 0.5, 0.6)
    t["skid_steer"] = tracks.skid_steer(3000.0, L=0.8, B=0.5, R=2.0, mu_t_max=0.6, rolling=0.05, v=1.0)
    t["min_turn_radius"] = tracks.min_turn_radius(3000.0, L=0.8, B=0.5, mu_t_max=0.6, rolling=0.05, F_max=1500.0)
    t["stability"] = tracks.stability(x_cg=0.0, z_cg=0.3, x_front=0.4, x_rear=-0.4, half_gauge=0.25, half_width=0.3, step_height=0.1,
                                      contact_front=0.45, contact_rear=-0.45)
    c = out["cutting"] = {}
    saw, wood = cutting.SAWS[0], cutting.WOODS[0]
    c["keys"] = [[x.key for x in cutting.SAWS], [x.key for x in cutting.WOODS]]
    c["get"] = [cutting.get_saw(cutting.SAWS[1].key).key, cutting.get_wood(cutting.WOODS[-1].key).key]
    c["saw"], c["wood"] = saw, wood
    c["cut_area"] = cutting.cut_area(0.1)
    c["chain_force"] = cutting.chain_force(saw)
    c["cut_rate_area"] = cutting.cut_rate_area(saw, wood)
    c["cut_time"] = cutting.cut_time(0.1, saw, wood)
    c["feed_force"] = cutting.feed_force(saw, wood)
    c["head_loads"] = [cutting.head_loads(saw, wood), cutting.head_loads(saw, wood, cut_from="bottom", dynamic=1.5)]
    c["branch"] = [cutting.branch_mass(0.1, 2.0, wood), cutting.branch_moment(0.1, 2.0, wood)]
    c["kerf_opens"] = [cutting.kerf_opens(0.1, 1.0, wood), cutting.kerf_opens(0.1, 1.0, wood, cut_from="bottom")]
    c["energy_per_cut"] = cutting.energy_per_cut(0.1, saw, wood, idle_s=2.0)
    c["cuts_per_charge"] = cutting.cuts_per_charge(340.0, 0.1, saw, wood, overhead_Wh=20.0)
    c["undercut_depth"] = cutting.undercut_depth(0.2)
    c["max_branch_for_bar"] = cutting.max_branch_for_bar(saw, 2)
    c["kickback_zone"] = cutting.kickback_zone(saw, 0.05)
    a = out["actuators"] = {}
    cat = actuators.CATALOG if isinstance(actuators.CATALOG, dict) else {v.key: v for v in actuators.CATALOG}
    a["catalog"] = {k: plain(v) for k, v in cat.items()}
    lin = actuators.LINEAR if isinstance(actuators.LINEAR, dict) else {v.key: v for v in actuators.LINEAR}
    a["linear"] = {k: plain(v) for k, v in lin.items()}
    jo = actuators.JOINTS if isinstance(actuators.JOINTS, dict) else {v.key: v for v in actuators.JOINTS}
    a["joints"] = {k: plain(v) for k, v in jo.items()}
    a["derived"] = {k: [v.holding_power(0.5 * v.rated_Nm), v.safety_factor(0.5 * v.rated_Nm)] for k, v in cat.items()}
    a["select"] = [actuators.select(5.0).key, actuators.select(50.0, kind=None, self_locking=True).key if any(v.self_locking for v in cat.values()) else None]
    return plain(out)


if __name__ == "__main__":                 # re-record: python parts_cases.py out.json (the modules live in Ogma now)
    from ogma import actuators, cutting, gears, soil, tracks
    json.dump(cases(gears, tracks, soil, cutting, actuators), open(sys.argv[1], "w"), indent=1)
