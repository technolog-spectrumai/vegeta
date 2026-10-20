"""Robotics reference results captured in Vegeta before the move to Chiron/Piccolo (docs/migration). The same
cases, with the same golden values, run in Chiron and Piccolo after the move: equal numbers prove the move kept
behaviour. Tiny models, short horizons."""
import pytest

mujoco = pytest.importorskip("mujoco")


def test_chiron_box_settles_on_its_feet(golden):
    from vegeta.chiron import ChironLab, Flat, FootSpec, Geom, Link, Robot
    geoms = [Geom("box", "box", (0.15, 0.1, 0.03), mass=3.0, role="body")]
    feet = []
    for i, (x, y) in enumerate(((0.12, 0.08), (0.12, -0.08), (-0.12, 0.08), (-0.12, -0.08))):
        geoms.append(Geom(f"pad{i}", "sphere", (0.02,), pos=(x, y, -0.05), mass=0.05, role="foot",
                          friction=(0.8, 0.005, 0.0001)))
        feet.append(FootSpec(f"pad{i}", f"pad{i}", [], "box"))
    robot = Robot("box", Link("box", geoms=geoms, log=True), feet=feet)
    lab = ChironLab(robot, Flat())
    lab.reset(base_pos=(0.0, 0.0, 0.12))
    for _ in range(300):
        lab.step()
    obs = lab.observe(sync=True)
    golden("robotics.chiron_box_300_steps", {"mass": robot.total_mass(), "base_pos": obs.base_pos}, rel=1e-6, abs=1e-9)


def test_races_reference_quad_and_mock_lap(golden):
    from races import courses, mock, refquad
    s = refquad.summary()
    golden("robotics.races_refquad", {k: v for k, v in s.items() if isinstance(v, float)})
    st = mock.run_mock(courses.hangar())["stats"]
    golden("robotics.races_mock_hangar", {k: st[k] for k in ("status", "lap time [s]", "energy [Wh]", "collisions")},
           rel=1e-9)


def test_races_mujoco_second_with_tuned_autopilot(golden):
    from races import autopilot, courses, sim
    r = sim.run_mujoco(courses.hangar(), control_dt=0.004, duration=1.0, **autopilot.tuned("hangar"))
    golden("robotics.races_mujoco_hangar_1s", {"last": r["history"][-1], "rows": len(r["history"])}, rel=1e-6, abs=1e-9)


def test_onboard_mission_closed_loop(golden):
    import vegeta.mission as vm
    from vegeta.mission.camera import IMX219_6MM
    from vegeta.mission.guidance import GuidanceConfig
    from vegeta.mission.sim import BirdField, BirdGroup, run_kinematic
    birds = BirdField([BirdGroup("black-headed gull", 3, (0, 250), "thermal", (55, 80), 35)], seed=3)
    cfg = vm.MissionConfig(camera=IMX219_6MM, guidance=GuidanceConfig(search_centre=(0, 250)))
    golden("robotics.mission_kinematic_20s", run_kinematic(birds, cfg, duration=20.0, seed=3).scores())
