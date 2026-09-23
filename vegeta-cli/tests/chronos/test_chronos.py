import json
import math

import numpy as np
import pytest

from vegeta import chronos
from vegeta.chronos.cli import main


def test_rainflow_astm_example():
    # ASTM E1049-85 figure 6 example: the standard's counts (range, count)
    cycles = chronos.rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2])
    got = sorted((round(r), c) for r, _, c in cycles)
    assert got == sorted([(3, 0.5), (4, 0.5), (4, 1.0), (8, 0.5), (9, 0.5), (8, 0.5), (6, 0.5)])
    sine = 5 + 2 * np.sin(np.linspace(0, 2 * math.pi * 10, 2001))
    cyc = chronos.rainflow(sine)
    assert sum(c for _, _, c in cyc) == pytest.approx(10.0, abs=0.5)         # ten sine periods
    assert all(abs(r - 4) < 0.02 and abs(m - 5) < 0.02 for r, m, c in cyc if c == 1.0 or r > 3.9)
    assert chronos.rainflow([1, 1, 1]) == []


def test_structure_amplification_and_margin():
    st = chronos.Structure((100.0, 300.0), damping_ratio=0.05)
    assert st.amplification(100.0)[0] == pytest.approx(1 / (2 * 0.05))         # resonance: 1/(2 zeta)
    assert st.amplification(10.0)[0] == pytest.approx(1.0, abs=0.02)            # quasi-static
    assert st.amplification(1000.0)[0] < 0.15                                   # isolated
    assert st.nearest_mode(180.0) == 300.0 or st.nearest_mode(180.0) == 100.0   # log-nearest
    assert st.margin(120.0) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        chronos.Structure((0.0,))


def mission():
    unb = chronos.Excitation("1P", 100.0, 0.5, "lateral")
    return chronos.Mission("m", (
        chronos.Segment("takeoff", 10, {"thrust": 6.0}, (unb,)),
        chronos.Segment("hover", 100, {"thrust": 4.0}, (unb,)),
        chronos.Segment("punch", 2, {"thrust": 8.0}, repeat=5),
        chronos.Segment("land", 5, {"thrust": 2.0, "landing": 30.0}),
    ), "test")


def test_mission_and_spectrum(tmp_path):
    m = mission()
    assert m.duration_s == 125 and m.patterns == ["thrust", "lateral", "landing"] and len(m.expanded()) == 8
    st = chronos.Structure((100.0,), 0.05)
    spec = chronos.build_spectrum(m, st)
    vib = [b for b in spec.blocks if b.pattern == "lateral"]
    assert [b.cycles for b in vib] == [1000.0, 10000.0]                      # f x t per segment
    assert all(b.amplitude == pytest.approx(0.5 * 10) for b in vib)         # resonance x10
    assert vib[1].mean == 0.0                                                # lateral has no static level
    thrust = [b for b in spec.blocks if b.pattern == "thrust"]
    assert sum(b.cycles for b in thrust) == pytest.approx(6.0)               # 4 punch cycles + (6-4) + ground-air-ground
    assert max(b.amplitude for b in thrust) == pytest.approx(4.0)            # 0 -> 8 N half cycle
    p = spec.save(tmp_path / "s.json")
    back = chronos.LoadSpectrum.load(p)
    assert back.total_cycles == spec.total_cycles and json.loads(p.read_text())["tool"] == "vegeta.chronos"
    assert len(spec.table()) == len(spec.blocks)
    m.table(); m.profile(); spec.plot(); st.campbell({"1P": 1, "3P": 3}, np.linspace(1000, 9000, 5), operating_rpm={"hover": 6000})


def test_hotspot_damage_and_life():
    curve = chronos.SNCurve("t", sigma_f=100.0, b=-0.1, ultimate=200.0)
    assert curve.cycles_to_failure(100.0) == pytest.approx(0.5)
    assert curve.cycles_to_failure(50.0, 100.0) < curve.cycles_to_failure(50.0, 0.0)   # Goodman
    spec = chronos.LoadSpectrum("x", 60.0, [chronos.Block("a", 0.0, 10.0, 100.0, "one")], ["a"])
    d = chronos.hotspot_damage(spec, {"a": 5.0}, curve)
    assert d["total"] == pytest.approx(100.0 / curve.cycles_to_failure(50.0))
    with pytest.raises(ValueError):
        chronos.hotspot_damage(spec, {}, curve)
    sim = chronos.simulate_life({"a": 0.01, "b": 0.0}, {"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 1.0}, 300, seed=None)
    assert sim.flights.count("a") == 150 and sim.flights_to_failure == pytest.approx(199.0) and sim.hours_to_failure == pytest.approx(298.0)
    rnd = chronos.simulate_life({"a": 0.01, "b": 0.0}, {"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 1.0}, 300, seed=1)
    assert 150 < rnd.flights_to_failure < 260
    assert math.isinf(chronos.simulate_life({"a": 1e-9}, {"a": 1.0}, {"a": 1.0}, 10).flights_to_failure)
    sim.plot()


def test_cli(tmp_path, capsys):
    f = tmp_path / "m.py"
    f.write_text("from vegeta import chronos\nM = chronos.Mission('m', (chronos.Segment('h', 60, {'t': 1.0}, "
                 "(chronos.Excitation('e', 50.0, 1.0, 'l'),)),))\n")
    out = tmp_path / "s.json"
    assert main(["spectrum", f"{f}:M", "--modes", "50,200", "-o", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["blocks"] == 2 and out.is_file()
    assert main(["life", "--damage", "m=0.01", "--hours", "m=0.5", "--usage", "m=1", "--flights", "50", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["final_damage"] == pytest.approx(0.5)
    assert main(["spectrum", f"{f}:NOPE", "-o", str(out)]) == 2
