import json

import cadquery as cq
import pytest

from vegeta.dedalus import BuildError, Design, Geometry, Parameter, design, load_step
from vegeta.dedalus.examples import Bracket, CantileverBeam, Cube, StreamlinedBody


def test_box_measurements_exact():
    g = CantileverBeam().generate(length=100, width=20, height=10)
    m = g.measure()
    assert m["valid"] and m["n_solids"] == 1 and m["n_faces"] == 6
    assert m["volume"] == pytest.approx(100 * 20 * 10)
    assert m["surface_area"] == pytest.approx(2 * (100 * 20 + 100 * 10 + 20 * 10))
    assert m["dimensions"] == pytest.approx([100, 20, 10])
    assert m["bbox_min"] == pytest.approx([0, -10, -5])
    assert m["center_of_mass"] == pytest.approx([50, 0, 0], abs=1e-9)


def test_bracket_holes_reduce_volume():
    import math

    p = Bracket().resolve()
    g = Bracket().generate(fillet=0.0)
    solid = p["length"] * p["width"] * p["thickness"]
    holes = 4 * math.pi * (p["hole_diameter"] / 2) ** 2 * p["thickness"]
    assert g.volume == pytest.approx(solid - holes, rel=1e-6)


def test_streamlined_body_is_valid_solid():
    g = StreamlinedBody().generate()
    assert g.measure()["valid"] and g.volume > 0
    assert g.dimensions[0] == pytest.approx(100, rel=1e-3)
    assert g.dimensions[1] == pytest.approx(20, rel=1e-2)


def test_step_and_stl_round_trip(tmp_path):
    g = Cube().generate(size=15)
    res = g.export(tmp_path)
    assert res.ok, res.messages
    assert res.artifacts["step"].stat().st_size > 0 and res.artifacts["stl"].stat().st_size > 84
    back = load_step(res.artifacts["step"])
    assert back.volume == pytest.approx(15 ** 3, rel=1e-9)
    # binary STL: 80-byte header + uint32 triangle count
    data = res.artifacts["stl"].read_bytes()
    n = int.from_bytes(data[80:84], "little")
    assert n >= 12 and len(data) == 84 + 50 * n
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "success" and summary["metadata"]["parameters"] == {"size": 15.0}


def test_decorator_design_and_source_identity():
    @design(parameters=[Parameter("r", 5.0, "mm", min=0.1)])
    def ball(p):
        return cq.Workplane().sphere(p["r"])

    g = ball.generate(r=2.0)
    import math

    assert g.volume == pytest.approx(4 / 3 * math.pi * 8, rel=1e-6)
    ident = ball.source_identity()
    assert ident["design"] == "ball" and len(ident["source_sha256"]) == 64
    assert ident == ball.source_identity()


def test_source_hash_changes_with_code():
    class A(Design):
        parameters = [Parameter("s", 1.0)]

        def build(self, p):
            return cq.Workplane().box(p["s"], 1, 1)

    class B(Design):
        parameters = [Parameter("s", 1.0)]

        def build(self, p):
            return cq.Workplane().box(p["s"], 2, 1)

    assert A().source_identity()["source_sha256"] != B().source_identity()["source_sha256"]


def test_build_failure_raises_clear_error_and_run_returns_failed(tmp_path):
    with pytest.raises(BuildError, match="Bracket"):
        Bracket().generate(hole_diameter=30.0)
    res = Bracket().run(tmp_path, hole_diameter=30.0)
    assert res.status == "failed" and "hole_diameter" in res.messages[0]
    assert "traceback" in res.metadata


def test_run_with_invalid_parameter_is_failed_result(tmp_path):
    res = Cube().run(tmp_path, size=-1.0)
    assert not res.ok and "below minimum" in res.messages[0]


def test_non_solid_reports_no_volume():
    face = cq.Workplane().rect(10, 10).val()  # a wire, make a face
    g = Geometry(cq.Face.makeFromWires(face))
    m = g.measure()
    assert m["volume"] is None and m["surface_area"] == pytest.approx(100)
    assert any("no closed solids" in s for s in g.messages)


def test_plots_return_figures():
    import matplotlib

    matplotlib.use("Agg")
    from vegeta.dedalus import measurements_table, plot_parameter_study, plot_views

    gs = [Cube().generate(size=s) for s in (5.0, 10.0, 15.0)]
    assert plot_views(gs[0]) is not None
    assert plot_parameter_study(gs, "size", "volume") is not None
    df = measurements_table(gs)
    assert list(df["volume"].round(6)) == [125.0, 1000.0, 3375.0]
