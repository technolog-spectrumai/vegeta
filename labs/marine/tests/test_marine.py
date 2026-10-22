"""Marine: the survey boat and the submarine build as CAD (Dedalus) and as assemblies with their propellers."""
import pytest

cq = pytest.importorskip("cadquery")
from vegeta import dedalus  # noqa: E402


@pytest.mark.parametrize("spec", ["survey_boat:SurveyBoat", "submarine:Submarine",
                                  "survey_boat_assembly:SurveyBoatAssembly", "submarine_assembly:SubmarineAssembly"])
def test_marine_designs_build(spec):
    design = dedalus.load_design(spec)
    p = design.resolve()
    if hasattr(design, "parts"):                                     # an assembly: its parts placed
        assert design.parts(p), spec
    shape = design.build(p)
    bb = (shape.val() if hasattr(shape, "val") else shape).BoundingBox()
    assert bb.xlen > 0 and bb.ylen > 0 and bb.zlen > 0
