"""Kestrel's structure in Talos (kestrel_fea, notebook 40 § 8b / notebook 39): the frame and joint CAD build and export,
the cases prepare for every bracing, the NOT RUN table and the plane-frame reference; nothing is solved here.

Run: cd /home/user/vegeta && python3 -m pytest -q designs/tests/test_kestrel_fea.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "designs"))

import kestrel_fea as kfe
import kestrel_structure as kst

MASS = 1.686


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return tmp_path_factory.mktemp("kestrel_fea")


def test_frame_and_joint_cad_build(work):
    bb = {}
    for br in ("x", "posts", "none"):
        g = kfe.frame_cad(bracing=br)
        v = g.val()
        assert v.isValid()
        bb[br] = v.BoundingBox()
        assert kfe.frame_step(work / "cad", bracing=br).stat().st_size > 1000
    assert bb["x"].ylen == pytest.approx(1000.0, abs=1.0) and bb["x"].zlen > bb["none"].zlen        # the posts stand proud of the spars
    j = kfe.joint_cad().val()
    assert j.isValid() and j.Volume() < 36 * 26 * 30 and kfe.joint_step(work / "cad").exists()


def test_lift_segments_sum_to_the_lift_less_the_inertia():
    rows = kfe.lift_segments(n=4.0, mass_kg=MASS, upper_share=0.5)
    m = kfe.masses()
    total = sum(f for _, _, _, f in rows)
    assert total == pytest.approx(4.0 * MASS * 9.80665 - 4.0 * 9.80665 * m["wing"] * 2, rel=1e-6)
    up = sum(f for w, _, _, f in rows if w == "upper")
    assert up == pytest.approx(total / 2, rel=1e-6) and len(rows) == 4 * (kfe.N_SEG + 2)


def test_cases_prepare_for_every_bracing_and_the_table_lists_them(work):
    cs = kfe.cases(work, mass_kg=MASS, bracings=("x", "posts"))
    names = [c.name for c in cs]
    assert names == ["frame_x_A", "frame_x_B", "frame_x_C", "frame_x_modes", "frame_posts_A", "frame_posts_B", "frame_posts_C", "frame_posts_modes", "joint_post"]
    a = cs[0].model
    assert len(a.supports) == 2 and sum(1 for ld in a.loads if ld.region.startswith("lift_")) == 4 * (kfe.N_SEG + 2)
    assert all(ld.fx < 0 for ld in cs[2].model.loads) and len(cs[3].model.masses) == 4
    joint = cs[-1]
    assert joint.build_axis == "z" and joint.strength_z == 27.0 and joint.model.loads[0].fz < -2.0
    res = kfe.solve(cs, work / "fea", run=False)
    t = kfe.table(cs, res)
    assert len(t) == 9 and (t["status"] == "NOT RUN").all() and kfe.summary(t)["NOT RUN"] == 9
    doc = kfe.save(t, work / "fea.json", meta={"quality": "test"})
    assert doc["summary"]["NOT RUN"] == 9 and len(doc["rows"]) == 9
    assert kfe.post_force(mass_kg=MASS) > 2.0


def test_plane_frame_reference_matches_kestrel_structure():
    ref = kfe.plane_frame_reference(mass_kg=MASS)
    f = kst.frame(n=4.0, mass_kg=MASS, bracing="braced", upper_share=0.5)
    assert float(ref.loc["frame_x_A", "spar stress, limit [MPa]"]) == pytest.approx(f["spar stress [MPa]"])
    assert float(ref.loc["frame_x_A", "spar stress, ultimate [MPa]"]) == pytest.approx(1.5 * f["spar stress [MPa]"])
    assert float(ref.loc["frame_posts_modes", "first mode, symmetric [Hz]"]) < float(ref.loc["frame_x_modes", "first mode, symmetric [Hz]"])


def test_region_boxes_hold_whole_faces(tmp_path):
    """The bug of the first FEA campaign: every region selected no surface (an unsplit tube is one face). The spars
    are built in segments now; each region box of the frame and of the joint holds at least one whole face."""
    import kestrel_fea as kfe
    for br in ("x", "posts"):
        faces = kfe.frame_faces(bracing=br)
        cases = kfe.frame_cases(tmp_path / br, mass_kg=1.69, bracing=br)
        sel = kfe.regions_select(cases[0].model.regions, faces)
        assert all(v >= 1 for v in sel.values()), {k: v for k, v in sel.items() if v == 0}
        assert all(sel[k] >= 1 for k in ("pod_upper", "pod_lower", "M1", "M2", "M3", "M4"))
    jc = kfe.joint_cases(tmp_path / "joint", mass_kg=1.69)[0]
    faces = []
    for f in kfe.joint_cad().val().Faces():
        bb = f.BoundingBox()
        faces.append((bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax, f.geomType()))
    sel = kfe.regions_select(jc.model.regions, faces)
    assert sel["spar_bore"] >= 1 and sel["socket_floor"] == 1
    assert kfe.frame_cad(bracing="x").val().isValid() and kfe.joint_cad().val().isValid()
