import numpy as np
from _helpers import box_part

from vegeta.fidia.checks import FAIL, PASS, WARN, check_export, check_parts, summarize


def by_name(checks, name, part=None):
    return [c for c in checks if c.name == name and (part is None or c.part == part)][0]


def test_seam_merge_makes_a_box_watertight():
    p = box_part()
    assert not p.to_trimesh(merge=False).is_watertight  # CadQuery-style repeated vertices are open
    cs = check_parts([p])
    assert by_name(cs, "watertight").status == PASS
    assert by_name(cs, "winding").status == PASS
    assert summarize(cs)["valid"]


def test_open_mesh_fails():
    p = box_part()
    p.triangles = p.triangles[:-2]
    cs = check_parts([p])
    assert by_name(cs, "watertight").status == FAIL
    assert not summarize(cs)["valid"]


def test_flipped_winding_fails():
    p = box_part()
    p.triangles[0] = p.triangles[0][::-1]
    assert by_name(check_parts([p]), "winding").status == FAIL


def test_inside_out_fails():
    p = box_part()
    p.triangles = p.triangles[:, ::-1].copy()
    c = by_name(check_parts([p]), "winding")
    assert c.status == FAIL and "inside out" in c.message


def test_invalid_brep_fails():
    p = box_part()
    p.brep = {"valid": False, "n_solids": 0, "volume_mm3": 0.0}
    cs = check_parts([p])
    assert {by_name(cs, n).status for n in ("brep_valid", "solid", "volume")} == {FAIL}


def test_size_against_plan():
    p = box_part(size=(100, 200, 300))
    assert by_name(check_parts([p], {"size_mm": [100, 200, 300]}), "size").status == PASS
    assert by_name(check_parts([p], {"size_mm": [200, 100, 300]}), "size").status == PASS  # turned about Z
    c = by_name(check_parts([p], {"size_mm": [100, 200, 600]}), "size")
    assert c.status == WARN and "50 % off" in c.message


def test_floating_part_warns():
    a = box_part("base", size=(100, 100, 10))
    b = box_part("lid", size=(20, 20, 5), offset=(0, 0, 50))
    c = by_name(check_parts([a, b]), "floating")
    assert c.status == WARN and c.value == ["lid"]
    touching = box_part("post", size=(20, 20, 40), offset=(0, 0, 10))
    assert by_name(check_parts([a, touching]), "floating").status == PASS
    exact = {"gap_mm": 0.5, "pairs": []}  # the runner's exact contacts win over bounding boxes
    assert by_name(check_parts([a, touching], contacts=exact), "floating").status == WARN
    assert by_name(check_parts([a, b], {"floating_ok": True}), "floating").status == PASS


def test_planned_parts_grounding_budget_and_colour():
    a = box_part("seat", size=(50, 50, 10), offset=(0, 0, 40))
    a.color_given = False
    cs = check_parts([a], {"parts": [{"name": "seat"}, {"name": "leg"}]}, max_triangles=5)
    assert by_name(cs, "planned_parts").status == WARN and "leg" in by_name(cs, "planned_parts").message
    assert by_name(cs, "grounded").status == WARN
    assert by_name(cs, "triangles").status == FAIL
    assert by_name(cs, "color").status == WARN


def test_several_bodies_and_degenerate_triangles_warn():
    a, b = box_part(), box_part(offset=(50, 0, 0))
    a.vertices = np.vstack([a.vertices, b.vertices])
    a.triangles = np.vstack([a.triangles, b.triangles + len(b.vertices)])
    a.vertices = np.vstack([a.vertices, [[0, 0, 0], [1, 0, 0], [2, 0, 0]]])
    n = len(a.vertices)
    a.triangles = np.vstack([a.triangles, [[n - 3, n - 2, n - 1]]])
    cs = check_parts([a])
    assert by_name(cs, "components").status == WARN
    assert by_name(cs, "degenerate").status == WARN


def test_no_parts_and_export_report():
    assert check_parts([])[0].status == FAIL
    rep = {"formats": {"glb": {"ok": True}, "obj": {"ok": False, "problems": ["material library model.mtl is missing"]}}}
    cs = check_export(rep)
    assert [c.status for c in cs] == [PASS, FAIL] and "mtl" in cs[1].message
    assert check_export({})[0].status == FAIL
