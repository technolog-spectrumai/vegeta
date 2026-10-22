"""Stage 4A — detachable components (docs/engineering.md is the oracle): a central body, a detachable arm, a motor and a
propeller with explicitly known masses, inertias, identities, attachments and connections; the assembly block refers
to the links and connections and repeats nothing; filtered by the component-related system tags; round trip against
the fixture; refusals with reasons; losses reported. No collision, no fragmentation."""
import json
from pathlib import Path

import numpy as np
import pytest

import yax
from yax import hub, schema
from yax import tags as tg

GOLD = json.loads((Path(__file__).parent / "fixtures" / "export2" / "mock_detachable.json").read_text())


def _assembly():
    L = GOLD["links"]

    def link(name, pos=(0, 0, 0), shape=None):
        return hub.Link(name, pos=pos, mass=L[name]["mass"], inertia=np.diag(L[name]["inertia_diag"]), shapes=[shape] if shape else [],
                        data={"material": "petg-cf"} if name in ("body", "arm") else {})
    body = link("body", shape=hub.Shape("hub", "box", (0.08, 0.08, 0.03)))
    arm = link("arm", pos=(0.09, 0.09, 0.0), shape=hub.Shape("tube", "capsule", (0.006, 0.11)))
    motor = link("motor", pos=(0.2, 0.09, 0.005), shape=hub.Shape("can", "cylinder", (0.014, 0.016)))
    prop = link("prop", pos=(0.2, 0.09, 0.021), shape=hub.Shape("disk", "cylinder", (0.0635, 0.001), role="visual"))
    joints = [hub.Joint("body_arm", "fixed", "body", "arm", pos=(0.09, 0.09, 0.0)), hub.Joint("arm_motor", "fixed", "arm", "motor", pos=(0.2, 0.09, 0.005)),
              hub.Joint("motor_prop", "continuous", "motor", "prop", pos=(0.2, 0.09, 0.021), axis=(0, 0, 1))]
    return hub.Machine("mockquadarm", links=[body, arm, motor, prop], joints=joints, producer={"tool": "test"},
                       data={"assembly": json.loads(json.dumps(GOLD["data:assembly"])), "connections": json.loads(json.dumps(GOLD["data:connections"])),
                             "materials": json.loads(json.dumps(GOLD["data:materials"]))})


def test_assembly_matches_the_oracle(tmp_path):
    m = _assembly()
    assert schema.problems(m) == []
    for ext in ("usda", "yx2"):
        yax.save(m, tmp_path / f"a.{ext}")
        back, _ = yax.read(tmp_path / f"a.{ext}")
        asm = back.data["assembly"]
        assert asm == GOLD["data:assembly"] and back.data["connections"] == GOLD["data:connections"]
        assert asm["components"]["arm"]["parent"] == "body" and asm["components"]["prop"]["parent"] == "motor" and asm["root"] == "body"
        assert asm["components"]["arm"]["attachment"]["pos"] == [0.09, 0.09, 0.0] and asm["components"]["prop"]["separation"]["calibrated"] is True
        assert asm["components"]["motor"]["separation"]["limits"] == {"tension_n": 400.0, "shear_n": 300.0, "basis": "datasheet"}
        assert back.link("prop").mass == 0.005 and back.link("arm").inertia[1, 1] == 2e-5 and back.link("arm").data["material"] == "petg-cf"
        assert {c["link"] for c in asm["components"].values()} == {lk.name for lk in back.links}    # identities: the links themselves
        assert schema.problems(back) == []
    t = tg.table(m)
    assert t["data:assembly/components/arm/separation/limits"] == ["00detach"] and t["data:assembly/components/body"] == ["00detach"]
    assert t["data:connections/body-arm-bolts/fasteners"] == ["00conn"] and t["link:prop/mass"] == ["00mass"]


def test_filtering_by_component_related_tags(tmp_path):
    m = _assembly()
    ukko = tg.Requirements(required=("00core", "00geom", "00mass", "00inert", "00detach", "00conn"), optional=("00mat", "00str"), consumer="ukko")
    assert tg.check(m, ukko, fmt="yx2").ok
    parts = tg.select(m, keep=("00core", "00geom", "00mass", "00inert", "00detach", "00conn"))
    assert set(parts.data) == {"assembly", "connections"} and "material" not in parts.link("body").data and parts.link("arm").mass == 0.03
    assert schema.problems(parts) == ["data:assembly/components/body: material 'petg-cf' is not in data['materials']",
                                      "data:assembly/components/arm: material 'petg-cf' is not in data['materials']"]   # dependents of excluded data, said
    v = tg.check(parts, ukko)
    assert v.ok and v.statuses["00mat"].status == tg.WITHHELD
    no = tg.select(m, drop=("00detach",))
    assert "assembly" not in no.data and no.data["connections"] and no.tags["withheld"] == ["00detach"]
    yax.save(no, tmp_path / "no.yx2")
    text = (tmp_path / "no.yx2").read_text()
    assert "separation" not in text and "mock pull test" not in text and "body-arm-bolts" in text
    v = tg.check(yax.read(tmp_path / "no.yx2")[0], ukko)
    assert not v.ok and v.statuses["00detach"].status == tg.WITHHELD


def test_refusals_and_losses(tmp_path):
    m = _assembly()
    c = m.data["assembly"]["components"]
    c["prop"]["parent"] = "wing"
    c["motor"]["connection"] = "glue"
    c["arm"]["link"] = "boom"
    c["body"]["detachable"] = True
    c["arm"]["attachment"]["frame"] = "link:nowhere"
    c["arm"]["separation"]["limits"]["tension"] = 5.0
    c["prop"]["separation"]["source"] = ""
    m.data["assembly"]["root"] = "hub"
    p = schema.problems(m)
    for expect in ("parent 'wing' is not a component", "connection 'glue' is not in data['connections']", "link 'boom' is not a link",
                   "a detachable component needs a parent", "attachment: frame must be", "separation/limits: tension: a number needs a unit suffix",
                   "a calibrated criterion needs its source", "root 'hub' is not a component"):
        assert any(expect in x for x in p), (expect, p)
    m = _assembly()
    m.data["assembly"]["components"]["body"]["parent"] = "prop"
    assert any("form a cycle of parents" in x for x in schema.problems(m))
    m = _assembly()
    del m.data["assembly"]["components"]["motor"]["separation"]              # unknown separation: absent, not assumed
    assert schema.problems(m) == []
    for ext in ("sdf", "xml", "urdf"):
        rep = yax.save(m, tmp_path / f"a.{ext}")
        assert any(d.startswith("data:assembly: the assembly (4 components, 3 detachable") for d in rep.dropped)
        text = (tmp_path / f"a.{ext}").read_text()
        assert "separation" not in text and "prop" in text                   # the links are written, the assembly is not
