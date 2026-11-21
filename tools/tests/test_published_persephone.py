"""Persephone, the 12-segment chimney Myropod (``labs/legged/designs/myropod.py`` defaults; Chiron model
``campaigns/benchmark/persephone/persephone_robot.py``): ``exports/persephone.yx2`` is published, serves Montezuma
(``tools/consumers/montezuma.json`` names it, with Cleopatra and Apheloria), and carries its CAD look, notebook 17's
mass budget (10.96 kg) and the CAD's size. Only Yax and the published files are needed."""
import json
from pathlib import Path

import numpy as np
import pytest

yax = pytest.importorskip("yax")
from yax import tags as tg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REQ = ROOT / "tools" / "consumers" / "montezuma.json"


@pytest.fixture(scope="module")
def persephone():
    m, _ = yax.read(ROOT / "exports" / "persephone.yx2")
    return m


def test_published_and_served_to_montezuma(persephone):
    wants = json.loads(REQ.read_text())["machines"]
    assert {"persephone", "myropod", "apheloria"} <= set(wants)
    e = json.loads((ROOT / "exports" / "MANIFEST.json").read_text())["machines"]["persephone"]
    assert e["file"] == "persephone.yx2" and e["lab"] == "legged" and e["consumers"]["montezuma"]
    assert tg.check(persephone, tg.Requirements.load(REQ), "yx2").ok


def test_mass_is_notebook_17s_budget(persephone):
    mass = sum(lk.mass or 0.0 for lk in persephone.links)
    assert mass == pytest.approx(10.96, abs=0.01)
    assert persephone.producer["machine"] == "persephone" and not persephone.producer["lumped"]


def test_articulated_twelve_segments(persephone):
    names = {lk.name for lk in persephone.links}
    assert {f"segment_{i}" for i in range(1, 13)} <= {n.replace(" ", "_") for n in names}
    assert len(persephone.joints) > 12 * 4 * 3                          # 4 three-joint legs per segment, body joints


def test_cad_look_and_size(persephone):
    """Head + 12 × (60 mm segment + 10 mm joint) + tether stub ≈ 1 m long; 44 mm segments with legs out ≈ 0.16 m
    wide; 34 mm body, antenna and standing legs ≈ 0.11 m high."""
    cad = [s for s in persephone.links[0].shapes if s.name == "cad"]
    assert len(cad) == 1 and cad[0].role == "visual" and len(cad[0].faces) > 1000
    L, W, H = np.ptp(np.asarray(cad[0].points), axis=0)
    assert 0.92 < L < 1.05 and 0.13 < W < 0.18 and 0.08 < H < 0.14
