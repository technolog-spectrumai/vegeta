"""The UFOs: fictional, unarmed, no AIS; each in the band the owner set (saucer 10-15 m, 60-80 m/s, hovers; cigar
15-25 m, 200-250 m/s, never below its minimum speed); the kinematic model moves them like a helicopter and like a jet
within their limits; the collision primitives hold the CAD; the flag carries the texture both ways round; the flak
figures are the pass geometry's."""
import math

import pytest

import ufo
import ufo_flag

DT = 0.05


def _fly(name, s, cmd, t_max, until=None):
    t = 0.0
    while t < t_max:
        s = ufo.step(name, s, cmd, DT)
        t += DT
        if until and until(s):
            break
    return s, t


def test_in_the_owners_bands():
    s, c = ufo.sizing("saucer"), ufo.sizing("cigar")
    assert 10 <= s["size"][0] <= 15 and 60 <= s["kinematics"]["v_max_m_s"] <= 80 and s["kinematics"]["hover"]
    assert 15 <= c["size"][0] <= 25 and 8 <= c["size"][1] <= 12 and 200 <= c["kinematics"]["v_cruise_m_s"]["low"] and c["kinematics"]["v_max_m_s"]["high"] <= 250
    assert not c["kinematics"]["hover"] and c["kinematics"]["v_min_m_s"] > 0
    for x in (s, c):
        assert x["fictional"] and x["signature"]["ais"] is None and x["basis"] and x["mass_kg"] > 0


def test_turns_follow_bank_and_speed():
    for name in ufo.CRAFT:
        for r in ufo.kinematics(name)["turn_by_speed"]:
            if r["v_m_s"] > 30:                                  # banked: rate = g tan(bank) / v
                assert r["rate_deg_s"] == pytest.approx(math.degrees(ufo.G * math.tan(math.radians(r["bank_deg"])) / r["v_m_s"]), rel=0.01)
    assert ufo.turn("saucer", 0.0)["rate_deg_s"] == 40.0
    assert ufo.turn("cigar", 200.0)["radius_m"] == pytest.approx(200.0 ** 2 / (ufo.G * math.sqrt(2.5 ** 2 - 1)), rel=0.01)
    assert ufo.turn("cigar", 10.0)["v_m_s"] == 70.0           # never slower than its minimum speed


def test_saucer_moves_like_a_helicopter():
    s = ufo.new_state("saucer")
    h, t = _fly("saucer", s, {"heading_deg": 180}, 20, lambda q: abs(abs(math.degrees(q["psi"])) - 180) < 0.5)
    assert t == pytest.approx(4.5, abs=0.2) and math.hypot(h["x"], h["y"]) < 1e-6        # turns on the spot
    f, t = _fly("saucer", s, {"u": 500}, 60, lambda q: q["u"] >= 77.9)
    assert f["u"] <= 78.0 and t == pytest.approx(26, abs=1) and f["pitch"] < 0           # nose down to accelerate
    b, _ = _fly("saucer", s, {"u": -50, "v": 50, "w": 50}, 20)
    assert b["u"] == -15.0 and b["v"] == 18.0 and b["w"] == 10.0                          # backward, sideways, climbing
    q, _ = _fly("saucer", ufo.new_state("saucer", speed=68), {"u": 68, "heading_deg": -90}, 5)
    assert q["roll"] == pytest.approx(math.radians(45), abs=1e-3) and q["r"] < 0           # right turn: right wing down


def test_cigar_moves_like_a_jet():
    c = ufo.new_state("cigar", z=100.0)
    slow, _ = _fly("cigar", c, {"speed": 0.0}, 120)
    fast, _ = _fly("cigar", c, {"speed": 999.0}, 120)
    assert slow["u"] == 70.0 and fast["u"] == 215.0
    hi, _ = _fly("cigar", ufo.new_state("cigar", z=9000.0, speed=230), {"speed": 999.0}, 60)
    assert hi["u"] == 245.0
    turned, t = _fly("cigar", c, {"speed": 200, "heading_deg": 180, "alt_m": 100}, 120, lambda q: abs(abs(math.degrees(q["psi"])) - 180) < 0.5)
    assert 28.0 <= t <= 36.0                                    # 28 s at 2.5 g, plus rolling in and out
    left, _ = _fly("cigar", c, {"speed": 200, "heading_deg": 90}, 5)
    assert left["roll"] < 0 and left["r"] > 0 and abs(left["roll"]) <= math.radians(66.5)
    up, _ = _fly("cigar", c, {"speed": 200, "alt_m": 2000}, 30)
    assert 0 < up["w"] <= 25.0 and up["pitch"] <= math.radians(25)


def test_flak_figures():
    gun = ufo.FLAK["aa_35"]
    r = ufo.pass_through(200.0, 100.0, 0.0, gun)
    assert r["in_range_s"] == pytest.approx(2 * math.sqrt(4000 ** 2 - 100 ** 2) / 200, abs=0.1)
    assert r["lead_m"] == pytest.approx(200 * 100 / gun["mean_m_s"], abs=0.1)
    assert ufo.pass_through(230.0, 9000.0, 0.0, gun)["in_range_s"] == 0.0
    f = ufo.flak("saucer")
    assert "hover" in f and f["passes"][0]["by_gun"]["hmg_12_7"]["0m"]["in_range_s"] > ufo.flak("cigar")["passes"][0]["by_gun"]["hmg_12_7"]["0m"]["in_range_s"]


def test_signatures():
    c = ufo.signatures("cigar")
    assert c["ir"]["skin_dT_K"] == pytest.approx(0.89 * 215 ** 2 / 2010, abs=0.1)
    for name in ufo.CRAFT:
        r = ufo.CRAFT[name]["rcs_m2"]
        assert r["front"] < r["side"] < r["top"]


def test_flag_mesh_and_texture(tmp_path):
    for name in ufo.CRAFT:
        m = ufo.flag_mesh(name)
        assert len(m["uv"]) == 2 * (len(m["vertices"]) // 3) and m["texture"] == "alien_jolly_roger.png"
        V = [m["vertices"][i:i + 3] for i in range(0, len(m["vertices"]), 3)]
        UV = [m["uv"][i:i + 2] for i in range(0, len(m["uv"]), 2)]
        for face in (0, 1):                                     # u runs aft (hoist forward), v runs up, on both faces
            q = {tuple(UV[i]): V[i] for i in range(4 * face, 4 * face + 4)}
            assert q[(1, 0)][0] < q[(0, 0)][0] and q[(0, 1)][2] > q[(0, 0)][2]
        w, h = ufo.CRAFT[name]["flag_m"]
        assert w / h == pytest.approx(ufo_flag.W / ufo_flag.H)
    from PIL import Image
    im = Image.open(ufo_flag.render(str(tmp_path / "f.png")))
    assert im.size == (1536, 1024)
    assert im.getpixel((10, 10)) == (0, 0, 0) and im.getpixel((768, 200)) == (255, 255, 255)   # black field, white head
    assert sum(im.getpixel((ufo_flag.HEAD_CX - 120, 402))) < 60                                  # a black eye


@pytest.mark.skipif(ufo.cq is None, reason="needs CadQuery")
@pytest.mark.parametrize("name", list(ufo.CRAFT))
def test_collision_holds_the_cad(name):
    import numpy as np
    shape = ufo.build(name).val()
    assert shape.isValid()
    verts, _ = shape.tessellate(20.0, 0.3)

    def inside(p, g, tol=0.05):
        x = np.array(p) - np.array(g["pos"])
        s = g["size"]
        if g["type"] == "ellipsoid":
            return sum((x[i] / (s[i] + tol)) ** 2 for i in range(3)) <= 1
        if g["type"] == "box":
            w, _, _, qz = g.get("quat", (1.0, 0.0, 0.0, 0.0))   # a box turned about z only
            yaw = 2 * math.atan2(qz, w)
            x = np.array([x[0] * math.cos(yaw) + x[1] * math.sin(yaw), -x[0] * math.sin(yaw) + x[1] * math.cos(yaw), x[2]])
            return all(abs(x[i]) <= s[i] + tol for i in range(3))
        a = np.array(g["axis"], float)
        h = x @ a
        if g["type"] == "cylinder":
            return abs(h) <= s[1] + tol and np.linalg.norm(x - h * a) <= s[0] + tol
        h = max(-s[1], min(s[1], h))
        return np.linalg.norm(x - h * a) <= s[0] + tol
    col = ufo.collision(name)
    pts = [(q.x / 1000, q.y / 1000, q.z / 1000) for q in verts]
    assert all(any(inside(p, g) for g in col) for p in pts)
    assert max(math.dist((0, 0, ufo.sizing(name)["size"][2] / 2 if name == "saucer" else 0), p) for p in pts) <= ufo.bounding_radius(name) + 0.05


def test_cigar_is_winged():
    c = ufo.CRAFT["cigar"]
    port = ufo.wing_planform(c, 1)
    assert port[1][1] == c["wing_span_m"] / 2 and port[1][0] < port[0][0]                  # swept back
    assert {g["name"] for g in ufo.collision("cigar")} >= {"wing_port", "wing_starboard"}
    assert ufo.bounding_radius("cigar") >= math.hypot(port[2][0], port[2][1])
