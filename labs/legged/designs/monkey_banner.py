"""The spider monkey on a march with a banner: a pole held across in both fists, at arm's length ahead of the chest,
a white cloth hanging from it with one word, "MONKEY", in red block letters on its face
(``space_kerchak_scenes --scene banner --machine spider``).

The arms are held in the carry while the legs walk (no swing: both fists keep the pole); the pole runs through both
fists' hollows (``space_units.fist_hole``) and the banner is a free body set rigidly on the right hand every frame.
Metres, x forward, y left, z up."""
from __future__ import annotations

import numpy as np
from vegeta.chiron import Geom, Link, Prop

import kerchak_scenarios as gs
import space_unit_specs as su
import space_units as sun

mm = 1e-3
P = su.SPIDER
CURL = 1.6
WORD = "MONKEY"
BANNER = {"over": 0.12, "drop": 0.30, "cloth_t": 0.004, "walk_m": 5.0}
#: the arms' carry [rad]: raised forward a little above level, the elbows a little bent, the fists' hollows on one line
CARRY = {"shoulder_pitch": -1.75, "shoulder_roll": 0.0, "shoulder_yaw": 0.0, "elbow": -0.35, "wrist_roll": 0.0, "wrist_pitch": 0.0}
WHITE, RED, WOOD = (0.97, 0.97, 0.95, 1.0), (0.80, 0.08, 0.10, 1.0), (0.62, 0.45, 0.25, 1.0)

#: a 5 x 7 block font (rows top to bottom) for the word
FONT = {"M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
        "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
        "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
        "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"]}


def hole():
    c, _, r = sun.fist_hole(dict(P), CURL)
    return np.array(c) * mm, r * mm


def carry(kin):
    """{joint: value}: both arms in the carry, both fists closed (``CARRY``)."""
    out = {}
    for side in ("L", "R"):
        for k, v in CARRY.items():
            j = f"{side}_{k}"
            if j in kin.index:
                out[j] = v
    return out


def banner_prop(span):
    """The banner in its own frame: the pole along y through the origin (the right fist's hollow) to the left fist's
    at ``span``, ``over`` past each; the cloth hanging from it on its front (+x), the word on the cloth's face."""
    B = BANNER
    _, rh = hole()
    W = span + 2 * B["over"]
    yc = span / 2
    xc = rh + B["cloth_t"] / 2 + 0.002                                     # the cloth hangs just in front of the pole
    g = [Geom("banner_pole", "capsule", (rh, W / 2), pos=(0.0, yc, 0.0), quat=(0.70710678, -0.70710678, 0.0, 0.0), mass=0.3,
              role="visual", rgba=WOOD),
         Geom("banner_cloth", "box", (B["cloth_t"] / 2, W / 2 - 0.01, B["drop"] / 2), pos=(xc, yc, -B["drop"] / 2), mass=0.2,
              role="visual", rgba=WHITE)]
    n = len(WORD)
    px = min((W - 0.06) / (n * 6 - 1), (B["drop"] - 0.06) / 7)            # the letters' pixel, the word filling the cloth
    y0 = yc - (n * 6 - 1) * px / 2                                          # read from the front (looking back along -x): the word runs towards +y
    z0 = -B["drop"] / 2 + 3.5 * px
    for i, ch in enumerate(WORD):
        for r, row in enumerate(FONT[ch]):
            for c, bit in enumerate(row):
                if bit == "1":
                    g.append(Geom(f"banner_{i}_{r}_{c}", "box", (0.0015, px / 2, px / 2),
                                  pos=(xc + B["cloth_t"] / 2 + 0.0015, y0 + (i * 6 + c + 0.5) * px, z0 - (r + 0.5) * px),
                                  mass=0.001, role="visual", rgba=RED))
    return Prop(Link("banner", pos=(-5.0, 0.0, 1.0), geoms=g), free=True)


def worn(kin, q):
    """{"banner": ("R_hand", R_off, p_off)} and the pole's span: the pole from the right fist's hollow to the left's,
    the cloth upright and forward."""
    kin.set(q)
    D = kin.data
    c_h, _ = hole()
    bR, bL = kin.body("R_hand"), kin.body("L_hand")
    Rr, pr = D.xmat[bR].reshape(3, 3).copy(), D.xpos[bR].copy()
    hr = pr + Rr @ c_h
    hl = D.xpos[bL] + D.xmat[bL].reshape(3, 3) @ c_h
    y = hl - hr
    span = float(np.linalg.norm(y))
    y /= span
    z = np.array([0.0, 0.0, 1.0]) - y[2] * y
    z /= np.linalg.norm(z)
    Rw = np.column_stack([np.cross(y, z), y, z])
    return {"banner": ("R_hand", Rr.T @ Rw, c_h.copy())}, span


def banner_props(p):
    ground = Prop(Link("street", pos=(0.0, 0.0, 0.0), geoms=[
        Geom("street", "box", (8.0, 3.0, 0.002), pos=(2.5, 0.0, -0.0015), mass=1.0, role="visual", rgba=(0.55, 0.56, 0.58, 1.0))]), free=False)
    return (ground, banner_prop(p.get("span", 0.3)))


def scenario_banner(kin, q_stand, p):
    """For now: the spider monkey walks with the banner held out ahead."""
    return [("gait", p["walk_m"] / 0.6, 0.0)]


gs.DEFAULTS["banner"] = dict(BANNER)
gs.SCENARIOS["banner"] = scenario_banner
gs.COMPOSITE["banner"] = banner_props
