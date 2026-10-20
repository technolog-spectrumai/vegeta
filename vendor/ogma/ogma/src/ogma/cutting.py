"""Chainsaw cutting of branches (the Perseus rovers, notebooks 37 and 38). SI: N, m, J, W, s.

A chainsaw removes the kerf (a slot ``kerf`` wide) across the branch; the energy it takes is the **specific cutting
energy** e_c [J/m³] of the wood times the kerf volume. e_c is the one number that sets cutting times and it is an
assumption to measure first: it comes from the cut-time derivation in ``WOODS`` (a 2 kW saw, 70 % of it at the
chain, cross-cutting a 200 mm pine in about 5 s removes π/4 · 0.2² · 0.0075 = 2.4e-4 m³ of kerf with 7 kJ: about
30 MJ/m³; hardwood about 1.5×; chain-saw research quotes 20–60 J/cm³ depending on species, moisture and chain
sharpness). From it:

* ``cut_time`` = e_c · A · kerf / (η P) — the time to cut a branch of cross-section A (the saw feeding at the rate
  the power allows; a dull chain or a timid feed is slower);
* ``chain_force`` = η P / v_chain — the tangential force the chain exerts on the wood (and the wood on the saw: with
  the chain running on the bar's underside it pulls the saw into the cut);
* ``feed_force`` = k_feed × chain force — the push into the wood, ``k_feed`` an assumption (chain-saw literature:
  the feed force is of the order of the cutting force; 1.5 here, conservative for the arm);
* ``head_loads`` — what the arm's head carries while cutting: the feed, the chain pull and the saw's weight;
* ``kerf_opens`` — whether a cut from above or below opens or pinches on a branch that overhangs the cut
  (cantilevered from the trunk: the top fibres are in tension, a top cut opens; a cut from below closes on the bar).
  The remedy is the forester's undercut (``undercut_depth``: a third of the diameter from below, then the top cut);
* ``branch_mass``, ``branch_moment`` — the green branch's mass and its bending moment at the cut;
* ``energy_per_cut``, ``cuts_per_charge`` — the battery side;
* ``max_branch_for_bar`` — the bar's reach in one pass (two passes from either side double it);
* ``kickback_zone`` — the bar nose's upper quadrant, which the controller keeps out of the wood.

Saws (``SAWS``) and woods (``WOODS``) are explicit catalogue entries with their sources, as ``actuators.py``.

    from ogma import cutting
    saw, wood = cutting.get_saw("chainsaw 2 kW, 36 V"), cutting.get_wood("softwood (spruce, pine), green")
    cutting.cut_time(0.10, saw, wood)          # s for a 100 mm branch
    cutting.cut_table([0.05, 0.10], [saw], [wood])
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["G", "Saw", "Wood", "SAWS", "WOODS", "get_saw", "get_wood", "saw_table", "wood_table", "cut_area",
           "cut_rate_area", "cut_time", "chain_force", "feed_force", "head_loads", "branch_mass", "branch_moment",
           "kerf_opens", "undercut_depth", "max_branch_for_bar", "energy_per_cut", "cuts_per_charge", "kickback_zone",
           "cut_table"]

G = 9.81


@dataclass(frozen=True)
class Saw:
    """A battery chainsaw as the rover's tool: ``power_W`` the rated electrical input, ``efficiency`` motor × chain
    drive (what reaches the cut), ``chain_speed`` [m/s], ``bar_length`` [m] (the usable bar beyond the body),
    ``kerf`` [m] (the chain's cutting width), ``mass_kg`` the powerhead without its battery plus bar, chain and the
    rover's mount (the rover's battery feeds it), ``voltage_V`` / ``current_A`` at rated power."""

    key: str
    power_W: float
    efficiency: float
    chain_speed: float
    bar_length: float
    kerf: float
    mass_kg: float
    voltage_V: float
    current_A: float
    source: str = ""

    @property
    def cutting_power(self) -> float:
        """Power at the chain [W]."""
        return self.power_W * self.efficiency

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Wood:
    """A wood class: ``density`` green [kg/m³], ``e_c`` specific cutting energy [J/m³], ``E`` [Pa], ``MOR`` the
    modulus of rupture (bending strength, green) [Pa], ``mu_steel`` friction of the bar on the wood."""

    key: str
    density: float
    e_c: float
    E: float
    MOR: float
    mu_steel: float
    source: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


SAWS = [
    Saw("chainsaw 2 kW, 36 V", 2000.0, 0.70, 24.0, 0.40, 0.0075, 6.5, 36.0, 55.0,
        "assumed: 36 V professional battery chainsaw class (2 kW motor, 3/8\" chain, 40 cm bar, 24 m/s); powerhead "
        "4.5 kg without the battery, bar and chain 1.0 kg, the rover's mount and guard 1.0 kg; motor 0.85 × chain "
        "drive 0.82 = 0.70 at the cut"),
    Saw("mini chainsaw 0.6 kW, 20 V", 600.0, 0.65, 8.0, 0.15, 0.005, 1.4, 20.0, 30.0,
        "assumed: 20 V one-hand pruning chainsaw class (6\" bar, 1/4\" chain, 8 m/s); 1.0 kg without the battery, "
        "0.4 kg of bar, chain and mount"),
]

WOODS = [
    Wood("softwood (spruce, pine), green", 850.0, 35e6, 10e9, 45e6, 0.40,
         "density: green spruce/pine 800-900 kg/m³ (Wood Handbook); e_c from the cut-time derivation in the module "
         "docstring (30 MJ/m³) rounded up for a chain in field condition; E and MOR green (Wood Handbook ch. 5)"),
    Wood("hardwood (oak, beech), green", 1000.0, 55e6, 12e9, 60e6, 0.45,
         "density: green oak/beech 950-1050 kg/m³; e_c 1.5× softwood (denser, harder; chain-saw tests: 40-60 J/cm³); "
         "E and MOR green (Wood Handbook ch. 5)"),
    Wood("garden shrub (hazel, privet, fruit), green", 900.0, 45e6, 8e9, 40e6, 0.45,
         "assumed: small-diameter garden wood between the two classes; knotty, with bark: e_c between softwood "
         "and hardwood, a lower E and MOR"),
    Wood("softwood, seasoned", 500.0, 45e6, 11e9, 70e6, 0.35,
         "dry (12 % moisture) spruce/pine: lighter and stiffer but harder to cut than green (chain-saw tests: dry "
         "wood takes ~30 % more energy); Wood Handbook ch. 5"),
]


def get_saw(key: str) -> Saw:
    for s in SAWS:
        if s.key == key:
            return s
    raise KeyError(f"no saw {key!r}; known: {[s.key for s in SAWS]}")


def get_wood(key_or_wood) -> Wood:
    if isinstance(key_or_wood, Wood):
        return key_or_wood
    for w in WOODS:
        if w.key == key_or_wood:
            return w
    raise KeyError(f"no wood {key_or_wood!r}; known: {[w.key for w in WOODS]}")


def saw_table() -> "pandas.DataFrame":
    import pandas as pd
    return pd.DataFrame([s.as_dict() for s in SAWS]).set_index("key")


def wood_table() -> "pandas.DataFrame":
    import pandas as pd
    return pd.DataFrame([w.as_dict() for w in WOODS]).set_index("key")


# ----------------------------------------------------------------------------------------------- the cut
def cut_area(d):
    """Cross-section [m²] of a round branch of diameter ``d`` [m]."""
    d = np.asarray(d, dtype=float)
    return math.pi / 4 * d * d


def cut_rate_area(saw: Saw, wood) -> float:
    """Kerf face area removed per second [m²/s] at the saw's cutting power: P_cut / (e_c · kerf)."""
    wood = get_wood(wood)
    return saw.cutting_power / (wood.e_c * saw.kerf)


def cut_time(d, saw: Saw, wood, feed: float = 1.0):
    """Time [s] to cut through a branch of diameter ``d`` [m]: e_c · A · kerf / (η P); ``feed`` < 1 is a feed held
    below the power's rate (a timid push, a dull chain)."""
    if not 0 < feed <= 1:
        raise ValueError("feed must be in (0, 1]")
    return cut_area(d) / (cut_rate_area(saw, wood) * feed)


def chain_force(saw: Saw) -> float:
    """Tangential cutting force [N] the chain exerts on the wood (= the wood on the saw, along the bar): P_cut / v."""
    return saw.cutting_power / saw.chain_speed


def feed_force(saw: Saw, wood=None, k_feed: float = 1.5) -> float:
    """Feed force [N] into the wood: ``k_feed`` × the chain force (assumption: of the order of the cutting force)."""
    return k_feed * chain_force(saw)


def head_loads(saw: Saw, wood=None, *, cut_from: str = "top", k_feed: float = 1.5, dynamic: float = 1.0) -> dict:
    """Forces [N] on the arm's head while cutting, in the head frame (x along the bar towards the nose, z up, the
    bar horizontal, the chain running towards the body on the bar's underside): ``feed`` the push into the wood
    (down for a cut from the top, up from below), ``chain_pull`` along −x (the chain drags the saw into the cut),
    ``weight`` the saw's, and the resultant; ``dynamic`` multiplies the cutting loads (vibration, a snag)."""
    if cut_from not in ("top", "bottom", "side"):
        raise ValueError("cut_from: 'top', 'bottom' or 'side'")
    Ft = chain_force(saw) * dynamic
    Fn = feed_force(saw, wood, k_feed) * dynamic
    W = saw.mass_kg * G
    if cut_from == "top":
        feed_vec = (0.0, 0.0, -Fn)
    elif cut_from == "bottom":
        feed_vec = (0.0, 0.0, Fn)
    else:
        feed_vec = (0.0, -Fn, 0.0)
    pull = (-Ft, 0.0, 0.0)
    weight = (0.0, 0.0, -W)
    total = tuple(a + b + c for a, b, c in zip(feed_vec, pull, weight))
    return {"feed_N": Fn, "chain_pull_N": Ft, "weight_N": W, "feed": feed_vec, "chain_pull": pull, "weight": weight,
            "total": total, "resultant_N": math.sqrt(sum(v * v for v in total))}


# ----------------------------------------------------------------------------------------------- the branch
def branch_mass(d, L, wood) -> float:
    """Mass [kg] of a round branch of diameter ``d`` and length ``L`` [m] (a cylinder of green wood)."""
    return float(get_wood(wood).density * cut_area(d) * L)


def branch_moment(d, L, wood) -> float:
    """Bending moment [N·m] at the cut from the overhanging part of the branch (``L`` beyond the cut) under its own
    weight: m g L / 2."""
    return branch_mass(d, L, wood) * G * L / 2


def kerf_opens(d: float, overhang: float, wood, cut_from: str = "top", *, extra_load_N: float = 0.0) -> dict:
    """Does the kerf open or close on the bar? A branch cantilevered from the trunk with ``overhang`` [m] beyond
    the cut bends under its weight (plus ``extra_load_N`` at its end — snow, a fork): the top fibres are in
    tension and the bottom in compression, so a cut from the top **opens** as it deepens and a cut from the bottom
    **pinches** (the classic trapped bar). A branch propped at its far end does the opposite. Returns the moment
    at the cut, the bending stress of the uncut section against the wood's MOR (the branch breaks before the cut is
    through when the moment is large: the hinge tears) and the verdict."""
    wood = get_wood(wood)
    M = branch_moment(d, overhang, wood) + extra_load_N * overhang
    S = math.pi * d ** 3 / 32                                   # section modulus of the uncut branch
    sigma = M / S if S > 0 else 0.0
    opens = {"top": True, "bottom": False, "side": None}[cut_from]
    if M <= 0:
        opens = None
    return {"moment_Nm": M, "stress_MPa": sigma / 1e6, "stress_over_MOR": sigma / wood.MOR, "opens": opens,
            "verdict": ("neutral (no overhang)" if opens is None else
                        "opens: the top fibres are in tension" if opens else
                        "pinches: the bottom fibres are in compression — undercut first, then cut from the top")}


def undercut_depth(d: float, fraction: float = 1 / 3) -> float:
    """The forester's undercut [m]: ``fraction`` of the diameter from below, before the top cut, so the branch
    breaks clean instead of tearing bark down the trunk or pinching the bar."""
    return fraction * d


def max_branch_for_bar(saw: Saw, passes: int = 1) -> float:
    """The largest branch [m] the bar cuts: its length in one pass, twice that from both sides."""
    if passes not in (1, 2):
        raise ValueError("passes: 1 or 2")
    return passes * saw.bar_length


# ----------------------------------------------------------------------------------------------- energy
def energy_per_cut(d, saw: Saw, wood, idle_s: float = 0.0, idle_W: float | None = None):
    """Electrical energy [Wh] of one cut: the kerf's e_c · A · kerf over the saw's efficiency, plus ``idle_s`` of
    the chain running free at ``idle_W`` (default 10 % of the rated power) before and after the cut."""
    wood = get_wood(wood)
    idle_W = 0.1 * saw.power_W if idle_W is None else idle_W
    E = wood.e_c * cut_area(d) * saw.kerf / saw.efficiency + idle_s * idle_W
    return E / 3600.0


def cuts_per_charge(battery_Wh: float, d, saw: Saw, wood, *, overhead_Wh: float = 0.0, reserve: float = 0.2,
                    idle_s: float = 2.0):
    """How many cuts of diameter ``d`` a ``battery_Wh`` battery gives (``reserve`` kept for driving home,
    ``overhead_Wh`` per cut for the arm's moves and the rover's electronics)."""
    per = energy_per_cut(d, saw, wood, idle_s=idle_s) + overhead_Wh
    return np.floor(battery_Wh * (1 - reserve) / per)


def kickback_zone(saw: Saw, bar_width: float) -> dict:
    """The bar nose: its radius (half the bar width) and the bar length the controller may put into the wood
    (the nose's upper quadrant kept clear: a cut on it throws the saw up and back)."""
    r = bar_width / 2
    return {"nose_radius_m": r, "usable_bar_m": saw.bar_length - r, "rule": "cut with the bar's underside, nose clear"}


def cut_table(diameters, saws, woods, feed: float = 1.0) -> "pandas.DataFrame":
    """Cut time [s], energy [Wh] and the branch's green mass per metre, per diameter × saw × wood."""
    rows = []
    for saw in saws:
        for wood in woods:
            w = get_wood(wood)
            for d in diameters:
                rows.append({"saw": saw.key, "wood": w.key, "d_mm": d * 1e3,
                             "cut_time_s": float(cut_time(d, saw, w, feed)), "energy_Wh": float(energy_per_cut(d, saw, w)),
                             "mass_per_m_kg": branch_mass(d, 1.0, w), "fits_bar": d <= saw.bar_length})
    import pandas as pd
    return pd.DataFrame(rows)
