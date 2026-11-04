"""Hall thrusters (``docs/hall.md``): the 0-D performance chain, the SPT-100 scaled to a power, the magnetised
checks, the magnetic circuit by reluctance, and the circuit's field solved on FiPy (axisymmetric magnetostatics on a
structured non-uniform grid: every region of the circuit is an r-z rectangle).

Units SI; the magnetic field in tesla (1 G = 1e-4 T).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

E_CHARGE = 1.602_176_634e-19
M_E = 9.109_383_7e-31
MU0 = 4e-7 * math.pi
G0 = 9.806_65
AMU = 1.660_539_07e-27
PROPELLANTS = {"xenon": 131.293 * AMU, "krypton": 83.798 * AMU, "argon": 39.948 * AMU}
IONISATION_EV = {"xenon": 12.13, "krypton": 14.0, "argon": 15.76}


# ------------------------------------------------------------------------------------------- the 0-D chain
@dataclass(frozen=True)
class HallPoint:
    """An operating point: ``v_d`` discharge voltage [V], ``mdot_a`` anode flow [kg/s], ``cathode_fraction`` of
    it through the cathode; the efficiency chain's factors (Goebel & Katz ch. 7): ``eta_m`` mass utilisation,
    ``eta_v`` voltage utilisation (the beam's energy over e V_d), ``gamma`` the beam's cos-divergence × multiple-ion
    correction, ``f_e`` the electron (non-beam) fraction of the discharge current."""

    v_d: float
    mdot_a: float
    propellant: str = "xenon"
    cathode_fraction: float = 0.08
    eta_m: float = 0.90
    eta_v: float = 0.90
    gamma: float = 0.95
    f_e: float = 0.25

    @property
    def m_ion(self) -> float:
        return PROPELLANTS[self.propellant]

    @property
    def v_ion(self) -> float:
        return math.sqrt(2 * E_CHARGE * self.eta_v * self.v_d / self.m_ion)

    @property
    def i_beam(self) -> float:
        return self.eta_m * self.mdot_a * E_CHARGE / self.m_ion

    @property
    def i_d(self) -> float:
        return self.i_beam / (1.0 - self.f_e)

    @property
    def power(self) -> float:
        return self.i_d * self.v_d

    @property
    def thrust(self) -> float:
        return self.gamma * self.eta_m * self.mdot_a * self.v_ion

    @property
    def mdot_total(self) -> float:
        return self.mdot_a * (1.0 + self.cathode_fraction)

    @property
    def isp(self) -> float:
        return self.thrust / (self.mdot_total * G0)

    @property
    def efficiency(self) -> float:
        """Total: thrust power over discharge power, with the cathode's flow charged."""
        return self.thrust ** 2 / (2 * self.mdot_total * self.power)

    @property
    def thrust_per_watt(self) -> float:
        return self.thrust / self.power

    def summary(self) -> dict:
        return {"propellant": self.propellant, "V_d": self.v_d, "mdot_a_mg_s": self.mdot_a * 1e6, "I_b": self.i_beam, "I_d": self.i_d,
                "power_W": self.power, "thrust_mN": self.thrust * 1e3, "v_ion_km_s": self.v_ion / 1e3, "isp_s": self.isp,
                "efficiency": self.efficiency, "mN_per_kW": self.thrust_per_watt * 1e6}


def point_for_power(power: float, v_d: float, propellant: str = "xenon", **factors) -> HallPoint:
    """The anode flow that draws ``power`` at ``v_d`` through the chain."""
    p0 = HallPoint(v_d, 1e-6, propellant, **factors)
    return HallPoint(v_d, 1e-6 * power / p0.power, propellant, **factors)


# ------------------------------------------------------------------------------------------- scaling
@dataclass(frozen=True)
class Channel:
    """The annular channel: mean diameter ``d`` [m], width ``h`` [m], length ``L`` [m]; exit field ``b_exit`` [T]."""

    d: float
    h: float
    L: float
    b_exit: float

    @property
    def r_in(self) -> float:
        return self.d / 2 - self.h / 2

    @property
    def r_out(self) -> float:
        return self.d / 2 + self.h / 2

    @property
    def area(self) -> float:
        return math.pi * (self.r_out ** 2 - self.r_in ** 2)


#: The SPT-100 as the reference: 1.35 kW at 300 V; channel outer Ø100 mm, width 15 mm, length 25 mm; ~0.02 T at the exit.
SPT100 = dict(power=1350.0, v_d=300.0, channel=Channel(d=0.085, h=0.015, L=0.025, b_exit=0.020), mdot_a=5.0e-6,
              thrust=0.083, isp=1600.0)
BHT200 = dict(power=200.0, v_d=250.0, thrust=0.0128, isp=1390.0, mdot_a=0.94e-6)


def scale_channel(power: float, v_d: float = 300.0, ref: dict = SPT100) -> Channel:
    """The reference channel scaled to ``power`` at fixed voltage and current density (Dannenmayer & Mazouffre):
    the current ∝ the channel's area, so every length ∝ √(P/P_ref · V_ref/V); the exit field ∝ 1/length (the
    electron Larmor radius must shrink with the width)."""
    c = ref["channel"]
    k = math.sqrt(power / ref["power"] * ref["v_d"] / v_d)
    return Channel(d=c.d * k, h=c.h * k, L=c.L * k, b_exit=c.b_exit / k)


# ------------------------------------------------------------------------------------------- magnetised?
def magnetised(ch: Channel, v_d: float, propellant: str = "xenon", te_ev: float = 20.0, n_e: float = 2e17,
               nu_ratio: float = 1.0) -> dict:
    """The conditions a Hall thruster lives on, with pass/fail: the electrons' Larmor radius ≪ the width, the
    ions' ≫ the length, the Hall parameter ≫ 1 (electron-neutral collisions at the channel's density plus the
    anomalous part, ``nu_ratio`` × classical). ``n_e`` the plasma density [1/m^3] near the exit."""
    B = ch.b_exit
    v_te = math.sqrt(E_CHARGE * te_ev / M_E)
    r_le = M_E * v_te / (E_CHARGE * B)
    m_i = PROPELLANTS[propellant]
    v_i = math.sqrt(2 * E_CHARGE * v_d / m_i)
    r_li = m_i * v_i / (E_CHARGE * B)
    w_ce = E_CHARGE * B / M_E
    # electron-neutral collisions: the neutral density from the flow is not known here; a representative 1e19 /m^3 at
    # the exit with sigma_en ~ 2e-19 m^2 (xenon, 20 eV), plus the anomalous share
    n_n = 1e19
    nu_en = n_n * 2e-19 * v_te * (1.0 + nu_ratio)
    hall = w_ce / nu_en
    return {"r_larmor_e_mm": r_le * 1e3, "r_larmor_i_mm": r_li * 1e3, "width_mm": ch.h * 1e3, "length_mm": ch.L * 1e3,
            "hall_parameter": hall, "f_ce_MHz": w_ce / (2 * math.pi) / 1e6, "B_G": B * 1e4,
            "electrons_magnetised": r_le < ch.h / 5, "ions_unmagnetised": r_li > 5 * ch.L, "hall_ok": hall > 10}


# ------------------------------------------------------------------------------------------- the circuit
@dataclass(frozen=True)
class Coil:
    turns: int
    current: float
    wire_d: float = 0.5e-3          # m, enamelled copper
    mean_radius: float = 0.01       # m
    rho_cu: float = 1.72e-8

    @property
    def ampere_turns(self) -> float:
        return self.turns * self.current

    @property
    def length(self) -> float:
        return self.turns * 2 * math.pi * self.mean_radius

    @property
    def resistance(self) -> float:
        return self.rho_cu * self.length / (math.pi * (self.wire_d / 2) ** 2)

    @property
    def power(self) -> float:
        return self.current ** 2 * self.resistance

    @property
    def mass(self) -> float:
        return 8960.0 * self.length * math.pi * (self.wire_d / 2) ** 2

    @property
    def current_density(self) -> float:
        """A/m^2 in the copper (in vacuum, without convection, keep it under ~3 A/mm^2)."""
        return self.current / (math.pi * (self.wire_d / 2) ** 2)


def reluctance_circuit(ch: Channel, wall: float = 2e-3, pole_gap_extra: float = 2e-3, iron_path: float = 0.12,
                       mu_r_iron: float = 1000.0, leakage: float = 1.6) -> dict:
    """The ampere-turns the gap needs: the radial gap between the pole pieces is the channel's width plus its two
    walls plus the clearance; the iron's path adds its reluctance; ``leakage`` (flux that does not cross the
    channel) scales the lot. Returns the gap, the gap's reluctance share and the ampere-turns for ``b_exit``."""
    g = ch.h + 2 * wall + pole_gap_extra
    ni = ch.b_exit * (g + iron_path / mu_r_iron) / MU0 * leakage
    return {"gap_m": g, "ampere_turns": ni, "iron_share": (iron_path / mu_r_iron) / (g + iron_path / mu_r_iron)}


def design_coils(ampere_turns: float, *, inner_radius: float, outer_radius: float, current: float = 0.8, wire_d: float = 0.63e-3,
                 outer_count: int = 4) -> dict:
    """Split the ampere-turns between the inner coil (round the core) and the outer coils (the same total: the
    flux through the gap is the same loop), size each."""
    n_in = int(math.ceil(ampere_turns / current))
    inner = Coil(n_in, current, wire_d, inner_radius)
    n_out = int(math.ceil(ampere_turns / current / outer_count))
    outer = Coil(n_out, current, wire_d, 0.006)           # each outer coil on its own 6 mm-radius core
    return {"inner": inner, "outer_each": outer, "outer_count": outer_count,
            "power_W": inner.power + outer_count * outer.power, "mass_kg": inner.mass + outer_count * outer.mass,
            "current_density_A_mm2": inner.current_density / 1e6}


# ------------------------------------------------------------------------------------------- magnetostatics on FiPy
@dataclass(frozen=True)
class CircuitSection:
    """The axisymmetric section of the magnetic circuit in metres (z along the thruster axis, the exit plane at
    z = L): the iron back plate, the core, the inner coil, the channel walls, the outer coil (one annular coil
    standing for the four), the yoke, the two front pole pieces."""

    ch: Channel
    wall: float = 2e-3
    plate_t: float = 5e-3
    core_r: float = 8e-3
    coil_in_w: float = 6e-3
    coil_out_w: float = 8e-3
    yoke_t: float = 4e-3
    pole_t: float = 4e-3
    pole_clear: float = 1e-3

    def regions(self) -> list:
        """[(name, r1, r2, z1, z2, kind)] with kind iron | coil | channel."""
        c, L = self.ch, self.ch.L
        r_wi, r_wo = c.r_in - self.wall, c.r_out + self.wall             # the channel's walls' outer faces
        core_top = L - self.pole_t
        regs = [("plate", 0.0, r_wo + self.coil_out_w + self.yoke_t, -self.plate_t, 0.0, "iron"),
                ("core", 0.0, self.core_r, 0.0, L, "iron"),
                ("coil_in", self.core_r, min(self.core_r + self.coil_in_w, r_wi - self.pole_clear), 0.0, core_top, "coil"),
                ("pole_in", 0.0, r_wi - self.pole_clear, core_top, L, "iron"),
                ("channel", c.r_in, c.r_out, 0.0, L, "channel"),
                ("coil_out", r_wo + self.pole_clear, r_wo + self.coil_out_w, 0.0, core_top, "coil"),
                ("yoke", r_wo + self.coil_out_w, r_wo + self.coil_out_w + self.yoke_t, 0.0, L, "iron"),
                ("pole_out", r_wo + self.pole_clear, r_wo + self.coil_out_w + self.yoke_t, core_top, L, "iron")]
        return regs


def _breaks(values, lo, hi, fine, far, box_lo, box_hi):
    """Grid lines from ``lo`` to ``hi`` through every value in ``values``: ``fine`` spacing inside [box_lo, box_hi],
    growing geometrically to ``far`` outside."""
    pts = sorted(set([lo, hi, box_lo, box_hi] + [float(v) for v in values if lo < v < hi]))
    edges = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        inside = a >= box_lo - 1e-12 and b <= box_hi + 1e-12
        if inside:
            n = max(1, int(round((b - a) / fine)))
            edges += list(np.linspace(a, b, n + 1)[1:])
        else:
            # geometric from the box outward (or inward towards it)
            n = max(2, int(round(math.log(1 + (b - a) * (1.4 - 1) / fine) / math.log(1.4))))
            t = np.linspace(0, 1, n + 1)
            g = (1.4 ** (n * t) - 1) / (1.4 ** n - 1)
            seg = a + (b - a) * (g if b > box_hi else 1 - g[::-1])
            edges += list(seg[1:])
    return np.array(edges)


def solve_magnetostatics(sec: CircuitSection, ampere_turns: float, *, mu_r_iron: float = 1000.0, r_dom: float | None = None,
                         z_lo: float | None = None, z_hi: float | None = None, lc_fine: float = 0.5e-3, lc_far: float = 6e-3) -> dict:
    """The vector potential on FiPy: ``∇·((1/(μ r)) ∇u) = −J`` with ``u = r A_θ``, ``B_r = −(1/r) ∂u/∂z``,
    ``B_z = (1/r) ∂u/∂r``, on a non-uniform structured (r, z) grid whose lines fall on every region's edges (the
    regions are rectangles; the axis is r = 0 with u = 0). The same ``ampere_turns`` in the inner and the outer
    coil, the same sense (both lie inside the flux loop core → inner pole → across the channel → outer pole →
    yoke → plate → core, so their magnetomotive forces add). Returns the fields, the radial field along the channel's midline, its mean at the exit plane and at the
    anode."""
    import fipy as fp
    c = sec.ch
    regs = sec.regions()
    r_box = max(r2 for _, _, r2, _, _, _ in regs)
    z_box_lo, z_box_hi = min(z1 for _, _, _, z1, _, _ in regs), max(z2 for _, _, _, _, z2, _ in regs)
    r_dom = r_dom or 3.0 * r_box
    z_lo = z_lo if z_lo is not None else z_box_lo - 2.5 * c.L
    z_hi = z_hi if z_hi is not None else z_box_hi + 4.0 * c.L
    r_edges = _breaks([r for _, r1, r2, _, _, _ in regs for r in (r1, r2)], 0.0, r_dom, lc_fine, lc_far, 0.0, r_box)
    z_edges = _breaks([z for _, _, _, z1, z2, _ in regs for z in (z1, z2)], z_lo, z_hi, lc_fine, lc_far, z_box_lo, z_box_hi)
    mesh = fp.Grid2D(dx=np.diff(r_edges), dy=np.diff(z_edges), nx=len(r_edges) - 1, ny=len(z_edges) - 1) + ((r_edges[0],), (z_edges[0],))
    r_c, z_c = np.asarray(mesh.cellCenters[0]), np.asarray(mesh.cellCenters[1])
    r_f = np.asarray(mesh.faceCenters[0])
    r_min = 0.25 * lc_fine
    mu_r = np.ones(mesh.numberOfCells)
    J = np.zeros(mesh.numberOfCells)
    masks = {}
    for name, r1, r2, z1, z2, kind in regs:
        m = (r_c > r1) & (r_c < r2) & (z_c > z1) & (z_c < z2)
        masks[name] = m
        if kind == "iron":
            mu_r[m] = mu_r_iron
        elif kind == "coil":
            J[m] = ampere_turns / ((r2 - r1) * (z2 - z1))          # both coils sit inside the flux loop: the same sense
    mu_c = fp.CellVariable(mesh=mesh, value=mu_r)
    coeff = fp.FaceVariable(mesh=mesh, value=1.0 / (MU0 * np.asarray(mu_c.harmonicFaceValue) * np.maximum(r_f, r_min)))
    u = fp.CellVariable(mesh=mesh, name="rA", value=0.0)
    u.constrain(0.0, mesh.exteriorFaces)                               # the axis (u = rA = 0) and the far field
    eq = fp.DiffusionTerm(coeff=coeff, var=u) == -fp.CellVariable(mesh=mesh, value=J)
    eq.solve(var=u)
    g = np.asarray(u.grad.value)
    rr = np.maximum(r_c, r_min)
    B_r, B_z = -g[1] / rr, g[0] / rr
    # the midline: the one column of cells nearest r = d/2 (a band would mix radii and saw-tooth the profile)
    r_cols = np.unique(r_c)
    r_mid = r_cols[np.argmin(np.abs(r_cols - c.d / 2))]
    mid = masks["channel"] & (r_c == r_mid)
    order = np.argsort(z_c[mid])
    z_line, br_line = z_c[mid][order], B_r[mid][order]
    exit_band, anode_band = z_line > 0.85 * c.L, z_line < 0.15 * c.L
    return {"mesh": mesh, "u": u, "B_r": B_r, "B_z": B_z, "r": r_c, "z": z_c, "masks": masks,
            "r_edges": r_edges, "z_edges": z_edges, "shape": (len(z_edges) - 1, len(r_edges) - 1),
            "z_line": z_line, "B_r_line": br_line,
            "B_r_exit": float(np.mean(np.abs(br_line[exit_band]))) if exit_band.any() else float("nan"),
            "B_r_anode": float(np.mean(np.abs(br_line[anode_band]))) if anode_band.any() else float("nan"),
            "cells": mesh.numberOfCells, "ampere_turns": ampere_turns}


def tune_ampere_turns(sec: CircuitSection, target: float, ni0: float, **kw) -> dict:
    """Linear in the current (iron unsaturated): one solve, then the ampere-turns scaled to hit ``target`` [T]."""
    r = solve_magnetostatics(sec, ni0, **kw)
    k = target / r["B_r_exit"] if r["B_r_exit"] > 0 else 1.0
    r2 = solve_magnetostatics(sec, ni0 * k, **kw)
    r2["scale_from_first"] = k
    return r2
