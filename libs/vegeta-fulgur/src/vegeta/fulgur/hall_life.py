"""A Hall thruster's heat, wear and test (``docs/hall.md``, H5): where the discharge power goes, the temperatures
a body radiating in vacuum settles at (the coils, the channel, the anode, magnets if any), the channel wall's
erosion by the ions and the life it sets, the permanent-magnet trade, and the vacuum test stand the thruster needs.

Every number here is a budget, not a measurement: the fractions are the published ranges for the class (Goebel &
Katz ch. 7; the SPT-100 and BHT-200 test reports), the sputter yield a fit to the BN data at 100–400 eV.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .hall import AMU, E_CHARGE, Channel, Coil, HallPoint, MU0

SIGMA_SB = 5.670_374e-8
K_B = 1.380_649e-23

#: Where the discharge power goes in a Hall thruster of this class (fractions of P_d): the beam carries
#: eta_v (1 - f_e) of it; the rest is split among the channel walls (ions and electrons hitting them), the anode
#: (the electron current's sheath and thermal energy), ionisation and radiation (lost as light and plume), and the
#: cathode coupling.
LOSS_SPLIT = {"walls": 0.45, "anode": 0.25, "radiation_plume": 0.22, "cathode_coupling": 0.08}


def power_budget(pt: HallPoint) -> dict:
    """Watts: the beam's power and the losses, by where they land."""
    p_beam = pt.i_beam * pt.eta_v * pt.v_d
    p_loss = pt.power - p_beam
    out = {"discharge": pt.power, "beam": p_beam}
    out.update({k: v * p_loss for k, v in LOSS_SPLIT.items()})
    return out


def radiative_temperature(power: float, area: float, emissivity: float = 0.8, t_env: float = 293.15) -> float:
    """The temperature [K] a surface of ``area`` settles at radiating ``power`` to a ``t_env`` chamber."""
    return (power / (emissivity * SIGMA_SB * area) + t_env ** 4) ** 0.25


@dataclass(frozen=True)
class ThermalBody:
    """The thruster as lumps that radiate: the body (iron and screen: most of the area), the channel's front
    (BN, hot), the anode (buried: radiates to the channel only), the coils (conducting to the body through their
    cores)."""

    ch: Channel
    body_area: float            # m^2 radiating (the screen's outside, the back plate's back)
    channel_front_area: float   # m^2 (the channel's rim and front poles)
    coil_power: float           # W
    coil_r_th: float = 4.0      # K/W from the coils to the body (a wound coil on an iron core, vacuum: no convection)
    anode_r_th: float = 10.0    # K/W from the anode to the body (stainless on the channel floor, through the BN)
    eps_body: float = 0.8       # black-anodised or oxidised iron
    eps_bn: float = 0.9


def temperatures(pt: HallPoint, tb: ThermalBody) -> dict:
    """Steady temperatures [°C] of the lumps for the budget's powers."""
    b = power_budget(pt)
    p_walls, p_anode = b["walls"], b["anode"]
    # the walls' and the anode's heat all leaves through the body's radiating faces and the channel front
    p_body = p_walls + p_anode + tb.coil_power
    t_body = radiative_temperature(p_body, tb.body_area + tb.channel_front_area, tb.eps_body)
    t_channel = radiative_temperature(p_walls, tb.channel_front_area, tb.eps_bn, t_env=t_body)   # the BN runs hotter: it radiates its own share from its front
    t_anode = t_body + p_anode * tb.anode_r_th
    t_coils = t_body + tb.coil_power * tb.coil_r_th
    k = lambda T: T - 273.15
    return {"body_C": k(t_body), "channel_C": k(t_channel), "anode_C": k(t_anode), "coils_C": k(t_coils),
            "coil_limit_C": 200.0, "SmCo_limit_C": 300.0, "NdFeB_limit_C": 150.0, "BN_limit_C": 900.0,
            "coils_ok": k(t_coils) < 200.0, "smco_ok": k(t_body) < 300.0, "ndfeb_ok": k(t_body) < 150.0}


def radiator_for(pt: HallPoint, tb: ThermalBody, t_coils_max_c: float = 190.0, t_env: float = 293.15) -> dict:
    """The radiating area the body needs for the coils to stay under ``t_coils_max_c`` (a radiator plate on the
    back, black, seeing the chamber): the body temperature that allows, and the area."""
    b = power_budget(pt)
    p_body = b["walls"] + b["anode"] + tb.coil_power
    t_body = t_coils_max_c + 273.15 - tb.coil_power * tb.coil_r_th
    area = p_body / (tb.eps_body * SIGMA_SB * (t_body ** 4 - t_env ** 4))
    return {"body_C": t_body - 273.15, "area_m2": area, "extra_m2": max(area - tb.body_area - tb.channel_front_area, 0.0),
            "disc_diameter_m": 2 * math.sqrt(max(area - tb.body_area - tb.channel_front_area, 0.0) / math.pi / 2)}   # a disc radiating both faces


# ------------------------------------------------------------------------------------------- erosion and life
N_BN = 1.0e29                              # atoms/m^3 of boron nitride (2.1 g/cm^3, 24.8 g/mol, two atoms)


def sputter_yield_bn(energy_ev: float) -> float:
    """Atoms per xenon ion on BN: a fit through the measured 0.02 at 100 eV, 0.07 at 300 eV (threshold ~ 50 eV)."""
    if energy_ev <= 50.0:
        return 0.0
    return 0.07 * ((energy_ev - 50.0) / 250.0) ** 1.1


def erosion(ch: Channel, n_exit: float = 2e17, te_ev: float = 20.0, wall_energy_ev: float | None = None,
            wall_thickness: float = 2e-3, zone_fraction: float = 0.3) -> dict:
    """The channel wall's erosion at the exit: the Bohm flux of ions to the wall over the last ``zone_fraction`` of
    the length, each hitting with the sheath's energy (5–6 T_e plus the acceleration they already have near the exit:
    ``wall_energy_ev``, default 6 T_e + 60 eV), the sputter yield, the rate [mm per 1000 h] and the hours to go
    through ``wall_thickness`` (the rate halves as the wall recedes and the ions graze: the usual 2× factor)."""
    m_i = 131.293 * AMU
    c_s = math.sqrt(E_CHARGE * te_ev / m_i)
    flux = 0.5 * n_exit * c_s                                   # Bohm flux [ions/(m^2 s)]
    e_wall = wall_energy_ev if wall_energy_ev is not None else 6.0 * te_ev + 60.0
    y = sputter_yield_bn(e_wall)
    rate = y * flux / N_BN                                      # m/s
    rate_mm_kh = rate * 3600e3 * 1e3
    life_h = 2.0 * wall_thickness / rate / 3600.0 if rate > 0 else float("inf")
    return {"ion_flux": flux, "wall_energy_eV": e_wall, "yield": y, "rate_mm_per_kh": rate_mm_kh, "life_h": life_h,
            "zone_length_mm": zone_fraction * ch.L * 1e3}


def total_impulse(pt: HallPoint, life_h: float) -> dict:
    """What the life buys: total impulse [N s] and the propellant it takes [kg]."""
    return {"total_impulse_Ns": pt.thrust * life_h * 3600, "propellant_kg": pt.mdot_total * life_h * 3600}


# ------------------------------------------------------------------------------------------- the magnet trade
def magnet_trade(ampere_turns_per_coil: float, gap_m: float, b_gap: float, coil_mass: float, coil_power: float,
                 n_coils: int = 2, material: str = "SmCo") -> dict:
    """Permanent magnets in place of the coils: the magnet length that supplies the gap's magnetomotive force at
    the material's working point (half its coercivity: B at half remanence), its mass, what is saved — and what
    is lost (no adjustment in test; the material's temperature limit)."""
    props = {"SmCo": dict(br=1.05, hc=7.5e5, rho=8400.0, t_max_c=300.0), "NdFeB": dict(br=1.25, hc=9.0e5, rho=7500.0, t_max_c=150.0)}[material]
    mmf = n_coils * ampere_turns_per_coil                       # A: the loop's magnetomotive force
    length = mmf / (0.5 * props["hc"])                          # m of magnet along the flux
    # the magnet's section carries the gap's flux at half remanence: area = B_gap A_gap / (Br/2); take the gap's
    # area as the channel exit annulus' (the field crosses it): ~ 2 pi r_mean * width
    area = (b_gap / (0.5 * props["br"]))
    mass = length * props["rho"] * area * 1e-3                  # per 1e-3 m^2 of gap area (reported per that)
    return {"material": material, "magnet_length_mm": length * 1e3, "mass_per_1e-3m2_gap_kg": mass,
            "saved_coil_mass_kg": coil_mass, "saved_power_W": coil_power, "t_max_C": props["t_max_c"],
            "note": "fixed field: no B scan in test; the iron circuit stays, lighter without the coil windows"}


# ------------------------------------------------------------------------------------------- the test stand
def facility(pt: HallPoint, p_max_mbar: float = 2e-5, t_gas: float = 293.15) -> dict:
    """What the vacuum chamber must do: the xenon throughput [mbar l/s] and the pumping speed [l/s] that holds the
    background under ``p_max_mbar`` (the class's rule: facility effects start above ~3e-5 mbar), the thrust
    balance's range and resolution, the supplies."""
    m_i = 131.293 * AMU
    q = pt.mdot_total / m_i * K_B * t_gas                       # Pa m^3/s
    q_mbar_l = q * 10.0                                         # 1 Pa m^3/s = 10 mbar l/s
    speed = q_mbar_l / p_max_mbar
    return {"throughput_mbar_l_s": q_mbar_l, "pumping_speed_l_s": speed, "chamber": "≥ 1 m diameter, 2 m long; cryopanels (xenon)",
            "thrust_balance": {"range_mN": 50.0, "resolution_mN": 0.2, "type": "inverted pendulum or torsion, in-situ calibrated by weights"},
            "supplies": {"discharge": f"0–400 V, 0–3 A ({pt.v_d:.0f} V, {pt.i_d:.2f} A at the point)", "coils": "0–2 A each, two channels",
                         "cathode": "keeper 0–30 V 0–2 A, heater 0–8 A", "flow": "two controllers 0–3 mg/s Xe (anode, cathode)"},
            "diagnostics": ["discharge current probe ≥ 1 MHz (the breathing mode)", "Faraday probe sweep of the plume (divergence)",
                            "retarding potential analyser (ion energy)", "thermocouples: body, channel rim, anode mount, coils", "camera"]}


TEST_PLAN = [
    ("T1 cathode alone", "ignite the hollow cathode, keeper current, 2 h", "stable keeper discharge"),
    ("T2 first light", "discharge at 200 V, 0.8 mg/s, coils at the design current", "ignition, I_d within 30 % of the 0-D"),
    ("T3 performance map", "V_d 200–350 V × flow 0.8–1.5 mg/s × coil current 0.6–1.2 A", "thrust, Isp, efficiency; the B scan the coils allow"),
    ("T4 thermal soak", "2 h at the design point", "temperatures at the lumps against the model; coils under 200 °C"),
    ("T5 oscillations", "current probe at 1 MHz at the design point", "the breathing frequency against the 1-D model"),
    ("T6 wear", "100 h at the design point", "the exit wall's profile against the erosion rate"),
]
