"""FW-H (Farassat 1A) against exact solutions, Gutin, and the incompressible limit."""
import numpy as np
import pytest

from mancha.acoustics import (
    SourceSurface,
    acoustic_field,
    dipole_potential,
    fwh,
    gutin,
    harmonics,
    monopole_potential,
    rigid_rotation,
    sphere_panels,
)

C, RHO0 = 340.0, 1.2
OMEGA = 2 * np.pi * 200.0
TAU = np.linspace(0, 0.02, 600)
OBS = np.array([[2.0, 0.5, 1.0], [-0.6, 0.2, 0.1]])          # a far and a near observer


def _permeable(phi, rotation, n_theta=24, n_phi=48):
    pts, nrm, dS = sphere_panels(0.3, n_theta, n_phi)
    k = rigid_rotation(pts, nrm, rotation, TAU)
    rho, u, p = acoustic_field(phi, k["y"], TAU[:, None], RHO0, C)
    s = SourceSurface("permeable", TAU, k["y"], k["v"], dS, n=k["n"], p=p, rho=rho, u=u, vdot=k["vdot"],
                      ndot=k["ndot"])
    r = fwh(s, OBS, C, RHO0)
    exact = acoustic_field(phi, OBS[:, None, :], r["t"][None, :], RHO0, C)[2]
    return np.abs(r["total"] - exact).max() / np.abs(exact).max()


@pytest.mark.parametrize("rotation", [0.0, 300.0])     # 300 rad/s: panels at Mach 0.26
@pytest.mark.parametrize("source", ["monopole", "dipole"])
def test_permeable_surface_reproduces_exact_fields(source, rotation):
    phi = (monopole_potential(1e-3, OMEGA, C) if source == "monopole"
           else dipole_potential(1e-3, OMEGA, C, axis=(0, 0.6, 0.8)))
    assert _permeable(phi, rotation) < 3e-3


def test_error_falls_with_panel_refinement():
    phi = dipole_potential(1e-3, OMEGA, C, axis=(0, 0.6, 0.8))
    coarse, fine = _permeable(phi, 300.0, 10, 20), _permeable(phi, 300.0, 20, 40)
    assert coarse / fine > 3.0                                  # ≈ second order (3.4 measured)


def test_rotating_compact_forces_agree_with_gutin():
    B, R, rpm = 2, 0.12, 6000.0
    om = rpm * 2 * np.pi / 60
    re, thrust, torque = 0.8 * R, 4.0, 0.08
    per = 720
    tau = np.arange(5 * per) * (2 * np.pi / om) / per
    ang = 2 * np.pi * np.arange(B) / B
    k = rigid_rotation(np.stack([0 * ang, re * np.cos(ang), re * np.sin(ang)], 1), None, om, tau)
    e_theta = k["v"] / np.linalg.norm(k["v"], axis=-1, keepdims=True)
    force = np.zeros_like(k["y"])
    force[..., 0] = -thrust / B                          # on the fluid: thrust pushes the prop towards +x
    force += torque / (B * re) * e_theta                 # blade drag drags the fluid round
    s = SourceSurface("forces", tau, k["y"], k["v"], np.ones(B), force=force, vdot=k["vdot"])
    for deg in (30, 90, 150):
        th = np.radians(deg)
        r = fwh(s, [[30 * np.cos(th), 30 * np.sin(th), 0.0]], C, RHO0)
        h = harmonics(r["t"], r["total"][0], B * om / (2 * np.pi), 3)
        g = gutin(thrust, torque, om, B, re, 30.0, th, C, 3)
        assert np.allclose(h[:2], g[:2], rtol=0.01)


def test_steady_force_is_the_incompressible_point_force_field():
    """A constant force on the fluid, translating slowly with the observer: p → F·x/(4π|x|³) as M → 0."""
    tau = np.linspace(0, 0.05, 200)
    force = np.broadcast_to(np.array([0.3, -0.2, 0.5]), (len(tau), 1, 3))
    s = SourceSurface("forces", tau, np.zeros((len(tau), 1, 3)), np.zeros((len(tau), 1, 3)), np.ones(1),
                      force=force)
    x = np.array([0.4, 0.3, -0.2])
    exact = force[0, 0] @ x / (4 * np.pi * np.linalg.norm(x) ** 3)
    static = fwh(s, [x], C, RHO0)["total"][0]
    assert np.allclose(static, exact, rtol=1e-10)
    moving = fwh(s, [x], C, RHO0, translation=(-3.4, 0, 0))["total"][0]     # Mach 0.01
    assert np.ptp(moving) < 1e-9 * abs(exact) and abs(moving.mean() / exact - 1) < 0.02


def test_harmonics_of_a_known_signal():
    t = np.linspace(0, 1, 2001)
    p = 3 * np.cos(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t + 0.3)
    h = harmonics(t, p, 5.0, 3)
    assert np.allclose(h, [3 / np.sqrt(2), 0.5 / np.sqrt(2), 0.0], atol=1e-3)


def test_link_forces_and_faces():
    from mancha import Simulation
    from mancha.diagnostics import solid_faces

    sim = Simulation(shape=(16, 12, 12), viscosity=0.05, velocity=(0.03, 0.01, 0))
    mask = np.zeros(sim.shape, bool)
    mask[6:9, 4:7, 5:8] = True
    sim.set_solid(mask).run(50)
    pos, forces = sim.solid_links()
    assert np.allclose(forces.sum(0), sim.solid_force())
    assert np.all(np.abs(pos - np.round(pos)) <= 0.5)
    centres, normals = solid_faces(mask)
    assert len(centres) == 54 and np.allclose(normals.sum(0), 0.0)
