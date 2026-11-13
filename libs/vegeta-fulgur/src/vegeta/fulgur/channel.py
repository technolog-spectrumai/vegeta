"""A Hall thruster's channel in one dimension (``docs/hall.md``, H4): the quasi-neutral fluid model along the
axis, marched in time on FiPy's 1-D grid, after Boeuf & Garrigues (1998). Along z from the anode (0) through the
exit (L) to the cathode plane (the domain's end):

* neutrals: ``∂n_n/∂t + ∂(n_n v_n)/∂z = −S``, ``S = n n_n k_i(T_e)`` the ionisation, ``v_n`` fixed;
* ions (cold, unmagnetised): ``∂n/∂t + ∂(n v)/∂z = S − n ν_w`` and ``∂(nv)/∂t + ∂(n v²)/∂z = (e/m_i) n E``
  (``ν_w`` the loss to the walls inside the channel: the Bohm flux over the width);
* electrons across B: drift–diffusion ``Γ_e = −n μ_⊥ E − μ_⊥ ∂(n T_e)/∂z`` with ``μ_⊥ = (e/m_e ν)/(1 + (ω_ce/ν)²)``
  and ``ν = ν_en + α_B ω_ce`` (electron–neutral collisions plus the anomalous Bohm share), the total current
  ``J = e(n v) − e Γ_e`` the same everywhere, and ``∫E dz = V_d``: Ohm's law then gives ``J`` in closed form each
  step and ``E(z)`` with it;
* electron energy (``T_e`` in eV): ``∂(3/2 n T_e)/∂t + ∂(5/2 T_e Γ_e − κ ∂T_e/∂z)/∂z = −Γ_e E − S ε_iz − n ν_w ε_w``,
  the heating by the field, the ionisation cost, the walls.

Outputs: the discharge current in time (steady, or the breathing mode at ~10–30 kHz), the profiles of density,
velocity, temperature, field and potential, the ionisation zone, the thrust and the mass utilisation at the exit —
the physics behind the 0-D chain's factors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .hall import AMU, E_CHARGE, IONISATION_EV, M_E, PROPELLANTS, Channel

K_B_EV = 1.0                                   # temperatures in eV throughout
SIGMA_EN = 2.0e-19                             # m^2, electron-neutral (xenon, ~20 eV)


def k_ionisation(te_ev, propellant: str = "xenon"):
    """The Maxwellian ionisation rate [m^3/s]: an Arrhenius fit anchored at ~4e-13 at 20 eV for xenon."""
    e_iz = IONISATION_EV[propellant]
    te = np.maximum(te_ev, 0.5)
    k0 = {"xenon": 1.8e-13, "krypton": 1.2e-13, "argon": 0.9e-13}[propellant]
    return k0 * np.sqrt(te) * np.exp(-e_iz / te)


@dataclass
class ChannelModel:
    """``ch`` the channel, ``b_of_z(z)`` the radial field [T] along the midline (from the magnetostatic solve or a
    fit), ``v_d`` the discharge voltage, ``mdot`` the anode flow [kg/s]; ``plume`` the length past the exit to the
    cathode plane as a multiple of L; ``alpha_in`` / ``alpha_out`` the Bohm coefficients inside and past the exit."""

    ch: Channel
    b_of_z: object
    v_d: float
    mdot: float
    propellant: str = "xenon"
    plume: float = 2.0
    n_cells: int = 150
    alpha_in: float = 1.0 / 160.0
    alpha_out: float = 1.0 / 16.0
    v_n: float = 250.0
    nu_wall_factor: float = 1.0
    eps_wall_factor: float = 7.0              # the energy an electron-ion pair takes to the wall, in T_e
    n_floor: float = 1e15
    te_min: float = 0.8
    te_max: float = 60.0
    history: list = field(default_factory=list)

    def __post_init__(self):
        import fipy as fp
        self.fp = fp
        L = self.ch.L
        self.L_dom = L * (1.0 + self.plume)
        self.dz = self.L_dom / self.n_cells
        self.mesh = fp.Grid1D(dx=self.dz, nx=self.n_cells)
        self.z = np.asarray(self.mesh.cellCenters[0])
        self.zf = np.asarray(self.mesh.faceCenters[0])
        self.inside = self.z < L
        self.m_i = PROPELLANTS[self.propellant]
        self.A = self.ch.area
        self.B = np.abs(np.asarray([self.b_of_z(zz) for zz in self.z]))
        self.Bf = np.abs(np.asarray([self.b_of_z(zz) for zz in self.zf]))
        # the variables
        n_n0 = self.mdot / (self.m_i * self.A * self.v_n)
        self.nn = fp.CellVariable(mesh=self.mesh, name="n_n", value=n_n0, hasOld=True)
        self.n = fp.CellVariable(mesh=self.mesh, name="n", value=1e17, hasOld=True)
        self.F = fp.CellVariable(mesh=self.mesh, name="nv", value=1e17 * 2000.0, hasOld=True)
        self.te = fp.CellVariable(mesh=self.mesh, name="T_e", value=5.0, hasOld=True)
        self.nn.constrain(n_n0, self.mesh.facesLeft)
        self.n.faceGrad.constrain(0.0, self.mesh.facesRight)
        self.nn.faceGrad.constrain(0.0, self.mesh.facesRight)
        self.F.faceGrad.constrain(0.0, self.mesh.facesRight)
        self.te.constrain(2.0, self.mesh.facesLeft)                     # cool electrons at the anode
        self.te.faceGrad.constrain(0.0, self.mesh.facesRight)
        self.t = 0.0
        self.E = np.zeros(self.n_cells)
        self.J = 0.0
        # the coefficients as Variables set once and updated each step (building FiPy terms every step is slow)
        nf = self.zf.shape
        self._vF = fp.FaceVariable(mesh=self.mesh, rank=1, value=[np.zeros(nf)])
        self._vn = fp.FaceVariable(mesh=self.mesh, rank=1, value=[np.full(nf, self.v_n)])
        self._ge = fp.FaceVariable(mesh=self.mesh, rank=1, value=[np.zeros(nf)])
        self._kappa = fp.FaceVariable(mesh=self.mesh, value=np.zeros(nf))
        cv = lambda: fp.CellVariable(mesh=self.mesh, value=0.0)
        self._nk, self._S, self._nuw, self._force, self._n15, self._heat, self._loss = (cv() for _ in range(7))
        self._eq_nn = (fp.TransientTerm(var=self.nn) + fp.UpwindConvectionTerm(coeff=self._vn, var=self.nn)
                       == -fp.ImplicitSourceTerm(coeff=self._nk, var=self.nn))
        self._eq_n = (fp.TransientTerm(var=self.n) + fp.UpwindConvectionTerm(coeff=self._vF, var=self.n)
                      == self._S - fp.ImplicitSourceTerm(coeff=self._nuw, var=self.n))
        self._eq_F = (fp.TransientTerm(var=self.F) + fp.UpwindConvectionTerm(coeff=self._vF, var=self.F)
                      == self._force - fp.ImplicitSourceTerm(coeff=self._nuw, var=self.F))
        self._eq_te = (fp.TransientTerm(coeff=self._n15, var=self.te) + fp.UpwindConvectionTerm(coeff=self._ge, var=self.te)
                       == fp.DiffusionTerm(coeff=self._kappa, var=self.te) + self._heat - fp.ImplicitSourceTerm(coeff=self._loss, var=self.te))

    # --- the closures
    def mobility(self, n_n, te):
        """μ_⊥ on the cells [m^2/(V s)]."""
        v_te = np.sqrt(E_CHARGE * np.maximum(te, self.te_min) / M_E)
        nu_en = n_n * SIGMA_EN * v_te
        w_ce = E_CHARGE * self.B / M_E
        alpha = np.where(self.inside, self.alpha_in, self.alpha_out)
        nu = nu_en + alpha * w_ce + 1e6
        return (E_CHARGE / (M_E * nu)) / (1.0 + (w_ce / nu) ** 2)

    def ohm(self, n, F, te, mu):
        """J [A/m^2] from ∫E dz = V_d, then E(z): E = (J/e − F − μ ∂(n T_e)/∂z) / (n μ), bounded (a reversed field
        only inside the channel, as at the anode sheath; never past the exit, where the potential falls monotonically
        to the cathode's; never above 300 kV/m) — with the bounds, J comes from a bisection on ∫E dz = V_d."""
        p = n * te
        dp = np.gradient(p, self.z)
        denom = n * mu
        lo = np.where(self.inside, -2e4, 0.0)
        def field(J_e):
            return np.clip((J_e - F - mu * dp) / denom, lo, 3e5)
        a, b = 0.0, 1e25
        for _ in range(60):
            mid = 0.5 * (a + b)
            if np.sum(field(mid)) * self.dz < self.v_d:
                a = mid
            else:
                b = mid
        J_e = 0.5 * (a + b)
        return J_e * E_CHARGE, field(J_e)

    # --- a step
    def step(self, dt: float):
        n = np.maximum(np.asarray(self.n.value), self.n_floor)
        nn = np.maximum(np.asarray(self.nn.value), 1e15)
        F = np.asarray(self.F.value)
        te = np.clip(np.asarray(self.te.value), self.te_min, self.te_max)
        k_i = k_ionisation(te, self.propellant)
        S = n * nn * k_i
        mu = self.mobility(nn, te)
        J, E = self.ohm(n, F, te, mu)
        self.J, self.E = J, E
        v = F / n
        c_s = np.sqrt(E_CHARGE * te / self.m_i)
        nu_w = np.where(self.inside, self.nu_wall_factor * 2.0 * c_s / self.ch.h, 0.0)
        Gamma_e = F - J / E_CHARGE                                         # electron flux (negative: towards the anode)
        v_face = np.interp(self.zf, self.z, v)
        v_face[self.zf >= self.ch.L] = np.maximum(v_face[self.zf >= self.ch.L], 0.0)   # past the exit ions leave, never return
        self._vF.setValue([v_face])
        self._ge.setValue([2.5 * np.interp(self.zf, self.z, Gamma_e)])
        self._kappa.setValue(np.interp(self.zf, self.z, 2.5 * n * mu * te))
        self._nk.setValue(n * k_i); self._S.setValue(S); self._nuw.setValue(nu_w)
        self._force.setValue((E_CHARGE / self.m_i) * n * E); self._n15.setValue(1.5 * n)
        self._heat.setValue(-Gamma_e * E)
        loss = S * IONISATION_EV[self.propellant] + n * nu_w * self.eps_wall_factor * te
        self._loss.setValue(loss / np.maximum(te, self.te_min))
        for var, eq in ((self.nn, self._eq_nn), (self.n, self._eq_n), (self.F, self._eq_F), (self.te, self._eq_te)):
            var.updateOld()
            eq.solve(var=var, dt=dt)
        self.n.setValue(np.maximum(self.n.value, self.n_floor))
        self.nn.setValue(np.maximum(self.nn.value, 1e15))
        vv = np.clip(np.asarray(self.F.value) / np.asarray(self.n.value), -3e3, 4e4)   # ions: a sheath's worth backwards at most
        self.F.setValue(np.asarray(self.n.value) * vv)
        self.te.setValue(np.clip(self.te.value, self.te_min, self.te_max))
        self.t += dt
        n_now, F_now = np.asarray(self.n.value), np.asarray(self.F.value)
        i_exit = int(np.searchsorted(self.z, self.ch.L))
        thrust_now = self.m_i * F_now[-1] * (F_now[-1] / max(n_now[-1], self.n_floor)) * self.A
        self.history.append((self.t, J * self.A, thrust_now, F_now[i_exit] * self.A))

    def run(self, t_end: float = 100e-6, dt: float = 1e-8, every: int = 1) -> dict:
        steps = int(round(t_end / dt))
        for k in range(steps):
            self.step(dt)
        return self.summary()

    # --- what it says
    def summary(self, tail: float = 0.4) -> dict:
        hist = np.array(self.history)
        t, I = hist[:, 0], hist[:, 1]
        last = t > t[-1] * (1 - tail)
        I_mean = float(I[last].mean())
        I_amp = float(I[last].max() - I[last].min()) / 2
        thrust_mean = float(hist[last, 2].mean())                             # over the breathing cycles
        flux_mean = float(hist[last, 3].mean())
        # the breathing frequency from the dominant line of the tail
        f_breath = float("nan")
        if last.sum() > 16:
            sig = I[last] - I_mean
            spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
            freqs = np.fft.rfftfreq(len(sig), d=np.mean(np.diff(t[last])))
            if len(spec) > 2 and spec[1:].max() > 0:
                f_breath = float(freqs[1:][np.argmax(spec[1:])])
        n = np.asarray(self.n.value); F = np.asarray(self.F.value); te = np.asarray(self.te.value); nn = np.asarray(self.nn.value)
        v = F / np.maximum(n, self.n_floor)
        phi = self.v_d - np.cumsum(self.E) * self.dz
        i_exit = int(np.searchsorted(self.z, self.ch.L))
        ion_flux_exit = F[i_exit] * self.A                                  # ions/s through the exit plane
        mdot_ions = ion_flux_exit * self.m_i
        thrust = float(self.m_i * F[-1] * v[-1] * self.A)                   # momentum flux at the domain's end
        S = n * nn * k_ionisation(te, self.propellant)
        return {"I_d": I_mean, "I_amp": I_amp, "I_rel_amp": I_amp / I_mean if I_mean else float("nan"), "f_breathing_Hz": f_breath,
                "thrust": thrust_mean, "thrust_now": thrust, "mass_utilisation": flux_mean * self.m_i / self.mdot,
                "mass_utilisation_now": mdot_ions / self.mdot, "v_exit": float(v[-1]), "te_max": float(te.max()),
                "z_ionisation_peak": float(self.z[np.argmax(S)]), "z_te_peak": float(self.z[np.argmax(te)]),
                "profiles": {"z": self.z, "n": n, "n_n": nn, "v": v, "T_e": te, "E": self.E, "phi": phi, "S": S, "B": self.B},
                "history": hist, "t_end": float(self.t)}
