"""The corona solved, not estimated: the electric field with the ions' space charge and the ions' drift through it,
on FiPy (``docs/plasma.md``, P3). Fulgur adds the physics around FiPy's solvers, not a solver:

* **Poisson** ``∇·(ε ∇φ) = −ρ_q`` — the emitter at ``V``, the grounds at 0, the frame a dielectric (``eps_r``),
  the open boundary at the far field's potential (0);
* **the ions** ``∇·(ρ_q μ E) = ∇·(D ∇ρ_q)``, steady, upwinded (FiPy's exponential scheme), ``E = −∇φ`` on the faces;
* **Kaptzov**: the ion density injected at the emitter faces, ``rho_inj``, is adjusted (secant iteration) until the
  field at those faces equals Peek's onset field — the ionisation layer is not resolved, its effect is;
* the outer loop sweeps Poisson and the ions to a fixed point; the **current** is the ion flux into the grounds,
  the **force** on the gas ``∫ ρ_q E dV`` (axial and radial), the thrust its axial part.

``CoronaProblem`` takes any FiPy mesh with face masks for the emitter and the grounds; ``coaxial`` builds the
wire-to-cylinder case (exact in 1-D: the benchmark), ``rings`` (in ``halka_fulgur``'s step) a Gmsh section of
Halka. Under ``vegeta.mock`` (``mock.install``) the solve returns the 1-D model's numbers at once.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .gas import EPS0, Air, DRY

D_ION = 5.0e-6          # m^2/s: ion diffusion (Einstein: mu kT/e ~ 5e-6 at 20 °C); it only smooths the fronts


@dataclass
class CoronaResult:
    V: float
    current: float                   # A (per the mesh's extent: a full revolution on cylindrical grids)
    force_z: float                   # N, the axial force on the gas (thrust)
    force_r: float                   # N, the radial force (converging: negative = inward)
    rho_inj: float                   # C/m^3 at the emitter faces
    e_emitter: float                 # V/m, the field at the emitter faces after Kaptzov
    iterations: int
    converged: bool
    phi: object = field(repr=False, default=None)
    rho: object = field(repr=False, default=None)
    mesh: object = field(repr=False, default=None)

    @property
    def power(self) -> float:
        return self.V * self.current

    @property
    def thrust_per_watt(self) -> float:
        return self.force_z / self.power if self.power else 0.0


class CoronaProblem:
    """A corona on a FiPy mesh. ``emitter`` / ``ground`` are face masks (FiPy FaceVariables or boolean arrays),
    ``e_onset`` the field Peek's law gives for the emitter's tip radius, ``eps_r`` a cell variable or 1."""

    def __init__(self, mesh, emitter, ground, e_onset: float, air: Air = DRY, eps_r=1.0, axisymmetric: bool = True,
                 polarity: str = "+"):
        import fipy as fp
        self.fp, self.mesh, self.air = fp, mesh, air
        self.emitter, self.ground, self.e_onset = emitter, ground, e_onset
        self.mu = air.mobility(polarity)
        self.axisymmetric = axisymmetric
        self.phi = fp.CellVariable(mesh=mesh, name="phi", value=0.0)
        self.rho = fp.CellVariable(mesh=mesh, name="rho", value=0.0)
        self.eps = eps_r * EPS0
        # FiPy's cylindrical grids measure volumes and areas per radian (weighted by r, no 2 pi): integrate per revolution
        self._scale = 2 * math.pi if axisymmetric else 1.0
        self._volumes = np.asarray(mesh.cellVolumes) * self._scale
        # the constraints, once (FiPy accumulates constrain() calls): the voltages as Variables so they can change
        self._V = fp.Variable(0.0)
        self._rho_inj = fp.Variable(0.0)
        self.phi.constrain(self._V, emitter)
        self.phi.constrain(0.0, ground)
        self._open_faces = mesh.exteriorFaces & ~emitter & ~ground
        self.rho.constrain(self._rho_inj, emitter)
        self.rho.faceGrad.constrain(0.0, ground)                     # outflow: the ions leave through the grounds
        self._poisson = fp.DiffusionTerm(coeff=self.eps, var=self.phi) == -self.rho

    # --- gradients: FiPy's cell-centred Gauss gradient on a cylindrical grid is taken on r-weighted volumes, so its
    # r-component is d(phi)/dr + phi/r and the field needs phi/r taken back off (checked against the exact coaxial
    # field). The face gradient's normal component is a direct difference between the two cells and is exact; it
    # is the only component the convection flux and the boundary currents use, so it stays as FiPy gives it.
    def _E_face(self):
        return -self.phi.faceGrad

    def _E_cell(self):
        g = self.phi.grad
        if not self.axisymmetric:
            return -g
        r = np.asarray(self.mesh.cellCenters[0])
        corr = np.asarray(self.phi.value) / r
        out = -np.asarray(g.value)
        out[0] += corr
        return out

    def set_open(self, open_boundary: bool):
        """The far boundary at 0 V (open, the default of ``solve``) or insulated (a closed cylinder)."""
        if open_boundary and not getattr(self, "_open_set", False):
            self.phi.constrain(0.0, self._open_faces)
            self._open_set = True

    # --- the two equations
    def _solve_poisson(self, V: float):
        self._V.setValue(V)
        self._poisson.solve(var=self.phi)

    def _solve_ions(self, rho_inj: float):
        """The steady ion transport for the present field: linear in rho, one solve."""
        fp = self.fp
        self._rho_inj.setValue(rho_inj)
        E = self._E_face()
        eq = fp.ExponentialConvectionTerm(coeff=self.mu * E, var=self.rho) == fp.DiffusionTerm(coeff=D_ION, var=self.rho)
        eq.solve(var=self.rho)
        self.rho.setValue(np.maximum(self.rho.value, 0.0))

    # --- the diagnostics
    def field_at(self, faces, signed: bool = True) -> float:
        """The mean normal field at ``faces`` [V/m]; ``signed``: positive when the field leaves the electrode into
        the gas (the face normals point out of the domain, into the electrode), negative when the space charge has
        reversed it — the sign the Kaptzov update must see."""
        g = self._E_face()
        n = self.mesh.faceNormals
        en = np.asarray((g * n).sum(axis=0))
        m = np.asarray(faces)
        return float(-en[m].mean()) if signed else float(np.abs(en[m]).mean())

    def current(self) -> float:
        """Ion flux into the ground faces: sum over faces of rho mu E_n A."""
        g = self._E_face()
        n = self.mesh.faceNormals
        en = np.asarray((g * n).sum(axis=0))
        m = np.asarray(self.ground)
        rho_f = np.asarray(self.rho.arithmeticFaceValue)
        area = np.asarray(self.mesh._faceAreas) * self._scale
        return float(np.sum(np.abs(rho_f[m] * self.mu * en[m] * area[m])))

    def force(self) -> tuple[float, float]:
        E = self._E_cell()
        f = np.asarray(self.rho.value) * self._volumes
        ez = E[1] if self.mesh.dim == 2 else E[0]
        er = E[0] if self.mesh.dim == 2 else np.zeros_like(ez)
        return float(np.sum(f * ez)), float(np.sum(f * er))

    # --- the loop
    def solve(self, V: float, *, open_boundary: bool = True, outer: int = 200, tol: float = 0.01, rho0: float | None = None,
              relax: float = 0.5, warm: bool = True) -> CoronaResult:
        """Kaptzov iteration at voltage ``V``: adjust the injected density until the emitter's field is ``e_onset``;
        inside it, sweep Poisson and the ions with the charge under-relaxed (the relaxation tightens when the
        current oscillates). Below onset (the Laplace field at the emitter under ``e_onset``) the current is zero.
        ``warm``: start from the last converged charge (a voltage sweep then tracks from point to point)."""
        self.set_open(open_boundary)
        rho_warm = getattr(self, "_warm", None) if warm else None
        self.rho.setValue(0.0)
        self._solve_poisson(V)
        e_lap = self.field_at(self.emitter)
        if e_lap <= self.e_onset:
            return CoronaResult(V, 0.0, 0.0, 0.0, 0.0, e_lap, 0, True, self.phi, self.rho, self.mesh)
        # a first guess: the excess field screened by the charge over the domain's length scale, or the warm start
        ell = float(np.ptp(np.asarray(self.mesh.cellCenters), axis=1).max()) / 2
        r = rho0 if rho0 is not None else EPS0 * (e_lap - self.e_onset) / ell
        if rho_warm is not None:
            self.rho.setValue(rho_warm[0])
            r = rho_warm[1]
            self._solve_poisson(V)
        e, I_prev, dI_prev, settled, step = e_lap, None, None, 0, 1.5
        last_sign = 0
        for k in range(outer):
            rho_old = self.rho.value.copy()
            self._solve_ions(r)
            self.rho.setValue(relax * self.rho.value + (1 - relax) * rho_old)
            self._solve_poisson(V)
            e = self.field_at(self.emitter)
            I = self.current()
            field_ok = abs(e - self.e_onset) < tol * self.e_onset
            current_ok = I_prev is not None and I > 0 and abs(I - I_prev) < tol * I
            settled = settled + 1 if (field_ok and current_ok) else 0
            if settled >= 2:
                fz, fr = self.force()
                self._warm = (self.rho.value.copy(), r)
                return CoronaResult(V, I, fz, fr, r, e, k + 1, True, self.phi, self.rho, self.mesh)
            if I_prev is not None:
                dI = I - I_prev
                if dI_prev is not None and dI * dI_prev < 0:        # the current oscillates: relax harder
                    relax = max(relax * 0.7, 0.1)
                dI_prev = dI
            I_prev = I
            # Kaptzov: more charge lowers the emitter's field; move the injection by the field's ratio, and damp
            # the step whenever the error changes sign (an overshoot)
            sign = 1 if e > self.e_onset else -1
            if last_sign and sign != last_sign:
                step = max(step * 0.6, 0.25)
            last_sign = sign
            ratio = e / self.e_onset
            r *= 0.5 if ratio <= 0 else float(np.clip(ratio ** step, 0.5, 2.0))
        fz, fr = self.force()
        return CoronaResult(V, self.current(), fz, fr, r, e, outer, False, self.phi, self.rho, self.mesh)


# ------------------------------------------------------------------------------------------- the geometries
def coaxial(a: float, R: float, length: float = 1.0, nr: int = 400, stretch: float = 4.0):
    """The wire-to-cylinder corona (wire radius ``a``, cylinder ``R``, ``length`` along z): a 1-D problem on FiPy's
    cylindrical grid, cells crowded at the wire (``stretch``: the last cell over the first). Exact answers:
    the Laplace field ``V / (r ln(R/a))``, and the force identity ``∫ ρ E dV = I (R − a) / μ``."""
    import fipy as fp
    k = np.linspace(0, 1, nr + 1)
    r_edges = a + (R - a) * (np.expm1(k * math.log(stretch)) / (stretch - 1))
    dr = np.diff(r_edges)
    mesh = fp.CylindricalGrid2D(dr=dr, dz=(length,), nr=nr, nz=1, origin=((a,), (0.0,)))
    emitter, ground = mesh.facesLeft, mesh.facesRight
    return mesh, emitter, ground


def townsend_coaxial_current(V: float, V0: float, a: float, R: float, length: float, air: Air = DRY) -> float:
    """Townsend's coaxial corona current (1914): ``I = 8 π ε0 μ V (V − V0) L / (R^2 ln(R/a))``."""
    return 0.0 if V <= V0 else 8 * math.pi * EPS0 * air.mobility() * V * (V - V0) * length / (R * R * math.log(R / a))
