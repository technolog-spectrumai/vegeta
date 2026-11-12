"""The corona solved, not estimated: the electric field with the ions' space charge and the ions' drift through it,
on FiPy (``docs/plasma.md``, P3). Fulgur adds the physics around FiPy's solvers, not a solver:

* **Poisson** ``∇·(ε ∇φ) = −ρ_q`` — the emitter at ``V``, the grounds at 0, the frame a dielectric (``eps_r``),
  the open boundary at the far field's potential (0);
* **the ions** ``∇·(ρ_q μ E) = ∇·(D ∇ρ_q)``, steady, upwinded (FiPy's upwind scheme: at these Péclet numbers the exponential one is the same and its matrix can go singular on unstructured meshes), ``E = −∇φ`` on the faces;
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

    def __init__(self, mesh, emitter, ground, e_onset: float, air: Air = DRY, eps_r=1.0, geometry: str = "cylindrical",
                 polarity: str = "+", inject=None):
        """``geometry``: ``cylindrical`` (FiPy's CylindricalGrid: r, z, volumes per radian), ``planar_axi`` (a planar
        mesh of the (r, z) section, e.g. from Gmsh: the equations carry the r weight), ``planar`` (a true 2-D slab per
        metre). ``inject``: the faces the Kaptzov injection applies to (the toothed edge); the whole ``emitter`` is
        at the voltage."""
        import fipy as fp
        self.fp, self.mesh, self.air = fp, mesh, air
        self.emitter, self.ground, self.e_onset = emitter, ground, e_onset
        self.inject = emitter if inject is None else inject
        self.mu = air.mobility(polarity)
        self.geometry = geometry
        self.axisymmetric = geometry == "cylindrical"
        self.phi = fp.CellVariable(mesh=mesh, name="phi", value=0.0)
        self.rho = fp.CellVariable(mesh=mesh, name="rho", value=0.0)
        self.eps = eps_r * EPS0
        if geometry == "cylindrical":
            # FiPy's cylindrical grids measure volumes and areas per radian (r-weighted, no 2 pi)
            self._wf, self._wc = 1.0, 1.0
            self._area_w = np.asarray(mesh._faceAreas) * 2 * math.pi
            self._volumes = np.asarray(mesh.cellVolumes) * 2 * math.pi
        elif geometry == "planar_axi":
            # a planar section: the r weight goes into the coefficients and the integrals (floored off the axis so
            # no face coefficient is exactly zero: the axis is a symmetry line, not a hole in the matrix)
            r_min = 1e-3 * float(np.abs(np.asarray(mesh.faceCenters[0])).max())
            self._wf = fp.FaceVariable(mesh=mesh, value=np.maximum(np.abs(np.asarray(mesh.faceCenters[0])), r_min))
            self._wc = fp.CellVariable(mesh=mesh, value=np.maximum(np.abs(np.asarray(mesh.cellCenters[0])), r_min))
            self._area_w = np.asarray(mesh._faceAreas) * 2 * math.pi * np.abs(np.asarray(mesh.faceCenters[0]))
            self._volumes = np.asarray(mesh.cellVolumes) * 2 * math.pi * np.abs(np.asarray(mesh.cellCenters[0]))
        else:
            self._wf, self._wc = 1.0, 1.0
            self._area_w = np.asarray(mesh._faceAreas)
            self._volumes = np.asarray(mesh.cellVolumes)
        # the constraints, once (FiPy accumulates constrain() calls): the voltages as Variables so they can change
        self._V = fp.Variable(0.0)
        self._rho_inj = fp.Variable(0.0)
        self.phi.constrain(self._V, emitter)
        self.phi.constrain(0.0, ground)
        axis = getattr(mesh, "physicalFaces", {}).get("axis", None)
        self._open_faces = mesh.exteriorFaces & ~emitter & ~ground
        if axis is not None:
            self._open_faces = self._open_faces & ~axis                  # the axis: symmetry (no flux), never 0 V
        self.rho.constrain(self._rho_inj, self.inject)
        # outflow everywhere but the injecting faces: the ions leave through the grounds and the open boundary
        # (a wall there would pile them up without bound wherever the field points out)
        self.rho.faceGrad.constrain(0.0, mesh.exteriorFaces & ~self.inject)
        self._poisson = fp.DiffusionTerm(coeff=self.eps * self._wf, var=self.phi) == -self.rho * self._wc

    # --- gradients: FiPy's cell-centred Gauss gradient on a cylindrical grid is taken on r-weighted volumes, so its
    # r-component is d(phi)/dr + phi/r and the field needs phi/r taken back off (checked against the exact coaxial
    # field). The face gradient's normal component is a direct difference between the two cells and is exact; it
    # is the only component the convection flux and the boundary currents use, so it stays as FiPy gives it.
    def _E_face(self):
        return -self.phi.faceGrad

    def _E_cell(self):
        g = self.phi.grad
        if not self.axisymmetric:
            return -np.asarray(g.value)
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
        eq = fp.UpwindConvectionTerm(coeff=self.mu * E * self._wf, var=self.rho) == fp.DiffusionTerm(coeff=D_ION * self._wf, var=self.rho)
        eq.solve(var=self.rho)
        self.rho.setValue(np.maximum(self.rho.value, 0.0))

    # --- the diagnostics
    def field_at(self, faces, signed: bool = True) -> float:
        """The mean normal field at ``faces`` [V/m]; ``signed``: positive when the field leaves the electrode into
        the gas (the face normals point out of the domain, into the electrode), negative when the space charge has
        reversed it — the sign the Kaptzov update must see."""
        g = self._E_face()
        n = self.mesh._orientedFaceNormals          # out of the owning cell: out of the domain on exterior faces
        en = np.asarray((g * n).sum(axis=0))
        m = np.asarray(faces)
        return float(-en[m].mean()) if signed else float(np.abs(en[m]).mean())

    def current(self) -> float:
        """Ion flux into the ground faces: sum over faces of rho mu E_n A."""
        g = self._E_face()
        n = self.mesh._orientedFaceNormals          # out of the owning cell: out of the domain on exterior faces
        en = np.asarray((g * n).sum(axis=0))
        m = np.asarray(self.ground)
        rho_f = np.asarray(self.rho.arithmeticFaceValue)
        return float(np.sum(np.abs(rho_f[m] * self.mu * en[m] * self._area_w[m])))

    def force(self) -> tuple[float, float]:
        E = np.asarray(self._E_cell())
        f = np.asarray(self.rho.value) * np.asarray(self._volumes)
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
        e_lap = self.field_at(self.inject)
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
            rho_new = self.rho.value.copy()
            self.rho.setValue(relax * rho_new + (1 - relax) * rho_old)
            self._solve_poisson(V)
            e = self.field_at(self.inject)
            I = self.current()
            field_ok = abs(e - self.e_onset) < tol * self.e_onset
            current_ok = I_prev is not None and I > 0 and abs(I - I_prev) < tol * I
            charge_ok = np.abs(rho_new - rho_old).max() < tol * max(rho_new.max(), 1e-30)   # the ions have settled too
            settled = settled + 1 if (field_ok and current_ok and charge_ok) else 0
            if settled >= 2:
                fz, fr = self.force()
                self._warm = (self.rho.value.copy(), r)
                return CoronaResult(V, I, fz, fr, r, e, k + 1, True, self.phi, self.rho, self.mesh)
            if I_prev is not None:
                dI = I - I_prev
                if dI_prev is not None and dI * dI_prev < 0:        # the current oscillates: relax harder
                    relax = max(relax * 0.7, 0.25)
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


def coaxial_section(a: float, R: float, length: float = 0.02, lc_wire: float = 20e-6, lc_wall: float = 1e-3):
    """The same coaxial corona as a Gmsh planar section (r from the wire to the cylinder, a short z extent with
    insulated ends) for ``geometry="planar_axi"``: the check that the planar form reproduces the exact solution."""
    import fipy as fp
    geo = f"""
Point(1) = {{{a}, 0, 0, {lc_wire}}}; Point(2) = {{{R}, 0, 0, {lc_wall}}}; Point(3) = {{{R}, {length}, 0, {lc_wall}}}; Point(4) = {{{a}, {length}, 0, {lc_wire}}};
Line(1) = {{1,2}}; Line(2) = {{2,3}}; Line(3) = {{3,4}}; Line(4) = {{4,1}};
Line Loop(1) = {{1,2,3,4}}; Plane Surface(1) = {{1}};
Physical Line("emitter") = {{4}}; Physical Line("ground") = {{2}}; Physical Line("ends") = {{1,3}};
Physical Surface("air") = {{1}};
"""
    mesh = fp.Gmsh2D(geo)
    return mesh, mesh.physicalFaces["emitter"], mesh.physicalFaces["ground"]


def ring_section(rings, r_outer: float, z_lo: float, z_hi: float, *, strip_h: float = 6e-3, strip_t: float = 0.2e-3,
                 ground_w: float = 10e-3, ground_h: float = 6e-3, edge_r: float = 0.1e-3, lc_edge: float = 15e-6,
                 lc_ground: float = 0.4e-3, lc_far: float = 5e-3):
    """A Gmsh section of a ring thruster for ``geometry="planar_axi"``: ``rings`` = [(radius, role, z)] in metres
    (``vegeta.fulgur.thruster``'s), the emitters as strips ``strip_h`` tall ending at their z with a rounded edge of
    radius ``edge_r`` (the teeth's effective tip), the grounds as ``ground_w x ground_h`` sections centred on their
    z. Physical lines: ``emitter`` (the whole strip), ``edge`` (its rounded end: where the Kaptzov injection is),
    ``ground``, ``axis``, ``outer``. Returns (mesh, emitter, ground, edge)."""
    import fipy as fp
    pts, lines, loops, phys = [], [], [], {"emitter": [], "edge": [], "ground": []}
    n = [0]
    def P(x, y, lc):
        n[0] += 1; pts.append(f"Point({n[0]}) = {{{x}, {y}, 0, {lc}}};"); return n[0]
    m = [0]
    def Ln(a, b):
        m[0] += 1; lines.append(f"Line({m[0]}) = {{{a},{b}}};"); return m[0]
    def Arc(a, c, b):
        m[0] += 1; lines.append(f"Circle({m[0]}) = {{{a},{c},{b}}};"); return m[0]
    # the domain: r in [0, r_outer], z in [z_lo, z_hi]
    p1, p2, p3, p4 = P(0, z_lo, lc_far), P(r_outer, z_lo, lc_far), P(r_outer, z_hi, lc_far), P(0, z_hi, lc_far)
    l1, l2, l3, l4 = Ln(p1, p2), Ln(p2, p3), Ln(p3, p4), Ln(p4, p1)
    loops.append(f"Line Loop(1) = {{{l1},{l2},{l3},{l4}}};")
    outer, axis = [l1, l2, l3], [l4]
    k = 1
    for r, role, z in rings:
        k += 1
        if role == "emitter":
            # a strip from z - strip_h to z - edge_r, closed by a semicircle of radius edge_r (= strip_t / 2 when equal)
            t = max(strip_t, 2 * edge_r)
            a = P(r - t / 2, z - strip_h, lc_ground); b = P(r + t / 2, z - strip_h, lc_ground)
            c = P(r + t / 2, z - t / 2, lc_edge); cc = P(r, z - t / 2, lc_edge); tip = P(r, z, lc_edge); d = P(r - t / 2, z - t / 2, lc_edge)
            la, lb, lc1, lc2, ld = Ln(a, b), Ln(b, c), Arc(c, cc, tip), Arc(tip, cc, d), Ln(d, a)   # two quarter arcs: Gmsh's arcs stay under pi
            loops.append(f"Line Loop({k}) = {{{la},{lb},{lc1},{lc2},{ld}}};")
            phys["emitter"] += [la, lb, lc1, lc2, ld]; phys["edge"] += [lc1, lc2]
        else:
            # a ground no wider than leaves 2 mm to its nearest neighbour ring (the hub ring is narrower than the rest)
            near = min(abs(r2 - r) for r2, _, _ in rings if r2 != r)
            w = min(ground_w, max(near - 2e-3, 2e-3))
            a = P(r - w / 2, z - ground_h / 2, lc_ground); b = P(r + w / 2, z - ground_h / 2, lc_ground)
            c = P(r + w / 2, z + ground_h / 2, lc_ground); d = P(r - w / 2, z + ground_h / 2, lc_ground)
            la, lb, lc_, ld = Ln(a, b), Ln(b, c), Ln(c, d), Ln(d, a)
            loops.append(f"Line Loop({k}) = {{{la},{lb},{lc_},{ld}}};")
            phys["ground"] += [la, lb, lc_, ld]
    geo = "\n".join(pts + lines + loops) + f"\nPlane Surface(1) = {{{','.join(str(i) for i in range(1, k + 1))}}};\n"
    geo += "".join(f'Physical Line("{name}") = {{{",".join(map(str, ids))}}};\n' for name, ids in phys.items() if ids)
    geo += f'Physical Line("axis") = {{{",".join(map(str, axis))}}};\nPhysical Line("outer") = {{{",".join(map(str, outer))}}};\n'
    geo += 'Physical Surface("air") = {1};\n'
    mesh = fp.Gmsh2D(geo)
    pf = mesh.physicalFaces
    return mesh, pf["emitter"] | pf["edge"], pf["ground"], pf["edge"]      # Gmsh files a line under one group only


def townsend_coaxial_current(V: float, V0: float, a: float, R: float, length: float, air: Air = DRY) -> float:
    """Townsend's coaxial corona current (1914): ``I = 8 π ε0 μ V (V − V0) L / (R^2 ln(R/a))``."""
    return 0.0 if V <= V0 else 8 * math.pi * EPS0 * air.mobility() * V * (V - V0) * length / (R * R * math.log(R / a))
