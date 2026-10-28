"""The Triton backend against the NumPy reference, configuration by configuration.

Runs on CUDA, or on the CPU when TRITON_INTERPRET=1 is set before the tests start (``./test.sh --interpret``);
skipped otherwise. Grids are tiny so the interpreter stays fast.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET", "0") in ("", "0"):
    pytest.skip("no CUDA and TRITON_INTERPRET is not set", allow_module_level=True)

from mancha import (  # noqa: E402
    BGK,
    D3Q19,
    WALE,
    InvalidParameterError,
    NumericalInstabilityError,
    RegularizedBGK,
    RotatingFrame,
    Simulation,
)
from mancha.backends import triton_kernels as K  # noqa: E402
from mancha.backends.triton_backend import link_mask  # noqa: E402
from mancha.boundaries import sphere  # noqa: E402

pytestmark = pytest.mark.gpu
SHAPE = (10, 7, 6)
RNG = np.random.default_rng(7)


def _pair(setup, steps=12, shape=SHAPE, **kw):
    """Run the same configuration on both backends from the same perturbed initial state."""
    rho = 1.0 + 0.01 * RNG.standard_normal(shape)
    u = 0.02 * RNG.standard_normal((3,) + shape)
    out = []
    for backend in ("numpy", "triton"):
        sim = Simulation(shape=shape, backend=backend, **kw)
        setup(sim)
        sim.initialize(density=rho, velocity=u)
        sim.run(steps)
        out.append(sim)
    return out


def _agree(a, b, tol=2e-5):
    fa, fb = a.populations(), b.populations()
    assert np.isfinite(fb).all()
    assert np.abs(fa - fb).max() < tol, np.abs(fa - fb).max()
    assert np.abs(a.density() - b.density()).max() < tol
    assert np.abs(a.velocity() - b.velocity()).max() < tol


def test_kernel_constants_match_the_lattice():
    lat = D3Q19()
    table = np.array(K.VELOCITIES)
    assert (table[:, :3] == lat.c).all()
    assert np.allclose(table[:, 3], lat.w)
    assert (table[:, 4] == lat.opposite).all()


def test_link_mask():
    solid = np.zeros((5, 4, 3), bool)
    solid[2, 1, 1] = True
    bits = link_mask(solid, D3Q19().c).reshape(solid.shape)
    assert (bits[2, 1, 1] >> K.SOLID_BIT.value) & 1
    i = 1                                          # c = (+1, 0, 0): pulls from x − 1
    assert (bits[3, 1, 1] >> i) & 1 and not (bits[1, 1, 1] >> i) & 1


def test_periodic():
    _agree(*_pair(lambda s: None, viscosity=0.05))


def test_body_force():
    _agree(*_pair(lambda s: (s.set_boundary("y-", "wall"), s.set_boundary("y+", "wall")), viscosity=0.1,
                  force=(1e-5, 2e-6, 0.0)))


def test_walls_and_moving_lid():
    def setup(s):
        for face in ("x-", "x+", "y-", "z-", "z+"):
            s.set_boundary(face, "wall")
        s.set_boundary("y+", "wall", velocity=(0.05, 0, 0.01))
    _agree(*_pair(setup, viscosity=0.05))


def test_inlet_and_outlet_on_every_axis():
    for axis in "xyz":
        others = [a for a in "xyz" if a != axis]

        def setup(s, axis=axis, others=others):
            u = [0.0, 0.0, 0.0]
            u["xyz".index(axis)] = 0.03
            s.set_boundary(f"{axis}-", "velocity", velocity=u)
            s.set_boundary(f"{axis}+", "pressure", density=1.002)
            s.set_boundary(f"{others[0]}-", "wall")
            s.set_boundary(f"{others[0]}+", "wall")
        _agree(*_pair(setup, viscosity=0.05))


def test_reversed_inlet_outlet_and_mixed_precedence():
    def setup(s):
        s.set_boundary("x+", "velocity", velocity=(-0.02, 0.01, 0))
        s.set_boundary("x-", "pressure", density=0.998)
        s.set_boundary("y-", "pressure", density=1.0)
        s.set_boundary("y+", "wall", velocity=(0.01, 0, 0))
        s.set_boundary("z-", "velocity", velocity=(0, 0, 0.01))
        s.set_boundary("z+", "wall")
    _agree(*_pair(setup, viscosity=0.08))


def test_solid_obstacle_with_inflow():
    def setup(s):
        s.set_boundary("x-", "velocity", velocity=(0.03, 0, 0))
        s.set_boundary("x+", "pressure", density=1.0)
        s.set_solid(sphere(SHAPE, (4, 3, 2.5), 1.6))
    a, b = _pair(setup, viscosity=0.05)
    _agree(a, b)
    assert np.allclose(a.solid_force(), b.solid_force(), atol=1e-5)


def test_solid_touching_a_periodic_wrap():
    mask = np.zeros(SHAPE, bool)
    mask[0, 2:4, :] = True                      # wraps across x
    _agree(*_pair(lambda s: s.set_solid(mask), viscosity=0.05))


def test_taylor_green_matches():
    from mancha.validation import taylor_green

    a = taylor_green(n=12, steps=30)
    b = taylor_green(n=12, steps=30, backend="triton")
    assert abs(a["decay_rate"] / b["decay_rate"] - 1) < 1e-4


def test_block_sizes_give_identical_results():
    runs = []
    for block in (16, 64, 128):
        sim = Simulation(shape=SHAPE, viscosity=0.05, velocity=(0.02, 0, 0), backend="triton", block_size=block)
        sim.set_boundary("y-", "wall").set_boundary("y+", "wall")
        sim.run(5)
        runs.append(sim.populations())
    assert all(np.array_equal(runs[0], r) for r in runs[1:])


def test_checkpoint_moves_between_backends(tmp_path):
    a = Simulation(shape=SHAPE, viscosity=0.05, velocity=(0.02, 0.01, 0), backend="triton")
    a.set_boundary("y-", "wall").set_boundary("y+", "wall")
    a.run(5)
    b = Simulation.restore(a.checkpoint(tmp_path / "c.npz"), backend="numpy", dtype=np.float32)
    a.run(5)
    b.run(5)
    assert np.abs(a.populations() - b.populations()).max() < 1e-6


def test_nan_detected_on_the_device():
    sim = Simulation(shape=SHAPE, viscosity=0.05, backend="triton", check_interval=3)
    f = sim.populations()
    f[5, 1, 1, 1] = np.inf
    sim.backend.set_populations(f)
    with pytest.raises(NumericalInstabilityError):
        sim.run(6)


@pytest.mark.parametrize("kw", [{"dtype": np.float64}, {"block_size": 100}, {"block_size": 8}])
def test_bad_backend_parameters(kw):
    with pytest.raises(InvalidParameterError):
        Simulation(shape=SHAPE, viscosity=0.05, backend="triton", **kw)


def test_unknown_collision_kind_is_refused():
    class Other(BGK):
        kind = "other"

    with pytest.raises(InvalidParameterError):
        Simulation(shape=SHAPE, collision=Other(0.8), backend="triton")


# ---------------------------------------------------------------- rotating frame, WALE, regularized collision
FRAME = RotatingFrame(0.004, center=(3.0, 2.5))


def test_rotating_frame_far_field_inlet_outlet():
    def setup(s):
        s.set_boundary("x-", "velocity", velocity=(0.02, 0, 0)).set_boundary("x+", "pressure", density=1.0)
        for face in ("y-", "y+", "z-", "z+"):
            s.set_boundary(face, "velocity", velocity=(0.02, 0, 0))
    _agree(*_pair(setup, viscosity=0.05, frame=FRAME))


def test_rotating_frame_with_blade_and_end_walls():
    blade = np.zeros(SHAPE, bool)
    blade[4:6, 1:6, 2:4] = True                    # an off-axis block, static in the frame

    def setup(s):
        s.set_boundary("x-", "wall").set_boundary("x+", "wall")
        s.set_solid(blade)
    a, b = _pair(setup, viscosity=0.05, frame=FRAME, force=(1e-5, 0, 0))
    _agree(a, b)
    fa, ma = a.solid_load()
    fb, mb = b.solid_load()
    assert np.allclose(fa, fb, atol=1e-5) and np.allclose(ma, mb, atol=1e-4)


def test_wale_with_walls_and_solids():
    mask = np.zeros(SHAPE, bool)
    mask[4:6, 2:4, 2:4] = True

    def setup(s):
        s.set_boundary("y-", "wall").set_boundary("y+", "wall", velocity=(0.04, 0, 0.01))
        s.set_boundary("z-", "velocity", velocity=(0.01, 0, 0.02)).set_boundary("z+", "pressure", density=1.0)
        s.set_solid(mask)
    _agree(*_pair(setup, viscosity=0.002, turbulence=WALE(0.5)))


def test_regularized_with_force():
    _agree(*_pair(lambda s: (s.set_boundary("y-", "wall"), s.set_boundary("y+", "wall")),
                  collision=RegularizedBGK(0.56), force=(1e-5, 2e-6, 0.0)))


def test_sponge_with_frame():
    from mancha import Sponge

    def setup(s):
        for face in ("y-", "y+", "z-", "z+"):
            s.set_boundary(face, "velocity", velocity=(0.01, 0, 0))
    _agree(*_pair(setup, collision=RegularizedBGK(0.52), frame=FRAME,
                  sponge=Sponge(width=2.5, viscosity=0.05, faces=("y-", "y+", "z+"))))


def test_everything_together():
    mask = np.zeros(SHAPE, bool)
    mask[4:6, 1:3, 2:5] = True

    def setup(s):
        s.set_boundary("x-", "velocity", velocity=(0.02, 0, 0)).set_boundary("x+", "pressure", density=1.0)
        for face in ("y-", "y+", "z-", "z+"):
            s.set_boundary(face, "velocity", velocity=(0.02, 0, 0))
        s.set_solid(mask)
    from mancha import Sponge

    _agree(*_pair(setup, collision=RegularizedBGK(0.505), frame=FRAME, turbulence=WALE(), force=(2e-6, 0, 0),
                  sponge=Sponge(width=2.0, viscosity=0.03)), tol=3e-5)


def test_wale_kernel_matches_the_reference():
    """The kernel's WALE viscosity on a random field, periodic and with walls and a solid, rotating frame or not
    (a wrong exponent here once passed every other test that ran few steps)."""
    import triton
    import triton.language as tl

    from mancha.turbulence import to_lab_gradient, velocity_gradient, wale_viscosity

    @triton.jit
    def probe(m_ptr, links_ptr, out_ptr, nx, ny, nz, N, cw2, om, PX: tl.constexpr, HAS_SOLID: tl.constexpr,
              HAS_FRAME: tl.constexpr, BLOCK: tl.constexpr):
        node = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = node < N
        z = node % nz
        y = (node // nz) % ny
        x = node // (ny * nz)
        if HAS_SOLID:
            links = tl.load(links_ptr + node, mask=valid, other=0)
        else:
            links = tl.zeros(node.shape, tl.int32)
        nut = K._wale(m_ptr, node, x, y, z, nx, ny, nz, N, valid, links, cw2, om, PX, True, True, HAS_SOLID,
                      HAS_FRAME)
        tl.store(out_ptr + node, nut, mask=valid)

    shape = (6, 5, 4)
    n = int(np.prod(shape))
    u = (0.02 * RNG.standard_normal((3,) + shape)).astype(np.float32)
    solid = np.zeros(shape, bool)
    solid[2, 1:3, 1] = True
    u[:, solid] = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for periodic_x, with_solid, om in ((True, False, 0.0), (False, True, 0.0), (False, True, 0.003)):
        out = torch.zeros(n, device=device)
        links = torch.from_numpy(link_mask(solid, D3Q19().c)).to(device)
        probe[(triton.cdiv(n, 64),)](torch.from_numpy(u.copy()).to(device), links, out, *shape, n, 0.25, om,
                                     PX=periodic_x, HAS_SOLID=with_solid, HAS_FRAME=om != 0.0, BLOCK=64)
        g = velocity_gradient(u.astype(np.float64), solid if with_solid else None, (periodic_x, True, True))
        if om:
            to_lab_gradient(g, om)
        ref = wale_viscosity(g, 0.5).reshape(-1)
        assert np.abs(out.cpu().numpy() - ref).max() < 1e-4 * ref.max()
