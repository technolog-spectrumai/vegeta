"""Triton kernels for D3Q19: fused pull-stream + boundaries + BGK (Guo forcing), and the macroscopic fields.

Memory layout (structure of arrays): population ``i`` of node ``n = (x·ny + y)·nz + z`` lives at
``f[i·N + n]``. A program handles ``BLOCK`` consecutive nodes, which share ``x`` and ``y`` except at row ends, so
for every direction the loads ``f[i·N + n − offset(c_i)]`` and the stores ``f_out[i·N + n]`` are contiguous runs
of ``BLOCK`` floats: coalesced (the loads are shifted by the neighbour offset, so they are not aligned to
128-byte lines; each warp touches at most one extra line per direction). Per node and step: 19 loads + 19 stores
of 4 bytes, plus a 4-byte link mask when solids are present.

Boundary handling follows :mod:`mancha.boundaries.conditions` exactly (same precedence, same formulas). The
face kinds are compile-time constants (one specialisation per boundary configuration); face velocities and
densities are read from a 24-float table. Solid links come from a per-node bit mask: bit ``i`` set when the
source of direction ``i`` is solid, bit 19 when the node itself is solid.

The numbering below must match :class:`mancha.lattice.D3Q19` (checked by the tests).
"""
from __future__ import annotations

import triton
import triton.language as tl

# (cx, cy, cz, weight, opposite) in D3Q19 order — see mancha.lattice.d3q19
W0 = 1.0 / 3.0
W1 = 1.0 / 18.0
W2 = 1.0 / 36.0
VELOCITIES = (
    (0, 0, 0, W0, 0),
    (1, 0, 0, W1, 2), (-1, 0, 0, W1, 1), (0, 1, 0, W1, 4), (0, -1, 0, W1, 3), (0, 0, 1, W1, 6), (0, 0, -1, W1, 5),
    (1, 1, 0, W2, 8), (-1, -1, 0, W2, 7), (1, -1, 0, W2, 10), (-1, 1, 0, W2, 9),
    (1, 0, 1, W2, 12), (-1, 0, -1, W2, 11), (1, 0, -1, W2, 14), (-1, 0, 1, W2, 13),
    (0, 1, 1, W2, 16), (0, -1, -1, W2, 15), (0, 1, -1, W2, 18), (0, -1, 1, W2, 17),
)
SOLID_BIT = 19
PERIODIC, PRESSURE, VELOCITY, MOVING_WALL, WALL = 0, 1, 2, 3, 4


@triton.jit
def _axis_key(key, s, n, C: tl.constexpr, CODE_LO: tl.constexpr, CODE_HI: tl.constexpr, RANK: tl.constexpr):
    """Fold one axis into the boundary key ``code·4 + rank`` (rank 2, 1, 0 for x, y, z) of the winning face."""
    if C == 1:
        key = tl.where(s < 0, tl.maximum(key, CODE_LO * 4 + RANK), key)
    elif C == -1:
        key = tl.where(s >= n, tl.maximum(key, CODE_HI * 4 + RANK), key)
    return key


@triton.jit
def _wrap(s, n):
    return tl.where(s < 0, s + n, tl.where(s >= n, s - n, s))


@triton.jit
def _pull(f_ptr, faces_ptr, node, x, y, z, nx, ny, nz, N, valid, links, rho_ref, ubx, uby, ubz, usq_b,
          I: tl.constexpr, CX: tl.constexpr, CY: tl.constexpr, CZ: tl.constexpr, W: tl.constexpr,
          OPP: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr,
          K4: tl.constexpr, K5: tl.constexpr, HAS_SOLID: tl.constexpr, HAS_FACES: tl.constexpr):
    """The population arriving at ``node`` along ``c_I``: pulled from ``node − c_I`` or supplied by a boundary."""
    if I == 0:
        return tl.load(f_ptr + node, mask=valid, other=0.0)
    sx = x - CX
    sy = y - CY
    sz = z - CZ
    src = (_wrap(sx, nx) * ny + _wrap(sy, ny)) * nz + _wrap(sz, nz)
    if HAS_FACES:
        key = tl.full(x.shape, -1, tl.int32)
        key = _axis_key(key, sx, nx, CX, K0, K1, 2)
        key = _axis_key(key, sy, ny, CY, K2, K3, 1)
        key = _axis_key(key, sz, nz, CZ, K4, K5, 0)
        kind = tl.where(key >= 0, key // 4, 0)
        on_face = kind > 0
    else:
        on_face = tl.zeros(x.shape, tl.int1)
    if HAS_SOLID:
        solid_link = ((links >> I) & 1) != 0
        bounce = on_face | solid_link
    else:
        bounce = on_face
    pulled = tl.load(f_ptr + I * N + src, mask=valid & (~bounce), other=0.0)
    back = tl.load(f_ptr + OPP * N + node, mask=valid & bounce, other=0.0)
    res = tl.where(bounce, back, pulled)                       # solid links and plain walls
    if HAS_FACES:
        rank = key % 4                                          # 2: x, 1: y, 0: z
        face = tl.where(rank == 2, 0 if CX == 1 else 1, tl.where(rank == 1, 2 if CY == 1 else 3, 4 if CZ == 1 else 5))
        face = tl.where(on_face, face, 0)
        uwx = tl.load(faces_ptr + face * 4 + 0, mask=on_face, other=0.0)
        uwy = tl.load(faces_ptr + face * 4 + 1, mask=on_face, other=0.0)
        uwz = tl.load(faces_ptr + face * 4 + 2, mask=on_face, other=0.0)
        rho_out = tl.load(faces_ptr + face * 4 + 3, mask=on_face, other=1.0)
        cuw = CX * uwx + CY * uwy + CZ * uwz
        res = tl.where((kind == 2) | (kind == 3), back + (6.0 * W) * rho_ref * cuw, res)
        cub = CX * ubx + CY * uby + CZ * ubz
        res = tl.where(kind == 1, -back + (2.0 * W) * rho_out * (1.0 + 4.5 * cub * cub - 1.5 * usq_b), res)
    return res


@triton.jit
def _relax_store(out_ptr, node, N, valid, fi, rho, ux, uy, uz, usq, fx, fy, fz, uf, omega, guo, solid_node,
                 rho_ref, I: tl.constexpr, CX: tl.constexpr, CY: tl.constexpr, CZ: tl.constexpr, W: tl.constexpr,
                 HAS_FORCE: tl.constexpr, HAS_SOLID: tl.constexpr):
    """BGK relaxation of one population (plus Guo's source) and its store; solid nodes get the rest state."""
    cu = CX * ux + CY * uy + CZ * uz
    feq = W * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    post = fi - omega * (fi - feq)
    if HAS_FORCE:
        cf = CX * fx + CY * fy + CZ * fz
        post += guo * W * (3.0 * (cf - uf) + 9.0 * cu * cf)
    if HAS_SOLID:
        post = tl.where(solid_node, W * rho_ref, post)
    tl.store(out_ptr + I * N + node, post, mask=valid)


@triton.jit
def _local(f_ptr, node, N, m, I: tl.constexpr):
    return tl.load(f_ptr + I * N + node, mask=m, other=0.0)


@triton.jit
def stream_collide_bgk(f_ptr, out_ptr, links_ptr, faces_ptr, nx, ny, nz, N, omega, rho_ref, fx, fy, fz,
                       K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr, K4: tl.constexpr,
                       K5: tl.constexpr, HAS_SOLID: tl.constexpr, HAS_FACES: tl.constexpr,
                       HAS_PRESSURE: tl.constexpr, HAS_FORCE: tl.constexpr, INT64: tl.constexpr,
                       BLOCK: tl.constexpr):
    """One time step for ``BLOCK`` nodes: pull (with boundaries), moments, BGK collision, store."""
    pid = tl.program_id(0)
    if INT64:
        node = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        node = pid * BLOCK + tl.arange(0, BLOCK)
    valid = node < N
    z = node % nz
    y = (node // nz) % ny
    x = node // (ny * nz)
    if HAS_SOLID:
        links = tl.load(links_ptr + node, mask=valid, other=0)
        solid_node = ((links >> SOLID_BIT) & 1) != 0
    else:
        links = tl.zeros(node.shape, tl.int32)
        solid_node = tl.zeros(node.shape, tl.int1)

    # velocity of the stored (post-collision) state at the node itself, needed by pressure faces only
    ubx = tl.zeros(node.shape, tl.float32)
    uby = tl.zeros(node.shape, tl.float32)
    ubz = tl.zeros(node.shape, tl.float32)
    usq_b = tl.zeros(node.shape, tl.float32)
    if HAS_PRESSURE:
        m = valid
        if K0 == 1:
            m = m & (x == 0)
        elif K1 == 1:
            m = m & (x == nx - 1)
        else:
            m = m & (x < 0)
        mp = m
        if K0 == 1 and K1 == 1:
            mp = valid & ((x == 0) | (x == nx - 1))
        my = valid & (((y == 0) & (K2 == 1)) | ((y == ny - 1) & (K3 == 1)))
        mz = valid & (((z == 0) & (K4 == 1)) | ((z == nz - 1) & (K5 == 1)))
        mp = mp | my | mz
        g0 = _local(f_ptr, node, N, mp, 0)
        g1 = _local(f_ptr, node, N, mp, 1)
        g2 = _local(f_ptr, node, N, mp, 2)
        g3 = _local(f_ptr, node, N, mp, 3)
        g4 = _local(f_ptr, node, N, mp, 4)
        g5 = _local(f_ptr, node, N, mp, 5)
        g6 = _local(f_ptr, node, N, mp, 6)
        g7 = _local(f_ptr, node, N, mp, 7)
        g8 = _local(f_ptr, node, N, mp, 8)
        g9 = _local(f_ptr, node, N, mp, 9)
        g10 = _local(f_ptr, node, N, mp, 10)
        g11 = _local(f_ptr, node, N, mp, 11)
        g12 = _local(f_ptr, node, N, mp, 12)
        g13 = _local(f_ptr, node, N, mp, 13)
        g14 = _local(f_ptr, node, N, mp, 14)
        g15 = _local(f_ptr, node, N, mp, 15)
        g16 = _local(f_ptr, node, N, mp, 16)
        g17 = _local(f_ptr, node, N, mp, 17)
        g18 = _local(f_ptr, node, N, mp, 18)
        rb = (g0 + g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9 + g10 + g11 + g12 + g13 + g14 + g15 + g16 + g17
              + g18)
        rb = tl.where(mp, rb, 1.0)
        ubx = (g1 - g2 + g7 - g8 + g9 - g10 + g11 - g12 + g13 - g14 - 0.5 * fx) / rb
        uby = (g3 - g4 + g7 - g8 - g9 + g10 + g15 - g16 + g17 - g18 - 0.5 * fy) / rb
        ubz = (g5 - g6 + g11 - g12 - g13 + g14 + g15 - g16 - g17 + g18 - 0.5 * fz) / rb
        usq_b = ubx * ubx + uby * uby + ubz * ubz

    a = (f_ptr, faces_ptr, node, x, y, z, nx, ny, nz, N, valid, links, rho_ref, ubx, uby, ubz, usq_b)
    f0 = _pull(*a, 0, 0, 0, 0, W0, 0, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f1 = _pull(*a, 1, 1, 0, 0, W1, 2, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f2 = _pull(*a, 2, -1, 0, 0, W1, 1, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f3 = _pull(*a, 3, 0, 1, 0, W1, 4, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f4 = _pull(*a, 4, 0, -1, 0, W1, 3, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f5 = _pull(*a, 5, 0, 0, 1, W1, 6, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f6 = _pull(*a, 6, 0, 0, -1, W1, 5, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f7 = _pull(*a, 7, 1, 1, 0, W2, 8, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f8 = _pull(*a, 8, -1, -1, 0, W2, 7, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f9 = _pull(*a, 9, 1, -1, 0, W2, 10, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f10 = _pull(*a, 10, -1, 1, 0, W2, 9, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f11 = _pull(*a, 11, 1, 0, 1, W2, 12, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f12 = _pull(*a, 12, -1, 0, -1, W2, 11, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f13 = _pull(*a, 13, 1, 0, -1, W2, 14, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f14 = _pull(*a, 14, -1, 0, 1, W2, 13, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f15 = _pull(*a, 15, 0, 1, 1, W2, 16, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f16 = _pull(*a, 16, 0, -1, -1, W2, 15, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f17 = _pull(*a, 17, 0, 1, -1, W2, 18, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)
    f18 = _pull(*a, 18, 0, -1, 1, W2, 17, K0, K1, K2, K3, K4, K5, HAS_SOLID, HAS_FACES)

    rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18
    rho = tl.where(valid, rho, 1.0)
    inv = 1.0 / rho
    ux = (f1 - f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14 + 0.5 * fx) * inv
    uy = (f3 - f4 + f7 - f8 - f9 + f10 + f15 - f16 + f17 - f18 + 0.5 * fy) * inv
    uz = (f5 - f6 + f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + 0.5 * fz) * inv
    usq = ux * ux + uy * uy + uz * uz
    uf = ux * fx + uy * fy + uz * fz
    guo = 1.0 - 0.5 * omega

    b = (out_ptr, node, N, valid)
    c = (rho, ux, uy, uz, usq, fx, fy, fz, uf, omega, guo, solid_node, rho_ref)
    _relax_store(*b, f0, *c, 0, 0, 0, 0, W0, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f1, *c, 1, 1, 0, 0, W1, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f2, *c, 2, -1, 0, 0, W1, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f3, *c, 3, 0, 1, 0, W1, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f4, *c, 4, 0, -1, 0, W1, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f5, *c, 5, 0, 0, 1, W1, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f6, *c, 6, 0, 0, -1, W1, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f7, *c, 7, 1, 1, 0, W2, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f8, *c, 8, -1, -1, 0, W2, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f9, *c, 9, 1, -1, 0, W2, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f10, *c, 10, -1, 1, 0, W2, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f11, *c, 11, 1, 0, 1, W2, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f12, *c, 12, -1, 0, -1, W2, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f13, *c, 13, 1, 0, -1, W2, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f14, *c, 14, -1, 0, 1, W2, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f15, *c, 15, 0, 1, 1, W2, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f16, *c, 16, 0, -1, -1, W2, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f17, *c, 17, 0, 1, -1, W2, HAS_FORCE, HAS_SOLID)
    _relax_store(*b, f18, *c, 18, 0, -1, 1, W2, HAS_FORCE, HAS_SOLID)


@triton.jit
def macroscopic(f_ptr, links_ptr, out_ptr, N, fx, fy, fz, HAS_SOLID: tl.constexpr, INT64: tl.constexpr,
                BLOCK: tl.constexpr):
    """Density and velocity of the stored state into ``out[4, N]`` (ρ, ux, uy, uz); zero velocity in solids."""
    pid = tl.program_id(0)
    if INT64:
        node = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        node = pid * BLOCK + tl.arange(0, BLOCK)
    valid = node < N
    g0 = _local(f_ptr, node, N, valid, 0)
    g1 = _local(f_ptr, node, N, valid, 1)
    g2 = _local(f_ptr, node, N, valid, 2)
    g3 = _local(f_ptr, node, N, valid, 3)
    g4 = _local(f_ptr, node, N, valid, 4)
    g5 = _local(f_ptr, node, N, valid, 5)
    g6 = _local(f_ptr, node, N, valid, 6)
    g7 = _local(f_ptr, node, N, valid, 7)
    g8 = _local(f_ptr, node, N, valid, 8)
    g9 = _local(f_ptr, node, N, valid, 9)
    g10 = _local(f_ptr, node, N, valid, 10)
    g11 = _local(f_ptr, node, N, valid, 11)
    g12 = _local(f_ptr, node, N, valid, 12)
    g13 = _local(f_ptr, node, N, valid, 13)
    g14 = _local(f_ptr, node, N, valid, 14)
    g15 = _local(f_ptr, node, N, valid, 15)
    g16 = _local(f_ptr, node, N, valid, 16)
    g17 = _local(f_ptr, node, N, valid, 17)
    g18 = _local(f_ptr, node, N, valid, 18)
    rho = g0 + g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9 + g10 + g11 + g12 + g13 + g14 + g15 + g16 + g17 + g18
    safe = tl.where(valid, rho, 1.0)
    ux = (g1 - g2 + g7 - g8 + g9 - g10 + g11 - g12 + g13 - g14 - 0.5 * fx) / safe
    uy = (g3 - g4 + g7 - g8 - g9 + g10 + g15 - g16 + g17 - g18 - 0.5 * fy) / safe
    uz = (g5 - g6 + g11 - g12 - g13 + g14 + g15 - g16 - g17 + g18 - 0.5 * fz) / safe
    if HAS_SOLID:
        solid = ((tl.load(links_ptr + node, mask=valid, other=0) >> SOLID_BIT) & 1) != 0
        ux = tl.where(solid, 0.0, ux)
        uy = tl.where(solid, 0.0, uy)
        uz = tl.where(solid, 0.0, uz)
    tl.store(out_ptr + node, rho, mask=valid)
    tl.store(out_ptr + N + node, ux, mask=valid)
    tl.store(out_ptr + 2 * N + node, uy, mask=valid)
    tl.store(out_ptr + 3 * N + node, uz, mask=valid)
