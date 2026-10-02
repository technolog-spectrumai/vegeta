"""Tonal noise of a ducted fan (rotor + stator): the rotor wakes striking the stator vanes, the spinning modes this
makes (Tyler & Sofrin 1962), which of them propagate, and their levels at an observer. First estimates for comparing
designs (blade and vane counts, rotor-stator spacing, rpm), not certification numbers.

- ``tyler_sofrin_modes``: the circumferential mode orders ``n = mB - sV`` a rotor of B blades and a stator of V vanes
  can make at m x BPF.
- ``cut_on_ratio``: whether a spinning mode propagates (> 1) or decays (< 1): ``m B M_tip (r/R_tip) / |n|``.
- ``vane_count_study``: the lowest mode of each harmonic for a list of vane counts and whether it is cut on — the
  Tyler–Sofrin rule for choosing V.
- ``rotor_wake_harmonics``: the rotor wakes a vane sees (Silverstein's wake width and depth, one wake per blade
  pitch), as harmonics of the blade-passing frequency.
- ``sears``: the Sears function amplitude, how much of a quasi-steady gust lift a vane actually develops.
- ``interaction_tones``: the unsteady vane lift (thin aerofoil in a transverse gust, Sears) of every vane, summed
  as stationary compact dipoles around the ring of vanes into spinning modes, at a far-field observer.

Conventions: the axis +x points downstream (the flow direction through the fan, towards the nozzle); ``angle_deg``
(theta) is measured from +x — 0 on the axis behind the fan (exhaust side), 90 in the fan plane, 180 on the axis in
front of the intake. The rotor turns towards +phi; ``azimuth_deg`` (phi_o) is measured from vane 0 in the sense of
rotation. A mode ``n > 0`` spins with the rotor (at m B Omega / n), ``n < 0`` against it, ``n = 0`` is a plane wave.

Not included: the duct itself (the modes radiate here as from the bare ring of vanes, so a cut-off mode decays by the
free-field Bessel drop-off; ``interaction_tones`` can add the exponential decay of cut-off modes along a length of
hard-walled duct as an option, but there is no inlet or nozzle termination, hub or liner), rotor-alone tones (the
steady-loading mode n = mB, cut off at subsonic tip speeds) and the rotor's response to the stator's potential field,
broadband noise (turbulence, tip clearance), the vane's non-compactness along its chord and span, and forward-flight
(Doppler) effects. Levels in dB re 20 uPa (air) or 1 uPa (water).
"""
from __future__ import annotations

import math

import numpy as np

from .noise import P_REF_AIR, P_REF_WATER, Medium, _bessel

LN2 = math.log(2.0)


def _count(value, name: str) -> int:
    if int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return int(value)


def _jn(n: int, x: float) -> float:
    """J_n(x) for any integer order (J_{-n} = (-1)^n J_n; the series fallback of ``_bessel`` needs n >= 0)."""
    j = float(_bessel(abs(n), np.array([float(x)]))[0])
    return -j if n < 0 and n % 2 else j


def _duct_cut_off(n: int) -> float:
    """k R_d at which the spinning mode n (first radial order) cuts on in a hard-walled cylindrical duct of radius R_d:
    j'_{n,1}, the first zero of J_n' (0 for the plane wave, which always propagates)."""
    n = abs(int(n))
    if n == 0:
        return 0.0
    try:
        from scipy.special import jnp_zeros
        return float(jnp_zeros(n, 1)[0])
    except ImportError:  # pragma: no cover - Olver's asymptotic form, within 2 % at n = 1 and better above
        return n + 0.8086 * n ** (1 / 3) + 0.0725 * n ** (-1 / 3)


def tyler_sofrin_modes(blades: int, vanes: int, harmonics: int = 3, s_range: int = 4) -> list[dict]:
    """The circumferential mode orders of rotor-stator interaction (Tyler & Sofrin 1962): at m x BPF the B rotor wakes
    sweeping over V vanes make the pressure patterns ``n = m B - s V`` (s any integer; the vane loading repeats every
    vane, so only these survive the sum over the vanes). For m = 1..``harmonics`` and s = -``s_range``..``s_range``,
    sorted by |n| within each harmonic (ties by s). Returns ``[{"harmonic", "s", "n"}, ...]``."""
    B, V = _count(blades, "blades"), _count(vanes, "vanes")
    if harmonics < 1 or s_range < 0:
        raise ValueError("harmonics >= 1 and s_range >= 0")
    rows = []
    for m in range(1, harmonics + 1):
        modes = [{"harmonic": m, "s": s, "n": m * B - s * V} for s in range(-s_range, s_range + 1)]
        rows += sorted(modes, key=lambda d: (abs(d["n"]), d["s"]))
    return rows


def cut_on_ratio(m: int, blades: int, n: int, tip_mach: float, radius_ratio: float = 1.0) -> float:
    """Cut-on ratio of the spinning mode n at m x BPF: ``(m B M_tip radius_ratio) / |n|`` (inf for the plane wave n = 0).

    The mode's pattern turns at m B Omega / n, so at radius r its phase speed along the circumference is
    m B Omega r / n; it propagates when that is supersonic, i.e. when the Bessel argument k r = m B M_tip (r / R_tip)
    exceeds the order |n| (a mode whose pattern moves subsonically decays away from the source). This is the
    free-field form of the criterion: in a hard-walled cylindrical duct of radius R a mode is cut on above
    k R = j'_{n,1} (the first zero of J_n'), j'_{n,1} ~ |n| + 0.81 |n|^(1/3) for large n (1.84 for n = 1, 3.05 for
    n = 2, 13.9 for n = 12). Using |n| marks modes cut on slightly early — a ratio between 1 and j'_{n,1}/|n| is still
    cut off in the duct — so it errs on the noisy side. ``radius_ratio`` = r / R_tip (1 for the duct wall)."""
    if tip_mach < 0 or radius_ratio < 0:
        raise ValueError("tip_mach and radius_ratio must be >= 0")
    if n == 0:
        return float("inf")
    return float(m * blades * tip_mach * radius_ratio / abs(n))


def vane_count_study(blades: int, vane_counts, tip_mach: float, harmonics: int = 2) -> list[dict]:
    """The Tyler–Sofrin vane-count rule: for every V in ``vane_counts`` the lowest-order interaction mode
    ``n = m B - s V`` of each harmonic m = 1..``harmonics`` and whether it propagates (``cut_on_ratio`` > 1 at the tip).

    The lowest |n| is the distance from m B to the nearest multiple of V. Harmonic m is cut off (all its interaction
    modes decay) when that distance exceeds m B M_tip: always so for V > m B (1 + M_tip) (the nearest multiples are
    0, i.e. the subsonic n = m B, and V), never for V < 2 m B M_tip (some multiple is within V/2), and between the two it
    depends on B and V. With B = 12 at M_tip = 0.4, 1 x BPF needs V >= 17 (V = 7 leaves n = 12 - 2 x 7 = -2 cut on); for
    sonic tips the rule becomes the classic V > 2 B. 2 x BPF is cut off for sure only above V = 2 B (1 + M_tip).
    Returns per V ``{"vanes", "modes": [{"harmonic", "n", "s", "cut_on_ratio", "cut_on"}], "cut_on_harmonics",
    "bpf_cut_off"}``."""
    B = _count(blades, "blades")
    if harmonics < 1:
        raise ValueError("harmonics must be >= 1")
    rows = []
    for vanes in vane_counts:
        V = _count(vanes, "vanes")
        modes = []
        for m in range(1, harmonics + 1):
            s = (m * B) // V                                   # the multiples of V either side of mB: s V <= mB < (s+1) V
            n_lo, n_hi = m * B - s * V, m * B - (s + 1) * V    # n_lo >= 0 > n_hi
            n, s = (n_lo, s) if n_lo <= -n_hi else (n_hi, s + 1)     # ties go to the smaller s, as in tyler_sofrin_modes
            ratio = cut_on_ratio(m, B, n, tip_mach)
            modes.append({"harmonic": m, "n": n, "s": s, "cut_on_ratio": ratio, "cut_on": ratio > 1.0})
        rows.append({"vanes": V, "modes": modes, "cut_on_harmonics": [d["harmonic"] for d in modes if d["cut_on"]],
                     "bpf_cut_off": not modes[0]["cut_on"]})
    return rows


def rotor_wake_harmonics(blades: int, chord_m: float, cd: float, spacing_m: float, radius_m: float, harmonics: int = 4,
                         n_samples: int = 4096, *, wake_angle_deg: float = 0.0) -> np.ndarray:
    """The rotor-wake velocity deficit a stator vane sees at ``radius_m``, ``spacing_m`` behind the rotor trailing edge,
    as a fraction of the blade's local relative velocity W: index 0 the circumferential mean, index m = 1..``harmonics``
    the amplitude at m x BPF.

    Each blade (chord c, profile drag coefficient ``cd`` including its losses) sheds a wake with, at a distance x
    downstream of the trailing edge (Silverstein, Katzoff & Bullivant 1939, in the form fan-noise prediction uses
    after Kemp & Sears 1955):

        centreline deficit   u_c / W = 2.42 sqrt(cd) / (x/c + 0.3)
        half width at half deficit   b = 0.68 c sqrt(cd (x/c + 0.15))
        profile   u / W = (u_c / W) exp(-ln2 (y / b)^2)          (Gaussian, y across the wake)

    The B wakes repeat every blade pitch 2 pi r / B around the circumference and sweep past the vane once per blade
    passage, so the deficit is sampled over one pitch (``n_samples`` points, overlapping wakes summed) and Fourier
    analysed. The mean is the wake's area over the pitch, u_c b sqrt(pi / ln2) / (2 pi r / B), and the harmonics are
    2 x mean x exp(-(pi m b / pitch)^2 / ln2) (Poisson summation: the Fourier coefficients of the repeated Gaussian are
    samples of its transform, overlapping or not): they fall off faster the wider the wake is against the pitch, so
    more spacing (a wider, shallower wake) weakens the higher harmonics most.

    ``wake_angle_deg``: the wake does not travel straight downstream but along the blade's relative exit flow, at this
    angle from the axis (often 40-60 deg in a fan). The path to the vane plane is then spacing / cos(angle), and a cut
    across the inclined wake sheet at the vane plane is b / cos(angle) wide along the circumference. 0 (default)
    takes the wake straight downstream.

    The integral of this wake, 3.5 cd c sqrt(x/c + 0.15) / (x/c + 0.3), is several times the momentum deficit
    cd c / 2 of the section drag alone at x ~ c: for a clean section it is an upper estimate (real fan-rotor wakes also
    carry tip-clearance and secondary-flow losses — put them in ``cd``, or scale the result)."""
    B = _count(blades, "blades")
    if chord_m <= 0 or cd < 0 or spacing_m < 0 or radius_m <= 0:
        raise ValueError("chord_m > 0, cd >= 0, spacing_m >= 0, radius_m > 0")
    if harmonics < 1 or n_samples < 2 * harmonics + 2:
        raise ValueError("harmonics >= 1 and n_samples >= 2 harmonics + 2")
    if not 0.0 <= wake_angle_deg < 90.0:
        raise ValueError("wake_angle_deg must be in [0, 90)")
    if cd == 0:
        return np.zeros(harmonics + 1)
    cos_b = math.cos(math.radians(wake_angle_deg))
    xc = spacing_m / cos_b / chord_m                                        # distance travelled along the wake, chords
    u_c = 2.42 * math.sqrt(cd) / (xc + 0.3)
    b = 0.68 * chord_m * math.sqrt(cd * (xc + 0.15)) / cos_b                # half width along the circumference
    pitch = 2 * math.pi * radius_m / B
    y = np.arange(n_samples) * pitch / n_samples
    copies = int(math.ceil(8 * b / pitch)) + 1                             # neighbouring wakes whose tails reach this pitch
    j = np.arange(-copies, copies + 1)
    deficit = u_c * np.exp(-LN2 * ((y[:, None] - j[None, :] * pitch) / b) ** 2).sum(axis=1)
    X = np.fft.rfft(deficit) / n_samples
    a = 2 * np.abs(X[:harmonics + 1])
    a[0] = X[0].real
    return a


def sears(reduced_frequency):
    """Amplitude of the Sears function, the usual fit ``|S(k)| ~ 1 / sqrt(1 + 2 pi k)``: the unsteady lift of a thin
    aerofoil in a sinusoidal transverse gust relative to its quasi-steady value, at the reduced frequency
    k = omega c / (2 U) (c chord, U the mean flow over it). 1 at k = 0 and -> 1 / sqrt(2 pi k) at high k, as the
    exact function; in between the fit is up to 8 % (0.7 dB) below the exact amplitude (most near k ~ 0.3).
    A float for a scalar, an array for an array."""
    k = np.asarray(reduced_frequency, dtype=float)
    if np.any(k < 0):
        raise ValueError("reduced_frequency must be >= 0")
    s = 1.0 / np.sqrt(1.0 + 2 * math.pi * k)
    return float(s) if s.ndim == 0 else s


def interaction_tones(blades: int, vanes: int, rpm: float, *, vane_radius_m: float, vane_chord_m: float,
                      vane_span_m: float, vane_inflow_m_s: float, gust_m_s, distance: float, medium: Medium,
                      angle_deg: float = 90.0, azimuth_deg: float = 0.0, stagger_deg: float = 0.0, harmonics: int = 4,
                      s_range: int = 6, duct_radius_m: float | None = None, duct_length_m: float = 0.0) -> list[dict]:
    """Rotor-stator interaction tones at m x BPF (m = 1..``harmonics``) at ``distance`` [m], ``angle_deg`` (theta, from
    the downstream axis) and ``azimuth_deg`` (phi_o, from vane 0 in the sense of rotation).

    Vane load. ``gust_m_s[m]`` is the upwash amplitude (velocity normal to the vane chord) at m x BPF on each vane,
    e.g. ``rotor_wake_harmonics(...)[m] x W x |sin(angle between the rotor relative flow and the vane chord)|`` (the
    wake deficit points along the rotor's relative flow W); index 0 is not used. A vane of chord c and span ``span``
    in a mean flow U (``vane_inflow_m_s``, the absolute velocity at the stator inlet) develops the unsteady lift
    (thin aerofoil in a transverse gust, Sears):

        L_m = pi rho c U w_m |S(k_m)| span,   k_m = omega_m c / (2 U),   omega_m = m B Omega

    normal to the chord. ``stagger_deg`` (xi) is the chord's angle from the axis, positive when the chord leans from
    leading to trailing edge towards the direction of rotation (a stator taking out the rotor's swirl); the normal to
    the chord direction (cos xi, sin xi) in (x, phi) is (-sin xi, cos xi), so the lift has an axial part
    F_a = L sin xi and a tangential part F_t = L cos xi of opposite signs in the (+x, +phi) axes.

    Radiation. The V vanes are stationary compact dipoles on a ring of radius R (``vane_radius_m``, where the vane
    loading is centred) at phi_v = 2 pi v / V; the rotor wakes reach vane v later by phi_v / Omega, so vane v carries
    the same force with the phase -m B phi_v. A force F e^{i omega t} on the fluid at y radiates, far away,
    p = i omega (x_hat . F) e^{i omega (t - r/c)} / (4 pi c r) with r ~ r_o - x_hat . y and
    x_hat . y_v = R sin(theta) cos(phi_v - phi_o). Expanding e^{i z cos a} = sum_n i^n J_n(z) e^{i n a} (Jacobi–Anger)
    and summing over the vanes keeps only n = m B - s V (the sum of e^{i (n - mB) phi_v} is V there, 0 elsewhere):

        p_m = (omega_m V / (4 pi c r)) | sum_s  i^n e^{-i n phi_o} [F_a cos(theta) J_n(z) - F_t n J_n(z) / (k R)] |,
        z = k R sin(theta),  k = omega_m / c,  n = m B - s V

    an amplitude; the rms is p_m / sqrt2 (as ``boreas.wake.rotating_tones``). n J_n(z) / (k R) is evaluated as
    sin(theta) (J_{n-1}(z) + J_{n+1}(z)) / 2, finite on the axis and for R = 0. Modes of order |n| above k R
    (``cut_on_ratio`` < 1) radiate weakly — J_n(z) falls off steeply once |n| > z — the plane wave n = 0 (V a divisor
    of m B) radiates along the axis. A single vane (V = 1) at R = 0 is the compact dipole omega F / (4 pi c r).

    Duct (optional). The ring above radiates into free space, where a mode below cut-off only loses the Bessel
    factor; inside a duct it also decays exponentially on its way out. With ``duct_radius_m`` (R_d) and
    ``duct_length_m`` (L, the length of duct between the stator and the nearer opening) every mode whose k R_d is below
    the hard-walled cut-off j'_{n,1} (first radial order) is multiplied by exp(-L sqrt(j'_{n,1}^2 - (k R_d)^2) / R_d);
    cut-on modes pass unchanged (no termination or liner model, no hub). A heuristic on top of the free-field ring, to
    rank vane counts, rpm and spacing; ``duct_attenuation_db`` (>= 0) reports it per mode. Default: no duct.

    The modes summed are s = s0 - ``s_range``..s0 + ``s_range`` around the lowest-order one, s0 = round(m B / V),
    widened where needed so that every mode left out has |n| > 1.5 k R + 12 (J_n below 1e-8 of the peak). The tone
    ``p_rms_Pa`` is the coherent sum at the observer's azimuth; ``p_rms_ring_Pa`` the rms over a ring of observers at
    the same theta (the modes are orthogonal in azimuth, so their energies add). Returns per harmonic ``{"harmonic",
    "frequency_hz", "p_rms_Pa", "spl_db", "p_rms_ring_Pa", "spl_ring_db", "dominant_n" (the mode loudest at the
    observer), "vane_force_N" (L_m amplitude), "reduced_frequency", "sears", "modes": [{"n", "s", "p_rms_Pa",
    "cut_on_ratio", "duct_attenuation_db"}] sorted by |n|}``."""
    B, V = _count(blades, "blades"), _count(vanes, "vanes")
    if rpm <= 0 or distance <= 0:
        raise ValueError("rpm and distance must be > 0")
    if vane_radius_m < 0 or vane_chord_m <= 0 or vane_span_m <= 0 or vane_inflow_m_s <= 0:
        raise ValueError("vane_radius_m >= 0; vane_chord_m, vane_span_m and vane_inflow_m_s > 0")
    if harmonics < 1 or s_range < 0:
        raise ValueError("harmonics >= 1 and s_range >= 0")
    if (duct_radius_m is not None and duct_radius_m <= 0) or duct_length_m < 0:
        raise ValueError("duct_radius_m > 0 (or None) and duct_length_m >= 0")
    gust = np.abs(np.asarray(gust_m_s, dtype=float).ravel())
    if len(gust) < harmonics + 1:
        raise ValueError(f"gust_m_s needs the harmonics 1..{harmonics} (index 0 = mean), got {len(gust)} values")
    omega, c, rho = rpm * 2 * math.pi / 60, medium.speed_of_sound, medium.density
    R, U, chord = float(vane_radius_m), float(vane_inflow_m_s), float(vane_chord_m)
    th, ph, xi = math.radians(angle_deg), math.radians(azimuth_deg), math.radians(stagger_deg)
    cos_t, sin_t = (0.0 if abs(v) < 1e-12 else v for v in (math.cos(th), math.sin(th)))    # exact zeros on and across the axis
    sin_x, cos_x = (0.0 if abs(v) < 1e-12 else v for v in (math.sin(xi), math.cos(xi)))
    p_ref = P_REF_AIR if medium.name == "air" else P_REF_WATER
    vane_mach = omega * R / c                                               # the cut-on ratios are taken at the vane radius
    rows = []
    for m in range(1, harmonics + 1):
        mB = m * B
        w_m = mB * omega
        k = w_m / c
        kred = w_m * chord / (2 * U)
        S = sears(kred)
        L = math.pi * rho * chord * U * float(gust[m]) * S * vane_span_m     # unsteady lift amplitude per vane
        Fa, Ft = L * sin_x, L * cos_x
        z = k * R * sin_t
        s0 = round(mB / V)
        s_lo, s_hi = s0 - s_range, s0 + s_range
        n_max = 1.5 * k * R + 12.0                                          # beyond this J_n(z <= kR) is negligible (< 1e-8)
        while mB - (s_lo - 1) * V <= n_max:                                 # modes left out below s_lo have n > n_max ...
            s_lo -= 1
        while (s_hi + 1) * V - mB <= n_max:                                 # ... and above s_hi n < -n_max
            s_hi += 1
        pre = w_m * V / (4 * math.pi * c * distance)
        total, energy, modes = 0j, 0.0, []
        for s in range(s_lo, s_hi + 1):
            n = mB - s * V
            a = Fa * cos_t * _jn(n, z) - Ft * sin_t * 0.5 * (_jn(n - 1, z) + _jn(n + 1, z))   # [F_a cos J_n - F_t n J_n / kR]
            att = 1.0
            if duct_radius_m is not None and duct_length_m > 0:
                kappa2 = _duct_cut_off(n) ** 2 - (k * duct_radius_m) ** 2       # > 0: the mode decays along the duct
                if kappa2 > 0:
                    att = math.exp(-duct_length_m * math.sqrt(kappa2) / duct_radius_m)
            a *= att
            total += np.exp(1j * n * (math.pi / 2 - ph)) * a                # i^n e^{-i n phi_o}
            energy += a * a
            modes.append({"n": n, "s": s, "p_rms_Pa": pre * abs(a) / math.sqrt(2),
                          "cut_on_ratio": cut_on_ratio(m, B, n, vane_mach),
                          "duct_attenuation_db": -20 * math.log10(max(att, 1e-300))})
        modes.sort(key=lambda d: (abs(d["n"]), d["s"]))
        p = pre * abs(total) / math.sqrt(2)
        p_ring = pre * math.sqrt(energy) / math.sqrt(2)
        dominant = max(modes, key=lambda d: d["p_rms_Pa"])
        rows.append({"harmonic": m, "frequency_hz": mB * rpm / 60, "p_rms_Pa": float(p),
                     "spl_db": 20 * math.log10(max(p, 1e-30) / p_ref), "p_rms_ring_Pa": float(p_ring),
                     "spl_ring_db": 20 * math.log10(max(p_ring, 1e-30) / p_ref), "dominant_n": dominant["n"],
                     "vane_force_N": L, "reduced_frequency": kred, "sears": S, "modes": modes})
    return rows
