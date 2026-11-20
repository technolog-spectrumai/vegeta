"""The sound of the recoilless 8 mm emitter (``bearing_recoilless``): one shot, or a burst of balls one after another.

The shot is the same 0-D internal ballistics as ``golf_recoilless.shot`` (plenum -> hammer valve -> chamber -> ball
in the 1.0 m tube; the chamber's rear -> throat -> 2:1 cone -> the back jet; after the ball leaves, the tube blows
down through the muzzle), run again here with its time histories recorded. The sound is built from them:

- **muzzle blast and back-blast**: each jet is an acoustic monopole, ``p(t) = (dm/dt)' / (4 pi r)`` — the rate of
  change of its mass flow (``mdot``) seen at the listener after the travel time. The muzzle's flow starts as a step
  when the ball leaves the tube (the sharp crack-like report of an air gun); the back jet rises with the chamber's
  pressure and dies with the blowdown;
- **jet roar**: the back jet's turbulent mixing noise (Lighthill's U^8 law, Lush's constant), band-limited around a
  Strouhal number 0.2 of its speed and exit diameter, shaped in time by its mass flow; the muzzle blowdown likewise;
- **the valve**: the hammer striking the 1/2" valve, a short decaying metallic ring at its opening (an assumed
  4.5 kHz, 3 ms decay, 0.3 Pa at 1 m) — the click before the bang;
- **the ground**: one reflection from flat ground (the emitter ``--height`` above it, the listener at ear height).

The ball itself leaves at Mach 0.82: no ballistic crack. A burst (``--shots`` N, ``--interval`` s apart) draws each
ball's air from the plenum, which the regulator refills towards its set pressure with a time constant ``--refill``
(an assumption: a hobby HPA regulator's flow) — so balls fired close together leave slower and sound weaker; each
shot's pressure, speed and level go into the table.

    python bearing_sound.py --shots 1                 # one shot -> bearing_1shot_10m.wav
    python bearing_sound.py --shots 5 --interval 0.12  # five balls in 0.48 s -> bearing_5shots_10m.wav
"""
from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np

import bearing_recoilless as br
import golf_gun as gg
import golf_recoilless as g

C_AIR, RHO_A, RATE = 343.0, 1.2, 96000          # the shot is fast: 96 kHz keeps the muzzle step sharp


def shot_history(p_gauge: float, throat_mm: float, dt: float = 2e-7, t_after: float = 0.03) -> dict:
    """``golf_recoilless.shot`` for the 8 mm emitter at ``p_gauge`` (bar g), recording the muzzle and back-jet mass flows,
    the jets' exit speeds and the valve's opening over time. Same equations, same constants."""
    case, proj, eps, friction = br.CASE, br.PROJ, br.EPS, br.FRICTION
    GAM, R, CP, PA, TA = g.GAM, g.R, g.CP, g.PA, g.TA
    v = case.valve
    p0i = PA + p_gauge * 1e5
    V0 = case.tank.volume_l * 1e-3
    m_t = m_t0 = p0i * V0 / (R * TA)
    a_valve = v.cd * v.area
    a_t = math.pi / 4 * (throat_mm * 1e-3) ** 2
    x0 = 0.02
    A_B = math.pi / 4 * proj["bore_m"] ** 2
    M_BALL = proj["mass"]
    travel = proj["barrel_m"] - x0 - proj["d_ball"]
    Vc = A_B * x0
    m_c = PA * Vc / (R * TA)
    E_c = PA * Vc / (GAM - 1)
    x = u = t = 0.0
    out_t, v_out = None, 0.0
    a_snd = math.sqrt(GAM * R * TA)
    rec = {"t": [], "mdot_muzzle": [], "mdot_back": [], "u_back": [], "u_muzzle": [], "p_c": [], "T_c": [], "u_ball": [],
           "p_front": []}
    while True:
        rho_t = m_t / V0
        p_t = p0i * (rho_t / (p0i / (R * TA))) ** GAM
        T_t = p_t / (rho_t * R)
        Vc_now = Vc + A_B * min(x, travel)
        p_c = (GAM - 1) * E_c / Vc_now
        T_c = p_c * Vc_now / (m_c * R)
        f = min(1.0, t / v.open_s) if out_t is None else max(0.0, 1.0 - (t - out_t) / v.open_s)
        mdot_in = g.mdot_orifice(a_valve * f, p_t, T_t, p_c)
        mdot_n = g.mdot_orifice(a_t, p_c, T_c, PA)
        if out_t is None:
            M = u / a_snd
            pf = PA * (1 + GAM * (GAM + 1) / 4 * M * M + GAM * M * math.sqrt(1 + ((GAM + 1) / 4) ** 2 * M * M))
            F = A_B * (p_c - pf)
            Fn = F - friction if (u > 0 or F > friction) else 0.0
            u = max(0.0, u + Fn / M_BALL * dt)
            x += u * dt
            mdot_f, p_front = 0.0, pf
            if x >= travel:
                out_t, v_out = t, u
        else:
            mdot_f, p_front = g.mdot_orifice(A_B, p_c, T_c, PA), PA
            if (t - out_t > t_after) or (p_c < PA * 1.02 and f == 0):
                break
        # the jets' exit speeds: sonic at a choked exit (the back cone's supersonic exit for the nozzle), else Bernoulli
        a_c = math.sqrt(GAM * R * T_c)
        u_b = (min(a_c * 1.6, math.sqrt(max(2 * CP * T_c * (1 - (PA / p_c) ** ((GAM - 1) / GAM)), 0.0))) if p_c > PA else 0.0)
        u_m = (min(a_c, math.sqrt(max(2 * CP * T_c * (1 - (PA / p_c) ** ((GAM - 1) / GAM)), 0.0))) if p_c > PA else 0.0)
        rec["t"].append(t); rec["mdot_muzzle"].append(mdot_f); rec["mdot_back"].append(mdot_n)
        rec["u_back"].append(u_b); rec["u_muzzle"].append(u_m)
        rec["p_c"].append(p_c); rec["T_c"].append(T_c); rec["p_front"].append(p_front)
        rec["u_ball"].append(u if out_t is None else 0.0)
        m_t -= mdot_in * dt
        m_c += (mdot_in - mdot_n - mdot_f) * dt
        E_c += (mdot_in * CP * T_t - (mdot_n + mdot_f) * CP * T_c - (p_c * A_B * u if out_t is None else 0.0)) * dt
        m_c, E_c = max(m_c, 1e-9), max(E_c, 1e-9)
        t += dt
        if t > 0.5:
            break
    out = {k: np.asarray(val) for k, val in rec.items()}
    out.update(v=v_out, barrel_s=out_t or t, air_kg=m_t0 - m_t, p_gauge=p_gauge)
    return out


def _resample(t, y, t_new):
    return np.interp(t_new, t, y, left=0.0, right=0.0)


def _band_noise(n, lo, hi, rng):
    F = np.fft.rfft(rng.standard_normal(n))
    fr = np.fft.rfftfreq(n, 1 / RATE)
    F[(fr < lo) | (fr > hi)] = 0
    x = np.fft.irfft(F, n)
    return x / max(float(np.std(x)), 1e-30)


def shot_sound(h: dict, *, distance: float, height: float, ear: float, behind_deg: float, throat_mm: float, seed: int,
               seconds: float) -> tuple[np.ndarray, dict]:
    """The pressure signal [Pa] at the listener for one recorded shot (time 0 = the valve starts to open)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    L_tube = br.BARREL_M
    # positions along the gun's axis (x forward, breech at 0): muzzle at +L, back-jet exit at -0.08 m; listener on the
    # ground plane's side, ``behind_deg`` from the muzzle direction
    th = math.radians(behind_deg)
    lis = np.array([distance * math.cos(th), distance * math.sin(th), ear])
    src = {"muzzle": np.array([L_tube, 0.0, height]), "back": np.array([-0.08, 0.0, height]), "valve": np.array([0.0, 0.0, height])}

    def paths(p):
        direct = float(np.linalg.norm(lis - p))
        mirror = p * np.array([1, 1, -1])
        refl = float(np.linalg.norm(lis - mirror))
        return [(direct, 1.0), (refl, 0.8)]                       # hard ground: 80 % of the pressure reflected

    sig = np.zeros(n)
    th_rec = h["t"]
    t_fine = np.arange(0, th_rec[-1], 1 / RATE)
    stats = {}
    for name, mkey, ukey, d_exit in (("muzzle", "mdot_muzzle", "u_muzzle", br.PROJ["bore_m"]),
                                     ("back", "mdot_back", "u_back", throat_mm * 1e-3 * math.sqrt(br.EPS))):
        mdot = _resample(th_rec, h[mkey], t_fine)
        U = _resample(th_rec, h[ukey], t_fine)
        mono = np.gradient(mdot, 1 / RATE) / (4 * math.pi)        # times 1/r per path
        # turbulent mixing noise: Lighthill / Lush, I = K rho U^8 D^2 / (c^5 r^2), K = 3e-5; its rms at 1 m
        prms_1m = np.sqrt(np.maximum(3e-5 * RHO_A * U ** 8 * d_exit ** 2 / C_AIR ** 5, 0.0) * RHO_A * C_AIR)
        U_peak = float(U.max()) if len(U) else 0.0
        f_peak = 0.2 * max(U_peak, 1.0) / d_exit
        roar = _band_noise(len(t_fine), 0.25 * f_peak, min(4 * f_peak, 0.45 * RATE), rng) * prms_1m
        for r, a in paths(src[name]):
            k = int(round(r / C_AIR * RATE))
            seg = (mono + roar) * a / r
            end = min(n, k + len(seg))
            if k < n:
                sig[k:end] += seg[: end - k]
        stats[name] = {"peak_mass_flow_g_s": float(mdot.max() * 1e3), "peak_jet_speed_m_s": U_peak, "roar_peak_hz": f_peak}
    # the valve's hammer click: 4.5 kHz, 3 ms decay, 0.3 Pa at 1 m (assumed)
    tc = np.arange(int(0.02 * RATE)) / RATE
    click = 0.3 * np.sin(2 * math.pi * 4500 * tc) * np.exp(-tc / 0.003)
    for r, a in paths(src["valve"]):
        k = int(round(r / C_AIR * RATE))
        if k < n:
            sig[k:k + len(click)] += (click * a / r)[: n - k]
    return sig, stats


def burst(n_shots: int, interval: float, refill_s: float, design: dict, t_after: float = 0.03) -> list[dict]:
    """Plenum pressure before each shot: each shot takes its air out of the 0.5 L plenum, the regulator refills it
    towards the set pressure with a first-order lag ``refill_s``."""
    R, TA = g.R, g.TA
    V0 = br.CASE.tank.volume_l * 1e-3
    p_set = design["p_bar_g"]
    p = p_set
    shots = []
    for i in range(n_shots):
        h = shot_history(p, design["throat_mm"], t_after=t_after)
        shots.append(h)
        m = (g.PA + p * 1e5) * V0 / (R * TA) - h["air_kg"]          # air left after this shot
        p_after = m * R * TA / V0 / 1e5 - g.PA / 1e5
        p = p_set + (p_after - p_set) * math.exp(-interval / refill_s) if refill_s > 0 else p_set
    return shots


def write_wav(path, x, full):
    pcm = np.clip(x / full, -1, 1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(RATE)
        f.writeframes((pcm * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--interval", type=float, default=0.12, help="time between shots [s]")
    ap.add_argument("--refill", type=float, default=0.25, help="regulator refill time constant of the plenum [s] (assumed)")
    ap.add_argument("--distance", type=float, default=10.0, help="listener distance from the breech [m]")
    ap.add_argument("--angle", type=float, default=90.0, help="listener angle from the muzzle direction [deg] (0 ahead, 180 behind)")
    ap.add_argument("--height", type=float, default=1.0, help="emitter height above the ground [m]")
    ap.add_argument("--ear", type=float, default=1.6, help="listener ear height [m]")
    ap.add_argument("--out", type=Path, default=Path("bearing_sounds"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("sizing the emitter (bearing_recoilless.design) ...")
    d = br.design(verbose=False)
    print(f"  {d['p_bar_g']:.1f} bar g, throat {d['throat_mm']:.2f} mm, {d['v']:.0f} m/s, back jet {d['jet_speed']:.0f} m/s")
    shots = burst(args.shots, args.interval, args.refill, d)
    seconds = args.interval * (args.shots - 1) + 1.0
    total = np.zeros(int(seconds * RATE))
    rows = []
    for i, h in enumerate(shots):
        s, st = shot_sound(h, distance=args.distance, height=args.height, ear=args.ear, behind_deg=args.angle,
                           throat_mm=d["throat_mm"], seed=i, seconds=seconds - i * args.interval)
        k = int(round(i * args.interval * RATE))
        total[k:k + len(s)] += s[: len(total) - k]
        peak = float(np.abs(s).max())
        rows.append({"shot": i + 1, "time_s": round(i * args.interval, 3), "plenum_bar_g": round(h["p_gauge"], 2),
                     "ball_speed_m_s": round(h["v"], 1), "energy_J": round(0.5 * br.PROJ["mass"] * h["v"] ** 2, 1),
                     "air_g": round(h["air_kg"] * 1e3, 2), "peak_Pa": round(peak, 2),
                     "peak_dB": round(20 * math.log10(max(peak, 1e-9) / 20e-6), 1), **{f"{k}_{kk}": round(vv, 1) for k, v in st.items() for kk, vv in v.items()}})
        print(f"  shot {i + 1}: plenum {h['p_gauge']:.1f} bar g, ball {h['v']:.0f} m/s, peak {rows[-1]['peak_dB']} dB at {args.distance:g} m")
    tag = f"{args.shots}shot{'s' if args.shots > 1 else ''}_{args.distance:g}m"
    full = 1.1 * float(np.abs(total).max())
    path = args.out / f"bearing_{tag}.wav"
    write_wav(path, total, full)
    (args.out / f"bearing_{tag}.json").write_text(json.dumps({
        "design": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()},
        "listener": {"distance_m": args.distance, "angle_deg": args.angle, "emitter_height_m": args.height, "ear_m": args.ear},
        "burst": {"shots": args.shots, "interval_s": args.interval, "refill_s": args.refill},
        "full_scale_Pa": full, "peak_dB_overall": round(20 * math.log10(full / 1.1 / 20e-6), 1), "shots": rows}, indent=1))
    print("wrote", path, f"(full scale {full:.1f} Pa = {20 * math.log10(full / 20e-6):.0f} dB peak)")


if __name__ == "__main__":
    main()
