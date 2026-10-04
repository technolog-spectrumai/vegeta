"""Life and fatigue of a structure from its modes and unit load cases: margins to the excitations, load spectra of the
missions, damage per mission, the static re-check at the hotspot, life under a usage mix. Lifted as they were from
notebook 08 Part 3 (cells 88, 95, 97, 98, 101, 103, 104) and 09b (cells 12, 18, 20, 21, 22, 24, 25), where both
notebooks had the same cells; the plots stay in the notebooks.

The FEA behind it (point masses, modal solve, unit cases on one mesh) is ``workflows._common.unit_fea``; the missions
and unit cases of each machine are in ``quad_life`` and ``fixed_wing_life``.
"""
from __future__ import annotations

from vegeta import chronos, talos

DAMPING = 0.03                                                              # 08 c88, 09b c12


def structure(freqs, source: str = "", damping: float = DAMPING) -> chronos.Structure:
    return chronos.Structure(tuple(freqs), damping_ratio=damping, source=source)


def margins(st: chronos.Structure, lines: dict, amplification_key: str = "amplification_1P") -> dict:
    """Each excitation line [Hz] against the nearest mode (08 c88; 09b c12 names the last column ``amplification``)."""
    return {k: {"1P_hz": f, "nearest_mode_hz": st.nearest_mode(f), "margin": st.margin(f),
                amplification_key: float(st.amplification(f)[0])} for k, f in lines.items()}


def spectra(missions: dict, st: chronos.Structure | None) -> dict:
    """08 c95 / 09b c18 (without saving and plotting)."""
    return {k: chronos.build_spectrum(m, st) for k, m in missions.items()}


def fatigue(unit_cases: dict, spectra_: dict, missions: dict, curve: talos.FatigueCurve, workdir=None, progress=False):
    """08 c97 / 09b c20: ``talos.assess_fatigue`` for each mission; returns ``(fatigue, life)`` (FatigueResult and the
    life table's rows by mission)."""
    from tqdm.auto import tqdm
    out = {}
    for k in tqdm(missions, desc="fatigue", disable=not progress):
        out[k] = talos.assess_fatigue(unit_cases, spectra_[k].to_dict(), curve,
                                      workdir=None if workdir is None else workdir / f"fatigue_{k}")
    life = {k: {"duration_min": missions[k].duration_h * 60, "damage_per_mission": f.result.metrics["damage_per_pass"],
                "missions_to_failure": f.result.metrics["passes_to_failure"],
                "hours_to_failure": f.result.metrics["hours_to_failure"],
                "hotspot": tuple(round(x, 1) for x in f.result.metrics["hotspot_location"])} for k, f in out.items()}
    return out, life


def worst(life: dict) -> str:
    """08 c98 / 09b c21: the mission with the most damage."""
    return max(life, key=lambda k: float(life[k]["damage_per_mission"]))


def static_recheck(fat: talos.FatigueResult, unit_cases: dict, spectra_: dict, yield_strength: float) -> dict:
    """08 c101 / 09b c22: the peak level of each pattern in each mission, combined at the worst mission's hotspot."""
    hot = fat.hotspot
    unit_vm = {k: float(talos.read_frd(r.artifacts["frd"]).von_mises[hot]) / load for k, (r, load) in unit_cases.items()}
    rows = {}
    for k, sp in spectra_.items():
        peak = {}
        for b in sp.blocks:
            peak[b.pattern] = max(peak.get(b.pattern, 0.0), abs(b.mean) + abs(b.amplitude))
        stress = sum(peak.get(pat, 0.0) * unit_vm[pat] for pat in unit_vm)
        rows[k] = {**{f"peak_{pat}": peak.get(pat, 0.0) for pat in unit_vm}, "hotspot_stress_MPa": stress,
                   "SF_yield": yield_strength / stress}
    return rows


def damage_rate(damage: dict, hours: dict, usage: dict) -> float:
    """09b c24: damage per flight hour under a usage mix."""
    return sum(usage[k] * damage[k] for k in usage) / sum(usage[k] * hours[k] for k in usage)


def usage_life(damage: dict, hours: dict, usage: dict, n_flights: int, seed=0) -> chronos.LifeSimulation:
    """08 c103 / 09b c24: flights and hours to failure under a usage mix."""
    return chronos.simulate_life(damage, hours, usage, n_flights=n_flights, seed=seed)


def balanced(sp: chronos.LoadSpectrum, factor: float = 2.5, pattern: str = "unbalance") -> chronos.LoadSpectrum:
    """08 c104: the spectrum with a better-balanced propeller (the unbalance amplitudes divided by ``factor``)."""
    return chronos.LoadSpectrum(sp.mission, sp.duration_s,
                                [chronos.Block(b.pattern, b.mean, b.amplitude / factor if b.pattern == pattern else b.amplitude,
                                               b.cycles, b.source) for b in sp.blocks],
                                sp.patterns)


def life_hours(damage: dict, hours: dict, mix: dict, n_flights: int = 200000) -> float:
    """08 c104: hours to failure under a usage mix (round robin, no random draws)."""
    return chronos.simulate_life(damage, hours, mix, n_flights=n_flights, seed=None).hours_to_failure


def nacelle_amplitude_mm(st: chronos.Structure | None, vib_result: talos.Result, vib_load: float, rpm: float,
                         unbalance_N: float) -> float:
    """09b c25: the motor's vibration amplitude, the static compliance of the vibration unit case times the dynamic
    amplification at the shaft frequency (1 without a structure)."""
    compliance = vib_result.metrics["max_displacement"] / vib_load              # mm per N, static
    f = rpm / 60
    daf = float(st.amplification(f)[0]) if st is not None else 1.0
    return daf * unbalance_N * compliance
