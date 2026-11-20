"""BUTEO (the lift + cruise VTOL, ``designs/buteo.py``): its three simulation phases. Each is restartable and writes
ONE timestamped file ``output/runs/buteo_<phase>_<time>_<host>.yaml`` (paste it back), with charts next to it.

    labs/aero/sims/buteo_cfd.sh -j 12          # 1. CFD (OpenFOAM): the wing-borne polar with booms, motors and feet
    labs/aero/sims/buteo_stability.sh -j 12    # 2. stability: wing-borne modes and tests at 22 m/s; the hover in wind
    labs/aero/sims/buteo_missions.sh -j 12     # 3. missions: the unattended cycle (take-off, transition, pad landing)
                                               #    over wind, the wing-borne card and chases, the BUTEO KPIs, per pack

1. **cfd** — an angle-of-attack sweep at 22 m/s, rotors stopped, pusher off. With six angles solved it is installed as
   ``designs/data/buteo_cfd_screening.yaml`` for phases 2 and 3 (``--no-install`` keeps the lattice model).
2. **stability** — wing-borne: trim, modes, doublets, gust, stall and recovery, with the model's post-stall moment
   and a weak one (0.35). Hover: 20 s over a point in 0, 4, 8, 12 and 16 m/s wind with gusts.
3. **missions** — per pack: the take-off and transition from the pad, the pad landing, both by Monte Carlo over wind
   (the pad's heading to the wind random; the approach into the wind); the operating wind; the cycle's energy, cycles
   per charge and recharge time; the wing-borne performance card and chases; the BUTEO KPIs and its gate (off the
   pad and back onto it).

``-j`` for phases 2 and 3 is capped to what the machine can carry; ``--force-cpu`` insists.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "designs"))
sys.path.append(str(HERE))

import astor_run as R                                                # noqa: E402  (the shared runner machinery)

DATA = HERE.parent / "designs" / "data" / "buteo_cfd_screening.yaml"
CHARGE_C = 1.0                    # HAND: the pad charges at 1C (Li-ion, long life)
ALERT_S = 2.0
HOVER_WINDS = (0.0, 4.0, 8.0, 12.0, 16.0)


def _meta(phase, a):
    import buteo_model as M
    C = M.get()
    m = R._meta(phase, a)
    m.update({"aircraft": "BUTEO", "aero_source": C["aero_source"], "mass_kg": round(C["mass"], 3), "pack": C["mass_budget"]["pack"]["name"]})
    return m


def _job(job):
    import numpy as np
    kind = job[0]
    t = time.time()
    try:
        import buteo_model as M
        import buteo_mujoco as S
        import arigo1_mujoco as amj
        if kind == "stab":
            old = amj.Sim
            amj.Sim = S.SimButeo
            try:
                out = amj.stability_tests("buteo", "pusher", job[1], job[2])
            finally:
                amj.Sim = old
            out["aircraft"] = "BUTEO"
        elif kind == "hover":
            out = S.hover_hold(S.SimButeo(), job[1], np.random.default_rng(int(job[1]) + 1))
        elif kind == "card":
            import arigo_perf as P
            import arigo_mission as AM
            pack, n = job[1], job[2]
            C = M.get(pack)
            sf = lambda bandit=False: S.SimButeo(bandit=bandit, C=C)
            calm_t, calm_l = S.takeoff_transition(sf()), S.approach_land(sf())
            C["equipment"] = {**C["equipment"], "pad_time_s": calm_t["t_safe_s"] or 30.0}
            out = P.card("BUTEO", C, sf, lambda name, sim, record=None: amj.manoeuvre(name, sim, record), M.CRUISE)
            out["aircraft"] = f"BUTEO, {pack}"
            rb = S.robustness(sf, n=n)
            hov = S.hover_hold(sf(), 12.0, np.random.default_rng(12))
            pm = out["point_mass"]
            E_use = C["mass_budget"]["pack"]["Wh_nominal"] * 0.85
            E_wing = AM.mission_energy(C, pm, 1500.0) / 3600.0
            E_cycle = (calm_t["energy_Wh"] or 0) + (calm_l.get("energy_Wh") or 0) + E_wing
            w8 = next((p for p in rb["per_wind"] if p["wind_m_s"] == 8.0), rb["per_wind"][0])
            env = [p for p in rb["per_wind"] if p["wind_m_s"] <= 12.0]
            m = C["mass"]
            out["vtol"] = {"takeoff_calm": calm_t, "landing_calm": calm_l, "robustness": rb, "hover_12": hov,
                           "cycle": {"takeoff_Wh": calm_t["energy_Wh"], "landing_Wh": calm_l.get("energy_Wh"),
                                     "wing_borne_Wh": round(E_wing, 1), "total_Wh": round(E_cycle, 1), "usable_Wh": round(E_use, 1),
                                     "rule": "take-off + a target 1.5 km out with 60 s at full power and the flight home + the pad landing"}}
            out["summary"].update({
                "launch": "pad", "cue_to_safe_s": round(ALERT_S + (calm_t["t_safe_s"] or 30.0), 1), "standby_h": 24.0,
                "pad_landing_calm_ok": bool(calm_l["ok"]), "pad_miss_p90_m": w8["miss_p90_m"], "pad_sink_m_s": calm_l.get("sink_m_s"),
                "launch_success_pct": round(100 * float(np.mean([p["launch"]["rate"] for p in env])), 1),
                "recovery_success_pct": round(100 * float(np.mean([p["recovery"]["rate"] for p in env])), 1),
                "operating_wind_m_s": rb["operating_wind_m_s"], "hover_rms_12_m": hov.get("rms_error_m"),
                "cycles_per_charge": round(E_use / E_cycle, 1), "turnaround_s": round(E_cycle / (C["mass_budget"]["pack"]["Wh_nominal"] * CHARGE_C) * 3600.0),
                "landing_miss_m": calm_l.get("miss_m"), "touchdown_energy_J": round(0.5 * m * (calm_l.get("sink_m_s") or 0) ** 2, 1),
                "hover_endurance_min": C["lift"]["hover_endurance_min"], "hover_power_W": C["lift"]["hover_power_W"]})
        else:
            out = {"error": f"unknown job {kind}"}
    except Exception as exc:                                          # noqa: BLE001
        import traceback
        out = {"error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-900:]}
    out["seconds"] = round(time.time() - t, 1)
    return "|".join(str(x) for x in job), out


def phase_stability(a) -> int:
    import buteo_model as M
    jobs = [("stab", M.CRUISE, k) for k in (1.0, 0.35)] + [("hover", W) for W in HOVER_WINDS]
    res = R._run_jobs(jobs, a, "buteo_stability_cache.yaml", "buteo stability", _job)
    runs = {f"cm{float(k.split('|')[2]):g}": v for k, v in res.items() if k.startswith("stab")}
    hover = {f"W{float(k.split('|')[1]):.0f}": v for k, v in res.items() if k.startswith("hover")}
    run = R._write("stability", {"run": _meta("stability", a), "results": {"buteo_mujoco": {"runs": runs, "hover": hover}}}, "buteo")
    for k, r in runs.items():
        print(k, r.get("verdict", r.get("error")))
    for k, r in hover.items():
        print("hover", k, {x: r.get(x) for x in ("ok", "rms_error_m", "max_tilt_deg", "saturated_share", "mean_power_W", "why")})
    print("->", run)
    return 0


def phase_missions(a) -> int:
    import arigo_charts as AC
    import buteo as B
    packs = a.packs or list(B.PACKS)
    res = R._run_jobs([("card", pk, a.samples) for pk in packs], a, "buteo_missions_cache.yaml", "buteo missions", _job)
    cards = {}
    for pk in packs:
        c = res[f"card|{pk}|{a.samples}"]
        if "error" in c:
            print("card", pk, "ERROR", c["error"], c.get("trace", "")[-400:])
            continue
        cards[f"buteo_{pk.replace(' ', '_')}"] = c
    MK = AC.mission_kpi(cards)
    items = AC.MISSIONS["buteo"][1]
    table = {}
    for k, c in cards.items():
        s = c["summary"]
        table[k] = {"name": c["aircraft"], "score": MK[k]["buteo"], "gates": {g: bool(f(s.get(g))) for g, f in AC.GATES["buteo"]},
                    "kpis": [{"kpi": lab, "value": s.get(key), "threshold": thr, "objective": obj, "score": MK[k]["per_metric"].get(key)}
                             for key, lab, thr, obj in items if key in s]}
    doc = {"run": {**_meta("missions", a), "packs": packs, "samples": a.samples}, "kpi": table,
           "results": {"buteo_missions": {"cards": cards}}}
    run = R._write("missions", doc, "buteo")
    for p in charts(run, doc):
        print("->", p)
    print("->", run)
    for k, t in table.items():
        s = cards[k]["summary"]
        print(f"  {t['name']:26s} score {t['score']:5.1f}  gates {t['gates']}  {s['mass_kg']:.2f} kg  cue-to-flight {s['cue_to_safe_s']} s  "
              f"operating wind {s['operating_wind_m_s']} m/s  pad miss p90 {s['pad_miss_p90_m']} m  cycles/charge {s['cycles_per_charge']}")
    return 0


def charts(run: Path, doc) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    T = doc["kpi"]
    if not T:
        return []
    out = []
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    labs = [x["kpi"] for x in next(iter(T.values()))["kpis"]]
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(labs) + 1.6), facecolor="#fcfcfb")
    y, h = np.arange(len(labs)), 0.8 / len(T)
    for i, (k, t) in enumerate(T.items()):
        sc = {x["kpi"]: x["score"] or 0 for x in t["kpis"]}
        ax.barh(y - 0.4 + h * (i + 0.5), [sc.get(l, 0) for l in labs], height=h, color=cols[i % 3], label=f"{t['name']} ({t['score']:.0f})")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7); ax.invert_yaxis(); ax.set_xlim(0, 105)
    ax.set_xlabel("score (0 = threshold, 100 = objective)", fontsize=8); ax.grid(axis="x", color="#e4e3df"); ax.set_axisbelow(True)
    ax.set_title("BUTEO: KPI scores per pack", fontsize=11, loc="left"); ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    out.append(run.with_name(run.stem + "_kpi.png")); fig.savefig(out[-1], dpi=110); plt.close(fig)
    c = next(iter(doc["results"]["buteo_missions"]["cards"].values()))
    per = c["vtol"]["robustness"]["per_wind"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), facecolor="#fcfcfb")
    w = [p["wind_m_s"] for p in per]
    axs[0].plot(w, [100 * p["launch"]["rate"] for p in per], color="#2a78d6", lw=2, marker="o", label="take-off and transition")
    axs[0].plot(w, [100 * p["recovery"]["rate"] for p in per], color="#e34948", lw=2, marker="s", label="pad landing")
    axs[0].axhline(95, color="#52514e", lw=1, ls=":"); axs[0].set_ylim(0, 105); axs[0].set_ylabel("success (%)", fontsize=8)
    axs[0].set_title("BUTEO: the cycle's two halves over wind (dotted: 95 %)", fontsize=10, loc="left")
    axs[1].plot(w, [p["miss_median_m"] for p in per], color="#2a78d6", lw=2, marker="o", label="median")
    axs[1].plot(w, [p["miss_p90_m"] for p in per], color="#eb6834", lw=2, marker="s", label="90th percentile")
    axs[1].axhline(0.22, color="#52514e", lw=1, ls=":"); axs[1].set_ylabel("miss from the pad's centre (m)", fontsize=8)
    axs[1].set_title("Pad landing accuracy (dotted: the pad's 0.22 m tolerance)", fontsize=10, loc="left")
    for ax in axs:
        ax.set_xlabel("steady wind (m/s), gusts 1.5x", fontsize=8); ax.grid(color="#e4e3df"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out.append(run.with_name(run.stem + "_cycle.png")); fig.savefig(out[-1], dpi=110); plt.close(fig)
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="buteo_<phase>.sh", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("phase", choices=("cfd", "stability", "missions"))
    ap.add_argument("-j", "--cpu", type=int, default=4, help="processes (cfd: OpenFOAM MPI; others: workers, capped to the machine)")
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="forget the phase's saved progress")
    ap.add_argument("--alphas", type=float, nargs="*", help="cfd: the angles of the sweep")
    ap.add_argument("--quality", default="screening")
    ap.add_argument("--full-cfd", action="store_true", help="cfd: the fine mesh; kept apart, not installed")
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--timeout", type=float, default=3 * 3600.0)
    ap.add_argument("--workdir", type=Path, default=HERE.parent / "_runs" / "buteo_cfd")
    ap.add_argument("--packs", nargs="*", help="missions: pack names (default: every option in buteo.PACKS)")
    ap.add_argument("--samples", type=int, default=12, help="missions: Monte Carlo take-offs and landings per wind")
    a = ap.parse_args(argv)
    if a.phase == "cfd":
        import buteo_cfd as cf
        a.alphas = a.alphas or list(cf.ALPHAS)
        return R.phase_cfd(a, cf, "buteo", DATA, _meta)
    return phase_stability(a) if a.phase == "stability" else phase_missions(a)


if __name__ == "__main__":
    sys.exit(main())
