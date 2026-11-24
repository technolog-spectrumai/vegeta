"""Everything for the TRANSONIC CARBON FAN in one go, one tqdm bar over the stages, one summary YAML.

STAGES
  1 design    mean-line sizing (static, 600 km/h) and thrust/power vs flight speed       -> fan_thrust_vs_speed.png
  2 cad       rotor + housing STEP/STL, one-blade comparison picture against the old EDF -> data in output/transonic/fan_cad/
  3 fea       spin + air load at the design rpm and 120 % overspeed (CalculiX)          -> fea_<time>_<host>.yaml
  4 vib       modes, Campbell diagram, resonance margins                                 -> vib_<time>_<host>.yaml
  5 cfd       rotor + housing in an MRF zone at 0.85/1.0/1.1 x rpm, static and 600 km/h -> cfd_<time>_<host>.yaml

USAGE (from the repository root, or scripts/transonic/transonic_fan.sh ...)
  scripts/transonic/transonic_fan.sh -j 12                       all stages
  scripts/transonic/transonic_fan.sh -j 12 --skip cfd            no OpenFOAM
  scripts/transonic/transonic_fan.sh -j 12 --compare             also the old 90 mm EDF in FEA/vib/CFD
  scripts/transonic/transonic_fan.sh -j 12 --quality smoke --element 1.5     a quick pass
  options of transonic_runs.py pass through: --quality, --compressible, --element, --modes, --southwell, --fresh

OUTPUT: output/transonic/runs/fan_<YYYYmmdd-HHMMSS>_<host>.yaml — the sizing, every stage's status and minutes, the key
numbers (max stress and safety factors, risky resonances, CFD thrust/power against the model) and the paths of the
stage files and pictures.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import transonic_runs as R  # noqa: E402

tr = R.tr
STAGES = ("design", "cad", "fea", "vib", "cfd")


def stage_design(out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V = np.linspace(0, 190, 39)
    rows = [tr.fan_meanline(tr.TFAN, float(v)) for v in V]
    drag = [tr.racer_thrust(max(float(v), 15.0)) for v in V]
    T = np.array([r["net_thrust_N"] for r in rows])
    top = next((float(v) * 3.6 for v, t, d in zip(V, T, drag) if v > 20 and t < d), None)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(V * 3.6, T, lw=2.2, color="#d03b3b", label="transonic fan, 90 000 rpm")
    ax[0].plot(V * 3.6, drag, ":", color="#555555", label="racer drag")
    ax[0].axvline(600, color="#999999", lw=1)
    ax[0].set(xlabel="flight speed [km/h]", ylabel="net thrust [N]"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].plot(V * 3.6, [r["shaft_kW"] for r in rows], lw=2.2, color="#d03b3b", label="shaft power [kW]")
    ax[1].plot(V * 3.6, [r["jet_m_s"] / 10 for r in rows], "--", color="#2a78d6", label="jet speed / 10 [m/s]")
    ax[1].set(xlabel="flight speed [km/h]"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.suptitle("Transonic carbon fan (mean line)", x=0.02, ha="left")
    fig.tight_layout()
    png = out / "fan_thrust_vs_speed.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    return dict(static=tr.fan_meanline(tr.TFAN, 0.0), at_600_kmh=tr.fan_meanline(tr.TFAN, tr.V_MAX), top_speed_kmh=top,
                curve=[dict(kmh=round(float(v) * 3.6), thrust_N=r["net_thrust_N"], shaft_kW=r["shaft_kW"], jet_m_s=r["jet_m_s"])
                       for v, r in zip(V, rows)], picture=str(png))


def stage_cad(out: Path) -> dict:
    d = out / "fan_cad"
    d.mkdir(parents=True, exist_ok=True)
    rot, hou = tr.edf_cad(tr.TFAN)
    files = {"rotor": rot.export(d / "rotor", formats=("step", "stl"), stl_tolerance=0.02).artifacts,
             "housing": hou.export(d / "housing", formats=("step", "stl"), stl_tolerance=0.05).artifacts}
    pic = None
    try:
        pic = blade_picture(d / "fan_blade_subsonic_vs_transonic.png")
    except Exception as e:                                          # noqa: BLE001 - no display: the CAD is still there
        print("  (blade picture skipped:", str(e)[:120], "- on a headless machine run under xvfb-run)")
    return dict(files={k: {f: str(p) for f, p in v.items()} for k, v in files.items()}, mass_g_rotor=rot.volume * tr.CFRP["rho"] * 1e6,
                picture=str(pic) if pic else None)


def blade_picture(path: Path) -> Path:
    import ducted_fan as df
    import pyvista as pv
    from vegeta.dedalus import viz
    pv.OFF_SCREEN = True

    def blade(d):
        rp, _ = tr._edf_p(d)
        kw = dict(blades=1, bore=0.0, spinner_length=0.0, **rp)
        for k in ("skew_deg", "thickness_root"):
            if d.get(k):
                kw[k] = d[k]
        hub = d["hub_diameter_mm"] / 2
        return viz.to_pyvista(df.EDFRotor().generate(**kw), 0.02).clip_box([-hub - 0.3, hub + 0.3, -hub - 0.3, hub + 0.3, -100, 100], invert=True)

    sub, tra = blade(tr.OLD_EDF), blade(tr.TFAN)
    pl = pv.Plotter(shape=(2, 2), window_size=(1700, 1300), off_screen=True)
    for i, (title, v) in enumerate((("isometric, true size", None), ("from the tip", (1, 0, 0)), ("along the axis", (0, 0, -1)),
                                    ("side, in the rotor plane", (0, 1, 0)))):
        pl.subplot(i // 2, i % 2)
        pl.add_mesh(sub, color="#2a78d6", smooth_shading=True, opacity=0.85, label="subsonic 90 mm (nb 25)")
        pl.add_mesh(tra, color="#d03b3b", smooth_shading=True, opacity=0.85, label="transonic 75 mm carbon")
        pl.add_text(title, font_size=11)
        pl.view_isometric() if v is None else pl.view_vector(v, viewup=(0, 1, 0) if v == (0, 0, -1) else (0, 0, 1))
        if i == 0:
            pl.add_legend(bcolor="white", size=(0.5, 0.12), loc="upper right")
        pl.reset_camera()
    pl.screenshot(str(path)); pl.close()
    return path


def _key_numbers(stage: str, path: Path) -> dict:
    d = yaml.safe_load(path.read_text())
    if stage == "fea":
        return {p: {c: {k: r[c][k] for k in ("max_von_mises_MPa", "safety_factor", "max_deflection_mm", "at_r_over_R")}
                    for c in ("at_design_rpm", "overspeed_120")} for p, r in d["parts"].items()}
    if stage == "vib":
        return {p: dict(modes_at_rest_hz=r["modes_at_rest_hz"], verdict=r["verdict"],
                        risky=[(c["mode"], c["order"], round(c["rpm"])) for c in r["risky_crossings"]]) for p, r in d["parts"].items()}
    return {k: dict(thrust_N=(c.get("cfd") or {}).get("thrust_N"), power_W=(c.get("cfd") or {}).get("power_W"),
                    model_thrust_N=(c.get("operating_point") or {}).get("thrust_N"), ratio=c.get("thrust_ratio_cfd_to_model"))
            for k, c in d["cases"].items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", "--cpu", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--skip", nargs="*", default=[], choices=STAGES)
    ap.add_argument("--only", nargs="*", default=None, choices=STAGES)
    ap.add_argument("--compare", action="store_true", help="also the old 90 mm EDF in FEA, vib and CFD")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--element", type=float, default=None)
    ap.add_argument("--modes", type=int, default=6)
    ap.add_argument("--southwell", type=float, nargs=2, default=(1.2, 0.2))
    ap.add_argument("--quality", default="screening", choices=tuple(R.QUALITY))
    ap.add_argument("--compressible", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args(argv)
    args.parts = ["transonic_fan"] + (["subsonic_edf_nb25"] if args.compare else [])
    stages = [s for s in (args.only or STAGES) if s not in args.skip]
    out = R.OUT / "runs"
    out.mkdir(parents=True, exist_ok=True)
    t_, host = R.stamp()
    doc = dict(kind="transonic_fan", run=dict(started=t_, host=host, cpu=args.cpu, stages=stages, compare=args.compare, argv=sys.argv[1:]),
               design=dict(fan=tr.TFAN, material=tr.CFRP, racer=tr.RACER), stages={})
    summary = out / f"fan_{t_}_{host}.yaml"
    try:
        from tqdm import tqdm
        bar = tqdm(total=len(stages), unit="stage", dynamic_ncols=True, desc="transonic fan")
        say = tqdm.write
    except ImportError:
        bar, say = None, print
    for s in stages:
        if bar:
            bar.set_postfix_str(s)
        t0 = time.time()
        rec = dict(status="ok")
        try:
            if s == "design":
                rec["result"] = stage_design(out)
            elif s == "cad":
                rec["result"] = stage_cad(R.OUT)
            else:
                path = {"fea": R.run_fea, "vib": R.run_vib, "cfd": R.run_cfd}[s](args)
                rec["file"] = str(path)
                rec["key_numbers"] = _key_numbers(s, path)
        except SystemExit as e:
            rec = dict(status="skipped", reason=str(e))
        except Exception as e:                                     # noqa: BLE001 - one stage failing keeps the others
            rec = dict(status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
        rec["minutes"] = round((time.time() - t0) / 60, 2)
        doc["stages"][s] = rec
        summary.write_text(yaml.safe_dump(R._plain(doc), sort_keys=False, width=140))      # after every stage
        say(f"[{s}] {rec['status']} in {rec['minutes']} min" + (f": {rec.get('error') or rec.get('reason')}" if rec["status"] != "ok" else ""))
        if bar:
            bar.update(1)
    if bar:
        bar.close()
    print("wrote", summary)


if __name__ == "__main__":
    main()
