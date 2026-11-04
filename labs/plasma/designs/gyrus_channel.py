"""Gyrus's channel in one dimension (``docs/hall.md``, H4): ``vegeta.fulgur.channel`` driven by H3's field along
the channel midline at the design point; the discharge current in time (the breathing mode), the time-averaged
current, thrust and mass utilisation against the 0-D chain, and the profiles.

    python gyrus_channel.py run [t_end_us] [out.npz]     # the run (minutes); saves the history and the profiles
    python gyrus_channel.py plot [in.npz] [out.png]      # the figure from a saved run
"""
from __future__ import annotations

import sys

import numpy as np

from vegeta.fulgur.channel import ChannelModel
from vegeta.fulgur.hall import point_for_power, scale_channel, tune_ampere_turns, CircuitSection

from gyrus import Gyrus


def midline_field(p=None, lc_fine: float = 0.5e-3):
    """B_r along the channel midline from z = 0 to the domain's end, from the magnetostatic solve: (z, B, solve)."""
    p = p or {q.name: q.default for q in Gyrus.parameters}
    sec = Gyrus.section(p)
    ch = sec.ch
    from vegeta.fulgur.hall import reluctance_circuit
    sol = tune_ampere_turns(sec, ch.b_exit, reluctance_circuit(ch)["ampere_turns"], lc_fine=lc_fine)
    r_cols = np.unique(sol["r"])
    r_mid = r_cols[np.argmin(np.abs(r_cols - ch.d / 2))]
    m = (sol["r"] == r_mid) & (sol["z"] >= 0)
    z, b = sol["z"][m], np.abs(sol["B_r"][m])
    o = np.argsort(z)
    return z[o], b[o], sol


def model(p=None, n_cells: int = 100, **kw) -> ChannelModel:
    p = p or {q.name: q.default for q in Gyrus.parameters}
    z, b, _ = midline_field(p)
    ch = scale_channel(p["power"], p["v_d"])
    pt = point_for_power(p["power"], p["v_d"])
    return ChannelModel(ch, lambda zz: float(np.interp(zz, z, b)), p["v_d"], pt.mdot_a, n_cells=n_cells, **kw)


def run(t_end_us: float = 300.0, out="gyrus_channel.npz", dt: float = 2e-8, n_cells: int = 80, report_every_us: float = 20.0) -> dict:
    p = {q.name: q.default for q in Gyrus.parameters}
    mdl = model(p, n_cells=n_cells)
    pt = point_for_power(p["power"], p["v_d"])
    t = 0.0
    while t < t_end_us - 1e-9:
        mdl.run(t_end=report_every_us * 1e-6, dt=dt)
        t += report_every_us
        s = mdl.summary()
        print(f"t={t:5.0f} us  I_d={s['I_d']:.3f} A (±{s['I_rel_amp']:.2f}, {s['f_breathing_Hz']/1e3:.0f} kHz)  T={s['thrust']*1e3:5.1f} mN  "
              f"eta_m={s['mass_utilisation']:.2f}  Te_max={s['te_max']:.1f} eV  z_ion={s['z_ionisation_peak']*1e3:.1f} mm", flush=True)
    s = mdl.summary(tail=0.6)
    pr = s["profiles"]
    np.savez(out, history=s["history"], **{k: v for k, v in pr.items()}, L=mdl.ch.L, I_d=s["I_d"], I_amp=s["I_amp"], f=s["f_breathing_Hz"],
             thrust=s["thrust"], eta_m=s["mass_utilisation"], v_exit=s["v_exit"], I_0d=pt.i_d, thrust_0d=pt.thrust, V_d=p["v_d"])
    return s


def plot(inp="gyrus_channel.npz", out="gyrus_channel.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BLUE, ORANGE, GREY = "#1f2328", "#6e7781", "#2f6db3", "#c8711a", "#9aa4ad"
    d = np.load(inp)
    h = d["history"]; z = d["z"] * 1e3; L = float(d["L"]) * 1e3
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax = axes[0, 0]
    ax.plot(h[:, 0] * 1e6, h[:, 1], color=BLUE, lw=1)
    ax.axhline(float(d["I_0d"]), color=ORANGE, ls=":", lw=1); ax.text(1, float(d["I_0d"]) + 0.05, f"0-D chain {float(d['I_0d']):.2f} A", color=ORANGE, fontsize=8)
    ax.axhline(float(d["I_d"]), color=INK, ls="--", lw=1); ax.text(1, float(d["I_d"]) - 0.25, f"1-D mean {float(d['I_d']):.2f} A, breathing {float(d['f'])/1e3:.0f} kHz", color=INK, fontsize=8)
    ax.set_xlabel("t [µs]"); ax.set_ylabel("discharge current [A]"); ax.set_title("the discharge current: the breathing mode", loc="left", fontsize=10, color=INK); ax.grid(color="#e6e8eb")
    panels = [("n", 1e-17, "plasma density [10¹⁷ m⁻³]", BLUE), ("n_n", 1e-19, "neutral density [10¹⁹ m⁻³]", GREY), ("T_e", 1.0, "electron temperature [eV]", ORANGE),
              ("phi", 1.0, "potential [V]", INK), ("v", 1e-3, "ion velocity [km/s]", BLUE)]
    for ax, (key, k, label, col) in zip([axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]], panels):
        ax.plot(z, d[key] * k, color=col, lw=2)
        ax.axvline(L, color=GREY, ls="--", lw=1); ax.text(L + 0.3, ax.get_ylim()[1] * 0.9, "exit", color=MUTED, fontsize=8)
        ax.set_xlabel("z [mm] (anode at 0)"); ax.set_ylabel(label); ax.grid(color="#e6e8eb")
        if key == "T_e":
            ax2 = ax.twinx(); ax2.plot(z, d["B"] * 1e4, color=GREY, lw=1, ls=":"); ax2.set_ylabel("B [G]", color=MUTED)
        if key == "n":
            ax2 = ax.twinx(); ax2.plot(z, d["S"] / d["S"].max(), color=ORANGE, lw=1, ls=":"); ax2.set_ylabel("ionisation rate (normalised)", color=ORANGE)
    fig.suptitle(f"Gyrus-300 channel, 1-D fluid at {float(d['V_d']):.0f} V: mean I_d {float(d['I_d']):.2f} A (0-D {float(d['I_0d']):.2f}), thrust {float(d['thrust'])*1e3:.1f} mN (0-D {float(d['thrust_0d'])*1e3:.1f}), "
                 f"mass utilisation {float(d['eta_m']):.2f}, ions leave at {float(d['v_exit'])/1e3:.1f} km/s", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["plot"]:
        print(plot(*sys.argv[2:4]))
    else:
        args = sys.argv[2:] if sys.argv[1:2] == ["run"] else sys.argv[1:]
        t_end = float(args[0]) if args else 300.0
        out = args[1] if len(args) > 1 else "gyrus_channel.npz"
        s = run(t_end, out)
        print("saved", out)
