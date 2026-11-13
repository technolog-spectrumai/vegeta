"""P1's chart: thrust per watt against the mean field (the 1/(mu E) law with the two presets on it) and the Mark 4C's
thrust at its spark limit through Warsaw's year.    python p1_chart.py [out.png]"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from vegeta.fulgur import Air, MARK4, MARK4C, climate, thrust_per_watt

G = 9.81
INK, MUTED, BLUE, ORANGE = "#1f2328", "#6e7781", "#2f6db3", "#c8711a"
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
E = np.linspace(0.2e6, 3e6, 200)
a.plot(E / 1e6, [thrust_per_watt(1.0, e) * 1e3 for e in E], color=BLUE, lw=2, label="1 / (μE), dry air")
for t, lab, off in ((MARK4.at_power(50.0, Air(rh=0.5)), "Mark IV at 50 W", (14, 30)),
                    (MARK4C.at_thrust(0.025 * G), "Mark 4C at 25 gf", (-20, -44))):
    d = min(g.d for g in MARK4.gaps()) if "IV" in lab else min(g.d for g in MARK4C.gaps())
    x, y = t["V"] / d / 1e6, t["thrust_per_watt"] * 1e3
    a.plot(x, y, "o", ms=9, color=ORANGE, mec="white", mew=2)
    a.annotate(f"{lab}\n{t['V']/1e3:.0f} kV, {t['thrust']/G*1e3:.0f} gf", (x, y), xytext=off, textcoords="offset points",
               color=INK, fontsize=9, ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a.set_xlabel("mean field across the gap  [MV/m]", color=INK); a.set_ylabel("thrust per watt  [mN/W]", color=INK)
a.set_title("The drift region's efficiency: lower fields, more thrust per watt", loc="left", color=INK, fontsize=11)
a.set_ylim(0, 20); a.grid(axis="y", color="#e6e8eb"); a.spines[["top", "right"]].set_visible(False); a.legend(frameon=False)
c = climate.conditions()
keys = list(climate.WARSAW) + ["humid_day", "indoor"]
res = [MARK4C.at_margin(1.0, c[k]) for k in keys]
cols = [BLUE if r["limited_by"] == "propagation" else ORANGE for r in res]
b.bar(range(len(keys)), [r["thrust"] / G * 1e3 for r in res], color=cols, width=0.72)
b.set_xticks(range(len(keys)), [k.replace("humid_day", "rainy\nday") for k in keys], color=INK)
b.set_ylabel("thrust at spark margin 1  [gf]", color=INK)
b.set_title("Mark 4C's most through Warsaw's year\nlimited by the air gap (blue) or by the wet surface (orange)", loc="left", color=INK, fontsize=11)
b.grid(axis="y", color="#e6e8eb"); b.spines[["top", "right"]].set_visible(False); b.set_axisbelow(True)
for i, r in enumerate(res):
    b.text(i, r["thrust"] / G * 1e3 + 0.6, f"{r['V']/1e3:.0f} kV", ha="center", fontsize=8, color=MUTED)
fig.suptitle("Fulgur P1: the 1-D corona model, fitted once on the Mark IV's 50 W", color=INK)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "plasma_p1.png"
fig.savefig(out, dpi=110)
print(out)
