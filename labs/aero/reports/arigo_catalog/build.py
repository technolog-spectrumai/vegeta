"""Build the ARIGO product catalogue / technical report (LaTeX -> PDF) from a stability run file.

    labs/aero/sims/arigo_catalog.sh --help

Figures are made here (CAD renders, simulator screenshots, CFD polars, chase tracks, launches, mass budgets, pack
trades) plus the run file's own charts (``arigo_charts``); tables and the numbers in the text come from the run file.
Nothing here runs CFD. Figures already made are kept on a rerun (``--fresh`` remakes them).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AERO = HERE.parents[1]
SIMS = AERO / "sims"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(AERO / "designs"))
sys.path.insert(0, str(SIMS))

import yaml                                                          # noqa: E402

LEAD = ["arigo-1_hybrid", "arigo-1_zero", "arigo-0_hybrid", "arigo-20"]        # the catalogue's products
SCENARIOS = {  # key: (title, [(summary key, label, unit, higher is better, format)])
    "takeoff": ("Launch: from the hand to safe flight", [
        ("takeoff_time_s", "time to safe flight", "s", False, "{:.1f}"), ("hand_launch_ok", "hand launch", "", True, "{}"),
        ("min_speed_m_s", "stall speed", "m/s", False, "{:.1f}"), ("max_climb_m_s", "max climb", "m/s", True, "{:.1f}"),
        ("mass_kg", "take-off mass", "kg", False, "{:.2f}")]),
    "zigzag": ("Zig-zag chase: a manoeuvring bandit at close range", [
        ("zigzag_time_to_5m_s", "time to 5 m", "s", False, "{:.1f}"), ("zigzag_share_within_10m", "time within 10 m", "%", True, "{:.0%}"),
        ("best_instant_turn_deg_s", "instant turn", "deg/s", True, "{:.0f}"), ("roll_rate_deg_s", "roll rate", "deg/s", True, "{:.0f}"),
        ("run_down_14_m_s_bandit_s", "run down 14 m/s", "s", False, "{:.1f}")]),
    "urban": ("Urban chase: low, tight, between buildings", [
        ("urban_time_to_5m_s", "time to 5 m", "s", False, "{:.1f}"), ("urban_share_within_10m", "time within 10 m", "%", True, "{:.0%}"),
        ("urban_crashed", "crashed", "", False, "{}"), ("tightest_sustained_radius_m", "tightest radius", "m", False, "{:.0f}"),
        ("best_sustained_turn_deg_s", "sustained turn", "deg/s", True, "{:.0f}")]),
    "open_field": ("Open-field long chase: a fast bandit over open ground", [
        ("open_field_time_to_5m_s", "time to 5 m", "s", False, "{:.1f}"), ("open_field_share_within_10m", "time within 10 m", "%", True, "{:.0%}"),
        ("max_speed_m_s", "max speed", "m/s", True, "{:.1f}"), ("run_down_18_m_s_bandit_s", "run down 18 m/s", "s", False, "{:.1f}"),
        ("max_climb_m_s", "max climb", "m/s", True, "{:.1f}")]),
    "patrol": ("Patrol: endurance and range on one pack", [
        ("endurance_min", "endurance", "min", True, "{:.0f}"), ("range_km", "range", "km", True, "{:.0f}"),
        ("mass_kg", "take-off mass", "kg", False, "{:.2f}"), ("pack", "pack", "", None, "{}"),
        ("best_endurance_pack", "best-endurance pack", "", None, "{}")]),
}


# ------------------------------------------------------------------------------------------------ helpers
def esc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def fmt(v, f="{:.1f}"):
    if v is None:
        return "--"
    if isinstance(v, bool):
        return "yes" if v else "no"
    try:
        return esc(f.format(v))
    except (ValueError, TypeError):
        return esc(v)


def fig(path, width=r"\textwidth", caption=None, label=None):
    s = [r"\begin{figure}[htbp]", r"\centering", rf"\includegraphics[width={width}]{{{path}}}"]
    if caption:
        s.append(rf"\caption{{{caption}}}")
    if label:
        s.append(rf"\label{{{label}}}")
    s.append(r"\end{figure}")
    return "\n".join(s)


def table(cards, keys, cols, caption, bold=True):
    """Configurations down, properties across; the best value of each numeric column in bold."""
    best = {}
    for k, _, _, hi, _ in cols:
        vals = [(cards[c]["summary"].get(k), c) for c in keys]
        num = [(v, c) for v, c in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if bold and hi is not None and num:
            best[k] = (max if hi else min)(num)[1]
    head = " & ".join([r"\textbf{configuration}"] + [rf"\textbf{{{esc(l)}}}" + (rf" \newline\footnotesize{{({esc(u)})}}" if u else "")
                                                      for _, l, u, _, _ in cols])
    spec = r">{\raggedright\arraybackslash}p{3.4cm}" + "".join(r">{\raggedleft\arraybackslash}p{%.2fcm}" % (11.0 / len(cols))
                                                                 for _ in cols)
    rows = []
    for c in keys:
        cells = [esc(cards[c]["aircraft"])]
        for k, _, _, _, f in cols:
            t = fmt(cards[c]["summary"].get(k), f)
            cells.append(rf"\textbf{{{t}}}" if best.get(k) == c else t)
        rows.append(" & ".join(cells) + r" \\")
    return "\n".join([r"\begin{table}[htbp]", r"\centering\small", rf"\caption{{{caption}}}", rf"\begin{{tabular}}{{{spec}}}",
                      r"\toprule", head + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def winner(cards, keys, k, hi):
    vals = [(cards[c]["summary"].get(k), c) for c in keys]
    vals = [(v, c) for v, c in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return (max if hi else min)(vals)[1] if vals else None


def kpi_rank(K, field):
    return sorted(K, key=lambda c: -(K[c].get(field) or 0))


# ------------------------------------------------------------------------------------------------ figures
def make_figures(out: Path, run_path: Path, doc: dict, skip_cad: bool, skip_sim: bool, fresh: bool, log):
    import figures as F
    fd = out / "fig"
    if fresh and fd.exists():
        shutil.rmtree(fd)
    fd.mkdir(parents=True, exist_ok=True)
    cards = doc["results"]["arigo_perf"]["cards"]
    made = {}
    steps = [("CAD renders", "cad"), ("simulator screenshots", "shots"), ("CFD polars", "polars"), ("chase tracks", "chase"),
             ("launches", "launch"), ("mass budgets and pack trades", "mass"), ("stability bars", "stab"), ("run charts", "charts")]
    try:
        from tqdm import tqdm
        bar = tqdm(total=len(steps), desc="catalogue figures", unit="step")
    except ImportError:
        bar = None
    for title, s in steps:
        if bar:
            bar.set_postfix_str(title)
        log(f"-- {title}")
        if s == "cad" and not skip_cad:
            if fresh or not (fd / "family_lineup.png").exists():
                F.cad_figures(fd, keys=("arigo-1", "arigo-1-zero", "arigo-0", "arigo-20", "arigo-2", "arigo-0-strut"), log=log)
            made["cad"] = True
        elif s == "shots" and not skip_sim:
            if fresh or not list(fd.glob("shot_*.png")):
                F.screenshots(fd, keys=("arigo-1", "arigo-0", "arigo-20"), log=log)
            made["shots"] = True
        elif s == "polars":
            F.polars(fd)
        elif s == "chase" and not skip_sim:
            cache = fd / "chase_summary.yaml"
            if fresh or not cache.exists():
                _, summ = F.chase_tracks(fd, log=log)
                cache.write_text(yaml.safe_dump(json.loads(json.dumps(summ, default=str)), sort_keys=False))
            made["chase"] = True
        elif s == "launch" and not skip_sim:
            F.launches(fd)
            made["launch"] = True
        elif s == "mass":
            F.mass_budgets(fd, cards)
            F.pack_trades(fd, cards)
        elif s == "stab":
            made["stab"] = F.stability_bars(fd, doc) is not None
        elif s == "charts":
            import arigo_charts
            cp = fd / "run.yaml"
            shutil.copy(run_path, cp)
            made["charts"] = [p.name for p in arigo_charts.charts(cp)]
        if bar:
            bar.update(1)
    if bar:
        bar.close()
    made["have"] = {p.name for p in fd.glob("*.png")}
    return made


# ------------------------------------------------------------------------------------------------ the document
def geometry():
    """Span, area, mean chord and CG of each product from the flight models (which read the CAD parameters)."""
    import arigo1_model as am
    import arigo20_model as m20
    import arigo2_model as m2
    out = {}
    for key, C in (("arigo-1_hybrid", am.coefficients("hybrid", "pusher")), ("arigo-1_zero", am.coefficients("zero", "pusher")),
                   ("arigo-0_hybrid", am.coefficients("hybrid", "tractor_lowcam")), ("arigo-20", m20.get()),
                   ("arigo-2_hybrid", m2.coefficients("hybrid"))):
        out[key] = {"b": C["b"], "S": C["S"], "c": C["c"], "x_cg_mm": C["x_cg_m"] * 1e3, "AR": C["b"] ** 2 / C["S"],
                    "aero": C.get("aero_source", "")}
    return out


def document(doc: dict, made: dict, run_name: str, version: str) -> str:
    cards = doc["results"]["arigo_perf"]["cards"]
    K = doc.get("kpi") or {}
    R = doc["results"]
    S = {k: c["summary"] for k, c in cards.items()}
    have = made["have"]
    try:
        G = geometry()
    except Exception as e:                                           # noqa: BLE001  the report still builds without it
        print("geometry from the models failed:", e)
        G = {}
    allk = list(cards)
    lead = [k for k in LEAD if k in cards]
    name = {k: cards[k]["aircraft"] for k in cards}
    rank_g = kpi_rank(K, "general")
    run = doc.get("run", {})

    def F(name_, w=r"\textwidth", cap=None, lab=None):
        return fig(f"fig/{name_}", w, cap, lab) if name_ in have else ""

    def n(k, f, spec="{:.1f}"):
        return fmt(S.get(k, {}).get(f), spec)

    L = []
    a = L.append
    a(r"""\documentclass[11pt,a4paper]{report}
\usepackage[english]{biznesplan_i18n}
\usepackage{tikz}
\usepackage{float}
\usepackage{tabularx}
\usepackage{xcolor}
\definecolor{arigozero}{HTML}{2a78d6}
\definecolor{arigoone}{HTML}{eb6834}
\definecolor{arigotwo}{HTML}{1baf7a}
\definecolor{arigotwenty}{HTML}{eda100}
\definecolor{riskred}{RGB}{170,45,45}
\definecolor{signalgray}{RGB}{90,90,90}
\BusinessPlanCompany{Basilisk Systems}
\BusinessPlanAuthorsText{Prepared by Basilisk Systems}
\BusinessPlanConfidentialText{Internal technical catalogue --- engineering estimates}
\BusinessPlanWatermarkText{BASILISK SYSTEMS}
\newcommand{\hand}[1]{\textcolor{riskred}{#1}}
\newcommand{\product}[2]{\textbf{\textcolor{#1}{#2}}}
\newcommand{\note}[1]{\textcolor{signalgray}{\small #1}}
\setcounter{tocdepth}{1}
\begin{document}
""")
    a(rf"""\BusinessPlanTitlePage
    {{\includegraphics[width=3.2cm]{{assets/basilisk_logo.png}}\\[0.4cm]Basilisk Systems}}
    {{ARIGO}}
    {{Compact autonomous interceptors}}
    {{Product Catalogue \& Technical Report}}
    {{ARIGO-0, ARIGO-1, ARIGO-20 --- what they are, why, and how they fly}}
    {{{esc(version)}}}
""")
    a(r"\tableofcontents" "\n" r"\newpage")

    # -------------------------------------------------------------- executive summary
    best_g, best_u, best_o = rank_g[0] if rank_g else None, (kpi_rank(K, "urban_chase") or [None])[0], (kpi_rank(K, "open_field_chase") or [None])[0]
    a(r"\chapter*{Executive Summary}" "\n" r"\addcontentsline{toc}{chapter}{Executive Summary}")
    a(F("family_lineup.png", r"\textwidth", "The ARIGO family: ARIGO-0 (nose tractor), ARIGO-1 (pusher), ARIGO-20 (twin ducted fans)."))
    a(rf"""ARIGO is a family of compact (about 1.2\,m span, under 2\,kg) electric fixed-wing aircraft that find, follow and
close on a small aerial target --- a \emph{{bandit}} --- autonomously. One airframe concept, three ways to push it:

\begin{{itemize}}
\item \product{{arigoone}}{{ARIGO-1}} --- a pusher. The camera sees a clean sky; the propeller is behind the tail boom.
  \textbf{{Our recommended product.}} Two wings: \emph{{hybrid}} (all-round) and \emph{{zero}} (longest endurance).
\item \product{{arigozero}}{{ARIGO-0}} --- a nose tractor with the camera low, looking through the propeller disc.
  Simpler and cheaper to build; the image is cleaned in software.
\item \product{{arigotwenty}}{{ARIGO-20}} --- two electric ducted fans at the rear: the fast interceptor for open ground.
\end{{itemize}}

\noindent Headline results (stability and performance run \texttt{{{esc(run_name)}}}):
""")
    if best_g:
        a(r"\begin{itemize}")
        a(rf"\item Best all-round configuration: \textbf{{{esc(name[best_g])}}} (general KPI {K[best_g]['general']:.1f}/100).")
        if best_u:
            a(rf"\item Best in the urban chase: \textbf{{{esc(name[best_u])}}} ({K[best_u]['urban_chase']:.1f}/100).")
        if best_o:
            a(rf"\item Best in the open-field long chase: \textbf{{{esc(name[best_o])}}} ({K[best_o]['open_field_chase']:.1f}/100).")
        we = winner(cards, allk, "endurance_min", True)
        a(rf"\item Longest endurance: \textbf{{{esc(name[we])}}}, {n(we, 'endurance_min', '{:.0f}')}\,min on its default pack.")
        wv = winner(cards, allk, "max_speed_m_s", True)
        a(rf"\item Fastest: \textbf{{{esc(name[wv])}}}, {n(wv, 'max_speed_m_s', '{:.0f}')}\,m/s level.")
        a(r"\end{itemize}")
    a(r"""\noindent\note{Every number in this document comes from a model or from the cited simulation run. Hand estimates
(masses, efficiencies, some propulsion values) are marked \hand{in red} or in the sources; screening CFD is not certification
data. Scores are relative to the configurations compared --- add or drop one and the others move.}""")

    # -------------------------------------------------------------- 1 intro
    a(r"\chapter{Introduction}")
    a(r"\section{What ARIGO is}")
    a(r"""ARIGO aircraft are small, hand-launched electric aeroplanes with a forward-looking camera and an on-board computer.
They patrol, detect a small drone or other aerial target, and chase it down using proportional navigation (PN) --- the
same guidance law as an interceptor missile: turn at a rate proportional to the rotation of the line of sight. The
mission is short and violent at the end (the chase) and long and quiet before it (the patrol); the design is the
compromise between the two.""")
    a(r"\section{Why a fixed wing}")
    a(r"""Multirotors hover, but a 1.5--2\,kg multirotor flies for 20--30 minutes and tops out near 20\,m/s. A fixed wing of
the same mass, on the same pack, patrols for over an hour and dashes faster than most small drones it will chase. What
it gives up --- the ability to stop --- matters less when the job is to catch something moving. The ARIGO family keeps
the airframe small enough to throw by hand, so it needs no runway and no launcher (except the ARIGO-20 dash variant).""")
    a(r"\section{How the numbers were made}")
    a(r"""\begin{enumerate}
\item \textbf{CAD} (Dedalus/CadQuery): every aircraft is a parametric model; the renders in this catalogue are those models.
\item \textbf{Aerodynamics}: a vortex-lattice model for the first estimate, then screening RANS CFD (OpenFOAM, $k$-$\omega$ SST)
  over the angle-of-attack range, propellers off. The CFD tables replace the lattice in the flight models.
\item \textbf{Propulsion}: propeller and ducted-fan maps from a propulsor library, sized to the airframe (static thrust and
  thrust at dash speed), electrical power from the maps.
\item \textbf{Mass budget}: geometry $\times$ areal or linear densities (\hand{hand estimates, $\pm$20\,\%}) plus catalogue
  components and the battery pack.
\item \textbf{Flight simulation}: a 6-DOF MuJoCo model flies the aircraft on the CFD tables with an autopilot: stability
  tests, manoeuvres, launches and the three chase scenarios against a scripted bandit.
\end{enumerate}""")
    a(rf"\noindent Run used: \texttt{{{esc(run_name)}}}, commit \texttt{{{esc(run.get('commit', '?'))}}}, host-side, "
      rf"{esc(run.get('started', ''))}.")

    # -------------------------------------------------------------- 2 products list
    a(r"\chapter{The Product Line}")
    a(r"""The catalogue covers three products; ARIGO-1 comes with two wings. Each is described in its own chapter.""")
    cols = [("mass_kg", "mass", "kg", None, "{:.2f}"), ("max_speed_m_s", "max speed", "m/s", True, "{:.0f}"),
            ("min_speed_m_s", "stall", "m/s", False, "{:.1f}"), ("endurance_min", "endurance", "min", True, "{:.0f}"),
            ("range_km", "range", "km", True, "{:.0f}"), ("takeoff_time_s", "launch", "s", False, "{:.1f}")]
    a(table(cards, lead, cols, "The products at a glance (default pack). Bold: best in the column."))
    a(r"\begin{table}[htbp]\centering\small\caption{Roles.}" "\n" r"\begin{tabularx}{\textwidth}{>{\raggedright}p{3.2cm}X}\toprule")
    a(r"\textbf{product} & \textbf{role} \\ \midrule")
    a(r"ARIGO-1 hybrid & All-round interceptor: urban and zig-zag chases, hand launch, over an hour of patrol. \\")
    a(r"ARIGO-1 zero & Long patrol: the longest endurance and range of the family, slightly less agile. \\")
    a(r"ARIGO-0 hybrid & Low-cost interceptor: nose tractor, camera through the disc, software cleans the image. \\")
    a(r"ARIGO-20 & Dash interceptor for open ground: the fastest and best climber; catapult launch, short endurance. \\")
    a(r"\bottomrule\end{tabularx}\end{table}")

    # -------------------------------------------------------------- 3 discarded
    a(r"\chapter{Discarded Ideas}")
    a(r"""Several designs were built, simulated and dropped. They are recorded here because the reasons are part of the design.""")
    a(r"\section{ARIGO-2: two tractor propellers on the wings}")
    a(F("cad_arigo-2_hero.png", r"0.6\textwidth", "ARIGO-2: twin wing-mounted tractors."))
    s2 = R.get("arigo2_mujoco", {}).get("runs", {}).get("hybrid_cm1", {})
    vmc = (s2.get("engine_out") or {}).get("V_mc_full_power_m_s")
    a(rf"""Two propellers clear the camera's view and give differential-thrust yaw. In simulation they added mass and drag
and gave nothing back: ARIGO-2 hybrid scored {fmt(K.get('arigo-2_hybrid', {}).get('general'))} on the general KPI against
{fmt(K.get('arigo-1_hybrid', {}).get('general'))} for ARIGO-1 hybrid. Worse, with one motor out at full power the rudder holds
the aircraft straight only above $V_{{mc}} \approx$ {fmt(vmc)}\,m/s --- far above the stall --- so a motor failure at launch or
in a slow turn is a departure. Dropped.""")
    a(r"\section{ARIGO-0 with a chin-strut camera}")
    a(F("cad_arigo-0-strut_hero.png", r"0.6\textwidth", "ARIGO-0 with the camera on a chin strut ahead of the disc."))
    a(rf"""Putting the camera on a strut ahead of the propeller clears the image but costs drag and a long, flexible mount
that vibrates. The flight results (\product{{magenta}}{{magenta}} in the charts) are below the low camera's: general KPI
{fmt(K.get('arigo-0_strut_hybrid', {}).get('general'))} against {fmt(K.get('arigo-0_hybrid', {}).get('general'))}. The camera
through the disc, with software to remove the blades, is the better answer (Chapter~\ref{{ch:arigo0}}).""")
    a(r"\section{The goshawk wing}")
    a(rf"""A short, thick, high-loaded wing taken from an earlier design. It was the weakest of the three wings on every
airframe: e.g. ARIGO-1 goshawk scored {fmt(K.get('arigo-1_goshawk', {}).get('general'))} general and flew
{n('arigo-1_goshawk', 'endurance_min', '{:.0f}')}\,min against {n('arigo-1_hybrid', 'endurance_min', '{:.0f}')}\,min on the hybrid wing.
Its stability needed a CG fix and it still lost. Dropped from the catalogue; its results remain in the comparison charts.""")
    a(r"\section{Tilt-rotor and tail-sitter VTOL}")
    a(r"""Vertical take-off would remove the hand launch. Both concepts were traded early and rejected on cost and mass: the
tilt mechanism or the extra lift motors weigh what an hour of pack weighs, add failure modes, and the mission does not
need to land precisely --- a belly landing or a net is enough.""")

    # -------------------------------------------------------------- 4 product chapters
    def product_chapter(key, title, label, colour, cad, intro, configs):
        a(rf"\chapter{{{title}}}\label{{{label}}}")
        a(F(f"cad_{cad}_hero.png", r"0.85\textwidth"))
        a(intro)
        a(r"\section{Design}")
        a(F(f"cad_{cad}_views.png", r"\textwidth", f"{title}: four views."))
        parts = sorted(p for p in have if p.startswith(f"cad_{cad}_part_"))
        if parts:
            a(r"\begin{figure}[htbp]\centering")
            for p in parts:
                nm = {"props": "propeller disc", "motors": "motors", "camera": "camera"}.get(p[len(f"cad_{cad}_part_"):-4], p[len(f"cad_{cad}_part_"):-4])
                a(rf"\begin{{subfigure}}{{0.32\textwidth}}\centering\includegraphics[width=\linewidth,height=3.2cm,keepaspectratio]{{fig/{p}}}"
                  rf"\caption{{{esc(nm)}}}\end{{subfigure}}\hfill")
            a(rf"\caption{{{title}: the parts.}}\end{{figure}}")
        g = G.get(configs[0])
        if g:
            a(r"\begin{table}[htbp]\centering\small\caption{Geometry and reference values (from the CAD parameters).}")
            a(r"\begin{tabular}{lr}\toprule")
            for lab_, v in (("span $b$", f"{g['b']:.3f} m"), ("wing area $S$", f"{g['S']:.4f} m$^2$"), ("mean chord", f"{g['c'] * 1e3:.0f} mm"),
                            ("aspect ratio", f"{g['AR']:.1f}"), ("CG from the nose", f"{g['x_cg_mm']:.0f} mm")):
                a(rf"{lab_} & {v} \\")
            a(r"\bottomrule\end{tabular}\end{table}")
        a(r"\section{Performance card}")
        cols = [("max_speed_m_s", "max speed", "m/s", True, "{:.1f}"), ("min_speed_m_s", "stall", "m/s", False, "{:.1f}"),
                ("max_climb_m_s", "climb", "m/s", True, "{:.1f}"), ("best_sustained_turn_deg_s", "sust. turn", "deg/s", True, "{:.0f}"),
                ("roll_rate_deg_s", "roll", "deg/s", True, "{:.0f}"), ("pitch_rate_deg_s", "pitch", "deg/s", True, "{:.0f}"),
                ("yaw_rate_deg_s", "yaw", "deg/s", True, "{:.0f}")]
        a(table(cards, configs, cols, f"{title}: flight envelope and agility.", bold=len(configs) > 1))
        cols = [("endurance_min", "endurance", "min", True, "{:.0f}"), ("range_km", "range", "km", True, "{:.0f}"),
                ("mass_kg", "mass", "kg", False, "{:.2f}"), ("pack", "pack", "", None, "{}"), ("propulsor", "propulsor", "", None, "{}")]
        a(table(cards, configs, cols, f"{title}: energy and mass.", bold=len(configs) > 1))
        for c in configs:
            pm = cards[c]["point_mass"]
            mb = pm.get("mass_budget") or {}
            if mb.get("items"):
                a(rf"\begin{{longtable}}{{p{{4.2cm}}rp{{8.4cm}}}}\caption{{{esc(name[c])}: mass budget.}}\\\toprule")
                a(r"\textbf{item} & \textbf{g} & \textbf{source} \\ \midrule\endhead")
                for it in mb["items"]:
                    src = esc(it.get("source", ""))
                    if "HAND" in src:
                        src = rf"\hand{{{src}}}"
                    a(rf"{esc(it['item'])} & {it['g']:.0f} & \footnotesize {src} \\")
                pk = mb.get("pack") or {}
                a(rf"\midrule empty & {mb.get('empty_g', 0):.0f} & \\ pack: {esc(pk.get('name', ''))} & {pk.get('g', 0):.0f} & "
                  rf"\footnotesize {esc(pk.get('source', ''))} \\ \textbf{{total}} & \textbf{{{mb.get('total_g', 0):.0f}}} & "
                  rf"pack share {100 * mb.get('pack_share', 0):.0f}\,\% \\")
                a(r"\bottomrule\end{longtable}")
            pr = pm.get("propulsion") or {}
            if pr:
                a(rf"\noindent\textbf{{Propulsion ({esc(name[c])}).}} {esc(pr.get('entry'))} $\times$ {pr.get('units')}, "
                  rf"{pr.get('power_w_per_unit', 0):.0f}\,W each; static thrust {pr.get('static_thrust_N', 0):.1f}\,N, "
                  rf"{pr.get('thrust_at_dash_N', 0):.1f}\,N at {pr.get('dash_speed_m_s', 0):.0f}\,m/s. "
                  rf"\note{{Selection: {esc(pr.get('why', ''))}.}}" "\n")
        a(r"\section{Stability}")
        a(stability_table(R, configs, name))
        a(r"\section{In flight}")
        shots = sorted(p for p in have if p.startswith(f"shot_{cad}_"))
        if shots:
            a(r"\begin{figure}[htbp]\centering")
            for p in shots:
                lab_ = p[len(f"shot_{cad}_"):-4].replace("_", " ")
                a(rf"\begin{{subfigure}}{{0.48\textwidth}}\centering\includegraphics[width=\linewidth]{{fig/{p}}}\caption{{{esc(lab_)}}}"
                  r"\end{subfigure}\hfill")
            a(rf"\caption{{{title} in the simulator (MuJoCo, 6-DOF on the CFD tables).}}\end{{figure}}")
        a(r"\section{Scores}")
        a(r"\begin{table}[htbp]\centering\small\caption{KPI scores (0--100, relative to all configurations in the run) and rank.}")
        a(r"\begin{tabular}{lrrrr}\toprule \textbf{configuration} & \textbf{general} & \textbf{chase} & \textbf{urban} & \textbf{open field} \\ \midrule")
        for c in configs:
            k_ = K.get(c, {})
            a(rf"{esc(name[c])} & {fmt(k_.get('general'))} (\#{rank_g.index(c) + 1 if c in rank_g else '-'}) & {fmt(k_.get('chase'))} & "
              rf"{fmt(k_.get('urban_chase'))} & {fmt(k_.get('open_field_chase'))} \\")
        a(r"\bottomrule\end{tabular}\end{table}")

    product_chapter("arigo-1", "ARIGO-1 --- the pusher", "ch:arigo1", "arigoone", "arigo-1", rf"""
\noindent\product{{arigoone}}{{ARIGO-1}} puts the motor at the back of the fuselage, between the wing and the tail. The camera
in the nose sees nothing but sky and the target; the propeller works in clean air behind the fuselage. The hybrid wing is
the all-round choice: {n('arigo-1_hybrid', 'endurance_min', '{:.0f}')}\,min endurance, {n('arigo-1_hybrid', 'range_km', '{:.0f}')}\,km
range, a {n('arigo-1_hybrid', 'tightest_sustained_radius_m', '{:.0f}')}\,m sustained turn radius, and it leaves the hand
in {n('arigo-1_hybrid', 'takeoff_time_s')}\,s. The zero wing trades some agility for the longest patrol in the family
({n('arigo-1_zero', 'endurance_min', '{:.0f}')}\,min).
""", [c for c in ("arigo-1_hybrid", "arigo-1_zero") if c in cards])
    a(F("cad_arigo-1-zero_hero.png", r"0.7\textwidth", "ARIGO-1 with the zero (long-endurance) wing."))

    occ = ((R.get("arigo0_camera") or {}).get("results") or [{}])[0]
    product_chapter("arigo-0", "ARIGO-0 --- the nose tractor", "ch:arigo0", "arigozero", "arigo-0", rf"""
\noindent\product{{arigozero}}{{ARIGO-0}} is the conventional layout: the motor in the nose, pulling. It is the simplest and
cheapest to build and balance, and the propeller in front protects nothing behind it from a nose-first landing. The camera
sits low in the nose and looks \emph{{through}} the propeller disc. Its CG was moved 15\,mm forward after the CFD screening
to give the same static margin as ARIGO-1 (about 12.5\,\% MAC).
""", [c for c in ("arigo-0_hybrid",) if c in cards])
    if occ:
        a(r"\section{The camera behind the propeller}")
        a(rf"""Geometry only (no optics): of the camera's image, {fmt((occ.get('behind_disc') or {}).get('image'), '{:.0%}')} lies behind
the propeller disc. A free-running propeller blocks on average {fmt((occ.get('free_running_global_mean') or {}).get('image'), '{:.1%}')}
of the image in any frame; fusing two frames half a revolution apart leaves
{fmt((occ.get('two_frame_fusion_residual') or {}).get('image'), '{:.1%}')} --- the spinner. With a global-shutter camera and
short exposures the blades are thin, sharp shapes that software can mask and fill from the neighbouring frame.""")
        mb = occ.get("motion_blur") or []
        if mb:
            a(r"\begin{table}[htbp]\centering\small\caption{Blade smear against exposure at 6000\,rpm.}\begin{tabular}{rrrr}\toprule")
            a(r"\textbf{exposure (ms)} & \textbf{smear (deg)} & \textbf{tip smear (mm)} & \textbf{image touched} \\ \midrule")
            for m in mb:
                a(rf"{m['exposure_ms']} & {m['smear_deg']:.1f} & {m['tip_smear_mm']:.0f} & {m['touched']['image']:.1%} \\".replace("%", r"\%"))
            a(r"\bottomrule\end{tabular}\end{table}")

    product_chapter("arigo-20", "ARIGO-20 --- the ducted-fan dash interceptor", "ch:arigo20", "arigotwenty", "arigo-20", rf"""
\noindent\product{{arigotwenty}}{{ARIGO-20}} has two electric ducted fans at the rear of the fuselage, on either side of an H
tail. It is built for speed: {n('arigo-20', 'max_speed_m_s', '{:.0f}')}\,m/s level and {n('arigo-20', 'max_climb_m_s', '{:.0f}')}\,m/s
climb. The fans are thirsty at low speed, so it patrols for {n('arigo-20', 'endurance_min', '{:.0f}')}\,min, and its stall speed of
{n('arigo-20', 'min_speed_m_s')}\,m/s needs a catapult (launch time {n('arigo-20', 'takeoff_time_s')}\,s). Its CG was moved 14\,mm
aft after the CFD screening to bring the static margin down to about 12.5\,\% MAC.
""", [c for c in ("arigo-20",) if c in cards])

    # -------------------------------------------------------------- 5 scenarios
    a(r"\chapter{Scenario Analyses}")
    a(r"""Each scenario is a mission phase. The tables list the four products plus, for reference, the discarded
configurations; bold marks the best value among all of them. The simulator's chase uses proportional navigation ($N=4$)
in the horizontal plane with height hold, speed control and a low-altitude floor; the bandit flies a scripted path.""")
    others = [c for c in allk if c not in lead]
    texts = {
        "takeoff": r"""A hand launch at 12\,m/s from 1.8\,m: the aircraft must climb away at full power before it touches the
ground and reach 10\,m at a safe speed. A stall speed near or above the throw speed means a catapult.""",
        "zigzag": r"""The bandit weaves at close range. What counts is roll rate (how fast the turn starts) and instantaneous turn
rate (how hard it can pull for a second).""",
        "urban": r"""Low, tight, between buildings: the bandit turns hard at low altitude. A small sustained radius and a low
minimum speed matter more than top speed; a crash scores zero.""",
        "open_field": r"""A fast bandit runs away over open ground. Speed and climb decide it: an aircraft that cannot out-run the
bandit only closes when the bandit turns.""",
        "patrol": r"""Before the chase, the aircraft waits: loitering at its best-endurance speed, the camera and computer on
(\hand{15\,W avionics, 85\,\% usable pack}).""",
    }
    figs = {"takeoff": ["launch.png"], "zigzag": ["chase_zigzag.png"], "urban": ["chase_urban.png"],
            "open_field": ["chase_open_field.png"], "patrol": ["pack_trades.png"]}
    for sc, (title, cols) in SCENARIOS.items():
        a(rf"\section{{{title}}}")
        a(texts[sc])
        for f_ in figs[sc]:
            a(F(f_, r"\textwidth", esc(title) + " (computed for this catalogue with the current models)."))
        a(table(cards, lead + others, cols, f"{title}: all configurations."))
        hi = cols[0][3]
        w = winner(cards, lead, cols[0][0], hi)
        if w:
            a(rf"\noindent\textbf{{Among the products, {esc(name[w])} leads on {esc(cols[0][1])}}} "
              rf"({fmt(S[w].get(cols[0][0]), cols[0][4])} {esc(cols[0][2])}).")
    a(r"\section{Stall, gusts and motor failure}")
    a(r"""All products are statically stable in pitch with a damped short period; the spiral mode is slowly divergent on
all of them (normal for a small aircraft, held by the autopilot). The stall-and-release test pulls to the stall and lets go
of the stick; the gust test is a 3\,m/s vertical gust at cruise.""")
    a(F("stability_bars.png", r"\textwidth", "Stability by configuration (MuJoCo linearised modes and the stall test)."))
    a(stability_table(R, lead + [c for c in others if c != "arigo-0_strut_goshawk"], name))
    e20 = (R.get("arigo20_mujoco", {}).get("runs", {}).get("cm1") or {}).get("engine_out") or {}
    if e20:
        a(rf"""\noindent\textbf{{ARIGO-20 with one fan out}} (open loop): bank reaches {fmt(e20.get('max_bank_deg'), '{:.0f}')}\,deg and
the aircraft loses {fmt(-e20.get('altitude_change_m', 0), '{:.0f}')}\,m --- the fans are close to the centreline, but the
autopilot must cut the good fan and glide. ARIGO-0 and ARIGO-1 have one motor: a failure is a glide.""")

    # -------------------------------------------------------------- 6 combined
    a(r"\chapter{Combined Analysis}")
    for p in made.get("charts") or []:
        cap = {"kpi": "KPI scores by configuration (hatching: winner densest, then runner-up and second runner-up).",
               "properties": "Every property of the performance cards, all configurations.",
               "curves": "Sustained turn rate and climb rate over speed."}
        c_ = next((v for k_, v in cap.items() if k_ in p), "")
        a(F(p, r"\textwidth", c_))
    a(F("cfd_polars.png", r"\textwidth", "The CFD polars behind the flight models."))
    a(F("mass_budgets.png", r"\textwidth", r"Mass budgets: where the grams go (\hand{structure masses are hand estimates})."))
    a(r"\begin{table}[htbp]\centering\small\caption{All configurations by general KPI.}\begin{tabular}{rlrrrr}\toprule")
    a(r"\# & \textbf{configuration} & \textbf{general} & \textbf{chase} & \textbf{urban} & \textbf{open field} \\ \midrule")
    for i, c in enumerate(rank_g, 1):
        k_ = K[c]
        a(rf"{i} & {esc(k_.get('name', c))} & {fmt(k_.get('general'))} & {fmt(k_.get('chase'))} & {fmt(k_.get('urban_chase'))} & "
          rf"{fmt(k_.get('open_field_chase'))} \\")
    a(r"\bottomrule\end{tabular}\end{table}")

    # -------------------------------------------------------------- 7 conclusions
    a(r"\chapter{Conclusions}")
    a(r"\begin{enumerate}")
    if best_g:
        a(rf"\item \textbf{{{esc(name[best_g])}}} is the best all-round configuration of this run (general {K[best_g]['general']:.1f}).")
    a(r"""\item \textbf{ARIGO-1 hybrid is the recommended product}: hand launch, over an hour of patrol, the tightest turns and the
  best urban chase; the camera sees a clean sky.
\item \textbf{ARIGO-1 zero} is the long-patrol option on the same fuselage --- swap the wing.
\item \textbf{ARIGO-0} is a few points behind ARIGO-1 everywhere; choose it on cost and simplicity, and budget the image
  processing that removes the propeller.
\item \textbf{ARIGO-20} is a specialist: unbeatable on speed and climb over open ground, poor in the urban chase, short
  endurance, needs a catapult.
\item \textbf{ARIGO-2}, the chin strut and the goshawk wing are dropped.
\end{enumerate}""")
    a(r"\section{Limits of this report and next steps}")
    a(r"""\begin{itemize}
\item CFD is screening quality (propellers off, coarse meshes); the power-on cases and a mesh study are next.
\item Structure masses are hand estimates ($\pm$20\,\%); weigh the first airframes and rerun.
\item The bandit is scripted, not adversarial; the chase scores rank the aircraft, they do not predict a kill probability.
\item Flight tests: hand launch, stall and the urban manoeuvres first, on ARIGO-1 hybrid.
\end{itemize}""")

    # -------------------------------------------------------------- glossary
    a(r"""\chapter*{Glossary}\addcontentsline{toc}{chapter}{Glossary}
\begin{description}[leftmargin=1.2cm, style=nextline]
\item[Bandit] The aerial target being chased.
\item[PN] Proportional navigation: turn rate proportional to the line-of-sight rate ($N=4$ here).
\item[CFD] Computational fluid dynamics; here RANS screening with OpenFOAM.
\item[MAC] Mean aerodynamic chord; the static margin is the distance from the CG to the neutral point in MAC.
\item[Dutch roll] The coupled yaw--roll oscillation; $\zeta$ is its damping ratio.
\item[Spiral mode] The slow tendency of a banked aircraft to tighten (diverge) or level (converge) its turn.
\item[$V_{mc}$] The slowest speed at which full rudder holds the aircraft straight with one motor out.
\item[EDF] Electric ducted fan.
\item[KPI] The scores used here: each property scaled 0--100 across the configurations, then averaged per mission.
\end{description}
\end{document}""")
    return "\n\n".join(x for x in L if x)


def stability_table(R, configs, name):
    """Linear modes, the stall and the gust from the stability runs (``arigo*_mujoco``) for the configurations listed."""
    rows = []
    for c in configs:
        fam, _, wing = c.partition("_")
        if fam == "arigo-20":
            x = (R.get("arigo20_mujoco", {}).get("runs") or {}).get("cm1")
        elif fam == "arigo-2":
            x = (R.get("arigo2_mujoco", {}).get("runs") or {}).get(f"{wing}_cm1")
        elif fam == "arigo-1":
            x = (R.get("arigo1_mujoco", {}).get("runs") or {}).get(f"{wing}_pusher_cm1")
        else:
            w = wing.replace("strut_", "")
            x = (R.get("arigo0_mujoco", {}).get("runs") or {}).get(f"{w}_{'tractor' if 'strut' in wing else 'tractor_lowcam'}_cm1")
        if not x or "linear_modes" not in x:
            continue
        lm, st, gu = x["linear_modes"], x.get("stall_and_release") or {}, x.get("vertical_gust_3m_s") or {}
        rows.append(rf"{esc(name.get(c, c))} & {lm['short_period']['zeta']:.2f} & {lm['dutch_roll']['zeta']:.3f} & "
                    rf"{lm['spiral']['t_half_or_double_s']:.0f} & {fmt(st.get('altitude_lost_m'), '{:.0f}')} & "
                    rf"{fmt((x.get('verdict') or {}).get('recovers_from_stall_hands_off'))} & {fmt(gu.get('peak_load_factor'), '{:.2f}')} \\")
    if not rows:
        return ""
    return "\n".join([r"\begin{table}[htbp]\centering\small\caption{Stability (MuJoCo, at 18\,m/s; ARIGO-20 at its test speed).}",
                      r"\begin{tabular}{lrrrrrr}\toprule",
                      r"\textbf{configuration} & \textbf{SP $\zeta$} & \textbf{DR $\zeta$} & \textbf{spiral $t_2$ (s)} & "
                      r"\textbf{stall loss (m)} & \textbf{recovers} & \textbf{gust $n$} \\ \midrule", *rows,
                      r"\bottomrule\end{tabular}\end{table}"])


# ------------------------------------------------------------------------------------------------ main
def newest_run():
    runs = sorted((SIMS / "output" / "runs").glob("arigo_stability_*.yaml"))
    if runs:
        return runs[-1]
    base = sorted((HERE / "data").glob("arigo_stability_*.yaml"))
    return base[-1] if base else None


def compile_pdf(out: Path, log) -> Path | None:
    tex = out / "arigo_catalog.tex"
    if shutil.which("latexmk"):
        cmd = [["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex.name]]
    elif shutil.which("pdflatex"):
        cmd = [["pdflatex", "-interaction=nonstopmode", tex.name]] * 2
    else:
        log("no latexmk/pdflatex on PATH: the .tex is ready, compile it yourself (see --help)")
        return None
    for c in cmd:
        r = subprocess.run(c, cwd=out, capture_output=True, text=True)
        if r.returncode:
            log(r.stdout[-3000:])
            log(f"LaTeX failed; see {out / 'arigo_catalog.log'}")
            return None
    return out / "arigo_catalog.pdf"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="arigo_catalog.sh", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--run", type=Path, help="stability run file (default: the newest sims/output/runs/arigo_stability_*.yaml, "
                                             "else the copy in reports/arigo_catalog/data/)")
    ap.add_argument("--out", type=Path, default=SIMS / "output" / "catalog", help="output folder (default: %(default)s)")
    ap.add_argument("--skip-cad", action="store_true", help="no CAD renders (faster; the figures are left out)")
    ap.add_argument("--skip-sim", action="store_true", help="no simulator screenshots, chase tracks or launches")
    ap.add_argument("--fresh", action="store_true", help="remake every figure (default: keep the ones already made)")
    ap.add_argument("--no-pdf", action="store_true", help="write the .tex only")
    a = ap.parse_args(argv)
    run = a.run or newest_run()
    if not run or not run.exists():
        print("no stability run file: run  labs/aero/sims/arigo_stability.sh  first, or pass --run FILE, or --compute")
        return 2
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"run file: {run}\noutput:   {out}")
    doc = yaml.safe_load(run.read_text())
    made = make_figures(out, run, doc, a.skip_cad, a.skip_sim, a.fresh, print)
    doc = yaml.safe_load((out / "fig" / "run.yaml").read_text())          # with the kpi section from arigo_charts
    shutil.copy(HERE / "biznesplan_i18n.sty", out)
    (out / "assets").mkdir(exist_ok=True)
    shutil.copy(HERE / "assets" / "basilisk_logo.png", out / "assets")
    version = f"draft {dt.date.today():%Y-%m-%d}, run {doc.get('run', {}).get('commit', '?')}"
    (out / "arigo_catalog.tex").write_text(document(doc, made, run.name, version))
    print("wrote", out / "arigo_catalog.tex")
    if not a.no_pdf:
        pdf = compile_pdf(out, print)
        if pdf:
            print("PDF:", pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
