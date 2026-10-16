"""Section polars: one ``analyze`` over two backends — **NeuralFoil** (a neural surrogate of XFOIL, a Python
package, seconds for hundreds of points, a confidence per point) and **XFOIL** itself (the executable on the PATH, a
panel method with a viscous boundary layer, the reference) — and the numbers a wing design reads from a polar.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .result import CommandRecord, Result
from .section import Section, flap_effectiveness, thin_airfoil

NEURALFOIL_MODEL = "large"
XFOIL_ITER = 200


@dataclass
class Polar:
    """A section's coefficients against the angle of attack at one Reynolds number. ``confidence`` is NeuralFoil's
    per-point confidence (1 for XFOIL points that converged; unconverged points are dropped)."""
    section: str
    Re: float
    alpha_deg: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    confidence: np.ndarray
    source: str
    mach: float = 0.0
    n_crit: float = 9.0
    notes: str = ""

    def __len__(self):
        return len(self.alpha_deg)

    @property
    def i_max(self) -> int:
        return int(np.argmax(self.cl))

    def attached(self) -> slice:
        """The points up to and including cl_max."""
        return slice(0, self.i_max + 1)

    def at_cl(self, cl: float, key: str):
        """``cd``, ``cm`` or ``alpha_deg`` on the attached branch at a lift coefficient (linear interpolation)."""
        s = self.attached()
        return float(np.interp(cl, self.cl[s], getattr(self, key)[s]))

    def summary(self, cl_design: float | None = None) -> dict:
        """cl_max and its angle, the zero-lift angle, the lift slope (per rad, a least-squares fit over the 8° above α₀), cd_min,
        the best L/D and its cl, the moment at zero lift and the aerodynamic centre (from the moment's slope), the lift
        kept 4° past the stall, the least confidence on the attached branch, and at ``cl_design``: α, cd, cm, L/D."""
        s = self.attached()
        a, cl, cd, cm = self.alpha_deg[s], self.cl[s], self.cd[s], self.cm[s]
        i = self.i_max
        a0 = float(np.interp(0.0, cl, a))
        lin = (a >= a0 - 0.5) & (a <= a0 + 8.0)
        cla = float(np.polyfit(np.radians(a[lin]), cl[lin], 1)[0]) if lin.sum() >= 2 else math.nan
        cm0 = float(np.interp(0.0, cl, cm))
        mid = (cl > 0.1) & (cl < 0.8 * cl[-1])
        dcm_dcl = float(np.polyfit(cl[mid], cm[mid], 1)[0]) if mid.sum() >= 3 else math.nan
        j = min(i + 4, len(self) - 1)
        out = {"section": self.section, "Re": self.Re, "source": self.source, "cl_max": float(cl[-1]), "alpha_cl_max_deg": float(a[-1]),
               "alpha0_deg": a0, "cl_alpha_per_rad": cla, "cd_min": float(cd.min()), "LD_max": float(np.max(cl / cd)),
               "cl_at_LD_max": float(cl[int(np.argmax(cl / cd))]), "cm0": cm0, "x_ac": 0.25 - dcm_dcl,
               "cl_4deg_past_stall_ratio": float(self.cl[j] / cl[-1]), "confidence_min_attached": float(self.confidence[s].min())}
        if cl_design is not None:
            out.update(cl_design=cl_design, alpha_design_deg=self.at_cl(cl_design, "alpha_deg"), cd_design=self.at_cl(cl_design, "cd"),
                       cm_design=self.at_cl(cl_design, "cm"), LD_design=cl_design / self.at_cl(cl_design, "cd"))
        return out

    def to_dict(self) -> dict:
        return {"section": self.section, "Re": self.Re, "mach": self.mach, "n_crit": self.n_crit, "source": self.source, "notes": self.notes,
                "alpha_deg": self.alpha_deg.tolist(), "cl": self.cl.tolist(), "cd": self.cd.tolist(), "cm": self.cm.tolist(),
                "confidence": self.confidence.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Polar":
        return cls(d["section"], d["Re"], *(np.asarray(d[k], float) for k in ("alpha_deg", "cl", "cd", "cm", "confidence")),
                   d["source"], d.get("mach", 0.0), d.get("n_crit", 9.0), d.get("notes", ""))


# ================================================================================================= backends
def backends() -> dict:
    """Which backends this machine has: {'neuralfoil': version or None, 'xfoil': path or None}."""
    try:
        import neuralfoil
        nf = getattr(neuralfoil, "__version__", None) or getattr(neuralfoil, "version", "installed")
    except ImportError:
        nf = None
    return {"neuralfoil": nf, "xfoil": shutil.which(os.environ.get("XFOIL", "xfoil"))}


def _neuralfoil(section: Section, alphas, Re, mach, n_crit, model_size) -> Polar:
    import neuralfoil as nf
    xy = section.coordinates()
    r = nf.get_aero_from_coordinates(xy, alpha=np.asarray(alphas, float), Re=float(Re), n_crit=float(n_crit), model_size=model_size)
    return Polar(section.name, float(Re), np.asarray(alphas, float), np.asarray(r["CL"], float), np.asarray(r["CD"], float),
                 np.asarray(r["CM"], float), np.asarray(r["analysis_confidence"], float), f"neuralfoil ({model_size})", mach, n_crit,
                 notes="a surrogate of XFOIL: read the confidence; low past the stall and for unusual shapes")


def _xfoil(section: Section, alphas, Re, mach, n_crit, workdir, timeout) -> tuple[Polar, CommandRecord]:
    exe = shutil.which(os.environ.get("XFOIL", "xfoil"))
    if exe is None:
        raise FileNotFoundError("xfoil is not on the PATH (set XFOIL=/path/to/xfoil)")
    alphas = np.asarray(alphas, float)
    tmp = Path(tempfile.mkdtemp(prefix="zephyr_", dir=workdir)) if workdir else Path(tempfile.mkdtemp(prefix="zephyr_"))
    dat = section.write_dat(tmp / "section.dat", n=100)
    pol = tmp / "polar.txt"
    cmds = ["PLOP", "G F", "", f"LOAD {dat.name}", "PANE", "OPER", f"VISC {Re:g}", f"MACH {mach:g}", "VPAR", f"N {n_crit:g}", "",
            f"ITER {XFOIL_ITER}", "PACC", pol.name, ""]
    cmds += [f"ALFA {a:g}" for a in alphas]                   # one point at a time: a failed point does not stop the sweep
    cmds += ["PACC", "", "QUIT"]
    script = "\n".join(cmds) + "\n"
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat()
    try:
        proc = subprocess.run([exe], input=script, capture_output=True, text=True, cwd=tmp, timeout=timeout)
        rec = CommandRecord([exe], str(tmp), proc.returncode, time.time() - t0, proc.stdout[-20000:], proc.stderr[-5000:], started)
    except subprocess.TimeoutExpired as exc:
        rec = CommandRecord([exe], str(tmp), None, time.time() - t0, (exc.stdout or "")[-20000:] if isinstance(exc.stdout, str) else "",
                            "", started, error=f"xfoil timed out after {timeout} s")
        raise RuntimeError(rec.error)
    if not pol.exists():
        raise RuntimeError("xfoil wrote no polar file; its output:\n" + proc.stdout[-2000:])
    rows = []
    for line in pol.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            try:
                rows.append([float(v) for v in parts[:5]])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError("xfoil converged at no point of the sweep")
    arr = np.array(rows)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    p = Polar(section.name, float(Re), arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 4], np.ones(len(arr)), "xfoil", mach, n_crit,
              notes=f"{len(arr)} of {len(alphas)} points converged" if len(arr) < len(alphas) else "")
    return p, rec


def analyze(section: Section, alphas=None, Re: float = 1e6, *, method: str = "auto", mach: float = 0.0, n_crit: float = 9.0,
            model_size: str = NEURALFOIL_MODEL, workdir=None, timeout: float = 120.0) -> Polar:
    """The polar of ``section`` over ``alphas`` [deg] at ``Re``. ``method``: ``neuralfoil``, ``xfoil`` or ``auto``
    (NeuralFoil when installed, else XFOIL). Raises when the chosen backend is missing; ``backends()`` says what is
    there. ``n_crit`` is the e^N transition criterion (9: a clean tunnel; 4-6: a turbulent, built-up small UAV wing)."""
    alphas = np.arange(-6.0, 20.0, 1.0) if alphas is None else np.asarray(alphas, float)
    have = backends()
    if method == "auto":
        method = "neuralfoil" if have["neuralfoil"] else "xfoil" if have["xfoil"] else None
        if method is None:
            raise RuntimeError("no section backend: pip install neuralfoil, or put xfoil on the PATH")
    if method == "neuralfoil":
        if not have["neuralfoil"]:
            raise RuntimeError("NeuralFoil is not installed (pip install neuralfoil)")
        return _neuralfoil(section, alphas, Re, mach, n_crit, model_size)
    if method == "xfoil":
        p, _ = _xfoil(section, alphas, Re, mach, n_crit, workdir, timeout)
        return p
    raise ValueError(f"method is 'auto', 'neuralfoil' or 'xfoil', not {method!r}")


def analyze_result(section: Section, alphas=None, Re: float = 1e6, *, cl_design: float | None = None, out: str | Path | None = None,
                   **kw) -> Result:
    """``analyze`` as a Result (the common shape): the summary as metrics, the polar JSON as an artifact when ``out`` is
    given, the XFOIL command recorded."""
    t0 = time.time()
    res = Result(kind="zephyr.polar", metadata={"section": section.to_dict(), "Re": Re, **{k: v for k, v in kw.items() if k != "workdir"}})
    try:
        method = kw.pop("method", "auto")
        if method == "xfoil" or (method == "auto" and not backends()["neuralfoil"]):
            p, rec = _xfoil(section, np.arange(-6.0, 20.0, 1.0) if alphas is None else np.asarray(alphas, float), Re,
                            kw.get("mach", 0.0), kw.get("n_crit", 9.0), kw.get("workdir"), kw.get("timeout", 120.0))
            res.execution.append(rec)
        else:
            p = analyze(section, alphas, Re, method=method, **kw)
    except Exception as exc:
        res.duration_s = time.time() - t0
        return res.fail(f"{type(exc).__name__}: {exc}")
    res.metrics = p.summary(cl_design)
    if out:
        from .export import save_polars
        res.artifacts["polar"] = save_polars(out, [p])
    res.duration_s = time.time() - t0
    return res


# ================================================================================================= flaps
def flap_sweep(section: Section, cf: float, Re: float, *, alpha_deg: float = 2.0, deltas=(-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0),
               method: str = "auto", **kw) -> dict:
    """A plain flap of chord fraction ``cf`` at a fixed angle of attack: cl, cd, cm per deflection; the slopes per
    radian fitted on ±5°; the effectiveness τ = (∂cl/∂δ)/(∂cl/∂α) against thin-airfoil theory's; the centre of the
    flap's lift x_cp/c = 1/4 − (∂cm/∂δ)/(∂cl/∂δ); the drag of a deflection above the undeflected section."""
    rows = []
    for d in deltas:
        p = analyze(section.flapped(cf, d), [alpha_deg], Re, method=method, **kw)
        rows.append((p.cl[0], p.cd[0], p.cm[0], p.confidence[0]))
    cl, cd, cm, conf = (np.array(v) for v in zip(*rows))
    base = analyze(section, [alpha_deg - 2.0, alpha_deg, alpha_deg + 2.0], Re, method=method, **kw)
    cla = float((base.cl[2] - base.cl[0]) / math.radians(4.0))
    dd = np.radians(np.asarray(deltas, float))
    k = np.abs(np.asarray(deltas)) <= 5.0
    dcl = float(np.polyfit(dd[k], cl[k], 1)[0])
    dcm = float(np.polyfit(dd[k], cm[k], 1)[0])
    i0 = list(deltas).index(0.0) if 0.0 in deltas else int(np.argmin(np.abs(deltas)))
    return {"section": section.name, "cf": cf, "Re": Re, "alpha_deg": alpha_deg, "deltas_deg": list(deltas), "cl": cl.tolist(), "cd": cd.tolist(),
            "cm": cm.tolist(), "confidence": conf.tolist(), "dcl_ddelta_per_rad": dcl, "dcm_ddelta_per_rad": dcm, "cl_alpha_per_rad": cla,
            "tau": dcl / cla, "tau_thin": flap_effectiveness(cf), "x_cp_flap": 0.25 - dcm / dcl,
            "cd_increment": (cd - cd[i0]).tolist(), "source": base.source}


def check_against_thin_airfoil(polar: Polar, section: Section) -> dict:
    """The polar's zero-lift angle and zero-lift moment against thin-airfoil theory's (the thickness and the viscosity
    move them a little: a few tenths of a degree, a few thousandths)."""
    ta = thin_airfoil(section)
    s = polar.summary()
    return {"alpha0_deg": s["alpha0_deg"], "alpha0_thin_deg": ta["alpha0_deg"], "d_alpha0_deg": s["alpha0_deg"] - ta["alpha0_deg"],
            "cm0": s["cm0"], "cm_c4_thin": ta["cm_c4"], "d_cm0": s["cm0"] - ta["cm_c4"],
            "cl_alpha_per_rad": s["cl_alpha_per_rad"], "cl_alpha_ratio_to_2pi": s["cl_alpha_per_rad"] / (2 * math.pi)}
