"""What the aero studies repeated (``from aero_kit import plain``): helpers that need nothing from a notebook."""
from __future__ import annotations

import numpy as np


def plain(x):
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def lower_skins(step):
    from vegeta import talos
    info = talos.inspect_step(step, units="mm-N-MPa")
    skins = [s for s in info.surfaces if s.kind == "BSpline surface" and s.area > 5e4]
    return [min((s for s in skins if (s.centroid[1] > 0) == right), key=lambda s: s.centroid[2]).tag
            for right in (False, True)]


def save_table(df, name, out):
    """``df`` to ``out/tables/<name>.csv``; returns ``df`` (to display it in the same line)."""
    df.to_csv(out / "tables" / f"{name}.csv")
    return df


def prop_of(B, blade, hub):
    """Boreas' propeller of notebook 25's blade (``blade``: the ``BLADE`` dict) with ``B`` blades on hub ``hub``."""
    from vegeta import boreas
    return boreas.Propeller.from_pitch(f"10x6, {B} blades", blade["diameter_mm"] / 1000, blade["pitch_mm"] / 1000, blades=B,
                                       chord_root_m=blade["chord_root_mm"] / 1000, chord_max_m=blade["chord_max_mm"] / 1000,
                                       chord_tip_m=blade["chord_tip_mm"] / 1000, hub_radius_m=hub["hub_diameter"] / 2000)


def cad_kw(B, blade, hub):
    """``PropellerCAD().generate`` keywords for that blade with ``B`` blades."""
    return dict(diameter=blade["diameter_mm"], pitch=blade["pitch_mm"], blades=B, chord_root=blade["chord_root_mm"],
                chord_max=blade["chord_max_mm"], chord_tip=blade["chord_tip_mm"], thickness=blade["thickness"], camber=blade["camber"],
                stations=8, **hub)


def pod_p(lay, pod, hub, pod_gap):
    """The pod's parameters for a layout (``tractor``/``pusher``)."""
    return dict(pod, layout=lay, hub_height=hub["hub_height"], hub_diameter=hub["hub_diameter"], gap=pod_gap[lay])
