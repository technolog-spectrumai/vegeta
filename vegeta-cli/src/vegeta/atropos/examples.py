"""Example settings: an illustration, not a validated profile. A diode laser's cut depends on the balsa's density,
the focus and the air assist: cut a test strip and measure the kerf before trusting any number here."""
from .settings import CutSettings, Machine, Operation, Sheet

D1_PRO_10W = Machine(name="xTool D1 Pro 10 W (example)", bed_width_mm=432.0, bed_height_mm=406.0,
                     travel_feed_mm_min=6000.0, s_max=1000)

# a 100 x 1000 mm balsa strip cut to the bed's length; the grain runs along the strip (X on the bed)
BALSA_400x100_1_5MM = Sheet("balsa 400 x 100 x 1.5 (example)", 400.0, 100.0, 1.5, 0.16, grain="x", margin_mm=4.0)
BALSA_400x100_2MM = Sheet("balsa 400 x 100 x 2 (example)", 400.0, 100.0, 2.0, 0.16, grain="x", margin_mm=4.0)
BALSA_400x100_3MM = Sheet("balsa 400 x 100 x 3 (example)", 400.0, 100.0, 3.0, 0.16, grain="x", margin_mm=4.0)
PLY_400x100_2MM = Sheet("birch plywood 400 x 100 x 2 (example)", 400.0, 100.0, 2.0, 0.68, grain="x", margin_mm=4.0)


def _ops(cut_feed: float, cut_passes: int, kerf: float) -> dict:
    return {
        "engrave": Operation("engrave", power_pct=12.0, feed_mm_min=3000.0, passes=1, kerf_mm=0.0, order=0),
        "score": Operation("score", power_pct=25.0, feed_mm_min=1500.0, passes=1, kerf_mm=0.0, order=1),
        "cut": Operation("cut", power_pct=100.0, feed_mm_min=cut_feed, passes=cut_passes, kerf_mm=kerf, order=2),
    }


SOURCE = "illustrative values for a 10 W diode on medium balsa, not measured on a machine"

D1_PRO_BALSA_1_5MM = CutSettings(BALSA_400x100_1_5MM, D1_PRO_10W, _ops(900.0, 1, 0.12), spacing_mm=3.0,
                                 name="D1 Pro, balsa 1.5 mm (example)", source=SOURCE)
D1_PRO_BALSA_2MM = CutSettings(BALSA_400x100_2MM, D1_PRO_10W, _ops(600.0, 2, 0.15), spacing_mm=3.0,
                               name="D1 Pro, balsa 2 mm (example)", source=SOURCE)
D1_PRO_BALSA_3MM = CutSettings(BALSA_400x100_3MM, D1_PRO_10W, _ops(400.0, 3, 0.18), spacing_mm=3.0,
                               name="D1 Pro, balsa 3 mm (example)", source=SOURCE)
D1_PRO_PLY_2MM = CutSettings(PLY_400x100_2MM, D1_PRO_10W, _ops(240.0, 3, 0.15), spacing_mm=3.0,
                             name="D1 Pro, birch ply 2 mm (example)", source=SOURCE)
