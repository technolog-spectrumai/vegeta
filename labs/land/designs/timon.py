"""Timon's CAD (Dedalus/CadQuery): the family's CAD (``pekari_2m._FamilyCAD``) on Timon's chassis — the main arm
(front left) and the second hand (front right), the deck with the module cradle and tool rack behind them, the rear
earth anchor (``stabilizers``: stowed / deployed).
"""
from pekari_2m import _make
from timon_robot import CHASSIS

Timon = _make("Timon", "timon_robot:TIMON", CHASSIS,
              "Timon: the fleet's servicing rover — battery module swaps, tool changes, recovery; sized for the job.")
