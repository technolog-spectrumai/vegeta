"""Calydon's CAD (Dedalus/CadQuery): the family's CAD (``pekari_2m._FamilyCAD``) on Calydon's chassis — the heavy
6-DOF arm and the auxiliary arm on the front corners, the deck between them, four swing-out stabilizers.

``part`` as ``pekari_2m.Pekari2M``; ``stabilizers``: ``stowed`` (legs folded against the hull sides), ``deployed``
(pads on the ground) or ``none``.
"""
from pekari_2m import _make
from calydon_robot import CHASSIS

Calydon = _make("Calydon", "calydon_robot:CALYDON", CHASSIS,
                "Calydon: the heavy Pekari derivative (the first Pumba) — reinforced, wider chassis, a 6-DOF heavy arm "
                "and an auxiliary arm, four stabilizers, the equipment deck.")
