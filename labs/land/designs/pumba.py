"""Pumba's CAD (Dedalus/CadQuery): the family's CAD (``pekari_2m._FamilyCAD``) on the light Pumba's chassis — the
6-DOF main arm and Pekari-2M's arm on the front corners, the deck between them, two swing-out front stabilizers.
The tracks are drawn as Pekari's link belt: the envelope of Pumba's rubber band track. Calydon: ``calydon.Calydon``.

``part`` as ``pekari_2m.Pekari2M``; ``stabilizers``: ``stowed``, ``deployed`` or ``none``.
"""
from pekari_2m import _make
from pumba_robot import CHASSIS

Pumba = _make("Pumba", "pumba_robot:PUMBA", CHASSIS,
              "Pumba: the light heavy derivative — a WILK-class chassis, a 6-DOF main arm and Pekari-2M's arm, two "
              "front stabilizers, the equipment deck.")
