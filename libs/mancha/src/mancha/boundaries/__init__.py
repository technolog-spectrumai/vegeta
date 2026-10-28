"""Boundary conditions: face conditions (periodic, wall, velocity, pressure) and voxel solids."""
from .conditions import FACES, BoundaryConditions, FaceCondition, Kind, LinkGroup, link_groups, make_condition
from .solids import box, sphere, validate_mask
from .sponge import Sponge

__all__ = ["FACES", "BoundaryConditions", "Sponge", "FaceCondition", "Kind", "LinkGroup", "box", "link_groups",
           "make_condition", "sphere", "validate_mask"]
