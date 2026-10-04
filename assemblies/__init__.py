"""Geometry only: how components are arranged into machines (a quadcopter, an aircraft, a boat, a submarine).

An assembly is a Dedalus ``Design`` whose ``parts(p)`` places components (``components/``) and returns them by name;
``build`` joins them into one compound, so an assembly generates, exports and loads by spec like any design. No
analysis: masses, loads, FEA and CFD stay in the notebooks. A notebook may use an assembly, a component, or its own
geometry::

    import sys; sys.path.insert(0, "..")
    from assemblies.quadcopter import Quadcopter
    quad = Quadcopter()
    parts = quad.generate_parts(wheelbase=280)        # {"frame": Shape, "motor_1": Shape, ..., "propeller_4": Shape}
    geometry = quad.generate(wheelbase=280)            # one compound (dedalus.Geometry)
"""
from .base import Assembly, component_parameters, pick

__all__ = ["Assembly", "component_parameters", "pick"]
