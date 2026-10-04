"""Geometry only: the machines' parametric CAD designs (Dedalus ``Design`` classes), shared by the notebooks.

A component defines geometry and nothing else: parameters, ``build``, the parts it can make, and the geometric
helpers that build needs. Analysis (masses, loads, FEA and CFD set-ups, flight, robots) stays in the notebooks, where
it is seen and run. A notebook may use a component or define its own geometry; nothing here forces either.

From ``notebooks/`` (the repository root on ``sys.path``)::

    import sys; sys.path.insert(0, "..")
    from components.quad_frame import QuadFrame
    frame = QuadFrame().generate(arm_width=14)
    # or by spec, as notebooks load designs today:
    from vegeta import dedalus
    frame_design = dedalus.load_design("components.quad_frame:QuadFrame")

The designs are copies of ``notebooks/designs/`` (which stays as it is: the notebooks and ``benchmark/`` use it), cut to
their geometry, with package-relative imports; ``components/tests`` checks that each builds the same shapes as the
original and imports no analysis.
"""
