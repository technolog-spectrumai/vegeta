"""Output: VTK ImageData (built-in writer, optional PyVista) and plots (optional matplotlib).
Fields, metadata and checkpoints are written by :meth:`mancha.Simulation.save` and ``checkpoint``."""
from .vtk import to_pyvista, write_vti

__all__ = ["to_pyvista", "write_vti"]
