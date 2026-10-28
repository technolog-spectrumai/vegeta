"""Aeroacoustics post-processing (outside the solver): the Ffowcs Williams–Hawkings analogy (Farassat 1A),
Gutin's steady-loading tones for comparison, exact fields for verification and rigid-rotation kinematics."""
from .fwh import SourceSurface, fwh, gutin, harmonics
from .sources import acoustic_field, dipole_potential, monopole_potential, rigid_rotation, rotate_x, sphere_panels

__all__ = ["SourceSurface", "acoustic_field", "dipole_potential", "fwh", "gutin", "harmonics",
           "monopole_potential", "rigid_rotation", "rotate_x", "sphere_panels"]
