"""Constants and units. SI everywhere; helpers for the units catalogues and datasheets use."""
import math

G = 9.80665                       # standard gravity [m/s²] (ISO 80000-3)
G_VEGETA = 9.81                   # what most of Vegeta's machine models used; kept where a moved model reproduces it
RHO0 = 1.225                      # ISA sea-level air density [kg/m³]
T0 = 288.15                       # ISA sea-level temperature [K]
P0 = 101325.0                     # ISA sea-level pressure [Pa]
R_AIR = 287.053                   # specific gas constant of dry air [J/(kg K)]
GAMMA_AIR = 1.4
NU_AIR = 1.46e-5                  # kinematic viscosity at sea level [m²/s]

MM = 1e-3
G_PER_KG = 1e3
RPM = 2 * math.pi / 60            # rad/s per rpm
INCH = 0.0254


def rpm_to_rad_s(rpm: float) -> float:
    return rpm * RPM


def rad_s_to_rpm(w: float) -> float:
    return w / RPM
