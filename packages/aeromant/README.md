# Aeromant

Standalone external aerodynamics on OpenFOAM. Aeromant never invents a CFD case: it copies a
known-good template case, inserts the STL and explicit values (velocity, viscosity, density,
reference area/length, moment centre), runs the template's pipeline and extracts Cd/Cl/Cm.
The complete case and all logs are kept. See `../../docs/aeromant.md`.
