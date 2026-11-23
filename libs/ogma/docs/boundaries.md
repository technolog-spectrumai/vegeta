# Who does what

| | Does | Never |
|---|---|---|
| **Yax** | defines the design package (STL + `yax.json`); writes, loads and checks it; mass properties of a mesh | imports numpy or any part type; converts to MJCF, SDF, glTF |
| **Ogma** (`og`) | standard-part physics as data + behaviour: dataclasses, pure functions, JSON catalogues; reads and writes its models inside Yax packages | imports Vegeta, Piccolo, Chiron, MuJoCo, a CAD kernel or pandas at import time; holds a machine's parts list; designs (computing a propeller map is Boreas's job in Vegeta — Ogma evaluates the map) |
| **Chiron** | runs Ogma models in MuJoCo (MJCF, contacts, servos, terrains, metrics) | owns part physics |
| **Vegeta** | computes the numbers (CAD, BEMT, FEA) and fills Ogma models while a machine is designed | is imported by Piccolo |
| **Piccolo** | develops hatched machines' behaviour on Ogma models from Yax packages | imports Vegeta |
