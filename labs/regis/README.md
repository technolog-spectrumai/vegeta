# Regis monsters

Static 3D models for [Regis](https://github.com/technolog-spectrumai/regis)' real-time combat screen. Regis shows each
monster kind as a stationary, deliberately robotic model; the models are Vegeta CAD designs exported here, offline.
Regis packages the GLBs and never needs Vegeta at runtime.

| Regis monster | Vegeta design | |
|---|---|---|
| ghoul | `labs/legged/designs/humanoid.py:Humanoid` | the humanoid walker |
| wraith | `labs/regis/designs/balloon.py:Balloon` | a tethered survey balloon (new, deliberately simple) |
| vampire | `labs/aero/designs/quadcopter_assembly.py:Quadcopter` | the X-frame quadcopter |
| wyvern | `labs/aero/designs/drako.py:Drako` | the Drako heavy-lift hexacopter |
| hound | `labs/legged/designs/robot_dog.py:RobotDog` | the Cerberus robot dog |
| leshy | `labs/land/designs/onager.py:OnagerSentinel` | the Onager Sentinel land robot |

```bash
python labs/regis/export_monsters.py            # all six into exports/regis/ (+ MANIFEST.json)
python labs/regis/export_monsters.py wraith     # one
```

The design's parts (`vegeta.fidia.runner.split_parts`) are tessellated with `vegeta.labkit.visual` and simplified
together to at most 6000 triangles (the budget of the `.yx2` CAD meshes), then written by `vegeta.fidia.export.export_scene`:
glTF 2.0 binary, Y up, metres, one node and one base-colour material per part, rest pose (no joints or motions).
Parts without a colour of their own take the monster's colour (Regis' map-marker colours).

Needs CadQuery and trimesh (as the rest of the labs). Licence: the repository's (internal use).
