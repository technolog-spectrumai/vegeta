# external/ — prepared heavy-tool runs (nothing here runs by itself)

| script | tool | what it prepares | needs |
|---|---|---|---|
| `bus_fea.py` | Gmsh + CalculiX through Talos | bus shell: STEP, quasi-static launch cases (axial, lateral), first modes | `ccx`, Gmsh (`./install_local.sh`) |
| `mujoco_capture.py` | MuJoCo | servicer + tumbling debris, soft weld capture, post-capture rates and constraint force | `pip install mujoco` |

Both write their inputs and print the plan without `--run`. With `--run` they call the real solver; with
`VEGETA_DEBUG=1` (the lab's tests) Talos is mocked and MuJoCo is not imported. Numbers from a mocked run mean nothing.

**These simulations need external verification** (the models here are preliminary): the bus FEA treats the honeycomb
panels as solid aluminium shell of equivalent density; launch loads are placeholders until a launcher user guide is
chosen; the MuJoCo capture is a rigid-body weld, not a contact model of the real gripper.
