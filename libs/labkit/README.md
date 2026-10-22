# vegeta.labkit — what every lab uses

The labs (`labs/aero`, `labs/land`, `labs/legged`, `labs/marine`, `labs/studio`) import the libraries and themselves,
never each other. What two labs need lives here:

| module | what |
|---|---|
| `bootstrap` | `lab("aero", DEBUG)`: the lab's folders on `sys.path`, the mocks (`vegeta.mock`) — a notebook's first cell |
| `machines` | the registry of exportable machines: each lab's `designs/machines.py` registers its robots; `tools/export.py` reads it |
| `control` | mission building blocks: `Phase` (one step of a mission), `smooth` (smoothstep) |
| `movie` | `end_card` (a movie's last frame with the outcome on it) |
| `kinematics` | `ik_two_link_planar` (a two-link leg in its plane) |
| `propeller`, `motor`, `assembly` | the generic CAD parts (Dedalus designs) and the assembly base class |

A module moves here (`tools/lab.py promote`) when a second lab needs it; until then it stays in its lab.
