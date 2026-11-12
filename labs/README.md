# labs/ — one per domain, each running the same loop

```
libs/ tools ──► notebooks/ ──► designs/ ──► sims/ (MuJoCo) ──► campaigns/ (data, optimisation) ──► export ──► Piccolo
                      │                                                                                     (tools/export.py)
                      └──► exp/ (limbo: what did not make it; nothing imports it)
```

| lab | what | machines |
|---|---|---|
| [aero](aero/README.md) | what flies: fixed-wing, VTOL (Kestrel), multirotors, propulsion, the races | Nisus, Nisus+, Falco, Merlin, Peregrine, Aguya, Velutina, Drongo, Kestrel |
| [land](land/README.md) | what rolls and tracks | Onager (Sentinel, Atlas, Manus, Sweeper), Pekari, Perseus, Rover |
| [legged](legged/README.md) | what walks | Robot dog, Myropods (Cleopatra, Persephone, Apheloria), the benchmarks |
| [marine](marine/README.md) | what floats and dives | Survey boat, Submarine |
| [space](space/README.md) | what orbits: LEO servicing, rendezvous, capture | Auseklis (debris-removal servicer, exp/) |
| [studio](studio/README.md) | the tools themselves: tutorials, the workbench, AI design | — |

The rules (`python tools/labs.py check`, in smoke): a lab imports `libs/` and itself, never another lab; nothing
imports `exp/`. What two labs need moves to `libs/labkit` (`python tools/lab.py promote <lab> <module> --to-libs`).
Labs exchange machines only as files written by Yax.

```bash
python tools/lab.py list                 # every machine's stage
python tools/lab.py test land            # a lab's tests, mocks on
python tools/lab.py run land sims        # its MuJoCo scenarios, mocks on
python tools/lab.py new-exp aero falcon  # a new limbo study with the bootstrap
python tools/lab.py readme               # regenerate the lab READMEs
./user_tests.sh                          # everything, mocks only
```
