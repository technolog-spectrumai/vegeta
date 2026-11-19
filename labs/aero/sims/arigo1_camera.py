"""ARIGO-1: the nose tractor's propeller in the camera's view, and what synchronised capture / frame fusion / shutter
type leave of it (``designs/arigo1_camera.py``). Seconds, one core. Runs the tractor (chin strut), tractor_lowcam
(through the disc) and the pusher (reference) on the hybrid wing.

Writes ``labs/aero/sims/output/arigo1_camera.yaml`` (PASTE BACK) and ``arigo1_camera.png``.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))
import arigo1 as a1  # noqa: E402
import arigo1_camera as cam  # noqa: E402

OUT = ROOT / "labs" / "aero" / "sims" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wing", default="hybrid", choices=tuple(a1.VARIANTS))
    ap.add_argument("--layouts", nargs="*", default=["tractor", "tractor_lowcam", "pusher"], choices=a1.PROPULSION)
    ap.add_argument("--lowcam-drops", type=float, nargs="*", default=[40.0, 55.0, 80.0], help="extra tractor_lowcam camera heights [mm]")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for lay in a.layouts:
        res.append(cam.analyse(a1.variant(a.wing, lay), f"{a.wing}_{lay}"))
        print(f"  {lay}: behind disc {res[-1]['behind_disc']}  synced {res[-1].get('synced_global_best_phase')}  fusion {res[-1].get('two_frame_fusion_residual')}")
    for d in a.lowcam_drops:
        res.append(cam.analyse(a1.variant(a.wing, "tractor_lowcam", lowcam_drop=d), f"{a.wing}_tractor_lowcam_{d:.0f}mm"))
    cam.picture(res, OUT / "arigo1_camera.png")
    doc = {"aircraft": "ARIGO-1", "note": cam.__doc__.split("\n\n")[1].replace("\n", " "), "results": [{k: v for k, v in r.items() if not k.startswith("_")} for r in res]}
    (OUT / "arigo1_camera.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(doc, default=float)), sort_keys=False, width=140))
    print(f"-> {OUT / 'arigo1_camera.yaml'}, {OUT / 'arigo1_camera.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
