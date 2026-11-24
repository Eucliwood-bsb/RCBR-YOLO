#!/usr/bin/env python3
"""Fresh-process reload check for the wrapped CR model (called by audit_integration.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))

import paths  # noqa: E402

META = json.loads((paths.STATE / "reload" / "meta.json").read_text(encoding="utf-8"))
sys.path.insert(0, META["src"])

from cr_integration import insert_context, unwrap_predictions  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402


def main() -> int:
    art = Path(META["art"])
    x = torch.load(art / "x.pt", map_location="cuda")
    y_ref = torch.load(art / "y.pt", map_location="cuda")
    result = {}

    # path 1: rebuild + state_dict
    yaml = json.loads((art / "yaml.json").read_text(encoding="utf-8"))
    m = DetectionModel(yaml, nc=META["nc"], verbose=False)
    insert_context(m, levels=tuple(META["levels"]), radii=tuple(META["radii"]))
    sd = torch.load(art / "cr_state.pt", map_location="cpu")
    missing, unexpected = m.load_state_dict(sd, strict=True)
    m = m.cuda().eval()
    with torch.no_grad():
        y, _ = unwrap_predictions(m(x))
    d = (y.float() - y_ref.float()).abs()
    result["state_dict"] = (f"rebuild+state_dict OK: max|Δy|={float(d.max()):.3e}, "
                            f"missing={len(missing)}, unexpected={len(unexpected)}")

    # path 2: full pickle (needs cr_integration importable, already on sys.path)
    try:
        m2 = torch.load(art / "cr_full.pt", map_location="cuda", weights_only=False).eval()
        with torch.no_grad():
            y2, _ = unwrap_predictions(m2(x))
        d2 = (y2.float() - y_ref.float()).abs()
        result["full_pickle"] = f"OK: max|Δy|={float(d2.max()):.3e}"
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        result["full_pickle"] = f"FAILED: {type(exc).__name__}: {exc}"

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
