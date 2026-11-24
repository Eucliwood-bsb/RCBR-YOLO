#!/usr/bin/env python3
"""Evaluate one RCBR-YOLO arm: export val/test, select val thresholds, freeze them on test.

Only `--splits val` runs during model selection; `--splits val,test` is used once the plan is frozen.

Writes runs/arms/<tag>/eval[_val_only].json and prints R@0.05, FPPI, AP.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate_operating_points import (BUDGETS, FIXED_CONF, class_aware_nms, coco_metrics,  # noqa: E402
                                       evaluate_file, load_gt, point, sweep)

import paths  # noqa: E402

BUD = tuple(str(b) for b in BUDGETS)
KEYS = ("threshold", "fppi", "precision", "recall", "f2", "tp", "fp", "fn",
        "empty_alarm_rate", "empty_fp_per_img", "defect_fp_per_img")


def export(weights: Path, out: Path, split: str) -> None:
    if out.is_file() and out.stat().st_size > 0:
        print(f"[skip export] {split} exists", flush=True)
        return
    cmd = [sys.executable, str(TASK / "export_preds.py"), "--weights", str(weights),
           "--split", split, "--out", str(out)]
    r = subprocess.run(cmd, cwd=str(TASK), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"export {split} failed: {r.stderr[-800:]}")
    print(r.stdout.strip().splitlines()[-1], flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--splits", default="val")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    run = paths.RUNS / "arms" / args.tag
    weights = run / "weights" / "last.pt"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    preds = run / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        export(weights, preds / f"{s}.json", s)

    val = evaluate_file(preds / "val.json", args.tag, "val")
    out = {"tag": args.tag, "weights": str(weights), "val": {
        "map_nms07": val["map_nms07"], "fixed_conf_0.25": val["fixed_conf_0.25"],
        "budgets": {b: val["budgets"][b] for b in BUD}}}
    if "test" in splits:
        _, gt_all, img_ids = load_gt("test")
        raw = json.loads((preds / "test.json").read_text(encoding="utf-8"))
        tn = class_aware_nms(raw, 0.7)
        ordered, tp = sweep(tn, gt_all, img_ids)
        n_gt = sum(len(v) for v in gt_all.values())
        test_out = {}
        for b in BUD:
            q = val["budgets"][b]
            if q is None:
                test_out[b] = None
                continue
            keep = [p for p in tn if p["score"] >= q["threshold"]]
            pt = point(ordered, tp, len(keep), len(img_ids), n_gt, gt_all, img_ids)
            pt["val_threshold"] = q["threshold"]
            pt["exceeds_budget"] = pt["fppi"] > float(b) + 1e-12
            test_out[b] = {k: pt[k] for k in KEYS + ("val_threshold", "exceeds_budget")}
        keep25 = [p for p in tn if p["score"] >= FIXED_CONF]
        q25 = point(ordered, tp, len(keep25), len(img_ids), n_gt, gt_all, img_ids)
        out["test"] = {"map_nms07": coco_metrics(tn, "test"),
                       "fixed_conf_0.25": {k: q25[k] for k in
                                           ("fppi", "precision", "recall", "f2", "tp", "fp", "fn")},
                       "budgets": test_out}

    suffix = args.out_suffix or ("" if "test" in splits else "_val_only")
    path = run / f"eval{suffix}.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    v = out["val"]
    line = (f"{args.tag:18s} val R@0.05 {v['budgets']['0.05']['recall']:.4f} "
            f"AP {v['map_nms07']['AP']:.4f}")
    if "test" in out:
        line += (f" | test R@0.05 {out['test']['budgets']['0.05']['recall']:.4f} "
                 f"FPPI {out['test']['budgets']['0.05']['fppi']:.3f} AP {out['test']['map_nms07']['AP']:.4f}")
    print(line, flush=True)
    print("written", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
