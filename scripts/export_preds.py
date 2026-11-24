#!/usr/bin/env python3
"""Export predictions for one split with the evaluation protocol.

Uses Ultralytics ``DetectionPredictor`` with the recorded export settings
(``imgsz=640, conf=0.001, iou=0.7, max_det=300, rect=True``) and writes a COCO
prediction list with one-based ``category_id`` (class index + 1), matching the
annotation files.

Wrapped checkpoints pickle ``CtxWrap`` from ``cr_integration``; ``src`` is put on
``sys.path`` before the checkpoint is loaded so a stored model can be unpickled
in a fresh process.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
sys.path.insert(0, str(ROOT / "src"))

import paths  # noqa: E402

SPLIT_IMAGES = {
    "train": paths.TRAIN_IMAGES,
    "val": paths.VAL_IMAGES,
    "test": paths.TEST_IMAGES,
}


def load_annotation(ann_dir: Path, split: str) -> dict:
    path = Path(ann_dir) / f"instances_{split}.json"
    paths.require(path, f"COCO annotation file for split '{split}'")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--images", default="", help="image directory (default: RCBR_*_IMAGES)")
    ap.add_argument("--ann-dir", default="", help="COCO annotation directory (default: RCBR_ANN_DIR)")
    args = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionPredictor

    ann_dir = Path(args.ann_dir) if args.ann_dir else paths.ANN_DIR
    img_dir = Path(args.images) if args.images else SPLIT_IMAGES[args.split]
    paths.require(img_dir, f"{args.split} image directory")

    gt = load_annotation(ann_dir, args.split)
    name2id = {im["file_name"]: im["id"] for im in gt["images"]}
    sizes = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}

    model = YOLO(args.weights).model.float().eval()
    pred = DetectionPredictor(
        overrides=dict(
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            save=False,
            verbose=False,
            task="detect",
            mode="predict",
            rect=True,
        )
    )
    pred.setup_model(model=model, verbose=False)

    rows, n_img = [], 0
    for r in pred(source=str(img_dir), stream=True):
        n_img += 1
        iid = name2id[Path(r.path).name]
        W, H = sizes[iid]
        if r.boxes is None:
            continue
        for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                           r.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = [float(v) for v in b]
            x1, y1 = max(0.0, min(x1, W - 1e-3)), max(0.0, min(y1, H - 1e-3))
            x2, y2 = max(x1 + 1e-3, min(x2, W)), max(y1 + 1e-3, min(y2, H))
            rows.append({"image_id": int(iid), "category_id": int(c) + 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows), encoding="utf-8")
    print(f"{args.split}: {n_img} images, {len(rows)} preds, imgsz={args.imgsz}, "
          f"conf={args.conf}, max_det={args.max_det} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
