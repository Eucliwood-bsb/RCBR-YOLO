#!/usr/bin/env python3
"""Unified operating-point evaluation for the ShoePrint local-image detection task.

Implements the project evaluation protocol (see `docs/PROTOCOL.md`) as far as it
applies to a pinhole-only detection subset:

  * FPPI  = (#FP over ALL evaluated local images) / (#evaluated local images),
            denominator includes the empty-target images           (§4.1)
  * match : same class, one-to-one, IoU >= 0.50, predictions in descending confidence,
            each prediction -> unmatched same-class GT with the largest IoU (§3.2)
  * TP/FP/FN, P = TP/(TP+FP), R = TP/(TP+FN), F2 = 5PR/(4P+R)      (§3.1, §6.1)
  * threshold per budget: among thresholds with FPPI <= b choose the one with the highest
            recall; ties -> lower FPPI; still tied -> higher threshold (deterministic)  (§4.3)
  * fixed supplementary point: confidence 0.25                      (§6.3)
  * mAP50 / mAP50-95 on the same NMS'd predictions, not truncated at 0.25  (§7)

Conventions that the document does not fix explicitly are stated here and in
reports/evaluation_protocol.md:
  * post-processing: one class-aware NMS at IoU 0.7 (the historical export already applied one;
    re-applying is a no-op in practice -- checked and reported);
  * a "threshold" is a real prediction score (or +inf for the empty prediction set), so every
    reported operating point is implementable -- no interpolation;
  * nGT == 0 -> recall = None ("not applicable"), precision/recall denominators 0 -> None;
  * F2 is reported only when both P and R are defined.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import paths

TASK = paths.ROOT
ANN_DIR = paths.ANN_DIR
BUDGETS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
FIXED_CONF = 0.25
MATCH_IOU = 0.5
NMS_IOU = 0.7


# ---------------------------------------------------------------- basic pieces
def iou_xywh(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = aw * ah + bw * bh - inter
    return 0.0 if u <= 0 else inter / u


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if u <= 0 else inter / u


def class_aware_nms(preds, iou_thr: float = NMS_IOU):
    """greedy per (image, class) NMS, score descending -- same rule as the historical protocol."""
    groups = {}
    for p in preds:
        groups.setdefault((p["image_id"], p["category_id"]), []).append(p)
    out = []
    for _, items in groups.items():
        items = sorted(items, key=lambda x: (-x["score"], x["image_id"], x["bbox"][0], x["bbox"][1]))
        boxes = [[p["bbox"][0], p["bbox"][1], p["bbox"][0] + p["bbox"][2],
                  p["bbox"][1] + p["bbox"][3]] for p in items]
        while boxes:
            keep = items[0]
            out.append(keep)
            rest_b, rest_i = [], []
            for j in range(1, len(boxes)):
                if iou_xyxy(boxes[0], boxes[j]) <= iou_thr:
                    rest_b.append(boxes[j])
                    rest_i.append(items[j])
            boxes, items = rest_b, rest_i
    return out


def load_gt(split: str = "val"):
    ann = json.loads((ANN_DIR / f"instances_{split}.json").read_text(encoding="utf-8"))
    gt = {}
    for a in ann["annotations"]:
        gt.setdefault(a["image_id"], []).append({"bbox": a["bbox"], "category_id": a["category_id"]})
    image_ids = [im["id"] for im in ann["images"]]
    return ann, gt, image_ids


# ---------------------------------------------------------------- the sweep
def sweep(preds, gt, image_ids):
    """Cumulative TP/FP at every prefix of the globally score-sorted prediction list.

    Processing predictions in descending score order is equivalent to per-image descending
    order, so each prefix corresponds to exactly one realisable threshold
    (threshold = score of the last included prediction).
    """
    ordered = sorted(preds, key=lambda p: (-p["score"], p["image_id"], p["bbox"][0], p["bbox"][1]))
    used = {i: [False] * len(gt.get(i, [])) for i in image_ids}
    tp = [0]
    n_tp = 0
    for p in ordered:
        gts = gt.get(p["image_id"], [])
        best, bi = MATCH_IOU, -1
        for j, g in enumerate(gts):
            if used[p["image_id"]][j] or g["category_id"] != p["category_id"]:
                continue
            v = iou_xywh(p["bbox"], g["bbox"])
            if v >= best and v > 0:
                best, bi = v, j
        if bi >= 0:
            used[p["image_id"]][bi] = True
            n_tp += 1
        tp.append(n_tp)
    return ordered, tp


def valid_cut_points(ordered):
    """prefix lengths that correspond to a realisable single threshold (no tie group is split)"""
    return [0] + [k for k in range(1, len(ordered) + 1)
                  if k == len(ordered) or ordered[k]["score"] < ordered[k - 1]["score"]]


def point(ordered, tp, k, n_images, n_gt, gt, image_ids):
    """metrics for the operating point that keeps the first k predictions"""
    fp = k - tp[k]
    thr = float("inf") if k == 0 else float(ordered[k - 1]["score"])
    rec = None if n_gt == 0 else tp[k] / n_gt
    prec = None if k == 0 else tp[k] / k
    f2 = None
    if prec is not None and rec is not None and (4 * prec + rec) > 0:
        f2 = 5 * prec * rec / (4 * prec + rec)
    # per-scene false alarms, using the same kept set
    kept = ordered[:k]
    by_img = {}
    for p in kept:
        by_img[p["image_id"]] = by_img.get(p["image_id"], 0) + 1
    matched_per_img = {}
    # recompute per-image TP for the kept prefix (needed for the A/B split)
    used = {i: [False] * len(gt.get(i, [])) for i in image_ids}
    for p in kept:
        gts = gt.get(p["image_id"], [])
        best, bi = MATCH_IOU, -1
        for j, g in enumerate(gts):
            if used[p["image_id"]][j] or g["category_id"] != p["category_id"]:
                continue
            v = iou_xywh(p["bbox"], g["bbox"])
            if v >= best and v > 0:
                best, bi = v, j
        if bi >= 0:
            used[p["image_id"]][bi] = True
            matched_per_img[p["image_id"]] = matched_per_img.get(p["image_id"], 0) + 1
    empty_ids = [i for i in image_ids if len(gt.get(i, [])) == 0]
    defect_ids = [i for i in image_ids if len(gt.get(i, [])) > 0]
    e_fp = sum(by_img.get(i, 0) for i in empty_ids)
    d_fp = sum(by_img.get(i, 0) - matched_per_img.get(i, 0) for i in defect_ids)
    return {
        "k": k, "threshold": thr, "fppi": fp / n_images, "tp": tp[k], "fp": fp,
        "fn": n_gt - tp[k], "precision": prec, "recall": rec, "f2": f2,
        "empty_images": len(empty_ids), "empty_alarm_images": sum(1 for i in empty_ids if by_img.get(i, 0) > 0),
        "empty_alarm_rate": (sum(1 for i in empty_ids if by_img.get(i, 0) > 0) / len(empty_ids)) if empty_ids else None,
        "empty_fp_total": e_fp, "empty_fp_per_img": e_fp / len(empty_ids) if empty_ids else None,
        "defect_images": len(defect_ids), "defect_fp_total": d_fp,
        "defect_fp_per_img": d_fp / len(defect_ids) if defect_ids else None,
        "preds_per_img": k / n_images,
    }


def select(points, budget):
    """max recall s.t. FPPI <= budget; ties -> lower FPPI; still tied -> higher threshold."""
    ok = [p for p in points if p["fppi"] <= budget + 1e-12]
    if not ok:
        return None
    best = None
    for p in ok:  # points are ordered by decreasing threshold
        if best is None:
            best = p
            continue
        r, rb = p["recall"], best["recall"]
        if r is None:
            continue
        if rb is None or r > rb + 1e-12:
            best = p
        elif abs(r - rb) <= 1e-12 and p["fppi"] < best["fppi"] - 1e-12:
            best = p
    return best


def coco_metrics(preds, split="val"):
    import contextlib
    import io

    from pycocotools.coco import COCO as COCOapi
    from pycocotools.cocoeval import COCOeval

    if not preds:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        gt_api = COCOapi(str(ANN_DIR / f"instances_{split}.json"))
        ev = COCOeval(gt_api, gt_api.loadRes(preds), "bbox")
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    def v(x):
        return None if x is None or x != x or x < 0 else float(x)

    return {"AP": v(ev.stats[0]), "AP50": v(ev.stats[1]), "AP75": v(ev.stats[2])}


def evaluate_file(pred_path: Path | str, label: str, split: str = "val") -> dict:
    pred_path = Path(pred_path)
    ann, gt, image_ids = load_gt(split)
    raw = json.loads(pred_path.read_text(encoding="utf-8"))
    preds = class_aware_nms(raw, NMS_IOU)
    n_images = len(image_ids)
    n_gt = sum(len(v) for v in gt.values())
    ordered, tp = sweep(preds, gt, image_ids)
    # a threshold is a SCORE VALUE: all predictions with that score must be kept together.
    # therefore only group boundaries (and the empty set) are valid operating points.
    valid_k = valid_cut_points(ordered)
    points = [point(ordered, tp, k, n_images, n_gt, gt, image_ids) for k in valid_k]

    out = {
        "label": label, "pred_file": str(pred_path), "split": split,
        "n_images": n_images, "n_gt": n_gt,
        "n_empty_images": sum(1 for i in image_ids if not gt.get(i)),
        "n_pred_raw": len(raw), "n_pred_after_nms": len(preds),
        "nms_removed": len(raw) - len(preds),
        "budgets": {}, "map_nms07": coco_metrics(preds, split),
        "curve": [{"threshold": p["threshold"] if p["k"] else None, "fppi": p["fppi"],
                   "recall": p["recall"], "tp": p["tp"], "fp": p["fp"]} for p in points],
    }
    for b in BUDGETS:
        p = select(points, b)
        out["budgets"][str(b)] = None if p is None else {
            k: (None if isinstance(v, float) and v != v else v) for k, v in p.items()}
    fixed = point(ordered, tp, sum(1 for p in preds if p["score"] >= FIXED_CONF), n_images, n_gt, gt, image_ids)
    out["fixed_conf_0.25"] = fixed
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/models.json",
                    help="JSON list of {label, pred, group, note}")
    ap.add_argument("--out-dir", default="evaluations/operating_points")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    man = json.loads((TASK / args.manifest).read_text(encoding="utf-8"))
    out_dir = TASK / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = skipped = 0
    for item in man:
        p = Path(item["pred"])
        if not p.is_file():
            print(f"[skip] {item['label']}: missing {p}", flush=True)
            skipped += 1
            continue
        res = evaluate_file(p, item["label"], args.split)
        res["group"] = item.get("group")
        res["note"] = item.get("note")
        (out_dir / f"{item['label']}.json").write_text(json.dumps(res, indent=2, default=float),
                                                      encoding="utf-8")
        b05 = res["budgets"]["0.05"]
        print(f"[{item['label']}] n={res['n_pred_after_nms']:5d} "
              f"FPPI0.05: thr={b05['threshold']:.3f} R={b05['recall']:.3f} P={b05['precision']:.3f} "
              f"FPPI={b05['fppi']:.3f} | FPPI1.0: R={res['budgets']['1.0']['recall']:.3f} "
              f"| conf.25: R={res['fixed_conf_0.25']['recall']:.3f} FPPI={res['fixed_conf_0.25']['fppi']:.2f} "
              f"| AP={res['map_nms07']['AP']:.4f}", flush=True)
        ok += 1
    print(f"\n{ok} evaluated, {skipped} skipped -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
