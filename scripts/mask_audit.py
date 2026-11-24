#!/usr/bin/env python3
"""Reliable-background mask audit for the CHBR ranking loss (read-only on model/data).

The CR prototype requires a `background_mask` but does not build one. This script measures, on the
real training split, how many candidates survive a conservative rule:

    positive  : assigner target_score > 0 for that class (one-to-many branch, TAL topk=10)
    excluded  : predicted box IoU with any GT > 0.5, or anchor centre inside a GT box grown by
                `guard_px` (same rule the ex12 rank loss used), or positive for any class
    background: neither  -> eligible for the ranking loss

It also reports the per-level split (P3/P4/P5) and dumps one visualisation. It does NOT train and
does NOT claim the mask is correct; it quantifies whether enough pairs exist to make the loss
non-degenerate, and whether the exclusions eat the candidates.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
sys.path.insert(0, str(ROOT / "src"))

import paths

DATA = paths.DATA_YAML
CKPT = paths.WEIGHTS
LEVELS = ((80, 80, 8), (40, 40, 16), (20, 20, 32))


def level_of(idx: np.ndarray) -> np.ndarray:
    bounds = np.cumsum([h * w for h, w, _ in LEVELS])
    out = np.zeros(idx.shape, dtype=np.int8)
    for k, b in enumerate(bounds[:-1]):
        out[idx >= b] = k + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--guard-px", type=float, default=2.0)
    ap.add_argument("--iou-exclude", type=float, default=0.5)
    ap.add_argument("--data", default=str(paths.DATA_YAML))
    ap.add_argument("--weights", default=str(paths.WEIGHTS))
    args = ap.parse_args()
    global DATA, CKPT
    DATA = Path(args.data)
    CKPT = Path(args.weights)

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import E2ELoss

    dev = torch.device("cuda")
    base = YOLO(str(CKPT)).model.float()
    m = DetectionModel(dict(base.yaml), nc=9, ch=3, verbose=False).float()
    m.load(base)
    m.args = get_cfg()
    m.args.epochs = 80
    m.args.mosaic = 0.0
    m = m.to(dev).eval()

    crit = E2ELoss(m)
    data = check_det_dataset(str(DATA))
    ds = build_yolo_dataset(m.args, data["train"], batch=args.batch, data=data, mode="val",
                            rect=False, stride=32)

    captured: list[dict] = []
    assigner = crit.one2many.assigner
    orig = assigner._forward

    def wrap(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **kw):
        out = orig(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **kw)
        tgt_labels, tgt_bboxes, tgt_scores, fg, tgt_idx = out
        with torch.no_grad():
            captured.append({
                "scores": pd_scores.detach(),
                "pboxes": pd_bboxes.detach(),
                "gt": gt_bboxes.detach(),
                "mask_gt": mask_gt.detach(),
                "tgt_scores": tgt_scores.detach(),
                "fg": fg.detach(),
            })
        return out

    assigner._forward = wrap

    rows = []
    viz = None
    n_done = 0
    with torch.no_grad():
        for bi in range(min(len(ds), args.limit)):
            s = ds[bi]
            n_gt = len(s["cls"])
            if n_gt == 0:
                rows.append({"file": Path(s["im_file"]).name, "n_gt": 0, "n_pos": 0,
                             "n_bg": 0, "n_excl": 0, "cls_with_pos": 0, "annotated": False})
                n_done += 1
                continue
            img = s["img"].to(dev).float()[None] / 255.0
            batch = {"img": img, "bboxes": s["bboxes"].to(dev).float(),
                     "cls": s["cls"].to(dev).float(),
                     "batch_idx": torch.zeros(n_gt, device=dev)}
            captured.clear()
            crit(m(img), batch)
            if not captured:
                continue
            c = captured[0]
            bs = c["scores"].shape[0]
            for b in range(bs):
                pb = c["pboxes"][b].float()                      # (A,4) px
                gt = c["gt"][b][c["mask_gt"][b, :, 0].bool()].float()
                ts = c["tgt_scores"][b]                          # (A,C)
                pos_c = ts > 1e-6
                pos_any = pos_c.any(dim=-1)
                if gt.numel():
                    lt = torch.max(pb[:, None, :2], gt[None, :, :2])
                    rb = torch.min(pb[:, None, 2:], gt[None, :, 2:])
                    wh = (rb - lt).clamp(min=0)
                    inter = wh[..., 0] * wh[..., 1]
                    ap = (pb[:, 2] - pb[:, 0]).clamp(min=0) * (pb[:, 3] - pb[:, 1]).clamp(min=0)
                    ag = (gt[:, 2] - gt[:, 0]).clamp(min=0) * (gt[:, 3] - gt[:, 1]).clamp(min=0)
                    iou = inter / (ap[:, None] + ag[None, :] - inter + 1e-9)
                    near_iou = (iou > args.iou_exclude).any(dim=1)
                    cx = (pb[:, 0] + pb[:, 2]) / 2
                    cy = (pb[:, 1] + pb[:, 3]) / 2
                    inside = (((cx[:, None] > gt[None, :, 0] - args.guard_px) &
                               (cx[:, None] < gt[None, :, 2] + args.guard_px)) &
                              ((cy[:, None] > gt[None, :, 1] - args.guard_px) &
                               (cy[:, None] < gt[None, :, 3] + args.guard_px))).any(dim=1)
                else:
                    near_iou = torch.zeros_like(pos_any)
                    inside = torch.zeros_like(pos_any)
                bg = ~pos_any & ~near_iou & ~inside
                excl = ~pos_any & (near_iou | inside)
                idx = np.arange(bg.numel())
                lv_pos = np.bincount(level_of(idx[pos_any.cpu().numpy()]), minlength=3)
                lv_bg = np.bincount(level_of(idx[bg.cpu().numpy()]), minlength=3)
                rows.append({
                    "file": Path(s["im_file"]).name, "n_gt": int(gt.shape[0]),
                    "n_pos": int(pos_any.sum()), "n_bg": int(bg.sum()),
                    "n_excl": int(excl.sum()),
                    "n_excl_iou": int((~pos_any & near_iou).sum()),
                    "n_excl_guard": int((~pos_any & inside & ~near_iou).sum()),
                    "cls_with_pos": int(pos_c.any(dim=0).sum()),
                    "pos_per_class": {str(i): int(v) for i, v in enumerate(pos_c.sum(0).tolist()) if v},
                    "lv_pos": [int(x) for x in lv_pos], "lv_bg": [int(x) for x in lv_bg],
                    "annotated": True,
                })
                if viz is None and int(gt.shape[0]) >= 6:
                    viz = {"pb": pb.cpu(), "pos": pos_any.cpu(), "bg": bg.cpu(),
                           "excl": excl.cpu(), "img": s["img"].cpu(), "file": Path(s["im_file"]).name}
            n_done += 1

    ann = [r for r in rows if r["annotated"]]
    empt = [r for r in rows if not r["annotated"]]
    tot_pos = int(sum(r["n_pos"] for r in ann))
    tot_bg = int(sum(r["n_bg"] for r in ann))
    tot_excl = int(sum(r["n_excl"] for r in ann))
    out = {
        "checkpoint": str(CKPT), "data": str(DATA), "n_images": n_done,
        "n_annotated": len(ann), "n_empty": len(empt),
        "guard_px": args.guard_px, "iou_exclude": args.iou_exclude,
        "totals": {"pos": tot_pos, "bg": tot_bg, "excluded": tot_excl,
                   "excluded_iou": int(sum(r.get("n_excl_iou", 0) for r in ann)),
                   "excluded_guard": int(sum(r.get("n_excl_guard", 0) for r in ann)),
                   "bg_over_pos": (tot_bg / tot_pos) if tot_pos else None},
        "per_image": {
            "pos_min": min(r["n_pos"] for r in ann), "pos_median": float(np.median([r["n_pos"] for r in ann])),
            "pos_max": max(r["n_pos"] for r in ann),
            "bg_min": min(r["n_bg"] for r in ann), "bg_median": float(np.median([r["n_bg"] for r in ann])),
            "bg_max": max(r["n_bg"] for r in ann),
            "images_without_bg": int(sum(1 for r in ann if r["n_bg"] == 0)),
            "images_without_pos": int(sum(1 for r in ann if r["n_pos"] == 0)),
        },
        "levels": {"pos": [int(sum(r["lv_pos"][i] for r in ann)) for i in range(3)],
                   "bg": [int(sum(r["lv_bg"][i] for r in ann)) for i in range(3)]},
        "pos_class_hist": {str(k): int(v) for k, v in
                           sorted(Counter(c for r in ann for c in r["pos_per_class"]).items())},
        "per_image_rows": rows,
    }
    rep = paths.REPORTS
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "mask_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- visualisation on the first image with >= 6 GT
    if viz is not None:
        import cv2
        img = (viz["img"].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        pb = viz["pb"].numpy()
        cx = ((pb[:, 0] + pb[:, 2]) / 2).round().astype(int)
        cy = ((pb[:, 1] + pb[:, 3]) / 2).round().astype(int)
        for i in range(len(pb)):
            if not (0 <= cx[i] < img.shape[1] and 0 <= cy[i] < img.shape[0]):
                continue
            if viz["bg"][i]:
                cv2.circle(img, (cx[i], cy[i]), 2, (160, 160, 160), -1)
            if viz["excl"][i]:
                cv2.circle(img, (cx[i], cy[i]), 3, (0, 165, 255), -1)
            if viz["pos"][i]:
                cv2.circle(img, (cx[i], cy[i]), 4, (0, 255, 0), -1)
        cv2.imwrite(str(rep / "mask_audit_example.png"), img[:, :, ::-1])
        out["viz"] = {"file": viz["file"], "png": str(rep / "mask_audit_example.png")}
        (rep / "mask_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- report
    L = ["# 可靠背景掩码审计（CR 排序损失的输入）", "",
         f"> {CKPT.name}（只读）· 训练 split 前 {n_done} 张 · guard={args.guard_px}px · "
         f"IoU 排除阈值 {args.iou_exclude}", "",
         "规则：正样本 = one-to-many 分配 `target_score > 0`；排除 = 预测框与任一 GT 的 IoU > 0.5，"
         "或锚点中心落在 GT 外扩 guard px 内；其余才是可用背景。", "",
         "## 1. 总量", "",
         "| 项 | 值 |", "| --- | ---: |",
         f"| 有标注图 / 空图 | {len(ann)} / {len(empt)} |",
         f"| 正候选总数 | {tot_pos} |",
         f"| **可用背景总数** | **{tot_bg}** |",
         f"| 被排除候选 | {tot_excl}（IoU {out['totals']['excluded_iou']} + guard "
         f"{out['totals']['excluded_guard']}） |",
         f"| 背景/正 比例 | {out['totals']['bg_over_pos']:.1f} |", "",
         "## 2. 逐图分布", "",
         "| 指标 | 正候选 | 可用背景 |", "| --- | ---: | ---: |",
         f"| min | {out['per_image']['pos_min']} | {out['per_image']['bg_min']} |",
         f"| median | {out['per_image']['pos_median']} | {out['per_image']['bg_median']} |",
         f"| max | {out['per_image']['pos_max']} | {out['per_image']['bg_max']} |",
         "", f"- 无正候选的图：{out['per_image']['images_without_pos']}；"
             f"无可用背景的图：{out['per_image']['images_without_bg']}", "",
         "## 3. 层级分布（P3/P4/P5）", "",
         f"- 正候选：{out['levels']['pos']}",
         f"- 可用背景：{out['levels']['bg']}", "",
         "## 4. 结论边界", "",
         "- 这只证明「按该规则能得到非退化的正/负集合」，**不证明排序损失有效**；",
         "- ex12 用同一排除规则训练过同构损失，结果是负的（R@0.05 0.430→0.377）；",
         "- CR 的 prototype 要求逐类掩码，本审计的正样本统计显示只有 pinhole 有分配，"
         "与 §6 的结论一致。", ""]
    (rep / "mask_audit.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("totals", "per_image", "levels")}, indent=1))
    print("written reports/mask_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
