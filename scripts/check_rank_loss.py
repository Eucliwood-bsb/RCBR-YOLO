#!/usr/bin/env python3
"""Checks for the CR ranking criterion (engineering only).

  1. vectorised `_rank_aux` equals the loop prototype `hard_background_ranking_loss` on synthetic data;
  2. beta = 0 is bit-identical to the stock `E2ELoss` on a real training batch;
  3. beta > 0 changes only the classification term (box/dfl identical), gradients finite;
  4. the one-to-one branch is untouched;
  5. magnitude report: aux vs cls loss (so the weight is not chosen blind).

Writes reports/rank_loss_check.json / .md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
sys.path.insert(0, str(ROOT / "src"))

from cr_rank_loss import CRRankDetectionLoss, CRRankE2ELoss  # noqa: E402
from cr_modules import hard_background_ranking_loss  # noqa: E402

import paths  # noqa: E402

DATA = paths.DATA_YAML
CKPT = paths.WEIGHTS
BETA = 0.05


def synthetic_equivalence() -> dict:
    """Compare the vectorised aux with the prototype on identical synthetic masks."""
    torch.manual_seed(0)
    b, a, nc = 2, 200, 3
    logits = torch.randn(b, a, nc)
    tgt = torch.zeros(b, a, nc)
    # 5 positives per image on class 0, 3 on class 1
    for i in range(b):
        tgt[i, torch.randint(0, a, (5,)), 0] = 0.9
        tgt[i, torch.randint(0, a, (3,)), 1] = 0.7
    pos = tgt > 1e-6
    pos_any = pos.any(-1)
    bg = (~pos_any)[:, :, None].expand(-1, -1, nc).clone()

    obj = SimpleNamespace(k_pos=32, k_neg=64, margin=0.5, iou_excl=0.5, guard_px=2.0, stats={})
    empty = torch.zeros(b, 1, 4)
    mask_gt = torch.zeros(b, 1, 1)
    stride = torch.ones(1, 1)
    aux_vec = CRRankDetectionLoss._rank_aux(obj, logits, tgt, torch.zeros(b, a, 4), empty, mask_gt, stride)

    # the prototype takes (B,N,C) masks and does its own top-k / mean
    aux_proto = hard_background_ranking_loss(logits, pos, bg, margin=0.5,
                                             max_positives=32, max_negatives=64)
    return {"vectorised": float(aux_vec), "prototype": float(aux_proto),
            "abs_diff": float((aux_vec - aux_proto).abs())}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
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

    out = {"beta": BETA, "synthetic": synthetic_equivalence()}
    assert out["synthetic"]["abs_diff"] < 1e-6, out["synthetic"]

    dev = torch.device("cuda")
    base = YOLO(str(CKPT)).model.float()
    m = DetectionModel(dict(base.yaml), nc=9, ch=3, verbose=False).float()
    m.load(base)
    m.args = get_cfg()
    m.args.epochs = 80
    m.args.mosaic = 0.0
    m = m.to(dev).train()

    data = check_det_dataset(str(DATA))
    ds = build_yolo_dataset(m.args, data["train"], batch=1, data=data, mode="val", rect=False, stride=32)

    sample = None
    for i in range(len(ds)):
        s = ds[i]
        if len(s["cls"]) > 0:
            sample = s
            break
    img = sample["img"].to(dev).float()[None] / 255.0
    n = len(sample["cls"])
    batch = {"img": img, "bboxes": sample["bboxes"].to(dev).float(),
             "cls": sample["cls"].to(dev).float(), "batch_idx": torch.zeros(n, device=dev)}

    stock = E2ELoss(m)
    cr0 = CRRankE2ELoss(m, beta=0.0)
    with torch.no_grad():
        preds = m(img)
        s_total, s_items = stock(preds, batch)
        z_total, z_items = cr0(preds, batch)
    out["beta0_bit_identical"] = {
        "total_equal": bool(torch.equal(s_total, z_total)),
        "total_stock": float(s_total.sum()), "total_cr0": float(z_total.sum()),
        "items_equal": {k: bool(torch.equal(s_items[k], z_items[k])) for k in s_items},
    }
    assert out["beta0_bit_identical"]["total_equal"], "beta=0 must be bit-identical"

    # ---- beta > 0: compare the one-to-many branch directly (E2ELoss returns the o2o items dict)
    cr = CRRankE2ELoss(m, beta=BETA)
    with torch.no_grad():
        s_o2m_vec, s_o2m_items = stock.one2many.loss(preds["one2many"], batch)
        b_o2m_vec, b_o2m_items = cr.one2many.loss(preds["one2many"], batch)
        b_total, _ = cr(preds, batch)
        z_total, _ = cr0(preds, batch)
    aux = cr.stats.get("rank_aux")
    out["beta_gt0"] = {
        "total": float(b_total.sum()),
        "total_beta0": float(z_total.sum()),
        "total_delta": float(b_total.sum() - z_total.sum()),
        "o2m_weight_at_call": float(cr.o2m),
        "aux": aux,
        "cls_stock": float(s_o2m_items["cls_loss"]), "cls_cr": float(b_o2m_items["cls_loss"]),
        "cls_delta": float(b_o2m_items["cls_loss"] - s_o2m_items["cls_loss"]),
        "expected_cls_delta": None if aux is None else BETA * aux,
        "box_equal": bool(torch.equal(s_o2m_items["box_loss"], b_o2m_items["box_loss"])),
        "other_items_equal": {k: bool(torch.equal(s_o2m_items[k], b_o2m_items[k]))
                              for k in s_o2m_items if k != "cls_loss"},
        "stats": {k: v for k, v in cr.stats.items()},
        "aux_over_cls": None if not aux else aux / float(s_o2m_items["cls_loss"]),
        "effective_weight_over_cls": None if not aux else
        BETA * aux / float(s_o2m_items["cls_loss"]),
    }

    head = m.model[-1]
    cv2_w = head.cv2[0][-1].weight
    cv3_w = head.cv3[0][-1].weight

    def grads(loss_obj):
        m.zero_grad(set_to_none=True)
        p = m(img)
        tot, _ = loss_obj(p, batch)
        tot.sum().backward()
        return (cv2_w.grad.detach().clone(), cv3_w.grad.detach().clone())

    g0_box, g0_cls = grads(E2ELoss(m))
    g1_box, g1_cls = grads(CRRankE2ELoss(m, beta=BETA))
    out["grad"] = {
        "box_grad_equal": bool(torch.equal(g0_box, g1_box)),
        "cls_grad_allclose": bool(torch.allclose(g0_cls, g1_cls)),
        "cls_grad_max_abs_delta": float((g0_cls - g1_cls).abs().max()),
        "finite": bool(torch.isfinite(g1_cls).all() and torch.isfinite(g1_box).all()),
    }
    m.zero_grad(set_to_none=True)

    # ---- one-to-one branch must be untouched
    with torch.no_grad():
        one2one_stock = E2ELoss(m).one2one(preds["one2one"], batch)[0]
        one2one_cr = cr.one2one(preds["one2one"], batch)[0]
    out["one2one_untouched"] = bool(torch.equal(one2one_stock, one2one_cr))
    assert out["one2one_untouched"]

    rep = paths.REPORTS
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "rank_loss_check.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    L = ["# CR 排序损失接入检查（工程）", "",
         f"> beta={BETA} · margin=0.5 · K+=32 / K-=64 · guard=2 px · 只挂在 one-to-many", "",
         "## 1. 向量化实现 vs 原型", "",
         "| 实现 | aux |", "| --- | ---: |",
         f"| 向量化 `_rank_aux` | {out['synthetic']['vectorised']:.8f} |",
         f"| 原型 `hard_background_ranking_loss` | {out['synthetic']['prototype']:.8f} |",
         f"| abs diff | {out['synthetic']['abs_diff']:.2e} |", "",
         "## 2. beta=0 与原生损失", "",
         f"- 总损失逐位相同：**{out['beta0_bit_identical']['total_equal']}** "
         f"（{out['beta0_bit_identical']['total_stock']:.6f}）",
         f"- 各分量逐位相同：{out['beta0_bit_identical']['items_equal']}", "",
         "## 3. beta>0", "",
         f"- aux={out['beta_gt0']['aux']:.4f}，占原生 one-to-many cls 项的 "
         f"{100 * (out['beta_gt0']['aux_over_cls'] or 0):.1f}%；实际加到损失上的份额 "
         f"{100 * (out['beta_gt0']['effective_weight_over_cls'] or 0):.2f}%",
         f"- cls 增量 {out['beta_gt0']['cls_delta']:.6f}，应等于 beta*aux="
         f"{out['beta_gt0']['expected_cls_delta']:.6f}",
         f"- 总损失增量 {out['beta_gt0']['total_delta']:.6f}"
         f"（= o2m 权重 {out['beta_gt0']['o2m_weight_at_call']:.2f} × beta × aux）",
         f"- box / 其余分量逐位不变：{out['beta_gt0']['box_equal']} / {out['beta_gt0']['other_items_equal']}",
         f"- 机制统计：{out['beta_gt0']['stats']}", "",
         "## 4. 梯度", "",
         f"- box 头梯度逐位不变：{out['grad']['box_grad_equal']}",
         f"- cls 头梯度改变：max|Δ|={out['grad']['cls_grad_max_abs_delta']:.3e}，有限={out['grad']['finite']}",
         f"- one-to-one 分支逐位不变：{out['one2one_untouched']}", "",
         "## 5. 边界", "",
         "- 只验证接入正确与量级，不验证效果；",
         "- 掩码规则与 `mask_audit` 相同（IoU>0.5 或 GT 外扩 2 px 排除任一类正候选）；",
         "- 本基准只有 pinhole 有正样本，其余类的 L_rank 恒为 0（逐类循环会跳过）。", ""]
    (rep / "rank_loss_check.md").write_text("\n".join(L), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
