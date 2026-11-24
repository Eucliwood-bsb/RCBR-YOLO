"""CR class-conditioned hard-background ranking, wired into YOLO26m's one-to-many branch.

Matches the reference prototype in `cr_modules.py`:

    L = L_det + beta * mean_{image, class with a valid pair} mean_{p in P, n in N} softplus(m + z_n - z_p)

    P : assigned positives (target_score > 0) for that class, lowest-logit K+ kept
    N : reliable background for that class (not positive for ANY class, predicted-box IoU with any
        GT <= 0.5, anchor centre not inside a GT box grown by `guard_px`), highest-logit K- kept

Design choices (differences from a sigmoid+hinge ranking loss used in earlier internal experiments):
  * logits + softplus instead of sigmoid probabilities + hinge;
  * per-class masks and a per-image-class mean, not pinhole-only;
  * only the one-to-many branch (inference branch), one-to-one stays stock;
  * the term is added after the loss gains, so `beta` is the literal weight of L_rank.

`beta = 0` takes exactly the stock code path (no extra ops), and the smoke check asserts
bit-identity against `v8DetectionLoss`.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors

NEG_INF = float("-inf")
POS_INF = float("inf")


class CRRankDetectionLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None,
                 beta: float = 0.0, margin: float = 0.5, k_pos: int = 32, k_neg: int = 64,
                 iou_excl: float = 0.5, guard_px: float = 2.0, stats: dict | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        self.beta = float(beta)
        self.margin = float(margin)
        self.k_pos = int(k_pos)
        self.k_neg = int(k_neg)
        self.iou_excl = float(iou_excl)
        self.guard_px = float(guard_px)
        self.stats = stats if stats is not None else {}

    # ------------------------------------------------------------------ aux
    def _rank_aux(self, logits, target_scores, pred_bboxes, gt_bboxes, mask_gt, stride_tensor):
        """Vectorised over (image, class); returns a scalar connected to `logits`."""
        b, a, nc = logits.shape
        with torch.no_grad():
            pos = target_scores > 1e-6                                  # (b,a,nc)
            pos_any = pos.any(dim=-1)                                   # (b,a)
            pb = pred_bboxes.detach().float() * stride_tensor.float()   # (b,a,4) px
            gt = gt_bboxes.float()                                      # (b,n,4) px
            valid_gt = mask_gt.squeeze(-1).bool()                       # (b,n)
            if gt.numel() and valid_gt.any():
                lt = torch.max(pb[:, :, None, :2], gt[:, None, :, :2])
                rb = torch.min(pb[:, :, None, 2:], gt[:, None, :, 2:])
                wh = (rb - lt).clamp(min=0)
                inter = wh[..., 0] * wh[..., 1]
                ap = (pb[:, :, 2] - pb[:, :, 0]).clamp(min=0) * (pb[:, :, 3] - pb[:, :, 1]).clamp(min=0)
                ag = (gt[:, :, 2] - gt[:, :, 0]).clamp(min=0) * (gt[:, :, 3] - gt[:, :, 1]).clamp(min=0)
                iou = inter / (ap[:, :, None] + ag[:, None, :] - inter + 1e-9)
                iou = torch.where(valid_gt[:, None, :], iou, torch.zeros_like(iou))
                near = (iou > self.iou_excl).any(dim=-1)                # (b,a)
                cx = (pb[:, :, 0] + pb[:, :, 2]) / 2
                cy = (pb[:, :, 1] + pb[:, :, 3]) / 2
                inside = (((cx[:, :, None] > gt[:, None, :, 0] - self.guard_px) &
                           (cx[:, :, None] < gt[:, None, :, 2] + self.guard_px)) &
                          ((cy[:, :, None] > gt[:, None, :, 1] - self.guard_px) &
                           (cy[:, :, None] < gt[:, None, :, 3] + self.guard_px)))
                inside = (inside & valid_gt[:, None, :]).any(dim=-1)    # (b,a)
            else:
                near = torch.zeros_like(pos_any)
                inside = torch.zeros_like(pos_any)
            excluded = near | inside
            bg = (~pos_any) & (~excluded)                               # (b,a)
            bgm = bg[:, :, None].expand(-1, -1, nc)
            n_excluded = int((excluded & ~pos_any).sum())

        # indices are chosen without grad; gather keeps the graph on the logits
        kp = min(self.k_pos, a)
        kn = min(self.k_neg, a)
        pos_pick = logits.masked_fill(~pos, POS_INF).topk(kp, dim=1, largest=False)
        neg_pick = logits.masked_fill(~bgm, NEG_INF).topk(kn, dim=1)
        pos_v = pos_pick.values.permute(0, 2, 1)   # (b,nc,kp)
        neg_v = neg_pick.values.permute(0, 2, 1)   # (b,nc,kn)
        pos_ok = torch.isfinite(pos_v)
        neg_ok = torch.isfinite(neg_v)

        pair = F.softplus(self.margin + neg_v[:, :, :, None] - pos_v[:, :, None, :])
        ok = neg_ok[:, :, :, None] & pos_ok[:, :, None, :]
        num = (pair * ok).sum(dim=(-1, -2))
        den = ok.sum(dim=(-1, -2))
        per_bc = num / den.clamp(min=1)
        use = den > 0
        if not bool(use.any()):
            return logits.sum() * 0.0
        aux = per_bc[use].mean()
        with torch.no_grad():
            self.stats["rank_aux"] = float(aux.detach())
            self.stats["rank_pairs"] = int(den[use].sum())
            self.stats["rank_pos"] = int(pos_ok.sum())
            self.stats["rank_neg"] = int(neg_ok.sum())
            self.stats["rank_excluded"] = n_excluded
            self.stats["rank_image_classes"] = int(use.sum())
        return aux

    # ------------------------------------------------------------------ loss
    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any],
                                      **assign_kwargs) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
            **assign_kwargs,
        )

        target_scores_sum = target_scores.sum().clamp_(min=1)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
            target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor,
        )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        if self.beta > 0.0:
            aux = self._rank_aux(pred_scores, target_scores, pred_bboxes, gt_bboxes, mask_gt, stride_tensor)
            loss[1] = loss[1] + self.beta * aux

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )


class CRRankE2ELoss(E2ELoss):
    """Stock one-to-one branch; the one-to-many branch carries the CR ranking term."""

    STAT_KEYS = ("rank_aux", "rank_pairs", "rank_pos", "rank_neg", "rank_excluded",
                 "rank_image_classes")

    def __init__(self, model, beta: float = 0.0, margin: float = 0.5, k_pos: int = 32,
                 k_neg: int = 64, iou_excl: float = 0.5, guard_px: float = 2.0,
                 stats: dict | None = None):
        super().__init__(model)
        self.stats = stats if stats is not None else {}
        self.beta = float(beta)
        self.one2many = CRRankDetectionLoss(
            model, tal_topk=10, beta=beta, margin=margin, k_pos=k_pos, k_neg=k_neg,
            iou_excl=iou_excl, guard_px=guard_px, stats=self.stats)

    def __call__(self, preds, batch):
        """E2ELoss plus per-epoch accumulators (logging only, no effect on the loss)."""
        out = super().__call__(preds, batch)
        st = self.stats
        st["n_batches"] = int(st.get("n_batches", 0)) + 1
        for k in self.STAT_KEYS:
            st["sum_" + k] = float(st.get("sum_" + k, 0.0)) + float(st.get(k, 0) or 0)
        return out
