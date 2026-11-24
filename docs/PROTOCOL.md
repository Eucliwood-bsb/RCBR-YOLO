# Recorded training and evaluation protocol

This file describes the protocol used for the four-arm comparison. It contains no
results and no machine-specific path. Everything below is implemented by the
scripts in `scripts/`.

## 1. Arms

| arm | context block (MRC) | ranking loss (CHBR) |
|---|---|---|
| A | no | no (stock criterion) |
| B | P3 + P4 | no |
| C | no | yes |
| D | P3 + P4 | yes |

A and B use the stock Ultralytics criterion; C and D replace the one-to-many
criterion with `CRRankE2ELoss(beta=beta)`. The one-to-one branch stays stock in
all arms.

## 2. Starting point and recipe

* Starting weights: COCO-pretrained `yolo26m.pt` (`nc=80`). The detector is
  rebuilt with the dataset's `nc`; the classification head's final convolutions
  are re-initialised (the recorded load transferred 756/768 tensors).
* Model: `ultralytics/cfg/models/26/yolo26.yaml`, scale `m`.
* Input 640×640; batch 8; nominal batch 64 (gradient accumulation 8); 80 epochs;
  `patience=80` (no early stopping); fixed `last.pt`; `save_period=-1`.
* SGD: `lr0=5e-3`, `lrf=0.01`, momentum 0.937, weight decay 5e-4, warmup 5 epochs.
  Ultralytics uses a **linear** `LambdaLR` (`cos_lr=false`), stepped per epoch;
  the realised last-epoch learning rate is `lr0 · ((1−79/80)·(1−lrf)+lrf)`,
  i.e. ≈2.24% of `lr0`, not `lrf·lr0`.
* AMP on, deterministic on, `rect=false` in training.
* Augmentation actually applied by the detection pipeline: random affine
  (scale gain U[0.5, 1.5], translation ±10%, no rotation/shear/perspective,
  114-pixel border when zooming out), horizontal flip p=0.5, and HSV jitter
  (H ±2.7°, S ×U[0.3,1.7], V ×U[0.6,1.4]). No mosaic, mixup, cutmix, copy-paste,
  vertical flip or rotation. `erasing`/`auto_augment` exist in the configuration
  but are consumed only by the classification dataloader, so they had no effect
  on these detection runs.

## 3. β selection (CHBR weight)

Select on the validation split only, on a single seed:

* grid `β ∈ {0.05, 0.2, 0.5}`, arm C, seed 42;
* criterion: validation R@FPPI 0.05 (highest wins; ties → larger β);
* the selected β is used for all C/D runs; the selection run counts as C's seed.

In the recorded experiment β = 0.5 was selected. The criterion and grid must be
fixed before looking at test results.

## 4. Prediction export

`scripts/export_preds.py`:

* native 640 input, `rect=True`;
* confidence 0.001, class-aware NMS at IoU 0.7, `max_det=300` per image;
* boxes clipped to the image, one-based `category_id` (class index + 1).

The same settings are used for every arm and for the external baselines of the
paper. The evaluator re-applies a class-aware NMS at IoU 0.7; for the YOLO arms
this removes ~0.1–0.2% of the boxes (it is not a no-op, but close).

## 5. Operating points and threshold freezing

* Seven validation FPPI budgets: **0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0**.
* Candidate thresholds are all realisable score values of the validation
  predictions (group boundaries), plus the empty prediction set (`+inf`).
  No fixed grid is used.
* For each model/seed/budget: keep the point with the highest recall subject to
  `FPPI ≤ budget`; ties → lower FPPI; because `TP+FP` is fixed by
  `(recall, FPPI)`, the documented third tie-break (higher threshold) is a
  defensive branch that does not change the outcome.
* Thresholds are frozen and applied to test **once**. Actual test FPPI is
  reported even when it exceeds the budget; the budget is not a test guarantee.
* All model, checkpoint, β and threshold decisions are made on validation only.
  Test is not used for any selection.

## 6. Matching and metrics

`src/evaluate_operating_points.py`:

* 50 validation images / 50 test images; FPPI denominator = all evaluated images
  (including empty ones).
* Matching: same class, one-to-one, predictions in descending confidence; each
  prediction is matched to the unmatched GT of the same class with the largest
  IoU, provided that IoU ≥ 0.50. Every unmatched prediction is a false positive.
* Ties in score: ordered by `(-score, image_id, x1, y1)`; equal-score
  predictions are kept as one group, so a threshold never splits a tie group.
* P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2PR/(P+R), F2 = 5PR/(4P+R).
* Fixed supplementary point: confidence 0.25, IoU 0.50.
* No ignore regions, no crowd handling in the custom matcher, no truncation or
  minimum-size filter. `iscrowd` is present in the COCO files and is zero for
  every annotation; pycocotools sees it for AP, the custom matcher ignores it.

## 7. AP

pycocotools `COCOeval` (bbox) on the same NMS'd predictions:

* IoU 0.50:0.95 step 0.05, 101-point interpolation, area range `all`,
  `maxDets=[1, 10, 100]`;
* predictions are **not** truncated at confidence 0.25 (the export floor 0.001 is
  used);
* AP is a category mean over categories that have ground truth. In the recorded
  data only *pinhole* has annotations, so the reported AP is a single-category AP;
  the other declared categories contribute no GT and are excluded.

## 8. Class coverage (important)

In the recorded data the dataset yaml declares nine output categories but only
`pinhole` has instance annotations (1,804 / 171 / 352 for train/val/test).
`misalignment` and `broken line` are excluded from annotation by the evaluation
specification and are **not** used as ignore regions: predictions of those
classes can only be false positives, and real instances in the images are
unlabelled background. Consequently every reported metric is pinhole-only, and a
"seven-class" description refers to the evaluation scope, not to seven annotated
classes.
