# RCBR-YOLO

Reference implementation of **RCBR-YOLO**, a YOLO26m-based detector that combines

* **MRC** — a multi-distance *relative context* block applied to the P3 and P4
  detection features, and
* **CHBR** — a class-conditioned *hard-background ranking* loss applied to the
  one-to-many training branch.

The repository contains code only. No dataset, no pretrained weights, no training
runs and no evaluated results are distributed here, and the code makes no
performance claim; numbers belong to the accompanying paper.

## Method

**MRC (`src/cr_modules.py`).** For each feature map ``F`` (C×H×W) and each
distance ``r ∈ {1,2,4}``, the eight neighbours of a 3×3 grid dilated by ``r`` are
averaged, excluding the centre and excluding invalid positions from the
denominator. A shared scoring network ``1×1 conv → SiLU → 1×1 conv`` predicts a
softmax weight per distance from ``[F, C_r, F−C_r]``; the weighted context and
difference are fused by ``1×1 conv → SiLU → 1×1 conv`` and added residually:

```
F_out = F + λ · h([F, C̄, D̄]),      λ initialised to 0.01 and learned
```

The block is inserted at layers **16 (P3, 256×80²)** and **19 (P4, 512×40²)**,
i.e. the two higher-resolution inputs of the `Detect` head (the head keeps
`f = [16, 19, 22]` and unchanged output shapes). The P5 node is not wrapped,
but it is downstream of the wrapped P3/P4 outputs through the bottom-up path.

**CHBR (`src/cr_rank_loss.py`).** On the one-to-many branch only:

```
L = L_det + β · mean_{image,class with a valid pair} mean_{p∈P,n∈N} softplus(m + z_n − z_p)
```

`P` are anchors assigned positive by the TAL assigner (`target_scores > 0`,
lowest-logit ``K₊=32`` kept); `N` are anchors that are not positive for any class
and whose decoded box has IoU ≤ 0.5 with every GT and whose centre is outside the
GT box grown by 2 px (highest-logit ``K₋=64`` kept). The term is added after the
loss gains and is only active when ``β > 0``; ``β = 0`` is bit-identical to the
stock criterion.

## Repository layout

```
src/
  cr_modules.py                  MRC block + reference CHBR prototype
  cr_integration.py              CtxWrap / insert_context / checkpoint helpers
  cr_rank_loss.py                vectorised CHBR criterion for YOLO26m
  evaluate_operating_points.py   FPPI / P / R / F2 / AP evaluation
  paths.py                       all filesystem locations (env-overridable)
scripts/
  train_arm.py                   train A/B/C/D (80-epoch recorded recipe)
  eval_arm.py                    export val/test and freeze val thresholds
  export_preds.py                COCO prediction export (conf 0.001, NMS 0.7)
  audit_integration.py           shapes, gradients, AMP, fuse, params/FLOPs, reload
  mask_audit.py                  background-mask statistics on the training split
  check_rank_loss.py             β=0 equivalence, gradients, loss magnitude
  equal_fppi.py                  descriptive equal-FPPI diagnostic
  summarise_arms.py              paired A/B/C/D summary
  make_tables.py                 report tables
  run_jobs.py                    sequential job runner
tests/
  test_relative_context.py       CPU unit tests (no GPU required)
```

The file names `cr_*` are retained on purpose: checkpoints saved by this code
pickle `CtxWrap` from `cr_integration`, so renaming the modules would break
existing weights.

## Requirements

* Python 3.12
* PyTorch 2.9.1 (CUDA 12.8 build), torchvision 0.24.1
* Ultralytics 8.4.198
* NumPy, OpenCV, pycocotools, matplotlib, psutil

```bash
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> **Note.** The experiments were run with a project-built `ultralytics 8.4.198`
> wheel whose default config contains a few non-upstream keys (`cls_remap` and
> distillation arguments). This code path only uses standard APIs, but if you
> reproduce with a different Ultralytics build, verify that `Detect.end2end` is
> `False` (the tested build is) and that the YOLO26 architecture is present.

## Data layout

The scripts never embed an absolute path. Defaults (all overridable, see
`src/paths.py`):

```
data/
  data.yaml                  Ultralytics dataset yaml (see configs/data.example.yaml)
  annotations/
    instances_train.json     COCO annotations
    instances_val.json
    instances_test.json
  images/{train,val,test}/
weights/
  yolo26m.pt                 COCO-pretrained checkpoint
```

Environment overrides: `RCBR_ROOT`, `RCBR_DATA_YAML`, `RCBR_ANN_DIR`,
`RCBR_WEIGHTS`, `RCBR_RUNS`, `RCBR_REPORTS`, `RCBR_STATE`,
`RCBR_TRAIN_IMAGES`, `RCBR_VAL_IMAGES`, `RCBR_TEST_IMAGES`.

The annotation file defines which classes have ground truth. In the recorded
benchmark only *pinhole* was annotated; the other declared classes contribute no
positives, so all reported metrics are effectively pinhole-only.

## Quick start

```bash
# 0) CPU unit tests
python -m unittest -v tests/test_relative_context.py

# 1) integration audit: shapes, gradients, AMP, fuse, params/FLOPs, fresh-process reload
python scripts/audit_integration.py --weights weights/yolo26m.pt

# 2) train one arm (A = stock model, B = MRC, C = CHBR, D = both)
python scripts/train_arm.py --arm B --seed 42 --epochs 80

# 3) validation-only evaluation (used for every model/threshold decision)
python scripts/eval_arm.py --tag B_s42 --splits val

# 4) once all decisions are frozen: evaluate test exactly once
python scripts/eval_arm.py --tag B_s42 --splits val,test

# 5) summary and tables
python scripts/summarise_arms.py --beta 0.5 --seeds 42,43,44
python scripts/make_tables.py --beta 0.5 --seeds 42,43,44
```

Arms: `A` stock YOLO26m, `B` + MRC, `C` + CHBR, `D` + both. The A arm may also be
a matched plain run produced elsewhere; place its `eval.json` under
`runs/arms/A_s<seed>/` and the summary treats it like any other arm.

Other checks:

```bash
python scripts/mask_audit.py --weights weights/yolo26m.pt       # positive/background counts
python scripts/check_rank_loss.py --weights weights/yolo26m.pt  # beta=0 equivalence, gradients
python scripts/equal_fppi.py --beta 0.5                         # descriptive only
```

## Evaluation protocol (summary)

Full details in `docs/PROTOCOL.md`.

* Export: class-aware NMS at IoU 0.7, confidence 0.001, `max_det=300`, `rect` 640.
* Seven validation FPPI budgets: 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0. For each
  model/seed/budget the threshold is the realisable score with the highest recall
  subject to `FPPI ≤ budget` (ties → lower FPPI). Thresholds are frozen and
  applied to test once; **actual test FPPI is reported even when it exceeds the
  budget**.
* Matching: same class, one-to-one, IoU ≥ 0.50, predictions in descending score;
  every unmatched prediction is a false positive. Only *pinhole* has ground truth
  in the recorded data.
* Fixed operating point: confidence 0.25, IoU 0.50.
* AP: pycocotools `COCOeval` (box), IoU 0.50:0.95, 101-point interpolation,
  `maxDets=[1,10,100]`, on the same NMS'd predictions (not truncated at 0.25).

## What is not included

Datasets, COCO annotations, pretrained weights, trained checkpoints, prediction
files, logs and generated reports are **not** part of this repository; `.gitignore`
excludes them. Latency was not measured under a controlled protocol, so no
latency claim is made anywhere and FLOPs are not used as a substitute.

## License

This project is licensed under the **GNU Affero General Public License v3.0**
(AGPL-3.0); see `LICENSE`. It depends on `ultralytics`, which is also AGPL-3.0.
If a modified version is offered as a network service, the AGPL's source-offer
obligation applies. See `NOTICE.md` for third-party details.
