#!/usr/bin/env python3
"""Descriptive equal-FPPI diagnostic (test thresholds picked for analysis only).

For each arm/seed, walks the test precision-recall curve and reports recall at the highest
operating point whose ACTUAL FPPI stays within each target. This is not model selection: the
frozen-threshold results (eval.json) remain the reported protocol numbers; this only answers
"at the same measured FPPI, is the ranking actually different?".

Writes reports/equal_fppi.json / .md
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate_operating_points import class_aware_nms, load_gt, sweep  # noqa: E402

import paths  # noqa: E402

TARGETS = (0.05, 0.1, 0.2, 0.5)
BETA = 0.5
SEEDS = (42, 43, 44)


def prediction_paths() -> dict[str, dict[int, Path]]:
    out: dict[str, dict[int, Path]] = {}
    for arm in ("A", "B", "C", "D"):
        out[arm] = {}
        for s in SEEDS:
            tag = f"{arm}_s{s}" + (f"_b{BETA:g}" if arm in ("C", "D") else "")
            out[arm][s] = paths.RUNS / "arms" / tag / "preds" / "test.json"
    return out


def curve(path: Path) -> tuple[np.ndarray, float, int]:
    """returns (fppi array indexed by k, tp array, n_gt); n_images fixed at 50."""
    _, gt_all, img_ids = load_gt("test")
    raw = json.loads(path.read_text(encoding="utf-8"))
    preds = class_aware_nms(raw, 0.7)
    ordered, tp = sweep(preds, gt_all, img_ids)
    n_gt = sum(len(v) for v in gt_all.values())
    tp_arr = np.asarray(tp, dtype=np.float64)                     # len = len(ordered)+1
    k = np.arange(tp_arr.size, dtype=np.float64)
    fppi = (k - tp_arr) / len(img_ids)
    return fppi, tp_arr, n_gt


def recall_at(fppi: np.ndarray, tp: np.ndarray, n_gt: int, target: float) -> tuple[float, float]:
    ok = np.nonzero(fppi <= target + 1e-12)[0]
    if ok.size == 0:
        return 0.0, 0.0
    k = int(ok[-1])
    return float(tp[k] / n_gt), float(fppi[k])


def main() -> int:
    out = {"targets": list(TARGETS), "beta": BETA, "seeds": list(SEEDS), "arms": {}}
    for arm, per in prediction_paths().items():
        rec = {"per_seed": {}}
        for s, p in per.items():
            if not p.is_file():
                continue
            fppi, tp, n_gt = curve(p)
            rec["per_seed"][str(s)] = {f"{t:g}": dict(zip(("recall", "fppi"),
                                                          recall_at(fppi, tp, n_gt, t)))
                                       for t in TARGETS}
        if not rec["per_seed"]:
            continue
        rec["mean"] = {}
        for t in TARGETS:
            vals = [v[f"{t:g}"]["recall"] for v in rec["per_seed"].values()]
            rec["mean"][f"{t:g}"] = (sum(vals) / len(vals),
                                     st.stdev(vals) if len(vals) > 1 else 0.0)
        out["arms"][arm] = rec

    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    (paths.REPORTS / "equal_fppi.json").write_text(json.dumps(out, indent=2, default=float),
                                                   encoding="utf-8")

    L = ["# 等 FPPI 诊断（test，描述性）", "",
         "> 对每个 arm/seed 沿 test 的 P–R 曲线取\"实际 FPPI ≤ 目标\"的最高操作点；",
         "> **这不是模型选择**，正式结果仍用冻结阈值的那张表（`arms_summary.md`）。",
         "> 目的：避免拿不同实际 FPPI 的召回互比。", "",
         "| 臂 | 指标 | " + " | ".join(f"FPPI≤{t:g}" for t in TARGETS) + " |",
         "| --- | --- |" + " ---: |" * len(TARGETS)]
    for arm, rec in out["arms"].items():
        L.append(f"| {arm} | recall (%) | " +
                 " | ".join(f"{100 * rec['mean'][f'{t:g}'][0]:.1f}±{100 * rec['mean'][f'{t:g}'][1]:.1f}"
                            for t in TARGETS) + " |")
    L += ["", "| 臂 | 实际 FPPI | " + " | ".join(f"{t:g}" for t in TARGETS) + " |",
          "| --- | --- |" + " ---: |" * len(TARGETS)]
    for arm, rec in out["arms"].items():
        L.append(f"| {arm} | achieved | " +
                 " | ".join(f"{rec['per_seed'][str(SEEDS[0])][f'{t:g}']['fppi']:.3f}"
                            for t in TARGETS) + " |")
    L += ["", "（最后一行取 seed 42 的实际 FPPI 作示例，完整逐 seed 见 JSON。）", ""]
    (paths.REPORTS / "equal_fppi.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
