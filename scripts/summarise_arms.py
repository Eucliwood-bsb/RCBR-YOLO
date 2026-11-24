#!/usr/bin/env python3
"""Assemble the A/B/C/D paired comparison from finished runs.

A/B/C/D: runs/arms/<tag>/eval.json written by eval_arm.py --splits val,test.
The A arm is either trained by train_arm.py --arm A or supplied as a matched plain
run whose eval.json is placed under runs/arms/A_s<seed>/.

Usage: python scripts/summarise_arms.py --beta 0.2 [--seeds 42,43,44]
Writes reports/arms_summary.json / .md (skips missing arms, so it can run incrementally).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
sys.path.insert(0, str(ROOT / "src"))

import paths  # noqa: E402

BUD = ("0.05", "0.1", "0.2", "0.5", "1.0", "2.0", "5.0")


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return sum(xs) / len(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def fmt(v, nd=3, pct=False):
    if v is None:
        return "--"
    m, s = v
    m = 100 * m if pct else m
    s = 100 * s if pct else s
    return f"{m:.{nd}f}±{s:.{nd}f}"


def arm_a(seed: int) -> dict | None:
    """Matched plain arm: same interface as the other runs."""
    return arm_run(f"A_s{seed}")


def arm_run(tag: str) -> dict | None:
    p = paths.RUNS / "arms" / tag / "eval.json"
    if not p.is_file():
        return None
    e = json.loads(p.read_text(encoding="utf-8"))
    return {"val": e["val"]["budgets"], "test": e["test"]["budgets"],
            "val_ap": e["val"]["map_nms07"], "test_ap": e["test"]["map_nms07"],
            "val_c25": e["val"]["fixed_conf_0.25"], "test_c25": e["test"]["fixed_conf_0.25"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--tex", default="", help="optional path for a LaTeX fragment")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    tags = {"A": {s: f"A_s{s}" for s in seeds},
            "B": {s: f"B_s{s}" for s in seeds},
            "C": {s: f"C_s{s}_b{args.beta:g}" for s in seeds},
            "D": {s: f"D_s{s}_b{args.beta:g}" for s in seeds}}
    data: dict[str, dict[int, dict]] = {}
    for arm in ("A", "B", "C", "D"):
        data[arm] = {}
        for s in seeds:
            d = arm_a(s) if arm == "A" else arm_run(tags[arm][s])
            if d is not None:
                data[arm][s] = d
    out = {"beta": args.beta, "seeds": seeds,
           "available": {a: sorted(data[a]) for a in data},
           "arms": {}}
    for arm, per in data.items():
        if not per:
            continue
        rec = {"n_seeds": len(per)}
        for b in BUD:
            rec[f"val_R@{b}"] = mean_sd([per[s]["val"][b]["recall"] if per[s]["val"][b] else None
                                         for s in per])
            rec[f"val_fppi@{b}"] = mean_sd([per[s]["val"][b]["fppi"] if per[s]["val"][b] else None
                                            for s in per])
            rec[f"test_R@{b}"] = mean_sd([per[s]["test"][b]["recall"] if per[s]["test"][b] else None
                                          for s in per])
            rec[f"test_f2@{b}"] = mean_sd([per[s]["test"][b]["f2"] if per[s]["test"][b] else None
                                           for s in per])
            rec[f"test_fppi@{b}"] = mean_sd([per[s]["test"][b]["fppi"] if per[s]["test"][b] else None
                                             for s in per])
        for split in ("val", "test"):
            for k in ("AP", "AP50", "AP75"):
                rec[f"{split}_{k}"] = mean_sd([per[s][f"{split}_ap"][k] for s in per])
            rec[f"{split}_conf025_recall"] = mean_sd([per[s][f"{split}_c25"]["recall"] for s in per])
            rec[f"{split}_conf025_precision"] = mean_sd([per[s][f"{split}_c25"]["precision"] for s in per])
            rec[f"{split}_conf025_f2"] = mean_sd([per[s][f"{split}_c25"]["f2"] for s in per])
            f1_values = []
            for s in per:
                p_, r_ = per[s][f"{split}_c25"]["precision"], per[s][f"{split}_c25"]["recall"]
                if p_ is not None and r_ is not None and p_ + r_ > 0:
                    f1_values.append(2 * p_ * r_ / (p_ + r_))
            rec[f"{split}_conf025_f1"] = mean_sd(f1_values)
            rec[f"{split}_conf025_fppi"] = mean_sd([per[s][f"{split}_c25"]["fppi"] for s in per])
        out["arms"][arm] = rec

    # paired deltas vs A (same seeds)
    paired = {}
    for arm in ("B", "C", "D"):
        common = sorted(set(data[arm]) & set(data["A"]))
        if not common:
            continue
        rows = []
        for s in common:
            a, x = data["A"][s], data[arm][s]
            rows.append({
                "seed": s,
                "d_test_R@0.05": x["test"]["0.05"]["recall"] - a["test"]["0.05"]["recall"],
                "d_test_fppi@0.05": x["test"]["0.05"]["fppi"] - a["test"]["0.05"]["fppi"],
                "d_val_R@0.05": x["val"]["0.05"]["recall"] - a["val"]["0.05"]["recall"],
                "d_test_AP": x["test_ap"]["AP"] - a["test_ap"]["AP"],
                "d_val_AP": x["val_ap"]["AP"] - a["val_ap"]["AP"],
            })
        paired[arm] = {"n": len(rows), "rows": rows,
                       "mean": {k: st.mean([r[k] for r in rows]) for k in rows[0] if k != "seed"},
                       "positive": {k: sum(r[k] > 0 for r in rows) for k in rows[0] if k != "seed"}}
    out["paired_vs_A"] = paired

    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    (paths.REPORTS / "arms_summary.json").write_text(json.dumps(out, indent=2, default=float),
                                                     encoding="utf-8")

    L = [f"# CR-YOLO 四臂汇总（β={args.beta:g}，seeds {seeds}）", "",
         f"可用臂：" + "，".join(f"{a} {sorted(data[a])}" for a in data if data[a]), "",
         "## 1. val：七预算召回（mean±sd，%）", "",
         "| 臂 | " + " | ".join(f"R@{b}" for b in BUD) + " |",
         "| --- |" + " ---: |" * len(BUD)]
    for arm, rec in out["arms"].items():
        L.append(f"| {arm} | " + " | ".join(fmt(rec[f'val_R@{b}'], 1, True) for b in BUD) + " |")
    L += ["", "## 2. test：实际 FPPI（冻结 val 阈值）", "",
          "| 臂 | " + " | ".join(f"{b}" for b in BUD) + " |",
          "| --- |" + " ---: |" * len(BUD)]
    for arm, rec in out["arms"].items():
        L.append(f"| {arm} | " + " | ".join(fmt(rec[f'test_fppi@{b}'], 3) for b in BUD) + " |")
    L += ["", "## 3. test：召回（%）", "",
          "| 臂 | " + " | ".join(f"{b}" for b in BUD) + " |",
          "| --- |" + " ---: |" * len(BUD)]
    for arm, rec in out["arms"].items():
        L.append(f"| {arm} | " + " | ".join(fmt(rec[f'test_R@{b}'], 1, True) for b in BUD) + " |")
    L += ["", "## 4. test：F₂（%）", "",
          "| 臂 | " + " | ".join(f"{b}" for b in BUD) + " |",
          "| --- |" + " ---: |" * len(BUD)]
    for arm, rec in out["arms"].items():
        L.append(f"| {arm} | " + " | ".join(fmt(rec[f'test_f2@{b}'], 1, True) for b in BUD) + " |")
    L += ["", "## 5. AP 与 conf 0.25", "",
          "| 臂 | val AP | test AP | test AP50 | test AP75 | test conf.25 R / FPPI |",
          "| --- | ---: | ---: | ---: | ---: | --- |"]
    for arm, rec in out["arms"].items():
        L.append(f"| {arm} | {fmt(rec['val_AP'], 4)} | {fmt(rec['test_AP'], 4)} | "
                 f"{fmt(rec['test_AP50'], 4)} | {fmt(rec['test_AP75'], 4)} | "
                 f"{fmt(rec['test_conf025_recall'], 3, True)} / {fmt(rec['test_conf025_fppi'], 3)} |")
    L += ["", "## 6. 配对差（同 seed，臂 − A）", "",
          "| 臂 | n | Δ test R@0.05 | Δ test FPPI@0.05 | Δ val R@0.05 | Δ test AP | Δ val AP | 正号数 |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for arm, p in paired.items():
        m, pos = p["mean"], p["positive"]
        L.append(f"| {arm} | {p['n']} | {100 * m['d_test_R@0.05']:+.1f}pp | "
                 f"{m['d_test_fppi@0.05']:+.3f} | {100 * m['d_val_R@0.05']:+.1f}pp | "
                 f"{m['d_test_AP']:+.4f} | {m['d_val_AP']:+.4f} | "
                 f"{pos['d_test_R@0.05']}/{p['n']} |")
    L += ["", "Paired deltas are descriptive; the decision criteria are in `docs/PROTOCOL.md`.", ""]
    (paths.REPORTS / "arms_summary.md").write_text("\n".join(L), encoding="utf-8")

    # ---- optional LaTeX in the paper's three-table layout
    if args.tex:
        COST = {"A": ("21.79", "74.4"), "B": ("22.36", "80.0"),
                "C": ("21.79", "74.4"), "D": ("22.36", "80.0")}
        NAME = {"A": "A: YOLO26m (plain)", "B": "B: + context module",
                "C": f"C: + ranking ($\\beta$={args.beta:g})",
                "D": f"D: + both ($\\beta$={args.beta:g})"}
        arms_present = [a for a in ("A", "B", "C", "D") if a in out["arms"]]
        T = ["% Generated by src/summarise_arms.py -- do not edit by hand.",
             f"% beta={args.beta:g}, seeds={seeds}, n per arm as reported below.",
             "", r"\begin{table*}[t]", r"\centering",
             r"\caption{Conventional object detection performance of the CR-YOLO arms on the "
             r"ShoePrint test set.}", r"\label{tab:cr_detection}", r"\small",
             r"\setlength{\tabcolsep}{5pt}", "", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lccccccccccc}", r"\toprule",
             "Model & Input size & P (\\%) $\\uparrow$ & R (\\%) $\\uparrow$ & F$_1$ (\\%) $\\uparrow$ "
             "& F$_2$ (\\%) $\\uparrow$ & mAP$_{50}$ (\\%) $\\uparrow$ & mAP$_{75}$ (\\%) $\\uparrow$ "
             "& mAP$_{50:95}$ (\\%) $\\uparrow$ & Params (M) $\\downarrow$ & FLOPs (G) $\\downarrow$ "
             "& Latency (ms) $\\downarrow$ \\\\", r"\midrule"]
        for a in arms_present:
            r = out["arms"][a]
            T.append(f"{NAME[a]} & 640 & {r['test_conf025_precision'][0]*100:.1f} & "
                     f"{r['test_conf025_recall'][0]*100:.1f} & {r['test_conf025_f1'][0]*100:.1f} & "
                     f"{r['test_conf025_f2'][0]*100:.1f} & {r['test_AP50'][0]*100:.1f} & "
                     f"{r['test_AP75'][0]*100:.1f} & {r['test_AP'][0]*100:.1f} & "
                     f"{COST[a][0]} & {COST[a][1]} & -- \\\\")
        T += [r"\bottomrule", r"\end{tabular}%", "}", "", r"\vspace{2pt}",
              r"\parbox{\textwidth}{\footnotesize", r"\textit{Note:} ...", "}", r"\end{table*}", ""]
        # table 2
        T += [r"\begin{table*}[t]", r"\centering",
              r"\caption{Test-set detection performance at operating points selected under "
              r"different validation-set FPPI budgets.}", r"\label{tab:cr_fppi_budget_results}",
              r"\small", r"\setlength{\tabcolsep}{4pt}", "", r"\resizebox{\textwidth}{!}{%",
              r"\begin{tabular}{lccccccc}", r"\toprule",
              r"& \multicolumn{7}{c}{Validation-set FPPI budget} \\", r"\cmidrule(lr){2-8}",
              "Model & 0.05 & 0.1 & 0.2 & 0.5 & 1.0 & 2.0 & 5.0 \\\\", r"\midrule"]
        for a in arms_present:
            r = out["arms"][a]
            cells = []
            for b in BUD:
                f_, rr, f2 = r[f"test_fppi@{b}"][0], r[f"test_R@{b}"][0], r[f"test_f2@{b}"][0]
                cells.append(f"{f_:.3f} / {rr*100:.1f} / {f2*100:.1f}")
            T.append(f"{NAME[a]} & " + " & ".join(cells) + r" \\")
        T += [r"\bottomrule", r"\end{tabular}%", "}", "", r"\vspace{2pt}",
              r"\parbox{\textwidth}{\footnotesize", r"\textit{Note:} ...", "}", r"\end{table*}", ""]
        Path(args.tex).write_text("\n".join(T) + "\n", encoding="utf-8")
        print("written", args.tex)

    print("\n".join(L[:14]))
    print("...")
    print("written reports/arms_summary.md / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
