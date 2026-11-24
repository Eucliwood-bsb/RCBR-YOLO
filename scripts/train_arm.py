#!/usr/bin/env python3
"""Train one RCBR-YOLO arm/seed with the recorded recipe (see docs/PROTOCOL.md).

  A  : stock YOLO26m (no context block, stock loss)
  B  : P3/P4 RelativeContextBlock, stock loss
  C  : stock model, L_det + beta * L_rank on one-to-many
  D  : both

Injection points: context blocks in `setup_model` (after pretrained weights load), criterion and
loss-statistics wrapper in `set_model_attributes` (after hyp/args, before optimizer and EMA).

Writes runs/arms/<tag>/weights/last.pt and runs/arms/<tag>/epoch_log.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
sys.path.insert(0, str(ROOT / "src"))

from cr_integration import context_named_parameters, insert_context  # noqa: E402
from cr_rank_loss import CRRankE2ELoss  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402
from ultralytics.utils.torch_utils import unwrap_model  # noqa: E402

import paths  # noqa: E402

DATA = paths.DATA_YAML
BASE = paths.WEIGHTS
STAT_KEYS = ("rank_aux", "rank_pairs", "rank_pos", "rank_neg", "rank_excluded", "rank_image_classes")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


_ARM = "B"
_BETA = 0.0


class ArmTrainer(DetectionTrainer):
    def __init__(self, *a, **kw):
        self.arm = _ARM
        self.beta = float(_BETA)
        self.info: dict = {}
        self.acc: dict = {"n_batches": 0, **{k: 0.0 for k in STAT_KEYS}}
        super().__init__(*a, **kw)

    def setup_model(self):
        ckpt = super().setup_model()
        m = unwrap_model(self.model)
        det = m.model[-1]
        self.info = {"arm": self.arm, "beta": self.beta, "nc": int(det.nc),
                     "f": list(det.f), "head_ch": None, "wrapped": []}
        from cr_integration import head_in_channels
        self.info["head_ch"] = head_in_channels(m)
        if self.arm in ("B", "D"):
            self.info["wrapped"] = insert_context(m, levels=(16, 19), radii=(1, 2, 4))
            self.info["head_ch_after"] = head_in_channels(m)
            self.info["f_after"] = list(m.model[-1].f)
        return ckpt

    def set_model_attributes(self):
        super().set_model_attributes()
        m = unwrap_model(self.model)
        if self.arm in ("C", "D"):
            if self.beta <= 0:
                raise ValueError("arm C/D requires beta > 0")
            m.criterion = CRRankE2ELoss(m, beta=self.beta, stats={})
            self.info["criterion"] = type(m.criterion).__name__
            self.info["beta"] = self.beta
        else:
            self.info["criterion"] = "stock"
        self.info["params"] = int(sum(p.numel() for p in m.parameters()))
        self.info["ctx_params"] = int(sum(p.numel() for _, p in context_named_parameters(m)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("A", "B", "C", "D"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--data", default="", help="dataset yaml (default: RCBR_DATA_YAML)")
    ap.add_argument("--weights", default="", help="starting checkpoint (default: RCBR_WEIGHTS)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from ultralytics import YOLO

    global _ARM, _BETA, DATA, BASE
    _ARM, _BETA = args.arm, args.beta
    if args.data:
        DATA = Path(args.data)
    if args.weights:
        BASE = Path(args.weights)
    tag = args.tag or f"{args.arm}_s{args.seed}" + (f"_b{args.beta:g}" if args.arm in ("C", "D") else "")
    run_dir = paths.RUNS / "arms" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "arm": args.arm, "seed": args.seed, "beta": args.beta, "epochs": args.epochs,
        "started": datetime.now().isoformat(timespec="seconds"),
        "data": str(DATA), "data_sha256": sha(DATA), "ckpt": str(BASE), "ckpt_sha256": sha(BASE),
        "cr_modules_sha256": sha(ROOT / "src" / "cr_modules.py"),
        "protocol": "docs/PROTOCOL.md",
        "recipe": "SGD lr0=5e-3 lrf=0.01 momentum=.937 wd=5e-4 warmup5 nbs64 batch8 amp deterministic; "
                  "mosaic0 close_mosaic10 scale.5 translate.1 fliplr.5; "
                  "erasing/randaugment are configured but inert for detection; "
                  "80ep patience80 last.pt",
        "env": {"torch": torch.__version__, "cuda": torch.cuda.get_device_name(0)},
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log = (run_dir / "epoch_log.jsonl").open("a", encoding="utf-8")
    holder: dict = {}

    def _start(tr):
        m = unwrap_model(tr.model)
        gp = context_named_parameters(m)
        ids = [id(p) for g in tr.optimizer.param_groups for p in g["params"]]
        info = {**tr.info,
                "optimizer_groups": len(tr.optimizer.param_groups),
                "ctx_in_optimizer": sum(id(p) in ids for _, p in gp),
                "ctx_total": len(gp),
                "criterion_at_start": type(getattr(m, "criterion", None)).__name__,
                "beta_at_start": getattr(getattr(m, "criterion", None), "beta", None)}
        meta["trainer"] = info
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("[trainer-check]", json.dumps(info)[:500], flush=True)

    def _epoch_end(tr):
        m = unwrap_model(tr.model)
        st = getattr(getattr(m, "criterion", None), "stats", {}) or {}
        nb = max(int(st.get("n_batches", 0)), 1)
        row = {"epoch": int(tr.epoch) + 1, "lr": float(tr.optimizer.param_groups[0]["lr"]),
               "rank_aux_mean": float(st.get("sum_rank_aux", 0.0)) / nb,
               "rank_pairs": float(st.get("sum_rank_pairs", 0.0)),
               "rank_pos": float(st.get("sum_rank_pos", 0.0)),
               "rank_neg": float(st.get("sum_rank_neg", 0.0)),
               "rank_excluded": float(st.get("sum_rank_excluded", 0.0)),
               "rank_image_classes": float(st.get("sum_rank_image_classes", 0.0)),
               "loss": {k: float(v) for k, v in (getattr(tr, "tloss", {}) or {}).items()},
               "metrics": {k: float(v) for k, v in (getattr(tr, "metrics", {}) or {}).items()
                           if isinstance(v, (int, float))}}
        log.write(json.dumps(row, default=float) + "\n")
        log.flush()
        for k in list(st):
            if k.startswith("sum_"):
                st[k] = 0.0
        st["n_batches"] = 0

    model = YOLO(str(BASE))
    model.add_callback("on_train_start", _start)
    model.add_callback("on_train_epoch_end", _epoch_end)
    t0 = time.time()
    model.train(trainer=ArmTrainer, data=str(DATA), epochs=args.epochs, imgsz=640, batch=args.batch,
                device=args.device, workers=args.workers, optimizer="SGD", lr0=0.005, lrf=0.01, momentum=0.937,
                weight_decay=0.0005, warmup_epochs=5.0, nbs=64, amp=True, deterministic=True,
                patience=args.epochs, close_mosaic=10, mosaic=0.0, scale=0.5, translate=0.1,
                degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mixup=0.0,
                copy_paste=0.0, erasing=0.4, auto_augment="randaugment", plots=False,
                save_period=-1, cache=False, val=True, seed=args.seed, max_det=300,
                project=str(paths.RUNS / "arms"), name=tag, exist_ok=True, verbose=True)

    last = run_dir / "weights" / "last.pt"
    meta.update({"status": "ok" if last.is_file() else "missing_last",
                 "elapsed_s": round(time.time() - t0, 1),
                 "last_sha256": sha(last) if last.is_file() else None,
                 "last_bytes": last.stat().st_size if last.is_file() else 0,
                 "cuda_max_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                 "ended": datetime.now().isoformat(timespec="seconds")})
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.close()
    print("[done]", tag, meta["status"], meta["elapsed_s"], flush=True)
    return 0 if meta["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
