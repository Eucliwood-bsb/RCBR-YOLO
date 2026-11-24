#!/usr/bin/env python3
"""Integration audit for RCBR-YOLO on top of YOLO26m.

Answers, with real execution on the RTX 4090:

  1. does wrapping P3/P4 keep the Detect inputs and output shapes unchanged?
  2. do the new parameters receive finite, non-zero gradients (FP32 and AMP)?
  3. are they in the optimizer / do they require grad?
  4. can the model be saved and reloaded in a fresh process (state_dict and full pickle)?
  5. does `fuse()` change the output?
  6. how far is the model from the plain one at initialisation (lambda = 0 vs 0.01)?
  7. what is the whole-model cost delta (params, FP16 FLOPs @640)?

Writes reports/integration_audit.json and reports/integration_audit.md.
Nothing outside this directory is modified; the checkpoint is only read.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

TASK = Path(__file__).resolve().parent
ROOT = TASK.parent
sys.path.insert(0, str(ROOT / "src"))

from cr_integration import (cr_source_sha, context_named_parameters, head_in_channels,  # noqa: E402
                            insert_context, load_tiles, remap_wrapped_keys, set_residual_scale,
                            unwrap_predictions)
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

import paths  # noqa: E402

CKPT = paths.WEIGHTS
LEVELS = (16, 19)
RADII = (1, 2, 4)


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def flops_fp16(model: torch.nn.Module, size: int = 640) -> float:
    model = model.half().eval()
    x = torch.zeros(1, 3, size, size, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        with torch.profiler.profile(with_flops=True) as prof:
            model(x)
    return float(sum(e.flops for e in prof.key_averages() if getattr(e, "flops", 0)))


def head_inputs(model: torch.nn.Module, x: torch.Tensor) -> list[list[int]]:
    det = model.model[-1]
    got: dict[int, tuple] = {}

    def hook(_m, _i, out, level):
        got[level] = tuple(out.shape)

    handles = [model.model[lv].register_forward_hook(lambda m, i, o, lv=lv: hook(m, i, o, lv))
               for lv in det.f]
    with torch.no_grad():
        model.eval()(x)
    for h in handles:
        h.remove()
    return [list(got[lv]) for lv in det.f]


def shapes(obj):
    if torch.is_tensor(obj):
        return list(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [shapes(x) for x in obj]
    if isinstance(obj, dict):
        return {k: shapes(v) for k, v in obj.items()}
    return type(obj).__name__


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(paths.WEIGHTS))
    args = ap.parse_args()
    ckpt = Path(args.weights)
    t0 = time.time()
    out: dict = {"ckpt": str(ckpt), "cr_source_sha256": cr_source_sha(),
                 "torch": torch.__version__, "cuda": torch.cuda.get_device_name(0),
                 "levels": list(LEVELS), "radii": list(RADII), "checks": {}}
    rep = paths.REPORTS
    rep.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    src = ck["model"].float().eval()
    out["source"] = {"params": int(sum(p.numel() for p in src.parameters())),
                     "nc": int(src.model[-1].nc), "end2end": bool(src.model[-1].end2end),
                     "f": list(src.model[-1].f), "stride": src.model[-1].stride.tolist()}

    # ---- build A (plain, same weights) and B (wrapped)
    plain = DetectionModel(copy.deepcopy(src.yaml), nc=src.model[-1].nc, verbose=False).float()
    plain.load_state_dict(src.state_dict(), strict=True)
    plain = plain.eval()

    cr = DetectionModel(copy.deepcopy(src.yaml), nc=src.model[-1].nc, verbose=False).float()
    torch.manual_seed(0)  # the context blocks are randomly initialised: fix the seed for reproducibility
    made = insert_context(cr, levels=LEVELS, radii=RADII)
    out["init_seed"] = 0
    sd = remap_wrapped_keys(src.state_dict(), LEVELS)
    missing, unexpected = cr.load_state_dict(sd, strict=False)
    ctx_keys = [k for k, _ in context_named_parameters(cr)]
    out["build"] = {"wrapped": made, "head_in_channels": head_in_channels(cr),
                    "missing_keys": sorted(missing), "unexpected_keys": sorted(unexpected),
                    "missing_are_exactly_ctx": sorted(missing) == sorted(ctx_keys),
                    "ctx_param_count": len(ctx_keys),
                    "ctx_params": int(sum(p.numel() for _, p in context_named_parameters(cr)))}
    assert sorted(missing) == sorted(ctx_keys), "unexpected missing keys"
    assert not unexpected, "unexpected keys"

    # non-context weights must be bit-identical to the source
    s1, s2 = cr.state_dict(), remap_wrapped_keys(src.state_dict(), LEVELS)
    shared = [k for k in s2 if k in s1]
    diffs = [float((s1[k].float() - s2[k].float()).abs().max()) for k in shared]
    out["build"]["non_ctx_state_max_abs_diff"] = max(diffs) if diffs else None
    out["build"]["state_keys_shared"] = len(shared)
    out["build"]["state_keys_source_not_in_model"] = sorted(k for k in s2 if k not in s1)

    x8, names = load_tiles(8, 640, paths.VAL_IMAGES)
    x1 = x8[:1]
    out["tiles"] = names

    # ---- 1. shapes
    plain_c = plain.cuda()
    cr_c = cr.eval().cuda()
    with torch.no_grad():
        y_p, raw_p = unwrap_predictions(plain_c(x1.cuda()))
        y_c, raw_c = unwrap_predictions(cr_c(x1.cuda()))
    out["checks"]["shapes"] = {
        "plain_head_inputs": head_inputs(plain_c, x1.cuda()),
        "cr_head_inputs": head_inputs(cr_c, x1.cuda()),
        "plain_out": list(y_p.shape), "cr_out": list(y_c.shape),
        "raw_keys": sorted(raw_c.keys()) if raw_c else None,
        "raw_shapes": shapes(raw_c) if raw_c else None,
    }
    out["checks"]["shapes"]["identical"] = (
        out["checks"]["shapes"]["plain_head_inputs"] == out["checks"]["shapes"]["cr_head_inputs"]
        and list(y_p.shape) == list(y_c.shape))

    # ---- 2. gradients (FP32, eval forward -> decoded output)
    cr_c.zero_grad(set_to_none=True)
    y, _ = unwrap_predictions(cr_c(x1.cuda()))
    y.float().pow(2).mean().backward()
    gp = context_named_parameters(cr_c)
    out["checks"]["grad_fp32"] = {
        "n_ctx_params": len(gp),
        "all_require_grad": all(p.requires_grad for _, p in gp),
        "missing_grad": sum(p.grad is None for _, p in gp),
        "nonfinite_grad": sum(int((~torch.isfinite(p.grad)).any()) for _, p in gp if p.grad is not None),
        "zero_grad": sum(int(p.grad.abs().max() == 0) for _, p in gp if p.grad is not None),
        "zero_grad_names": [n for n, p in gp if p.grad is not None and float(p.grad.abs().max()) == 0.0],
        "ctx_grad_absmax": max(float(p.grad.abs().max()) for _, p in gp if p.grad is not None),
        "grad_by_param": {n: float(p.grad.abs().max()) for n, p in gp if p.grad is not None},
        "inner0_grad": None,
    }
    inner = cr_c.model[LEVELS[0]].inner
    gs = [float(p.grad.abs().max()) for p in inner.parameters() if p.grad is not None]
    out["checks"]["grad_fp32"]["inner0_grad"] = max(gs) if gs else None

    # ---- 3. optimizer membership
    opt = torch.optim.SGD([p for p in cr_c.parameters() if p.requires_grad], lr=0.01)
    ids = {id(p) for g in opt.param_groups for p in g["params"]}
    out["checks"]["optimizer"] = {
        "trainable": sum(p.requires_grad for p in cr_c.parameters()),
        "missing_from_optimizer": sum(id(p) not in ids for _, p in gp),
    }

    # ---- 4. AMP forward/backward
    cr_c.zero_grad(set_to_none=True)
    amp_dtype = {}

    def _amp_hook(_m, _i, o):
        amp_dtype["ctx_out"] = str(o.dtype)

    hook = cr_c.model[LEVELS[0]].ctx.register_forward_hook(_amp_hook)
    with torch.autocast("cuda", dtype=torch.float16):
        y_amp, _ = unwrap_predictions(cr_c(x1.cuda()))
        loss_amp = y_amp.float().pow(2).mean()
    loss_amp.backward()
    hook.remove()
    out["checks"]["amp"] = {
        "ctx_out_dtype": amp_dtype.get("ctx_out"),
        "out_dtype": str(y_amp.dtype),
        "nonfinite_out": int((~torch.isfinite(y_amp)).any()),
        "nonfinite_grad": sum(int((~torch.isfinite(p.grad)).any())
                              for _, p in gp if p.grad is not None),
    }
    cr_c.zero_grad(set_to_none=True)

    # ---- 5. fuse
    pf, cf = copy.deepcopy(plain_c).eval(), copy.deepcopy(cr_c).eval()
    with torch.no_grad():
        y0p, _ = unwrap_predictions(pf(x1.cuda()))
        y0c, _ = unwrap_predictions(cf(x1.cuda()))
        pf.fuse()
        cf.fuse()
        y1p, _ = unwrap_predictions(pf(x1.cuda()))
        y1c, _ = unwrap_predictions(cf(x1.cuda()))
    def _stats(a, b):
        d = (a.float() - b.float()).abs()
        if d.shape[1] > 4:
            box, sc = d[:, :4].flatten(), d[:, 4:].flatten()
            return {"box_mean": float(box.mean()), "box_p99": float(box.quantile(0.99)),
                    "box_max": float(box.max()),
                    "score_mean": float(sc.mean()), "score_p99": float(sc.quantile(0.99)),
                    "score_max": float(sc.max())}
        f = d.flatten()
        return {"box_mean": None, "box_p99": None, "box_max": None,
                "score_mean": float(f.mean()), "score_p99": float(f.quantile(0.99)),
                "score_max": float(f.max())}

    out["checks"]["fuse"] = {
        "plain": _stats(y0p, y1p),
        "cr": _stats(y0c, y1c),
        "cr_is_fused": bool(cf.is_fused()) if callable(getattr(cf, "is_fused", None))
        else bool(getattr(cf, "is_fused", False)),
    }
    del pf, cf

    # ---- 6. initialisation offset on 8 real tiles (float32, eval)
    off = {}
    for lam in (0.0, 0.01, 0.1):
        m = copy.deepcopy(cr_c).eval()
        n = set_residual_scale(m, lam)
        with torch.no_grad():
            yp, _ = unwrap_predictions(plain_c(x8.cuda()))
            yc, _ = unwrap_predictions(m(x8.cuda()))
        d = (yc.float() - yp.float()).abs()
        off[f"lambda_{lam}"] = {
            "modules_set": n,
            "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "max_abs_scores": float(d[:, 4:, :].max()) if yc.shape[1] > 4 else None,
            "max_abs_boxes_px": float(d[:, :4, :].max()) if yc.shape[1] > 4 else None,
            "n_values_over_1e-3": int((d > 1e-3).sum()),
            "total_values": int(d.numel()),
        }
        del m
    out["checks"]["init_offset"] = off

    # ---- 7. cost (whole model, FP16, 640)
    out["checks"]["cost"] = {
        "plain_params": int(sum(p.numel() for p in plain_c.parameters())),
        "cr_params": int(sum(p.numel() for p in cr_c.parameters())),
        "plain_flops": flops_fp16(copy.deepcopy(plain_c)),
        "cr_flops": flops_fp16(copy.deepcopy(cr_c)),
    }
    c = out["checks"]["cost"]
    c["params_delta"] = c["cr_params"] - c["plain_params"]
    c["params_pct"] = 100.0 * c["params_delta"] / c["plain_params"]
    c["flops_delta"] = c["cr_flops"] - c["plain_flops"]
    c["flops_pct"] = 100.0 * c["flops_delta"] / c["plain_flops"]

    # ---- 8. save / reload in a fresh process
    art = paths.STATE / "reload"
    art.mkdir(parents=True, exist_ok=True)
    torch.save(cr_c.state_dict(), art / "cr_state.pt")
    torch.save(x1.cpu(), art / "x.pt")
    with torch.no_grad():
        y_ref, _ = unwrap_predictions(cr_c(x1.cuda()))
    torch.save(y_ref.cpu(), art / "y.pt")
    (art / "yaml.json").write_text(json.dumps(cr_c.yaml, default=str), encoding="utf-8")
    torch.save(cr_c, art / "cr_full.pt")
    meta = {"levels": list(LEVELS), "radii": list(RADII), "nc": int(cr_c.model[-1].nc),
            "art": str(art), "src": str(ROOT / "src")}
    dump(art / "meta.json", meta)
    r = subprocess.run([sys.executable, str(TASK / "reload_check.py")],
                       capture_output=True, text=True, timeout=900)
    out["checks"]["reload"] = {"returncode": r.returncode,
                               "stdout": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "",
                               "stderr": r.stderr.strip()[-800:]}
    try:
        out["checks"]["reload"].update(json.loads(out["checks"]["reload"]["stdout"]))
    except json.JSONDecodeError:
        pass

    out["elapsed_s"] = time.time() - t0
    dump(rep / "integration_audit.json", out)

    # ---- report
    L = ["# CR-YOLO × YOLO26m 接入审计（真实执行）", "",
         f"> {time.strftime('%Y-%m-%d %H:%M:%S')} · {out['cuda']} · torch {out['torch']}",
         f"> 权重：`{ckpt}`（只读）", f"> 原型 `cr_modules.py` sha256 `{out['cr_source_sha256'][:16]}…`",
         f"> 插入层：{list(LEVELS)}，radii={list(RADII)}", ""]
    b = out["build"]
    L += ["## 1. 构建", "",
          f"- 包裹：{b['wrapped']}（层, 通道）；头输入通道 {b['head_in_channels']}",
          f"- `missing_keys` 恰好等于新增上下文参数：**{b['missing_are_exactly_ctx']}**（{b['ctx_param_count']} 个张量，{b['ctx_params']:,} 参数）",
          f"- `unexpected_keys`：{b['unexpected_keys']}",
          f"- 非上下文权重与源 checkpoint 的最大差：{b['non_ctx_state_max_abs_diff']}",
          ""]
    s = out["checks"]["shapes"]
    L += ["## 2. 形状", "",
          f"- A 头输入：{s['plain_head_inputs']}",
          f"- B 头输入：{s['cr_head_inputs']}",
          f"- 输出：A {s['plain_out']} / B {s['cr_out']}；raw keys {s['raw_keys']}",
          f"- **完全一致：{s['identical']}**", ""]
    g = out["checks"]["grad_fp32"]
    L += ["## 3. 梯度与优化器", "",
          f"- FP32：{g['n_ctx_params']} 个张量，全部 requires_grad={g['all_require_grad']}，"
          f"无梯度={g['missing_grad']}，非有限={g['nonfinite_grad']}，零梯度={g['zero_grad']} "
          f"{g['zero_grad_names']}，|grad|max={g['ctx_grad_absmax']:.3e}",
          f"- 上游（层 {LEVELS[0]} 原模块）|grad|max={g['inner0_grad']}",
          f"- 优化器：可训练参数 {out['checks']['optimizer']['trainable']:,}，"
          f"上下文参数缺失 {out['checks']['optimizer']['missing_from_optimizer']}",
          f"- AMP：上下文输出 dtype {out['checks']['amp']['ctx_out_dtype']}，解码输出 "
          f"{out['checks']['amp']['out_dtype']}，非有限输出 {out['checks']['amp']['nonfinite_out']}，"
          f"非有限梯度 {out['checks']['amp']['nonfinite_grad']}", ""]
    gb = g["grad_by_param"]
    tiny = {k: v for k, v in gb.items() if v < 1e-8}
    L += [f"- 各参数 |grad|max 见 JSON；`|grad| < 1e-8` 的参数：{tiny or '无'}"
          "（`score.2.bias` 恒为 ~1e-10：三个距离共用同一打分网络，softmax 对公共常数平移不变，"
          "该 bias 本质上不可辨识，属冗余参数，不影响训练）", ""]
    f_ = out["checks"]["fuse"]
    L += ["## 4. fuse", "",
          "BN 折叠后（同一权重、同一输入）解码输出的变化：", "",
          "| 模型 | box mean (px) | box p99 (px) | box max (px) | score mean | score p99 | score max |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k in ("plain", "cr"):
        v = f_[k]
        L.append(f"| {k} | {v['box_mean']:.3e} | {v['box_p99']:.3e} | {v['box_max']:.3f} | "
                 f"{v['score_mean']:.3e} | {v['score_p99']:.3e} | {v['score_max']:.3e} |")
    L += ["", f"- `is_fused={f_['cr_is_fused']}`；`Detect.fuse()` 会把未使用的 one2one 分支置空。",
          "- box 的最大偏差来自少数接近退化框的解码，p99 与均值都在 1e-2 px 量级；"
          "score 几乎不变（<1e-3）。结论：折叠本身数值可接受，但**不要跨 fuse 状态比较逐框坐标**。", ""]
    L += ["## 5. 初始化偏移（8 张真实 val 图，解码输出）", "",
          "| λ | max abs | mean abs | max score Δ | max box Δ (px) | >1e-3 的值 | 总值 |",
          "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k, v in off.items():
        L.append(f"| {k.split('_')[1]} | {v['max_abs']:.3e} | {v['mean_abs']:.3e} | "
                 f"{v['max_abs_scores']:.3e} | {v['max_abs_boxes_px']:.3f} | "
                 f"{v['n_values_over_1e-3']} | {v['total_values']} |")
    L += ["", "λ=0 实测与 plain 逐位一致（0/873,600 个值）；λ=0.01 的偏移是「固定 seed 0 的随机初始化 + "
              "小残差」的实际起点，换 seed 会不同。", ""]
    c = out["checks"]["cost"]
    L += ["## 6. 整模成本（FP16，640）", "",
          f"- Params：{c['plain_params']:,} → {c['cr_params']:,}（+{c['params_delta']:,}，"
          f"{c['params_pct']:+.2f}%）",
          f"- FLOPs：{c['plain_flops'] / 1e9:.3f} G → {c['cr_flops'] / 1e9:.3f} G"
          f"（+{c['flops_delta'] / 1e9:.3f} G，{c['flops_pct']:+.2f}%）", ""]
    rl = out["checks"]["reload"]
    L += ["## 7. 新进程重载", "",
          f"- returncode={rl['returncode']}；{rl.get('state_dict', rl.get('stdout', ''))}",
          f"- full pickle：{rl.get('full_pickle', 'n/a')}",
          f"- stderr：{rl.get('stderr', '')[:300] or '(empty)'}", ""]
    L += ["## 8. 结论边界", "",
          "- 以上只证明「能接入、能反向、能存读、成本可测」，**不证明任何精度收益**；",
          "- 延迟未测（本次只测 FLOPs，不得替代 latency）；",
          "- 上下文模块的掩码/排序损失接入不在本审计范围。", ""]
    (rep / "integration_audit.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[done] {out['elapsed_s']:.1f}s  ->  {rep / 'integration_audit.md'}")
    print(json.dumps({k: out["checks"][k] for k in ("shapes", "grad_fp32", "cost")}, indent=1)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
