"""Insert the CR relative-context blocks into YOLO26m without touching its code.

The block itself lives in `cr_modules.py`; this module only wraps the two
head-input stages in place, so that

  * the Sequential indices do not move (Detect keeps `f = [16, 19, 22]`),
  * the Detect interface (list of 3 tensors, same channels) does not change,
  * the stock criterion, validator, export and NMS code paths keep working.

`CtxWrap` is defined at module level (not in __main__) so a saved model can be unpickled in a new
process as long as this directory is on sys.path.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

import paths
from cr_modules import RelativeContextBlock


def cr_source_sha() -> str:
    """sha256 of the block implementation actually imported."""
    return hashlib.sha256(Path(__file__).with_name("cr_modules.py").read_bytes()).hexdigest()


def head_in_channels(model: nn.Module) -> list[int]:
    """Input channels of each Detect level, read from the head itself (no dummy forward)."""
    det = model.model[-1]
    ch = []
    for seq in det.cv2:  # box branch always starts with ultralytics Conv
        first = seq[0]
        conv = getattr(first, "conv", first)
        ch.append(int(conv.in_channels))
    return ch


class CtxWrap(nn.Module):
    """inner(x) -> context -> out, carrying the metadata the model list expects."""

    def __init__(self, inner: nn.Module, ctx: nn.Module, index: int) -> None:
        super().__init__()
        self.inner = inner
        self.ctx = ctx
        self.i = index
        self.f = getattr(inner, "f", -1)
        self.type = f"{type(inner).__name__}+RelativeContextBlock"
        self.np = sum(p.numel() for p in inner.parameters()) + sum(p.numel() for p in ctx.parameters())

    def forward(self, x):
        return self.ctx(self.inner(x))


def insert_context(model: nn.Module, levels=(16, 19), radii=(1, 2, 4),
                   residual_init: float = 0.01, verbose: bool = False) -> list[tuple[int, int]]:
    """Wrap the given layer indices in place. Returns [(level, channels), ...]."""
    det = model.model[-1]
    mapping = list(zip(list(det.f), head_in_channels(model)))
    made = []
    for level, c in mapping:
        if level not in levels:
            continue
        inner = model.model[level]
        if isinstance(inner, CtxWrap):
            raise RuntimeError(f"layer {level} is already wrapped")
        ctx = RelativeContextBlock(c, radii=radii, residual_init=residual_init)
        model.model[level] = CtxWrap(inner, ctx, level)
        made.append((level, c))
    if verbose:
        for level, c in made:
            print(f"wrapped layer {level}: {c} channels, radii={tuple(radii)}, lambda0={residual_init}")
    missing = [lv for lv, _ in mapping if lv in levels and lv not in [m[0] for m in made]]
    if missing:
        raise RuntimeError(f"requested levels not found: {missing}")
    return made


def set_residual_scale(model: nn.Module, value: float) -> int:
    n = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, RelativeContextBlock):
                m.residual_scale.fill_(float(value))
                n += 1
    return n


def context_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if ".ctx." in n]


def remap_wrapped_keys(sd: dict, levels) -> dict:
    """Rename `model.<lv>.foo` -> `model.<lv>.inner.foo` for wrapped levels.

    Needed because wrapping moves the original module under `.inner`, so a plain
    `load_state_dict` from an unwrapped checkpoint would report every key of those layers as
    both unexpected and missing. During training the order is the other way round (load, then
    wrap), so this is only for state-dict round-trips.
    """
    out = dict(sd)
    for lv in levels:
        p = f"model.{lv}."
        for k in [k for k in out if k.startswith(p) and not k.startswith(p + "inner.")]:
            out[p + "inner." + k[len(p):]] = out.pop(k)
    return out


def unwrap_predictions(out):
    """model(x) in eval returns (decoded, raw) or decoded; return (decoded, raw|None)."""
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        return out[0], out[1]
    return out, None


def load_tiles(n: int = 8, size: int = 640, img_dir: Path | None = None) -> torch.Tensor:
    """Real validation tiles, resized to `size` (comparison only, not an evaluation)."""
    import cv2
    import numpy as np

    img_dir = Path(img_dir) if img_dir is not None else paths.VAL_IMAGES
    files = sorted(img_dir.glob("*.png"))[:n]
    if not files:
        raise FileNotFoundError(img_dir)
    xs = []
    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        xs.append(torch.from_numpy(im).permute(2, 0, 1))
    return torch.stack(xs), [f.name for f in files]
