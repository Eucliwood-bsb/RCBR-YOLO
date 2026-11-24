# Method (paper draft, code-consistent)

This note mirrors `src/cr_modules.py` and `src/cr_rank_loss.py`. It contains no
experimental results.

## Task

Local-image detection of shoe-print defects: the detector receives a 640×640
local crop and predicts boxes, category labels and confidence scores. No
reference image is used at inference. Training, validation and test are
disjoint at the source-image level.

## Multi-distance relative context (MRC)

For a feature map ``F ∈ R^{C×H×W}`` and a distance ``r ∈ {1,2,4}``, the context
at position ``p`` is the mean of the eight neighbours of a 3×3 grid dilated by
``r``:

```
C_r(p) = Σ_{δ∈D} v(p+rδ) F(p+rδ) / Σ_{δ∈D} v(p+rδ),
D = {-1,0,1}² \ {(0,0)}
```

``v(q)=1`` for a valid position and 0 outside the map, so boundary positions use
only their valid neighbours and the denominator is position-specific. The centre
is never included; when no neighbour is valid the context is zero.

The relative feature is ``D_r(p) = F(p) − C_r(p)``. A shared scoring network ``g``
(of the form 1×1 conv → SiLU → 1×1 conv) produces softmax weights over distances:

```
α_r(p) = exp(g([F(p), C_r(p), D_r(p)])) / Σ_s exp(g([F(p), C_s(p), D_s(p)]))
C̄ = Σ_r α_r C_r,    D̄ = Σ_r α_r D_r
```

and a fusion network ``h`` (1×1 conv → SiLU → 1×1 conv) gives

```
F_out(p) = F(p) + λ h([F(p), C̄(p), D̄(p)])
```

with a learnable scalar ``λ`` initialised to 0.01. Independent blocks are inserted
at layer 16 (P3, 256×80²) and layer 19 (P4, 512×40²), i.e. immediately before the
`Detect` head consumes those tensors. The wrapper keeps the module list indices,
`Detect.f = [16, 19, 22]` and all output shapes unchanged. The P5 node is not
wrapped; because the bottom-up path is computed from the wrapped P3/P4 outputs,
the P5 feature is nevertheless affected.

Because the eight neighbours are samples of a dilated 3×3 grid, diagonal
neighbours are at Euclidean distance ``r√2``; "eight neighbours at distance r"
should be read as Chebyshev spacing, not as a ring of radius ``r``.

## Class-conditioned hard-background ranking (CHBR)

Let ``z_{b,i,c}`` be the classification logit of anchor ``i`` for class ``c`` in
image ``b``. Positives are the anchors assigned by the detector's one-to-many TAL
assigner (``target_scores > 0``). A candidate is *reliable background* when

* it is not positive for **any** class,
* its decoded box has IoU ≤ 0.5 with every valid GT box, and
* its box centre is outside every GT box grown by 2 pixels.

No padding, crowd or ignore masks are applied by this implementation.

For each image and class with both sets non-empty, keep the ``K₊=32`` positive
anchors with the lowest logits and the ``K₋=64`` background anchors with the
highest logits, and average a softplus ranking loss over their Cartesian product:

```
L^{rank}_{b,c} = mean_{p∈P, n∈N} log(1 + exp(m + z_{b,n,c} − z_{b,p,c}))
```

with margin ``m = 0.5``. The image-class losses are averaged over image-class
pairs that contain at least one valid pair; with no valid pair the term is a
graph-connected zero. The total objective is

```
L = L_det + β · L_rank
```

applied to the one-to-many branch only, after the box/cls/dfl gains. ``β = 0``
takes the stock code path bit-identically (verified by `check_rank_loss.py`).
The mask is recomputed from the current (detached) predictions at every step, so
the background set drifts during training; it is not a fixed external mask.

## Inference and evaluation

The ranking term is training-only. Inference uses the one-to-many branch of the
tested Ultralytics build (`Detect.end2end = False`), followed by class-aware NMS
at IoU 0.7. Evaluation uses seven validation FPPI budgets with frozen thresholds;
see `docs/PROTOCOL.md`.
