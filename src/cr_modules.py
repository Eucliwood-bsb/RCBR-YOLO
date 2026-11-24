"""Relative-context block (MRC) and class-conditioned hard-background ranking (CHBR).

``RelativeContextBlock`` is inserted into the detector by ``cr_integration.py``.
The ranking loss is re-implemented in vectorised form in ``cr_rank_loss.py``;
the function below is kept as the reference prototype used by the unit tests.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RelativeContextBlock(nn.Module):
    """Aggregate sparse multi-distance context with residual fusion.

    Input and output:
        x: [batch, channels, height, width]

    At each distance, the eight neighboring positions of a 3x3 grid
    are averaged. Invalid positions beyond the feature map are
    excluded from the denominator; the center is always excluded.
    """

    def __init__(
        self,
        channels: int,
        radii: Sequence[int] = (1, 2, 4),
        hidden_channels: int | None = None,
        residual_init: float = 0.01,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")
        if not radii or any(int(r) != r or r <= 0 for r in radii):
            raise ValueError("radii must contain positive integers")
        if len(set(radii)) != len(radii):
            raise ValueError("radii must be unique")

        hidden = hidden_channels if hidden_channels is not None else max(
            16, channels // 4
        )
        if hidden <= 0:
            raise ValueError("hidden_channels must be positive")

        self.channels = int(channels)
        self.radii = tuple(int(r) for r in radii)

        # Same scoring network is used for each context distance.
        self.score = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_init))
        )

        neighbor_kernel = torch.ones(1, 1, 3, 3)
        neighbor_kernel[0, 0, 1, 1] = 0.0
        self.register_buffer(
            "neighbor_kernel", neighbor_kernel, persistent=False
        )

    def _context(self, x: Tensor, radius: int) -> Tensor:
        """Return the valid-neighbor mean at a single distance."""
        _, channels, height, width = x.shape
        kernel = self.neighbor_kernel.to(dtype=x.dtype)

        # Depthwise convolution sums the same eight offsets per channel.
        weights = kernel.expand(channels, 1, 3, 3).contiguous()
        summed = F.conv2d(
            x,
            weights,
            padding=radius,
            dilation=radius,
            groups=channels,
        )

        # Count valid spatial neighbors; broadcast over batch/channels.
        valid = x.new_ones((1, 1, height, width))
        count = F.conv2d(
            valid,
            kernel,
            padding=radius,
            dilation=radius,
        )
        # A position with no valid neighbors gets zero context.
        return summed / count.clamp_min(1.0)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {x.shape[1]}"
            )

        contexts = []
        differences = []
        scores = []

        for radius in self.radii:
            context = self._context(x, radius)
            difference = x - context
            score_input = torch.cat((x, context, difference), dim=1)
            contexts.append(context)
            differences.append(difference)
            scores.append(self.score(score_input))

        # Shape: [B, number_of_radii, H, W].
        weights = torch.softmax(torch.cat(scores, dim=1), dim=1)

        context_mean = torch.zeros_like(x)
        difference_mean = torch.zeros_like(x)
        for index, (context, difference) in enumerate(
            zip(contexts, differences)
        ):
            weight = weights[:, index : index + 1]
            context_mean = context_mean + weight * context
            difference_mean = difference_mean + weight * difference

        fused = self.fuse(torch.cat((x, context_mean, difference_mean), dim=1))
        return x + self.residual_scale * fused


def hard_background_ranking_loss(
    logits: Tensor,
    positive_mask: Tensor,
    background_mask: Tensor,
    margin: float = 0.5,
    max_positives: int = 32,
    max_negatives: int = 64,
) -> Tensor:
    """Rank same-class positives above reliable background candidates.

    Args:
        logits:
            Raw classification logits with shape [B, N, C].
        positive_mask:
            Boolean [B, N, C]. True only for a candidate assigned
            positive to that class.
        background_mask:
            Boolean [B, N, C]. True only for reliable background
            candidates eligible for that class. The caller is
            responsible for excluding ambiguous candidates.
        margin:
            Additive logit-space margin.
        max_positives:
            Maximum low-logit positives per image/class.
        max_negatives:
            Maximum high-logit backgrounds per image/class.

    Returns:
        Mean over nonempty image-class losses. If no valid pair
        exists, returns a graph-connected scalar zero.

    Note:
        A candidate positive for ANY class cannot be background for
        any class in this conservative prototype.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be [B,N,C], got {tuple(logits.shape)}")
    if positive_mask.shape != logits.shape:
        raise ValueError("positive_mask shape must match logits")
    if background_mask.shape != logits.shape:
        raise ValueError("background_mask shape must match logits")
    if positive_mask.dtype != torch.bool or background_mask.dtype != torch.bool:
        raise TypeError("candidate masks must have boolean dtype")
    if positive_mask.device != logits.device or background_mask.device != logits.device:
        raise ValueError("logits and masks must be on the same device")
    if max_positives <= 0 or max_negatives <= 0:
        raise ValueError("candidate limits must be positive")
    if margin < 0:
        raise ValueError("margin must be nonnegative")

    # Conservative cross-class safety check.
    positive_candidate = positive_mask.any(dim=-1, keepdim=True)
    if torch.any(background_mask & positive_candidate):
        raise ValueError(
            "a candidate marked positive for any class cannot be "
            "marked background"
        )

    batch_size, _, num_classes = logits.shape
    image_class_losses = []

    for batch_index in range(batch_size):
        for class_index in range(num_classes):
            class_logits = logits[batch_index, :, class_index]
            positives = class_logits[
                positive_mask[batch_index, :, class_index]
            ]
            negatives = class_logits[
                background_mask[batch_index, :, class_index]
            ]

            if positives.numel() == 0 or negatives.numel() == 0:
                continue

            num_pos = min(max_positives, positives.numel())
            num_neg = min(max_negatives, negatives.numel())
            hard_pos = torch.topk(
                positives, k=num_pos, largest=False
            ).values
            hard_neg = torch.topk(
                negatives, k=num_neg, largest=True
            ).values

            # Pair shape: [number of selected positives, negatives].
            pair_losses = F.softplus(
                float(margin)
                + hard_neg.unsqueeze(0)
                - hard_pos.unsqueeze(1)
            )
            image_class_losses.append(pair_losses.mean())

    if not image_class_losses:
        return logits.sum() * 0.0

    return torch.stack(image_class_losses).mean()
