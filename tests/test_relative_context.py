"""CPU smoke tests for the relative-context block and the ranking-loss prototype.

Run from the repository root:
    python -m unittest -v tests/test_relative_context.py

Passing these tests does not establish detection performance.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from cr_modules import RelativeContextBlock, hard_background_ranking_loss


class TestRelativeContextBlock(unittest.TestCase):
    def test_shape_finite_and_backward(self):
        torch.manual_seed(10)
        module = RelativeContextBlock(channels=8)
        x = torch.randn(2, 8, 9, 11, requires_grad=True)

        output = module(x)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all().item())

        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertIsNotNone(module.residual_scale.grad)

    def test_boundary_mean_excludes_invalid_neighbors(self):
        module = RelativeContextBlock(channels=1, radii=(1,))
        x = torch.ones(1, 1, 3, 3)
        context = module._context(x, radius=1)

        # Every position has at least one valid neighbor here.
        self.assertTrue(torch.allclose(context, torch.ones_like(context)))

    def test_no_valid_neighbor_is_zero(self):
        module = RelativeContextBlock(channels=1, radii=(1,))
        x = torch.ones(1, 1, 1, 1)
        context = module._context(x, radius=1)
        self.assertTrue(torch.equal(context, torch.zeros_like(context)))

    def test_bad_channels_raise(self):
        module = RelativeContextBlock(channels=4)
        with self.assertRaises(ValueError):
            module(torch.randn(1, 3, 5, 5))


class TestRankingLoss(unittest.TestCase):
    def test_known_pair_and_gradients(self):
        # One positive with logit 2 and one background with logit 1.
        logits = torch.tensor([[[2.0], [1.0]]], requires_grad=True)
        pos = torch.tensor([[[True], [False]]])
        neg = torch.tensor([[[False], [True]]])

        loss = hard_background_ranking_loss(
            logits, pos, neg, margin=0.5
        )
        expected = F.softplus(torch.tensor(-0.5))
        self.assertTrue(torch.allclose(loss, expected))

        loss.backward()
        self.assertLess(logits.grad[0, 0, 0].item(), 0.0)
        self.assertGreater(logits.grad[0, 1, 0].item(), 0.0)

    def test_empty_pairs_return_connected_zero(self):
        logits = torch.randn(1, 3, 2, requires_grad=True)
        pos = torch.zeros_like(logits, dtype=torch.bool)
        neg = torch.zeros_like(logits, dtype=torch.bool)

        loss = hard_background_ranking_loss(logits, pos, neg)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_cross_class_positive_cannot_be_background(self):
        logits = torch.zeros(1, 2, 2)
        pos = torch.zeros_like(logits, dtype=torch.bool)
        neg = torch.zeros_like(logits, dtype=torch.bool)
        pos[0, 0, 0] = True
        neg[0, 0, 1] = True

        with self.assertRaises(ValueError):
            hard_background_ranking_loss(logits, pos, neg)

    def test_hard_candidate_selection(self):
        # Select the lowest positive and highest background.
        logits = torch.tensor(
            [[[2.0], [0.0], [-2.0], [1.0]]],
            requires_grad=True,
        )
        pos = torch.tensor([[[True], [True], [False], [False]]])
        neg = torch.tensor([[[False], [False], [True], [True]]])

        loss = hard_background_ranking_loss(
            logits,
            pos,
            neg,
            margin=0.5,
            max_positives=1,
            max_negatives=1,
        )
        expected = F.softplus(torch.tensor(1.5))
        self.assertTrue(torch.allclose(loss, expected))


if __name__ == "__main__":
    unittest.main()
