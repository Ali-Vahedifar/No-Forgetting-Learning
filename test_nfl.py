"""
Unit Tests for NFL, NFL+, and NFL+LoRA

Verifies correctness of all components including the new LoRA variant.

Anonymous submission - NeurIPS 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import unittest

from models.nfl import (
    KnowledgeDistillationLoss,
    UnderCompleteAutoEncoder,
    MultiHeadClassifier,
    NFLModel,
    NFLPlusModel,
    NFLPlusLoRAModel,
)
from models.backbone import resnet18, get_backbone
from utils.metrics import (
    compute_average_accuracy,
    compute_backward_transfer,
    compute_plasticity,
    compute_stability,
    compute_plasticity_stability,
    AccuracyMatrix,
    MetricsTracker,
)


class TestKnowledgeDistillationLoss(unittest.TestCase):

    def test_loss_computation(self):
        kd = KnowledgeDistillationLoss(temperature=2.0)
        soft = torch.randn(4, 10)
        out = torch.randn(4, 10)
        loss = kd(soft, out)
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0)

    def test_identical_inputs_minimal_gradient(self):
        """Identical inputs should produce zero gradient (at optimum)."""
        kd = KnowledgeDistillationLoss(temperature=2.0)
        t = torch.randn(4, 10, requires_grad=True)
        loss = kd(t.detach(), t)
        loss.backward()
        # Gradient should be near zero since we're at the minimum
        self.assertLess(t.grad.abs().max().item(), 1e-5)


class TestAutoEncoder(unittest.TestCase):

    def test_forward_shape(self):
        ae = UnderCompleteAutoEncoder(512, 128, num_old_classes=10)
        x = torch.randn(4, 512)
        self.assertEqual(ae(x).shape, (4, 512))

    def test_encode_decode_shapes(self):
        ae = UnderCompleteAutoEncoder(512, 128, num_old_classes=10)
        x = torch.randn(4, 512)
        z = ae.encode(x)
        self.assertEqual(z.shape, (4, 128))
        r = ae.decode(z)
        self.assertEqual(r.shape, (4, 512))

    def test_bias_correction_shape(self):
        ae = UnderCompleteAutoEncoder(512, 128, num_old_classes=10)
        x = torch.randn(4, 512)
        gamma = ae.compute_bias_correction(x)
        self.assertEqual(gamma.shape, (4, 10))

    def test_update_old_classes(self):
        ae = UnderCompleteAutoEncoder(512, 128, num_old_classes=10)
        ae.update_num_old_classes(20)
        x = torch.randn(4, 512)
        gamma = ae.compute_bias_correction(x)
        self.assertEqual(gamma.shape, (4, 20))


class TestMultiHeadClassifier(unittest.TestCase):

    def test_single_head(self):
        clf = MultiHeadClassifier(512, 10)
        f = torch.randn(4, 512)
        self.assertEqual(clf(f).shape, (4, 10))

    def test_add_task_cil(self):
        clf = MultiHeadClassifier(512, 10)
        clf.add_task_head(10)
        clf.add_task_head(10)
        self.assertEqual(clf.num_tasks, 3)
        self.assertEqual(clf.total_classes, 30)
        f = torch.randn(4, 512)
        self.assertEqual(clf(f).shape, (4, 30))

    def test_til_mode(self):
        clf = MultiHeadClassifier(512, 10)
        clf.add_task_head(10)
        f = torch.randn(4, 512)
        self.assertEqual(clf(f, task_id=0).shape, (4, 10))
        self.assertEqual(clf(f, task_id=1).shape, (4, 10))


class TestNFLModel(unittest.TestCase):

    def test_forward(self):
        bb = resnet18(input_size='cifar')
        m = NFLModel(bb, 512, 10)
        x = torch.randn(2, 3, 32, 32)
        self.assertEqual(m(x).shape, (2, 10))

    def test_add_task(self):
        bb = resnet18(input_size='cifar')
        m = NFLModel(bb, 512, 10)
        m.add_task(10)
        x = torch.randn(2, 3, 32, 32)
        self.assertEqual(m(x).shape, (2, 20))

    def test_features(self):
        bb = resnet18(input_size='cifar')
        m = NFLModel(bb, 512, 10)
        x = torch.randn(2, 3, 32, 32)
        self.assertEqual(m.get_features(x).shape, (2, 512))


class TestNFLPlusModel(unittest.TestCase):

    def test_forward(self):
        bb = resnet18(input_size='cifar')
        m = NFLPlusModel(bb, 512, 10)
        x = torch.randn(2, 3, 32, 32)
        self.assertEqual(m(x).shape, (2, 10))

    def test_autoencoder_integrated(self):
        bb = resnet18(input_size='cifar')
        m = NFLPlusModel(bb, 512, 10)
        x = torch.randn(2, 3, 32, 32)
        f = m.get_features(x)
        r = m.autoencoder(f)
        self.assertEqual(r.shape, f.shape)

    def test_bias_corrected_logits(self):
        bb = resnet18(input_size='cifar')
        m = NFLPlusModel(bb, 512, 10)
        x = torch.randn(2, 3, 32, 32)
        f = m.get_features(x)
        h = torch.randn(2, 10)
        corrected = m.compute_bias_corrected_logits(f, h)
        self.assertEqual(corrected.shape, (2, 10))


class TestNFLPlusLoRAModel(unittest.TestCase):
    """Test NFL+LoRA model — requires timm."""

    @unittest.skipUnless(
        __import__('importlib').util.find_spec('timm') is not None,
        "timm not installed"
    )
    def test_forward(self):
        from models.vit_lora import get_vit_lora_backbone
        bb = get_vit_lora_backbone(pretrained=False, lora_rank=4)
        m = NFLPlusLoRAModel(bb, bb.feature_dim, 10)
        x = torch.randn(2, 3, 224, 224)
        out = m(x)
        self.assertEqual(out.shape, (2, 10))

    @unittest.skipUnless(
        __import__('importlib').util.find_spec('timm') is not None,
        "timm not installed"
    )
    def test_add_task(self):
        from models.vit_lora import get_vit_lora_backbone
        bb = get_vit_lora_backbone(pretrained=False, lora_rank=4)
        m = NFLPlusLoRAModel(bb, bb.feature_dim, 10)
        m.add_task(10)
        x = torch.randn(2, 3, 224, 224)
        self.assertEqual(m(x).shape, (2, 20))

    @unittest.skipUnless(
        __import__('importlib').util.find_spec('timm') is not None,
        "timm not installed"
    )
    def test_lora_merge_and_reset(self):
        from models.vit_lora import get_vit_lora_backbone
        bb = get_vit_lora_backbone(pretrained=False, lora_rank=4)
        m = NFLPlusLoRAModel(bb, bb.feature_dim, 10)
        x = torch.randn(2, 3, 224, 224)

        # Get output before merge
        out_before = m(x).detach()

        # Merge and reset
        bb.merge_and_reset_all()

        # After merge+reset with fresh zeros, output should be same
        out_after = m(x).detach()
        self.assertTrue(
            torch.allclose(out_before, out_after, atol=1e-4),
            "Merge-and-reset should preserve model output"
        )

    @unittest.skipUnless(
        __import__('importlib').util.find_spec('timm') is not None,
        "timm not installed"
    )
    def test_fisher_penalty_zero_at_init(self):
        from models.vit_lora import get_vit_lora_backbone
        bb = get_vit_lora_backbone(pretrained=False, lora_rank=4)
        penalty = bb.compute_fisher_penalty()
        # At init, LoRA B=0, so penalty should be near 0
        self.assertLess(penalty.item(), 1e-6)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.acc_matrix = np.array([
            [0.95, 0.00, 0.00, 0.00, 0.00],
            [0.85, 0.92, 0.00, 0.00, 0.00],
            [0.78, 0.85, 0.90, 0.00, 0.00],
            [0.72, 0.80, 0.85, 0.88, 0.00],
            [0.68, 0.75, 0.82, 0.85, 0.90],
        ])

    def test_average_accuracy(self):
        acc = compute_average_accuracy(self.acc_matrix)
        expected = np.mean([0.68, 0.75, 0.82, 0.85, 0.90])
        self.assertAlmostEqual(acc, expected, places=5)

    def test_backward_transfer_negative(self):
        bwt = compute_backward_transfer(self.acc_matrix)
        self.assertLess(bwt, 0)

    def test_plasticity_stability_range(self):
        ps, p, s = compute_plasticity_stability(self.acc_matrix)
        self.assertGreaterEqual(ps, 0)
        self.assertLessEqual(ps, 1)
        self.assertGreaterEqual(p, 0)
        self.assertGreaterEqual(s, 0)


class TestBackbone(unittest.TestCase):

    def test_resnet18_cifar(self):
        m = resnet18(input_size='cifar')
        self.assertEqual(m(torch.randn(2, 3, 32, 32)).shape, (2, 512))

    def test_resnet18_tiny(self):
        m = resnet18(input_size='tiny')
        self.assertEqual(m(torch.randn(2, 3, 64, 64)).shape, (2, 512))

    def test_resnet18_imagenet(self):
        m = resnet18(input_size='imagenet')
        self.assertEqual(m(torch.randn(2, 3, 224, 224)).shape, (2, 512))

    def test_factory(self):
        bb = get_backbone('resnet18', 'cifar100')
        self.assertIsInstance(bb, nn.Module)
        self.assertEqual(bb.feature_dim, 512)


class TestMetricsTracker(unittest.TestCase):

    def test_full_tracking(self):
        t = MetricsTracker(3)
        t.update(0, [0.95])
        t.update(1, [0.85, 0.92])
        t.update(2, [0.80, 0.88, 0.90])
        final = t.get_final_metrics()
        self.assertIsNotNone(final.accuracy)
        self.assertIsNotNone(final.backward_transfer)
        self.assertIsNotNone(final.plasticity_stability)


if __name__ == '__main__':
    unittest.main(verbosity=2)
