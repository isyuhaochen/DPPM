import tempfile
import unittest
from pathlib import Path

import torch

from dppm.memory import DPPM, Delta, Evidence, MatchedRPMem, RPMem
from dppm.runner import load_weights, save_weights


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.q = torch.randn(6, 2, 3, 16)

    def test_evidence_is_permutation_invariant(self):
        model = Evidence(16)
        torch.nn.init.normal_(model.score.weight, std=0.05)
        torch.testing.assert_close(model(self.q), model(self.q[[4, 0, 5, 1, 2, 3]]))

    def test_delta_preserves_constant_inputs(self):
        for calibrated in [False, True]:
            model = Delta(16, 4, calibrated)
            q = self.q[:1].expand(8, -1, -1, -1)
            torch.testing.assert_close(model(q), q[:1], rtol=2e-5, atol=2e-5)

    def test_first_session_identity(self):
        for model in [RPMem(16), Evidence(16), Delta(16, 4, True), DPPM(16)]:
            torch.testing.assert_close(model(self.q[:1]), self.q[:1], rtol=2e-5, atol=2e-5)

    def test_order_sensitive_paths(self):
        for model in [RPMem(16), Delta(16), DPPM(16)]:
            self.assertGreater(float((model(self.q) - model(self.q.flip(0))).abs().max()), 1e-4)

    def test_both_paths_receive_gradients(self):
        model = DPPM(16)
        model(self.q).square().mean().backward()
        for name in [
            "evidence.score.weight",
            "revision.rate.weight",
            "revision.key.weight",
            "mix_logit",
        ]:
            grad = dict(model.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)

    def test_parameter_budget_and_matched_initialization(self):
        self.assertEqual(sum(p.numel() for p in DPPM().parameters()), 527364)
        self.assertEqual(sum(p.numel() for p in RPMem().parameters()), 524800)
        self.assertEqual(sum(p.numel() for p in MatchedRPMem().parameters()), 527364)
        torch.testing.assert_close(MatchedRPMem(16)(self.q), RPMem(16)(self.q), rtol=0, atol=0)

    def test_portable_checkpoint_rejects_wrong_backbone(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.pt"
            save_weights(path, DPPM(), "dppm", {"backbone": "qwen4b"}, 42)
            restored = load_weights(path, "dppm", {"backbone": "qwen4b"}, "cpu")
            self.assertEqual(sum(p.numel() for p in restored.parameters()), 527364)
            with self.assertRaises(ValueError):
                load_weights(path, "dppm", {"backbone": "gemma2b"}, "cpu")


if __name__ == "__main__":
    unittest.main()
