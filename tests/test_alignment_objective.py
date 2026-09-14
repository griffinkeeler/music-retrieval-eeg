import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scripts.train import save_checkpoint
from src.model import (
    CLIPLoss,
    CosineRegressionLoss,
    build_alignment_objective,
    restore_alignment_objective,
)
from src.model.clip_loss import sequence_similarity_logits


class AlignmentObjectiveTests(unittest.TestCase):
    def test_cosine_regression_matches_full_sequence_retrieval_cosine(self):
        torch.manual_seed(4)
        eeg = torch.randn(5, 7, 6)
        audio = torch.randn(5, 7, 4)

        actual = CosineRegressionLoss()(eeg, audio)
        expected = 1.0 - sequence_similarity_logits(eeg, audio).diagonal().mean()

        torch.testing.assert_close(actual, expected)

    def test_cosine_regression_uses_only_matched_pairs(self):
        torch.manual_seed(8)
        eeg = torch.randn(4, 6, 5)
        audio = torch.randn(4, 6, 5)
        objective = CosineRegressionLoss()

        batched_loss = objective(
            eeg,
            audio,
            song_ids=torch.tensor([1, 1, 2, 3]),
            window_idxs=torch.tensor([0, 1, 0, 0]),
        )
        separate_loss = torch.stack(
            [objective(eeg[index:index + 1], audio[index:index + 1]) for index in range(4)]
        ).mean()

        torch.testing.assert_close(batched_loss, separate_loss)

    def test_cosine_regression_is_zero_for_identical_targets(self):
        targets = torch.randn(3, 8, 5)
        loss = CosineRegressionLoss()(targets, targets)
        torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)

    def test_objective_factory_defaults_legacy_configs_to_infonce(self):
        legacy = build_alignment_objective({"clip": {"temperature": 0.2}})
        regression = build_alignment_objective(
            {"objective": {"type": "cosine_regression"}}
        )

        self.assertIsInstance(legacy, CLIPLoss)
        self.assertIsInstance(regression, CosineRegressionLoss)
        self.assertAlmostEqual(float(legacy.log_temperature.detach().exp()), 0.2)

    def test_restore_supports_legacy_and_objective_aware_checkpoints(self):
        original_clip = CLIPLoss(temperature=0.13)
        restored_clip = restore_alignment_objective(
            {"clip": {"temperature": 0.07}},
            {"clip_state_dict": original_clip.state_dict()},
        )
        restored_regression = restore_alignment_objective(
            {"objective": {"type": "infonce"}},
            {
                "objective_name": "cosine_regression",
                "objective_state_dict": CosineRegressionLoss().state_dict(),
            },
        )

        self.assertIsInstance(restored_clip, CLIPLoss)
        torch.testing.assert_close(
            restored_clip.log_temperature,
            original_clip.log_temperature,
        )
        self.assertIsInstance(restored_regression, CosineRegressionLoss)

    def test_checkpoint_records_objective_and_legacy_state_keys(self):
        eeg_encoder = torch.nn.Linear(2, 2)
        audio_projection = torch.nn.Linear(2, 2)
        objective = CosineRegressionLoss()
        optimizer = torch.optim.AdamW(
            list(eeg_encoder.parameters()) + list(audio_projection.parameters())
        )
        config = OmegaConf.create(
            {"objective": {"type": "cosine_regression"}}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                epoch=2,
                eeg_encoder=eeg_encoder,
                optimizer=optimizer,
                objective=objective,
                audio_projection=audio_projection,
                args=config,
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

        self.assertEqual(checkpoint["objective_name"], "cosine_regression")
        self.assertEqual(checkpoint["objective_state_dict"], {})
        self.assertEqual(checkpoint["clip_state_dict"], {})

    def test_cosine_regression_rejects_unpaired_batch_sizes(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            CosineRegressionLoss()(
                torch.randn(2, 4),
                torch.randn(3, 4),
            )


if __name__ == "__main__":
    unittest.main()
