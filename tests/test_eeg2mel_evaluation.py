import unittest

import torch

from src.evaluation import precompute_negative_mse_matrix
from src.evaluation.test_eeg2mel_baseline import (
    evaluate_nearest_mse_retrieval,
    evaluate_retrieval_space,
    parse_args,
    reconstruction_metrics,
)


class EEG2MelEvaluationTests(unittest.TestCase):
    def test_mert_evaluation_is_cli_opt_in(self):
        self.assertFalse(parse_args([]).mert_space)
        self.assertTrue(parse_args(["--mert-space"]).mert_space)

    def test_reconstruction_metrics_are_exact_for_perfect_mels(self):
        target = torch.linspace(-1.0, 1.0, 2 * 12 * 12).reshape(2, 12, 12)

        metrics = reconstruction_metrics(target, target)

        self.assertEqual(metrics["n_examples"], 2)
        self.assertEqual(metrics["mse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["ssim_mean"], 1.0, places=5)
        self.assertAlmostEqual(metrics["matched_cosine_mean"], 1.0, places=6)

    def test_negative_mse_scores_rank_exact_targets_first(self):
        target = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ]
        )

        scores = precompute_negative_mse_matrix(target, target)

        self.assertTrue(torch.equal(scores.argmax(dim=1), torch.arange(3)))
        self.assertTrue(torch.allclose(scores.diag(), torch.zeros(3)))

    def test_default_mel_evaluation_includes_shared_song_metrics(self):
        generator = torch.Generator().manual_seed(4)
        target = torch.randn(4, 12, 12, generator=generator)
        subject_ids = [0, 0, 1, 1]
        song_ids = [21, 21, 22, 22]
        window_idxs = [0, 1, 0, 1]

        results, song_rows, section_rows = evaluate_retrieval_space(
            target,
            target,
            subject_ids=subject_ids,
            song_ids=song_ids,
            window_idxs=window_idxs,
            section_ids=[-1, -1, -1, -1],
            ks=(1,),
            gap=0,
            n_perms=2,
            n_within_song_shuffles=2,
            n_song_search_perms=2,
            song_search_permutation_batch_size=1,
            song_search_marginal_temperature=0.07,
            rng=0,
        )

        self.assertEqual(results["observed"]["across_song_top1"], 1.0)
        self.assertIn("song_search_marginal_top1", results["song_search"])
        self.assertEqual(len(song_rows), 4)
        self.assertEqual(section_rows, [])

    def test_nearest_mse_is_a_separate_default_retrieval_view(self):
        target = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ]
        )

        results = evaluate_nearest_mse_retrieval(
            target,
            target,
            song_ids=[21, 22, 23],
            window_idxs=[0, 0, 0],
            ks=(1,),
            gap=0,
        )

        self.assertEqual(results["distance_metric"], "mean_squared_error")
        self.assertEqual(results["observed"]["across_song_top1"], 1.0)


if __name__ == "__main__":
    unittest.main()
