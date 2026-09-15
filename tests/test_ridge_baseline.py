import unittest
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from src.evaluation.baselines import RidgeEEGToAudio
from src.evaluation.train_all_ridge_splits import ridge_metric_rows
from src.evaluation.train_ridge_baseline import (
    _ridge_settings,
    dataset_to_ridge_arrays,
    pool_audio_window,
    regression_metrics,
)


class _ToyDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class RidgeBaselineTests(unittest.TestCase):
    def test_temporal_audio_is_mean_pooled_while_eeg_shape_is_preserved(self):
        samples = [
            {
                "eeg": torch.arange(6, dtype=torch.float64).reshape(2, 3) + i,
                "audio": torch.tensor(
                    [[1.0 + i, 3.0 + i], [2.0 + i, 6.0 + i]]
                ),
            }
            for i in range(3)
        ]

        eeg, audio = dataset_to_ridge_arrays(_ToyDataset(samples))

        self.assertEqual(eeg.shape, (3, 2, 3))
        self.assertEqual(audio.shape, (3, 2))
        self.assertEqual(eeg.dtype, np.float32)
        self.assertEqual(audio.dtype, np.float32)
        np.testing.assert_allclose(audio[0], [2.0, 4.0])

    def test_ridge_fits_flattened_eeg_and_returns_audio_vectors(self):
        rng = np.random.default_rng(7)
        eeg = rng.normal(size=(30, 2, 4)).astype(np.float32)
        weights = rng.normal(size=(8, 3)).astype(np.float32)
        audio = eeg.reshape(30, -1) @ weights

        model = RidgeEEGToAudio(alpha=1e-5).fit(eeg, audio)
        predicted = model.predict(eeg)

        self.assertEqual(predicted.shape, audio.shape)
        np.testing.assert_allclose(predicted, audio, atol=2e-4)

    def test_ridge_rejects_unpooled_audio_targets(self):
        with self.assertRaisesRegex(ValueError, "Pool temporal audio"):
            RidgeEEGToAudio().fit(
                np.ones((4, 2, 3), dtype=np.float32),
                np.ones((4, 5, 2), dtype=np.float32),
            )

    def test_regression_metrics_are_exact_for_perfect_predictions(self):
        target = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        metrics = regression_metrics(target, target)

        self.assertEqual(metrics["mse"], 0.0)
        self.assertEqual(metrics["r2_variance_weighted"], 1.0)
        self.assertAlmostEqual(metrics["matched_cosine_mean"], 1.0)

    def test_pool_audio_window_rejects_unsupported_layouts(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            pool_audio_window(np.ones((2, 3, 4), dtype=np.float32))

    def test_ridge_settings_include_marginal_temperature(self):
        config = OmegaConf.create(
            {
                "testing": {
                    "n_perms": 10,
                    "n_within_song_shuffles": 11,
                    "song_search_n_perms": 12,
                    "song_search_permutation_batch_size": 4,
                    "song_search_marginal_temperature": 0.09,
                },
                "ridge_baseline": {},
            }
        )
        cli_args = SimpleNamespace(
            alpha=None,
            n_perms=None,
            n_within_song_shuffles=None,
            n_song_search_perms=None,
            song_search_permutation_batch_size=None,
            song_search_marginal_temperature=0.2,
        )

        settings = _ridge_settings(config, cli_args)

        self.assertEqual(settings["song_search_marginal_temperature"], 0.2)

    def test_ridge_csv_rows_include_marginal_song_identification(self):
        observed = {}
        chance = {}
        null = {}
        for regime in (
            "across_song",
            "within_song",
            "across_song_no_same_song_negatives",
        ):
            key = f"{regime}_top1"
            observed[key] = 0.4
            chance[f"{key}_chance"] = 0.1
            null[f"{key}_null_mean"] = 0.11
            null[f"{key}_p"] = 0.02
            null[f"{key}_p_lower"] = 0.99

        song_search = {
            "song_search_song_top1": 0.5,
            "song_search_song_top1_chance": 0.1,
            "song_search_song_top1_null_mean": 0.11,
            "song_search_song_top1_p": 0.01,
            "song_search_song_top1_p_lower": 1.0,
            "song_search_marginal_top1": 0.6,
            "song_search_marginal_top1_chance": 0.1,
            "song_search_marginal_top1_null_mean": 0.12,
            "song_search_marginal_top1_p": 0.005,
            "song_search_marginal_top1_p_lower": 1.0,
        }
        shuffle = {
            f"within_song_{source}_shuffle_top1_{suffix}": value
            for source in ("audio", "eeg")
            for suffix, value in (
                ("mean", 0.1),
                ("p", 0.5),
                ("p_lower", 0.5),
            )
        }
        payload = {
            "num_train_examples": 20,
            "num_examples": 10,
            "ks": [1],
            "regression_metrics": {
                "mse": 0.1,
                "r2_variance_weighted": 0.2,
                "matched_cosine_mean": 0.3,
            },
            "results": {
                "observed": observed,
                "chance": chance,
                "null": null,
                "lift": {},
                "song_search": song_search,
                "within_song_shuffle": shuffle,
            },
        }

        rows = ridge_metric_rows(payload)
        marginal = next(
            row
            for row in rows
            if row["metric"] == "song_identification_marginal_top1"
        )

        self.assertEqual(marginal["value"], 0.6)
        self.assertEqual(marginal["chance"], 0.1)
        self.assertAlmostEqual(marginal["gap"], 0.5)
        self.assertEqual(marginal["null_mean"], 0.12)
        self.assertEqual(marginal["p_upper"], 0.005)


if __name__ == "__main__":
    unittest.main()
