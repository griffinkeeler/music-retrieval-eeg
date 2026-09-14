import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.evaluation import retrieval


class RetrievalOptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.eeg = torch.randn(10, 7)
        self.audio = torch.randn(10, 7)
        self.song_ids = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        self.window_idxs = [0, 0, 1, 2, 3, 0, 1, 1, 2, 3]
        self.ks = (1, 2, 3)
        self.cache = retrieval.build_retrieval_evaluation_cache(
            self.eeg,
            self.audio,
            self.song_ids,
            self.window_idxs,
            gap=0,
        )

    def test_similarity_cache_matches_vector_and_sequence_similarity(self):
        torch.testing.assert_close(
            retrieval.precompute_similarity_matrix(self.eeg, self.audio),
            retrieval.sequence_similarity_logits(self.eeg, self.audio),
        )

        sequence_eeg = torch.randn(8, 4, 3)
        sequence_audio = torch.randn(8, 4, 5)
        torch.testing.assert_close(
            retrieval.precompute_similarity_matrix(sequence_eeg, sequence_audio),
            retrieval.sequence_similarity_logits(sequence_eeg, sequence_audio),
        )

        vector_audio = torch.randn(8, 4)
        torch.testing.assert_close(
            retrieval.precompute_similarity_matrix(sequence_eeg, vector_audio),
            retrieval.sequence_similarity_logits(sequence_eeg, vector_audio),
        )

        vector_eeg = torch.randn(8, 4)
        torch.testing.assert_close(
            retrieval.precompute_similarity_matrix(vector_eeg, sequence_audio),
            retrieval.sequence_similarity_logits(vector_eeg, sequence_audio),
        )

        torch.testing.assert_close(
            retrieval.precompute_similarity_matrix(
                sequence_eeg * 1e-10,
                sequence_audio * 1e-10,
            ),
            retrieval.sequence_similarity_logits(
                sequence_eeg * 1e-10,
                sequence_audio * 1e-10,
            ),
        )

    def test_cached_pools_and_scores_match_query_loop(self):
        permutation = np.random.default_rng(3).permutation(len(self.audio))

        for regime in retrieval.RETRIEVAL_REGIMES:
            expected_pools = [
                retrieval._candidate_indices_for_query(
                    query_idx,
                    self.song_ids,
                    self.window_idxs,
                    regime=regime,
                    gap=0,
                )
                for query_idx in range(len(self.eeg))
            ]
            pool = self.cache.pools[regime]
            actual_pools = [
                pool.indices[query_idx, pool.valid_mask[query_idx]].tolist()
                for query_idx in range(len(self.eeg))
            ]
            self.assertEqual(actual_pools, expected_pools)

            observed = retrieval._score_cached_candidate_pool(
                self.cache,
                regime,
                self.ks,
            )
            permuted = retrieval._score_cached_candidate_pool(
                self.cache,
                regime,
                self.ks,
                audio_permutation=permutation,
            )
            for k in self.ks:
                self.assertAlmostEqual(
                    observed[k],
                    retrieval.retrieval_with_candidate_pool(
                        self.eeg,
                        self.audio,
                        self.song_ids,
                        self.window_idxs,
                        k=k,
                        regime=regime,
                        gap=0,
                    ),
                )
                self.assertAlmostEqual(
                    permuted[k],
                    retrieval.retrieval_with_candidate_pool(
                        self.eeg,
                        self.audio[permutation],
                        self.song_ids,
                        self.window_idxs,
                        k=k,
                        regime=regime,
                        gap=0,
                    ),
                )

    def test_combined_candidate_p_values_match_independent_query_loop(self):
        n_perm = 7
        actual = retrieval.evaluate_candidate_pool_p_values(
            self.eeg,
            self.audio,
            self.window_idxs,
            self.song_ids,
            ks=self.ks,
            n_perm=n_perm,
            cache=self.cache,
        )

        for regime in retrieval.RETRIEVAL_REGIMES:
            observed = {
                k: retrieval.retrieval_with_candidate_pool(
                    self.eeg,
                    self.audio,
                    self.song_ids,
                    self.window_idxs,
                    k=k,
                    regime=regime,
                )
                for k in self.ks
            }
            null_scores = {k: [] for k in self.ks}
            rng = np.random.default_rng(0)
            for _ in range(n_perm):
                permutation = rng.permutation(len(self.audio))
                for k in self.ks:
                    null_scores[k].append(
                        retrieval.retrieval_with_candidate_pool(
                            self.eeg,
                            self.audio[permutation],
                            self.song_ids,
                            self.window_idxs,
                            k=k,
                            regime=regime,
                        )
                    )

            for k in self.ks:
                values = np.asarray(null_scores[k])
                expected_upper = (
                    1 + np.sum(values >= observed[k])
                ) / (n_perm + 1)
                expected_lower = (
                    1 + np.sum(values <= observed[k])
                ) / (n_perm + 1)
                self.assertEqual(
                    actual[f"{regime}_top{k}_p"],
                    expected_upper,
                )
                self.assertEqual(
                    actual[f"{regime}_top{k}_p_lower"],
                    expected_lower,
                )

    def test_within_song_index_shuffles_match_embedding_shuffles(self):
        old_rng = np.random.default_rng(5)
        new_rng = np.random.default_rng(5)

        for _ in range(3):
            shuffled_audio = retrieval._shuffle_audio_within_song(
                self.audio,
                self.song_ids,
                old_rng,
            )
            expected_audio = {
                k: retrieval.retrieval_with_candidate_pool(
                    self.eeg,
                    shuffled_audio,
                    self.song_ids,
                    self.window_idxs,
                    k=k,
                    regime="within_song",
                )
                for k in self.ks
            }
            shuffled_eeg = retrieval._shuffle_eeg_within_song(
                self.eeg,
                self.song_ids,
                old_rng,
            )
            expected_eeg = {
                k: retrieval.retrieval_with_candidate_pool(
                    shuffled_eeg,
                    self.audio,
                    self.song_ids,
                    self.window_idxs,
                    k=k,
                    regime="within_song",
                )
                for k in self.ks
            }

            audio_permutation = retrieval._within_song_permutation_indices(
                self.song_ids,
                new_rng,
            )
            actual_audio = retrieval._score_cached_candidate_pool(
                self.cache,
                "within_song",
                self.ks,
                audio_permutation=audio_permutation,
            )
            eeg_permutation = retrieval._within_song_permutation_indices(
                self.song_ids,
                new_rng,
            )
            actual_eeg = retrieval._score_cached_candidate_pool(
                self.cache,
                "within_song",
                self.ks,
                eeg_permutation=eeg_permutation,
            )

            for k in self.ks:
                self.assertAlmostEqual(actual_audio[k], expected_audio[k])
                self.assertAlmostEqual(actual_eeg[k], expected_eeg[k])

    def test_shuffle_and_section_nulls_include_lower_tail_p_values(self):
        n_perm = 5
        shuffle_metrics = retrieval.evaluate_within_song_shuffle_baseline(
            self.eeg,
            self.audio,
            self.song_ids,
            self.window_idxs,
            ks=self.ks,
            n_shuffle=n_perm,
            cache=self.cache,
        )
        section_metrics = retrieval.evaluate_section_null_regime(
            self.eeg,
            self.audio,
            self.song_ids,
            self.window_idxs,
            section_ids=[0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
            ks=self.ks,
            n_perm=n_perm,
            cache=self.cache,
        )

        for k in self.ks:
            for prefix in (
                "within_song_audio_shuffle",
                "within_song_eeg_shuffle",
            ):
                self.assertIn(f"{prefix}_top{k}_p_lower", shuffle_metrics)
            self.assertIn(
                f"within_song_section_top{k}_p_lower",
                section_metrics,
            )

    def test_section_retrieval_supports_time_preserving_embeddings(self):
        sequence_eeg = torch.randn(10, 4, 3)
        sequence_audio = torch.randn(10, 4, 5)
        section_ids = [0, 0, 1, 1, -1, 0, 1, 1, 2, 2]
        cache = retrieval.build_retrieval_evaluation_cache(
            sequence_eeg,
            sequence_audio,
            self.song_ids,
            self.window_idxs,
        )
        actual = retrieval.evaluate_section_regime(
            sequence_eeg,
            sequence_audio,
            self.song_ids,
            self.window_idxs,
            section_ids,
            ks=self.ks,
            cache=cache,
        )

        for k in self.ks:
            expected = retrieval.retrieval_with_candidate_pool_section(
                sequence_eeg,
                sequence_audio,
                self.song_ids,
                self.window_idxs,
                section_ids,
                k=k,
            )
            self.assertAlmostEqual(
                actual[f"within_song_section_top{k}"],
                expected,
            )

    def test_song_search_cache_matches_existing_query_loop(self):
        cache = retrieval.build_song_search_evaluation_cache(
            self.eeg,
            self.audio,
            self.song_ids,
            self.window_idxs,
        )
        results = retrieval.evaluate_song_search(
            self.eeg,
            self.audio,
            self.song_ids,
            self.window_idxs,
            cache=cache,
        )

        for query_idx, result in enumerate(results):
            grouped_scores = retrieval.similarity_scores_grouped_by_song(
                self.eeg[query_idx],
                self.audio,
                self.song_ids,
                self.window_idxs,
            )
            expected_ranked_songs = []
            for song_id, entries in grouped_scores.items():
                best_entry = max(entries, key=lambda item: item["score"])
                expected_ranked_songs.append(
                    {
                        "song_id": song_id,
                        "best_window_idx": best_entry["window_idx"],
                        "best_score": best_entry["score"],
                    }
                )
            expected_ranked_songs.sort(
                key=lambda item: item["best_score"],
                reverse=True,
            )

            self.assertEqual(
                result["predicted_song_id"],
                expected_ranked_songs[0]["song_id"],
            )
            for actual_song, expected_song in zip(
                    result["ranked_songs"],
                    expected_ranked_songs,
            ):
                self.assertEqual(actual_song["song_id"], expected_song["song_id"])
                self.assertEqual(
                    actual_song["best_window_idx"],
                    expected_song["best_window_idx"],
                )
                self.assertAlmostEqual(
                    actual_song["best_score"],
                    expected_song["best_score"],
                    places=6,
                )

        observed, n_correct = retrieval._song_search_top1_accuracy(cache)
        self.assertAlmostEqual(
            observed,
            np.mean([result["song_correct"] for result in results]),
        )
        self.assertEqual(
            n_correct,
            sum(result["song_correct"] for result in results),
        )
        marginal_observed, marginal_n_correct = (
            retrieval._song_search_marginal_top1_accuracy(cache)
        )
        self.assertAlmostEqual(
            marginal_observed,
            np.mean([result["marginal_song_correct"] for result in results]),
        )
        self.assertEqual(
            marginal_n_correct,
            sum(result["marginal_song_correct"] for result in results),
        )
        for result in results:
            self.assertAlmostEqual(
                sum(
                    song["marginal_probability"]
                    for song in result["marginal_ranked_songs"]
                ),
                1.0,
                places=6,
            )

    def test_song_search_marginal_corrects_for_unequal_window_counts(self):
        cache = retrieval.SongSearchEvaluationCache(
            similarities=torch.tensor([[0.2, 0.2, 0.3]]),
            query_song_ids=(2,),
            query_window_idxs=(0,),
            candidate_song_ids=(1, 1, 2),
            candidate_window_idxs=(0, 1, 0),
            candidate_n_repeats=(1, 1, 1),
        )

        observed, n_correct = retrieval._song_search_marginal_top1_accuracy(
            cache,
            temperature=1.0,
        )

        self.assertEqual(observed, 1.0)
        self.assertEqual(n_correct, 1)
        unnormalized_song_1 = torch.exp(torch.tensor(0.2)) * 2
        unnormalized_song_2 = torch.exp(torch.tensor(0.3))
        self.assertGreater(unnormalized_song_1, unnormalized_song_2)

    def test_song_search_marginal_rejects_invalid_temperature(self):
        cache = retrieval.build_song_search_evaluation_cache(
            self.eeg,
            self.audio,
            self.song_ids,
            self.window_idxs,
        )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            retrieval._song_search_marginal_top1_accuracy(
                cache,
                temperature=0,
            )

    def test_song_search_permutation_matches_manual_column_shuffles(self):
        cache = retrieval.build_song_search_evaluation_cache(
            self.eeg,
            self.audio,
            self.song_ids,
            self.window_idxs,
        )
        n_perm = 9
        seed = 23
        marginal_temperature = 0.2
        actual = retrieval.evaluate_song_search_permutation_test(
            self.eeg,
            self.audio,
            self.song_ids,
            self.window_idxs,
            n_perm=n_perm,
            rng=seed,
            permutation_batch_size=3,
            cache=cache,
            marginal_temperature=marginal_temperature,
        )
        unbatched = retrieval.evaluate_song_search_permutation_test(
            self.eeg,
            self.audio,
            self.song_ids,
            self.window_idxs,
            n_perm=n_perm,
            rng=seed,
            permutation_batch_size=1,
            cache=cache,
            marginal_temperature=marginal_temperature,
        )

        candidate_songs = tuple(dict.fromkeys(cache.candidate_song_ids))
        positions_by_song = [
            [
                position
                for position, candidate_song_id in enumerate(cache.candidate_song_ids)
                if candidate_song_id == song_id
            ]
            for song_id in candidate_songs
        ]
        song_positions = {
            song_id: position for position, song_id in enumerate(candidate_songs)
        }
        true_positions = torch.as_tensor(
            [song_positions[song_id] for song_id in cache.query_song_ids]
        )
        rng = np.random.default_rng(seed)
        null_correct_counts = []
        marginal_null_correct_counts = []
        for _ in range(n_perm):
            permutation = torch.as_tensor(
                rng.permutation(cache.similarities.shape[1]),
                dtype=torch.long,
            )
            shuffled_scores = cache.similarities.index_select(1, permutation)
            best_by_song = torch.stack(
                [
                    shuffled_scores[:, positions].amax(dim=1)
                    for positions in positions_by_song
                ],
                dim=1,
            )
            null_correct_counts.append(
                int((best_by_song.argmax(dim=1) == true_positions).sum())
            )
            marginal_by_song = torch.stack(
                [
                    torch.logsumexp(
                        shuffled_scores[:, positions] / marginal_temperature,
                        dim=1,
                    )
                    - np.log(len(positions))
                    for positions in positions_by_song
                ],
                dim=1,
            )
            marginal_null_correct_counts.append(
                int((marginal_by_song.argmax(dim=1) == true_positions).sum())
            )

        null_correct_counts = np.asarray(null_correct_counts)
        null_scores = null_correct_counts / len(cache.query_song_ids)
        _, observed_correct = retrieval._song_search_top1_accuracy(cache)
        marginal_null_correct_counts = np.asarray(marginal_null_correct_counts)
        marginal_null_scores = (
            marginal_null_correct_counts / len(cache.query_song_ids)
        )
        _, marginal_observed_correct = (
            retrieval._song_search_marginal_top1_accuracy(
                cache,
                temperature=marginal_temperature,
            )
        )
        expected = {
            "song_search_song_top1_null_mean": float(null_scores.mean()),
            "song_search_song_top1_null_std": float(null_scores.std(ddof=0)),
            "song_search_song_top1_p": float(
                (1 + np.sum(null_correct_counts >= observed_correct)) / (n_perm + 1)
            ),
            "song_search_song_top1_p_lower": float(
                (1 + np.sum(null_correct_counts <= observed_correct)) / (n_perm + 1)
            ),
            "song_search_marginal_top1_null_mean": float(
                marginal_null_scores.mean()
            ),
            "song_search_marginal_top1_null_std": float(
                marginal_null_scores.std(ddof=0)
            ),
            "song_search_marginal_top1_p": float(
                (
                    1
                    + np.sum(
                        marginal_null_correct_counts >= marginal_observed_correct
                    )
                )
                / (n_perm + 1)
            ),
            "song_search_marginal_top1_p_lower": float(
                (
                    1
                    + np.sum(
                        marginal_null_correct_counts <= marginal_observed_correct
                    )
                )
                / (n_perm + 1)
            ),
        }
        for key, expected_value in expected.items():
            self.assertAlmostEqual(actual[key], expected_value)
            self.assertAlmostEqual(unbatched[key], expected_value)

        self.assertEqual(actual["song_search_song_top1_n_perm"], n_perm)
        self.assertEqual(actual["song_search_marginal_top1_n_perm"], n_perm)
        self.assertEqual(
            actual["song_search_marginal_temperature"],
            marginal_temperature,
        )
        self.assertEqual(actual["song_search_n_candidate_songs"], 2)
        self.assertEqual(actual["song_search_song_top1_chance"], 0.5)

    def test_song_search_summary_csv_includes_permutation_metrics(self):
        cache = retrieval.build_song_search_evaluation_cache(
            self.eeg,
            self.audio,
            self.song_ids,
            self.window_idxs,
        )
        results = retrieval.evaluate_song_search(
            self.eeg,
            self.audio,
            self.song_ids,
            self.window_idxs,
            cache=cache,
        )
        summary = retrieval.summarize_song_search_results(results)
        summary.update(
            retrieval.evaluate_song_search_permutation_test(
                self.eeg,
                self.audio,
                self.song_ids,
                self.window_idxs,
                n_perm=3,
                cache=cache,
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "song_search_summary.csv"
            retrieval.save_song_search_summary_csv(summary, output_path)
            with output_path.open(newline="") as csv_file:
                saved_metrics = {
                    row["metric"]: row for row in csv.DictReader(csv_file)
                }

        self.assertEqual(float(saved_metrics["analytic_chance"]["value"]), 0.5)
        self.assertIn("permutation_null_mean", saved_metrics)
        self.assertIn("permutation_p_value", saved_metrics)
        self.assertIn("permutation_p_value_lower", saved_metrics)
        self.assertIn("marginal_observed_song_accuracy", saved_metrics)
        self.assertIn("marginal_permutation_null_mean", saved_metrics)
        self.assertIn("marginal_permutation_p_value", saved_metrics)
        self.assertIn("marginal_permutation_p_value_lower", saved_metrics)
        self.assertEqual(float(saved_metrics["n_permutations"]["value"]), 3)

    def test_test_metrics_csv_includes_lower_tail_diagnostics(self):
        from scripts.test import save_test_metrics_csv

        regimes = retrieval.RETRIEVAL_REGIMES
        ks = (1, 5, 10)
        test_results = {
            f"{regime}_top{k}": 0.5
            for regime in regimes
            for k in ks
        }
        chance_results = {
            f"{regime}_top{k}_chance": 0.1
            for regime in regimes
            for k in ks
        }
        p_values = {}
        for regime in regimes:
            for k in ks:
                p_values[f"{regime}_top{k}_p"] = 0.02
                p_values[f"{regime}_top{k}_p_lower"] = 0.99
        p_values["within_song_audio_shuffle_top1_p"] = 0.05
        p_values["within_song_audio_shuffle_top1_p_lower"] = 0.96

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_metrics.csv"
            save_test_metrics_csv(
                output_path=output_path,
                n_test_windows=10,
                test_results=test_results,
                candidate_chance_results=chance_results,
                song_search_summary={
                    "song_search_song_top1": 0.5,
                    "song_search_marginal_top1": 0.6,
                    "song_search_mean_localization_error": 1.0,
                    "song_search_song_top1_p": 0.03,
                    "song_search_song_top1_p_lower": 0.98,
                    "song_search_marginal_top1_p": 0.02,
                    "song_search_marginal_top1_p_lower": 0.99,
                },
                section_coverage={"section_eval_n_scored": 8},
                section_regime_results={"within_song_section_top1": 0.5},
                section_chance_results={"within_song_section_top1_chance": 0.1},
                section_null_results={
                    "within_song_section_top1_null_mean": 0.2,
                    "within_song_section_top1_p": 0.04,
                    "within_song_section_top1_p_lower": 0.97,
                },
                p_values=p_values,
            )
            with output_path.open(newline="") as csv_file:
                saved_metrics = {
                    row["metric"]: row
                    for row in csv.DictReader(csv_file)
                }

        self.assertEqual(
            float(saved_metrics["across_song_r_at_1_p_lower"]["value"]),
            0.99,
        )
        self.assertEqual(
            float(saved_metrics["song_identification_top1_p_lower"]["value"]),
            0.98,
        )
        self.assertEqual(
            float(
                saved_metrics[
                    "song_identification_marginal_top1_p_lower"
                ]["value"]
            ),
            0.99,
        )
        self.assertEqual(
            float(saved_metrics["section_top1_p_lower"]["value"]),
            0.97,
        )
        self.assertEqual(
            float(
                saved_metrics[
                    "within_song_audio_shuffle_top1_p_lower"
                ]["value"]
            ),
            0.96,
        )

    def test_collect_embeddings_stores_raw_audio_and_aligns_only_for_loss(self):
        class IdentityEEGEncoder(torch.nn.Module):
            def forward(self, eeg, subject_ids):
                return eeg

        class RecordingClipLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_shapes = None

            def forward(self, eeg, audio, song_ids, window_idxs):
                self.input_shapes = (tuple(eeg.shape), tuple(audio.shape))
                return torch.zeros((), device=eeg.device)

        batch_size = 3
        feature_dim = 4
        time_steps = 5
        batch = {
            "eeg": torch.randn(batch_size, feature_dim, time_steps),
            "audio": torch.randn(batch_size, feature_dim),
            "subject_id": torch.tensor([0, 1, 2]),
            "song_id": torch.tensor([1, 1, 2]),
            "window_idx": torch.tensor([0, 1, 0]),
            "section_id": torch.tensor([0, 0, 1]),
        }
        clip = RecordingClipLoss()

        collected = retrieval.collect_embeddings(
            [batch],
            IdentityEEGEncoder(),
            torch.nn.Identity(),
            clip,
            torch.device("cpu"),
        )

        self.assertEqual(
            clip.input_shapes,
            (
                (batch_size, feature_dim, time_steps),
                (batch_size, feature_dim, time_steps),
            ),
        )
        self.assertEqual(
            tuple(collected["eeg_embeds"].shape),
            (batch_size, feature_dim, time_steps),
        )
        self.assertEqual(
            tuple(collected["audio_embeds"].shape),
            (batch_size, feature_dim),
        )
        torch.testing.assert_close(collected["audio_embeds"], batch["audio"])
        torch.testing.assert_close(
            retrieval.precompute_similarity_matrix(
                collected["eeg_embeds"],
                collected["audio_embeds"],
            ),
            retrieval.sequence_similarity_logits(
                collected["eeg_embeds"],
                collected["audio_embeds"],
            ),
        )


if __name__ == "__main__":
    unittest.main()
