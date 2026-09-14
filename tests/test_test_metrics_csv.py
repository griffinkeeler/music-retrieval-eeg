import csv
import tempfile
import unittest
from pathlib import Path

from scripts.test import save_test_metrics_csv


class TestMetricsCsvTests(unittest.TestCase):
    def test_candidate_pool_chance_is_saved_for_every_retrieval_regime(self):
        ks = (1, 5, 10)
        regimes = (
            "across_song",
            "within_song",
            "across_song_no_same_song_negatives",
        )
        test_results = {
            f"{regime}_top{k}": 0.5
            for regime in regimes
            for k in ks
        }
        candidate_chance_results = {
            f"{regime}_top{k}_chance": 0.1
            for regime in regimes
            for k in ks
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = save_test_metrics_csv(
                output_path=Path(temp_dir) / "test_metrics.csv",
                n_test_windows=12,
                test_results=test_results,
                candidate_chance_results=candidate_chance_results,
                song_search_summary={
                    "song_search_song_top1": 0.5,
                    "song_search_marginal_top1": 0.6,
                    "song_search_marginal_top1_chance": 0.2,
                    "song_search_mean_localization_error": 1.0,
                },
                section_coverage={"section_eval_n_scored": 10},
                section_regime_results={"within_song_section_top1": 0.5},
                section_chance_results={"within_song_section_top1_chance": 0.1},
                section_null_results={"within_song_section_top1_null_mean": 0.2},
            )

            with output_path.open(newline="") as csv_file:
                rows_by_metric = {
                    row["metric"]: row
                    for row in csv.DictReader(csv_file)
                }

        for regime in regimes:
            for k in ks:
                row = rows_by_metric[f"{regime}_r_at_{k}"]
                self.assertEqual(float(row["value"]), 0.5)
                self.assertEqual(float(row["chance"]), 0.1)
                self.assertEqual(float(row["gap"]), 0.4)

        marginal_row = rows_by_metric["song_identification_marginal_top1"]
        self.assertEqual(float(marginal_row["value"]), 0.6)
        self.assertEqual(float(marginal_row["chance"]), 0.2)
        self.assertAlmostEqual(float(marginal_row["gap"]), 0.4)

    def test_section_metrics_are_optional(self):
        ks = (1, 5, 10)
        regimes = (
            "across_song",
            "within_song",
            "across_song_no_same_song_negatives",
        )
        test_results = {
            f"{regime}_top{k}": 0.5
            for regime in regimes
            for k in ks
        }
        candidate_chance_results = {
            f"{regime}_top{k}_chance": 0.1
            for regime in regimes
            for k in ks
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = save_test_metrics_csv(
                output_path=Path(temp_dir) / "test_metrics.csv",
                n_test_windows=12,
                test_results=test_results,
                candidate_chance_results=candidate_chance_results,
                song_search_summary={
                    "song_search_song_top1": 0.5,
                    "song_search_mean_localization_error": 1.0,
                },
                section_coverage={},
                section_regime_results={},
                section_chance_results={},
                section_null_results={},
            )

            with output_path.open(newline="") as csv_file:
                metric_names = {row["metric"] for row in csv.DictReader(csv_file)}

        self.assertNotIn("section_scored_windows", metric_names)
        self.assertNotIn("section_top1", metric_names)
