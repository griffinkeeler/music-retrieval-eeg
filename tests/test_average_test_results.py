import csv
import statistics
import tempfile
import unittest
from pathlib import Path

from scripts.average_test_results import (
    AGGREGATE_FIELDS,
    ALL_FOLDS_SIGNIFICANT_FIELD,
    METRIC_FIELDS,
    SPLIT_DIRECTORY_LABELS,
    SPLIT_LABELS,
    SIGNIFICANCE_THRESHOLD,
    average_test_results,
)


class AverageTestResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.run_dir = Path(self.temp_dir.name) / "example-25splits"
        self.run_dir.mkdir()
        self.run_prefix = "example-25split"

    def _write_fold(
        self, split_label, fold, unit="proportion", p_upper=None
    ):
        directory_label = SPLIT_DIRECTORY_LABELS[split_label]
        fold_dir = self.run_dir / f"{self.run_prefix}-{directory_label}-fold{fold}"
        fold_dir.mkdir(parents=True)
        with (fold_dir / "test_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            rows = [
                {
                    "metric": "score_top1",
                    "value": 0.2 + fold * 0.1,
                    "chance": 0.1 + fold * 0.05,
                    "gap": 0.1 + fold * 0.05,
                    "unit": unit,
                },
                {
                    "metric": "error",
                    "value": fold + 10,
                    "chance": "",
                    "gap": "",
                    "unit": "windows",
                },
            ]
            if p_upper is not None:
                rows.append(
                    {
                        "metric": "score_top1_p_upper",
                        "value": p_upper,
                        "chance": "",
                        "gap": "",
                        "unit": "probability",
                    }
                )
            writer.writerows(rows)

    def _write_complete_run(self, p_values_by_split=None):
        p_values_by_split = p_values_by_split or {}
        for split_label in SPLIT_LABELS:
            for fold in range(5):
                p_values = p_values_by_split.get(split_label)
                self._write_fold(
                    split_label,
                    fold,
                    p_upper=None if p_values is None else p_values[fold],
                )

    def test_writes_five_csvs_with_fold_means_and_sample_sds(self):
        self._write_complete_run()

        output_paths = average_test_results(self.run_dir)

        self.assertEqual(len(output_paths), 5)
        expected_names = {
            f"example_{split_label}_5fold_average_metrics.csv"
            for split_label in SPLIT_LABELS
        }
        self.assertSetEqual({path.name for path in output_paths}, expected_names)

        with output_paths[0].open(newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(tuple(reader.fieldnames), AGGREGATE_FIELDS)
            rows = list(reader)

        score, error = rows
        score_values = [0.2 + fold * 0.1 for fold in range(5)]
        score_chances = [0.1 + fold * 0.05 for fold in range(5)]
        score_gaps = [
            value - chance for value, chance in zip(score_values, score_chances)
        ]
        score_kappas = [
            (value - chance) / (1.0 - chance)
            for value, chance in zip(score_values, score_chances)
        ]

        self.assertEqual(score["metric"], "score_top1")
        self.assertAlmostEqual(float(score["value"]), statistics.mean(score_values))
        self.assertAlmostEqual(float(score["value_sd"]), statistics.stdev(score_values))
        self.assertAlmostEqual(float(score["chance"]), statistics.mean(score_chances))
        self.assertAlmostEqual(
            float(score["chance_sd"]), statistics.stdev(score_chances)
        )
        self.assertAlmostEqual(float(score["gap"]), statistics.mean(score_gaps))
        self.assertAlmostEqual(float(score["gap_sd"]), statistics.stdev(score_gaps))
        self.assertAlmostEqual(float(score["kappa"]), statistics.mean(score_kappas))
        self.assertAlmostEqual(float(score["kappa_sd"]), statistics.stdev(score_kappas))
        self.assertEqual(score["n_folds"], "5")
        self.assertEqual(score["unit"], "proportion")
        self.assertEqual(score[ALL_FOLDS_SIGNIFICANT_FIELD], "")

        self.assertEqual(error["metric"], "error")
        self.assertEqual(error["value"], "12")
        self.assertAlmostEqual(
            float(error["value_sd"]), statistics.stdev(range(10, 15))
        )
        for field in ("chance", "chance_sd", "kappa", "kappa_sd", "gap", "gap_sd"):
            self.assertEqual(error[field], "")
        self.assertEqual(error["n_folds"], "5")
        self.assertEqual(error["unit"], "windows")
        self.assertEqual(error[ALL_FOLDS_SIGNIFICANT_FIELD], "")

    def test_records_whether_every_fold_p_value_meets_the_threshold(self):
        all_below_or_equal = [
            SIGNIFICANCE_THRESHOLD * fraction / 5 for fraction in range(1, 6)
        ]
        one_above = [
            SIGNIFICANCE_THRESHOLD / 5,
            SIGNIFICANCE_THRESHOLD / 5,
            SIGNIFICANCE_THRESHOLD / 5,
            SIGNIFICANCE_THRESHOLD / 5,
            SIGNIFICANCE_THRESHOLD * 1.001,
        ]
        self._write_complete_run(
            p_values_by_split={
                "songout": all_below_or_equal,
                "subjectout": one_above,
            }
        )

        output_paths = average_test_results(self.run_dir)
        outputs_by_split = {
            path.name.split("_")[-4]: path for path in output_paths
        }

        def p_row(split):
            with outputs_by_split[split].open(newline="") as handle:
                return next(
                    row
                    for row in csv.DictReader(handle)
                    if row["metric"] == "score_top1_p_upper"
                )

        significant = p_row("songout")
        not_significant = p_row("subjectout")
        self.assertEqual(significant[ALL_FOLDS_SIGNIFICANT_FIELD], "true")
        self.assertAlmostEqual(
            float(significant["value"]), statistics.mean(all_below_or_equal)
        )
        self.assertEqual(not_significant[ALL_FOLDS_SIGNIFICANT_FIELD], "false")
        self.assertLess(
            float(not_significant["value"]), SIGNIFICANCE_THRESHOLD
        )

    def test_missing_fold_stops_before_writing_outputs(self):
        self._write_complete_run()
        missing_dir = self.run_dir / f"{self.run_prefix}-subject_song_out-fold4"
        (missing_dir / "test_metrics.csv").unlink()
        missing_dir.rmdir()

        with self.assertRaisesRegex(FileNotFoundError, "subject_song_out-fold4"):
            average_test_results(self.run_dir)

        self.assertEqual(list(self.run_dir.glob("*average_metrics.csv")), [])

    def test_inconsistent_units_stop_before_writing_outputs(self):
        self._write_complete_run()
        bad_path = (
            self.run_dir / f"{self.run_prefix}-song_out-fold4" / "test_metrics.csv"
        )
        with bad_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["unit"] = "percent"
        with bad_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaisesRegex(ValueError, "inconsistent units"):
            average_test_results(self.run_dir)

        self.assertEqual(list(self.run_dir.glob("*average_metrics.csv")), [])


if __name__ == "__main__":
    unittest.main()
