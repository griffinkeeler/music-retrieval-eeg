import csv
import tempfile
import unittest
from pathlib import Path

from scripts.format_average_metrics_table import SPLITS
from scripts.format_table1 import METHODS, format_table1, load_table1


class FormatTable1Tests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.final_results_dir = Path(self.temp_dir.name) / "final_results"
        self.run_dirs = {}

        for method_index, (key, _, default_run) in enumerate(METHODS):
            run_dir = self.final_results_dir / default_run
            average_dir = run_dir / "average_metrics"
            average_dir.mkdir(parents=True)
            self.run_dirs[key] = run_dir
            for split_index, (split, _) in enumerate(SPLITS):
                self._write_average_csv(
                    average_dir,
                    default_run,
                    split,
                    method_index,
                    split_index,
                )

    @staticmethod
    def _write_average_csv(
        average_dir,
        prefix,
        split,
        method_index,
        split_index,
    ):
        path = average_dir / f"{prefix}_{split}_5fold_average_metrics.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "metric",
                    "value",
                    "value_sd",
                    "kappa",
                    "kappa_sd",
                    "n_folds",
                    "unit",
                ),
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "metric": "song_identification_marginal_top1",
                        "value": 0.10 + method_index * 0.01 + split_index * 0.001,
                        "value_sd": 0.003,
                        "kappa": 0.01 + method_index * 0.01,
                        "kappa_sd": 0.004,
                        "n_folds": 5,
                        "unit": "proportion",
                    },
                    {
                        "metric": "within_song_r_at_1",
                        "value": 0.05 + method_index * 0.01 + split_index * 0.001,
                        "value_sd": 0.005,
                        "kappa": 0.04 - method_index * 0.01,
                        "kappa_sd": 0.006,
                        "n_folds": 5,
                        "unit": "proportion",
                    },
                ]
            )

    def test_loads_all_methods_in_paper_order(self):
        methods = load_table1(self.run_dirs)

        self.assertEqual(
            [method.label for method in methods],
            [
                "Subject Layer On",
                "Subject Layer Off",
                "Cosine Regression",
                "EEG2Mel",
            ],
        )
        self.assertEqual(
            [row.label for row in methods[0].rows],
            [
                "Subjects",
                "Songs",
                "Subjects + Songs",
                "Chunks",
                "Random Segments",
            ],
        )

    def test_markdown_bolds_best_unrounded_kappa_in_each_panel(self):
        table = format_table1(self.run_dirs, "markdown")

        self.assertIn("*(a) Marginalized Song Identification*", table)
        self.assertIn("*(b) Within-Song Retrieval*", table)
        self.assertIn("**.040 ± .004**", table)
        self.assertIn("**.040 ± .006**", table)
        self.assertIn("| Subjects | .100 ± .003 | .010 ± .004 |", table)

    def test_latex_matches_table_i_structure_and_bolding(self):
        table = format_table1(self.run_dirs, "latex")

        self.assertIn(r"\begin{table*}[t]", table)
        self.assertIn(r"\multicolumn{2}{c}{EEG2Mel}", table)
        self.assertIn(
            r"\multicolumn{9}{l}{\textit{(a) Marginalized Song Identification}}",
            table,
        )
        self.assertIn(r"$\mathbf{.040 \pm .004}$", table)
        self.assertIn(r"$\mathbf{.040 \pm .006}$", table)

    def test_rejects_an_incomplete_method_mapping(self):
        incomplete = dict(self.run_dirs)
        incomplete.pop("eeg2mel")

        with self.assertRaisesRegex(ValueError, "missing eeg2mel"):
            load_table1(incomplete)


if __name__ == "__main__":
    unittest.main()
