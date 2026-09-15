import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.average_test_results import (
    ALL_FOLDS_SIGNIFICANT_FIELD,
    RIDGE_AGGREGATE_FIELDS,
)
from scripts.format_average_metrics_table import (
    FINAL_TABLE_FILENAME,
    SPLITS,
    concatenate_results_tables,
    discover_average_metric_csvs,
    format_run_table,
    load_table,
    main,
    wrap_standalone_latex,
)
from scripts.significance import (
    SIGNIFICANCE_THRESHOLD,
    SIGNIFICANCE_THRESHOLD_LABEL,
)


class FormatAverageMetricsTableTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.run_dir = Path(self.temp_dir.name) / "example-25splits"
        self.average_dir = self.run_dir / "average_metrics"
        self.average_dir.mkdir(parents=True)

    def _write_average_csv(
        self,
        split,
        within_song_suffix="top_1",
        marginal_p_upper=0.05,
        within_song_p_upper=0.05,
        marginal_all_folds_significant=False,
        within_song_all_folds_significant=False,
    ):
        split_index = [name for name, _ in SPLITS].index(split)
        filename = f"example_model_{split}_5fold_average_metrics.csv"
        path = self.average_dir / filename
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "metric",
                    "value",
                    "value_sd",
                    "kappa",
                    "kappa_sd",
                    ALL_FOLDS_SIGNIFICANT_FIELD,
                    "n_folds",
                    "unit",
                ),
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "metric": "song_identification_marginal_top1",
                        "value": 0.011 + split_index * 0.01,
                        "value_sd": 0.003,
                        "kappa": -0.0001 if split == "subjectout" else 0.009,
                        "kappa_sd": 0.002,
                        "n_folds": 5,
                        "unit": "proportion",
                    },
                    {
                        "metric": f"within_song_{within_song_suffix}",
                        "value": 0.051 + split_index * 0.01,
                        "value_sd": 0.004,
                        "kappa": 0.041,
                        "kappa_sd": 0.005,
                        "n_folds": 5,
                        "unit": "proportion",
                    },
                    {
                        "metric": "song_identification_marginal_top1_p_upper",
                        "value": marginal_p_upper,
                        "value_sd": 0.0,
                        ALL_FOLDS_SIGNIFICANT_FIELD: str(
                            marginal_all_folds_significant
                        ).lower(),
                        "n_folds": 5,
                        "unit": "probability",
                    },
                    {
                        "metric": f"within_song_{within_song_suffix}_p_upper",
                        "value": within_song_p_upper,
                        "value_sd": 0.0,
                        ALL_FOLDS_SIGNIFICANT_FIELD: str(
                            within_song_all_folds_significant
                        ).lower(),
                        "n_folds": 5,
                        "unit": "probability",
                    },
                ]
            )
        return path

    def _write_complete_set(self, within_song_suffix="top_1", p_values=None):
        p_values = p_values or {}
        for split, _ in SPLITS:
            self._write_average_csv(
                split,
                within_song_suffix=within_song_suffix,
                **p_values.get(split, {}),
            )

    def _write_complete_ridge_set(self):
        for split_index, (split, _) in enumerate(SPLITS):
            path = (
                self.average_dir
                / f"ridge_model_{split}_5fold_average_metrics.csv"
            )
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=RIDGE_AGGREGATE_FIELDS,
                )
                writer.writeheader()
                for metric, value, kappa, p_upper, significant in (
                    (
                        "song_identification_marginal_top1",
                        0.02 + split_index * 0.01,
                        0.01,
                        0.02,
                        True,
                    ),
                    (
                        "within_song_r_at_1",
                        0.06 + split_index * 0.01,
                        0.04,
                        0.08,
                        False,
                    ),
                ):
                    writer.writerow(
                        {
                            "space": "pooled_mert",
                            "metric": metric,
                            "value": value,
                            "value_sd": 0.003,
                            "kappa": kappa,
                            "kappa_sd": 0.002,
                            "p_upper": p_upper,
                            ALL_FOLDS_SIGNIFICANT_FIELD: str(
                                significant
                            ).lower(),
                            "n_folds": 5,
                            "unit": "proportion",
                        }
                    )

    @staticmethod
    def _write_results_table(
        final_results_dir,
        run_name,
        marginal_kappa,
        within_kappa,
    ):
        run_dir = final_results_dir / run_name
        run_dir.mkdir(parents=True)
        rows = [
            (
                f"{row_label} & $.100 \\pm .001$ "
                f"& ${marginal_kappa:.3f} \\pm .002$ "
                f"& $.050 \\pm .003$ "
                f"& ${within_kappa:.3f} \\pm .004$ \\\\"
            )
            for _, row_label in SPLITS
        ]
        (run_dir / "results_table.tex").write_text(
            "\\begin{table}[t]\n"
            "\\begin{tabular}{lcccc}\n"
            + "\n".join(rows)
            + "\n\\end{tabular}\n"
            "\\end{table}\n",
            encoding="utf-8",
        )

    def test_loads_top_1_metrics_in_table_i_row_order(self):
        self._write_complete_set()

        prefix, rows = load_table(self.run_dir)

        self.assertEqual(prefix, "example_model")
        self.assertEqual(
            [row.label for row in rows],
            [
                "Subjects",
                "Songs",
                "Subjects + Songs",
                "Chunks",
                "Random Segments",
            ],
        )
        self.assertAlmostEqual(
            rows[0].marginal_song_identification.value, 0.011
        )
        self.assertAlmostEqual(rows[-1].within_song_retrieval.value, 0.091)

    def test_supports_repository_within_song_r_at_1_metric_alias(self):
        self._write_complete_set(within_song_suffix="r_at_1")

        table = format_run_table(self.run_dir)

        self.assertIn("Subjects                      .011 ± .003", table)
        self.assertIn("Random Segments               .051 ± .003", table)

    def test_supports_ridge_p_values_embedded_in_metric_rows(self):
        self._write_complete_ridge_set()

        prefix, rows = load_table(self.run_dir)

        self.assertEqual(prefix, "ridge_model")
        self.assertAlmostEqual(
            rows[0].marginal_song_identification.p_upper,
            0.02,
        )
        self.assertTrue(
            rows[0].marginal_song_identification.all_folds_significant
        )
        self.assertAlmostEqual(rows[0].within_song_retrieval.p_upper, 0.08)
        self.assertFalse(rows[0].within_song_retrieval.all_folds_significant)

    def test_formats_markdown_and_latex_like_table_i(self):
        self._write_complete_set()

        markdown = format_run_table(self.run_dir, "markdown")
        latex = format_run_table(self.run_dir, "latex")

        self.assertIn(
            "| Evaluation Regime | Marginal Song Identification Top-1",
            markdown,
        )
        self.assertIn("| Subjects | .011 ± .003 | .000 ± .002 |", markdown)
        self.assertIn(
            r"\multicolumn{2}{c}{Marginal Song Identification}", latex
        )
        self.assertIn(r"Subjects & $.011 \pm .003$ & $.000 \pm .002$", latex)
        self.assertNotIn("-0.000", latex)

    def test_marks_r_at_1_only_when_all_five_fold_p_values_meet_threshold(self):
        self._write_complete_set(
            p_values={
                "subjectout": {
                    "marginal_p_upper": SIGNIFICANCE_THRESHOLD,
                    "marginal_all_folds_significant": True,
                    "within_song_p_upper": SIGNIFICANCE_THRESHOLD,
                },
                "songout": {
                    "marginal_p_upper": SIGNIFICANCE_THRESHOLD / 5,
                    "within_song_p_upper": SIGNIFICANCE_THRESHOLD,
                    "within_song_all_folds_significant": True,
                },
            }
        )

        text = format_run_table(self.run_dir)
        markdown = format_run_table(self.run_dir, "markdown")
        latex = format_run_table(self.run_dir, "latex")

        self.assertIn("Subjects                     .011* ± .003", text)
        significance_note = (
            f"* p-upper ≤ {SIGNIFICANCE_THRESHOLD_LABEL} in all five folds"
        )
        self.assertIn(significance_note, text)
        self.assertIn("| Subjects | .011* ± .003 |", markdown)
        self.assertIn(significance_note, markdown)
        self.assertIn(
            "| Songs | .021 ± .003 | .009 ± .002 | .061* ± .004 |",
            markdown,
        )
        self.assertIn(r"Subjects & $.011^{*} \pm .003$", latex)
        self.assertIn(
            r"Songs & $.021 \pm .003$ & $.009 \pm .002$ "
            r"& $.061^{*} \pm .004$",
            latex,
        )
        self.assertIn(
            rf"$p_{{\mathrm{{upper}}}} \leq {SIGNIFICANCE_THRESHOLD_LABEL}$",
            latex,
        )

    def test_reports_an_incomplete_five_csv_set(self):
        for split, _ in SPLITS[:-1]:
            self._write_average_csv(split)

        with self.assertRaisesRegex(FileNotFoundError, "randomsegmentout"):
            discover_average_metric_csvs(self.run_dir)

    def test_named_average_directory_takes_priority_over_legacy_root_copies(self):
        self._write_complete_set()
        for source in self.average_dir.glob("*.csv"):
            duplicate = self.run_dir / source.name
            duplicate.write_bytes(source.read_bytes())

        prefix, paths = discover_average_metric_csvs(self.run_dir)

        self.assertEqual(prefix, "example_model")
        self.assertTrue(
            all(path.parent == self.average_dir.resolve() for path in paths.values())
        )

    def test_combines_run_results_as_one_table_in_the_supplied_order(self):
        final_results_dir = Path(self.temp_dir.name) / "final_results"
        self._write_results_table(final_results_dir, "second-run", 0.02, 0.05)
        self._write_results_table(final_results_dir, "first-run", 0.03, 0.04)

        output_path = concatenate_results_tables(
            ["second-run", "first-run"],
            final_results_dir,
        )

        self.assertEqual(
            output_path,
            (final_results_dir / FINAL_TABLE_FILENAME).resolve(),
        )
        output = output_path.read_text(encoding="utf-8")
        self.assertEqual(output.count(r"\begin{table*}[t]"), 1)
        self.assertNotIn(r"\begin{table}[t]", output)
        self.assertIn(r"\begin{tabular}{lcccc}", output)
        self.assertLess(
            output.index(r"\multicolumn{2}{c}{Second Run}"),
            output.index(r"\multicolumn{2}{c}{First Run}"),
        )
        self.assertIn(
            r"\multicolumn{5}{l}{\textit{(a) Within-Song Retrieval}}",
            output,
        )
        self.assertIn(
            r"\multicolumn{5}{l}{\textit{(b) Marginalized Song Identification}}",
            output,
        )
        self.assertLess(
            output.index("(a) Within-Song Retrieval"),
            output.index("(b) Marginalized Song Identification"),
        )
        self.assertIn(r"$\mathbf{0.030 \pm .002}$", output)
        self.assertIn(r"$\mathbf{0.050 \pm .004}$", output)

    def test_concatenate_runs_argument_writes_final_table(self):
        final_results_dir = Path(self.temp_dir.name) / "final_results"
        self._write_results_table(final_results_dir, "run-a", 0.01, 0.02)
        self._write_results_table(final_results_dir, "run-b", 0.03, 0.04)

        with redirect_stdout(io.StringIO()):
            output_path = main(
                [
                    "--concatenate-runs",
                    "run-a",
                    "run-b",
                    "--standalone",
                    "--final-results-dir",
                    str(final_results_dir),
                ]
            )

        self.assertEqual(output_path.name, FINAL_TABLE_FILENAME)
        output = output_path.read_text(encoding="utf-8")
        self.assertTrue(output.startswith("\\documentclass{article}\n"))
        self.assertEqual(output.count(r"\begin{table*}[t]"), 1)
        self.assertTrue(output.endswith("\\end{document}\n"))

    def test_concatenation_stops_when_a_run_table_is_missing(self):
        final_results_dir = Path(self.temp_dir.name) / "final_results"
        self._write_results_table(
            final_results_dir,
            "existing-run",
            0.01,
            0.02,
        )

        with self.assertRaisesRegex(FileNotFoundError, "missing-run"):
            concatenate_results_tables(
                ["existing-run", "missing-run"],
                final_results_dir,
            )

        self.assertFalse((final_results_dir / FINAL_TABLE_FILENAME).exists())

    def test_wraps_table_as_a_standalone_latex_document(self):
        document = wrap_standalone_latex(
            "\\begin{table}\nContents\n\\end{table}\n"
        )

        self.assertEqual(
            document,
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n\n"
            "\\begin{document}\n"
            "\\begin{table}\n"
            "Contents\n"
            "\\end{table}\n"
            "\\end{document}",
        )


if __name__ == "__main__":
    unittest.main()
