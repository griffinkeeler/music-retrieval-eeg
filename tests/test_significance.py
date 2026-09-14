import unittest

from scripts import average_test_results, format_average_metrics_table
from scripts.significance import (
    ALL_FOLDS_SIGNIFICANT_FIELD,
    SIGNIFICANCE_THRESHOLD,
    SIGNIFICANCE_THRESHOLD_LABEL,
)


class SignificanceConfigurationTests(unittest.TestCase):
    def test_aggregation_and_formatting_share_derived_threshold_values(self):
        self.assertEqual(
            average_test_results.SIGNIFICANCE_THRESHOLD,
            SIGNIFICANCE_THRESHOLD,
        )
        self.assertEqual(
            average_test_results.ALL_FOLDS_SIGNIFICANT_FIELD,
            ALL_FOLDS_SIGNIFICANT_FIELD,
        )
        self.assertEqual(
            format_average_metrics_table.ALL_FOLDS_SIGNIFICANT_FIELD,
            ALL_FOLDS_SIGNIFICANT_FIELD,
        )
        self.assertEqual(
            format_average_metrics_table.SIGNIFICANCE_THRESHOLD_LABEL,
            SIGNIFICANCE_THRESHOLD_LABEL,
        )
        expected_label = format(SIGNIFICANCE_THRESHOLD, ".15g")
        if expected_label.startswith("0."):
            expected_label = expected_label[1:]
        self.assertEqual(SIGNIFICANCE_THRESHOLD_LABEL, expected_label)
        threshold_token = (
            format(SIGNIFICANCE_THRESHOLD, ".15g")
            .replace("-", "minus_")
            .replace("+", "")
            .replace(".", "_")
        )
        self.assertEqual(
            ALL_FOLDS_SIGNIFICANT_FIELD,
            "all_folds_p_le_" + threshold_token,
        )


if __name__ == "__main__":
    unittest.main()
