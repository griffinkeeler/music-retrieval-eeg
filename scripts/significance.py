"""Shared configuration for fold-level significance annotations."""


SIGNIFICANCE_THRESHOLD = 0.05


def _format_threshold_label(threshold):
    """Return a compact decimal label suitable for table captions."""
    label = format(threshold, ".15g")
    return label[1:] if label.startswith("0.") else label


def _format_threshold_field_token(threshold):
    """Return a stable identifier token derived from the numeric threshold."""
    return (
        format(threshold, ".15g")
        .replace("-", "minus_")
        .replace("+", "")
        .replace(".", "_")
    )


SIGNIFICANCE_THRESHOLD_LABEL = _format_threshold_label(SIGNIFICANCE_THRESHOLD)
ALL_FOLDS_SIGNIFICANT_FIELD = (
    "all_folds_p_le_" + _format_threshold_field_token(SIGNIFICANCE_THRESHOLD)
)
