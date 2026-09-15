"""Summarize test metrics across the five folds of all five split families."""

import argparse
import csv
import math
import re
from pathlib import Path

if __package__:
    from scripts.significance import (
        ALL_FOLDS_SIGNIFICANT_FIELD,
        SIGNIFICANCE_THRESHOLD,
    )
else:
    from significance import (
        ALL_FOLDS_SIGNIFICANT_FIELD,
        SIGNIFICANCE_THRESHOLD,
    )


SPLIT_LABELS = (
    "chunkout",
    "randomsegmentout",
    "songout",
    "subjectout",
    "subjectsongout",
)
SPLIT_DIRECTORY_LABELS = {
    "chunkout": "chunk_out",
    "randomsegmentout": "random_segment_out",
    "songout": "song_out",
    "subjectout": "subject_out",
    "subjectsongout": "subject_song_out",
}
SPLIT_DIRECTORY_ALIASES = {
    output_label: (output_label, directory_label)
    for output_label, directory_label in SPLIT_DIRECTORY_LABELS.items()
}
DIRECTORY_TO_OUTPUT_LABEL = {
    directory_label: output_label
    for output_label, directory_labels in SPLIT_DIRECTORY_ALIASES.items()
    for directory_label in directory_labels
}
FOLDS = range(5)
METRIC_FIELDS = ("metric", "value", "chance", "gap", "unit")
NUMERIC_FIELDS = ("value", "chance", "gap")
AGGREGATE_FIELDS = (
    "metric",
    "value",
    "value_sd",
    "chance",
    "chance_sd",
    "kappa",
    "kappa_sd",
    "gap",
    "gap_sd",
    ALL_FOLDS_SIGNIFICANT_FIELD,
    "n_folds",
    "unit",
)
RIDGE_METRIC_FIELDS = (
    "space",
    "metric",
    "value",
    "chance",
    "gap",
    "chance_ratio",
    "null_mean",
    "p_upper",
    "p_lower",
    "unit",
)
RIDGE_NUMERIC_FIELDS = (
    "value",
    "chance",
    "gap",
    "chance_ratio",
    "null_mean",
    "p_upper",
    "p_lower",
)
RIDGE_AGGREGATE_FIELDS = (
    "space",
    "metric",
    "value",
    "value_sd",
    "chance",
    "chance_sd",
    "kappa",
    "kappa_sd",
    "gap",
    "gap_sd",
    "chance_ratio",
    "chance_ratio_sd",
    "null_mean",
    "null_mean_sd",
    "p_upper",
    "p_upper_sd",
    "p_lower",
    "p_lower_sd",
    ALL_FOLDS_SIGNIFICANT_FIELD,
    "n_folds",
    "unit",
)
METRIC_SCHEMAS = {
    METRIC_FIELDS: {
        "identity_fields": ("metric",),
        "numeric_fields": NUMERIC_FIELDS,
        "aggregate_fields": AGGREGATE_FIELDS,
    },
    RIDGE_METRIC_FIELDS: {
        "identity_fields": ("space", "metric"),
        "numeric_fields": RIDGE_NUMERIC_FIELDS,
        "aggregate_fields": RIDGE_AGGREGATE_FIELDS,
    },
}
METRIC_RELATIVE_PATHS = (
    Path("test_metrics.csv"),
    Path("benchmarks/ridge_baseline_test_metrics.csv"),
)
FOLD_RUN_PATTERN = re.compile(
    rf"^(?P<prefix>.+)-(?P<split>"
    rf"{'|'.join(re.escape(label) for label in DIRECTORY_TO_OUTPUT_LABEL)})-fold"
    r"(?P<fold>\d+)$"
)


def _discover_fold_runs(run_dir):
    """Find the single complete 25-run set represented by ``run_dir``."""
    run_dir = Path(run_dir)
    matches_by_prefix = {}

    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        match = FOLD_RUN_PATTERN.fullmatch(child.name)
        if match is None:
            continue
        split_label = DIRECTORY_TO_OUTPUT_LABEL[match.group("split")]
        key = (split_label, int(match.group("fold")))
        prefix_matches = matches_by_prefix.setdefault(match.group("prefix"), {})
        if key in prefix_matches:
            raise ValueError(
                f"Run directory contains duplicate aliases for {split_label} "
                f"fold {key[1]} under prefix {match.group('prefix')!r}."
            )
        prefix_matches[key] = child

    expected = {(split_label, fold) for split_label in SPLIT_LABELS for fold in FOLDS}
    complete_prefixes = [
        prefix
        for prefix, matches in matches_by_prefix.items()
        if set(matches) == expected
    ]

    if len(complete_prefixes) == 1:
        prefix = complete_prefixes[0]
        return prefix, matches_by_prefix[prefix]
    if len(complete_prefixes) > 1:
        prefixes = ", ".join(sorted(complete_prefixes))
        raise ValueError(
            f"Run directory contains more than one complete 25-fold run: {prefixes}"
        )

    if not matches_by_prefix:
        raise FileNotFoundError(
            f"No fold run directories were found in {run_dir}. Expected names like "
            "<prefix>-chunk_out-fold0."
        )

    best_prefix, found = max(
        matches_by_prefix.items(),
        key=lambda item: len(set(item[1]) & expected),
    )
    missing = sorted(expected - set(found))
    unexpected = sorted(set(found) - expected)
    details = []
    if missing:
        details.append(
            "missing "
            + ", ".join(
                f"{best_prefix}-{SPLIT_DIRECTORY_LABELS[split_label]}-fold{fold}"
                for split_label, fold in missing
            )
        )
    if unexpected:
        details.append(
            "unexpected "
            + ", ".join(
                f"{best_prefix}-{SPLIT_DIRECTORY_LABELS[split_label]}-fold{fold}"
                for split_label, fold in unexpected
            )
        )
    raise FileNotFoundError(
        f"Run directory does not contain the required 25 folds for "
        f"{best_prefix}: {'; '.join(details)}"
    )


def _find_metric_path(fold_dir):
    """Find the supported metrics CSV stored in one fold directory."""
    candidates = [fold_dir / relative_path for relative_path in METRIC_RELATIVE_PATHS]
    present = [path for path in candidates if path.is_file()]
    if len(present) == 1:
        return present[0]
    if len(present) > 1:
        raise ValueError(
            f"Fold directory contains more than one supported metrics CSV: {fold_dir}"
        )
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing test metrics file; expected one of: {expected}")


def _read_metric_rows(path):
    """Read one supported metrics CSV and reject incomplete values."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing test metrics file: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames not in METRIC_SCHEMAS:
            expected = " or ".join(str(list(fields)) for fields in METRIC_SCHEMAS)
            raise ValueError(
                f"Unexpected columns in {path}: {reader.fieldnames}. "
                f"Expected {expected}."
            )
        rows = list(reader)
    schema = METRIC_SCHEMAS[fieldnames]

    if not rows:
        raise ValueError(f"Test metrics file is empty: {path}")

    seen_identities = set()
    for row in rows:
        if None in row or any(row[field] is None for field in fieldnames):
            raise ValueError(f"Malformed row in {path}: {row}")
        for field in schema["identity_fields"]:
            row[field] = row[field].strip()
            if not row[field]:
                raise ValueError(f"Found an empty {field} in {path}")
        identity = tuple(row[field] for field in schema["identity_fields"])
        if identity in seen_identities:
            raise ValueError(f"Duplicate metric identity {identity!r} in {path}")
        seen_identities.add(identity)
        row["unit"] = row["unit"].strip()

        for field in schema["numeric_fields"]:
            text = row[field].strip()
            row[field] = text
            if field == "value" and not text:
                raise ValueError(f"Metric {identity!r} has no value in {path}")
            if not text:
                continue
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(
                    f"Metric {identity!r} has a non-numeric {field} in "
                    f"{path}: {text!r}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"Metric {identity!r} has a non-finite {field} in "
                    f"{path}: {text!r}"
                )

    return rows, schema


def _format_number(value):
    """Format a summary statistic without adding avoidable CSV precision noise."""
    return format(value, ".17g")


def _mean(values):
    return math.fsum(values) / len(values)


def _sample_sd(values):
    """Return the sample SD, using folds as the sample (Bessel correction)."""
    if len(values) < 2:
        raise ValueError("At least two folds are required to calculate sample SD.")
    mean = _mean(values)
    squared_deviations = [(value - mean) ** 2 for value in values]
    return math.sqrt(math.fsum(squared_deviations) / (len(values) - 1))


def _uses_kappa(metric):
    """Return whether a metric is a top-1 score with a chance baseline."""
    return metric.endswith(("_r_at_1", "_top1"))


def _fold_kappa(metric, value, chance):
    """Calculate chance-corrected agreement for one fold."""
    if chance == 1.0:
        raise ValueError(
            f"Metric {metric!r} has chance=1, so its fold-level kappa is undefined."
        )
    return (value - chance) / (1.0 - chance)


def _average_fold_rows(metric_paths):
    """Calculate fold means and sample SDs for aligned metric tables."""
    loaded_folds = [_read_metric_rows(path) for path in metric_paths]
    fold_rows = [rows for rows, _ in loaded_folds]
    schemas = [schema for _, schema in loaded_folds]
    schema = schemas[0]
    if any(candidate is not schema for candidate in schemas[1:]):
        raise ValueError("Metric CSV schemas do not match across folds.")

    reference_rows = fold_rows[0]
    identity_fields = schema["identity_fields"]
    reference_identities = [
        tuple(row[field] for field in identity_fields) for row in reference_rows
    ]

    for path, rows in zip(metric_paths[1:], fold_rows[1:]):
        identities = [tuple(row[field] for field in identity_fields) for row in rows]
        if identities != reference_identities:
            raise ValueError(
                f"Metric identities or ordering in {path} do not match "
                f"{metric_paths[0]}."
            )

    averaged_rows = []
    for row_index, reference in enumerate(reference_rows):
        metric = reference["metric"]
        identity = tuple(reference[field] for field in identity_fields)
        units = {rows[row_index]["unit"] for rows in fold_rows}
        if len(units) != 1:
            raise ValueError(
                f"Metric {identity!r} uses inconsistent units across folds: "
                f"{sorted(units)}"
            )

        averaged = {
            **{field: reference[field] for field in identity_fields},
            ALL_FOLDS_SIGNIFICANT_FIELD: "",
            "n_folds": str(len(fold_rows)),
            "unit": reference["unit"],
        }
        for field in schema["numeric_fields"]:
            texts = [rows[row_index][field] for rows in fold_rows]
            present = [bool(text) for text in texts]
            if any(present) and not all(present):
                raise ValueError(
                    f"Metric {identity!r} has inconsistent blank {field} values "
                    "across folds."
                )
            if all(present):
                values = [float(text) for text in texts]
                averaged[field] = _format_number(_mean(values))
                averaged[f"{field}_sd"] = _format_number(_sample_sd(values))
                if field == "value" and metric.endswith("_p_upper"):
                    averaged[ALL_FOLDS_SIGNIFICANT_FIELD] = (
                        "true"
                        if all(
                            value <= SIGNIFICANCE_THRESHOLD for value in values
                        )
                        else "false"
                    )
            else:
                averaged[field] = ""
                averaged[f"{field}_sd"] = ""

        # Ridge metrics store their permutation p-value beside the observed
        # score instead of as a separate ``*_p_upper`` metric row.
        if "p_upper" in schema["numeric_fields"]:
            p_upper_values = [rows[row_index]["p_upper"] for rows in fold_rows]
            if all(p_upper_values):
                averaged[ALL_FOLDS_SIGNIFICANT_FIELD] = (
                    "true"
                    if all(
                        float(value) <= SIGNIFICANCE_THRESHOLD
                        for value in p_upper_values
                    )
                    else "false"
                )

        if _uses_kappa(metric) and all(
            rows[row_index][field]
            for rows in fold_rows
            for field in ("value", "chance")
        ):
            kappas = [
                _fold_kappa(
                    metric,
                    float(rows[row_index]["value"]),
                    float(rows[row_index]["chance"]),
                )
                for rows in fold_rows
            ]
            averaged["kappa"] = _format_number(_mean(kappas))
            averaged["kappa_sd"] = _format_number(_sample_sd(kappas))
        else:
            averaged["kappa"] = ""
            averaged["kappa_sd"] = ""
        averaged_rows.append(averaged)

    return averaged_rows, schema["aggregate_fields"]


def _write_metric_rows(path, rows, fieldnames):
    path = Path(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def average_test_results(run_dir, output_dir=None, output_prefix=None):
    """Validate a 5-family x 5-fold run and write one summary CSV per family."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_prefix, fold_dirs = _discover_fold_runs(run_dir)
    if output_prefix is None:
        output_prefix = (
            run_prefix[: -len("-25split")]
            if run_prefix.endswith("-25split")
            else run_prefix
        )
    else:
        output_prefix = output_prefix.strip()
        if not output_prefix or Path(output_prefix).name != output_prefix:
            raise ValueError("Output prefix must be a non-blank filename prefix.")

    # Build every aggregate before writing any file, so an invalid fold cannot
    # leave behind a mixture of fresh and stale summary CSVs.
    aggregates = []
    for split_label in SPLIT_LABELS:
        metric_paths = [
            _find_metric_path(fold_dirs[(split_label, fold)]) for fold in FOLDS
        ]
        rows, fieldnames = _average_fold_rows(metric_paths)
        filename = f"{output_prefix}_{split_label}_5fold_average_metrics.csv"
        aggregates.append((filename, rows, fieldnames))

    output_dir = run_dir if output_dir is None else Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for filename, rows, fieldnames in aggregates:
        output_path = output_dir / filename
        _write_metric_rows(output_path, rows, fieldnames)
        output_paths.append(output_path)

    return output_paths


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Summarize test_metrics.csv with mean +/- sample SD across folds "
            "0-4 for chunk-out, random-segment-out, song-out, subject-out, "
            "and subject-song-out runs."
        )
    )
    parser.add_argument(
        "run_name",
        help=(
            "Run name under runs/ (or a run directory path) containing the "
            "25 fold run directories."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Runs root used to resolve a run name. Defaults to <repo>/runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to the supplied run directory.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Optional filename prefix for the five summary CSVs. Defaults to "
            "the discovered run prefix."
        ),
    )
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = args.runs_dir.resolve() if args.runs_dir else repo_root / "runs"

    supplied_path = Path(args.run_name)
    if supplied_path.is_absolute() or supplied_path.is_dir():
        run_dir = supplied_path
    else:
        run_dir = runs_dir / supplied_path

    try:
        output_paths = average_test_results(
            run_dir,
            args.output_dir,
            args.output_prefix,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    for output_path in output_paths:
        print(output_path)
    return output_paths


if __name__ == "__main__":
    main()
