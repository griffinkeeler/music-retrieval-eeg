"""Combine four complete paper runs into the final Table I layout."""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from scripts.format_average_metrics_table import (
    TableRow,
    _format_mean_sd,
    _format_result,
    _latex_escape,
    load_table,
)


METHODS = (
    ("subject_layer_on", "Subject Layer On", "table1-infonce-subject-on"),
    ("subject_layer_off", "Subject Layer Off", "table1-infonce-subject-off"),
    ("ridge_regression", "Ridge Regression", "table1-ridge-regression"),
    ("eeg2mel", "EEG2Mel", "table1-eeg2mel"),
)
PANELS = (
    (
        "marginal_song_identification",
        "(a) Marginalized Song Identification",
    ),
    ("within_song_retrieval", "(b) Within-Song Retrieval"),
)


@dataclass(frozen=True)
class MethodResults:
    """The five Table I rows for one model condition."""

    key: str
    label: str
    run_prefix: str
    rows: tuple[TableRow, ...]


def _resolve_run_dir(value, final_results_dir):
    path = Path(value)
    if path.is_absolute() or path.is_dir():
        return path
    return Path(final_results_dir) / path


def load_table1(run_dirs):
    """Load and validate the four method runs required by Table I."""
    expected_keys = {key for key, _, _ in METHODS}
    supplied_keys = set(run_dirs)
    if supplied_keys != expected_keys:
        missing = sorted(expected_keys - supplied_keys)
        unexpected = sorted(supplied_keys - expected_keys)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError("Invalid Table I method set: " + "; ".join(details))

    methods = []
    reference_labels = None
    for key, label, _ in METHODS:
        prefix, rows = load_table(run_dirs[key])
        row_labels = tuple(row.label for row in rows)
        if reference_labels is None:
            reference_labels = row_labels
        elif row_labels != reference_labels:
            raise ValueError(
                f"Split rows for {label} do not match the other Table I methods."
            )
        methods.append(
            MethodResults(
                key=key,
                label=label,
                run_prefix=prefix,
                rows=tuple(rows),
            )
        )
    return tuple(methods)


def _panel_result(row, panel_key):
    return getattr(row, panel_key)


def _best_kappa_indexes(methods, row_index, panel_key):
    values = [
        _panel_result(method.rows[row_index], panel_key).kappa
        for method in methods
    ]
    best = max(values)
    return {
        index
        for index, value in enumerate(values)
        if math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12)
    }


def _format_kappa(result, *, bold=False, latex=False):
    value = _format_mean_sd(result.kappa, result.kappa_sd, latex=latex)
    if latex:
        return rf"$\mathbf{{{value}}}$" if bold else f"${value}$"
    return f"**{value}**" if bold else value


def format_markdown_table(methods):
    """Return a two-panel Markdown version of Table I."""
    lines = [
        "**Table I. Mean R@1 and chance-normalized κ "
        "(mean ± sample SD across five folds).**"
    ]
    header = "| Data Split | " + " | ".join(
        cell
        for method in methods
        for cell in (f"{method.label} R@1", f"{method.label} κ")
    ) + " |"
    divider = "| --- | " + " | ".join(
        "---:" for _ in range(2 * len(methods))
    ) + " |"

    for panel_key, panel_label in PANELS:
        lines.extend(["", f"*{panel_label}*", "", header, divider])
        for row_index, reference_row in enumerate(methods[0].rows):
            best_indexes = _best_kappa_indexes(methods, row_index, panel_key)
            cells = []
            for method_index, method in enumerate(methods):
                result = _panel_result(method.rows[row_index], panel_key)
                cells.extend(
                    [
                        _format_result(result),
                        _format_kappa(
                            result,
                            bold=method_index in best_indexes,
                        ),
                    ]
                )
            lines.append(
                f"| {reference_row.label} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines)


def format_text_table(methods):
    """Return a compact plain-text Table I representation."""
    markdown = format_markdown_table(methods)
    return markdown.replace("**", "").replace("*", "")


def format_latex_table(methods):
    """Return a paper-ready two-panel LaTeX Table I."""
    column_count = 1 + 2 * len(methods)
    lines = [
        r"\begin{table*}[t]",
        (
            r"\caption{Mean marginalized song identification and top-1 "
            r"within-song retrieval performance ($R@1$) with chance-normalized "
            r"$\kappa$ (mean $\pm$ sample SD across five folds). Bold indicates "
            r"the highest $\kappa$ across methods for each data split.}"
        ),
        r"\centering",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l*{8}{c}}",
        r"\toprule",
        "& "
        + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{_latex_escape(method.label)}}}"
            for method in methods
        )
        + r" \\",
        " ".join(
            rf"\cmidrule(lr){{{start}-{start + 1}}}"
            for start in range(2, column_count, 2)
        ),
        "Data Split & "
        + " & ".join(r"$R@1$ & $\kappa$" for _ in methods)
        + r" \\",
        r"\midrule",
    ]

    for panel_index, (panel_key, panel_label) in enumerate(PANELS):
        lines.append(
            rf"\multicolumn{{{column_count}}}{{l}}{{\textit{{{_latex_escape(panel_label)}}}}} \\"
        )
        for row_index, reference_row in enumerate(methods[0].rows):
            best_indexes = _best_kappa_indexes(methods, row_index, panel_key)
            cells = []
            for method_index, method in enumerate(methods):
                result = _panel_result(method.rows[row_index], panel_key)
                cells.extend(
                    [
                        f"${_format_result(result, latex=True)}$",
                        _format_kappa(
                            result,
                            bold=method_index in best_indexes,
                            latex=True,
                        ),
                    ]
                )
            lines.append(
                f"{_latex_escape(reference_row.label)} & "
                + " & ".join(cells)
                + r" \\",
            )
        if panel_index < len(PANELS) - 1:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


FORMATTERS = {
    "text": format_text_table,
    "markdown": format_markdown_table,
    "latex": format_latex_table,
}


def format_table1(run_dirs, output_format="latex"):
    """Load the supplied runs and return a combined Table I."""
    try:
        formatter = FORMATTERS[output_format]
    except KeyError as exc:
        raise ValueError(
            f"Unknown output format {output_format!r}; choose from "
            f"{', '.join(FORMATTERS)}."
        ) from exc
    return formatter(load_table1(run_dirs))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Combine four complete five-family runs into Table I."
    )
    for key, label, default_run in METHODS:
        parser.add_argument(
            "--" + key.replace("_", "-"),
            default=default_run,
            help=(
                f"{label} run name under runs/final_results or a directory path "
                f"(default: {default_run})."
            ),
        )
    parser.add_argument(
        "--format",
        choices=tuple(FORMATTERS),
        default="latex",
        help="Output format (default: latex).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file; stdout is used when omitted.",
    )
    parser.add_argument(
        "--final-results-dir",
        type=Path,
        default=None,
        help="Run root (default: <repo>/runs/final_results).",
    )
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    final_results_dir = (
        args.final_results_dir.resolve()
        if args.final_results_dir
        else repo_root / "runs" / "final_results"
    )
    run_dirs = {
        key: _resolve_run_dir(getattr(args, key), final_results_dir)
        for key, _, _ in METHODS
    }

    try:
        table = format_table1(run_dirs, args.format)
        if args.output is None:
            print(table)
        else:
            output_path = args.output.resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(table + "\n")
            print(output_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
