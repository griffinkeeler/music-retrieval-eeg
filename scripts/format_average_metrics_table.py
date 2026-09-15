"""Format one run's five average-metrics CSVs as a paper-style table."""

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from scripts.significance import (
        ALL_FOLDS_SIGNIFICANT_FIELD,
        SIGNIFICANCE_THRESHOLD_LABEL,
    )
else:
    from significance import (
        ALL_FOLDS_SIGNIFICANT_FIELD,
        SIGNIFICANCE_THRESHOLD_LABEL,
    )


SPLITS = (
    ("subjectout", "Subjects"),
    ("songout", "Songs"),
    ("subjectsongout", "Subjects + Songs"),
    ("chunkout", "Chunks"),
    ("randomsegmentout", "Random Segments"),
)
METRIC_ALIASES = {
    "marginal_song_identification": ("song_identification_marginal_top1",),
    "within_song": ("within_song_top_1", "within_song_r_at_1"),
}
STATISTIC_FIELDS = ("value", "value_sd", "kappa", "kappa_sd")
AVERAGE_DIRECTORY_NAMES = ("average-metrics", "average_metrics")
RESULTS_TABLE_FILENAME = "results_table.tex"
FINAL_TABLE_FILENAME = "final_table.tex"
AVERAGE_FILE_PATTERN = re.compile(
    r"^(?P<prefix>.+)_(?P<split>"
    + "|".join(split for split, _ in SPLITS)
    + r")_5fold_average_metrics\.csv$"
)


@dataclass(frozen=True)
class RetrievalResult:
    """R@1 and kappa summaries, plus fold-level significance metadata."""

    value: float
    value_sd: float
    kappa: float
    kappa_sd: float
    p_upper: float | None = None
    all_folds_significant: bool = False


@dataclass(frozen=True)
class TableRow:
    """One evaluation regime in the formatted results table."""

    label: str
    marginal_song_identification: RetrievalResult
    within_song_retrieval: RetrievalResult


@dataclass(frozen=True)
class LatexRunTable:
    """One parsed per-run LaTeX table ready for combined-table formatting."""

    label: str
    path: Path
    rows: tuple[tuple[str, str, str, str], ...]


def _candidate_directories(run_dir):
    """Return average-metrics locations in priority order."""
    run_dir = Path(run_dir)
    if run_dir.name in AVERAGE_DIRECTORY_NAMES:
        return (run_dir,)

    named_directories = tuple(
        run_dir / name
        for name in AVERAGE_DIRECTORY_NAMES
        if (run_dir / name).is_dir()
    )
    if named_directories:
        return named_directories

    # Older result folders sometimes stored their aggregate CSVs at the run root.
    return (run_dir,)


def discover_average_metric_csvs(run_dir):
    """Find exactly one complete five-CSV aggregate set for ``run_dir``."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    complete_sets = []
    incomplete_sets = []
    expected_splits = {split for split, _ in SPLITS}

    for directory in _candidate_directories(run_dir):
        files_by_prefix = {}
        for path in directory.glob("*_5fold_average_metrics.csv"):
            match = AVERAGE_FILE_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            prefix = match.group("prefix")
            split = match.group("split")
            prefix_files = files_by_prefix.setdefault(prefix, {})
            if split in prefix_files:
                raise ValueError(
                    f"Duplicate average-metrics CSV for {split!r} in {directory}."
                )
            prefix_files[split] = path

        for prefix, paths in files_by_prefix.items():
            if set(paths) == expected_splits:
                complete_sets.append((prefix, paths))
            else:
                incomplete_sets.append((directory, prefix, set(paths)))

    if len(complete_sets) == 1:
        return complete_sets[0]
    if len(complete_sets) > 1:
        prefixes = ", ".join(sorted(prefix for prefix, _ in complete_sets))
        raise ValueError(
            "Run directory contains more than one complete average-metrics set: "
            f"{prefixes}. Pass one average-metrics directory instead."
        )

    if incomplete_sets:
        directory, prefix, found = max(
            incomplete_sets, key=lambda item: len(item[2])
        )
        missing = ", ".join(sorted(expected_splits - found))
        raise FileNotFoundError(
            f"Average-metrics set {prefix!r} in {directory} is incomplete; "
            f"missing: {missing}."
        )

    raise FileNotFoundError(
        f"No files matching '*_5fold_average_metrics.csv' were found in {run_dir}."
    )


def _parse_finite_number(row, field, metric, path):
    text = (row.get(field) or "").strip()
    if not text:
        raise ValueError(f"Metric {metric!r} has no {field!r} value in {path}.")
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Metric {metric!r} has a non-numeric {field!r} value in "
            f"{path}: {text!r}."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Metric {metric!r} has a non-finite {field!r} value in {path}."
        )
    return value


def _parse_all_folds_significant(row, metric, path):
    """Read the explicit all-five-fold threshold result from an aggregate row."""
    text = (row.get(ALL_FOLDS_SIGNIFICANT_FIELD) or "").strip().lower()
    if not text:
        return False
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(
        f"Metric {metric!r} has an invalid {ALL_FOLDS_SIGNIFICANT_FIELD!r} "
        f"value in {path}: {text!r}; expected 'true' or 'false'."
    )


def _read_retrieval_metrics(path):
    """Read marginal song-identification and within-song top-1 metrics."""
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {"metric", *STATISTIC_FIELDS}
        missing_fields = sorted(required - fieldnames)
        if missing_fields:
            raise ValueError(
                f"Missing required columns in {path}: {', '.join(missing_fields)}."
            )
        rows = list(reader)

    results = {}
    for retrieval_name, aliases in METRIC_ALIASES.items():
        matches = [
            row
            for row in rows
            if (row.get("metric") or "").strip() in aliases
        ]
        if not matches:
            raise ValueError(
                f"Missing {retrieval_name.replace('_', ' ')} top-1 metric in {path}; "
                f"expected one of: {', '.join(aliases)}."
            )
        if len(matches) > 1:
            names = ", ".join((row.get("metric") or "").strip() for row in matches)
            raise ValueError(
                f"Found multiple {retrieval_name.replace('_', ' ')} top-1 rows "
                f"in {path}: {names}."
            )

        row = matches[0]
        metric = row["metric"].strip()
        if "n_folds" in fieldnames and (row.get("n_folds") or "").strip() != "5":
            raise ValueError(
                f"Metric {metric!r} in {path} is not a five-fold average "
                f"(n_folds={row.get('n_folds')!r})."
            )
        if "unit" in fieldnames and (row.get("unit") or "").strip() not in (
            "",
            "proportion",
        ):
            raise ValueError(
                f"Metric {metric!r} in {path} must use proportion units, not "
                f"{row.get('unit')!r}."
            )

        values = {
            field: _parse_finite_number(row, field, metric, path)
            for field in STATISTIC_FIELDS
        }
        p_metric_aliases = {f"{alias}_p_upper" for alias in aliases}
        p_matches = [
            candidate
            for candidate in rows
            if (candidate.get("metric") or "").strip() in p_metric_aliases
        ]
        if len(p_matches) > 1:
            names = ", ".join(
                (candidate.get("metric") or "").strip()
                for candidate in p_matches
            )
            raise ValueError(
                f"Found multiple {retrieval_name.replace('_', ' ')} upper-tail "
                f"p-value metrics in {path}: {names}."
            )

        p_upper = None
        all_folds_significant = False
        if p_matches:
            p_row = p_matches[0]
            p_metric = p_row["metric"].strip()
            if (
                "n_folds" in fieldnames
                and (p_row.get("n_folds") or "").strip() != "5"
            ):
                raise ValueError(
                    f"Metric {p_metric!r} in {path} is not a five-fold average "
                    f"(n_folds={p_row.get('n_folds')!r})."
                )
            if "unit" in fieldnames and (p_row.get("unit") or "").strip() not in (
                "",
                "probability",
            ):
                raise ValueError(
                    f"Metric {p_metric!r} in {path} must use probability units, "
                    f"not {p_row.get('unit')!r}."
                )
            p_upper = _parse_finite_number(p_row, "value", p_metric, path)
            if not 0.0 <= p_upper <= 1.0:
                raise ValueError(
                    f"Metric {p_metric!r} in {path} must be between 0 and 1."
                )
            all_folds_significant = _parse_all_folds_significant(
                p_row, p_metric, path
            )
        elif "p_upper" in fieldnames and (row.get("p_upper") or "").strip():
            # Ridge aggregate CSVs keep the p-value on the metric row.
            p_upper = _parse_finite_number(row, "p_upper", metric, path)
            if not 0.0 <= p_upper <= 1.0:
                raise ValueError(
                    f"Metric {metric!r} in {path} must have a p-value "
                    "between 0 and 1."
                )
            all_folds_significant = _parse_all_folds_significant(
                row, metric, path
            )

        results[retrieval_name] = RetrievalResult(
            **values,
            p_upper=p_upper,
            all_folds_significant=all_folds_significant,
        )

    return results


def load_table(run_dir):
    """Load a run name and the five rows needed for the results table."""
    prefix, paths = discover_average_metric_csvs(run_dir)
    rows = []
    for split, label in SPLITS:
        metrics = _read_retrieval_metrics(paths[split])
        rows.append(
            TableRow(
                label=label,
                marginal_song_identification=metrics[
                    "marginal_song_identification"
                ],
                within_song_retrieval=metrics["within_song"],
            )
        )
    return prefix, rows


def _format_decimal(value, latex=False):
    """Match Table I's three-decimal style, including omitted leading zeros."""
    rounded = f"{value:.3f}"
    if rounded == "-0.000":
        rounded = ".000"
    elif rounded.startswith("-0."):
        rounded = "-" + rounded[2:]
    elif rounded.startswith("0."):
        rounded = rounded[1:]
    if latex and rounded.startswith("-"):
        rounded = r"\mathord{-}" + rounded[1:]
    return rounded


def _format_result(result, latex=False):
    return _format_mean_sd(
        result.value,
        result.value_sd,
        latex=latex,
        significant=result.all_folds_significant,
    )


def _format_mean_sd(mean, sd, latex=False, significant=False):
    separator = r" \pm " if latex else " ± "
    marker = r"^{*}" if latex and significant else "*" if significant else ""
    return (
        _format_decimal(mean, latex=latex)
        + marker
        + separator
        + _format_decimal(sd, latex=latex)
    )


def _display_name(prefix):
    return " ".join(word for word in re.split(r"[-_]+", prefix) if word)


def format_text_table(prefix, rows):
    """Return an aligned terminal table."""
    headers = (
        "Evaluation Regime",
        "Marginal Song ID Top-1",
        "Marginal Song ID κ",
        "Within-Song R@1",
        "Within-Song κ",
    )
    body = [
        (
            row.label,
            _format_result(row.marginal_song_identification),
            _format_mean_sd(
                row.marginal_song_identification.kappa,
                row.marginal_song_identification.kappa_sd,
            ),
            _format_result(row.within_song_retrieval),
            _format_mean_sd(
                row.within_song_retrieval.kappa,
                row.within_song_retrieval.kappa_sd,
            ),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in body))
        for index in range(len(headers))
    ]

    def format_line(cells):
        return "  ".join(
            cell.ljust(width) if index == 0 else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(cells, widths))
        )

    divider = "  ".join("-" * width for width in widths)
    title = (
        f"{_display_name(prefix)} (mean ± SD across five folds; "
        f"* p-upper ≤ {SIGNIFICANCE_THRESHOLD_LABEL} in all five folds)"
    )
    return "\n".join(
        [title, format_line(headers), divider, *(format_line(row) for row in body)]
    )


def format_markdown_table(prefix, rows):
    """Return a GitHub-flavored Markdown table."""
    lines = [
        f"**{_display_name(prefix)}** (mean ± SD across five folds; "
        f"* p-upper ≤ {SIGNIFICANCE_THRESHOLD_LABEL} in all five folds)",
        "",
        "| Evaluation Regime | Marginal Song Identification Top-1 | "
        "Marginal Song Identification κ | "
        "Within-Song Retrieval R@1 | Within-Song κ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        marginal_kappa = _format_mean_sd(
            row.marginal_song_identification.kappa,
            row.marginal_song_identification.kappa_sd,
        )
        within_kappa = _format_mean_sd(
            row.within_song_retrieval.kappa,
            row.within_song_retrieval.kappa_sd,
        )
        lines.append(
            f"| {row.label} | "
            f"{_format_result(row.marginal_song_identification)} | "
            f"{marginal_kappa} | {_format_result(row.within_song_retrieval)} | "
            f"{within_kappa} |"
        )
    return "\n".join(lines)


def _latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def format_latex_table(prefix, rows):
    """Return an IEEE-paper-style LaTeX table using booktabs rules."""
    model_name = _latex_escape(_display_name(prefix))
    lines = [
        r"\begin{table}[t]",
        (
            r"\caption{Mean top-1 retrieval performance ($R@1$) and "
            r"chance-normalized $\kappa$ (mean $\pm$ SD across five folds) for "
            + model_name
            + r". An asterisk denotes $p_{\mathrm{upper}} \leq "
            + SIGNIFICANCE_THRESHOLD_LABEL
            + r"$ in all "
            + r"five folds.}"
        ),
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Marginal Song Identification} "
        r"& \multicolumn{2}{c}{Within-Song Retrieval} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Evaluation Regime & $R@1$ & $\kappa$ & $R@1$ & $\kappa$ \\",
        r"\midrule",
    ]
    for row in rows:
        marginal_kappa = _format_mean_sd(
            row.marginal_song_identification.kappa,
            row.marginal_song_identification.kappa_sd,
            latex=True,
        )
        within_kappa = _format_mean_sd(
            row.within_song_retrieval.kappa,
            row.within_song_retrieval.kappa_sd,
            latex=True,
        )
        lines.append(
            f"{_latex_escape(row.label)} "
            f"& ${_format_result(row.marginal_song_identification, latex=True)}$ "
            f"& ${marginal_kappa}$ "
            f"& ${_format_result(row.within_song_retrieval, latex=True)}$ "
            f"& ${within_kappa}$ \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


FORMATTERS = {
    "text": format_text_table,
    "markdown": format_markdown_table,
    "latex": format_latex_table,
}


def format_run_table(run_dir, output_format="text"):
    """Load and format one run's complete average-metrics set."""
    try:
        formatter = FORMATTERS[output_format]
    except KeyError as exc:
        raise ValueError(
            f"Unknown output format {output_format!r}; choose from "
            f"{', '.join(FORMATTERS)}."
        ) from exc
    prefix, rows = load_table(run_dir)
    return formatter(prefix, rows)


def wrap_standalone_latex(table_content):
    """Wrap one or more table environments in a compilable LaTeX document."""
    table_content = table_content.strip()
    if not table_content:
        raise ValueError("Cannot create a standalone document from empty LaTeX.")
    return "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage{booktabs}",
            "",
            r"\begin{document}",
            table_content,
            r"\end{document}",
        ]
    )


def _combined_run_label(run_name):
    """Return a compact column heading for a final-results run name."""
    compact_name = re.sub(r"[^a-z0-9]", "", run_name.lower())
    if "eeg2mel" in compact_name:
        return "EEG2Mel"
    if "ridgeregression" in compact_name:
        return "Ridge Regression"
    if "cosineregression" in compact_name:
        return "Cosine Regression"
    if "subjectlayeroff" in compact_name or "subjectoff" in compact_name:
        return "Subject Layer Off"
    if "subjectlayeron" in compact_name or "subjecton" in compact_name:
        return "Subject Layer On"
    return _display_name(run_name).title()


def _parse_latex_run_table(table_path, run_name):
    """Extract the four metric cells for every split from results_table.tex."""
    table_path = Path(table_path)
    table = table_path.read_text(encoding="utf-8")
    if not table.strip():
        raise ValueError(f"LaTeX table is empty: {table_path}")

    rows = []
    for _, row_label in SPLITS:
        row_pattern = re.compile(
            rf"^{re.escape(_latex_escape(row_label))}\s*&\s*"
            rf"(.+?)\s*&\s*(.+?)\s*&\s*(.+?)\s*&\s*(.+?)\s*\\\\\s*$",
            re.MULTILINE,
        )
        matches = row_pattern.findall(table)
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {row_label!r} result row in "
                f"{table_path}, found {len(matches)}."
            )
        rows.append(tuple(cell.strip() for cell in matches[0]))

    return LatexRunTable(
        label=_combined_run_label(run_name),
        path=table_path,
        rows=tuple(rows),
    )


def _latex_cell_mean(cell, table_path):
    """Read the leading mean from a generated LaTeX mean-plus-SD cell."""
    normalized = cell.replace(r"\mathord{-}", "-")
    match = re.search(r"-?(?:\d+(?:\.\d*)?|\.\d+)", normalized)
    if match is None:
        raise ValueError(
            f"Could not read a numeric mean from {cell!r} in {table_path}."
        )
    return float(match.group())


def _bold_latex_cell(cell):
    """Bold one generated math-mode LaTeX cell without changing its value."""
    cell = cell.strip()
    if cell.startswith("$") and cell.endswith("$"):
        content = cell[1:-1]
        if content.startswith(r"\mathbf{"):
            return cell
        return rf"$\mathbf{{{content}}}$"
    return rf"\textbf{{{cell}}}"


def _format_combined_latex_table(run_tables):
    """Format parsed run tables as one dynamic, two-panel wide table."""
    if not run_tables:
        raise ValueError("At least one parsed run table is required.")

    column_count = 1 + 2 * len(run_tables)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Mean marginalized song identification and top-1 "
            r"within-song retrieval performance ($R@1$) with chance-normalized "
            r"$\kappa$ (mean $\pm$ SD across five folds). Bold indicates the "
            r"highest $\kappa$ across methods for each data split. An asterisk "
            r"denotes $p_{\mathrm{upper}} \leq "
            + SIGNIFICANCE_THRESHOLD_LABEL
            + r"$ in all five folds.}"
        ),
        r"\label{tab:retrieval_results}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        rf"\begin{{tabular}}{{{'l' + 'cc' * len(run_tables)}}}",
        r"\toprule",
        "& "
        + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{_latex_escape(run_table.label)}}}"
            for run_table in run_tables
        )
        + r" \\",
        *(
            rf"\cmidrule(lr){{{start}-{start + 1}}}"
            for start in range(2, column_count, 2)
        ),
        "Data Split & "
        + " & ".join(r"$R@1$ & $\kappa$" for _ in run_tables)
        + r" \\",
        r"\midrule",
    ]

    panels = (
        ("(a) Marginalized Song Identification", 0),
        ("(b) Within-Song Retrieval", 2),
    )
    for panel_index, (panel_label, first_cell) in enumerate(panels):
        lines.append(
            rf"\multicolumn{{{column_count}}}{{l}}{{\textit{{{panel_label}}}}} \\"
        )
        for row_index, (_, row_label) in enumerate(SPLITS):
            kappa_cells = [
                run_table.rows[row_index][first_cell + 1]
                for run_table in run_tables
            ]
            kappa_values = [
                _latex_cell_mean(cell, run_table.path)
                for cell, run_table in zip(kappa_cells, run_tables)
            ]
            best_kappa = max(kappa_values)

            cells = []
            for run_index, run_table in enumerate(run_tables):
                result_cells = run_table.rows[row_index]
                cells.append(result_cells[first_cell])
                kappa_cell = result_cells[first_cell + 1]
                if math.isclose(
                    kappa_values[run_index],
                    best_kappa,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    kappa_cell = _bold_latex_cell(kappa_cell)
                cells.append(kappa_cell)
            lines.append(
                f"{_latex_escape(row_label)} & "
                + " & ".join(cells)
                + r" \\",
            )
        if panel_index < len(panels) - 1:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def concatenate_results_tables(run_names, final_results_dir, standalone=False):
    """Combine per-run LaTeX results into one wide, two-panel table."""
    if not run_names:
        raise ValueError("At least one run name is required for concatenation.")

    final_results_dir = Path(final_results_dir).resolve()
    run_tables = []
    for run_name in run_names:
        run_name = str(run_name)
        run_path = Path(run_name)
        if (
            not run_name.strip()
            or run_path.is_absolute()
            or len(run_path.parts) != 1
            or run_path.name in (".", "..")
        ):
            raise ValueError(
                f"Invalid run name {run_name!r}; expected a directory name "
                f"under {final_results_dir}."
            )

        table_path = final_results_dir / run_name / RESULTS_TABLE_FILENAME
        if not table_path.is_file():
            raise FileNotFoundError(
                f"Run {run_name!r} has no {RESULTS_TABLE_FILENAME}: {table_path}"
            )
        run_tables.append(_parse_latex_run_table(table_path, run_name))

    output = _format_combined_latex_table(run_tables)
    if standalone:
        output = wrap_standalone_latex(output)

    output_path = final_results_dir / FINAL_TABLE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output + "\n", encoding="utf-8")
    return output_path


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Format the five average-metrics CSVs for a final-results run as "
            "a Table-I-style top-1 retrieval table."
        )
    )
    parser.add_argument(
        "run",
        nargs="?",
        help=(
            "Run name under runs/final_results, a run directory path, or its "
            "average-metrics directory."
        ),
    )
    parser.add_argument(
        "--concatenate-runs",
        nargs="+",
        metavar="RUN",
        help=(
            "Combine each named run's results_table.tex, in the supplied "
            "order, as column groups in one final_table.tex under "
            "runs/final_results."
        ),
    )
    parser.add_argument(
        "--format",
        choices=tuple(FORMATTERS),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help=(
            "Wrap LaTeX output in a complete document with the booktabs "
            "package so it can be compiled and previewed directly."
        ),
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
        help=(
            "Directory used to resolve a run name "
            "(default: <repo>/runs/final_results)."
        ),
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

    if args.concatenate_runs:
        if args.run is not None:
            parser.error(
                "RUN cannot be combined with --concatenate-runs."
            )
        if args.output is not None:
            parser.error(
                "--output cannot be combined with --concatenate-runs; the "
                f"output is {FINAL_TABLE_FILENAME} under the final-results "
                "directory."
            )
        try:
            output_path = concatenate_results_tables(
                args.concatenate_runs,
                final_results_dir,
                standalone=args.standalone,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            parser.error(str(exc))
        print(output_path)
        return output_path

    if args.run is None:
        parser.error("RUN is required unless --concatenate-runs is used.")
    if args.standalone and args.format != "latex":
        parser.error("--standalone requires --format latex.")

    supplied_path = Path(args.run)
    if supplied_path.is_absolute() or supplied_path.is_dir():
        run_dir = supplied_path
    else:
        run_dir = final_results_dir / supplied_path

    try:
        table = format_run_table(run_dir, args.format)
        if args.standalone:
            table = wrap_standalone_latex(table)
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
