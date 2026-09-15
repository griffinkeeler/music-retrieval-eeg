"""Train and evaluate ridge baselines on the 25 original five-second splits."""

import argparse
import csv
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from scripts.average_test_results import average_test_results
from src.config import load_config, resolve_split_directory


LOGGER = logging.getLogger(__name__)
SPLIT_FAMILIES = (
    ("chunk_out", "chunkout"),
    ("random_segment_out", "randomsegmentout"),
    ("song_out", "songout"),
    ("subject_out", "subjectout"),
    ("subject_song_out", "subjectsongout"),
)
RETRIEVAL_REGIMES = (
    ("across_song", "across_song"),
    ("within_song", "within_song"),
    (
        "across_song_no_same_song_negatives",
        "across_song_no_same_song_negatives",
    ),
)
METRIC_FIELDS = (
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
COMBINED_ID_FIELDS = (
    "split_family",
    "fold",
    "split_filename",
    "run_name",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train ridge baselines on all five folds of the five original "
            "split families and save per-split and combined metrics CSVs."
        )
    )
    parser.add_argument("--config", default="configs/ridge_paper.yaml")
    parser.add_argument(
        "--split-run-name",
        default=None,
        help=(
            "Optional run containing source CSVs under runs/<name>/splits/. "
            "Defaults to split_directory in the config."
        ),
    )
    parser.add_argument(
        "--output-run-prefix",
        default=None,
        help=(
            "Optional name for the grouped output directory and prefix for "
            "the 25 ridge fold runs. "
            "Defaults to run_name in the config."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Combined metrics CSV path. Defaults to "
            "runs/final_results/<output-run-prefix>/benchmarks/."
        ),
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--n-perms", type=int, default=None)
    parser.add_argument("--n-within-song-shuffles", type=int, default=None)
    parser.add_argument("--n-song-search-perms", type=int, default=None)
    parser.add_argument(
        "--song-search-permutation-batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--song-search-marginal-temperature",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Reuse an existing per-run ridge result JSON instead of retraining.",
    )
    return parser.parse_args(argv)


def expected_split_specs(split_dir):
    """Return and validate the fixed five-family by five-fold split matrix."""
    split_dir = Path(split_dir)
    specs = []
    missing = []
    for family, run_label in SPLIT_FAMILIES:
        for fold in range(5):
            split_path = split_dir / f"{family}_fold{fold}.csv"
            if not split_path.is_file():
                missing.append(split_path)
            specs.append(
                {
                    "family": family,
                    "run_label": run_label,
                    "fold": fold,
                    "path": split_path,
                }
            )
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required ridge split files:\n{preview}")
    return specs


def _empty_metric_row(metric, value, unit):
    return {
        "space": "pooled_mert",
        "metric": metric,
        "value": value,
        "chance": "",
        "gap": "",
        "chance_ratio": "",
        "null_mean": "",
        "p_upper": "",
        "p_lower": "",
        "unit": unit,
    }


def ridge_metric_rows(payload):
    """Convert one ridge result payload to the EEG2MEL-style CSV schema."""
    results = payload["results"]
    observed = results["observed"]
    chance = results["chance"]
    null = results["null"]
    lift = results.get("lift", {})
    rows = [
        _empty_metric_row("train_windows", payload["num_train_examples"], "count"),
        _empty_metric_row("test_windows", payload["num_examples"], "count"),
    ]

    for regime, label in RETRIEVAL_REGIMES:
        for k in payload["ks"]:
            key = f"{regime}_top{k}"
            value = observed[key]
            chance_value = chance[f"{key}_chance"]
            rows.append(
                {
                    "space": "pooled_mert",
                    "metric": f"{label}_r_at_{k}",
                    "value": value,
                    "chance": chance_value,
                    "gap": lift.get(
                        f"{key}_minus_chance", value - chance_value
                    ),
                    "chance_ratio": lift.get(f"{key}_chance_ratio", ""),
                    "null_mean": null[f"{key}_null_mean"],
                    "p_upper": null[f"{key}_p"],
                    "p_lower": null[f"{key}_p_lower"],
                    "unit": "proportion",
                }
            )

    song_search = results.get("song_search")
    if song_search is not None:
        for metric_name, key_prefix in (
            ("song_identification_top1", "song_search_song_top1"),
            (
                "song_identification_marginal_top1",
                "song_search_marginal_top1",
            ),
        ):
            if key_prefix not in song_search:
                continue
            value = song_search[key_prefix]
            chance_value = song_search[f"{key_prefix}_chance"]
            rows.append(
                {
                    "space": "pooled_mert",
                    "metric": metric_name,
                    "value": value,
                    "chance": chance_value,
                    "gap": song_search.get(
                        f"{key_prefix}_minus_chance",
                        value - chance_value,
                    ),
                    "chance_ratio": (
                        value / chance_value if chance_value > 0 else ""
                    ),
                    "null_mean": song_search[f"{key_prefix}_null_mean"],
                    "p_upper": song_search[f"{key_prefix}_p"],
                    "p_lower": song_search[f"{key_prefix}_p_lower"],
                    "unit": "proportion",
                }
            )

    shuffle = results["within_song_shuffle"]
    for source, label in (
        ("within_song_audio_shuffle", "within_song_audio_shuffle"),
        ("within_song_eeg_shuffle", "within_song_prediction_shuffle"),
    ):
        for k in payload["ks"]:
            rows.append(
                {
                    "space": "pooled_mert",
                    "metric": f"{label}_r_at_{k}",
                    "value": shuffle[f"{source}_top{k}_mean"],
                    "chance": "",
                    "gap": "",
                    "chance_ratio": "",
                    "null_mean": "",
                    "p_upper": shuffle[f"{source}_top{k}_p"],
                    "p_lower": shuffle[f"{source}_top{k}_p_lower"],
                    "unit": "proportion",
                }
            )

    diagnostics = payload["regression_metrics"]
    for metric, unit in (
        ("mse", "pooled_mert_squared"),
        ("r2_variance_weighted", "proportion"),
        ("matched_cosine_mean", "cosine"),
    ):
        rows.append(
            _empty_metric_row(
                f"regression_{metric}", diagnostics[metric], unit
            )
        )
    return rows


def write_metrics_csv(rows, output_path, fieldnames=METRIC_FIELDS):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _append_override(command, flag, value):
    if value is not None:
        command.extend((flag, str(value)))


def _train_one_split(
    *,
    base_dir,
    base_config,
    temp_dir,
    split_spec,
    output_run_name,
    alpha=None,
    n_perms=None,
    n_within_song_shuffles=None,
    n_song_search_perms=None,
    song_search_permutation_batch_size=None,
    song_search_marginal_temperature=None,
    skip_completed=False,
):
    split_path = split_spec["path"].resolve()
    cfg = OmegaConf.create(OmegaConf.to_container(base_config, resolve=False))
    cfg.run_name = output_run_name
    cfg.training.filename = str(split_path)
    cfg.testing.filename = str(split_path)
    ridge_cfg = cfg.get("ridge_baseline", {}) or {}
    output_filename = str(
        ridge_cfg.get("output_filename", "ridge_baseline_test.json")
    )
    result_path = (
        Path(base_dir)
        / "runs"
        / output_run_name
        / "benchmarks"
        / output_filename
    )

    reuse_completed = False
    if skip_completed and result_path.is_file():
        existing_payload = json.loads(result_path.read_text())
        existing_song_search = existing_payload.get("results", {}).get(
            "song_search",
            {},
        )
        reuse_completed = "song_search_marginal_top1" in existing_song_search
        if not reuse_completed:
            LOGGER.info(
                "Existing ridge result predates marginal song identification; "
                "recomputing: %s",
                result_path,
            )

    if not reuse_completed:
        temp_config_path = (
            Path(temp_dir)
            / f"{split_spec['family']}_fold{split_spec['fold']}.yaml"
        )
        OmegaConf.save(cfg, temp_config_path)
        command = [
            sys.executable,
            "-m",
            "src.evaluation.train_ridge_baseline",
            "--config",
            str(temp_config_path),
        ]
        _append_override(command, "--alpha", alpha)
        _append_override(command, "--n-perms", n_perms)
        _append_override(
            command,
            "--n-within-song-shuffles",
            n_within_song_shuffles,
        )
        _append_override(
            command,
            "--n-song-search-perms",
            n_song_search_perms,
        )
        _append_override(
            command,
            "--song-search-permutation-batch-size",
            song_search_permutation_batch_size,
        )
        _append_override(
            command,
            "--song-search-marginal-temperature",
            song_search_marginal_temperature,
        )
        subprocess.run(command, cwd=base_dir, check=True)
    else:
        LOGGER.info("Reusing completed ridge result: %s", result_path)

    if not result_path.is_file():
        raise FileNotFoundError(
            f"Ridge training completed without producing {result_path}."
        )
    return result_path, json.loads(result_path.read_text())


def main(cli_args=None):
    cli_args = parse_args() if cli_args is None else cli_args
    base_dir = Path(__file__).resolve().parents[2]
    config_path = Path(cli_args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    base_config = load_config(config_path)
    output_run_prefix = (
        cli_args.output_run_prefix
        if cli_args.output_run_prefix is not None
        else str(base_config["run_name"])
    )
    output_run_prefix = output_run_prefix.strip()
    if (
        not output_run_prefix
        or Path(output_run_prefix).name != output_run_prefix
        or output_run_prefix in {".", ".."}
    ):
        raise ValueError(
            "Output run prefix must be a non-blank directory name without slashes."
        )
    output_run_dir = base_dir / "runs" / "final_results" / output_run_prefix
    model_run_prefix = f"final_results/{output_run_prefix}/{output_run_prefix}"

    split_dir = (
        base_dir / "runs" / cli_args.split_run_name / "splits"
        if cli_args.split_run_name is not None
        else resolve_split_directory(base_dir, base_config)
    )
    split_specs = expected_split_specs(split_dir)
    combined_path = (
        Path(cli_args.output_csv)
        if cli_args.output_csv is not None
        else output_run_dir / "benchmarks" / "ridge_all_splits_metrics.csv"
    )
    if not combined_path.is_absolute():
        combined_path = base_dir / combined_path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    combined_rows = []
    with tempfile.TemporaryDirectory(prefix="ridge-all-splits-") as temp_dir:
        for index, split_spec in enumerate(split_specs, start=1):
            output_run_name = (
                f"{model_run_prefix}-{split_spec['run_label']}"
                f"-fold{split_spec['fold']}"
            )
            LOGGER.info(
                "Ridge split %d/%d: %s -> %s",
                index,
                len(split_specs),
                split_spec["path"],
                output_run_name,
            )
            result_path, payload = _train_one_split(
                base_dir=base_dir,
                base_config=base_config,
                temp_dir=temp_dir,
                split_spec=split_spec,
                output_run_name=output_run_name,
                alpha=cli_args.alpha,
                n_perms=cli_args.n_perms,
                n_within_song_shuffles=cli_args.n_within_song_shuffles,
                n_song_search_perms=getattr(
                    cli_args,
                    "n_song_search_perms",
                    None,
                ),
                song_search_permutation_batch_size=(
                    getattr(
                        cli_args,
                        "song_search_permutation_batch_size",
                        None,
                    )
                ),
                song_search_marginal_temperature=getattr(
                    cli_args,
                    "song_search_marginal_temperature",
                    None,
                ),
                skip_completed=cli_args.skip_completed,
            )
            metric_rows = ridge_metric_rows(payload)
            per_run_csv = result_path.with_name("ridge_baseline_test_metrics.csv")
            write_metrics_csv(metric_rows, per_run_csv)

            for row in metric_rows:
                combined_rows.append(
                    {
                        "split_family": split_spec["family"],
                        "fold": split_spec["fold"],
                        "split_filename": split_spec["path"].name,
                        "run_name": output_run_name,
                        **row,
                    }
                )
            write_metrics_csv(
                combined_rows,
                combined_path,
                fieldnames=COMBINED_ID_FIELDS + METRIC_FIELDS,
            )
            LOGGER.info("Saved per-run ridge metrics to %s", per_run_csv)

    LOGGER.info("Saved all-split ridge metrics to %s", combined_path)
    average_paths = average_test_results(
        output_run_dir,
        output_dir=output_run_dir / "average_metrics",
        output_prefix=output_run_prefix,
    )
    LOGGER.info(
        "Saved five-fold ridge averages to %s",
        average_paths[0].parent,
    )
    return combined_path


if __name__ == "__main__":
    main()
