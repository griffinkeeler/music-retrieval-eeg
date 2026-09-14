"""Run the main evaluation script on a checkpoint and report retrieval, permutation, shuffle, and song-search metrics."""

import argparse
import csv
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.model import (
    build_audio_projection,
    resolve_alignment_objective_name,
    resolve_alignment_settings,
    restore_alignment_objective,
)
from src.data import EEGMusicWindowDataset
from src.config import load_config, resolve_split_path
from src.encoders import EEGEncoder
from src.evaluation import (log_split_metrics,
                            log_chance_metrics,
                            collect_embeddings,
                            build_retrieval_evaluation_cache,
                            evaluate_candidate_pool_regimes,
                            evaluate_candidate_pool_chance_regimes,
                            evaluate_candidate_pool_p_values,
                            evaluate_candidate_pool_null_regimes,
                            log_candidate_pool_null_metrics,
                            evaluate_within_song_shuffle_baseline,
                            log_p_metrics,
                            log_within_song_shuffle_metrics,
                            evaluate_song_search,
                            build_song_search_evaluation_cache,
                            evaluate_song_search_permutation_test,
                            summarize_song_search_results,
                            log_song_search_metrics,
                            save_song_search_results_csv,
                            save_song_search_summary_csv,
                            similarity_scores_for_query_index,
                            log_similarity_scores_grouped_by_song,
                            summarize_section_coverage,
                            log_section_coverage,
                            evaluate_section_regime,
                            chance_section_regime,
                            log_section_metrics,
                            log_section_chance_metrics,
                            evaluate_section_null_regime,
                            log_section_null_metrics,
                            log_section_p_metrics,
                            evaluate_section_search,
                            summarize_section_search_results,
                            log_section_search_metrics,
                            save_section_search_results_csv,)


def _test_metric_row(metric, value, unit, chance=None):
    return {
        "metric": metric,
        "value": value,
        "chance": "" if chance is None else chance,
        "gap": "" if chance is None else value - chance,
        "unit": unit,
    }


def save_test_metrics_csv(
    output_path,
    n_test_windows,
    test_results,
    candidate_chance_results,
    song_search_summary,
    section_coverage,
    section_regime_results,
    section_chance_results,
    section_null_results,
    p_values=None,
):
    """Save the headline metrics for one completed test run.

    Chance and gap are populated only when the evaluator computes an exact,
    directly comparable candidate-pool chance value. Gap is reported as the
    additive difference ``value - chance``. Permutation results retain the
    existing upper-tail p-values and add explicitly named lower-tail diagnostics.
    """
    rows = [_test_metric_row("test_windows", n_test_windows, "count")]
    if section_coverage:
        rows.append(
            _test_metric_row(
                "section_scored_windows",
                section_coverage["section_eval_n_scored"],
                "count",
            )
        )

    for regime_name, metric_prefix in (
        ("across_song", "across_song"),
        ("within_song", "within_song"),
        (
            "across_song_no_same_song_negatives",
            "across_song_no_same_song_negatives",
        ),
    ):
        for k in (1, 5, 10):
            key = f"{regime_name}_top{k}"
            rows.append(
                _test_metric_row(
                    f"{metric_prefix}_r_at_{k}",
                    test_results[key],
                    "proportion",
                    chance=candidate_chance_results[f"{key}_chance"],
                )
            )

    rows.extend(
        [
            _test_metric_row(
                "song_identification_top1",
                song_search_summary["song_search_song_top1"],
                "proportion",
                chance=song_search_summary.get("song_search_song_top1_chance"),
            ),
            _test_metric_row(
                "localization_mean_error",
                song_search_summary["song_search_mean_localization_error"],
                "windows",
            ),
        ]
    )
    if "song_search_marginal_top1" in song_search_summary:
        rows.append(
            _test_metric_row(
                "song_identification_marginal_top1",
                song_search_summary["song_search_marginal_top1"],
                "proportion",
                chance=song_search_summary.get(
                    "song_search_marginal_top1_chance"
                ),
            )
        )

    if p_values is not None:
        for regime_name, metric_prefix in (
            ("across_song", "across_song"),
            ("within_song", "within_song"),
            (
                "across_song_no_same_song_negatives",
                "across_song_no_same_song_negatives",
            ),
        ):
            for k in (1, 5, 10):
                upper_key = f"{regime_name}_top{k}_p"
                lower_key = f"{regime_name}_top{k}_p_lower"
                if upper_key in p_values:
                    rows.append(
                        _test_metric_row(
                            f"{metric_prefix}_r_at_{k}_p_upper",
                            p_values[upper_key],
                            "probability",
                        )
                    )
                if lower_key in p_values:
                    rows.append(
                        _test_metric_row(
                            f"{metric_prefix}_r_at_{k}_p_lower",
                            p_values[lower_key],
                            "probability",
                        )
                    )

        for diagnostic_name, prefix in (
            ("within_song_audio_shuffle", "within_song_audio_shuffle"),
            ("within_song_eeg_shuffle", "within_song_eeg_shuffle"),
        ):
            for k in (1, 5, 10):
                for tail_name, key_suffix in (
                    ("upper", "_p"),
                    ("lower", "_p_lower"),
                ):
                    key = f"{prefix}_top{k}{key_suffix}"
                    if key in p_values:
                        rows.append(
                            _test_metric_row(
                                f"{diagnostic_name}_top{k}_p_{tail_name}",
                                p_values[key],
                                "probability",
                            )
                        )

    for metric_name, summary_key in (
        ("song_identification_top1_p_upper", "song_search_song_top1_p"),
        ("song_identification_top1_p_lower", "song_search_song_top1_p_lower"),
        (
            "song_identification_marginal_top1_p_upper",
            "song_search_marginal_top1_p",
        ),
        (
            "song_identification_marginal_top1_p_lower",
            "song_search_marginal_top1_p_lower",
        ),
    ):
        if summary_key in song_search_summary:
            rows.append(
                _test_metric_row(
                    metric_name,
                    song_search_summary[summary_key],
                    "probability",
                )
            )

    for k in (1, 5, 10):
        for tail_name, key_suffix in (("upper", "_p"), ("lower", "_p_lower")):
            key = f"within_song_section_top{k}{key_suffix}"
            if key in section_null_results:
                rows.append(
                    _test_metric_row(
                        f"section_top{k}_p_{tail_name}",
                        section_null_results[key],
                        "probability",
                    )
                )

    if section_regime_results and section_chance_results and section_null_results:
        section_top1 = section_regime_results["within_song_section_top1"]
        section_chance_top1 = section_chance_results[
            "within_song_section_top1_chance"
        ]
        section_null_top1 = section_null_results[
            "within_song_section_top1_null_mean"
        ]
        rows.extend(
            [
                _test_metric_row(
                    "section_top1",
                    section_top1,
                    "proportion",
                    chance=section_chance_top1,
                ),
                _test_metric_row(
                    "section_null_top1",
                    section_null_top1,
                    "proportion",
                ),
                _test_metric_row(
                    "section_lift_over_null",
                    section_top1 - section_null_top1,
                    "proportion",
                ),
            ]
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("metric", "value", "chance", "gap", "unit"),
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on the configured test split.",
    )
    parser.add_argument(
        "--config",
        default="configs/chunk_out.yaml",
        help="Path to a paper experiment config.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name override. Defaults to config run_name.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help=(
            "Optional checkpoint override. Defaults to best.pt then eeg_encoder.pt "
            "for the run."
        ),
    )
    return parser.parse_args()


def _default_checkpoint_path(base_dir, run_name, explicit_path=None):
    if explicit_path is not None:
        checkpoint_path = Path(explicit_path)
        return checkpoint_path if checkpoint_path.is_absolute() else (base_dir / checkpoint_path)

    checkpoint_dir = base_dir / "runs" / "checkpoints" / run_name
    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "eeg_encoder.pt"
    checkpoint_path = (
        best_checkpoint_path if best_checkpoint_path.exists()
        else last_checkpoint_path
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    cli_args = parse_args()
    base_dir = PROJECT_ROOT
    config_path = Path(cli_args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    args = load_config(config_path)

    run_name = cli_args.run_name or args["run_name"]
    log_dir = base_dir / "runs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(args, log_dir / "config.yaml")

    log_name = args["testing"]["log_name"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / log_name),
            logging.StreamHandler(),
        ],
    )

    checkpoint_path = _default_checkpoint_path(
        base_dir=base_dir,
        run_name=run_name,
        explicit_path=cli_args.checkpoint_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logging.info("Loaded checkpoint from %s", checkpoint_path)
    checkpoint_config = checkpoint.get("config", {})

    use_subject_layer = checkpoint_config.get(
        "ablations", {}
    ).get(
        "use_subject_layer",
        args["ablations"]["use_subject_layer"],
    )
    effective_config = checkpoint_config if checkpoint_config else args
    alignment_settings = resolve_alignment_settings(effective_config)

    eeg_encoder = EEGEncoder(
        embedding_dim=alignment_settings["eeg_embedding_dim"],
        use_subject_layer=use_subject_layer,
        use_projection_head=alignment_settings["use_projection_head"],
    ).to(device)
    eeg_encoder.load_state_dict(checkpoint["eeg_encoder_state_dict"])
    eeg_encoder.eval()

    audio_projection = build_audio_projection(
        768,
        alignment_settings["audio_out_features"],
        mode=alignment_settings["audio_projection_mode"],
    ).to(device)
    audio_projection.load_state_dict(checkpoint["audio_projection_state_dict"])
    audio_projection.eval()

    objective = restore_alignment_objective(
        effective_config,
        checkpoint,
    ).to(device)
    objective.eval()
    logging.info(
        "Evaluation objective: %s",
        checkpoint.get(
            "objective_name",
            resolve_alignment_objective_name(effective_config),
        ),
    )

    test_filename = args["testing"]["filename"]
    test_filepath = resolve_split_path(base_dir, args, test_filename)

    test_dataset = EEGMusicWindowDataset(
        metadata_path=test_filepath,
        split="test",
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args["training"]["batch_size"],
        shuffle=False,
    )

    logging.info(
        "Loaded the test dataset with %s windows across %s batches. "
        "| Training Filename: %s",
        len(test_dataset),
        len(test_loader),
        test_filename,
    )

    outputs = collect_embeddings(
        dataloader=test_loader,
        eeg_encoder=eeg_encoder,
        audio_projection=audio_projection,
        clip=objective,
        device=device,
    )

    eeg_embeds = outputs["eeg_embeds"]
    audio_embeds = outputs["audio_embeds"]
    subject_ids = outputs["subject_ids"]
    song_ids = outputs["song_ids"]
    window_idxs = outputs["window_idxs"]
    section_ids = outputs["section_ids"]

    candidate_gap_size = args["retrieval"]["candidate_gap_size"]
    n_perms = args["testing"]["n_perms"]
    n_within_song_shuffles = args["testing"]["n_within_song_shuffles"]
    n_song_search_perms = args["testing"].get("song_search_n_perms", n_perms)
    song_search_permutation_batch_size = args["testing"].get(
        "song_search_permutation_batch_size",
        16,
    )
    song_search_marginal_temperature = float(
        args["testing"].get("song_search_marginal_temperature", 0.07)
    )

    logging.info("Precomputing the retrieval similarity matrix and candidate pools.")
    retrieval_cache = build_retrieval_evaluation_cache(
        eeg_emb=eeg_embeds,
        audio_emb=audio_embeds,
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=candidate_gap_size,
    )

    test_results = evaluate_candidate_pool_regimes(
        eeg_embeds,
        audio_embeds,
        song_ids=song_ids,
        window_idxs=window_idxs,
        ks=(1, 5, 10),
        gap=candidate_gap_size,
        cache=retrieval_cache,
    )
    test_results["loss"] = outputs["loss"]

    log_split_metrics("Test", test_results)

    candidate_chance_results = evaluate_candidate_pool_chance_regimes(
        song_ids=song_ids,
        window_idxs=window_idxs,
        ks=(1, 5, 10),
        gap=candidate_gap_size,
    )
    log_chance_metrics("Test", candidate_chance_results)

    within_song_shuffle_metrics = evaluate_within_song_shuffle_baseline(
        eeg_emb=eeg_embeds,
        audio_emb=audio_embeds,
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=candidate_gap_size,
        n_shuffle=n_within_song_shuffles,
        cache=retrieval_cache,
    )

    log_within_song_shuffle_metrics("Test", within_song_shuffle_metrics)

    logging.info("Precomputing the cached song-search similarity matrix.")
    song_search_cache = build_song_search_evaluation_cache(
        eeg_emb=eeg_embeds,
        audio_emb=audio_embeds,
        song_ids=song_ids,
        window_idxs=window_idxs,
    )
    song_search_results = evaluate_song_search(
        eeg_emb=eeg_embeds,
        audio_emb=audio_embeds,
        song_ids=song_ids,
        window_idxs=window_idxs,
        subject_ids=subject_ids,
        cache=song_search_cache,
        marginal_temperature=song_search_marginal_temperature,
    )
    song_search_summary = summarize_song_search_results(song_search_results)
    song_search_permutation_summary = evaluate_song_search_permutation_test(
        eeg_emb=eeg_embeds,
        audio_emb=audio_embeds,
        song_ids=song_ids,
        window_idxs=window_idxs,
        n_perm=n_song_search_perms,
        rng=0,
        permutation_batch_size=song_search_permutation_batch_size,
        cache=song_search_cache,
        marginal_temperature=song_search_marginal_temperature,
    )
    if not math.isclose(
        song_search_summary["song_search_song_top1"],
        song_search_permutation_summary[
            "song_search_song_top1_permutation_observed"
        ],
        abs_tol=1e-7,
    ):
        raise RuntimeError(
            "Cached song-search accuracy does not match the detailed evaluator."
        )
    if not math.isclose(
        song_search_summary["song_search_marginal_top1"],
        song_search_permutation_summary[
            "song_search_marginal_top1_permutation_observed"
        ],
        abs_tol=1e-7,
    ):
        raise RuntimeError(
            "Cached marginal song-search accuracy does not match the detailed "
            "evaluator."
        )
    song_search_summary.update(song_search_permutation_summary)
    log_song_search_metrics("Test", song_search_summary)
    song_search_file_name = args["testing"]["song_search_filename"]
    song_search_csv_path = log_dir / song_search_file_name
    save_song_search_results_csv(song_search_results, song_search_csv_path)
    logging.info("Saved song-search per-query results to %s", song_search_csv_path)
    song_search_summary_csv_path = log_dir / "song_search_summary.csv"
    save_song_search_summary_csv(song_search_summary, song_search_summary_csv_path)
    logging.info("Saved song-search summary to %s", song_search_summary_csv_path)

    section_coverage = {}
    section_regime_results = {}
    section_chance_results = {}
    section_null_results = {}
    if any(section_id != -1 for section_id in section_ids):
        # Section-aware retrieval is optional when callers supply section_id.
        section_coverage = summarize_section_coverage(section_ids)
        log_section_coverage("Test", section_coverage)

        section_regime_results = evaluate_section_regime(
            eeg_emb=eeg_embeds,
            audio_emb=audio_embeds,
            song_ids=song_ids,
            window_idxs=window_idxs,
            section_ids=section_ids,
            gap=candidate_gap_size,
            cache=retrieval_cache,
        )
        log_section_metrics("Test", section_regime_results)

        section_chance_results = chance_section_regime(
            song_ids=song_ids,
            window_idxs=window_idxs,
            section_ids=section_ids,
            gap=candidate_gap_size,
        )
        log_section_chance_metrics("Test", section_chance_results)

        section_null_results = evaluate_section_null_regime(
            eeg_emb=eeg_embeds,
            audio_emb=audio_embeds,
            song_ids=song_ids,
            window_idxs=window_idxs,
            section_ids=section_ids,
            gap=candidate_gap_size,
            n_perm=n_perms,
            cache=retrieval_cache,
        )
        log_section_null_metrics("Test", section_null_results)
        log_section_p_metrics("Test", section_null_results)

        section_search_results = evaluate_section_search(
            eeg_emb=eeg_embeds,
            audio_emb=audio_embeds,
            song_ids=song_ids,
            window_idxs=window_idxs,
            section_ids=section_ids,
            subject_ids=subject_ids,
            gap=candidate_gap_size,
            cache=retrieval_cache,
        )
        section_search_summary = summarize_section_search_results(
            section_search_results
        )
        log_section_search_metrics("Test", section_search_summary)
        section_search_csv_path = log_dir / "section_search_results.csv"
        save_section_search_results_csv(
            section_search_results,
            section_search_csv_path,
        )
        logging.info(
            "Saved section-search per-query results to %s",
            section_search_csv_path,
        )
    else:
        logging.info("Skipping optional section metrics: section_id is unavailable.")

    similarity_scores = similarity_scores_for_query_index(
        query_index=0,
        eeg_emb=eeg_embeds,
        audio_emb=audio_embeds,
        song_ids=song_ids,
        window_idxs=window_idxs,
    )

    query_idx = 0

    log_similarity_scores_grouped_by_song(
        split_name="Test",
        grouped_scores=similarity_scores,
        query_index=query_idx,
        query_song_id=song_ids[query_idx],
        query_window_idx=window_idxs[query_idx],
    )

    logging.info("Computing permutation-based p-values last.")
    p_values = evaluate_candidate_pool_p_values(
        eeg_emb=eeg_embeds,
        audio_emb=audio_embeds,
        window_idxs=window_idxs,
        song_ids=song_ids,
        gap=candidate_gap_size,
        n_perm=n_perms,
        within_song_shuffle_metrics=within_song_shuffle_metrics,
        cache=retrieval_cache,
    )

    log_p_metrics("Test", p_values)

    candidate_pool_null_regimes = evaluate_candidate_pool_null_regimes(
        eeg_emb=eeg_embeds,
        audio_emb=audio_embeds,
        window_idxs=window_idxs,
        song_ids=song_ids,
        gap=candidate_gap_size,
        n_perm=n_perms,
        cache=retrieval_cache,
    )
    log_candidate_pool_null_metrics("Test", candidate_pool_null_regimes)

    test_metrics_csv_path = save_test_metrics_csv(
        output_path=log_dir / "test_metrics.csv",
        n_test_windows=len(test_dataset),
        test_results=test_results,
        candidate_chance_results=candidate_chance_results,
        song_search_summary=song_search_summary,
        section_coverage=section_coverage,
        section_regime_results=section_regime_results,
        section_chance_results=section_chance_results,
        section_null_results=section_null_results,
        p_values=p_values,
    )
    logging.info("Saved per-run test metrics to %s", test_metrics_csv_path)
