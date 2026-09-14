"""Evaluate a trained EEG2Mel baseline checkpoint with the shared retrieval benchmark.

For each test-set row: reconstruct a 5-second waveform from the model's 5
predicted 1-second mel-spectrograms (concatenated, then one Griffin-Lim
pass), embed it with this project's own frozen MERT extractor, and compare
its full temporal sequence against the real MERT sequence. It runs the same
candidate-pool, song-search, and permutation evaluations as the main
contrastive model and writes the same test_metrics.csv schema.
"""

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.encoders import MERTFeatureExtractor
from src.evaluation import (
    build_retrieval_evaluation_cache,
    build_song_search_evaluation_cache,
    chance_section_regime,
    evaluate_candidate_pool_p_values,
    evaluate_embedding_benchmark,
    evaluate_section_null_regime,
    evaluate_section_regime,
    evaluate_section_search,
    evaluate_song_search,
    evaluate_song_search_permutation_test,
    log_candidate_pool_metrics,
    log_candidate_pool_null_metrics,
    log_chance_metrics,
    log_lift_metrics,
    log_null_metrics,
    log_p_metrics,
    log_section_chance_metrics,
    log_section_coverage,
    log_section_metrics,
    log_section_null_metrics,
    log_section_p_metrics,
    log_section_search_metrics,
    log_song_search_metrics,
    log_split_metrics,
    log_within_song_shuffle_metrics,
    save_section_search_results_csv,
    save_song_search_results_csv,
    save_song_search_summary_csv,
    summarize_section_coverage,
    summarize_section_search_results,
    summarize_song_search_results,
)
from src.evaluation.eeg2mel_baseline import (
    EEG2MelBaseline,
    EEG2MelEvalDataset,
    reconstruct_and_embed,
    resolve_eeg2mel_settings,
)
from scripts.test import save_test_metrics_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the EEG2Mel baseline against the shared retrieval benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/eeg2mel_paper.yaml",
        help="Path to the standalone paper EEG2Mel config.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name override. Defaults to config run_name.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional checkpoint override. Defaults to best.pt then eeg2mel.pt for the run.",
    )
    return parser.parse_args()


def resolve_split_path(base_dir, run_name, split_filename):
    split_path = Path(split_filename)
    if split_path.is_absolute():
        return split_path
    return base_dir / "runs" / run_name / "splits" / split_path


def _default_checkpoint_path(base_dir, run_name, explicit_path=None):
    if explicit_path is not None:
        checkpoint_path = Path(explicit_path)
        return checkpoint_path if checkpoint_path.is_absolute() else (base_dir / checkpoint_path)

    checkpoint_dir = base_dir / "runs" / "checkpoints" / run_name
    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "eeg2mel.pt"
    checkpoint_path = (
        best_checkpoint_path if best_checkpoint_path.exists() else last_checkpoint_path
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    cli_args = parse_args()
    base_dir = Path(__file__).parents[2]
    config_path = Path(cli_args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    args = OmegaConf.load(config_path)
    settings = resolve_eeg2mel_settings(args)

    run_name = cli_args.run_name or args["run_name"]
    log_dir = base_dir / "runs" / run_name
    benchmark_dir = log_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "eeg2mel_baseline_test.log"),
            logging.StreamHandler(),
        ],
    )
    OmegaConf.save(args, benchmark_dir / "eeg2mel_baseline_config.yaml")

    checkpoint_path = _default_checkpoint_path(
        base_dir=base_dir, run_name=run_name, explicit_path=cli_args.checkpoint_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logging.info("Loaded checkpoint from %s", checkpoint_path)

    model = EEG2MelBaseline(
        psd_shape=settings["psd_shape"],
        spec_shape=settings["spec_shape"],
        mel_sample_rate=settings["mel_sample_rate"],
        n_fft=settings["n_fft"],
        n_mels=settings["n_mels"],
        hop_length=settings["hop_length"],
        min_db=settings["mel_min_db"],
        max_db=settings["mel_max_db"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logging.info("Loading frozen MERT extractor for reconstructed-audio embedding.")
    mert_extractor = MERTFeatureExtractor().to(device).eval()

    test_path = resolve_split_path(base_dir, run_name, args["testing"]["filename"])
    test_dataset = EEG2MelEvalDataset(
        metadata_path=test_path, split="test", eeg_sample_rate=settings["eeg_sample_rate"],
    )
    test_dataloader = DataLoader(test_dataset, batch_size=settings["batch_size"], shuffle=False)
    logging.info("Loaded %d test windows from %s.", len(test_dataset), test_path)

    reconstructed_embeds = []
    real_embeds = []
    subject_ids = []
    song_ids = []
    window_idxs = []
    section_ids = []
    for batch in test_dataloader:
        batch_embeds = reconstruct_and_embed(
            model, batch["eeg_psd_stack"], mert_extractor, device,
        )
        reconstructed_embeds.append(batch_embeds.cpu())
        real_embeds.append(batch["audio"].cpu())
        subject_ids.extend(int(value) for value in batch["subject_id"])
        song_ids.extend(int(value) for value in batch["song_id"])
        window_idxs.extend(int(value) for value in batch["window_idx"])
        section_ids.extend(int(value) for value in batch["section_id"])

    reconstructed_embeds = torch.cat(reconstructed_embeds).numpy()
    real_embeds = torch.cat(real_embeds).numpy()

    candidate_gap_size = int(args["retrieval"]["candidate_gap_size"])
    song_search_marginal_temperature = float(
        args["testing"].get("song_search_marginal_temperature", 0.07)
    )
    benchmark = evaluate_embedding_benchmark(
        eeg_emb=reconstructed_embeds,
        audio_emb=real_embeds,
        song_ids=song_ids,
        window_idxs=window_idxs,
        ks=settings["ks"],
        gap=candidate_gap_size,
        n_perm=settings["n_perms"],
        n_within_song_shuffles=settings["n_within_song_shuffles"],
        song_search_marginal_temperature=song_search_marginal_temperature,
        rng=0,
    )

    test_results = dict(benchmark["observed"])
    # EEG2Mel is trained with mel reconstruction loss rather than the main
    # contrastive objective, so there is no directly comparable test loss.
    test_results["loss"] = float("nan")
    log_split_metrics("EEG2Mel", test_results)
    log_candidate_pool_metrics("EEG2Mel", benchmark["candidate_pool"])
    log_chance_metrics("EEG2Mel", benchmark["chance"])
    log_null_metrics("EEG2Mel", benchmark["null"])
    log_within_song_shuffle_metrics("EEG2Mel", benchmark["within_song_shuffle"])
    log_lift_metrics("EEG2Mel", benchmark["lift"])

    retrieval_cache = build_retrieval_evaluation_cache(
        eeg_emb=reconstructed_embeds,
        audio_emb=real_embeds,
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=candidate_gap_size,
    )

    logging.info("Precomputing the cached song-search similarity matrix.")
    song_search_cache = build_song_search_evaluation_cache(
        eeg_emb=reconstructed_embeds,
        audio_emb=real_embeds,
        song_ids=song_ids,
        window_idxs=window_idxs,
    )
    song_search_results = evaluate_song_search(
        eeg_emb=reconstructed_embeds,
        audio_emb=real_embeds,
        song_ids=song_ids,
        window_idxs=window_idxs,
        subject_ids=subject_ids,
        cache=song_search_cache,
        marginal_temperature=song_search_marginal_temperature,
    )
    song_search_summary = summarize_song_search_results(song_search_results)
    song_search_permutation_summary = evaluate_song_search_permutation_test(
        eeg_emb=reconstructed_embeds,
        audio_emb=real_embeds,
        song_ids=song_ids,
        window_idxs=window_idxs,
        n_perm=int(args["testing"].get("song_search_n_perms", settings["n_perms"])),
        rng=0,
        permutation_batch_size=int(
            args["testing"].get("song_search_permutation_batch_size", 16)
        ),
        cache=song_search_cache,
        marginal_temperature=song_search_marginal_temperature,
    )
    if not math.isclose(
        song_search_summary["song_search_song_top1"],
        song_search_permutation_summary["song_search_song_top1_permutation_observed"],
        abs_tol=1e-7,
    ):
        raise RuntimeError("Cached song-search accuracy does not match the detailed evaluator.")
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
    log_song_search_metrics("EEG2Mel", song_search_summary)

    song_search_csv_path = log_dir / args["testing"].get(
        "song_search_filename", "song_search_results.csv"
    )
    save_song_search_results_csv(song_search_results, song_search_csv_path)
    logging.info("Saved song-search per-query results to %s", song_search_csv_path)
    song_search_summary_csv_path = log_dir / "song_search_summary.csv"
    save_song_search_summary_csv(song_search_summary, song_search_summary_csv_path)
    logging.info("Saved song-search summary to %s", song_search_summary_csv_path)

    section_coverage = {}
    section_regime_results = {}
    section_chance_results = {}
    section_null_results = {}
    section_search_summary = {}
    if any(section_id != -1 for section_id in section_ids):
        section_coverage = summarize_section_coverage(section_ids)
        log_section_coverage("EEG2Mel", section_coverage)
        section_regime_results = evaluate_section_regime(
            eeg_emb=reconstructed_embeds,
            audio_emb=real_embeds,
            song_ids=song_ids,
            window_idxs=window_idxs,
            section_ids=section_ids,
            gap=candidate_gap_size,
            cache=retrieval_cache,
        )
        log_section_metrics("EEG2Mel", section_regime_results)
        section_chance_results = chance_section_regime(
            song_ids=song_ids,
            window_idxs=window_idxs,
            section_ids=section_ids,
            gap=candidate_gap_size,
        )
        log_section_chance_metrics("EEG2Mel", section_chance_results)
        section_null_results = evaluate_section_null_regime(
            eeg_emb=reconstructed_embeds,
            audio_emb=real_embeds,
            song_ids=song_ids,
            window_idxs=window_idxs,
            section_ids=section_ids,
            gap=candidate_gap_size,
            n_perm=settings["n_perms"],
            cache=retrieval_cache,
        )
        log_section_null_metrics("EEG2Mel", section_null_results)
        log_section_p_metrics("EEG2Mel", section_null_results)

        section_search_results = evaluate_section_search(
            eeg_emb=reconstructed_embeds,
            audio_emb=real_embeds,
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
        log_section_search_metrics("EEG2Mel", section_search_summary)
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

    logging.info("Computing permutation-based p-values last.")
    p_values = evaluate_candidate_pool_p_values(
        eeg_emb=reconstructed_embeds,
        audio_emb=real_embeds,
        window_idxs=window_idxs,
        song_ids=song_ids,
        gap=candidate_gap_size,
        n_perm=settings["n_perms"],
        within_song_shuffle_metrics=benchmark["within_song_shuffle"],
        cache=retrieval_cache,
    )
    log_p_metrics("EEG2Mel", p_values)
    log_candidate_pool_null_metrics("EEG2Mel", benchmark["null"])

    test_metrics_csv_path = save_test_metrics_csv(
        output_path=log_dir / "test_metrics.csv",
        n_test_windows=len(test_dataset),
        test_results=test_results,
        candidate_chance_results=benchmark["chance"],
        song_search_summary=song_search_summary,
        section_coverage=section_coverage,
        section_regime_results=section_regime_results,
        section_chance_results=section_chance_results,
        section_null_results=section_null_results,
        p_values=p_values,
    )
    logging.info("Saved per-run test metrics to %s", test_metrics_csv_path)

    result_payload = {
        "model_label": "eeg2mel_baseline",
        "source_type": "eeg2mel_gradient_regression",
        "split_filename": str(args["testing"]["filename"]),
        "split_name": "test",
        "ks": list(settings["ks"]),
        "candidate_gap_size": candidate_gap_size,
        "n_perms": settings["n_perms"],
        "n_within_song_shuffles": settings["n_within_song_shuffles"],
        "num_examples": len(test_dataset),
        "audio_pooling": settings["audio_pooling"],
        "griffin_lim_rand_init": model.griffin_lim_transform.rand_init,
        "checkpoint_path": str(checkpoint_path),
        "results": benchmark,
        "song_search": song_search_summary,
        "p_values": p_values,
    }
    if section_coverage:
        result_payload["section"] = {
            "coverage": section_coverage,
            "observed": section_regime_results,
            "chance": section_chance_results,
            "null": section_null_results,
            "search": section_search_summary,
        }
    output_path = benchmark_dir / settings["output_filename"]
    output_path.write_text(json.dumps(_jsonable(result_payload), indent=2))
    logging.info("Saved EEG2Mel benchmark results to %s", output_path)
