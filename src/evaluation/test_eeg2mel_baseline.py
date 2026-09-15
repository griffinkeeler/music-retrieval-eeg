"""Evaluate EEG2Mel in mel space, with reconstructed-audio MERT as an option.

The default follows ``feature/eeg2mel``: compare predicted and target mel
spectrograms directly using reconstruction diagnostics, cosine retrieval, and
nearest-MSE retrieval. Passing ``--mert-space`` additionally runs the slower
predicted-mel -> Griffin-Lim waveform -> frozen-MERT evaluation.
"""

import argparse
import csv
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from scripts.test import save_test_metrics_csv
from src.evaluation import (
    RETRIEVAL_REGIMES,
    build_retrieval_evaluation_cache,
    build_song_search_evaluation_cache,
    chance_section_regime,
    evaluate_candidate_pool_chance_regimes,
    evaluate_candidate_pool_null_regimes,
    evaluate_candidate_pool_regimes,
    evaluate_section_null_regime,
    evaluate_section_regime,
    evaluate_section_search,
    evaluate_song_search,
    evaluate_song_search_permutation_test,
    evaluate_within_song_shuffle_baseline,
    log_candidate_pool_metrics,
    log_chance_metrics,
    log_lift_metrics,
    log_null_metrics,
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
    summarize_candidate_pool_sizes,
    summarize_section_coverage,
    summarize_section_search_results,
    summarize_song_search_results,
)
from src.evaluation.eeg2mel_baseline import (
    EEG2MelBaseline,
    EEG2MelEvalDataset,
    concatenate_sub_window_mels,
    embed_predicted_mels,
    predict_mel_windows,
    resolve_eeg2mel_settings,
)


RICH_METRIC_FIELDS = (
    "space",
    "metric",
    "value",
    "chance",
    "null_mean",
    "p_upper",
    "p_lower",
    "unit",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate EEG2Mel in mel space and optionally in reconstructed "
            "MERT space."
        )
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
        help=(
            "Optional checkpoint override. Defaults to best.pt then "
            "eeg2mel.pt for the run."
        ),
    )
    parser.add_argument(
        "--mert-space",
        action="store_true",
        help=(
            "Also reconstruct waveforms and run frozen-MERT retrieval. "
            "Mel-space evaluation is always run."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Evaluation device: auto, cpu, cuda, or mps.",
    )
    parser.add_argument("--n-perms", type=int, default=None)
    parser.add_argument("--n-within-song-shuffles", type=int, default=None)
    return parser.parse_args(argv)


def _resolve_device(device_name="auto"):
    device_name = str(device_name)
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")
    return device


def resolve_split_path(base_dir, run_name, split_filename):
    split_path = Path(split_filename)
    if split_path.is_absolute():
        return split_path
    return base_dir / "runs" / run_name / "splits" / split_path


def _default_checkpoint_path(base_dir, run_name, explicit_path=None):
    if explicit_path is not None:
        checkpoint_path = Path(explicit_path)
        return (
            checkpoint_path
            if checkpoint_path.is_absolute()
            else base_dir / checkpoint_path
        )

    checkpoint_dir = base_dir / "runs" / "checkpoints" / run_name
    for filename in ("best.pt", "eeg2mel.pt"):
        checkpoint_path = checkpoint_dir / filename
        if checkpoint_path.exists():
            return checkpoint_path
    raise FileNotFoundError(f"Checkpoint not found under: {checkpoint_dir}")


def _ssim_per_sample(predicted, target, data_range=2.0, chunk_size=32):
    """Compute windowed SSIM without adding a scikit-image dependency."""
    predicted = torch.as_tensor(predicted, dtype=torch.float32, device="cpu")
    target = torch.as_tensor(target, dtype=torch.float32, device="cpu")
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("SSIM inputs must share shape [N,H,W].")

    coordinates = torch.arange(11, dtype=torch.float32) - 5
    gaussian = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    kernel = torch.outer(gaussian, gaussian).reshape(1, 1, 11, 11)
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    scores = []
    for start in range(0, len(predicted), int(chunk_size)):
        x = predicted[start : start + chunk_size].unsqueeze(1)
        y = target[start : start + chunk_size].unsqueeze(1)
        x = F.pad(x, (5, 5, 5, 5), mode="reflect")
        y = F.pad(y, (5, 5, 5, 5), mode="reflect")
        mu_x = F.conv2d(x, kernel)
        mu_y = F.conv2d(y, kernel)
        mu_x_sq = mu_x.square()
        mu_y_sq = mu_y.square()
        mu_xy = mu_x * mu_y
        sigma_x = F.conv2d(x.square(), kernel) - mu_x_sq
        sigma_y = F.conv2d(y.square(), kernel) - mu_y_sq
        sigma_xy = F.conv2d(x * y, kernel) - mu_xy
        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
        scores.append((numerator / denominator.clamp_min(1e-12)).mean((1, 2, 3)))
    return torch.cat(scores)


def reconstruction_metrics(predicted, target, data_range=2.0):
    """Return the reconstruction metrics used by ``feature/eeg2mel``."""
    predicted = torch.as_tensor(predicted, dtype=torch.float32, device="cpu")
    target = torch.as_tensor(target, dtype=torch.float32, device="cpu")
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("Reconstruction inputs must share shape [N,H,W].")
    if len(predicted) == 0:
        raise ValueError("At least one reconstruction is required.")
    data_range = float(data_range)
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("Reconstruction data_range must be finite and positive.")

    error = predicted - target
    per_sample_mse = error.square().flatten(1).mean(1)
    per_sample_mae = error.abs().flatten(1).mean(1)
    per_sample_psnr = 10.0 * torch.log10(
        torch.tensor(data_range**2) / per_sample_mse.clamp_min(1e-12)
    )
    per_sample_cosine = F.cosine_similarity(
        predicted.flatten(1),
        target.flatten(1),
        dim=1,
        eps=1e-8,
    )
    per_sample_ssim = _ssim_per_sample(
        predicted,
        target,
        data_range=data_range,
    )
    return {
        "n_examples": int(len(predicted)),
        "mse": float(per_sample_mse.mean()),
        "mae": float(per_sample_mae.mean()),
        "ssim_mean": float(per_sample_ssim.mean()),
        "psnr_db_mean": float(per_sample_psnr.mean()),
        "matched_cosine_mean": float(per_sample_cosine.mean()),
    }


def _benchmark_lifts(observed, chance, nulls, ks):
    lifts = {}
    for regime in RETRIEVAL_REGIMES:
        for k in ks:
            key = f"{regime}_top{k}"
            chance_value = chance[f"{key}_chance"]
            observed_value = observed[key]
            null_value = nulls[f"{key}_null_mean"]
            lifts[f"{key}_minus_chance"] = float(
                observed_value - chance_value
            )
            lifts[f"{key}_chance_ratio"] = float(
                observed_value / chance_value if chance_value else math.nan
            )
            lifts[f"{key}_minus_null_mean"] = float(
                observed_value - null_value
            )
    return lifts


def evaluate_retrieval_space(
    predicted,
    target,
    *,
    subject_ids,
    song_ids,
    window_idxs,
    section_ids,
    ks,
    gap,
    n_perms,
    n_within_song_shuffles,
    n_song_search_perms,
    song_search_permutation_batch_size,
    song_search_marginal_temperature,
    rng=0,
):
    """Run the shared retrieval benchmark for one representation space."""
    cache = build_retrieval_evaluation_cache(
        predicted,
        target,
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=gap,
    )
    observed = evaluate_candidate_pool_regimes(
        predicted,
        target,
        song_ids,
        window_idxs,
        ks,
        gap,
        cache,
    )
    chance = evaluate_candidate_pool_chance_regimes(
        song_ids,
        window_idxs,
        ks,
        gap,
    )
    nulls = evaluate_candidate_pool_null_regimes(
        predicted,
        target,
        window_idxs,
        song_ids,
        ks,
        gap,
        n_perms,
        rng,
        cache,
    )
    within_song_shuffle = evaluate_within_song_shuffle_baseline(
        predicted,
        target,
        song_ids,
        window_idxs,
        ks,
        gap,
        n_within_song_shuffles,
        rng,
        cache,
    )

    song_cache = build_song_search_evaluation_cache(
        predicted,
        target,
        song_ids,
        window_idxs,
    )
    song_rows = evaluate_song_search(
        predicted,
        target,
        song_ids,
        window_idxs,
        subject_ids=subject_ids,
        cache=song_cache,
        marginal_temperature=song_search_marginal_temperature,
    )
    song_summary = summarize_song_search_results(song_rows)
    song_null = evaluate_song_search_permutation_test(
        predicted,
        target,
        song_ids,
        window_idxs,
        n_perm=n_song_search_perms,
        rng=rng,
        permutation_batch_size=song_search_permutation_batch_size,
        cache=song_cache,
        marginal_temperature=song_search_marginal_temperature,
    )
    for metric in ("song", "marginal"):
        observed_key = f"song_search_{metric}_top1"
        permutation_key = f"{observed_key}_permutation_observed"
        if not math.isclose(
            song_summary[observed_key],
            song_null[permutation_key],
            abs_tol=1e-7,
        ):
            raise RuntimeError(
                f"Song-search cache produced inconsistent {metric} accuracy."
            )
    song_summary.update(song_null)

    section_coverage = summarize_section_coverage(section_ids)
    section = {"coverage": section_coverage}
    section_rows = []
    if section_coverage["section_eval_n_scored"] > 0:
        section["observed"] = evaluate_section_regime(
            predicted,
            target,
            song_ids,
            window_idxs,
            section_ids,
            ks,
            gap,
            cache,
        )
        section["chance"] = chance_section_regime(
            song_ids,
            window_idxs,
            section_ids,
            ks,
            gap,
        )
        section["null"] = evaluate_section_null_regime(
            predicted,
            target,
            song_ids,
            window_idxs,
            section_ids,
            ks,
            gap,
            n_perms,
            rng,
            cache,
        )
        section_rows = evaluate_section_search(
            predicted,
            target,
            song_ids,
            window_idxs,
            section_ids,
            subject_ids=subject_ids,
            gap=gap,
            cache=cache,
        )
        section["search"] = summarize_section_search_results(section_rows)
    else:
        section["skipped_reason"] = (
            "No test windows have a non-boundary section_id."
        )

    result = {
        "observed": observed,
        "chance": chance,
        "null": nulls,
        "within_song_shuffle": within_song_shuffle,
        "candidate_pool": summarize_candidate_pool_sizes(
            song_ids,
            window_idxs,
            gap=gap,
        ),
        "lift": _benchmark_lifts(observed, chance, nulls, ks),
        "song_search": song_summary,
        "section": section,
    }
    return result, song_rows, section_rows


def evaluate_nearest_mse_retrieval(
    predicted,
    target,
    song_ids,
    window_idxs,
    ks,
    gap,
):
    """Rank exact five-second mel targets from lowest pairwise MSE."""
    cache = build_retrieval_evaluation_cache(
        predicted,
        target,
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=gap,
        score_metric="negative_mse",
    )
    return {
        "distance_metric": "mean_squared_error",
        "ranking_direction": "lowest_mse_first",
        "observed": evaluate_candidate_pool_regimes(
            predicted,
            target,
            song_ids,
            window_idxs,
            ks,
            gap,
            cache,
        ),
        "chance": evaluate_candidate_pool_chance_regimes(
            song_ids,
            window_idxs,
            ks,
            gap,
        ),
        "candidate_pool": summarize_candidate_pool_sizes(
            song_ids,
            window_idxs,
            gap=gap,
        ),
    }


def _log_space(label, results):
    observed = dict(results["observed"])
    observed["loss"] = float("nan")
    log_split_metrics(label, observed)
    log_candidate_pool_metrics(label, results["candidate_pool"])
    log_chance_metrics(label, results["chance"])
    log_null_metrics(label, results["null"])
    log_within_song_shuffle_metrics(label, results["within_song_shuffle"])
    log_lift_metrics(label, results["lift"])
    log_song_search_metrics(label, results["song_search"])
    section = results["section"]
    log_section_coverage(label, section["coverage"])
    if "observed" in section:
        log_section_metrics(label, section["observed"])
        log_section_chance_metrics(label, section["chance"])
        log_section_null_metrics(label, section["null"])
        log_section_p_metrics(label, section["null"])
        log_section_search_metrics(label, section["search"])


def _save_space_details(output_dir, song_rows, song_summary, section_rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_song_search_results_csv(song_rows, output_dir / "song_search_results.csv")
    save_song_search_summary_csv(
        song_summary,
        output_dir / "song_search_summary.csv",
    )
    if section_rows:
        save_section_search_results_csv(
            section_rows,
            output_dir / "section_search_results.csv",
        )


def _retrieval_rows(space, results, ks):
    rows = []
    for regime in RETRIEVAL_REGIMES:
        for k in ks:
            key = f"{regime}_top{k}"
            nulls = results.get("null", {})
            rows.append(
                {
                    "space": space,
                    "metric": f"{regime}_r_at_{k}",
                    "value": results["observed"][key],
                    "chance": results["chance"][f"{key}_chance"],
                    "null_mean": nulls.get(f"{key}_null_mean", ""),
                    "p_upper": nulls.get(f"{key}_p", ""),
                    "p_lower": nulls.get(f"{key}_p_lower", ""),
                    "unit": "proportion",
                }
            )
    return rows


def _headline_rows(space, results, ks, reconstruction=None):
    rows = _retrieval_rows(space, results, ks)
    song = results["song_search"]
    for metric_name, key_prefix in (
        ("song_identification_top1", "song_search_song_top1"),
        (
            "song_identification_marginal_top1",
            "song_search_marginal_top1",
        ),
    ):
        rows.append(
            {
                "space": space,
                "metric": metric_name,
                "value": song[key_prefix],
                "chance": song.get(f"{key_prefix}_chance", ""),
                "null_mean": song.get(f"{key_prefix}_null_mean", ""),
                "p_upper": song.get(f"{key_prefix}_p", ""),
                "p_lower": song.get(f"{key_prefix}_p_lower", ""),
                "unit": "proportion",
            }
        )
    rows.append(
        {
            "space": space,
            "metric": "localization_mean_error",
            "value": song["song_search_mean_localization_error"],
            "chance": "",
            "null_mean": "",
            "p_upper": "",
            "p_lower": "",
            "unit": "five_second_windows",
        }
    )

    section = results["section"]
    if "observed" in section:
        key = "within_song_section_top1"
        rows.append(
            {
                "space": space,
                "metric": "section_top1",
                "value": section["observed"][key],
                "chance": section["chance"][f"{key}_chance"],
                "null_mean": section["null"][f"{key}_null_mean"],
                "p_upper": section["null"][f"{key}_p"],
                "p_lower": section["null"][f"{key}_p_lower"],
                "unit": "proportion",
            }
        )

    for source, prefix in (
        ("within_song_audio_shuffle", "within_song_audio_shuffle"),
        ("within_song_eeg_shuffle", "within_song_prediction_shuffle"),
    ):
        for k in ks:
            rows.append(
                {
                    "space": space,
                    "metric": f"{prefix}_r_at_{k}",
                    "value": results["within_song_shuffle"][
                        f"{source}_top{k}_mean"
                    ],
                    "chance": "",
                    "null_mean": "",
                    "p_upper": results["within_song_shuffle"][
                        f"{source}_top{k}_p"
                    ],
                    "p_lower": results["within_song_shuffle"][
                        f"{source}_top{k}_p_lower"
                    ],
                    "unit": "proportion",
                }
            )

    if reconstruction is not None:
        for scope, metrics in reconstruction.items():
            for metric, unit in (
                ("mse", "normalized_mel_squared"),
                ("mae", "normalized_mel"),
                ("ssim_mean", "index"),
                ("psnr_db_mean", "decibels"),
                ("matched_cosine_mean", "cosine"),
            ):
                rows.append(
                    {
                        "space": space,
                        "metric": f"{scope}_{metric}",
                        "value": metrics[metric],
                        "chance": "",
                        "null_mean": "",
                        "p_upper": "",
                        "p_lower": "",
                        "unit": unit,
                    }
                )
    return rows


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_default_test_metrics(log_dir, n_examples, results):
    section = results["section"]
    p_values = {
        **results["null"],
        **results["within_song_shuffle"],
    }
    observed = dict(results["observed"])
    observed["loss"] = float("nan")
    return save_test_metrics_csv(
        output_path=Path(log_dir) / "test_metrics.csv",
        n_test_windows=n_examples,
        test_results=observed,
        candidate_chance_results=results["chance"],
        song_search_summary=results["song_search"],
        section_coverage=section["coverage"],
        section_regime_results=section.get("observed", {}),
        section_chance_results=section.get("chance", {}),
        section_null_results=section.get("null", {}),
        p_values=p_values,
    )


def main(cli_args=None):
    cli_args = parse_args() if cli_args is None else cli_args
    base_dir = Path(__file__).resolve().parents[2]
    config_path = Path(cli_args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    args = OmegaConf.load(config_path)
    settings = resolve_eeg2mel_settings(args)

    run_name = cli_args.run_name or str(args["run_name"])
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
        base_dir=base_dir,
        run_name=run_name,
        explicit_path=cli_args.checkpoint_path,
    )
    device = _resolve_device(cli_args.device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
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

    use_mert_space = bool(cli_args.mert_space or settings["mert_space_enabled"])
    test_path = resolve_split_path(
        base_dir,
        run_name,
        args["testing"]["filename"],
    )
    test_dataset = EEG2MelEvalDataset(
        metadata_path=test_path,
        split="test",
        eeg_sample_rate=settings["eeg_sample_rate"],
        include_audio=use_mert_space,
    )
    if len(test_dataset) == 0:
        raise ValueError("The EEG2Mel test split is empty.")
    batch_size = (
        settings["mert_batch_size"]
        if use_mert_space
        else settings["evaluation_batch_size"]
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    logging.info(
        "Loaded %d test windows from %s; default evaluation space is mel.",
        len(test_dataset),
        test_path,
    )

    mert_extractor = None
    if use_mert_space:
        from src.encoders import MERTFeatureExtractor

        logging.info(
            "Loading frozen MERT extractor for optional reconstructed-audio "
            "evaluation."
        )
        mert_extractor = MERTFeatureExtractor(
            model_name=settings["mert_model_name"]
        ).to(device).eval()

    predicted_mel_stacks = []
    target_mel_stacks = []
    predicted_mert = []
    target_mert = []
    subject_ids = []
    song_ids = []
    window_idxs = []
    section_ids = []
    for batch in test_dataloader:
        batch_predicted_stack, _ = predict_mel_windows(
            model,
            batch["eeg_psd_stack"],
            device,
        )
        predicted_mel_stacks.append(batch_predicted_stack.cpu())
        target_mel_stacks.append(batch["mel_target_stack"].cpu())
        if use_mert_space:
            predicted_mert.append(
                embed_predicted_mels(
                    model,
                    batch_predicted_stack,
                    mert_extractor,
                )
            )
            target_mert.append(batch["audio"].cpu())
        subject_ids.extend(int(value) for value in batch["subject_id"])
        song_ids.extend(int(value) for value in batch["song_id"])
        window_idxs.extend(int(value) for value in batch["window_idx"])
        section_ids.extend(int(value) for value in batch["section_id"])

    predicted_mel_stack = torch.cat(predicted_mel_stacks)
    target_mel_stack = torch.cat(target_mel_stacks)
    predicted_one_second = predicted_mel_stack.flatten(0, 1)
    target_one_second = target_mel_stack.flatten(0, 1)
    predicted_five_second = concatenate_sub_window_mels(predicted_mel_stack)
    target_five_second = concatenate_sub_window_mels(target_mel_stack)
    reconstruction = {
        "one_second": reconstruction_metrics(
            predicted_one_second,
            target_one_second,
            data_range=2.0,
        ),
        "five_second": reconstruction_metrics(
            predicted_five_second,
            target_five_second,
            data_range=2.0,
        ),
    }

    if settings["save_representations"]:
        torch.save(
            {
                "schema_version": 1,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "split_path": str(test_path.resolve()),
                "predicted_mel_stack": predicted_mel_stack,
                "target_mel_stack": target_mel_stack,
                "predicted_mel": predicted_five_second,
                "target_mel": target_five_second,
                "subject_ids": subject_ids,
                "song_ids": song_ids,
                "window_idxs": window_idxs,
                "section_ids": section_ids,
            },
            benchmark_dir / "eeg2mel_mel_representations.pt",
        )

    testing_cfg = args.get("testing", {})
    n_perms = (
        cli_args.n_perms
        if cli_args.n_perms is not None
        else settings["n_perms"]
    )
    n_shuffles = (
        cli_args.n_within_song_shuffles
        if cli_args.n_within_song_shuffles is not None
        else settings["n_within_song_shuffles"]
    )
    n_song_perms = int(testing_cfg.get("song_search_n_perms", n_perms))
    song_permutation_batch_size = int(
        testing_cfg.get("song_search_permutation_batch_size", 16)
    )
    marginal_temperature = float(
        testing_cfg.get("song_search_marginal_temperature", 0.07)
    )
    gap = int(args.get("retrieval", {}).get("candidate_gap_size", 0))
    seed = int(args.get("training", {}).get("seed", 0))
    common_arguments = {
        "subject_ids": subject_ids,
        "song_ids": song_ids,
        "window_idxs": window_idxs,
        "section_ids": section_ids,
        "ks": settings["ks"],
        "gap": gap,
        "n_perms": int(n_perms),
        "n_within_song_shuffles": int(n_shuffles),
        "n_song_search_perms": n_song_perms,
        "song_search_permutation_batch_size": song_permutation_batch_size,
        "song_search_marginal_temperature": marginal_temperature,
        "rng": seed,
    }

    logging.info("Running default mel-space retrieval and controls.")
    mel_results, mel_song_rows, mel_section_rows = evaluate_retrieval_space(
        predicted_five_second,
        target_five_second,
        **common_arguments,
    )
    _log_space("EEG2Mel mel", mel_results)
    _save_space_details(
        benchmark_dir / "mel_space",
        mel_song_rows,
        mel_results["song_search"],
        mel_section_rows,
    )
    mel_nearest_mse = evaluate_nearest_mse_retrieval(
        predicted_five_second,
        target_five_second,
        song_ids,
        window_idxs,
        settings["ks"],
        gap,
    )

    result_payload = {
        "model_label": "eeg2mel_baseline",
        "source_type": "eeg2mel_gradient_regression",
        "default_evaluation_space": "mel_space",
        "split_filename": str(args["testing"]["filename"]),
        "split_name": "test",
        "ks": list(settings["ks"]),
        "candidate_gap_size": gap,
        "n_perms": int(n_perms),
        "n_within_song_shuffles": int(n_shuffles),
        "num_examples": len(test_dataset),
        "num_one_second_examples": len(test_dataset) * predicted_mel_stack.shape[1],
        "audio_pooling": settings["audio_pooling"],
        "checkpoint_path": str(checkpoint_path),
        "comparison_note": (
            "Mel-space reconstruction and retrieval are the default EEG2Mel "
            "evaluation. MERT-space scores are an optional reconstructed-audio "
            "comparison."
        ),
        "reconstruction_metrics": reconstruction,
        "mel_space": mel_results,
        "mel_space_nearest_mse": mel_nearest_mse,
        # Backward-compatible aliases now point to the default mel space.
        "results": mel_results,
        "song_search": mel_results["song_search"],
        "p_values": {
            **mel_results["null"],
            **mel_results["within_song_shuffle"],
        },
    }
    headline_rows = _headline_rows(
        "mel_space",
        mel_results,
        settings["ks"],
        reconstruction,
    )
    headline_rows.extend(
        _retrieval_rows(
            "mel_space_nearest_mse",
            mel_nearest_mse,
            settings["ks"],
        )
    )

    if use_mert_space:
        logging.info("Running optional reconstructed-audio MERT-space evaluation.")
        predicted_mert = torch.cat(predicted_mert)
        target_mert = torch.cat(target_mert)
        mert_results, mert_song_rows, mert_section_rows = evaluate_retrieval_space(
            predicted_mert,
            target_mert,
            **common_arguments,
        )
        _log_space("EEG2Mel MERT", mert_results)
        _save_space_details(
            benchmark_dir / "mert_space",
            mert_song_rows,
            mert_results["song_search"],
            mert_section_rows,
        )
        result_payload["mert_space"] = mert_results
        headline_rows.extend(
            _headline_rows("mert_space", mert_results, settings["ks"])
        )
    else:
        result_payload["mert_space"] = {
            "enabled": False,
            "note": (
                "Pass --mert-space or set "
                "eeg2mel_baseline.evaluation.mert_space.enabled=true to run "
                "the reconstructed-audio MERT comparison."
            ),
        }

    test_metrics_path = _save_default_test_metrics(
        log_dir,
        len(test_dataset),
        mel_results,
    )
    logging.info("Saved default mel-space metrics to %s", test_metrics_path)

    rich_metrics_path = benchmark_dir / "eeg2mel_baseline_test_metrics.csv"
    with rich_metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RICH_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(headline_rows)

    output_path = benchmark_dir / settings["output_filename"]
    output_path.write_text(json.dumps(_jsonable(result_payload), indent=2))
    logging.info(
        "Saved EEG2Mel evaluation to %s and %s",
        output_path,
        rich_metrics_path,
    )
    return result_payload


if __name__ == "__main__":
    main()
