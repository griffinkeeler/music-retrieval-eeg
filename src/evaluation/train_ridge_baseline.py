"""Train and benchmark a ridge-regression EEG-to-audio baseline."""

import argparse
import gc
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from omegaconf import OmegaConf

from src.data import EEGMusicWindowDataset
from src.config import load_config, resolve_split_path
from src.evaluation import (
    RidgeEEGToAudio,
    evaluate_embedding_benchmark,
    log_candidate_pool_metrics,
    log_chance_metrics,
    log_lift_metrics,
    log_null_metrics,
    log_split_metrics,
    log_within_song_shuffle_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a ridge-regression EEG-to-audio baseline and evaluate it "
            "with the same candidate-pool retrieval benchmark as the main model."
        )
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
        "--alpha",
        type=float,
        default=None,
        help="Optional ridge alpha override.",
    )
    parser.add_argument(
        "--n-perms",
        type=int,
        default=None,
        help="Optional number of candidate-pool permutations.",
    )
    parser.add_argument(
        "--n-within-song-shuffles",
        type=int,
        default=None,
        help="Optional number of within-song audio and EEG shuffles.",
    )
    parser.add_argument(
        "--n-song-search-perms",
        type=int,
        default=None,
        help="Optional number of song-identification permutations.",
    )
    parser.add_argument(
        "--song-search-permutation-batch-size",
        type=int,
        default=None,
        help="Optional number of song-identification permutations per batch.",
    )
    parser.add_argument(
        "--song-search-marginal-temperature",
        type=float,
        default=None,
        help="Optional softmax temperature for marginalized song identification.",
    )
    return parser.parse_args()


def _to_float32_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def pool_audio_window(audio, pooling="mean"):
    """Convert one vector or temporal audio embedding to one target vector."""
    audio = _to_float32_numpy(audio)
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise ValueError(
            "Each audio embedding must have shape [D] or [D, T], "
            f"got {audio.shape}."
        )
    if pooling == "mean":
        return audio.mean(axis=-1, dtype=np.float32)
    raise ValueError("ridge_baseline.audio_pooling must be 'mean'.")


def dataset_to_ridge_arrays(dataset, audio_pooling="mean"):
    """Load a dataset into float32 EEG windows and pooled audio targets.

    Arrays are preallocated and temporal audio embeddings are pooled one at a
    time. This avoids retaining every full MERT sequence in memory.
    """
    if len(dataset) == 0:
        raise ValueError(
            "Cannot train or evaluate the ridge baseline on an empty split."
        )

    first_sample = dataset[0]
    first_eeg = _to_float32_numpy(first_sample["eeg"])
    first_audio = pool_audio_window(first_sample["audio"], pooling=audio_pooling)
    if first_eeg.ndim < 1:
        raise ValueError("Each EEG window must have at least one feature dimension.")

    eeg = np.empty((len(dataset), *first_eeg.shape), dtype=np.float32)
    audio = np.empty((len(dataset), first_audio.shape[0]), dtype=np.float32)
    eeg[0] = first_eeg
    audio[0] = first_audio

    for idx in range(1, len(dataset)):
        sample = dataset[idx]
        eeg_window = _to_float32_numpy(sample["eeg"])
        audio_target = pool_audio_window(
            sample["audio"],
            pooling=audio_pooling,
        )
        if eeg_window.shape != first_eeg.shape:
            raise ValueError(
                "All EEG windows must have the same shape: "
                f"expected {first_eeg.shape}, got {eeg_window.shape} at index {idx}."
            )
        if audio_target.shape != first_audio.shape:
            raise ValueError(
                "All pooled audio targets must have the same shape: "
                f"expected {first_audio.shape}, got {audio_target.shape} at index {idx}."
            )
        eeg[idx] = eeg_window
        audio[idx] = audio_target

    return eeg, audio


def regression_metrics(predicted, target):
    """Return scalar regression diagnostics for predicted audio vectors."""
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError(
            "Predicted and target audio embeddings must have matching [N, D] shapes."
        )

    residual = predicted - target
    mse = float(np.mean(np.square(residual), dtype=np.float64))
    target_centered = target - target.mean(axis=0, keepdims=True)
    total_variation = float(
        np.sum(np.square(target_centered), dtype=np.float64)
    )
    r2 = float(
        1.0
        - np.sum(np.square(residual), dtype=np.float64) / total_variation
        if total_variation > 0
        else np.nan
    )

    numerator = np.sum(predicted * target, axis=1, dtype=np.float64)
    denominator = np.linalg.norm(predicted, axis=1) * np.linalg.norm(target, axis=1)
    cosine = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return {
        "mse": mse,
        "r2_variance_weighted": r2,
        "matched_cosine_mean": float(cosine.mean()),
    }


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


def _ridge_settings(args, cli_args):
    ridge_args = args.get("ridge_baseline", {}) or {}
    testing_args = args["testing"]
    cli_n_song_search_perms = getattr(cli_args, "n_song_search_perms", None)
    cli_song_search_batch_size = getattr(
        cli_args,
        "song_search_permutation_batch_size",
        None,
    )
    cli_song_search_marginal_temperature = getattr(
        cli_args,
        "song_search_marginal_temperature",
        None,
    )
    alpha = (
        cli_args.alpha
        if cli_args.alpha is not None
        else ridge_args.get("alpha", 1.0)
    )
    n_perms = (
        cli_args.n_perms
        if cli_args.n_perms is not None
        else ridge_args.get("n_perms", testing_args["n_perms"])
    )
    n_within_song_shuffles = (
        cli_args.n_within_song_shuffles
        if cli_args.n_within_song_shuffles is not None
        else ridge_args.get(
            "n_within_song_shuffles",
            testing_args.get("n_within_song_shuffles", 200),
        )
    )
    n_song_search_perms = (
        cli_n_song_search_perms
        if cli_n_song_search_perms is not None
        else ridge_args.get(
            "n_song_search_perms",
            testing_args.get("song_search_n_perms", n_perms),
        )
    )
    song_search_permutation_batch_size = (
        cli_song_search_batch_size
        if cli_song_search_batch_size is not None
        else ridge_args.get(
            "song_search_permutation_batch_size",
            testing_args.get("song_search_permutation_batch_size", 16),
        )
    )
    song_search_marginal_temperature = (
        cli_song_search_marginal_temperature
        if cli_song_search_marginal_temperature is not None
        else ridge_args.get(
            "song_search_marginal_temperature",
            testing_args.get("song_search_marginal_temperature", 0.07),
        )
    )
    settings = {
        "alpha": float(alpha),
        "solver": str(ridge_args.get("solver", "auto")),
        "audio_pooling": str(ridge_args.get("audio_pooling", "mean")),
        "ks": tuple(int(k) for k in ridge_args.get("ks", [1, 5, 10])),
        "n_perms": int(n_perms),
        "n_within_song_shuffles": int(n_within_song_shuffles),
        "n_song_search_perms": int(n_song_search_perms),
        "song_search_permutation_batch_size": int(
            song_search_permutation_batch_size
        ),
        "song_search_marginal_temperature": float(
            song_search_marginal_temperature
        ),
        "output_filename": str(
            ridge_args.get("output_filename", "ridge_baseline_test.json")
        ),
    }
    if settings["alpha"] < 0:
        raise ValueError("ridge_baseline.alpha must be non-negative.")
    if not settings["ks"] or any(k < 1 for k in settings["ks"]):
        raise ValueError("ridge_baseline.ks must contain positive integers.")
    if settings["n_perms"] < 1:
        raise ValueError("ridge_baseline.n_perms must be at least 1.")
    if settings["n_within_song_shuffles"] < 1:
        raise ValueError(
            "ridge_baseline.n_within_song_shuffles must be at least 1."
        )
    if settings["n_song_search_perms"] < 1:
        raise ValueError("ridge_baseline.n_song_search_perms must be at least 1.")
    if settings["song_search_permutation_batch_size"] < 1:
        raise ValueError(
            "ridge_baseline.song_search_permutation_batch_size must be at least 1."
        )
    if (
        not np.isfinite(settings["song_search_marginal_temperature"])
        or settings["song_search_marginal_temperature"] <= 0
    ):
        raise ValueError(
            "ridge_baseline.song_search_marginal_temperature must be finite "
            "and positive."
        )
    if settings["audio_pooling"] != "mean":
        raise ValueError("ridge_baseline.audio_pooling must be 'mean'.")
    return settings


def main(cli_args=None):
    cli_args = cli_args or parse_args()
    base_dir = Path(__file__).parents[2]
    config_path = Path(cli_args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    args = load_config(config_path)

    run_name = cli_args.run_name or args["run_name"]
    settings = _ridge_settings(args, cli_args)
    log_dir = base_dir / "runs" / run_name
    benchmark_dir = log_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "ridge_baseline.log"),
            logging.StreamHandler(),
        ],
    )
    OmegaConf.save(args, benchmark_dir / "ridge_baseline_config.yaml")

    train_path = resolve_split_path(
        base_dir,
        args,
        args["training"]["filename"],
    )
    test_path = resolve_split_path(
        base_dir,
        args,
        args["testing"]["filename"],
    )
    train_dataset = EEGMusicWindowDataset(metadata_path=train_path, split="train")
    test_dataset = EEGMusicWindowDataset(metadata_path=test_path, split="test")
    logging.info(
        "Loaded %d training windows from %s and %d test windows from %s.",
        len(train_dataset),
        train_path,
        len(test_dataset),
        test_path,
    )

    logging.info(
        "Loading training arrays and mean-pooling temporal audio embeddings."
    )
    eeg_train, audio_train = dataset_to_ridge_arrays(
        train_dataset,
        audio_pooling=settings["audio_pooling"],
    )
    logging.info(
        "Fitting ridge baseline with alpha=%g, solver=%s, EEG shape=%s, target shape=%s.",
        settings["alpha"],
        settings["solver"],
        tuple(eeg_train.shape),
        tuple(audio_train.shape),
    )
    ridge_baseline = RidgeEEGToAudio(
        alpha=settings["alpha"],
        solver=settings["solver"],
    ).fit(eeg_train, audio_train)

    checkpoint_dir = base_dir / "runs" / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "ridge_baseline.joblib"
    joblib.dump(
        {
            "model": ridge_baseline,
            "audio_pooling": settings["audio_pooling"],
            "train_split_path": str(train_path),
            "config": OmegaConf.to_container(args, resolve=True),
        },
        checkpoint_path,
    )
    logging.info("Saved ridge baseline to %s", checkpoint_path)

    # The fitted scaler/model retain only learned statistics and coefficients.
    # Release the large raw training arrays before loading the test split.
    del eeg_train, audio_train
    gc.collect()

    logging.info("Loading test arrays.")
    eeg_test, audio_test = dataset_to_ridge_arrays(
        test_dataset,
        audio_pooling=settings["audio_pooling"],
    )
    predicted_audio = ridge_baseline.predict(eeg_test)
    diagnostics = regression_metrics(predicted_audio, audio_test)
    del eeg_test
    gc.collect()

    song_ids = test_dataset.metadata["song_id"].astype(int).tolist()
    window_idxs = test_dataset.metadata["window_idx"].astype(int).tolist()
    candidate_gap_size = int(args["retrieval"]["candidate_gap_size"])
    benchmark = evaluate_embedding_benchmark(
        eeg_emb=predicted_audio,
        audio_emb=audio_test,
        song_ids=song_ids,
        window_idxs=window_idxs,
        ks=settings["ks"],
        gap=candidate_gap_size,
        n_perm=settings["n_perms"],
        n_within_song_shuffles=settings["n_within_song_shuffles"],
        n_song_search_perms=settings["n_song_search_perms"],
        song_search_permutation_batch_size=settings[
            "song_search_permutation_batch_size"
        ],
        song_search_marginal_temperature=settings[
            "song_search_marginal_temperature"
        ],
        rng=0,
    )

    observed = dict(benchmark["observed"])
    observed["loss"] = diagnostics["mse"]
    logging.info(
        "Ridge regression diagnostics: MSE=%.6g R2=%.4f matched-cosine=%.4f",
        diagnostics["mse"],
        diagnostics["r2_variance_weighted"],
        diagnostics["matched_cosine_mean"],
    )
    log_split_metrics("Ridge", observed)
    log_candidate_pool_metrics("Ridge", benchmark["candidate_pool"])
    log_chance_metrics("Ridge", benchmark["chance"])
    log_null_metrics("Ridge", benchmark["null"])
    log_within_song_shuffle_metrics("Ridge", benchmark["within_song_shuffle"])
    log_lift_metrics("Ridge", benchmark["lift"])
    song_search = benchmark["song_search"]
    logging.info(
        "Ridge song identification: top1=%.4f chance=%.4f null=%.4f "
        "p_upper=%.6f p_lower=%.6f",
        song_search["song_search_song_top1"],
        song_search["song_search_song_top1_chance"],
        song_search["song_search_song_top1_null_mean"],
        song_search["song_search_song_top1_p"],
        song_search["song_search_song_top1_p_lower"],
    )
    logging.info(
        "Ridge marginal song identification: top1=%.4f chance=%.4f null=%.4f "
        "p_upper=%.6f p_lower=%.6f temperature=%.6g",
        song_search["song_search_marginal_top1"],
        song_search["song_search_marginal_top1_chance"],
        song_search["song_search_marginal_top1_null_mean"],
        song_search["song_search_marginal_top1_p"],
        song_search["song_search_marginal_top1_p_lower"],
        song_search["song_search_marginal_temperature"],
    )

    result_payload = {
        "model_label": "ridge_baseline",
        "source_type": "ridge_regression",
        "split_filename": str(args["testing"]["filename"]),
        "split_name": "test",
        "ks": list(settings["ks"]),
        "candidate_gap_size": candidate_gap_size,
        "n_perms": settings["n_perms"],
        "n_within_song_shuffles": settings["n_within_song_shuffles"],
        "n_song_search_perms": settings["n_song_search_perms"],
        "song_search_permutation_batch_size": settings[
            "song_search_permutation_batch_size"
        ],
        "song_search_marginal_temperature": settings[
            "song_search_marginal_temperature"
        ],
        "num_train_examples": len(train_dataset),
        "num_examples": len(test_dataset),
        "alpha": settings["alpha"],
        "solver": settings["solver"],
        "audio_pooling": settings["audio_pooling"],
        "checkpoint_path": str(checkpoint_path),
        "regression_metrics": diagnostics,
        "results": benchmark,
    }
    output_path = benchmark_dir / settings["output_filename"]
    output_path.write_text(json.dumps(_jsonable(result_payload), indent=2))
    logging.info("Saved ridge benchmark results to %s", output_path)
    return result_payload


if __name__ == "__main__":
    main()
