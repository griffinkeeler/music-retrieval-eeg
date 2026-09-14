import argparse
import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data import AcrossSongBatchSampler, EEGMusicWindowDataset, WithinSongBatchSampler
from src.config import load_config, resolve_split_path
from src.encoders import EEGEncoder
from src.evaluation import evaluate_epoch, log_split_metrics
from src.model import (
    build_alignment_objective,
    build_audio_projection,
    resolve_alignment_objective_name,
    resolve_alignment_settings,
    run_loss_epoch,
)


def get_early_stopping_monitor_value(val_metrics, monitor_name):
    if monitor_name == "val_loss":
        return val_metrics["loss"]
    if monitor_name in val_metrics:
        return val_metrics[monitor_name]
    raise ValueError(
        f"Unsupported early stopping monitor '{monitor_name}'. "
        "Use 'val_loss' or a validation metric returned by evaluate_epoch()."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the EEG-to-music alignment model."
    )
    parser.add_argument(
        "--config",
        default="configs/chunk_out.yaml",
        help="Path to a paper experiment config.",
    )
    return parser.parse_args()


def seed_training(seed):
    """Seed training RNGs so matched runs start from the same randomness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_dataloader(train_dataset, batch_size, sampler_regime, seed=0):
    if sampler_regime == "within_song":
        sampler = WithinSongBatchSampler(
            train_dataset.metadata,
            batch_size=batch_size,
            seed=seed,
        )
    elif sampler_regime == "across_song":
        sampler = AcrossSongBatchSampler(
            train_dataset.metadata,
            batch_size=batch_size,
            seed=seed,
        )
    else:
        raise ValueError(
            "sampler regime must be one of ['within_song', 'across_song']"
        )

    return DataLoader(
        train_dataset,
        batch_sampler=sampler,
    )


def metric_improved(current_value, best_value, mode):
    if best_value is None:
        return True
    if mode == "max":
        return current_value > best_value
    if mode == "min":
        return current_value < best_value
    raise ValueError("switch_mode must be one of ['max', 'min']")


def save_checkpoint(
    checkpoint_path,
    epoch,
    eeg_encoder,
    optimizer,
    objective,
    audio_projection,
    args,
):
    objective_state_dict = objective.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "eeg_encoder_state_dict": eeg_encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "objective_name": resolve_alignment_objective_name(args),
            "objective_state_dict": objective_state_dict,
            # Keep the legacy key so older checkpoint consumers can still read
            # newly trained InfoNCE models.
            "clip_state_dict": objective_state_dict,
            "audio_projection_state_dict": audio_projection.state_dict(),
            "config": OmegaConf.to_container(args, resolve=True),
        },
        checkpoint_path,
    )

if __name__ == "__main__":
    # ---------- Logging Setup --------
    # The base directory file path
    base_dir = PROJECT_ROOT
    cli_args = parse_args()

    config_path = Path(cli_args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    args = load_config(config_path)
    training_seed = int(args["training"].get("seed", 0))
    seed_training(training_seed)

    run_name = args["run_name"]
    log_dir = base_dir / "runs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(args, log_dir / "config.yaml")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )

    # Setup TensorBoard writer
    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
    logging.info("Training seed: %d", training_seed)

    train_filename = args["training"]["filename"]
    train_filepath = resolve_split_path(base_dir, args, train_filename)

    train_dataset = EEGMusicWindowDataset(metadata_path=train_filepath, split="train")
    validation_enabled = bool(args["training"].get("validation", True))
    evaluate_train_split = bool(args["training"].get("evaluate_train_split", False))

    batch_size = args["training"]["batch_size"]
    default_sampler_regime = args["retrieval"]["regime"]
    candidate_gap_size = args["retrieval"]["candidate_gap_size"]
    train_dataloaders = {
        "across_song": build_train_dataloader(
            train_dataset=train_dataset,
            batch_size=batch_size,
            sampler_regime="across_song",
            seed=training_seed,
        ),
        "within_song": build_train_dataloader(
            train_dataset=train_dataset,
            batch_size=batch_size,
            sampler_regime="within_song",
            seed=training_seed,
        ),
    }

    sampler_schedule = args["training"].get("sampler_schedule")
    schedule_enabled = bool(sampler_schedule and sampler_schedule.get("enabled", False))
    if schedule_enabled and not validation_enabled:
        raise ValueError("training.sampler_schedule requires training.validation=True")

    if schedule_enabled:
        phase1_regime = str(sampler_schedule["phase1_regime"])
        phase2_regime = str(sampler_schedule["phase2_regime"])
        switch_metric = str(sampler_schedule["switch_metric"])
        switch_mode = str(sampler_schedule.get("switch_mode", "max"))
        switch_patience = int(sampler_schedule.get("patience", 5))
        min_epochs_before_switch = int(
            sampler_schedule.get("min_epochs_before_switch", 10)
        )
        max_phase1_epochs = sampler_schedule.get("max_phase1_epochs")
        if max_phase1_epochs is not None:
            max_phase1_epochs = int(max_phase1_epochs)

        if phase1_regime not in train_dataloaders or phase2_regime not in train_dataloaders:
            raise ValueError(
                "training.sampler_schedule phase regimes must be in "
                "['within_song', 'across_song']"
            )
        if phase1_regime == phase2_regime:
            raise ValueError(
                "training.sampler_schedule phase1_regime and phase2_regime must differ"
            )
        if max_phase1_epochs is not None and max_phase1_epochs < 1:
            raise ValueError("training.sampler_schedule.max_phase1_epochs must be >= 1")

        active_sampler_regime = phase1_regime
        best_switch_metric = None
        epochs_without_improvement = 0
        phase_switched = False
    else:
        active_sampler_regime = default_sampler_regime
        phase1_regime = None
        phase2_regime = None
        switch_metric = None
        switch_mode = None
        switch_patience = None
        min_epochs_before_switch = None
        max_phase1_epochs = None
        best_switch_metric = None
        epochs_without_improvement = 0
        phase_switched = False

    train_dataloader = train_dataloaders[active_sampler_regime]

    if validation_enabled:
        val_dataset = EEGMusicWindowDataset(metadata_path=train_filepath, split="val")
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        logging.info(
            "Loaded %s training windows across %s batches and %s validation"
            " windows across %s batches.",
            len(train_dataset),
            len(train_dataloaders[active_sampler_regime]),
            len(val_dataset),
            len(val_dataloader),
        )
    else:
        val_dataloader = None
        logging.info(
            "Loaded %s training windows across %s batches. Validation disabled.",
            len(train_dataset),
            len(train_dataloader),
        )

    logging.info(
        "Training File: %s | Regime: %s | Temporal Gap Size: %s",
        train_filename,
        active_sampler_regime,
        candidate_gap_size,
    )
    if schedule_enabled:
        logging.info(
            "Sampler schedule enabled: %s -> %s | switch metric=%s mode=%s "
            "patience=%d min_epochs_before_switch=%d max_phase1_epochs=%s",
            phase1_regime,
            phase2_regime,
            switch_metric,
            switch_mode,
            switch_patience,
            min_epochs_before_switch,
            (
                str(max_phase1_epochs)
                if max_phase1_epochs is not None
                else "none"
            ),
        )
    logging.info("Loaded config from %s", config_path)

    # --------- Alignment Training ---------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alignment_settings = resolve_alignment_settings(args)
    objective_name = resolve_alignment_objective_name(args)

    eeg_encoder = EEGEncoder(
        embedding_dim=alignment_settings["eeg_embedding_dim"],
        use_subject_layer=args["ablations"]["use_subject_layer"],
        use_projection_head=alignment_settings["use_projection_head"],
    ).to(device)
    objective = build_alignment_objective(args).to(device)

    audio_projection = build_audio_projection(
        768,
        alignment_settings["audio_out_features"],
        mode=alignment_settings["audio_projection_mode"],
    ).to(device)
    freeze_audio_projection = bool(
        args.get("ablations", {}).get("freeze_audio_projection", False)
    )
    if freeze_audio_projection:
        for param in audio_projection.parameters():
            param.requires_grad = False

    optimizer_params = (
        list(eeg_encoder.parameters())
        + list(objective.parameters())
        + [param for param in audio_projection.parameters() if param.requires_grad]
    )
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=args["training"]["learning_rate"],
    )

    logging.info(
        "Alignment mode: %s | EEG embedding dim: %s | Audio out dim: %s",
        "direct_768" if alignment_settings["use_direct_alignment"] else "projected_512",
        alignment_settings["eeg_embedding_dim"],
        alignment_settings["audio_out_features"],
    )
    logging.info("Training objective: %s", objective_name)
    audio_projection_trainable = any(
        param.requires_grad for param in audio_projection.parameters()
    )
    logging.info("Audio projection trainable: %s", audio_projection_trainable)

    num_training_epochs = args["training"]["num_training_epochs"]
    checkpoint_dir = base_dir / "runs" / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    periodic_checkpoint_dir = checkpoint_dir / "by_epoch"
    periodic_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = checkpoint_dir / "eeg_encoder.pt"
    best_checkpoint_path = checkpoint_dir / "best.pt"
    best_val_loss = float("inf")
    best_epoch = None
    periodic_checkpoint_every = int(args["training"].get("checkpoint_every", 0) or 0)
    early_stopping_cfg = args["training"].get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False))
    early_stopping_patience = int(early_stopping_cfg.get("patience", 0) or 0)
    early_stopping_min_delta = float(early_stopping_cfg.get("min_delta", 0.0) or 0.0)
    early_stopping_monitor = str(early_stopping_cfg.get("monitor", "val_loss"))
    early_stopping_mode = str(early_stopping_cfg.get("mode", "min"))
    if early_stopping_mode not in {"min", "max"}:
        raise ValueError("early_stopping.mode must be one of ['min', 'max']")
    if early_stopping_enabled and not validation_enabled:
        raise ValueError("Early stopping requires training.validation to be True.")
    early_stopping_best = None
    early_stopping_bad_epochs = 0

    for epoch in range(num_training_epochs):
        train_dataloader = train_dataloaders[active_sampler_regime]

        train_loss = run_loss_epoch(
            dataloader=train_dataloader,
            eeg_encoder=eeg_encoder,
            audio_projection=audio_projection,
            objective=objective,
            device=device,
            optimizer=optimizer,
        )

        logging.info(
            "Epoch %d/%d | sampler=%s",
            epoch + 1,
            num_training_epochs,
            active_sampler_regime,
        )
        writer.add_scalar("TrainOptimize/Loss", train_loss["loss"], epoch)
        writer.add_text("TrainSampler/ActiveRegime", active_sampler_regime, epoch)
        logging.info("  %-5s optimize | loss=%.4f", "Train", train_loss["loss"])

        if evaluate_train_split:
            train_metrics = evaluate_epoch(
                dataloader=train_dataloader,
                eeg_encoder=eeg_encoder,
                audio_projection=audio_projection,
                clip=objective,
                device=device,
                gap=candidate_gap_size,
            )
            log_split_metrics("Train", train_metrics)

            for metric, score in train_metrics.items():
                writer.add_scalar(f"TrainEval/{metric}", score, epoch)
        if validation_enabled:
            val_metrics = evaluate_epoch(
                dataloader=val_dataloader,
                eeg_encoder=eeg_encoder,
                audio_projection=audio_projection,
                clip=objective,
                device=device,
                gap=candidate_gap_size,
            )
            log_split_metrics("Val", val_metrics)
            for metric, score in val_metrics.items():
                writer.add_scalar(f"ValEval/{metric}", score, epoch)

            if schedule_enabled and not phase_switched:
                if switch_metric not in val_metrics:
                    raise KeyError(
                        f"Switch metric '{switch_metric}' not found in validation metrics"
                    )

                current_switch_metric = float(val_metrics[switch_metric])
                if metric_improved(
                    current_value=current_switch_metric,
                    best_value=best_switch_metric,
                    mode=switch_mode,
                ):
                    best_switch_metric = current_switch_metric
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                writer.add_scalar(
                    f"SamplerSchedule/{switch_metric}",
                    current_switch_metric,
                    epoch,
                )
                writer.add_scalar(
                    "SamplerSchedule/EpochsWithoutImprovement",
                    epochs_without_improvement,
                    epoch,
                )

                should_switch_on_plateau = (
                    epoch + 1 >= min_epochs_before_switch
                    and epochs_without_improvement >= switch_patience
                )
                should_switch_on_max_phase1 = (
                    max_phase1_epochs is not None
                    and epoch + 1 >= max_phase1_epochs
                )

                if should_switch_on_plateau or should_switch_on_max_phase1:
                    active_sampler_regime = phase2_regime
                    phase_switched = True
                    if should_switch_on_plateau:
                        logging.info(
                            "Switching sampler regime from %s to %s at epoch %d "
                            "after %d epochs without %s improvement. Best %s=%.4f",
                            phase1_regime,
                            phase2_regime,
                            epoch + 1,
                            epochs_without_improvement,
                            switch_metric,
                            switch_metric,
                            best_switch_metric,
                        )
                    else:
                        logging.info(
                            "Switching sampler regime from %s to %s at epoch %d "
                            "after reaching max_phase1_epochs=%d.",
                            phase1_regime,
                            phase2_regime,
                            epoch + 1,
                            max_phase1_epochs,
                        )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch + 1
                save_checkpoint(
                    checkpoint_path=best_checkpoint_path,
                    epoch=best_epoch,
                    eeg_encoder=eeg_encoder,
                    optimizer=optimizer,
                    objective=objective,
                    audio_projection=audio_projection,
                    args=args,
                )
                logging.info(
                    "Saved new best checkpoint at epoch %d with val loss %.4f "
                    "to %s",
                    best_epoch,
                    best_val_loss,
                    best_checkpoint_path,
                )

            if early_stopping_enabled:
                monitor_value = get_early_stopping_monitor_value(
                    val_metrics, early_stopping_monitor
                )
                if early_stopping_best is None:
                    improved = True
                elif early_stopping_mode == "min":
                    improved = monitor_value < (
                        early_stopping_best - early_stopping_min_delta
                    )
                else:
                    improved = monitor_value > (
                        early_stopping_best + early_stopping_min_delta
                    )

                if improved:
                    early_stopping_best = monitor_value
                    early_stopping_bad_epochs = 0
                    logging.info(
                        "Early stopping monitor improved: %s=%.4f",
                        early_stopping_monitor,
                        monitor_value,
                    )
                else:
                    early_stopping_bad_epochs += 1
                    logging.info(
                        "Early stopping patience: %d/%d without sufficient %s improvement.",
                        early_stopping_bad_epochs,
                        early_stopping_patience,
                        early_stopping_monitor,
                    )

        if periodic_checkpoint_every > 0 and (epoch + 1) % periodic_checkpoint_every == 0:
            periodic_checkpoint_path = (
                periodic_checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
            )
            save_checkpoint(
                checkpoint_path=periodic_checkpoint_path,
                epoch=epoch + 1,
                eeg_encoder=eeg_encoder,
                optimizer=optimizer,
                objective=objective,
                audio_projection=audio_projection,
                args=args,
            )
            logging.info(
                "Saved periodic checkpoint at epoch %d to %s",
                epoch + 1,
                periodic_checkpoint_path,
            )

        if (
            early_stopping_enabled
            and early_stopping_bad_epochs >= early_stopping_patience
        ):
            logging.info(
                "Stopping early at epoch %d after %d epochs without sufficient %s improvement.",
                epoch + 1,
                early_stopping_bad_epochs,
                early_stopping_monitor,
            )
            break

    writer.close()

    # --------- EEG Encoder Checkpoints ---------
    save_checkpoint(
        checkpoint_path=last_checkpoint_path,
        epoch=num_training_epochs,
        eeg_encoder=eeg_encoder,
        optimizer=optimizer,
        objective=objective,
        audio_projection=audio_projection,
        args=args,
    )
    logging.info("Saved final checkpoint to %s", last_checkpoint_path)

    if validation_enabled and best_epoch is not None:
        logging.info(
            "Best validation checkpoint: epoch %d | val loss %.4f | path %s",
            best_epoch,
            best_val_loss,
            best_checkpoint_path,
        )
