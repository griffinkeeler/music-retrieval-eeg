"""Train the EEG2Mel baseline: gradient-based MSE regression to mel-spectrograms.

Mirrors src/model/train.py's epoch loop, checkpointing, and early-stopping
structure, but trains a much simpler EEG2MelBaseline (see
src/evaluation/eeg2mel_baseline.py) via plain MSE regression rather than a
contrastive objective. Reads training.filename/testing.filename the same way
the main model does, so it works unmodified against every
existing split type (and their round-robin fold variants) via the same
SLURM-array fold-looping convention used elsewhere in this repo.
"""

import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.evaluation.eeg2mel_baseline import (
    EEG2MelBaseline,
    EEG2MelSubWindowDataset,
    resolve_eeg2mel_settings,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the EEG2Mel baseline (EEG PSD -> mel-spectrogram regression)."
    )
    parser.add_argument(
        "--config",
        default="configs/eeg2mel_paper.yaml",
        help="Path to the standalone paper EEG2Mel config.",
    )
    return parser.parse_args()


def resolve_split_path(base_dir, run_name, split_filename):
    """Resolve an absolute split path or a filename within a run's split folder."""
    split_path = Path(split_filename)
    if split_path.is_absolute():
        return split_path
    return base_dir / "runs" / run_name / "splits" / split_path


def save_checkpoint(checkpoint_path, epoch, model, optimizer, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(args, resolve=True),
        },
        checkpoint_path,
    )


def run_epoch(dataloader, model, device, optimizer=None):
    """Run one epoch of MSE regression. Trains if optimizer is given, else evaluates."""
    is_training = optimizer is not None
    model.train(is_training)
    loss_fn = torch.nn.MSELoss()

    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        eeg_psd = batch["eeg_psd"].to(device)
        mel_target = batch["mel_target"].to(device)

        with torch.set_grad_enabled(is_training):
            predicted_mel = model(eeg_psd)
            loss = loss_fn(predicted_mel, mel_target)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {"loss": total_loss / max(num_batches, 1)}


if __name__ == "__main__":
    base_dir = Path(__file__).parents[2]
    cli_args = parse_args()

    config_path = Path(cli_args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    args = OmegaConf.load(config_path)
    settings = resolve_eeg2mel_settings(args)

    run_name = args["run_name"]
    log_dir = base_dir / "runs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "eeg2mel_baseline_train.log"),
            logging.StreamHandler(),
        ],
    )

    train_filepath = resolve_split_path(base_dir, run_name, args["training"]["filename"])
    train_dataset = EEG2MelSubWindowDataset(
        metadata_path=train_filepath, split="train", eeg_sample_rate=settings["eeg_sample_rate"],
    )
    validation_enabled = bool(args["training"].get("validation", True))

    train_dataloader = DataLoader(
        train_dataset, batch_size=settings["batch_size"], shuffle=True,
    )

    if validation_enabled:
        val_dataset = EEG2MelSubWindowDataset(
            metadata_path=train_filepath, split="val", eeg_sample_rate=settings["eeg_sample_rate"],
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=settings["batch_size"], shuffle=False,
        )
        logging.info(
            "Loaded %d training sub-windows across %d batches and %d validation "
            "sub-windows across %d batches.",
            len(train_dataset), len(train_dataloader), len(val_dataset), len(val_dataloader),
        )
    else:
        val_dataloader = None
        logging.info(
            "Loaded %d training sub-windows across %d batches. Validation disabled.",
            len(train_dataset), len(train_dataloader),
        )
    logging.info("Training file: %s | Loaded config from %s", train_filepath, config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])

    checkpoint_dir = base_dir / "runs" / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    periodic_checkpoint_dir = checkpoint_dir / "by_epoch"
    periodic_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = checkpoint_dir / "eeg2mel.pt"
    best_checkpoint_path = checkpoint_dir / "best.pt"

    best_val_loss = float("inf")
    best_epoch = None
    early_stopping_best = None
    early_stopping_bad_epochs = 0
    if settings["early_stopping_enabled"] and not validation_enabled:
        raise ValueError("eeg2mel_baseline early stopping requires training.validation to be True.")

    for epoch in range(settings["num_training_epochs"]):
        train_metrics = run_epoch(train_dataloader, model, device, optimizer=optimizer)
        logging.info(
            "Epoch %d/%d | train loss=%.6f",
            epoch + 1, settings["num_training_epochs"], train_metrics["loss"],
        )

        if validation_enabled:
            val_metrics = run_epoch(val_dataloader, model, device, optimizer=None)
            logging.info("  Val loss=%.6f", val_metrics["loss"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch + 1
                save_checkpoint(best_checkpoint_path, best_epoch, model, optimizer, args)
                logging.info(
                    "Saved new best checkpoint at epoch %d with val loss %.6f to %s",
                    best_epoch, best_val_loss, best_checkpoint_path,
                )

            if settings["early_stopping_enabled"]:
                monitor_value = val_metrics["loss"]
                if early_stopping_best is None:
                    improved = True
                elif settings["early_stopping_mode"] == "min":
                    improved = monitor_value < (early_stopping_best - settings["early_stopping_min_delta"])
                else:
                    improved = monitor_value > (early_stopping_best + settings["early_stopping_min_delta"])

                if improved:
                    early_stopping_best = monitor_value
                    early_stopping_bad_epochs = 0
                else:
                    early_stopping_bad_epochs += 1
                    logging.info(
                        "Early stopping patience: %d/%d without sufficient improvement.",
                        early_stopping_bad_epochs, settings["early_stopping_patience"],
                    )

        if settings["checkpoint_every"] > 0 and (epoch + 1) % settings["checkpoint_every"] == 0:
            periodic_checkpoint_path = periodic_checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
            save_checkpoint(periodic_checkpoint_path, epoch + 1, model, optimizer, args)
            logging.info("Saved periodic checkpoint at epoch %d to %s", epoch + 1, periodic_checkpoint_path)

        if (
            settings["early_stopping_enabled"]
            and early_stopping_bad_epochs >= settings["early_stopping_patience"]
        ):
            logging.info(
                "Stopping early at epoch %d after %d epochs without sufficient improvement.",
                epoch + 1, early_stopping_bad_epochs,
            )
            break

    save_checkpoint(last_checkpoint_path, settings["num_training_epochs"], model, optimizer, args)
    logging.info("Saved final checkpoint to %s", last_checkpoint_path)

    if validation_enabled and best_epoch is not None:
        logging.info(
            "Best validation checkpoint: epoch %d | val loss %.6f | path %s",
            best_epoch, best_val_loss, best_checkpoint_path,
        )
