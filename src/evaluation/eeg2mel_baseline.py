"""EEG2Mel baseline: EEG power-spectral-density regression to mel-spectrograms.

Modeled on the EEG2Mel baseline described in Qing et al.'s "Channel-Oriented
Design for EEG-to-Music Reconstruction" (see qing_reference/, a read-only
vendored reference -- nothing here imports from it). Unlike that reference,
which windows EEG/audio at 1 second, this project's windows are 5 seconds.
Training operates at the original 1-second sub-window granularity (matching
the reference architecture's conv-stack sizing and giving 5x more training
examples per cached row). At evaluation time, 5 predicted 1-second
mel-spectrograms are concatenated into one 5-second spectrogram. Mel-space
reconstruction and retrieval are the default evaluation, matching the
``feature/eeg2mel`` branch. Griffin-Lim reconstruction followed by frozen MERT
embedding remains available as an optional comparison.
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy import signal
from torch import nn

from src.data import EEGMusicWindowDataset

SUB_WINDOWS_PER_ROW = 5


def eeg_psd_transform(eeg, fs=125.0):
    """Compute a per-channel power spectral density for a raw EEG window.

    Args:
        eeg: EEG window, shape [channels, samples].
        fs: EEG sampling rate in Hz.

    Returns:
        PSD array with shape [channels, samples // 2 + 1].
    """
    eeg = np.asarray(eeg, dtype=np.float64)
    _, psd = signal.periodogram(eeg, fs=fs, axis=-1)
    return psd


def _conv2d_output_size(size, kernel_size, stride, padding):
    return (size + 2 * padding - kernel_size) // stride + 1


def eeg2mel_flatten_dim(psd_shape):
    """Deterministically compute the encoder's flattened feature size.

    Mirrors the exact Conv2d(k=4) x4 (stride 1) + Conv2d(k=4, stride 2) +
    MaxPool2d(2) stack below, so the first Linear layer's input size is
    always consistent with psd_shape without running a dummy forward pass.
    """
    height, width = psd_shape
    layer_specs = [(4, 1, 1)] * 4 + [(4, 2, 1)]
    for kernel_size, stride, padding in layer_specs:
        height = _conv2d_output_size(height, kernel_size, stride, padding)
        width = _conv2d_output_size(width, kernel_size, stride, padding)
    height //= 2  # MaxPool2d(kernel_size=2)
    width //= 2
    return 128 * height * width


class EEG2MelBaseline(nn.Module):
    """Predict a mel-spectrogram from an EEG power-spectral-density window."""

    def __init__(
        self,
        psd_shape=(125, 63),
        spec_shape=(64, 47),
        mel_sample_rate=24000,
        n_fft=1024,
        n_mels=64,
        hop_length=512,
        min_db=-70.0,
        max_db=14.0,
    ):
        super().__init__()
        self.psd_shape = tuple(psd_shape)
        self.spec_shape = tuple(spec_shape)
        self.min_db = float(min_db)
        self.max_db = float(max_db)
        self.griffin_lim_seed = 0

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(8), nn.Dropout(0.1),
            nn.Conv2d(8, 16, kernel_size=4, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(16), nn.Dropout(0.1),
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.Dropout(0.1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
        )
        self.flatten = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(128), nn.Flatten(),
        )

        flat_dim = eeg2mel_flatten_dim(self.psd_shape)
        self.linear = nn.Sequential(
            nn.Linear(flat_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            nn.Linear(128, self.spec_shape[0] * self.spec_shape[1]),
        )

        # Reconstruction path: predicted mel-spectrogram -> waveform.
        self.inverse_mel_transform = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=mel_sample_rate,
        )
        self.griffin_lim_transform = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, hop_length=hop_length, rand_init=True,
        )

    def forward(self, eeg_psd):
        """eeg_psd: [B, *psd_shape] -> predicted mel-spectrogram [B, *spec_shape]."""
        x = self.encoder(eeg_psd.unsqueeze(1))
        x = self.flatten(x)
        x = self.linear(x)
        return x.reshape(x.size(0), *self.spec_shape)

    @torch.no_grad()
    def to_wave(self, mel_spec):
        """Normalized mel-spectrogram in [-1, 1] -> reconstructed waveform.

        Griffin-Lim runs on CPU because torchaudio's CUDA implementation
        fails on some PyTorch/CUDA builds for both zero-phase and random
        initialization. The surrounding EEG2Mel and MERT inference remains
        on GPU. A forked, fixed-seed RNG context makes reconstruction
        repeatable without changing the caller's process-wide random state.
        """
        mel_spec = mel_spec.clamp(-1, 1)
        mel_spec = (mel_spec + 1) * (self.max_db - self.min_db) / 2 + self.min_db
        mel_spec = 2 ** mel_spec
        linear_spectrogram = self.inverse_mel_transform(mel_spec)
        cuda_devices = []
        if linear_spectrogram.is_cuda:
            device_index = linear_spectrogram.device.index
            cuda_devices = [
                torch.cuda.current_device() if device_index is None else device_index
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.griffin_lim_seed)
            transform = self.griffin_lim_transform
            return torchaudio.functional.griffinlim(
                linear_spectrogram.cpu(),
                transform.window.cpu(),
                transform.n_fft,
                transform.hop_length,
                transform.win_length,
                transform.power,
                transform.n_iter,
                transform.momentum,
                transform.length,
                transform.rand_init,
            )

    @torch.no_grad()
    def generate_wave(self, eeg_psd):
        predicted_mel = self.forward(eeg_psd)
        return self.to_wave(predicted_mel)


def resolve_eeg2mel_settings(args):
    """Resolve eeg2mel_baseline: config, applying defaults and testing:// fallbacks."""
    cfg = args.get("eeg2mel_baseline", {}) or {}
    testing_args = args["testing"]
    early_stopping_cfg = cfg.get("early_stopping", {}) or {}
    evaluation_cfg = cfg.get("evaluation", {}) or {}
    mert_space_cfg = evaluation_cfg.get("mert_space", {}) or {}

    settings = {
        "eeg_sample_rate": float(cfg.get("eeg_sample_rate", 125.0)),
        "sub_window_seconds": float(cfg.get("sub_window_seconds", 1.0)),
        "mel_sample_rate": int(cfg.get("mel_sample_rate", 24000)),
        "n_fft": int(cfg.get("n_fft", 1024)),
        "n_mels": int(cfg.get("n_mels", 64)),
        "hop_length": int(cfg.get("hop_length", 512)),
        "mel_min_db": float(cfg.get("mel_min_db", -70.0)),
        "mel_max_db": float(cfg.get("mel_max_db", 14.0)),
        "learning_rate": float(cfg.get("learning_rate", 0.0003)),
        "batch_size": int(cfg.get("batch_size", 16)),
        "num_training_epochs": int(cfg.get("num_training_epochs", 100)),
        "checkpoint_every": int(cfg.get("checkpoint_every", 0) or 0),
        "early_stopping_enabled": bool(early_stopping_cfg.get("enabled", False)),
        "early_stopping_patience": int(early_stopping_cfg.get("patience", 0) or 0),
        "early_stopping_min_delta": float(early_stopping_cfg.get("min_delta", 0.0) or 0.0),
        "early_stopping_monitor": str(early_stopping_cfg.get("monitor", "val_loss")),
        "early_stopping_mode": str(early_stopping_cfg.get("mode", "min")),
        "ks": tuple(int(k) for k in cfg.get("ks", [1, 5, 10])),
        "audio_pooling": str(cfg.get("audio_pooling", "sequence")),
        "n_perms": int(cfg.get("n_perms", testing_args["n_perms"])),
        "n_within_song_shuffles": int(
            cfg.get("n_within_song_shuffles", testing_args.get("n_within_song_shuffles", 200))
        ),
        "evaluation_batch_size": int(
            evaluation_cfg.get("batch_size") or cfg.get("batch_size", 16)
        ),
        "save_representations": bool(
            evaluation_cfg.get("save_representations", True)
        ),
        "mert_space_enabled": bool(mert_space_cfg.get("enabled", False)),
        "mert_model_name": str(
            mert_space_cfg.get("model_name", "m-a-p/MERT-v1-95M")
        ),
        "mert_batch_size": int(mert_space_cfg.get("batch_size", 2)),
        "output_filename": str(cfg.get("output_filename", "eeg2mel_baseline_test.json")),
    }

    if settings["early_stopping_mode"] not in {"min", "max"}:
        raise ValueError("eeg2mel_baseline.early_stopping.mode must be one of ['min', 'max'].")
    if settings["audio_pooling"] != "sequence":
        raise ValueError("eeg2mel_baseline.audio_pooling must be 'sequence'.")
    if not settings["ks"] or any(k < 1 for k in settings["ks"]):
        raise ValueError("eeg2mel_baseline.ks must contain positive integers.")
    if settings["evaluation_batch_size"] < 1:
        raise ValueError("eeg2mel_baseline.evaluation.batch_size must be at least 1.")
    if settings["mert_batch_size"] < 1:
        raise ValueError(
            "eeg2mel_baseline.evaluation.mert_space.batch_size must be at least 1."
        )

    settings["psd_shape"] = (
        125,
        int(round(settings["sub_window_seconds"] * settings["eeg_sample_rate"])) // 2 + 1,
    )
    frames_per_sub_window = 1 + int(
        round(settings["sub_window_seconds"] * settings["mel_sample_rate"])
    ) // settings["hop_length"]
    settings["spec_shape"] = (settings["n_mels"], frames_per_sub_window)

    return settings


def _resolve_relative_path(path_value, base_dir):
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path


class EEG2MelSubWindowDataset(EEGMusicWindowDataset):
    """Exposes each 5-second metadata row as 5 independent 1-second examples.

    Reloads the parent row's cached EEG/audio files once per sub-window
    access rather than once per row; with a shuffled DataLoader, sub-windows
    of the same row are not accessed consecutively, so a simple last-row
    cache would rarely help. Kept simple since this is a baseline, not the
    main model's hot path.
    """

    def __init__(self, metadata_path, split=None, eeg_sample_rate=125.0):
        super().__init__(metadata_path, split=split)
        self.eeg_sample_rate = float(eeg_sample_rate)
        self.samples_per_sub_window = int(round(self.eeg_sample_rate))

    def __len__(self):
        return len(self.metadata) * SUB_WINDOWS_PER_ROW

    def __getitem__(self, idx):
        row_idx, sub_idx = divmod(idx, SUB_WINDOWS_PER_ROW)
        row = self.metadata.iloc[row_idx]
        sample = super().__getitem__(row_idx)

        eeg = sample["eeg"]
        start = sub_idx * self.samples_per_sub_window
        end = start + self.samples_per_sub_window
        eeg_sub = eeg[:, start:end].numpy()
        eeg_psd = torch.as_tensor(
            eeg_psd_transform(eeg_sub, fs=self.eeg_sample_rate), dtype=torch.float32
        )

        mel_path = _resolve_relative_path(row["mel_path"], self.base_dir)
        mel_sample = torch.load(mel_path, weights_only=False)
        mel_targets = mel_sample["mel_targets"]
        if not isinstance(mel_targets, torch.Tensor):
            mel_targets = torch.as_tensor(mel_targets)
        mel_target = mel_targets[sub_idx].float()

        return {
            "eeg_psd": eeg_psd,
            "mel_target": mel_target,
            "song_id": sample["song_id"],
            "window_idx": sample["window_idx"],
            "subject_id": sample["subject_id"],
            "sub_idx": sub_idx,
        }


class EEG2MelEvalDataset(EEGMusicWindowDataset):
    """Exposes each 5-second row with all 5 stacked sub-window PSDs.

    Used at evaluation time: the model runs on all 5 sub-windows, and the 5
    predicted mel-spectrograms are concatenated back into one 5-second
    spectrogram before reconstruction, matching the main model's own
    5-second retrieval granularity.
    """

    def __init__(
        self,
        metadata_path,
        split=None,
        eeg_sample_rate=125.0,
        include_audio=True,
    ):
        super().__init__(metadata_path, split=split)
        self.eeg_sample_rate = float(eeg_sample_rate)
        self.samples_per_sub_window = int(round(self.eeg_sample_rate))
        self.include_audio = bool(include_audio)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        if self.include_audio:
            sample = super().__getitem__(idx)
            eeg = sample["eeg"]
        else:
            eeg_path = _resolve_relative_path(row["eeg_path"], self.base_dir)
            eeg_payload = torch.load(eeg_path, weights_only=False)
            eeg = eeg_payload["epoch"]
            if not isinstance(eeg, torch.Tensor):
                eeg = torch.as_tensor(eeg)

        mel_path = _resolve_relative_path(row["mel_path"], self.base_dir)
        mel_payload = torch.load(mel_path, weights_only=False)
        mel_targets = mel_payload["mel_targets"]
        if not isinstance(mel_targets, torch.Tensor):
            mel_targets = torch.as_tensor(mel_targets)
        mel_targets = mel_targets.float()
        if mel_targets.shape[0] != SUB_WINDOWS_PER_ROW:
            raise ValueError(
                f"Expected {SUB_WINDOWS_PER_ROW} mel targets in {mel_path}, "
                f"got shape {tuple(mel_targets.shape)}."
            )

        psd_stack = []
        for sub_idx in range(SUB_WINDOWS_PER_ROW):
            start = sub_idx * self.samples_per_sub_window
            end = start + self.samples_per_sub_window
            eeg_sub = eeg[:, start:end].numpy()
            psd_stack.append(eeg_psd_transform(eeg_sub, fs=self.eeg_sample_rate))
        eeg_psd_stack = torch.as_tensor(np.stack(psd_stack), dtype=torch.float32)

        result = {
            "eeg_psd_stack": eeg_psd_stack,
            "mel_target_stack": mel_targets,
            "subject_id": int(row["subject_id"]),
            "song_id": int(row["song_id"]),
            "window_idx": int(row["window_idx"]),
            "section_id": int(row.get("section_id", -1)),
        }
        if self.include_audio:
            result["audio"] = sample["audio"]
        return result


def concatenate_sub_window_mels(predicted_mel_stack):
    """[B, 5, n_mels, T_sub] -> [B, n_mels, 5 * T_sub], in sub-window order."""
    batch_size, n_sub, n_mels, n_frames = predicted_mel_stack.shape
    return predicted_mel_stack.permute(0, 2, 1, 3).reshape(
        batch_size, n_mels, n_sub * n_frames
    )


@torch.no_grad()
def predict_mel_windows(model, eeg_psd_stack, device):
    """Predict one- and five-second mel windows for an evaluation batch."""
    model.eval()
    batch_size, n_sub = eeg_psd_stack.shape[:2]
    flat_psd = eeg_psd_stack.reshape(
        batch_size * n_sub,
        *eeg_psd_stack.shape[2:],
    ).to(device)
    flat_prediction = model(flat_psd)
    predicted = flat_prediction.reshape(
        batch_size,
        n_sub,
        *flat_prediction.shape[1:],
    )
    return predicted, concatenate_sub_window_mels(predicted)


@torch.no_grad()
def embed_predicted_mels(model, predicted_mel_stack, mert_extractor):
    """Invert predicted mel windows and encode the waveforms with frozen MERT."""
    concatenated_mel = concatenate_sub_window_mels(predicted_mel_stack)
    waveforms = model.to_wave(concatenated_mel)
    waveform_arrays = [
        waveform.astype(np.float32) for waveform in waveforms.cpu().numpy()
    ]
    return mert_extractor.encode_arrays(waveform_arrays).cpu()


@torch.no_grad()
def reconstruct_and_embed(model, eeg_psd_stack, mert_extractor, device):
    """Reconstruct one 5-second waveform per row and embed it with frozen MERT.

    Args:
        model: EEG2MelBaseline, in eval mode.
        eeg_psd_stack: [B, 5, *psd_shape] stacked per-sub-window PSDs.
        mert_extractor: frozen MERTFeatureExtractor from src.encoders.
        device: torch device the model and MERT extractor live on.

    Returns:
        [B, 768, T] final-layer MERT sequences for the reconstructed audio.
    """
    predicted_mel, _ = predict_mel_windows(model, eeg_psd_stack, device)
    return embed_predicted_mels(model, predicted_mel, mert_extractor)
