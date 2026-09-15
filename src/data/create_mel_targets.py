"""Build ground-truth mel-spectrogram targets for the EEG2Mel baseline.

Each 5-second metadata row's audio span is split into 5 consecutive 1-second
sub-windows (matching EEG2MelSubWindowDataset's training granularity, see
src/evaluation/eeg2mel_baseline.py), and each sub-window's mel-spectrogram is
cached to disk. A `mel_path` column is added to the metadata CSV in place,
mirroring how create_window_metadata.py caches MERT embeddings via
`audio_path`. EEGMusicWindowDataset ignores unknown columns, so this is
non-breaking for the main model and ridge baseline.

Note: create_window_metadata.py's __main__ fully regenerates the metadata
CSV. If that script is rerun, this one must be rerun afterward to restore
the `mel_path` column.
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from omegaconf import OmegaConf

SUB_WINDOWS_PER_ROW = 5
DEFAULT_SONG_DIR = Path("data/songs")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create cached mel targets for the paper EEG2Mel baseline."
    )
    parser.add_argument(
        "--config",
        default="configs/table1_eeg2mel.yaml",
        help="Path to the standalone paper EEG2Mel config.",
    )
    return parser.parse_args()


def _normalize_mel(mel_power, min_db, max_db):
    """Invert of EEG2MelBaseline.to_wave's de-normalization: power -> [-1, 1]."""
    mel_db = torch.clamp(mel_power, min=1e-10).log2()
    return 2.0 * (mel_db - min_db) / (max_db - min_db) - 1.0


def _compute_mel_targets(
    window_audio,
    mel_transform,
    sub_window_seconds,
    sample_rate,
    min_db,
    max_db,
):
    samples_per_sub_window = int(round(sub_window_seconds * sample_rate))
    target_length = SUB_WINDOWS_PER_ROW * samples_per_sub_window

    if len(window_audio) < target_length:
        window_audio = np.pad(window_audio, (0, target_length - len(window_audio)))
    else:
        window_audio = window_audio[:target_length]

    targets = []
    for sub_idx in range(SUB_WINDOWS_PER_ROW):
        start = sub_idx * samples_per_sub_window
        end = start + samples_per_sub_window
        sub_audio = torch.as_tensor(window_audio[start:end], dtype=torch.float32)
        mel_power = mel_transform(sub_audio)
        targets.append(_normalize_mel(mel_power, min_db, max_db))

    return torch.stack(targets)  # [SUB_WINDOWS_PER_ROW, n_mels, n_frames]


def _resolve_song_wav_path(song_dir, song_id):
    """Resolve one full-song WAV from the repository's flat song directory."""
    song_wav_path = Path(song_dir) / f"song{int(song_id)}.wav"
    if not song_wav_path.is_file():
        raise FileNotFoundError(
            f"Song audio not found: {song_wav_path}. Expected the repository "
            "layout data/songs/song<id>.wav."
        )
    return song_wav_path


def add_mel_targets_to_metadata(
    metadata_path,
    full_song_dir=None,
    mel_sample_rate=24000,
    n_fft=1024,
    n_mels=64,
    hop_length=512,
    sub_window_seconds=1.0,
    min_db=-70.0,
    max_db=14.0,
    mel_output_dir=None,
    base_dir=None,
):
    """Write mel targets and attach their paths, optionally isolating a rerun.

    mel_output_dir is relative to base_dir unless absolute. The default keeps
    the original data/mel_targets layout; microvolt reruns use a separate cache.
    """
    base_dir = Path(base_dir) if base_dir is not None else Path(__file__).parents[2]
    metadata_path = Path(metadata_path)
    metadata = pd.read_csv(metadata_path)
    full_song_dir = (
        Path(full_song_dir)
        if full_song_dir is not None
        else base_dir / DEFAULT_SONG_DIR
    )

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=mel_sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length,
    )

    mel_dir_root = (
        Path(mel_output_dir) if mel_output_dir is not None else Path("data/mel_targets")
    )
    if not mel_dir_root.is_absolute():
        mel_dir_root = base_dir / mel_dir_root
    mel_paths = {}

    for song_id in sorted(metadata["song_id"].unique()):
        song_wav_path = _resolve_song_wav_path(full_song_dir, song_id)
        audio_array, _ = librosa.load(song_wav_path, sr=mel_sample_rate)

        mel_dir = mel_dir_root / f"song{song_id}"
        mel_dir.mkdir(parents=True, exist_ok=True)

        song_rows = metadata[metadata["song_id"] == song_id]
        for window_idx in sorted(song_rows["window_idx"].unique()):
            row = song_rows[song_rows["window_idx"] == window_idx].iloc[0]
            start_sample = int(round(float(row["start_time_sec"]) * mel_sample_rate))
            end_sample = int(round(float(row["end_time_sec"]) * mel_sample_rate))
            window_audio = audio_array[start_sample:end_sample]

            mel_targets = _compute_mel_targets(
                window_audio,
                mel_transform,
                sub_window_seconds=sub_window_seconds,
                sample_rate=mel_sample_rate,
                min_db=min_db,
                max_db=max_db,
            )

            mel_path = mel_dir / f"song{song_id}_win{window_idx:03d}.pt"
            torch.save({"mel_targets": mel_targets}, mel_path)
            mel_paths[(int(song_id), int(window_idx))] = str(mel_path.relative_to(base_dir))

    metadata["mel_path"] = metadata.apply(
        lambda row: mel_paths[(int(row["song_id"]), int(row["window_idx"]))], axis=1
    )
    metadata.to_csv(metadata_path, index=False)


if __name__ == "__main__":
    base_dir = Path(__file__).parents[2]
    cli_args = parse_args()
    config_path = Path(cli_args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    config = OmegaConf.load(config_path)
    settings = config.get("eeg2mel_baseline", {}) or {}
    meta_path = base_dir / "data" / "metadata" / config["metadata"]["filename"]
    add_mel_targets_to_metadata(
        metadata_path=meta_path,
        mel_sample_rate=int(settings.get("mel_sample_rate", 24000)),
        n_fft=int(settings.get("n_fft", 1024)),
        n_mels=int(settings.get("n_mels", 64)),
        hop_length=int(settings.get("hop_length", 512)),
        sub_window_seconds=float(settings.get("sub_window_seconds", 1.0)),
        min_db=float(settings.get("mel_min_db", -70.0)),
        max_db=float(settings.get("mel_max_db", 14.0)),
        mel_output_dir=settings.get("mel_output_dir"),
    )
