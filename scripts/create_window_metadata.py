import argparse
import math
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SONG_DIR = Path("data/songs")
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import pandas as pd
import torch
from mne import make_fixed_length_epochs

from scripts.load_data import load_preprocessed
from src.encoders import MERTFeatureExtractor

EEG_UNIT = "uV"
VOLTS_TO_MICROVOLTS = 1e6


def _validate_window_length(window_length):
    if isinstance(window_length, bool):
        raise ValueError("window_length must be a finite number greater than 0.")
    try:
        window_length = float(window_length)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "window_length must be a finite number greater than 0."
        ) from error
    if not math.isfinite(window_length) or window_length <= 0:
        raise ValueError("window_length must be a finite number greater than 0.")
    return window_length


def _window_length_label(window_length):
    """Return the directory label used to isolate one window duration."""
    window_length = _validate_window_length(window_length)
    return f"{window_length:.15g}s"


def _compute_audio_embedding(audio_encoder, audio_array):
    inputs = audio_encoder.processor(
        [audio_array],
        sampling_rate=audio_encoder.target_sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(next(audio_encoder.model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = audio_encoder.model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[-1]  # [1, T, 768]
    return hidden.squeeze(0).transpose(0, 1).contiguous().cpu()


def _create_audio_embeddings_from_wav(audio_encoder, song_audio_path, window_length, audio_dir, song_id):
    audio_array, sample_rate = librosa.load(song_audio_path, sr=audio_encoder.target_sr)
    samples_per_window = int(round(window_length * sample_rate))
    if samples_per_window <= 0:
        raise ValueError("window_length must be greater than 0.")

    full_window_count = len(audio_array) // samples_per_window
    if full_window_count == 0:
        raise ValueError(
            f"Audio file {song_audio_path} is shorter than one {window_length}-second window."
        )

    audio_paths_by_window = {}
    for window_idx in range(full_window_count):
        start_sample = window_idx * samples_per_window
        end_sample = start_sample + samples_per_window
        audio_window = audio_array[start_sample:end_sample]
        audio_path = audio_dir / f"song{song_id}_win{window_idx:03d}.pt"
        audio_emb = _compute_audio_embedding(audio_encoder, audio_window)

        torch.save({"audio_embed": audio_emb}, audio_path)
        audio_paths_by_window[window_idx] = audio_path

    return audio_paths_by_window


def _create_audio_embeddings_from_clip_dir(audio_encoder, song_dir, audio_dir, song_id):
    audio_paths_by_window = {}
    clip_paths = sorted(
        song_dir.glob("clip_*.wav"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )

    for window_idx, clip_path in enumerate(clip_paths):
        audio_path = audio_dir / f"song{song_id}_win{window_idx:03d}.pt"
        audio_emb = audio_encoder([clip_path]).squeeze(0).cpu()

        torch.save({"audio_embed": audio_emb}, audio_path)
        audio_paths_by_window[window_idx] = audio_path

    return audio_paths_by_window


def _get_song_audio_files(full_song_dir):
    song_audio_files = {}
    for wav_path in sorted(full_song_dir.glob("song*.wav")):
        song_label = wav_path.stem
        song_id = int(song_label.removeprefix("song"))
        song_audio_files[song_id] = wav_path
    return song_audio_files


def create_window_metadata(
        window_length,
        metadata_path,
        song_audio_path=None,
        song_id=None,
        full_song_dir=None,
        base_dir=None,
):
    """Create aligned metadata and fixed-microvolt EEG/MERT artifacts.

    MNE receives volts from load_preprocessed. After windowing, a fixed 1e6
    multiplier converts EEG to microvolts; no recording-, subject-, or
    dataset-level location/scale statistics are fitted. EEG tensors are stored
    under ``data/eeg/uv/<duration>/`` to avoid overwriting legacy robust-scaled
    tensors. MERT features remain under ``data/audio/mert/<duration>/``.
    """
    rows = []
    window_length = _validate_window_length(window_length)
    window_label = _window_length_label(window_length)
    base_dir = (
        Path(base_dir) if base_dir is not None else PROJECT_ROOT
    )
    metadata_path = Path(metadata_path)
    audio_encoder = MERTFeatureExtractor()
    full_song_dir = (
        Path(full_song_dir)
        if full_song_dir is not None
        else base_dir / DEFAULT_SONG_DIR
    )

    if song_audio_path is not None:
        if song_id is None:
            raise ValueError("song_id must be provided when song_audio_path is set.")
        song_audio_paths = {song_id: Path(song_audio_path)}
    else:
        if full_song_dir.exists():
            song_audio_paths = _get_song_audio_files(full_song_dir)
            if not song_audio_paths:
                raise ValueError(f"No .wav files found in {full_song_dir}.")
        else:
            song_audio_paths = {}

    if song_audio_paths:
        song_ids = sorted(song_audio_paths)
    else:
        song_ids = list(range(21, 31))

    for current_song_id in song_ids:
        data_path = base_dir / "data" / "processed" / f"song{current_song_id}_Imputed.mat"
        song_dir = base_dir / "data" / "songs" / f"song{current_song_id}"
        audio_dir = (
            base_dir
            / "data"
            / "audio"
            / "mert"
            / window_label
            / f"song{current_song_id}"
        )
        audio_dir.mkdir(parents=True, exist_ok=True)

        if current_song_id in song_audio_paths:
            audio_paths_by_window = _create_audio_embeddings_from_wav(
                audio_encoder=audio_encoder,
                song_audio_path=song_audio_paths[current_song_id],
                window_length=window_length,
                audio_dir=audio_dir,
                song_id=current_song_id,
            )
        else:
            audio_paths_by_window = _create_audio_embeddings_from_clip_dir(
                audio_encoder=audio_encoder,
                song_dir=song_dir,
                audio_dir=audio_dir,
                song_id=current_song_id,
            )

        for subject_id in range(0, 20):
            raw = load_preprocessed(
                file_path=str(data_path),
                subject_id=subject_id,
                song_id=current_song_id,
            )
            epochs = make_fixed_length_epochs(raw, duration=window_length, preload=True)
            # Keep MNE's data in volts. Multiplication creates a separate array
            # in microvolts without fitting statistics or modifying the epochs.
            epoch_data = epochs.get_data(copy=False) * VOLTS_TO_MICROVOLTS

            shared_window_count = min(len(epoch_data), len(audio_paths_by_window))
            if shared_window_count < len(epoch_data) or shared_window_count < len(audio_paths_by_window):
                warnings.warn(
                    "Window count mismatch for "
                    f"song {current_song_id}, subject {subject_id}: "
                    f"{len(epoch_data)} EEG windows vs {len(audio_paths_by_window)} audio windows. "
                    f"Using first {shared_window_count} aligned windows."
                )

            for window_idx in range(shared_window_count):
                eeg_dir = (
                    base_dir
                    / "data"
                    / "eeg"
                    / "uv"
                    / window_label
                    / f"song{current_song_id}"
                    / f"subject{subject_id}"
                )
                eeg_dir.mkdir(parents=True, exist_ok=True)

                eeg_path = eeg_dir / f"sub{subject_id:02d}_song{current_song_id}_win{window_idx:03d}.pt"
                audio_path = audio_paths_by_window[window_idx]
                start = window_idx * window_length
                end = start + window_length

                torch.save(
                    {"epoch": torch.as_tensor(epoch_data[window_idx]), "eeg_unit": EEG_UNIT},
                    eeg_path,
                )

                rows.append({
                    "window_uid": f"sub{subject_id:02d}_song{current_song_id}_win{window_idx:03d}",
                    "subject_id": subject_id,
                    "song_id": current_song_id,
                    "window_idx": window_idx,
                    "window_length_sec": window_length,
                    "start_time_sec": start,
                    "end_time_sec": end,
                    "eeg_path": str(eeg_path.relative_to(base_dir)),
                    "eeg_unit": EEG_UNIT,
                    "audio_path": str(audio_path.relative_to(base_dir)),
                })

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(metadata_path, index=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create fixed-microvolt EEG windows and frozen MERT features."
    )
    parser.add_argument("--window-length", type=float, default=5.0)
    parser.add_argument(
        "--output",
        default="data/metadata/five_sec_windows.csv",
        help="Output metadata CSV; defaults to the canonical five-second metadata file.",
    )
    parser.add_argument(
        "--full-song-dir",
        default=str(DEFAULT_SONG_DIR),
        help="Directory containing song<id>.wav files.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    base_dir = PROJECT_ROOT
    cli_args = parse_args()
    metadata_path = Path(cli_args.output)
    if not metadata_path.is_absolute():
        metadata_path = base_dir / metadata_path
    full_song_dir = Path(cli_args.full_song_dir)
    if not full_song_dir.is_absolute():
        full_song_dir = base_dir / full_song_dir
    create_window_metadata(
        window_length=cli_args.window_length,
        metadata_path=metadata_path,
        full_song_dir=full_song_dir,
        base_dir=base_dir,
    )
