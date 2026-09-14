from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class EEGMusicWindowDataset(Dataset):
    def __init__(self, metadata_path, split=None):
        """
        Contains EEG and song audio tensors with their corresponding song
        and subject ID's.

        Args:
            metadata_path: The path to the csv file.
            split: train, val, or test split.
        """
        self.metadata_path = Path(metadata_path).resolve()
        self.metadata = pd.read_csv(self.metadata_path)
        self.base_dir = self.metadata_path.parents[3]

        if split is not None:
            self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        eeg_path = Path(row["eeg_path"])
        audio_path = Path(row["audio_path"])

        if not eeg_path.is_absolute():
            eeg_path = self.base_dir / eeg_path
        if not audio_path.is_absolute():
            audio_path = self.base_dir / audio_path

        eeg_sample = torch.load(eeg_path, weights_only=False)
        audio_sample = torch.load(audio_path, weights_only=False)

        eeg = eeg_sample["epoch"]
        audio = audio_sample["audio_embed"]

        if not isinstance(eeg, torch.Tensor):
            eeg = torch.as_tensor(eeg)
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)

        return {
            "eeg": eeg,
            "audio": audio,
            "subject_id": int(row["subject_id"]),
            "window_idx": int(row["window_idx"]),
            "song_id": int(row["song_id"]),
            "section_id": int(row.get("section_id", -1)),
            "start_time": float(row["start_time_sec"]),
            "end_time": float(row["end_time_sec"]),
        }
