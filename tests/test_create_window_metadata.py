import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from mne import create_info, use_log_level
from mne.io import RawArray

import scripts.create_window_metadata as metadata_module
from src.data import EEGMusicWindowDataset


class _FakeEpochs:
    def __init__(self, duration):
        self.epoch_data = np.zeros(
            (1, 125, int(round(125 * duration))),
            dtype=np.float32,
        )

    def get_data(self, copy=False):
        return self.epoch_data


class WindowLengthNamespaceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base_dir = Path(self.temp_dir.name)

    def _generate_metadata(self, window_length):
        metadata_path = (
            self.base_dir
            / "data"
            / "metadata"
            / f"{metadata_module._window_length_label(window_length)}.csv"
        )

        def fake_audio_embeddings(
                audio_encoder,
                song_audio_path,
                window_length,
                audio_dir,
                song_id,
        ):
            return {0: audio_dir / f"song{song_id}_win000.pt"}

        def fake_epochs(raw, duration, preload):
            return _FakeEpochs(duration)

        with (
            patch.object(metadata_module, "MERTFeatureExtractor", return_value=object()),
            patch.object(
                metadata_module,
                "_create_audio_embeddings_from_wav",
                side_effect=fake_audio_embeddings,
            ),
            patch.object(metadata_module, "load_preprocessed", return_value=object()),
            patch.object(
                metadata_module,
                "make_fixed_length_epochs",
                side_effect=fake_epochs,
            ),
            patch.object(metadata_module.torch, "save"),
        ):
            metadata_module.create_window_metadata(
                window_length=window_length,
                metadata_path=metadata_path,
                song_audio_path=self.base_dir / "song21.wav",
                song_id=21,
                base_dir=self.base_dir,
            )

        return pd.read_csv(metadata_path)

    def test_window_length_labels_are_stable_and_readable(self):
        self.assertEqual(metadata_module._window_length_label(1), "1s")
        self.assertEqual(metadata_module._window_length_label(5.0), "5s")
        self.assertEqual(metadata_module._window_length_label(0.5), "0.5s")
        self.assertNotEqual(
            metadata_module._window_length_label(0.3333333),
            metadata_module._window_length_label(0.3333334),
        )

    def test_cli_defaults_to_flat_repository_song_directory(self):
        cli_args = metadata_module.parse_args([])

        self.assertEqual(cli_args.full_song_dir, "data/songs")
        self.assertEqual(cli_args.output, "data/metadata/five_sec_windows.csv")

    def test_invalid_window_lengths_are_rejected(self):
        for value in (0, -1, float("inf"), float("nan"), True, "invalid"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "window_length"):
                    metadata_module._window_length_label(value)

    def test_one_and_five_second_runs_use_disjoint_artifact_paths(self):
        one_second = self._generate_metadata(1.0)
        five_second = self._generate_metadata(5.0)

        self.assertEqual(set(one_second["window_length_sec"]), {1.0})
        self.assertEqual(set(five_second["window_length_sec"]), {5.0})

        one_eeg_paths = set(one_second["eeg_path"])
        five_eeg_paths = set(five_second["eeg_path"])
        one_audio_paths = set(one_second["audio_path"])
        five_audio_paths = set(five_second["audio_path"])

        self.assertTrue(one_eeg_paths.isdisjoint(five_eeg_paths))
        self.assertTrue(one_audio_paths.isdisjoint(five_audio_paths))
        self.assertTrue(
            all(path.startswith("data/eeg/uv/1s/") for path in one_eeg_paths)
        )
        self.assertTrue(
            all(path.startswith("data/eeg/uv/5s/") for path in five_eeg_paths)
        )
        self.assertEqual(
            one_audio_paths,
            {"data/audio/mert/1s/song21/song21_win000.pt"},
        )
        self.assertEqual(
            five_audio_paths,
            {"data/audio/mert/5s/song21/song21_win000.pt"},
        )
        self.assertEqual(set(five_second["eeg_unit"]), {"uV"})


class FixedMicrovoltTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_path = Path(self.temp_dir.name)
        # Two five-second windows, with nonzero offsets and different channel
        # amplitudes: centering or per-channel scaling must not remove these.
        self.volts = np.array([
            np.linspace(-3e-6, 6e-6, 100),
            np.linspace(10e-6, 30e-6, 100),
        ])

    def _generate(self, name, volts):
        base_dir = self.temp_path / name
        raw = RawArray(volts.copy(), create_info(["E1", "E2"], 10, "eeg"), verbose=False)

        def fake_audio_embeddings(**kwargs):
            paths = {}
            for window_idx in range(2):
                path = kwargs["audio_dir"] / f"song21_win{window_idx:03d}.pt"
                torch.save({"audio_embed": torch.zeros(3, 4)}, path)
                paths[window_idx] = path
            return paths

        with (
            use_log_level("ERROR"),
            patch.object(metadata_module, "MERTFeatureExtractor", return_value=object()),
            patch.object(
                metadata_module, "_create_audio_embeddings_from_wav",
                side_effect=fake_audio_embeddings,
            ),
            patch.object(metadata_module, "load_preprocessed", return_value=raw),
        ):
            metadata_path = base_dir / "data" / "metadata" / "windows_uv.csv"
            metadata_module.create_window_metadata(
                window_length=5,
                metadata_path=metadata_path,
                song_audio_path=base_dir / "song21.wav",
                song_id=21,
                base_dir=base_dir,
            )

        # Reusing Raw across subjects must not repeatedly multiply its data.
        np.testing.assert_array_equal(raw.get_data(), volts)
        return base_dir, pd.read_csv(metadata_path)

    def test_saved_windows_are_fixed_microvolts_without_centering(self):
        base_dir, metadata = self._generate("original", self.volts)
        self.assertEqual(len(metadata), 40)
        self.assertEqual(set(metadata["eeg_unit"]), {"uV"})
        for row in metadata.itertuples():
            saved = torch.load(base_dir / row.eeg_path, weights_only=True)
            start = row.window_idx * 50
            expected = torch.as_tensor(self.volts[:, start:start + 50] * 1e6)
            torch.testing.assert_close(saved["epoch"], expected, rtol=0, atol=0)
            self.assertEqual(saved["eeg_unit"], "uV")
            self.assertTrue(row.eeg_path.startswith("data/eeg/uv/5s/"))
            self.assertEqual(row.start_time_sec, row.window_idx * 5)
            self.assertEqual(row.end_time_sec, (row.window_idx + 1) * 5)

    def test_other_windows_do_not_influence_scaling(self):
        base_dir, metadata = self._generate("original", self.volts)
        changed = self.volts.copy()
        changed[:, 50:] = changed[:, 50:] * 10000 + 1
        changed_dir, changed_metadata = self._generate("changed", changed)

        for subject_id in (0, 19):
            first = metadata[(metadata.subject_id == subject_id) & (metadata.window_idx == 0)].iloc[0]
            other = changed_metadata[
                (changed_metadata.subject_id == subject_id) & (changed_metadata.window_idx == 0)
            ].iloc[0]
            original_tensor = torch.load(base_dir / first.eeg_path, weights_only=True)["epoch"]
            changed_tensor = torch.load(changed_dir / other.eeg_path, weights_only=True)["epoch"]
            torch.testing.assert_close(original_tensor, changed_tensor, rtol=0, atol=0)

    def test_existing_dataset_loads_microvolts_without_rescaling(self):
        base_dir, metadata = self._generate("dataset", self.volts)
        metadata["split"] = "train"
        metadata["section_id"] = 0
        split_path = base_dir / "runs" / "uv-test" / "splits" / "test.csv"
        split_path.parent.mkdir(parents=True)
        metadata.to_csv(split_path, index=False)

        dataset = EEGMusicWindowDataset(split_path, split="train")
        torch.testing.assert_close(
            dataset[0]["eeg"], torch.as_tensor(self.volts[:, :50] * 1e6), rtol=0, atol=0,
        )


if __name__ == "__main__":
    unittest.main()
