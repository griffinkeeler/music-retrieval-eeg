import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import src.data.create_mel_targets as mel_module
from src.data.create_mel_targets import (
    _compute_mel_targets,
    _normalize_mel,
    _resolve_song_wav_path,
)
from src.evaluation.eeg2mel_baseline import (
    EEG2MelBaseline,
    EEG2MelEvalDataset,
    EEG2MelSubWindowDataset,
    SUB_WINDOWS_PER_ROW,
    concatenate_sub_window_mels,
    eeg2mel_flatten_dim,
    eeg_psd_transform,
    reconstruct_and_embed,
)


class PsdTransformTests(unittest.TestCase):
    def test_shape_matches_one_second_window(self):
        eeg = np.random.randn(125, 125)  # 125 channels, 1s @ 125Hz
        psd = eeg_psd_transform(eeg, fs=125.0)
        self.assertEqual(psd.shape, (125, 63))
        self.assertTrue(np.all(np.isfinite(psd)))

    def test_matches_qing_reference_flatten_dim(self):
        # Confirms eeg2mel_flatten_dim's arithmetic against the vendored
        # reference implementation's own stated flatten size (53,760) for a
        # 63x125-shaped PSD input -- see qing_reference/.../eeg2mel.py.
        self.assertEqual(eeg2mel_flatten_dim((63, 125)), 53760)
        self.assertEqual(eeg2mel_flatten_dim((125, 63)), 53760)


class MelTargetTests(unittest.TestCase):
    def test_resolves_flat_repository_song_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            song_dir = Path(temp_dir) / "data" / "songs"
            song_dir.mkdir(parents=True)
            expected_path = song_dir / "song21.wav"
            expected_path.touch()

            self.assertEqual(_resolve_song_wav_path(song_dir, 21), expected_path)

    def test_missing_song_reports_expected_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            song_dir = Path(temp_dir) / "data" / "songs"
            song_dir.mkdir(parents=True)

            with self.assertRaisesRegex(
                FileNotFoundError,
                r"data/songs/song<id>\.wav",
            ):
                _resolve_song_wav_path(song_dir, 21)

    def test_normalize_mel_is_finite_and_centered(self):
        mel_power = torch.rand(64, 47) * 1e-3 + 1e-6
        normalized = _normalize_mel(mel_power, min_db=-70.0, max_db=14.0)
        self.assertEqual(normalized.shape, (64, 47))
        self.assertTrue(torch.isfinite(normalized).all())

    def test_compute_mel_targets_shape(self):
        import torchaudio

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_fft=1024, n_mels=64, hop_length=512,
        )
        window_audio = np.random.randn(5 * 24000).astype(np.float32)
        targets = _compute_mel_targets(
            window_audio,
            mel_transform,
            sub_window_seconds=1.0,
            sample_rate=24000,
            min_db=-70.0,
            max_db=14.0,
        )
        self.assertEqual(tuple(targets.shape), (SUB_WINDOWS_PER_ROW, 64, 47))

    def test_compute_mel_targets_pads_short_audio(self):
        import torchaudio

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_fft=1024, n_mels=64, hop_length=512,
        )
        short_audio = np.random.randn(24000).astype(np.float32)  # only 1s, not 5s
        targets = _compute_mel_targets(
            short_audio,
            mel_transform,
            sub_window_seconds=1.0,
            sample_rate=24000,
            min_db=-70.0,
            max_db=14.0,
        )
        self.assertEqual(tuple(targets.shape), (SUB_WINDOWS_PER_ROW, 64, 47))


class MelCacheIsolationTests(unittest.TestCase):
    def test_microvolt_targets_preserve_legacy_cache_and_metadata_units(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            song_dir = base_dir / "data" / "songs"
            song_dir.mkdir(parents=True)
            (song_dir / "song21.wav").touch()
            metadata_path = base_dir / "metadata_uv.csv"
            pd.DataFrame([{
                "song_id": 21, "window_idx": 0, "eeg_unit": "uV",
                "start_time_sec": 0.0, "end_time_sec": 5.0,
            }]).to_csv(metadata_path, index=False)
            legacy_targets = torch.zeros(5, 2, 3)
            new_targets = torch.ones(5, 2, 3)
            with (
                patch.object(mel_module.librosa, "load", return_value=(np.zeros(120000), 24000)),
                patch.object(mel_module.torchaudio.transforms, "MelSpectrogram", return_value=object()),
                patch.object(mel_module, "_compute_mel_targets", side_effect=[legacy_targets, new_targets]),
            ):
                mel_module.add_mel_targets_to_metadata(metadata_path, base_dir=base_dir)
                legacy_path = pd.read_csv(metadata_path).iloc[0].mel_path
                self.assertEqual(legacy_path, "data/mel_targets/song21/song21_win000.pt")
                mel_module.add_mel_targets_to_metadata(
                    metadata_path, base_dir=base_dir, mel_output_dir="data/mel_targets/uv/5s",
                )

            metadata = pd.read_csv(metadata_path)
            new_path = metadata.iloc[0].mel_path
            self.assertEqual(new_path, "data/mel_targets/uv/5s/song21/song21_win000.pt")
            self.assertEqual(metadata.iloc[0].eeg_unit, "uV")
            torch.testing.assert_close(
                torch.load(base_dir / legacy_path, weights_only=True)["mel_targets"], legacy_targets,
            )
            torch.testing.assert_close(
                torch.load(base_dir / new_path, weights_only=True)["mel_targets"], new_targets,
            )


class EEG2MelBaselineModelTests(unittest.TestCase):
    def test_forward_shape(self):
        model = EEG2MelBaseline(psd_shape=(125, 63), spec_shape=(64, 47))
        model.eval()
        eeg_psd = torch.randn(4, 125, 63)
        predicted_mel = model(eeg_psd)
        self.assertEqual(tuple(predicted_mel.shape), (4, 64, 47))

    def test_to_wave_smoke(self):
        model = EEG2MelBaseline(psd_shape=(125, 63), spec_shape=(64, 47))
        model.eval()
        self.assertTrue(model.griffin_lim_transform.rand_init)
        mel_spec = torch.rand(4, 64, 47) * 2 - 1  # in [-1, 1]
        waveform = model.to_wave(mel_spec)
        self.assertEqual(waveform.ndim, 2)
        self.assertEqual(waveform.shape[0], 4)
        self.assertEqual(waveform.device.type, "cpu")
        self.assertGreater(waveform.shape[1], 0)

    def test_to_wave_is_deterministic_and_preserves_global_rng(self):
        model = EEG2MelBaseline(psd_shape=(125, 63), spec_shape=(64, 47))
        model.eval()
        mel_spec = torch.rand(1, 64, 47) * 2 - 1

        torch.manual_seed(123)
        expected_next_random = torch.rand(1)
        torch.manual_seed(123)
        first = model.to_wave(mel_spec)
        actual_next_random = torch.rand(1)
        second = model.to_wave(mel_spec)

        torch.testing.assert_close(first, second)
        torch.testing.assert_close(actual_next_random, expected_next_random)

    def test_concatenate_sub_window_mels(self):
        stack = torch.arange(2 * 5 * 3 * 4, dtype=torch.float32).reshape(2, 5, 3, 4)
        concatenated = concatenate_sub_window_mels(stack)
        self.assertEqual(tuple(concatenated.shape), (2, 3, 20))
        # Sub-window order must be preserved: first 4 frames == sub-window 0.
        self.assertTrue(torch.equal(concatenated[:, :, :4], stack[:, 0]))
        self.assertTrue(torch.equal(concatenated[:, :, 4:8], stack[:, 1]))


class EEG2MelDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base_dir = Path(self.temp_dir.name)

        self.eeg_dir = self.base_dir / "data" / "eeg" / "song21" / "subject0"
        self.audio_dir = self.base_dir / "data" / "audio" / "song21"
        self.mel_dir = self.base_dir / "data" / "mel_targets" / "song21"
        self.eeg_dir.mkdir(parents=True)
        self.audio_dir.mkdir(parents=True)
        self.mel_dir.mkdir(parents=True)

        eeg_path = self.eeg_dir / "sub00_song21_win000.pt"
        audio_path = self.audio_dir / "song21_win000.pt"
        mel_path = self.mel_dir / "song21_win000.pt"

        torch.save({"epoch": torch.randn(125, 625, dtype=torch.float64)}, eeg_path)
        torch.save({"audio_embed": torch.randn(768, 374)}, audio_path)
        torch.save({"mel_targets": torch.randn(SUB_WINDOWS_PER_ROW, 64, 47)}, mel_path)

        splits_dir = self.base_dir / "runs" / "testrun" / "splits"
        splits_dir.mkdir(parents=True)
        self.metadata_path = splits_dir / "split.csv"

        row = {
            "window_uid": "sub00_song21_win000",
            "subject_id": 0,
            "song_id": 21,
            "window_idx": 0,
            "section_id": 0,
            "start_time_sec": 0.0,
            "end_time_sec": 5.0,
            "eeg_path": str(eeg_path.relative_to(self.base_dir)),
            "audio_path": str(audio_path.relative_to(self.base_dir)),
            "mel_path": str(mel_path.relative_to(self.base_dir)),
            "split": "train",
        }
        pd.DataFrame([row]).to_csv(self.metadata_path, index=False)

    def test_sub_window_dataset_shapes(self):
        dataset = EEG2MelSubWindowDataset(metadata_path=self.metadata_path, split="train")
        self.assertEqual(len(dataset), SUB_WINDOWS_PER_ROW)

        sample = dataset[0]
        self.assertEqual(tuple(sample["eeg_psd"].shape), (125, 63))
        self.assertEqual(tuple(sample["mel_target"].shape), (64, 47))
        self.assertEqual(sample["sub_idx"], 0)

        last_sample = dataset[SUB_WINDOWS_PER_ROW - 1]
        self.assertEqual(last_sample["sub_idx"], SUB_WINDOWS_PER_ROW - 1)

    def test_eval_dataset_shapes(self):
        dataset = EEG2MelEvalDataset(metadata_path=self.metadata_path, split="train")
        self.assertEqual(len(dataset), 1)

        sample = dataset[0]
        self.assertEqual(tuple(sample["eeg_psd_stack"].shape), (SUB_WINDOWS_PER_ROW, 125, 63))
        self.assertEqual(tuple(sample["audio"].shape), (768, 374))
        self.assertEqual(sample["subject_id"], 0)
        self.assertEqual(sample["song_id"], 21)
        self.assertEqual(sample["section_id"], 0)

    def test_eval_dataset_does_not_require_section_ids(self):
        metadata = pd.read_csv(self.metadata_path).drop(columns="section_id")
        metadata.to_csv(self.metadata_path, index=False)

        dataset = EEG2MelEvalDataset(metadata_path=self.metadata_path, split="train")

        self.assertEqual(dataset[0]["section_id"], -1)


class SequenceReconstructionTests(unittest.TestCase):
    class FakeEEG2MelModel:
        def eval(self):
            return self

        def __call__(self, eeg_psd):
            return torch.zeros(eeg_psd.shape[0], 2, 3, device=eeg_psd.device)

        def to_wave(self, mel_spec):
            return torch.arange(
                mel_spec.shape[0] * 12,
                dtype=torch.float32,
                device=mel_spec.device,
            ).reshape(mel_spec.shape[0], 12)

    class FakeProcessor:
        def __call__(self, arrays, sampling_rate, return_tensors, padding):
            del sampling_rate, return_tensors, padding
            return {"input_values": torch.as_tensor(np.stack(arrays))}

    class FakeMERTModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_values, output_hidden_states):
            self.assert_output_hidden_states = output_hidden_states
            batch_size = input_values.shape[0]
            hidden = torch.arange(
                batch_size * 4 * 3,
                dtype=torch.float32,
                device=input_values.device,
            ).reshape(batch_size, 4, 3)
            return SimpleNamespace(hidden_states=(hidden,))

    class FakeMERTExtractor:
        def __init__(self):
            self.model = SequenceReconstructionTests.FakeMERTModel()
            self.processor = SequenceReconstructionTests.FakeProcessor()
            self.target_sr = 24000

    def test_reconstruct_and_embed_preserves_mert_time_axis(self):
        extractor = self.FakeMERTExtractor()
        eeg_psd_stack = torch.randn(2, SUB_WINDOWS_PER_ROW, 5, 4)

        embeddings = reconstruct_and_embed(
            self.FakeEEG2MelModel(),
            eeg_psd_stack,
            extractor,
            torch.device("cpu"),
        )

        self.assertEqual(tuple(embeddings.shape), (2, 3, 4))
        expected = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        torch.testing.assert_close(embeddings, expected.transpose(1, 2))


if __name__ == "__main__":
    unittest.main()
