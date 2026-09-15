import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from src.config import load_config
from src.config import resolve_split_path


class PaperConfigTests(unittest.TestCase):
    def test_each_split_config_inherits_shared_settings(self):
        project_root = Path(__file__).resolve().parents[1]
        expected = {
            "chunk_out": "chunk_out_fold0.csv",
            "random_segment_out": "random_segment_out_fold0.csv",
            "song_out": "song_out_fold0.csv",
            "subject_out": "subject_out_fold0.csv",
            "subject_song_out": "subject_song_out_fold0.csv",
        }

        for split_name, split_filename in expected.items():
            with self.subTest(split=split_name):
                config = load_config(
                    project_root / "configs" / f"{split_name}.yaml"
                )
                self.assertEqual(config.training.filename, split_filename)
                self.assertEqual(config.testing.filename, split_filename)
                self.assertEqual(config.training.seed, 1)
                self.assertEqual(config.metadata.filename, "five_sec_windows.csv")
                self.assertNotIn("extends", config)
                self.assertEqual(
                    resolve_split_path(project_root, config, split_filename),
                    project_root
                    / "runs"
                    / "icassp2027-uv-splits"
                    / "splits"
                    / split_filename,
                )

                enabled_families = [
                    family
                    for family, settings in config.splits.items()
                    if settings.file_name is not None
                ]
                self.assertEqual(enabled_families, [split_name])

    def test_config_cycles_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first = temp_path / "first.yaml"
            second = temp_path / "second.yaml"
            first.write_text("extends: second.yaml\n")
            second.write_text("extends: first.yaml\n")

            with self.assertRaisesRegex(ValueError, "cycle"):
                load_config(first)

    def test_main_and_baseline_configs_inherit_uv_defaults(self):
        project_root = Path(__file__).resolve().parents[1]
        base = load_config(project_root / "configs" / "base.yaml")
        cases = (
            ("all_splits.yaml", "icassp2027-uv-splits", "infonce"),
            (
                "infonce_paper.yaml",
                "icassp2027-uv-infonce-25split",
                "infonce",
            ),
            (
                "cosine_regression_paper.yaml",
                "cosine-regression-uv-25split",
                "cosine_regression",
            ),
            (
                "ridge_paper.yaml",
                "ridge-uv-25split",
                "infonce",
            ),
            (
                "table1_infonce_subject_on.yaml",
                "table1-infonce-subject-on",
                "infonce",
            ),
            (
                "table1_ridge_regression.yaml",
                "table1-ridge-regression",
                "infonce",
            ),
        )
        for name, run_name, objective in cases:
            with self.subTest(config=name):
                config = load_config(project_root / "configs" / name)
                self.assertEqual(config.run_name, run_name)
                self.assertEqual(config.metadata.filename, "five_sec_windows.csv")
                self.assertEqual(config.split_directory, "runs/icassp2027-uv-splits/splits")
                self.assertEqual(config.objective.type, objective)
                self.assertTrue(config.ablations.use_subject_layer)
                self.assertEqual(config.splits, base.splits)
                self.assertEqual(config.training.seed, base.training.seed)
                self.assertEqual(config.training.learning_rate, base.training.learning_rate)
                self.assertEqual(config.training.batch_size, base.training.batch_size)

    def test_table1_subject_layer_off_only_changes_name_and_ablation(self):
        project_root = Path(__file__).resolve().parents[1]
        subject_on = load_config(
            project_root / "configs" / "table1_infonce_subject_on.yaml"
        )
        subject_off = load_config(
            project_root / "configs" / "table1_infonce_subject_off.yaml"
        )

        self.assertTrue(subject_on.ablations.use_subject_layer)
        self.assertFalse(subject_off.ablations.use_subject_layer)
        subject_on.ablations.use_subject_layer = False
        subject_on.run_name = subject_off.run_name
        self.assertEqual(subject_on, subject_off)

    def test_cosine_config_only_changes_main_experiment_name_and_objective(self):
        project_root = Path(__file__).resolve().parents[1]
        main = load_config(project_root / "configs" / "infonce_paper.yaml")
        cosine = load_config(
            project_root / "configs" / "cosine_regression_paper.yaml"
        )

        self.assertEqual(main.objective.type, "infonce")
        self.assertEqual(cosine.objective.type, "cosine_regression")
        main.objective.type = cosine.objective.type
        main.run_name = cosine.run_name
        self.assertEqual(main, cosine)

    def test_table1_ridge_config_only_changes_ridge_experiment_name(self):
        project_root = Path(__file__).resolve().parents[1]
        ridge = load_config(project_root / "configs" / "ridge_paper.yaml")
        table1_ridge = load_config(
            project_root / "configs" / "table1_ridge_regression.yaml"
        )

        ridge.run_name = table1_ridge.run_name
        self.assertEqual(ridge, table1_ridge)

    def test_eeg2mel_config_is_standalone_and_uses_uv_inputs(self):
        project_root = Path(__file__).resolve().parents[1]
        config = OmegaConf.load(project_root / "configs" / "table1_eeg2mel.yaml")
        base = load_config(project_root / "configs" / "base.yaml")
        self.assertNotIn("extends", config)
        self.assertEqual(config.run_name, "eeg2mel_uv")
        self.assertEqual(config.metadata.filename, "five_sec_windows.csv")
        self.assertEqual(config.eeg2mel_baseline.mel_output_dir, "data/mel_targets/uv/5s")
        self.assertFalse(config.eeg2mel_baseline.evaluation.mert_space.enabled)
        self.assertEqual(config.eeg2mel_baseline.evaluation.batch_size, 16)
        self.assertEqual(config.splits, base.splits)
        self.assertEqual(config.training.seed, base.training.seed)


if __name__ == "__main__":
    unittest.main()
