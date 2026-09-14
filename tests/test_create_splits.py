import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.create_splits import (
    _created_split_paths,
    _parse_randomize_config,
    create_random_segment_out_split,
    create_within_song_split,
)


class RandomSegmentOutTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_path = Path(self.temp_dir.name)
        self.metadata_path = self.temp_path / "metadata.csv"

        rows = []
        for subject_id in (0, 1, 2):
            for song_id in (21, 22):
                for window_idx in range(12):
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "song_id": song_id,
                            "window_idx": window_idx,
                            "section_id": 0,
                        }
                    )
        pd.DataFrame(rows).to_csv(self.metadata_path, index=False)

    def _create_split(self, filename, seed, same_across_subjects):
        return create_random_segment_out_split(
            metadata_path=self.metadata_path,
            output_path=self.temp_path / filename,
            train_frac=0.5,
            val_frac=0.25,
            test_frac=0.25,
            same_across_subjects=same_across_subjects,
            random_seed=seed,
        )

    def test_shared_randomization_assigns_each_song_window_identically(self):
        split = self._create_split("shared.csv", seed=42, same_across_subjects=True)

        assignments = split.pivot(
            index=["song_id", "window_idx"],
            columns="subject_id",
            values="split",
        )
        for subject_id in assignments.columns[1:]:
            self.assertTrue(assignments.iloc[:, 0].equals(assignments[subject_id]))

        counts = split.groupby(["subject_id", "song_id", "split"]).size()
        for subject_id in (0, 1, 2):
            for song_id in (21, 22):
                self.assertEqual(counts[subject_id, song_id, "train"], 6)
                self.assertEqual(counts[subject_id, song_id, "val"], 3)
                self.assertEqual(counts[subject_id, song_id, "test"], 3)

    def test_random_seed_is_reproducible_in_both_randomization_modes(self):
        for same_across_subjects in (False, True):
            with self.subTest(same_across_subjects=same_across_subjects):
                first = self._create_split(
                    f"first-{same_across_subjects}.csv",
                    seed=17,
                    same_across_subjects=same_across_subjects,
                )
                second = self._create_split(
                    f"second-{same_across_subjects}.csv",
                    seed=17,
                    same_across_subjects=same_across_subjects,
                )
                different_seed = self._create_split(
                    f"different-{same_across_subjects}.csv",
                    seed=18,
                    same_across_subjects=same_across_subjects,
                )

                self.assertListEqual(first["split"].tolist(), second["split"].tolist())
                self.assertNotEqual(
                    first["split"].tolist(),
                    different_seed["split"].tolist(),
                )

    def test_legacy_boolean_randomize_config_remains_supported(self):
        self.assertEqual(_parse_randomize_config(True), (True, False, 0))
        self.assertEqual(
            _parse_randomize_config(
                {"enabled": True, "same_across_subjects": True, "seed": 123}
            ),
            (True, True, 123),
        )

    def test_created_split_paths_use_fold_suffixes_in_fold_mode(self):
        output_path = self.temp_path / "song_out.csv"

        self.assertEqual(_created_split_paths(output_path), [output_path])
        self.assertEqual(
            _created_split_paths(output_path, n_folds=3),
            [
                self.temp_path / "song_out_fold0.csv",
                self.temp_path / "song_out_fold1.csv",
                self.temp_path / "song_out_fold2.csv",
            ],
        )


if __name__ == "__main__":
    unittest.main()
