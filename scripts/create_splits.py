import argparse
import logging
import random
import sys
from math import isclose
from numbers import Integral, Real
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from omegaconf import ListConfig

from src.config import load_config, resolve_split_directory


def _load_metadata(metadata_path):
    metadata_path = Path(metadata_path)
    return pd.read_csv(metadata_path)


def _ensure_output_path(output_path, default_path):
    output_path = Path(output_path) if output_path is not None else Path(default_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _created_split_paths(output_path, n_folds=1):
    """Return the CSV paths produced by a split generator."""
    output_path = Path(output_path)
    if n_folds is not None and n_folds > 1:
        return [
            output_path.with_name(f"{output_path.stem}_fold{fold_idx}.csv")
            for fold_idx in range(n_folds)
        ]
    return [output_path]


def _normalize_song_ids(song_ids):
    return _normalize_ids(song_ids, "song_id")


def _normalize_ids(ids, id_name):
    if ids is None:
        normalized_ids = []
    elif isinstance(ids, (list, tuple, set, ListConfig)):
        normalized_ids = [int(value) for value in ids]
    else:
        normalized_ids = [int(ids)]

    if not normalized_ids:
        raise ValueError(f"At least one {id_name} must be provided.")

    return normalized_ids


def _resolve_split_ids(
    available_ids,
    train_ids,
    val_ids,
    test_ids,
    *,
    id_name,
    randomize=False,
    n_train=None,
    n_val=None,
    n_test=None,
    random_seed=0,
    rng=None,
):
    """Resolve explicit or randomized, mutually exclusive ID assignments."""
    available_ids = {int(value) for value in available_ids}

    if randomize:
        counts = {
            "train": n_train,
            "val": n_val,
            "test": n_test,
        }
        for split_name, count in counts.items():
            if not isinstance(count, Integral) or isinstance(count, bool) or count < 0:
                raise ValueError(
                    f"n_{split_name} for randomized {id_name} splits must be a "
                    f"non-negative integer, but received {count!r}."
                )

        n_requested = sum(counts.values())
        if n_requested > len(available_ids):
            raise ValueError(
                f"Randomized {id_name} splits request {n_requested} IDs, but only "
                f"{len(available_ids)} are available in the metadata."
            )

        rng = rng if rng is not None else random.Random(random_seed)
        selected_ids = rng.sample(sorted(available_ids), n_requested)
        train_end = counts["train"]
        val_end = train_end + counts["val"]
        return (
            selected_ids[:train_end],
            selected_ids[train_end:val_end],
            selected_ids[val_end:],
        )

    split_ids = (
        _normalize_ids(train_ids, id_name),
        _normalize_ids(val_ids, id_name),
        _normalize_ids(test_ids, id_name),
    )
    train_set, val_set, test_set = (set(ids) for ids in split_ids)
    overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    if overlap:
        raise ValueError(
            f"{id_name.capitalize()} splits must be disjoint, but found overlapping "
            f"IDs: {sorted(overlap)}"
        )

    requested_ids = train_set | val_set | test_set
    missing_ids = sorted(requested_ids - available_ids)
    if missing_ids:
        raise ValueError(
            f"Requested {id_name} values are missing from metadata: {missing_ids}"
        )

    return split_ids


def _round_robin_folds(available_ids, n_val, n_test, n_folds, *, random_seed=0, rng=None):
    """Partition available_ids into n_folds wrap-around (train, val, test) tuples.

    Shuffles the id pool once, then for each fold slides a fixed-size
    val+test window across that fixed order, wrapping past the end. The
    window's leading n_val slice is val, the rest of the window is test,
    everything outside the window is train. Unlike _resolve_split_ids's
    randomize branch, there is no unused pool: every id not in a fold's
    window is train for that fold, so n_val + n_test must be <= the number
    of available ids.
    """
    ids = sorted({int(value) for value in available_ids})
    n = len(ids)
    window = n_val + n_test
    if window > n:
        raise ValueError(
            f"Round-robin folds requested a val+test window of {window} ids, "
            f"but only {n} are available."
        )
    if not isinstance(n_folds, Integral) or isinstance(n_folds, bool) or n_folds < 1:
        raise ValueError(f"n_folds must be a positive integer, but received {n_folds!r}.")

    order = ids[:]
    rng = rng if rng is not None else random.Random(random_seed)
    rng.shuffle(order)

    step = n / n_folds
    folds = []
    for fold_idx in range(n_folds):
        start = round(fold_idx * step) % n
        val_ids, test_ids, in_window = [], [], set()
        for offset in range(window):
            pos = (start + offset) % n
            in_window.add(pos)
            (val_ids if offset < n_val else test_ids).append(order[pos])
        train_ids = [order[pos] for pos in range(n) if pos not in in_window]
        folds.append((train_ids, val_ids, test_ids))
    return folds


def _resolve_split_id_folds(
    available_ids,
    *,
    id_name,
    n_val,
    n_test,
    n_folds,
    random_seed=0,
    rng=None,
):
    """Round-robin fold analog of _resolve_split_ids's randomized branch.

    train has no explicit count here (it is always "everything not held out
    this fold"), so unlike _resolve_split_ids, callers do not pass n_train.
    """
    for count_name, count in (("n_val", n_val), ("n_test", n_test)):
        if not isinstance(count, Integral) or isinstance(count, bool) or count < 0:
            raise ValueError(
                f"{count_name} for round-robin {id_name} folds must be a "
                f"non-negative integer, but received {count!r}."
            )
    return _round_robin_folds(
        available_ids,
        n_val,
        n_test,
        n_folds,
        random_seed=random_seed,
        rng=rng,
    )


def _filter_metadata_by_song_ids(metadata, include_song_ids=None, exclude_song_ids=None):
    filtered = metadata.copy()

    if include_song_ids is not None:
        include_song_ids = _normalize_song_ids(include_song_ids)
        filtered = filtered[filtered["song_id"].isin(include_song_ids)].copy()

    if exclude_song_ids is not None:
        exclude_song_ids = _normalize_song_ids(exclude_song_ids)
        filtered = filtered[~filtered["song_id"].isin(exclude_song_ids)].copy()

    return filtered


def _validate_split_fractions(train_frac, val_frac, test_frac=None):
    if test_frac is None:
        test_frac = 1.0 - train_frac - val_frac

    fractions = {
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
    }
    for name, fraction in fractions.items():
        if (
            not isinstance(fraction, Real)
            or isinstance(fraction, bool)
            or not 0 <= fraction <= 1
        ):
            raise ValueError(
                f"{name} must be a number between 0 and 1, but received "
                f"{fraction!r}."
            )

    if not isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError(
            "train_frac, val_frac, and test_frac must sum to 1.0, but sum to "
            f"{sum(fractions.values()):g}."
        )

    return train_frac, val_frac, test_frac


def _split_labels(n_windows, train_frac, val_frac, test_frac=None):
    train_frac, val_frac, test_frac = _validate_split_fractions(
        train_frac,
        val_frac,
        test_frac,
    )
    n_train_windows = int(n_windows * train_frac)
    n_val_windows = int(n_windows * val_frac)
    n_test_windows = int(n_windows * test_frac)
    # Assign fractional rounding remainder to test so every window is retained.
    n_test_windows += n_windows - (
        n_train_windows + n_val_windows + n_test_windows
    )
    return (
        ["train"] * n_train_windows
        + ["val"] * n_val_windows
        + ["test"] * n_test_windows
    )


def _fold_window_counts(n_windows, train_frac, val_frac, test_frac):
    """Resolve (n_train, n_val, n_test) window counts for one song's fold windows."""
    train_frac, val_frac, test_frac = _validate_split_fractions(
        train_frac,
        val_frac,
        test_frac,
    )
    n_train = int(n_windows * train_frac)
    n_val = int(n_windows * val_frac)
    n_test = int(n_windows * test_frac)
    # Assign fractional rounding remainder to test so every window is retained.
    n_test += n_windows - (n_train + n_val + n_test)
    if n_val <= 0 or n_test <= 0:
        raise ValueError("Not enough windows for requested fractions.")
    return n_train, n_val, n_test


def _parse_randomize_config(randomize_config):
    if isinstance(randomize_config, bool):
        # Backwards compatibility with the original `randomize: true/false` config.
        return randomize_config, False, 0

    return (
        randomize_config.get("enabled", False),
        randomize_config.get("same_across_subjects", False),
        randomize_config.get("seed", 0),
    )


def _parse_partition_randomize_config(randomize_config):
    if isinstance(randomize_config, bool):
        return randomize_config, {}, 0

    return (
        randomize_config.get("enabled", False),
        randomize_config,
        randomize_config.get("seed", 0),
    )


def create_song_out_csv(
        metadata_path,
        output_path=None,
        train_song_id=(),
        val_song_ids=(),
        test_song_ids=(),
        randomize=False,
        n_train_songs=None,
        n_val_songs=None,
        n_test_songs=None,
        random_seed=0,
        n_folds=None,
):
    metadata_path = Path(metadata_path)
    output_path = _ensure_output_path(
        output_path,
        metadata_path.parent / "leave_song_out.csv",
    )

    metadata = _load_metadata(metadata_path)

    if n_folds is not None and n_folds > 1:
        if not randomize:
            raise ValueError("n_folds > 1 requires randomize.enabled=True for song_out.")
        fold_id_sets = _resolve_split_id_folds(
            metadata["song_id"].unique(),
            id_name="song_id",
            n_val=n_val_songs,
            n_test=n_test_songs,
            n_folds=n_folds,
            random_seed=random_seed,
        )
        fold_dfs = []
        for fold_idx, (train_ids, val_ids, test_ids) in enumerate(fold_id_sets):
            split_rows = []
            for split_name, song_ids in (
                ("train", train_ids),
                ("val", val_ids),
                ("test", test_ids),
            ):
                song_windows = metadata[metadata["song_id"].isin(song_ids)].copy()
                song_windows["split"] = split_name
                split_rows.append(song_windows)
            fold_df = pd.concat(split_rows, ignore_index=True)
            fold_df.to_csv(
                output_path.parent / f"{output_path.stem}_fold{fold_idx}.csv",
                index=False,
            )
            fold_dfs.append(fold_df)
        return fold_dfs

    train_song_ids, val_song_ids, test_song_ids = _resolve_split_ids(
        metadata["song_id"].unique(),
        train_song_id,
        val_song_ids,
        test_song_ids,
        id_name="song_id",
        randomize=randomize,
        n_train=n_train_songs,
        n_val=n_val_songs,
        n_test=n_test_songs,
        random_seed=random_seed,
    )

    split_rows = []
    for split_name, song_ids in (
        ("train", train_song_ids),
        ("val", val_song_ids),
        ("test", test_song_ids),
    ):
        song_windows = metadata[metadata["song_id"].isin(song_ids)].copy()
        song_windows["split"] = split_name
        split_rows.append(song_windows)

    if not split_rows:
        raise ValueError("No song rows were assigned to any split.")

    split_df = pd.concat(split_rows, ignore_index=True)
    split_df.to_csv(output_path, index=False)
    return split_df


def _create_segment_split(
    metadata_path,
    output_path=None,
    train_frac=0.70,
    val_frac=0.15,
    randomize=False,
    include_song_ids=None,
    exclude_song_ids=None,
    same_across_subjects=False,
    random_seed=0,
    test_frac=None,
):
    metadata_path = Path(metadata_path)
    output_path = _ensure_output_path(
        output_path,
        metadata_path.parent / "random_segment_out.csv",
    )

    metadata = _load_metadata(metadata_path)
    metadata = _filter_metadata_by_song_ids(
        metadata,
        include_song_ids=include_song_ids,
        exclude_song_ids=exclude_song_ids,
    )
    rng = random.Random(random_seed)

    shared_song_splits = {}
    if randomize and same_across_subjects:
        for song_id in sorted(metadata["song_id"].unique()):
            window_indices = sorted(
                metadata.loc[metadata["song_id"] == song_id, "window_idx"].unique()
            )
            split_labels = _split_labels(
                len(window_indices),
                train_frac,
                val_frac,
                test_frac,
            )
            rng.shuffle(split_labels)
            shared_song_splits[song_id] = dict(zip(window_indices, split_labels))

    split_rows = []

    for subject_id in sorted(metadata["subject_id"].unique()):
        subject_windows = metadata[metadata["subject_id"] == subject_id]

        for song_id in sorted(subject_windows["song_id"].unique()):
            song_windows = subject_windows[subject_windows["song_id"] == song_id].copy()
            song_windows = song_windows.sort_values("window_idx").reset_index(drop=True)

            n_windows = len(song_windows)
            if n_windows == 0:
                continue

            split_labels = _split_labels(
                n_windows,
                train_frac,
                val_frac,
                test_frac,
            )

            if randomize and same_across_subjects:
                song_windows["split"] = song_windows["window_idx"].map(
                    shared_song_splits[song_id]
                )
            else:
                if randomize:
                    rng.shuffle(split_labels)
                song_windows["split"] = split_labels
            split_rows.append(song_windows)

    if not split_rows:
        raise ValueError(f"No rows found in metadata file: {metadata_path}")

    split_df = pd.concat(split_rows, ignore_index=True)
    split_df.to_csv(output_path, index=False)
    return split_df


def create_random_segment_out_split(
    metadata_path,
    output_path=None,
    train_frac=0.70,
    val_frac=0.15,
    test_frac=None,
    same_across_subjects=False,
    random_seed=0,
    include_song_ids=None,
    exclude_song_ids=None,
    n_folds=None,
):
    """Create one reproducible split with randomized segments within each song."""
    if output_path is None:
        output_path = Path(metadata_path).parent / "random_segment_out.csv"
    output_path = Path(output_path)

    if n_folds is not None and n_folds > 1:
        if not same_across_subjects:
            raise ValueError(
                "random_segment_out fold mode always shares splits across "
                "subjects for the same song; set same_across_subjects to True "
                "(or omit it) when n_folds > 1."
            )
        return _create_segment_split_folds(
            metadata_path=metadata_path,
            output_dir=output_path.parent,
            file_stem=output_path.stem,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            n_folds=n_folds,
            random_seed=random_seed,
            include_song_ids=include_song_ids,
            exclude_song_ids=exclude_song_ids,
        )

    return _create_segment_split(
        metadata_path=metadata_path,
        output_path=output_path,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        randomize=True,
        same_across_subjects=same_across_subjects,
        random_seed=random_seed,
        include_song_ids=include_song_ids,
        exclude_song_ids=exclude_song_ids,
    )


def _create_segment_split_folds(
    metadata_path,
    output_dir,
    file_stem,
    train_frac=0.70,
    val_frac=0.10,
    test_frac=None,
    n_folds=5,
    random_seed=0,
    include_song_ids=None,
    exclude_song_ids=None,
):
    """Round-robin fold analog of _create_segment_split with same_across_subjects=True.

    Each song's window_idx pool is shuffled once (shared across every subject
    who heard that song), then a val+test window is round-robined across that
    fixed order per fold, wrapping past the end - the same mechanism as
    _round_robin_folds, applied per song instead of to a single id pool.
    """
    metadata = _load_metadata(metadata_path)
    metadata = _filter_metadata_by_song_ids(
        metadata,
        include_song_ids=include_song_ids,
        exclude_song_ids=exclude_song_ids,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(random_seed)

    song_fold_sets = {}
    for song_id in sorted(metadata["song_id"].unique()):
        window_indices = sorted(
            metadata.loc[metadata["song_id"] == song_id, "window_idx"].unique()
        )
        _, n_val, n_test = _fold_window_counts(
            len(window_indices),
            train_frac,
            val_frac,
            test_frac,
        )
        song_fold_sets[song_id] = _round_robin_folds(
            window_indices,
            n_val,
            n_test,
            n_folds,
            rng=rng,
        )

    fold_dfs = []
    for fold_idx in range(n_folds):
        shared_song_splits = {}
        for song_id, folds in song_fold_sets.items():
            train_windows, val_windows, test_windows = folds[fold_idx]
            labels = {}
            labels.update({w: "train" for w in train_windows})
            labels.update({w: "val" for w in val_windows})
            labels.update({w: "test" for w in test_windows})
            shared_song_splits[song_id] = labels

        split_rows = []
        for subject_id in sorted(metadata["subject_id"].unique()):
            subject_windows = metadata[metadata["subject_id"] == subject_id]

            for song_id in sorted(subject_windows["song_id"].unique()):
                song_windows = subject_windows[subject_windows["song_id"] == song_id].copy()
                song_windows = song_windows.sort_values("window_idx").reset_index(drop=True)
                if len(song_windows) == 0:
                    continue

                song_windows["split"] = song_windows["window_idx"].map(
                    shared_song_splits[song_id]
                )
                split_rows.append(song_windows)

        if not split_rows:
            raise ValueError(f"No rows found in metadata file: {metadata_path}")

        fold_df = pd.concat(split_rows, ignore_index=True)
        fold_df.to_csv(output_dir / f"{file_stem}_fold{fold_idx}.csv", index=False)
        fold_dfs.append(fold_df)

    return fold_dfs


def create_subject_out(
    metadata_path,
    output_path=None,
    train_subjects=(),
    val_subjects=(),
    test_subjects=(),
    randomize=False,
    n_train_subjects=None,
    n_val_subjects=None,
    n_test_subjects=None,
    random_seed=0,
    n_folds=None,
):
    metadata_path = Path(metadata_path)
    output_path = _ensure_output_path(
        output_path,
        metadata_path.parent / "leave_one_subject_out.csv",
    )

    metadata = _load_metadata(metadata_path)

    if n_folds is not None and n_folds > 1:
        if not randomize:
            raise ValueError("n_folds > 1 requires randomize.enabled=True for subject_out.")
        fold_id_sets = _resolve_split_id_folds(
            metadata["subject_id"].unique(),
            id_name="subject_id",
            n_val=n_val_subjects,
            n_test=n_test_subjects,
            n_folds=n_folds,
            random_seed=random_seed,
        )
        fold_dfs = []
        for fold_idx, (train_ids, val_ids, test_ids) in enumerate(fold_id_sets):
            split_rows = []
            for split_name, subject_ids in (
                ("train", train_ids),
                ("val", val_ids),
                ("test", test_ids),
            ):
                subject_windows = metadata[metadata["subject_id"].isin(subject_ids)].copy()
                subject_windows["split"] = split_name
                split_rows.append(subject_windows)
            fold_df = pd.concat(split_rows, ignore_index=True)
            fold_df.to_csv(
                output_path.parent / f"{output_path.stem}_fold{fold_idx}.csv",
                index=False,
            )
            fold_dfs.append(fold_df)
        return fold_dfs

    train_subjects, val_subjects, test_subjects = _resolve_split_ids(
        metadata["subject_id"].unique(),
        train_subjects,
        val_subjects,
        test_subjects,
        id_name="subject_id",
        randomize=randomize,
        n_train=n_train_subjects,
        n_val=n_val_subjects,
        n_test=n_test_subjects,
        random_seed=random_seed,
    )

    split_rows = []
    for subject_id in train_subjects:
        subject_windows = metadata[metadata["subject_id"] == subject_id].copy()
        subject_windows["split"] = ["train"] * len(subject_windows)
        split_rows.append(subject_windows)

    for subject_id in val_subjects:
        subject_windows = metadata[metadata["subject_id"] == subject_id].copy()
        subject_windows["split"] = ["val"] * len(subject_windows)
        split_rows.append(subject_windows)

    for subject_id in test_subjects:
        subject_windows = metadata[metadata["subject_id"] == subject_id].copy()
        subject_windows["split"] = ["test"] * len(subject_windows)
        split_rows.append(subject_windows)

    if not split_rows:
        raise ValueError("No subject rows were assigned to any LOSO split.")

    split_df = pd.concat(split_rows, ignore_index=True)
    split_df.to_csv(output_path, index=False)
    return split_df


def create_subject_song_out(
    metadata_path,
    output_path=None,
    train_song_ids=(),
    train_subject_ids=(),
    val_song_ids=(),
    val_subject_ids=(),
    test_song_ids=(),
    test_subject_ids=(),
    randomize=False,
    n_train_songs=None,
    n_train_subjects=None,
    n_val_songs=None,
    n_val_subjects=None,
    n_test_songs=None,
    n_test_subjects=None,
    random_seed=0,
    n_folds=None,
):
    """Create splits that hold out both songs and subjects simultaneously."""
    metadata_path = Path(metadata_path)
    output_path = _ensure_output_path(
        output_path,
        metadata_path.parent / "leave_subject_song_out.csv",
    )
    metadata = _load_metadata(metadata_path)

    rng = random.Random(random_seed)

    if n_folds is not None and n_folds > 1:
        if not randomize:
            raise ValueError("n_folds > 1 requires randomize.enabled=True for subject_song_out.")
        song_fold_sets = _resolve_split_id_folds(
            metadata["song_id"].unique(),
            id_name="song_id",
            n_val=n_val_songs,
            n_test=n_test_songs,
            n_folds=n_folds,
            rng=rng,
        )
        subject_fold_sets = _resolve_split_id_folds(
            metadata["subject_id"].unique(),
            id_name="subject_id",
            n_val=n_val_subjects,
            n_test=n_test_subjects,
            n_folds=n_folds,
            rng=rng,
        )
        fold_dfs = []
        for fold_idx, (song_splits_f, subject_splits_f) in enumerate(
            zip(song_fold_sets, subject_fold_sets)
        ):
            split_rows = []
            for split_name, song_ids, subject_ids in zip(
                ("train", "val", "test"), song_splits_f, subject_splits_f
            ):
                split_windows = metadata[
                    metadata["song_id"].isin(song_ids)
                    & metadata["subject_id"].isin(subject_ids)
                ].copy()
                if song_ids and subject_ids and split_windows.empty:
                    raise ValueError(
                        f"No metadata rows match the {split_name} song and "
                        f"subject assignments for fold {fold_idx}."
                    )
                split_windows["split"] = split_name
                split_rows.append(split_windows)
            fold_df = pd.concat(split_rows, ignore_index=True)
            if fold_df.empty:
                raise ValueError(
                    f"No rows were assigned to any subject-song split for fold {fold_idx}."
                )
            fold_df.to_csv(
                output_path.parent / f"{output_path.stem}_fold{fold_idx}.csv",
                index=False,
            )
            fold_dfs.append(fold_df)
        return fold_dfs

    song_splits = _resolve_split_ids(
        metadata["song_id"].unique(),
        train_song_ids,
        val_song_ids,
        test_song_ids,
        id_name="song_id",
        randomize=randomize,
        n_train=n_train_songs,
        n_val=n_val_songs,
        n_test=n_test_songs,
        rng=rng,
    )
    subject_splits = _resolve_split_ids(
        metadata["subject_id"].unique(),
        train_subject_ids,
        val_subject_ids,
        test_subject_ids,
        id_name="subject_id",
        randomize=randomize,
        n_train=n_train_subjects,
        n_val=n_val_subjects,
        n_test=n_test_subjects,
        rng=rng,
    )

    split_rows = []
    for split_name, song_ids, subject_ids in zip(
        ("train", "val", "test"), song_splits, subject_splits
    ):
        split_windows = metadata[
            metadata["song_id"].isin(song_ids)
            & metadata["subject_id"].isin(subject_ids)
        ].copy()
        if song_ids and subject_ids and split_windows.empty:
            raise ValueError(
                f"No metadata rows match the {split_name} song and subject assignments."
            )
        split_windows["split"] = split_name
        split_rows.append(split_windows)

    split_df = pd.concat(split_rows, ignore_index=True)
    if split_df.empty:
        raise ValueError("No rows were assigned to any subject-song split.")

    split_df.to_csv(output_path, index=False)
    return split_df


def create_chunk_out_folds(
    metadata_path,
    output_dir,
    train_frac=0.70,
    val_frac=0.10,
    n_folds=5,
    gap_windows=0,
    file_name="chunk_out.csv",
    include_song_ids=None,
    exclude_song_ids=None,
    test_frac=None,
    prefix=None,
):
    train_frac, val_frac, test_frac = _validate_split_fractions(
        train_frac,
        val_frac,
        test_frac,
    )
    metadata = _load_metadata(metadata_path)
    metadata = _filter_metadata_by_song_ids(
        metadata,
        include_song_ids=include_song_ids,
        exclude_song_ids=exclude_song_ids,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if prefix is not None:
        # Compatibility for callers of the former repeated-split function.
        fold_stem = str(prefix)
    else:
        file_name = Path(file_name).name
        if not file_name:
            raise ValueError("file_name must provide a base name for chunk-out folds.")
        if Path(file_name).suffix not in ("", ".csv"):
            raise ValueError("chunk_out.file_name must use a .csv extension.")
        fold_stem = Path(file_name).stem

    grouped = []
    for subject_id in sorted(metadata["subject_id"].unique()):
        subject_windows = metadata[metadata["subject_id"] == subject_id]
        for song_id in sorted(subject_windows["song_id"].unique()):
            song_windows = subject_windows[subject_windows["song_id"] == song_id].copy()
            song_windows = song_windows.sort_values("window_idx").reset_index(drop=True)
            if len(song_windows) > 0:
                grouped.append(((subject_id, song_id), song_windows))

    for fold_idx in range(n_folds):
        split_rows = []

        for (_, _), song_windows in grouped:
            n = len(song_windows)
            n_train = int(n * train_frac)
            n_val = int(n * val_frac)
            n_test = int(n * test_frac)
            # Assign fractional rounding remainder to test so every window is retained.
            n_test += n - (n_train + n_val + n_test)

            if n_test <= 0 or n_val <= 0:
                raise ValueError("Not enough windows for requested fractions.")

            max_start = n - n_val - n_test
            if n_folds == 1:
                val_start = n_train
            else:
                val_start = round(fold_idx * max_start / (n_folds - 1))

            test_start = val_start + n_val
            val_end = test_start
            test_end = test_start + n_test

            labels = ["train"] * n

            for i in range(val_start, val_end):
                labels[i] = "val"
            for i in range(test_start, test_end):
                labels[i] = "test"

            # Optional buffer: drop windows near val/test from training
            if gap_windows > 0:
                for i in range(max(0, val_start - gap_windows), min(n, test_end + gap_windows)):
                    if labels[i] == "train":
                        labels[i] = "gap"

            fold_df = song_windows.copy()
            fold_df["split"] = labels
            fold_df = fold_df[fold_df["split"] != "gap"].reset_index(drop=True)
            split_rows.append(fold_df)

        split_df = pd.concat(split_rows, ignore_index=True)
        split_df.to_csv(output_dir / f"{fold_stem}_fold{fold_idx}.csv", index=False)


# Legacy import aliases; new code should use the chunk/random-segment names above.
create_segment_out_split = _create_segment_split
create_within_song_split = _create_segment_split
create_repeated_segment_out_splits = create_chunk_out_folds
create_repeated_within_song_splits = create_chunk_out_folds

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create deterministic train/validation/test split CSVs."
    )
    parser.add_argument(
        "--config",
        default="configs/all_splits.yaml",
        help="Path to a paper experiment config.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    base_dir = PROJECT_ROOT
    cli_args = parse_args()
    args_path = Path(cli_args.config)
    if not args_path.is_absolute():
        args_path = base_dir / args_path
    args = load_config(args_path)
    chunk_out_config = args["splits"].get("chunk_out")
    random_segment_out_config = args["splits"].get("random_segment_out")
    chunk_out_fname = (
        chunk_out_config.get("file_name")
        if chunk_out_config is not None
        else None
    )
    random_segment_out_fname = (
        random_segment_out_config.get("file_name")
        if random_segment_out_config is not None
        else None
    )
    loso_fname = args["splits"]["subject_out"]["file_name"]
    song_fname = args["splits"]["song_out"]["file_name"]
    subject_song_config = args["splits"].get("subject_song_out")
    subject_song_fname = (
        subject_song_config.get("file_name")
        if subject_song_config is not None
        else None
    )

    metadata_path = base_dir / "data" / "metadata" / args["metadata"]["filename"]
    split_output_dir = resolve_split_directory(base_dir, args)
    random_segment_out_path = None
    if random_segment_out_fname is not None:
        random_segment_out_path = split_output_dir / random_segment_out_fname

    loso_path = None
    if loso_fname is not None:
        loso_path = split_output_dir / loso_fname

    song_path = None
    if song_fname is not None:
        song_path = split_output_dir / song_fname

    subject_song_path = None
    if subject_song_fname is not None:
        subject_song_path = (
            split_output_dir / subject_song_fname
        )

    # Leave-one-song-out split
    train_song_id = args["splits"]["song_out"].get("train_song_id")
    val_song_id = args["splits"]["song_out"].get("val_song_id")
    test_song_id = args["splits"]["song_out"].get("test_song_id")
    song_randomize, song_randomize_config, song_random_seed = (
        _parse_partition_randomize_config(
            args["splits"]["song_out"].get("randomize", False)
        )
    )

    gap_windows = args["retrieval"]["candidate_gap_size"]
    song_n_folds = song_randomize_config.get("n_folds", 1)

    if song_fname is not None:
        create_song_out_csv(
            metadata_path=metadata_path,
            output_path=song_path,
            train_song_id=train_song_id,
            val_song_ids=val_song_id,
            test_song_ids=test_song_id,
            randomize=song_randomize,
            n_train_songs=song_randomize_config.get("n_train_songs"),
            n_val_songs=song_randomize_config.get("n_val_songs"),
            n_test_songs=song_randomize_config.get("n_test_songs"),
            random_seed=song_random_seed,
            n_folds=song_n_folds,
        )
    # song_out only produces a single song_out.csv when unfolded; folded
    # song_out writes song_out_fold*.csv instead, so fall back to the raw
    # metadata (same rows, unless song_out is configured to filter out songs).
    segment_metadata_path = (
        song_path if song_path is not None and song_n_folds <= 1 else metadata_path
    )

    if chunk_out_fname is not None:
        create_chunk_out_folds(
            metadata_path=segment_metadata_path,
            output_dir=split_output_dir,
            file_name=chunk_out_fname,
            train_frac=chunk_out_config["train_frac"],
            val_frac=chunk_out_config["val_frac"],
            test_frac=chunk_out_config.get("test_frac"),
            n_folds=chunk_out_config["n_folds"],
            gap_windows=gap_windows,
            include_song_ids=chunk_out_config.get("include_song_ids"),
            exclude_song_ids=chunk_out_config.get("exclude_song_ids"),
        )
    if random_segment_out_fname is not None:
        random_segment_n_folds = random_segment_out_config.get("n_folds", 1)
        create_random_segment_out_split(
            metadata_path=segment_metadata_path,
            output_path=random_segment_out_path,
            train_frac=random_segment_out_config["train_frac"],
            val_frac=random_segment_out_config["val_frac"],
            test_frac=random_segment_out_config.get("test_frac"),
            same_across_subjects=random_segment_out_config.get(
                "same_across_subjects", False
            ),
            random_seed=random_segment_out_config.get("seed", 0),
            include_song_ids=random_segment_out_config.get("include_song_ids"),
            exclude_song_ids=random_segment_out_config.get("exclude_song_ids"),
            n_folds=random_segment_n_folds,
        )
    subject_out_config = args["splits"]["subject_out"]
    train_subjects = subject_out_config.get(
        "train_subject_ids", subject_out_config.get("train_subjects")
    )
    val_subjects = subject_out_config.get(
        "val_subject_ids", subject_out_config.get("val_subjects")
    )
    test_subjects = subject_out_config.get(
        "test_subject_ids", subject_out_config.get("test_subjects")
    )
    subject_randomize, subject_randomize_config, subject_random_seed = (
        _parse_partition_randomize_config(
            subject_out_config.get("randomize", False)
        )
    )

    subject_n_folds = subject_randomize_config.get("n_folds", 1)
    if loso_fname is not None:
        create_subject_out(
            metadata_path=metadata_path,
            output_path=loso_path,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            randomize=subject_randomize,
            n_train_subjects=subject_randomize_config.get("n_train_subjects"),
            n_val_subjects=subject_randomize_config.get("n_val_subjects"),
            n_test_subjects=subject_randomize_config.get("n_test_subjects"),
            random_seed=subject_random_seed,
            n_folds=subject_n_folds,
        )
    if subject_song_fname is not None:
        combined_randomize, combined_randomize_config, combined_random_seed = (
            _parse_partition_randomize_config(
                subject_song_config.get("randomize", False)
            )
        )
        combined_n_folds = combined_randomize_config.get("n_folds", 1)
        create_subject_song_out(
            metadata_path=metadata_path,
            output_path=subject_song_path,
            train_song_ids=subject_song_config.get("train_song_ids"),
            train_subject_ids=subject_song_config.get("train_subject_ids"),
            val_song_ids=subject_song_config.get("val_song_ids"),
            val_subject_ids=subject_song_config.get("val_subject_ids"),
            test_song_ids=subject_song_config.get("test_song_ids"),
            test_subject_ids=subject_song_config.get("test_subject_ids"),
            randomize=combined_randomize,
            n_train_songs=combined_randomize_config.get("n_train_songs"),
            n_train_subjects=combined_randomize_config.get("n_train_subjects"),
            n_val_songs=combined_randomize_config.get("n_val_songs"),
            n_val_subjects=combined_randomize_config.get("n_val_subjects"),
            n_test_songs=combined_randomize_config.get("n_test_songs"),
            n_test_subjects=combined_randomize_config.get("n_test_subjects"),
            random_seed=combined_random_seed,
            n_folds=combined_n_folds,
        )
