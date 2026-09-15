"""Core retrieval, permutation, shuffle-baseline, and song-search metrics used across evaluation scripts."""

from dataclasses import dataclass
import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.model.clip_loss import align_temporal_resolution, sequence_similarity_logits


RETRIEVAL_REGIMES = (
    "across_song",
    "within_song",
    "across_song_no_same_song_negatives",
)


@dataclass(frozen=True)
class CandidatePoolCache:
    """Dense, padded candidate pools and exact-match masks for one regime."""

    indices: torch.Tensor
    valid_mask: torch.Tensor
    exact_positive_mask: torch.Tensor


@dataclass(frozen=True)
class RetrievalEvaluationCache:
    """Reusable similarities and candidate-pool structure for retrieval evaluation."""

    similarities: torch.Tensor
    pools: dict
    song_ids: tuple
    window_idxs: tuple
    gap: int


@dataclass(frozen=True)
class SongSearchEvaluationCache:
    """Cached query-to-unique-window scores used by song search and its null."""

    similarities: torch.Tensor
    query_song_ids: tuple
    query_window_idxs: tuple
    candidate_song_ids: tuple
    candidate_window_idxs: tuple
    candidate_n_repeats: tuple


def precompute_similarity_matrix(eeg_emb, audio_emb):
    """Compute every EEG/audio cosine similarity once on CPU.

    Mixed temporal/vector embeddings are handled without expanding vectors
    across time.  For example, EEG shaped ``[N, D, T]`` and audio shaped
    ``[N, D]`` are reduced to two ``[N, D]`` tensors before matrix
    multiplication.  This is mathematically equivalent to expanding audio to
    ``[N, D, T]`` and computing flattened cosine similarity, without the
    multi-gigabyte expanded and normalized tensors.
    """
    eeg_emb = torch.as_tensor(
        eeg_emb, dtype=torch.float32, device="cpu"
    ).contiguous()
    audio_emb = torch.as_tensor(
        audio_emb, dtype=torch.float32, device="cpu"
    ).contiguous()

    if eeg_emb.ndim not in (2, 3) or audio_emb.ndim not in (2, 3):
        raise ValueError(
            "EEG and audio embeddings must be vectors [N, D] or "
            "sequences [N, D, T]."
        )
    if eeg_emb.shape[1] != audio_emb.shape[1]:
        raise ValueError(
            "EEG and audio feature dimensions must match, "
            f"got {eeg_emb.shape[1]} and {audio_emb.shape[1]}."
        )

    with torch.no_grad():
        if eeg_emb.ndim == 2 and audio_emb.ndim == 2:
            eeg_normalized = F.normalize(eeg_emb, dim=1)
            audio_normalized = F.normalize(audio_emb, dim=1)
            return eeg_normalized @ audio_normalized.T

        if eeg_emb.ndim == 3 and audio_emb.ndim == 2:
            time_steps = eeg_emb.shape[-1]
            eeg_norms = eeg_emb.norm(dim=(1, 2)).clamp_min(1e-8)
            expanded_audio_norms = (
                audio_emb.norm(dim=1) * math.sqrt(time_steps)
            ).clamp_min(1e-8)
            pooled_eeg = eeg_emb.sum(dim=-1) / eeg_norms.unsqueeze(1)
            scaled_audio = audio_emb / expanded_audio_norms.unsqueeze(1)
            return pooled_eeg @ scaled_audio.T

        if eeg_emb.ndim == 2 and audio_emb.ndim == 3:
            time_steps = audio_emb.shape[-1]
            expanded_eeg_norms = (
                eeg_emb.norm(dim=1) * math.sqrt(time_steps)
            ).clamp_min(1e-8)
            audio_norms = audio_emb.norm(dim=(1, 2)).clamp_min(1e-8)
            scaled_eeg = eeg_emb / expanded_eeg_norms.unsqueeze(1)
            pooled_audio = audio_emb.sum(dim=-1) / audio_norms.unsqueeze(1)
            return scaled_eeg @ pooled_audio.T

        if eeg_emb.shape[-1] != audio_emb.shape[-1]:
            audio_emb = F.interpolate(
                audio_emb,
                size=eeg_emb.shape[-1],
                mode="linear",
                align_corners=False,
            )

        eeg_flat = eeg_emb.reshape(eeg_emb.shape[0], -1)
        audio_flat = audio_emb.reshape(audio_emb.shape[0], -1)
        similarities = eeg_flat @ audio_flat.T
        eeg_norms = eeg_flat.norm(dim=1).clamp_min(1e-8)
        audio_norms = audio_flat.norm(dim=1).clamp_min(1e-8)
        return similarities / (eeg_norms.unsqueeze(1) * audio_norms.unsqueeze(0))


def precompute_negative_mse_matrix(predicted, target):
    """Compute pairwise negative-MSE scores for equally shaped representations.

    Retrieval helpers rank larger scores first, so returning ``-MSE`` makes
    the target with the lowest mean-squared error the top-ranked candidate.
    """
    predicted = torch.as_tensor(
        predicted, dtype=torch.float32, device="cpu"
    ).contiguous()
    target = torch.as_tensor(
        target, dtype=torch.float32, device="cpu"
    ).contiguous()

    if predicted.ndim < 2 or target.ndim < 2:
        raise ValueError("MSE retrieval inputs must include a batch dimension.")
    if tuple(predicted.shape[1:]) != tuple(target.shape[1:]):
        raise ValueError(
            "MSE retrieval inputs must have identical non-batch shapes, "
            f"got {tuple(predicted.shape[1:])} and {tuple(target.shape[1:])}."
        )

    predicted_flat = predicted.flatten(1)
    target_flat = target.flatten(1)
    n_features = predicted_flat.shape[1]
    if n_features == 0:
        raise ValueError("MSE retrieval inputs must contain at least one feature.")

    with torch.no_grad():
        squared_distances = (
            predicted_flat.square().sum(dim=1, keepdim=True)
            + target_flat.square().sum(dim=1).unsqueeze(0)
            - 2.0 * (predicted_flat @ target_flat.T)
        )
        squared_distances.clamp_min_(0.0)
        return -squared_distances / n_features


def _dense_candidate_pool(rows, song_ids, window_idxs, device):
    if not rows or any(not row for row in rows):
        raise ValueError("Candidate pool is empty for at least one query.")

    n_queries = len(rows)
    width = max(len(row) for row in rows)
    indices = torch.zeros((n_queries, width), dtype=torch.long, device=device)
    valid_mask = torch.zeros((n_queries, width), dtype=torch.bool, device=device)
    exact_positive_mask = torch.zeros(
        (n_queries, width), dtype=torch.bool, device=device
    )

    for query_idx, row in enumerate(rows):
        row_tensor = torch.as_tensor(row, dtype=torch.long, device=device)
        row_width = len(row)
        indices[query_idx, :row_width] = row_tensor
        valid_mask[query_idx, :row_width] = True
        exact_positive_mask[query_idx, :row_width] = torch.as_tensor(
            [
                song_ids[candidate_idx] == song_ids[query_idx]
                and window_idxs[candidate_idx] == window_idxs[query_idx]
                for candidate_idx in row
            ],
            dtype=torch.bool,
            device=device,
        )

    if not torch.all(exact_positive_mask.any(dim=1)):
        raise ValueError("No matching positive found in candidate pool.")

    return CandidatePoolCache(
        indices=indices,
        valid_mask=valid_mask,
        exact_positive_mask=exact_positive_mask,
    )


def build_retrieval_evaluation_cache(
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
        gap=0,
        score_metric="cosine",
):
    """Precompute scores and all candidate pools used during testing."""
    song_ids = tuple(int(x) for x in song_ids)
    window_idxs = tuple(int(x) for x in window_idxs)
    if len(song_ids) != len(window_idxs):
        raise ValueError("song_ids and window_idxs must have the same length.")
    if len(song_ids) != len(eeg_emb) or len(song_ids) != len(audio_emb):
        raise ValueError("Embeddings and retrieval metadata must have the same length.")

    if score_metric == "cosine":
        similarities = precompute_similarity_matrix(eeg_emb, audio_emb)
    elif score_metric == "negative_mse":
        similarities = precompute_negative_mse_matrix(eeg_emb, audio_emb)
    else:
        raise ValueError(
            "score_metric must be 'cosine' or 'negative_mse', "
            f"got {score_metric!r}."
        )
    if similarities.shape != (len(song_ids), len(song_ids)):
        raise ValueError("Expected a square EEG/audio similarity matrix.")

    unique_indices = []
    unique_indices_by_song = {}
    seen = set()
    for idx, (song_id, window_idx) in enumerate(zip(song_ids, window_idxs)):
        key = (song_id, window_idx)
        if key in seen:
            continue
        seen.add(key)
        unique_indices.append(idx)
        unique_indices_by_song.setdefault(song_id, []).append(idx)

    rows_by_regime = {regime: [] for regime in RETRIEVAL_REGIMES}
    for query_idx, (query_song_id, query_window_idx) in enumerate(
            zip(song_ids, window_idxs)
    ):
        rows_by_regime["across_song"].append(unique_indices)
        rows_by_regime["within_song"].append([
            candidate_idx
            for candidate_idx in unique_indices_by_song[query_song_id]
            if (
                window_idxs[candidate_idx] == query_window_idx
                or abs(window_idxs[candidate_idx] - query_window_idx) > gap
            )
        ])
        rows_by_regime["across_song_no_same_song_negatives"].append(
            [query_idx] + [
                candidate_idx
                for candidate_idx in unique_indices
                if song_ids[candidate_idx] != query_song_id
            ]
        )

    pools = {
        regime: _dense_candidate_pool(
            rows,
            song_ids=song_ids,
            window_idxs=window_idxs,
            device=similarities.device,
        )
        for regime, rows in rows_by_regime.items()
    }
    return RetrievalEvaluationCache(
        similarities=similarities,
        pools=pools,
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=int(gap),
    )


def _resolve_retrieval_cache(
        cache,
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
        gap,
):
    if cache is None:
        return build_retrieval_evaluation_cache(
            eeg_emb=eeg_emb,
            audio_emb=audio_emb,
            song_ids=song_ids,
            window_idxs=window_idxs,
            gap=gap,
        )

    normalized_song_ids = tuple(int(x) for x in song_ids)
    normalized_window_idxs = tuple(int(x) for x in window_idxs)
    if cache.song_ids != normalized_song_ids or cache.window_idxs != normalized_window_idxs:
        raise ValueError("Retrieval cache metadata does not match the requested evaluation.")
    if cache.gap != int(gap):
        raise ValueError("Retrieval cache gap does not match the requested evaluation.")
    return cache


def _score_cached_candidate_pool(
        cache,
        regime,
        ks,
        audio_permutation=None,
        eeg_permutation=None,
        positive_mask=None,
        query_mask=None,
):
    """Score every requested k from one cached score matrix per permutation."""
    ks = tuple(int(k) for k in ks)
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain at least one positive integer.")

    pool = cache.pools[regime]
    device = cache.similarities.device
    n_queries = cache.similarities.shape[0]
    row_indices = torch.arange(n_queries, dtype=torch.long, device=device)
    if eeg_permutation is not None:
        row_indices = torch.as_tensor(
            eeg_permutation, dtype=torch.long, device=device
        )

    candidate_indices = pool.indices
    if audio_permutation is not None:
        audio_permutation = torch.as_tensor(
            audio_permutation, dtype=torch.long, device=device
        )
        candidate_indices = audio_permutation[candidate_indices]

    scores = cache.similarities[row_indices[:, None], candidate_indices]
    scores = scores.masked_fill(~pool.valid_mask, float("-inf"))

    if positive_mask is None:
        positive_mask = pool.exact_positive_mask
    positive_mask = torch.as_tensor(positive_mask, dtype=torch.bool, device=device)

    if query_mask is None:
        query_mask = torch.ones(n_queries, dtype=torch.bool, device=device)
    else:
        query_mask = torch.as_tensor(query_mask, dtype=torch.bool, device=device)
    n_scored = int(query_mask.sum().item())
    if n_scored == 0:
        raise ValueError("No queries were eligible for retrieval scoring.")

    metrics = {}
    for k in ks:
        # Repeated audio windows create exact ties after permutation. Retain
        # the existing per-k selection definition instead of forcing smaller
        # k values to use a Top-max(k) prefix with different boundary ties.
        ranking_width = min(k, scores.shape[1])
        top_positions = torch.topk(
            scores,
            k=ranking_width,
            dim=1,
            sorted=True,
        ).indices
        hits = positive_mask.gather(1, top_positions).any(dim=1)
        metrics[k] = float(hits[query_mask].float().mean().item())
    return metrics


def _section_masks(cache, section_ids):
    section_ids = torch.as_tensor(
        [int(x) for x in section_ids],
        dtype=torch.long,
        device=cache.similarities.device,
    )
    if section_ids.numel() != cache.similarities.shape[0]:
        raise ValueError("section_ids must match the number of embeddings.")

    pool = cache.pools["within_song"]
    query_mask = section_ids != -1
    positive_mask = (
        pool.valid_mask
        & query_mask[:, None]
        & (section_ids[pool.indices] == section_ids[:, None])
    )
    return positive_mask, query_mask


def _deduplicate_candidate_indices(candidate_indices, song_ids, window_idxs):
    deduped = []
    seen = set()

    for j in candidate_indices:
        key = (song_ids[j], window_idxs[j])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(j)

    return deduped


def _candidate_indices_for_query(i, song_ids, window_idxs, regime="across_song", gap=0):
    n = len(song_ids)
    all_indices = list(range(n))

    if regime == "across_song":
        candidate_indices = all_indices
    elif regime == "within_song":
        candidate_indices = [
            j for j in all_indices
            if song_ids[j] == song_ids[i]
            and (
                window_idxs[j] == window_idxs[i]
                or abs(window_idxs[j] - window_idxs[i]) > gap
            )
        ]
    elif regime == "across_song_no_same_song_negatives":
        candidate_indices = [i] + [j for j in all_indices if song_ids[j] != song_ids[i]]
    else:
        raise ValueError(f"Unknown regime: {regime}")

    return _deduplicate_candidate_indices(
        candidate_indices,
        song_ids,
        window_idxs,
    )


def _shuffle_audio_within_song(audio_emb, song_ids, rng):
    shuffled_audio = torch.as_tensor(audio_emb, dtype=torch.float32).clone()
    song_ids = [int(x) for x in song_ids]

    song_to_indices = {}
    for idx, song_id in enumerate(song_ids):
        song_to_indices.setdefault(song_id, []).append(idx)

    for indices in song_to_indices.values():
        permuted_indices = rng.permutation(indices)
        shuffled_audio[indices] = shuffled_audio[permuted_indices]

    return shuffled_audio


def _shuffle_eeg_within_song(eeg_emb, song_ids, rng):
    shuffled_eeg = torch.as_tensor(eeg_emb, dtype=torch.float32).clone()
    song_ids = [int(x) for x in song_ids]

    song_to_indices = {}
    for idx, song_id in enumerate(song_ids):
        song_to_indices.setdefault(song_id, []).append(idx)

    for indices in song_to_indices.values():
        permuted_indices = rng.permutation(indices)
        shuffled_eeg[indices] = shuffled_eeg[permuted_indices]

    return shuffled_eeg


def _unique_song_window_entries(audio_emb, song_ids, window_idxs):
    audio_emb = torch.as_tensor(audio_emb, dtype=torch.float32)
    song_ids = [int(x) for x in song_ids]
    window_idxs = [int(x) for x in window_idxs]

    grouped = {}
    for idx, (song_id, window_idx) in enumerate(zip(song_ids, window_idxs)):
        song_windows = grouped.setdefault(song_id, {})
        song_windows.setdefault(window_idx, []).append(audio_emb[idx])

    unique_entries = {}
    for song_id, song_windows in grouped.items():
        entries = []
        for window_idx in sorted(song_windows):
            stacked = torch.stack(song_windows[window_idx], dim=0)
            entries.append(
                {
                    "window_idx": window_idx,
                    "audio_emb": stacked.mean(dim=0),
                    "n_repeats": stacked.shape[0],
                }
            )
        unique_entries[song_id] = entries

    return unique_entries


def build_song_search_evaluation_cache(
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
):
    """Compute EEG-to-unique-audio-window similarities once for song search.

    Repeated audio embeddings for the same ``(song_id, window_idx)`` are
    averaged exactly as in :func:`similarity_scores_grouped_by_song` before the
    score matrix is built.
    """
    query_song_ids = tuple(int(x) for x in song_ids)
    query_window_idxs = tuple(int(x) for x in window_idxs)
    if len(query_song_ids) != len(query_window_idxs):
        raise ValueError("song_ids and window_idxs must have the same length.")
    if len(query_song_ids) != len(eeg_emb) or len(query_song_ids) != len(audio_emb):
        raise ValueError("Embeddings and song-search metadata must have the same length.")

    unique_entries = _unique_song_window_entries(
        audio_emb,
        query_song_ids,
        query_window_idxs,
    )
    candidate_embeddings = []
    candidate_song_ids = []
    candidate_window_idxs = []
    candidate_n_repeats = []
    for song_id, entries in unique_entries.items():
        for entry in entries:
            candidate_embeddings.append(entry["audio_emb"])
            candidate_song_ids.append(int(song_id))
            candidate_window_idxs.append(int(entry["window_idx"]))
            candidate_n_repeats.append(int(entry["n_repeats"]))

    if not candidate_embeddings:
        raise ValueError("No unique audio windows were available for song search.")

    unique_audio_emb = torch.stack(candidate_embeddings, dim=0)
    similarities = precompute_similarity_matrix(eeg_emb, unique_audio_emb)
    return SongSearchEvaluationCache(
        similarities=similarities,
        query_song_ids=query_song_ids,
        query_window_idxs=query_window_idxs,
        candidate_song_ids=tuple(candidate_song_ids),
        candidate_window_idxs=tuple(candidate_window_idxs),
        candidate_n_repeats=tuple(candidate_n_repeats),
    )


def _resolve_song_search_cache(
        cache,
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
):
    if cache is None:
        return build_song_search_evaluation_cache(
            eeg_emb=eeg_emb,
            audio_emb=audio_emb,
            song_ids=song_ids,
            window_idxs=window_idxs,
        )

    normalized_song_ids = tuple(int(x) for x in song_ids)
    normalized_window_idxs = tuple(int(x) for x in window_idxs)
    if cache.query_song_ids != normalized_song_ids:
        raise ValueError("Song-search cache song IDs do not match the requested evaluation.")
    if cache.query_window_idxs != normalized_window_idxs:
        raise ValueError("Song-search cache window indices do not match the requested evaluation.")
    return cache


def _song_search_layout(cache):
    candidate_songs = tuple(dict.fromkeys(cache.candidate_song_ids))
    song_to_position = {
        song_id: position for position, song_id in enumerate(candidate_songs)
    }
    candidate_positions_by_song = [
        torch.as_tensor(
            [
                position
                for position, candidate_song_id in enumerate(cache.candidate_song_ids)
                if candidate_song_id == song_id
            ],
            dtype=torch.long,
            device=cache.similarities.device,
        )
        for song_id in candidate_songs
    ]
    try:
        query_song_positions = torch.as_tensor(
            [song_to_position[song_id] for song_id in cache.query_song_ids],
            dtype=torch.long,
            device=cache.similarities.device,
        )
    except KeyError as exc:
        raise ValueError(f"Query song {exc.args[0]} has no audio candidates.") from exc
    return candidate_songs, candidate_positions_by_song, query_song_positions


def _song_search_top1_accuracy(cache):
    _, candidate_positions_by_song, query_song_positions = _song_search_layout(cache)
    per_song_scores = torch.stack(
        [
            cache.similarities.index_select(1, positions).amax(dim=1)
            for positions in candidate_positions_by_song
        ],
        dim=1,
    )
    predicted_song_positions = per_song_scores.argmax(dim=1)
    hits = predicted_song_positions == query_song_positions
    n_correct = int(hits.sum().item())
    return n_correct / len(cache.query_song_ids), n_correct


def _song_search_marginal_log_scores(
        similarities,
        candidate_positions_by_song,
        temperature=0.07,
):
    """Marginalize window evidence with an equal prior over candidate songs.

    Each song score is ``logmeanexp(similarity / temperature)`` over its unique
    audio windows. This is equivalent, for ranking, to summing window softmax
    probabilities within a song and correcting for unequal window counts.
    """
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Song-search marginal temperature must be finite and positive.")

    return torch.stack(
        [
            torch.logsumexp(
                similarities.index_select(-1, positions) / temperature,
                dim=-1,
            )
            - math.log(positions.numel())
            for positions in candidate_positions_by_song
        ],
        dim=-1,
    )


def _song_search_marginal_top1_accuracy(cache, temperature=0.07):
    """Return top-1 song accuracy after marginalizing unique audio windows."""
    _, candidate_positions_by_song, query_song_positions = _song_search_layout(cache)
    per_song_scores = _song_search_marginal_log_scores(
        cache.similarities,
        candidate_positions_by_song,
        temperature=temperature,
    )
    predicted_song_positions = per_song_scores.argmax(dim=1)
    hits = predicted_song_positions == query_song_positions
    n_correct = int(hits.sum().item())
    return n_correct / len(cache.query_song_ids), n_correct


def _empirical_tail_p_values(null_values, observed):
    """Return corrected one-sided upper- and lower-tail empirical p-values."""
    null_values = np.asarray(null_values)
    if null_values.size == 0:
        raise ValueError("At least one null value is required.")

    denominator = null_values.size + 1
    return (
        float((1 + np.sum(null_values >= observed)) / denominator),
        float((1 + np.sum(null_values <= observed)) / denominator),
    )


def evaluate_song_search_permutation_test(
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
        n_perm=1000,
        rng=0,
        permutation_batch_size=16,
        cache=None,
        marginal_temperature=0.07,
):
    """Evaluate max-window and marginalized song top-1 permutation nulls.

    The EEG-to-unique-audio-window similarity matrix is computed once. For
    each permutation, its audio columns are reassigned to the fixed candidate
    song/window slots. Both the existing per-song maximum and the
    length-normalized probability marginal are recomputed. Permutations are
    processed in bounded batches to avoid materializing the full
    ``n_perm x n_queries x n_candidates`` tensor.
    """
    n_perm = int(n_perm)
    permutation_batch_size = int(permutation_batch_size)
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1.")
    if permutation_batch_size < 1:
        raise ValueError("permutation_batch_size must be at least 1.")

    cache = _resolve_song_search_cache(
        cache=cache,
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
    )
    candidate_songs, candidate_positions_by_song, query_song_positions = (
        _song_search_layout(cache)
    )
    observed_accuracy, observed_correct = _song_search_top1_accuracy(cache)
    marginal_accuracy, marginal_correct = _song_search_marginal_top1_accuracy(
        cache,
        temperature=marginal_temperature,
    )
    n_candidates = cache.similarities.shape[1]
    rng = np.random.default_rng(rng)
    null_correct_counts = []
    marginal_null_correct_counts = []
    completed = 0

    with torch.no_grad():
        while completed < n_perm:
            current_batch_size = min(permutation_batch_size, n_perm - completed)
            permutations = np.stack(
                [rng.permutation(n_candidates) for _ in range(current_batch_size)],
                axis=0,
            )
            permutation_tensor = torch.as_tensor(
                permutations,
                dtype=torch.long,
                device=cache.similarities.device,
            )
            max_score_parts = []
            marginal_score_parts = []
            for positions in candidate_positions_by_song:
                permuted_scores = cache.similarities[
                    :,
                    permutation_tensor.index_select(1, positions),
                ]
                max_score_parts.append(permuted_scores.amax(dim=2).T)
                marginal_score_parts.append(
                    torch.logsumexp(
                        permuted_scores / float(marginal_temperature),
                        dim=2,
                    ).T
                    - math.log(positions.numel())
                )

            per_song_scores = torch.stack(max_score_parts, dim=2)
            predicted_song_positions = per_song_scores.argmax(dim=2)
            batch_correct_counts = (
                predicted_song_positions == query_song_positions.unsqueeze(0)
            ).sum(dim=1)
            null_correct_counts.extend(batch_correct_counts.cpu().tolist())

            marginal_per_song_scores = torch.stack(
                marginal_score_parts,
                dim=2,
            )
            marginal_predicted_positions = marginal_per_song_scores.argmax(dim=2)
            marginal_batch_correct_counts = (
                marginal_predicted_positions == query_song_positions.unsqueeze(0)
            ).sum(dim=1)
            marginal_null_correct_counts.extend(
                marginal_batch_correct_counts.cpu().tolist()
            )

            previous_completed = completed
            completed += current_batch_size
            if completed == n_perm or completed // 50 > previous_completed // 50:
                logging.info(
                    "Song-search permutation progress: %d/%d",
                    completed,
                    n_perm,
                )

    null_correct_counts = np.asarray(null_correct_counts, dtype=np.int64)
    null_scores = null_correct_counts / len(cache.query_song_ids)
    marginal_null_correct_counts = np.asarray(
        marginal_null_correct_counts,
        dtype=np.int64,
    )
    marginal_null_scores = (
        marginal_null_correct_counts / len(cache.query_song_ids)
    )
    analytic_chance = 1.0 / len(candidate_songs)
    p_upper, p_lower = _empirical_tail_p_values(
        null_correct_counts,
        observed_correct,
    )
    marginal_p_upper, marginal_p_lower = _empirical_tail_p_values(
        marginal_null_correct_counts,
        marginal_correct,
    )
    return {
        "song_search_n_queries": len(cache.query_song_ids),
        "song_search_n_candidate_songs": len(candidate_songs),
        "song_search_song_top1_n_correct": observed_correct,
        "song_search_song_top1_permutation_observed": observed_accuracy,
        "song_search_song_top1_chance": float(analytic_chance),
        "song_search_song_top1_minus_chance": float(
            observed_accuracy - analytic_chance
        ),
        "song_search_song_top1_null_mean": float(null_scores.mean()),
        "song_search_song_top1_null_std": float(null_scores.std(ddof=0)),
        "song_search_song_top1_minus_null": float(
            observed_accuracy - null_scores.mean()
        ),
        "song_search_song_top1_p": p_upper,
        "song_search_song_top1_p_lower": p_lower,
        "song_search_song_top1_n_perm": n_perm,
        "song_search_marginal_temperature": float(marginal_temperature),
        "song_search_marginal_top1_n_correct": marginal_correct,
        "song_search_marginal_top1_permutation_observed": marginal_accuracy,
        "song_search_marginal_top1_chance": float(analytic_chance),
        "song_search_marginal_top1_minus_chance": float(
            marginal_accuracy - analytic_chance
        ),
        "song_search_marginal_top1_null_mean": float(
            marginal_null_scores.mean()
        ),
        "song_search_marginal_top1_null_std": float(
            marginal_null_scores.std(ddof=0)
        ),
        "song_search_marginal_top1_minus_null": float(
            marginal_accuracy - marginal_null_scores.mean()
        ),
        "song_search_marginal_top1_p": marginal_p_upper,
        "song_search_marginal_top1_p_lower": marginal_p_lower,
        "song_search_marginal_top1_n_perm": n_perm,
    }


def topk_accuracy_from_logits(logits, labels, k=10):
    """
    logits: [B, B]
    labels: [B]
    """
    k = min(k, logits.shape[1])
    top_k_preds = logits.topk(k, dim=1).indices
    correct = top_k_preds.eq(labels.unsqueeze(1))
    top_k_acc = correct.any(dim=1).float().mean()
    return top_k_acc


def topk_retrieval_accuracy(eeg_emb, audio_emb, k=10, bidirectional=False):
    """
    Args:
        eeg_emb:   [B, D]
        audio_emb: [B, D]
        k: Top-K accuracy
        bidirectional: Determines if top-k accuracy is for both
        EEG -> Audio and Audio -> EEG.
    Returns:
        Top-k retrieval accuracy from EEG -> Audio.
        If bidirectional, top-k accuracy from EEG <-> Audio.
    """
    eeg_emb = torch.as_tensor(eeg_emb, dtype=torch.float32)
    audio_emb = torch.as_tensor(audio_emb, dtype=torch.float32)

    logits = sequence_similarity_logits(eeg_emb, audio_emb)
    targets = torch.arange(logits.shape[0], device=logits.device)

    eeg_to_audio_topk = topk_accuracy_from_logits(logits, targets, k)
    audio_to_eeg_topk = topk_accuracy_from_logits(logits.T, targets, k)

    if bidirectional:
        return (eeg_to_audio_topk + audio_to_eeg_topk) / 2
    return eeg_to_audio_topk


def retrieval_metric(eeg_emb, audio_emb, k=1, bidirectional=True):
    score = topk_retrieval_accuracy(
        eeg_emb,
        audio_emb,
        k=k,
        bidirectional=bidirectional,
    )
    return float(score.item())


def permutation_null(
        eeg_emb,
        audio_emb,
        metric_fn,
        n_perm=1000,
        rng=None,
        regime=None,
        k=None,
):
    rng = np.random.default_rng(rng)
    scores = []

    for perm_idx in range(n_perm):
        perm = rng.permutation(len(audio_emb))
        shuffled_audio = audio_emb[perm]
        score = metric_fn(eeg_emb, shuffled_audio)
        scores.append(score)

        if (perm_idx + 1) % 50 == 0 or perm_idx + 1 == n_perm:
            if regime is not None and k is not None:
                logging.info(
                    "Permutation progress [%s top%d]: %d/%d",
                    regime,
                    k,
                    perm_idx + 1,
                    n_perm,
                )
            else:
                logging.info("Permutation progress: %d/%d", perm_idx + 1, n_perm)

    return np.array(scores)


def compute_p_value(
        eeg_emb,
        audio_emb,
        k,
        window_idxs,
        song_ids,
        regime,
        gap=0,
        n_perm=1000,
        cache=None,
):
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1.")
    cache = _resolve_retrieval_cache(
        cache=cache,
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=gap,
    )
    observed_topk = _score_cached_candidate_pool(cache, regime, (k,))[k]

    rng = np.random.default_rng(0)
    null_topk = []
    for perm_idx in range(n_perm):
        permutation = rng.permutation(len(audio_emb))
        null_topk.append(
            _score_cached_candidate_pool(
                cache,
                regime,
                (k,),
                audio_permutation=permutation,
            )[k]
        )
        if (perm_idx + 1) % 50 == 0 or perm_idx + 1 == n_perm:
            logging.info(
                "Permutation progress [%s top%d]: %d/%d",
                regime,
                k,
                perm_idx + 1,
                n_perm,
            )

    null_topk = np.asarray(null_topk, dtype=np.float64)
    topk_p = (1 + np.sum(null_topk >= observed_topk)) / (len(null_topk) + 1)
    return topk_p


def retrieval_with_candidate_pool(
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
        k=10,
        regime="across_song",
        gap=0,
):
    n = len(eeg_emb)
    correct = 0

    eeg_emb = torch.as_tensor(eeg_emb, dtype=torch.float32)
    audio_emb = torch.as_tensor(audio_emb, dtype=torch.float32)
    song_ids = [int(x) for x in song_ids]
    window_idxs = [int(x) for x in window_idxs]

    for i in range(n):
        candidate_indices = _candidate_indices_for_query(
            i,
            song_ids,
            window_idxs,
            regime=regime,
            gap=gap,
        )
        candidate_audio = audio_emb[candidate_indices]
        query_eeg = eeg_emb[i].unsqueeze(0)

        similarities = sequence_similarity_logits(query_eeg, candidate_audio).squeeze(0)
        effective_k = min(k, len(candidate_indices))
        topk_positions = torch.topk(similarities, k=effective_k).indices.tolist()

        correct_positions = [
            pos for pos, j in enumerate(candidate_indices)
            if song_ids[j] == song_ids[i] and window_idxs[j] == window_idxs[i]
        ]

        if any(pos in topk_positions for pos in correct_positions):
            correct += 1

    return correct / n


def similarity_scores_grouped_by_song(
        eeg_query,
        audio_emb,
        song_ids,
        window_idxs,
):
    eeg_query = torch.as_tensor(eeg_query, dtype=torch.float32)
    if eeg_query.ndim == 1:
        eeg_query = eeg_query.reshape(1, -1)
    elif eeg_query.ndim == 2:
        eeg_query = eeg_query.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected EEG query shape: {tuple(eeg_query.shape)}")

    unique_entries = _unique_song_window_entries(audio_emb, song_ids, window_idxs)
    deduped_scores = {}

    for song_id, song_windows in unique_entries.items():
        entries = []
        for song_window in song_windows:
            window_idx = song_window["window_idx"]
            audio_vector = torch.as_tensor(song_window["audio_emb"], dtype=torch.float32)
            if audio_vector.ndim == 1:
                audio_vector = audio_vector.reshape(1, -1)
            elif audio_vector.ndim == 2:
                audio_vector = audio_vector.unsqueeze(0)
            else:
                raise ValueError(
                    f"Unexpected audio candidate shape: {tuple(audio_vector.shape)}"
                )
            score = float(sequence_similarity_logits(eeg_query, audio_vector).squeeze().item())
            entries.append(
                {
                    "window_idx": window_idx,
                    "score": score,
                    "n_repeats": song_window["n_repeats"],
                }
            )
        deduped_scores[song_id] = entries

    return deduped_scores


def similarity_scores_for_query_index(
        query_index,
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
):
    return similarity_scores_grouped_by_song(
        eeg_query=eeg_emb[query_index],
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
    )


def evaluate_song_search(
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
        subject_ids=None,
        top_k_true_song_windows=3,
        cache=None,
        marginal_temperature=0.07,
):
    eeg_emb = torch.as_tensor(eeg_emb, dtype=torch.float32)
    song_ids = [int(x) for x in song_ids]
    window_idxs = [int(x) for x in window_idxs]
    if subject_ids is not None:
        subject_ids = [int(x) for x in subject_ids]

    cache = _resolve_song_search_cache(
        cache=cache,
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
    )
    candidate_songs, candidate_positions_by_song, _ = _song_search_layout(cache)
    marginal_log_scores = _song_search_marginal_log_scores(
        cache.similarities,
        candidate_positions_by_song,
        temperature=marginal_temperature,
    )
    marginal_probabilities = torch.softmax(marginal_log_scores, dim=1)
    results = []

    for query_index in range(len(eeg_emb)):
        query_song_id = song_ids[query_index]
        query_window_idx = window_idxs[query_index]

        per_song_summary = []
        for song_position, (candidate_song_id, positions) in enumerate(zip(
                candidate_songs,
                candidate_positions_by_song,
        )):
            entries = [
                {
                    "window_idx": cache.candidate_window_idxs[position],
                    "score": float(cache.similarities[query_index, position].item()),
                    "n_repeats": cache.candidate_n_repeats[position],
                }
                for position in positions.tolist()
            ]
            best_entry = max(entries, key=lambda item: item["score"])
            top_true_entries = sorted(
                entries,
                key=lambda item: item["score"],
                reverse=True,
            )[:top_k_true_song_windows]
            per_song_summary.append(
                {
                    "song_id": candidate_song_id,
                    "best_window_idx": best_entry["window_idx"],
                    "best_score": best_entry["score"],
                    "marginal_log_score": float(
                        marginal_log_scores[query_index, song_position].item()
                    ),
                    "marginal_probability": float(
                        marginal_probabilities[query_index, song_position].item()
                    ),
                    "top_windows": top_true_entries,
                }
            )

        ranked_songs = sorted(
            per_song_summary,
            key=lambda item: item["best_score"],
            reverse=True,
        )
        predicted_song_id = ranked_songs[0]["song_id"]
        marginal_ranked_songs = sorted(
            per_song_summary,
            key=lambda item: item["marginal_log_score"],
            reverse=True,
        )
        marginal_predicted_song_id = marginal_ranked_songs[0]["song_id"]
        true_song_summary = next(
            item for item in per_song_summary if item["song_id"] == query_song_id
        )
        localization_error = abs(
            true_song_summary["best_window_idx"] - query_window_idx
        )

        result = {
            "query_index": query_index,
            "true_song_id": query_song_id,
            "true_window_idx": query_window_idx,
            "predicted_song_id": predicted_song_id,
            "song_correct": predicted_song_id == query_song_id,
            "marginal_predicted_song_id": marginal_predicted_song_id,
            "marginal_song_correct": marginal_predicted_song_id == query_song_id,
            "marginal_temperature": float(marginal_temperature),
            "true_song_best_window_idx": true_song_summary["best_window_idx"],
            "true_song_best_score": true_song_summary["best_score"],
            "true_song_marginal_log_score": true_song_summary[
                "marginal_log_score"
            ],
            "true_song_marginal_probability": true_song_summary[
                "marginal_probability"
            ],
            "localization_error": localization_error,
            "true_song_top_windows": true_song_summary["top_windows"],
            "ranked_songs": ranked_songs,
            "marginal_ranked_songs": marginal_ranked_songs,
        }
        if subject_ids is not None:
            result["subject_id"] = subject_ids[query_index]

        results.append(result)

    return results


def evaluate_section_search(
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
        section_ids,
        subject_ids=None,
        gap=0,
        cache=None,
):
    """Per-query Tier 2 detail, analogous to evaluate_song_search but scoped
    within a song rather than ranking across songs: for each query with a
    valid section_id (boundary windows, section_id == -1, are skipped, same
    exclusion as retrieval_with_candidate_pool_section), finds the single
    best-matching candidate in the same-song pool and records whether its
    section_id matches. Feeds the per-song section confusion matrix
    and the section-search result CSV
    written by test.py.
    """
    eeg_emb = torch.as_tensor(eeg_emb, dtype=torch.float32)
    audio_emb = torch.as_tensor(audio_emb, dtype=torch.float32)
    song_ids = [int(x) for x in song_ids]
    window_idxs = [int(x) for x in window_idxs]
    section_ids = [int(x) for x in section_ids]
    if subject_ids is not None:
        subject_ids = [int(x) for x in subject_ids]
    if cache is not None:
        cache = _resolve_retrieval_cache(
            cache=cache,
            eeg_emb=eeg_emb,
            audio_emb=audio_emb,
            song_ids=song_ids,
            window_idxs=window_idxs,
            gap=gap,
        )

    results = []

    for query_index in range(len(eeg_emb)):
        if section_ids[query_index] == -1:
            continue

        if cache is None:
            candidate_indices = _candidate_indices_for_query(
                query_index,
                song_ids,
                window_idxs,
                regime="within_song",
                gap=gap,
            )
            similarities = sequence_similarity_logits(
                eeg_emb[query_index].unsqueeze(0),
                audio_emb[candidate_indices],
            ).squeeze(0)
        else:
            pool = cache.pools["within_song"]
            valid = pool.valid_mask[query_index]
            candidate_tensor = pool.indices[query_index, valid]
            candidate_indices = candidate_tensor.tolist()
            similarities = cache.similarities[query_index, candidate_tensor]
        best_pos = int(torch.argmax(similarities).item())
        best_j = candidate_indices[best_pos]

        result = {
            "query_index": query_index,
            "song_id": song_ids[query_index],
            "true_window_idx": window_idxs[query_index],
            "true_section_id": section_ids[query_index],
            "predicted_window_idx": window_idxs[best_j],
            "predicted_section_id": section_ids[best_j],
            "section_correct": section_ids[best_j] == section_ids[query_index],
        }
        if subject_ids is not None:
            result["subject_id"] = subject_ids[query_index]

        results.append(result)

    return results


def summarize_section_search_results(results):
    if not results:
        raise ValueError("No section-search results were provided.")

    accuracy = float(
        np.mean([result["section_correct"] for result in results], dtype=np.float64)
    )
    return {
        "section_search_top1": accuracy,
        "section_search_n_scored": len(results),
    }


def log_section_search_metrics(split_name, summary):
    logging.info(
        "  %-5s sec-srch | top1=%.4f (n=%d scored)",
        split_name,
        summary["section_search_top1"],
        summary["section_search_n_scored"],
    )


def save_section_search_results_csv(results, output_path):
    pd.DataFrame(results).to_csv(str(output_path), index=False)


def summarize_song_search_results(
        results,
        localization_tolerances=(0, 1, 5),
):
    if not results:
        raise ValueError("No song-search results were provided.")

    song_accuracy = float(
        np.mean([result["song_correct"] for result in results], dtype=np.float64)
    )
    has_marginal_results = all(
        "marginal_song_correct" in result for result in results
    )
    localization_errors = np.asarray(
        [result["localization_error"] for result in results],
        dtype=np.float64,
    )

    summary = {
        "song_search_song_top1": song_accuracy,
        "song_search_mean_localization_error": float(localization_errors.mean()),
        "song_search_median_localization_error": float(np.median(localization_errors)),
    }
    if has_marginal_results:
        summary["song_search_marginal_top1"] = float(
            np.mean(
                [result["marginal_song_correct"] for result in results],
                dtype=np.float64,
            )
        )
        summary["song_search_marginal_temperature"] = float(
            results[0]["marginal_temperature"]
        )

    for tolerance in localization_tolerances:
        summary[f"song_search_loc_within_{tolerance}"] = float(
            np.mean(localization_errors <= tolerance)
        )

    if all("subject_id" in result for result in results):
        subject_scores = {}
        for result in results:
            subject_scores.setdefault(result["subject_id"], []).append(result)

        by_subject = {}
        for subject_id, subject_results in sorted(subject_scores.items()):
            subject_summary = {
                "song_top1": float(
                    np.mean([item["song_correct"] for item in subject_results], dtype=np.float64)
                ),
                "mean_localization_error": float(
                    np.mean(
                        [item["localization_error"] for item in subject_results],
                        dtype=np.float64,
                    )
                ),
                "n_queries": len(subject_results),
            }
            if has_marginal_results:
                subject_summary["marginal_top1"] = float(
                    np.mean(
                        [item["marginal_song_correct"] for item in subject_results],
                        dtype=np.float64,
                    )
                )
            by_subject[subject_id] = subject_summary
        summary["song_search_by_subject"] = by_subject

    return summary


def log_song_search_metrics(split_name, summary):
    logging.info(
        "  %-5s song-search | song-top1=%.4f loc-mean=%.2f loc-median=%.2f",
        split_name,
        summary["song_search_song_top1"],
        summary["song_search_mean_localization_error"],
        summary["song_search_median_localization_error"],
    )
    if "song_search_marginal_top1" in summary:
        logging.info(
            "  %-5s song-marg  | top1=%.4f temperature=%.6g",
            split_name,
            summary["song_search_marginal_top1"],
            summary["song_search_marginal_temperature"],
        )

    tolerance_keys = sorted(
        key for key in summary if key.startswith("song_search_loc_within_")
    )
    if tolerance_keys:
        formatted = " ".join(
            f"{key.removeprefix('song_search_')}={summary[key]:.4f}"
            for key in tolerance_keys
        )
        logging.info("  %-5s              | %s", split_name, formatted)

    if "song_search_song_top1_p" in summary:
        if "song_search_song_top1_p_lower" in summary:
            logging.info(
                "  %-5s song-null   | chance=%.4f null=%.4f±%.4f "
                "p-upper=%.4g p-lower=%.4g (n=%d)",
                split_name,
                summary["song_search_song_top1_chance"],
                summary["song_search_song_top1_null_mean"],
                summary["song_search_song_top1_null_std"],
                summary["song_search_song_top1_p"],
                summary["song_search_song_top1_p_lower"],
                summary["song_search_song_top1_n_perm"],
            )
        else:
            logging.info(
                "  %-5s song-null   | chance=%.4f null=%.4f±%.4f "
                "p-upper=%.4g (n=%d)",
                split_name,
                summary["song_search_song_top1_chance"],
                summary["song_search_song_top1_null_mean"],
                summary["song_search_song_top1_null_std"],
                summary["song_search_song_top1_p"],
                summary["song_search_song_top1_n_perm"],
            )

    if "song_search_marginal_top1_p" in summary:
        logging.info(
            "  %-5s marg-null   | chance=%.4f null=%.4f±%.4f "
            "p-upper=%.4g p-lower=%.4g (n=%d)",
            split_name,
            summary["song_search_marginal_top1_chance"],
            summary["song_search_marginal_top1_null_mean"],
            summary["song_search_marginal_top1_null_std"],
            summary["song_search_marginal_top1_p"],
            summary["song_search_marginal_top1_p_lower"],
            summary["song_search_marginal_top1_n_perm"],
        )

    by_subject = summary.get("song_search_by_subject")
    if by_subject:
        for subject_id, subject_metrics in by_subject.items():
            marginal_text = (
                f" marginal-top1={subject_metrics['marginal_top1']:.4f}"
                if "marginal_top1" in subject_metrics
                else ""
            )
            logging.info(
                "  %-5s              | subject=%s song-top1=%.4f%s "
                "loc-mean=%.2f n=%d",
                split_name,
                subject_id,
                subject_metrics["song_top1"],
                marginal_text,
                subject_metrics["mean_localization_error"],
                subject_metrics["n_queries"],
            )


def save_song_search_results_csv(results, output_path):
    rows = []

    for result in results:
        row = {
            "query_index": result["query_index"],
            "subject_id": result.get("subject_id"),
            "true_song_id": result["true_song_id"],
            "true_window_idx": result["true_window_idx"],
            "predicted_song_id": result["predicted_song_id"],
            "song_correct": result["song_correct"],
            "marginal_predicted_song_id": result.get(
                "marginal_predicted_song_id"
            ),
            "marginal_song_correct": result.get("marginal_song_correct"),
            "marginal_temperature": result.get("marginal_temperature"),
            "true_song_best_window_idx": result["true_song_best_window_idx"],
            "true_song_best_score": result["true_song_best_score"],
            "true_song_marginal_log_score": result.get(
                "true_song_marginal_log_score"
            ),
            "true_song_marginal_probability": result.get(
                "true_song_marginal_probability"
            ),
            "localization_error": result["localization_error"],
        }

        ranked_songs = result.get("ranked_songs", [])
        for rank, ranked_song in enumerate(ranked_songs, start=1):
            row[f"rank{rank}_song_id"] = ranked_song["song_id"]
            row[f"rank{rank}_best_window_idx"] = ranked_song["best_window_idx"]
            row[f"rank{rank}_best_score"] = ranked_song["best_score"]

        marginal_ranked_songs = result.get("marginal_ranked_songs", [])
        for rank, ranked_song in enumerate(marginal_ranked_songs, start=1):
            row[f"marginal_rank{rank}_song_id"] = ranked_song["song_id"]
            row[f"marginal_rank{rank}_log_score"] = ranked_song[
                "marginal_log_score"
            ]
            row[f"marginal_rank{rank}_probability"] = ranked_song[
                "marginal_probability"
            ]

        top_windows = result.get("true_song_top_windows", [])
        for rank, top_window in enumerate(top_windows, start=1):
            row[f"true_song_top_window_{rank}"] = top_window["window_idx"]
            row[f"true_song_top_score_{rank}"] = top_window["score"]

        rows.append(row)

    output_path = str(output_path)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_song_search_summary_csv(summary, output_path):
    """Save scalar song-identification and permutation metrics."""
    metric_specs = (
        ("n_queries", "song_search_n_queries", "count"),
        ("candidate_songs_per_query", "song_search_n_candidate_songs", "count"),
        ("n_correct", "song_search_song_top1_n_correct", "count"),
        ("observed_song_accuracy", "song_search_song_top1", "proportion"),
        ("analytic_chance", "song_search_song_top1_chance", "proportion"),
        ("accuracy_minus_chance", "song_search_song_top1_minus_chance", "proportion"),
        ("permutation_null_mean", "song_search_song_top1_null_mean", "proportion"),
        ("permutation_null_std", "song_search_song_top1_null_std", "proportion"),
        ("accuracy_minus_null", "song_search_song_top1_minus_null", "proportion"),
        ("permutation_p_value", "song_search_song_top1_p", "probability"),
        ("permutation_p_value_lower", "song_search_song_top1_p_lower", "probability"),
        ("n_permutations", "song_search_song_top1_n_perm", "count"),
        ("marginal_temperature", "song_search_marginal_temperature", "temperature"),
        ("marginal_n_correct", "song_search_marginal_top1_n_correct", "count"),
        ("marginal_observed_song_accuracy", "song_search_marginal_top1", "proportion"),
        ("marginal_analytic_chance", "song_search_marginal_top1_chance", "proportion"),
        ("marginal_accuracy_minus_chance", "song_search_marginal_top1_minus_chance", "proportion"),
        ("marginal_permutation_null_mean", "song_search_marginal_top1_null_mean", "proportion"),
        ("marginal_permutation_null_std", "song_search_marginal_top1_null_std", "proportion"),
        ("marginal_accuracy_minus_null", "song_search_marginal_top1_minus_null", "proportion"),
        ("marginal_permutation_p_value", "song_search_marginal_top1_p", "probability"),
        ("marginal_permutation_p_value_lower", "song_search_marginal_top1_p_lower", "probability"),
        ("marginal_n_permutations", "song_search_marginal_top1_n_perm", "count"),
    )
    rows = [
        {"metric": metric, "value": summary[key], "unit": unit}
        for metric, key, unit in metric_specs
        if key in summary
    ]
    output_path = str(output_path)
    pd.DataFrame(rows, columns=("metric", "value", "unit")).to_csv(
        output_path,
        index=False,
    )
    return output_path


def log_similarity_scores_grouped_by_song(
        split_name,
        grouped_scores,
        query_index=None,
        query_song_id=None,
        query_window_idx=None,
        top_k=None,
):
    if (
        query_index is not None
        or query_song_id is not None
        or query_window_idx is not None
    ):
        logging.info(
            "  %-5s query    | index=%s song=%s window=%s",
            split_name,
            query_index if query_index is not None else "n/a",
            query_song_id if query_song_id is not None else "n/a",
            query_window_idx if query_window_idx is not None else "n/a",
        )

    logging.info("  %-5s similarity scores by song", split_name)

    for song_id in sorted(grouped_scores):
        entries = grouped_scores[song_id]
        if top_k is not None:
            display_entries = sorted(
                entries,
                key=lambda item: item["score"],
                reverse=True,
            )[:top_k]
        else:
            display_entries = entries

        formatted_scores = ", ".join(
            (
                f"(window={item['window_idx']}, score={item['score']:.4f}, "
                f"repeats={item['n_repeats']})"
                if item.get("n_repeats", 1) > 1
                else f"(window={item['window_idx']}, score={item['score']:.4f})"
            )
            for item in display_entries
        )

        logging.info(
            "  %-5s          | song=%s %s",
            split_name,
            song_id,
            formatted_scores if formatted_scores else "(no scores)",
        )


def evaluate_candidate_pool_regimes(
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
        ks=(1, 5, 10),
        gap=0,
        cache=None,
):
    cache = _resolve_retrieval_cache(
        cache=cache,
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=gap,
    )
    results = {}

    for regime in RETRIEVAL_REGIMES:
        regime_metrics = _score_cached_candidate_pool(cache, regime, ks)
        for k, value in regime_metrics.items():
            results[f"{regime}_top{k}"] = value

    return results


def _chance_hit_probability(n_candidates, n_correct, k):
    """Exact (hypergeometric) probability that a uniformly random top-k draw
    without replacement from a pool of n_candidates includes at least one of
    n_correct marked-positive items.

    Not min(k, n_correct) / n_candidates: that formula only happens to be
    correct when n_correct == 1 and k == 1 simultaneously -- e.g. exact
    window matching always has exactly one true positive, so it looked
    right there, but silently understates chance for k > 1 (Tier 1's own
    top5/top10) and for n_correct > 1 (Tier 2 section matching, where many
    candidates can legitimately share a query's section).
    """
    effective_k = min(k, n_candidates)
    if n_correct >= n_candidates:
        return 1.0
    prob_zero_hits = (
        math.comb(n_candidates - n_correct, effective_k) / math.comb(n_candidates, effective_k)
    )
    return 1.0 - prob_zero_hits


def chance_retrieval_with_candidate_pool(
        song_ids,
        window_idxs,
        k=10,
        regime="across_song",
        gap=0,
):
    song_ids = [int(x) for x in song_ids]
    window_idxs = [int(x) for x in window_idxs]
    correct_probabilities = []

    for i in range(len(song_ids)):
        candidate_indices = _candidate_indices_for_query(
            i,
            song_ids,
            window_idxs,
            regime=regime,
            gap=gap,
        )
        if not candidate_indices:
            raise ValueError("Candidate pool is empty for at least one query.")

        correct_positions = [
            pos for pos, j in enumerate(candidate_indices)
            if song_ids[j] == song_ids[i] and window_idxs[j] == window_idxs[i]
        ]
        if not correct_positions:
            raise ValueError("No matching positive found in candidate pool.")

        correct_probabilities.append(
            _chance_hit_probability(len(candidate_indices), len(correct_positions), k)
        )

    return float(np.mean(correct_probabilities))


def evaluate_candidate_pool_chance_regimes(
        song_ids,
        window_idxs,
        ks=(1, 5, 10),
        gap=0,
):
    results = {}

    for regime in RETRIEVAL_REGIMES:
        for k in ks:
            results[f"{regime}_top{k}_chance"] = chance_retrieval_with_candidate_pool(
                song_ids=song_ids,
                window_idxs=window_idxs,
                k=k,
                regime=regime,
                gap=gap,
            )

    return results


def retrieval_with_candidate_pool_section(
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
        section_ids,
        k=10,
        gap=0,
):
    """Tier 2 (section) retrieval: same-song candidate pool as the
    `within_song` regime, but correctness relaxes from an exact window_idx
    match to a section_id match -- chorus 1 mistaken for chorus 2 counts as
    correct. Queries whose section_id is -1 (below the 80% overlap
    threshold in align_class_windows.py) are skipped entirely: their true
    section is ambiguous, so scoring them would reward or punish a coin
    flip. Candidates with section_id == -1 are left in the pool untouched --
    they can never satisfy the correctness check against a real (non -1)
    query section, so no separate filtering is needed for them.
    """
    eeg_emb = torch.as_tensor(eeg_emb, dtype=torch.float32)
    audio_emb = torch.as_tensor(audio_emb, dtype=torch.float32)
    song_ids = [int(x) for x in song_ids]
    window_idxs = [int(x) for x in window_idxs]
    section_ids = [int(x) for x in section_ids]

    correct = 0
    n_scored = 0

    for i in range(len(eeg_emb)):
        if section_ids[i] == -1:
            continue
        n_scored += 1

        candidate_indices = _candidate_indices_for_query(
            i,
            song_ids,
            window_idxs,
            regime="within_song",
            gap=gap,
        )
        candidate_audio = audio_emb[candidate_indices]
        query_eeg = eeg_emb[i].unsqueeze(0)

        similarities = sequence_similarity_logits(query_eeg, candidate_audio).squeeze(0)
        effective_k = min(k, len(candidate_indices))
        topk_positions = torch.topk(similarities, k=effective_k).indices.tolist()

        correct_positions = [
            pos for pos, j in enumerate(candidate_indices)
            if section_ids[j] == section_ids[i]
        ]

        if any(pos in topk_positions for pos in correct_positions):
            correct += 1

    if n_scored == 0:
        raise ValueError("No queries with a valid section_id (-1 excluded) were found.")

    return correct / n_scored


def chance_retrieval_with_candidate_pool_section(
        song_ids,
        window_idxs,
        section_ids,
        k=10,
        gap=0,
):
    song_ids = [int(x) for x in song_ids]
    window_idxs = [int(x) for x in window_idxs]
    section_ids = [int(x) for x in section_ids]
    correct_probabilities = []

    for i in range(len(song_ids)):
        if section_ids[i] == -1:
            continue

        candidate_indices = _candidate_indices_for_query(
            i,
            song_ids,
            window_idxs,
            regime="within_song",
            gap=gap,
        )
        if not candidate_indices:
            raise ValueError("Candidate pool is empty for at least one query.")

        correct_positions = [
            pos for pos, j in enumerate(candidate_indices)
            if section_ids[j] == section_ids[i]
        ]
        if not correct_positions:
            raise ValueError("No matching positive found in candidate pool.")

        correct_probabilities.append(
            _chance_hit_probability(len(candidate_indices), len(correct_positions), k)
        )

    if not correct_probabilities:
        raise ValueError("No queries with a valid section_id (-1 excluded) were found.")

    return float(np.mean(correct_probabilities))


def evaluate_section_regime(
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
        section_ids,
        ks=(1, 5, 10),
        gap=0,
        cache=None,
):
    cache = _resolve_retrieval_cache(
        cache=cache,
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=gap,
    )
    positive_mask, query_mask = _section_masks(cache, section_ids)
    metrics = _score_cached_candidate_pool(
        cache,
        "within_song",
        ks,
        positive_mask=positive_mask,
        query_mask=query_mask,
    )
    return {
        f"within_song_section_top{k}": value
        for k, value in metrics.items()
    }


def chance_section_regime(
        song_ids,
        window_idxs,
        section_ids,
        ks=(1, 5, 10),
        gap=0,
):
    return {
        f"within_song_section_top{k}_chance": chance_retrieval_with_candidate_pool_section(
            song_ids=song_ids,
            window_idxs=window_idxs,
            section_ids=section_ids,
            k=k,
            gap=gap,
        )
        for k in ks
    }


def evaluate_section_null_regime(
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
        section_ids,
        ks=(1, 5, 10),
        gap=0,
        n_perm=1000,
        rng=0,
        cache=None,
):
    """Permutation-null baseline for the section regime: repeatedly shuffles
    audio embeddings (keeping song_id/window_idx/section_id labels fixed, so
    candidate-pool structure -- including each section's size -- is
    preserved) and re-scores, giving an empirical null distribution rather
    than a single point-estimate chance value. More rigorous than
    chance_section_regime alone: a section with many members could look
    "above chance" by the simple formula through noise alone, and this
    catches that the same way evaluate_candidate_pool_null_regimes already
    does for Tier 1.
    """
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1.")
    cache = _resolve_retrieval_cache(
        cache=cache,
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=gap,
    )
    positive_mask, query_mask = _section_masks(cache, section_ids)
    observed = _score_cached_candidate_pool(
        cache,
        "within_song",
        ks,
        positive_mask=positive_mask,
        query_mask=query_mask,
    )
    null_scores = {int(k): [] for k in ks}
    rng = np.random.default_rng(rng)

    for perm_idx in range(n_perm):
        permutation = rng.permutation(cache.similarities.shape[1])
        permuted = _score_cached_candidate_pool(
            cache,
            "within_song",
            ks,
            audio_permutation=permutation,
            positive_mask=positive_mask,
            query_mask=query_mask,
        )
        for k, value in permuted.items():
            null_scores[k].append(value)

        if (perm_idx + 1) % 50 == 0 or perm_idx + 1 == n_perm:
            logging.info(
                "Permutation progress [within_song_section top1/top5/top10]: %d/%d",
                perm_idx + 1,
                n_perm,
            )

    summary = {}
    for k in (int(k) for k in ks):
        values = np.asarray(null_scores[k], dtype=np.float64)
        p_upper, p_lower = _empirical_tail_p_values(values, observed[k])
        summary[f"within_song_section_top{k}_null_mean"] = float(values.mean())
        summary[f"within_song_section_top{k}_null_std"] = float(values.std(ddof=0))
        summary[f"within_song_section_top{k}_p"] = p_upper
        summary[f"within_song_section_top{k}_p_lower"] = p_lower

    return summary


def log_section_null_metrics(split_name, metrics):
    logging.info(
        "  %-5s sec-null | top1=%.4f±%.4f top5=%.4f±%.4f top10=%.4f±%.4f",
        split_name,
        metrics["within_song_section_top1_null_mean"],
        metrics["within_song_section_top1_null_std"],
        metrics["within_song_section_top5_null_mean"],
        metrics["within_song_section_top5_null_std"],
        metrics["within_song_section_top10_null_mean"],
        metrics["within_song_section_top10_null_std"],
    )


def log_section_p_metrics(split_name, metrics):
    logging.info(
        "  %-5s sec-p-upper | top1=%.4f top5=%.4f top10=%.4f",
        split_name,
        metrics["within_song_section_top1_p"],
        metrics["within_song_section_top5_p"],
        metrics["within_song_section_top10_p"],
    )
    lower_keys = [
        f"within_song_section_top{k}_p_lower"
        for k in (1, 5, 10)
    ]
    if all(key in metrics for key in lower_keys):
        logging.info(
            "  %-5s sec-p-lower | top1=%.4f top5=%.4f top10=%.4f",
            split_name,
            *(metrics[key] for key in lower_keys),
        )


def summarize_section_coverage(section_ids):
    section_ids = [int(x) for x in section_ids]
    n_total = len(section_ids)
    n_boundary = sum(1 for s in section_ids if s == -1)
    return {
        "section_eval_n_total": n_total,
        "section_eval_n_excluded_boundary": n_boundary,
        "section_eval_n_scored": n_total - n_boundary,
    }


def log_section_metrics(split_name, metrics):
    logging.info(
        "  %-5s section  | top1=%.4f top5=%.4f top10=%.4f",
        split_name,
        metrics["within_song_section_top1"],
        metrics["within_song_section_top5"],
        metrics["within_song_section_top10"],
    )


def log_section_chance_metrics(split_name, metrics):
    logging.info(
        "  %-5s sec-chnc | top1=%.4f top5=%.4f top10=%.4f",
        split_name,
        metrics["within_song_section_top1_chance"],
        metrics["within_song_section_top5_chance"],
        metrics["within_song_section_top10_chance"],
    )


def log_section_coverage(split_name, metrics):
    logging.info(
        "  %-5s section  | scored=%d/%d windows (%d excluded as boundary, section_id=-1)",
        split_name,
        metrics["section_eval_n_scored"],
        metrics["section_eval_n_total"],
        metrics["section_eval_n_excluded_boundary"],
    )


def summarize_candidate_pool_sizes(
        song_ids,
        window_idxs,
        regimes=RETRIEVAL_REGIMES,
        gap=0,
):
    song_ids = [int(x) for x in song_ids]
    window_idxs = [int(x) for x in window_idxs]
    summary = {}

    for regime in regimes:
        sizes = [
            len(_candidate_indices_for_query(i, song_ids, window_idxs, regime=regime, gap=gap))
            for i in range(len(song_ids))
        ]
        size_array = np.asarray(sizes, dtype=np.float64)
        summary[f"{regime}_pool_size_mean"] = float(size_array.mean())
        summary[f"{regime}_pool_size_min"] = int(size_array.min())
        summary[f"{regime}_pool_size_max"] = int(size_array.max())

    return summary


def _candidate_pool_permutation_samples(cache, ks, n_perm, rng=0):
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1.")

    observed = {
        regime: _score_cached_candidate_pool(cache, regime, ks)
        for regime in RETRIEVAL_REGIMES
    }
    null_scores = {
        regime: {int(k): [] for k in ks}
        for regime in RETRIEVAL_REGIMES
    }
    rng = np.random.default_rng(rng)

    for perm_idx in range(n_perm):
        permutation = rng.permutation(cache.similarities.shape[1])
        for regime in RETRIEVAL_REGIMES:
            permuted = _score_cached_candidate_pool(
                cache,
                regime,
                ks,
                audio_permutation=permutation,
            )
            for k, value in permuted.items():
                null_scores[regime][k].append(value)

        if (perm_idx + 1) % 50 == 0 or perm_idx + 1 == n_perm:
            logging.info(
                "Permutation progress [all candidate pools, all top-k]: %d/%d",
                perm_idx + 1,
                n_perm,
            )

    return observed, {
        regime: {
            k: np.asarray(values, dtype=np.float64)
            for k, values in regime_scores.items()
        }
        for regime, regime_scores in null_scores.items()
    }


def evaluate_candidate_pool_p_values(
        eeg_emb,
        audio_emb,
        window_idxs,
        song_ids,
        ks=(1, 5, 10),
        gap=0,
        n_perm=1000,
        within_song_shuffle_metrics=None,
        cache=None,
):
    cache = _resolve_retrieval_cache(
        cache=cache,
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=gap,
    )
    observed, null_scores = _candidate_pool_permutation_samples(
        cache,
        ks=ks,
        n_perm=n_perm,
        rng=0,
    )
    p_values = {}
    for regime in RETRIEVAL_REGIMES:
        for k in (int(k) for k in ks):
            p_upper, p_lower = _empirical_tail_p_values(
                null_scores[regime][k],
                observed[regime][k],
            )
            p_values[f"{regime}_top{k}_p"] = p_upper
            p_values[f"{regime}_top{k}_p_lower"] = p_lower

    if within_song_shuffle_metrics is not None:
        for k in ks:
            for prefix in ("within_song_audio_shuffle", "within_song_eeg_shuffle"):
                for suffix in ("_p", "_p_lower"):
                    key = f"{prefix}_top{k}{suffix}"
                    if key in within_song_shuffle_metrics:
                        p_values[key] = float(within_song_shuffle_metrics[key])

    return p_values


def evaluate_candidate_pool_null_regimes(
        eeg_emb,
        audio_emb,
        window_idxs,
        song_ids,
        ks=(1, 5, 10),
        gap=0,
        n_perm=1000,
        rng=0,
        cache=None,
):
    cache = _resolve_retrieval_cache(
        cache=cache,
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=gap,
    )
    observed, null_scores = _candidate_pool_permutation_samples(
        cache,
        ks=ks,
        n_perm=n_perm,
        rng=rng,
    )
    summary = {}
    for regime in RETRIEVAL_REGIMES:
        for k in (int(k) for k in ks):
            values = null_scores[regime][k]
            p_upper, p_lower = _empirical_tail_p_values(
                values,
                observed[regime][k],
            )
            summary[f"{regime}_top{k}_null_mean"] = float(values.mean())
            summary[f"{regime}_top{k}_null_std"] = float(values.std(ddof=0))
            summary[f"{regime}_top{k}_p"] = p_upper
            summary[f"{regime}_top{k}_p_lower"] = p_lower

    return summary


def log_candidate_pool_null_metrics(split_name, null_regimes):
    logging.info(
        "  %-5s seg-null | across   top1=%.4f±%.4f top5=%.4f±%.4f top10=%.4f±%.4f",
        split_name,
        null_regimes["across_song_top1_null_mean"],
        null_regimes["across_song_top1_null_std"],
        null_regimes["across_song_top5_null_mean"],
        null_regimes["across_song_top5_null_std"],
        null_regimes["across_song_top10_null_mean"],
        null_regimes["across_song_top10_null_std"],
    )
    logging.info(
        "  %-5s          | within   top1=%.4f±%.4f top5=%.4f±%.4f top10=%.4f±%.4f",
        split_name,
        null_regimes["within_song_top1_null_mean"],
        null_regimes["within_song_top1_null_std"],
        null_regimes["within_song_top5_null_mean"],
        null_regimes["within_song_top5_null_std"],
        null_regimes["within_song_top10_null_mean"],
        null_regimes["within_song_top10_null_std"],
    )
    logging.info(
        "  %-5s          | no-negs  top1=%.4f±%.4f top5=%.4f±%.4f top10=%.4f±%.4f",
        split_name,
        null_regimes["across_song_no_same_song_negatives_top1_null_mean"],
        null_regimes["across_song_no_same_song_negatives_top1_null_std"],
        null_regimes["across_song_no_same_song_negatives_top5_null_mean"],
        null_regimes["across_song_no_same_song_negatives_top5_null_std"],
        null_regimes["across_song_no_same_song_negatives_top10_null_mean"],
        null_regimes["across_song_no_same_song_negatives_top10_null_std"],
    )


def _within_song_permutation_indices(song_ids, rng):
    song_to_indices = {}
    for idx, song_id in enumerate(int(x) for x in song_ids):
        song_to_indices.setdefault(song_id, []).append(idx)

    permutation = np.arange(len(song_ids), dtype=np.int64)
    for indices in song_to_indices.values():
        permutation[indices] = rng.permutation(indices)
    return permutation


def evaluate_within_song_shuffle_baseline(
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
        ks=(1, 5, 10),
        gap=0,
        n_shuffle=200,
        rng=0,
        cache=None,
):
    if n_shuffle < 1:
        raise ValueError("n_shuffle must be at least 1.")
    cache = _resolve_retrieval_cache(
        cache=cache,
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=gap,
    )
    rng = np.random.default_rng(rng)
    audio_scores = {int(k): [] for k in ks}
    eeg_scores = {int(k): [] for k in ks}
    observed_scores = _score_cached_candidate_pool(
        cache,
        "within_song",
        ks,
    )

    for shuffle_idx in range(n_shuffle):
        audio_permutation = _within_song_permutation_indices(song_ids, rng)
        audio_metrics = _score_cached_candidate_pool(
            cache,
            "within_song",
            ks,
            audio_permutation=audio_permutation,
        )
        for k, value in audio_metrics.items():
            audio_scores[k].append(value)

        eeg_permutation = _within_song_permutation_indices(song_ids, rng)
        eeg_metrics = _score_cached_candidate_pool(
            cache,
            "within_song",
            ks,
            eeg_permutation=eeg_permutation,
        )
        for k, value in eeg_metrics.items():
            eeg_scores[k].append(value)

        if (shuffle_idx + 1) % 50 == 0 or shuffle_idx + 1 == n_shuffle:
            logging.info(
                "Within-song shuffle progress: %d/%d",
                shuffle_idx + 1,
                n_shuffle,
            )

    summary = {}
    for k in (int(k) for k in ks):
        audio_values = np.asarray(audio_scores[k], dtype=np.float64)
        eeg_values = np.asarray(eeg_scores[k], dtype=np.float64)
        audio_p_upper, audio_p_lower = _empirical_tail_p_values(
            audio_values,
            observed_scores[k],
        )
        eeg_p_upper, eeg_p_lower = _empirical_tail_p_values(
            eeg_values,
            observed_scores[k],
        )

        summary[f"within_song_shuffle_top{k}_mean"] = float(audio_values.mean())
        summary[f"within_song_shuffle_top{k}_std"] = float(audio_values.std(ddof=0))
        summary[f"within_song_shuffle_top{k}_p"] = audio_p_upper
        summary[f"within_song_shuffle_top{k}_p_lower"] = audio_p_lower
        summary[f"within_song_audio_shuffle_top{k}_mean"] = float(audio_values.mean())
        summary[f"within_song_audio_shuffle_top{k}_std"] = float(audio_values.std(ddof=0))
        summary[f"within_song_audio_shuffle_top{k}_p"] = audio_p_upper
        summary[f"within_song_audio_shuffle_top{k}_p_lower"] = audio_p_lower
        summary[f"within_song_eeg_shuffle_top{k}_mean"] = float(eeg_values.mean())
        summary[f"within_song_eeg_shuffle_top{k}_std"] = float(eeg_values.std(ddof=0))
        summary[f"within_song_eeg_shuffle_top{k}_p"] = eeg_p_upper
        summary[f"within_song_eeg_shuffle_top{k}_p_lower"] = eeg_p_lower

    return summary


def evaluate_embedding_benchmark(
        eeg_emb,
        audio_emb,
        song_ids,
        window_idxs,
        ks=(1, 5, 10),
        gap=0,
        n_perm=1000,
        n_within_song_shuffles=200,
        rng=0,
        n_song_search_perms=None,
        song_search_permutation_batch_size=16,
        song_search_marginal_temperature=0.07,
):
    """Evaluate candidate-pool retrieval and song-level identification.

    Song identification includes both the existing best-window metric and a
    length-normalized probability marginal over each song's unique windows.
    """
    if n_song_search_perms is None:
        n_song_search_perms = n_perm

    cache = build_retrieval_evaluation_cache(
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=gap,
    )
    observed = evaluate_candidate_pool_regimes(
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
        ks=ks,
        gap=gap,
        cache=cache,
    )
    chance = evaluate_candidate_pool_chance_regimes(
        song_ids=song_ids,
        window_idxs=window_idxs,
        ks=ks,
        gap=gap,
    )
    nulls = evaluate_candidate_pool_null_regimes(
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
        ks=ks,
        gap=gap,
        n_perm=n_perm,
        rng=rng,
        cache=cache,
    )
    within_song_shuffle = evaluate_within_song_shuffle_baseline(
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
        ks=ks,
        gap=gap,
        n_shuffle=n_within_song_shuffles,
        rng=rng,
        cache=cache,
    )
    pool_sizes = summarize_candidate_pool_sizes(
        song_ids=song_ids,
        window_idxs=window_idxs,
        gap=gap,
    )
    song_search_cache = build_song_search_evaluation_cache(
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
    )
    song_search = evaluate_song_search_permutation_test(
        eeg_emb=eeg_emb,
        audio_emb=audio_emb,
        song_ids=song_ids,
        window_idxs=window_idxs,
        n_perm=n_song_search_perms,
        rng=rng,
        permutation_batch_size=song_search_permutation_batch_size,
        cache=song_search_cache,
        marginal_temperature=song_search_marginal_temperature,
    )
    song_search["song_search_song_top1"] = song_search[
        "song_search_song_top1_permutation_observed"
    ]
    song_search["song_search_marginal_top1"] = song_search[
        "song_search_marginal_top1_permutation_observed"
    ]

    lifts = {}
    for regime in RETRIEVAL_REGIMES:
        for k in ks:
            metric_key = f"{regime}_top{k}"
            chance_key = f"{metric_key}_chance"
            null_key = f"{metric_key}_null_mean"
            observed_value = observed[metric_key]
            chance_value = chance[chance_key]
            null_value = nulls[null_key]
            lifts[f"{metric_key}_minus_chance"] = float(observed_value - chance_value)
            lifts[f"{metric_key}_chance_ratio"] = float(
                observed_value / chance_value if chance_value > 0 else np.nan
            )
            lifts[f"{metric_key}_minus_null_mean"] = float(observed_value - null_value)

    return {
        "observed": observed,
        "chance": chance,
        "null": nulls,
        "within_song_shuffle": within_song_shuffle,
        "candidate_pool": pool_sizes,
        "lift": lifts,
        "song_search": song_search,
    }


def evaluate_epoch(
        dataloader,
        eeg_encoder,
        audio_projection,
        clip,
        device,
        gap=0,
):
    eeg_encoder.eval()
    audio_projection.eval()
    clip.eval()

    total_loss = 0.0
    total_examples = 0
    all_eeg_embeds = []
    all_audio_embeds = []
    all_song_ids = []
    all_window_idxs = []

    with torch.no_grad():
        for batch in dataloader:
            eeg = batch["eeg"].float().to(device)
            audio_features = batch["audio"].float().to(device)

            subject_ids = batch["subject_id"].long().to(device)
            song_ids = batch["song_id"].long().to(device)
            window_idxs = batch["window_idx"].long().to(device)

            eeg_embed = eeg_encoder(eeg, subject_ids)
            audio_embed = audio_projection(audio_features)
            aligned_eeg_embed, aligned_audio_embed = align_temporal_resolution(
                eeg_embed, audio_embed
            )

            loss = clip(
                aligned_eeg_embed,
                aligned_audio_embed,
                song_ids,
                window_idxs,
            )

            all_eeg_embeds.append(eeg_embed.cpu())
            all_audio_embeds.append(audio_embed.cpu())
            all_song_ids.extend(batch["song_id"].tolist())
            all_window_idxs.extend(batch["window_idx"].tolist())

            batch_size = eeg.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    all_eeg_embeds = torch.cat(all_eeg_embeds, dim=0)
    all_audio_embeds = torch.cat(all_audio_embeds, dim=0)

    results = evaluate_candidate_pool_regimes(
        all_eeg_embeds,
        all_audio_embeds,
        song_ids=all_song_ids,
        window_idxs=all_window_idxs,
        ks=(1, 5, 10),
        gap=gap,
    )
    results["loss"] = total_loss / total_examples
    return results


def log_split_metrics(split_name, metrics):
    logging.info(
        "  %-5s loss=%.4f | across   top1=%.4f top5=%.4f top10=%.4f",
        split_name,
        metrics["loss"],
        metrics["across_song_top1"],
        metrics["across_song_top5"],
        metrics["across_song_top10"],
    )
    logging.info(
        "  %-5s             | within   top1=%.4f top5=%.4f top10=%.4f",
        split_name,
        metrics["within_song_top1"],
        metrics["within_song_top5"],
        metrics["within_song_top10"],
    )
    logging.info(
        "  %-5s             | no-negs  top1=%.4f top5=%.4f top10=%.4f",
        split_name,
        metrics["across_song_no_same_song_negatives_top1"],
        metrics["across_song_no_same_song_negatives_top5"],
        metrics["across_song_no_same_song_negatives_top10"],
    )


def log_p_metrics(split_name, p_values):
    for tail_label, suffix in (("p-upper", "_p"), ("p-lower", "_p_lower")):
        logging.info(
            "  %-5s %-7s | across   top1=%.4f top5=%.4f top10=%.4f",
            split_name,
            tail_label,
            p_values[f"across_song_top1{suffix}"],
            p_values[f"across_song_top5{suffix}"],
            p_values[f"across_song_top10{suffix}"],
        )
        logging.info(
            "  %-5s         | within   top1=%.4f top5=%.4f top10=%.4f",
            split_name,
            p_values[f"within_song_top1{suffix}"],
            p_values[f"within_song_top5{suffix}"],
            p_values[f"within_song_top10{suffix}"],
        )
        for label, prefix in (
            ("a-shuf", "within_song_audio_shuffle"),
            ("e-shuf", "within_song_eeg_shuffle"),
        ):
            keys = [f"{prefix}_top{k}{suffix}" for k in (1, 5, 10)]
            if all(key in p_values for key in keys):
                logging.info(
                    "  %-5s         | %-8s top1=%.4f top5=%.4f top10=%.4f",
                    split_name,
                    label,
                    *(p_values[key] for key in keys),
                )
        logging.info(
            "  %-5s         | no-negs  top1=%.4f top5=%.4f top10=%.4f",
            split_name,
            p_values[f"across_song_no_same_song_negatives_top1{suffix}"],
            p_values[f"across_song_no_same_song_negatives_top5{suffix}"],
            p_values[f"across_song_no_same_song_negatives_top10{suffix}"],
        )


def log_within_song_shuffle_metrics(split_name, metrics):
    logging.info(
        "  %-5s shuffle  | a-shuf   top1=%.4f±%.4f top5=%.4f±%.4f top10=%.4f±%.4f",
        split_name,
        metrics["within_song_audio_shuffle_top1_mean"],
        metrics["within_song_audio_shuffle_top1_std"],
        metrics["within_song_audio_shuffle_top5_mean"],
        metrics["within_song_audio_shuffle_top5_std"],
        metrics["within_song_audio_shuffle_top10_mean"],
        metrics["within_song_audio_shuffle_top10_std"],
    )
    if all(
        key in metrics
        for key in (
            "within_song_eeg_shuffle_top1_mean",
            "within_song_eeg_shuffle_top1_std",
            "within_song_eeg_shuffle_top5_mean",
            "within_song_eeg_shuffle_top5_std",
            "within_song_eeg_shuffle_top10_mean",
            "within_song_eeg_shuffle_top10_std",
        )
    ):
        logging.info(
            "  %-5s          | e-shuf   top1=%.4f±%.4f top5=%.4f±%.4f top10=%.4f±%.4f",
            split_name,
            metrics["within_song_eeg_shuffle_top1_mean"],
            metrics["within_song_eeg_shuffle_top1_std"],
            metrics["within_song_eeg_shuffle_top5_mean"],
            metrics["within_song_eeg_shuffle_top5_std"],
            metrics["within_song_eeg_shuffle_top10_mean"],
            metrics["within_song_eeg_shuffle_top10_std"],
        )


def log_chance_metrics(split_name, metrics):
    logging.info(
        "  %-5s chance   | across   top1=%.4f top5=%.4f top10=%.4f",
        split_name,
        metrics["across_song_top1_chance"],
        metrics["across_song_top5_chance"],
        metrics["across_song_top10_chance"],
    )
    logging.info(
        "  %-5s          | within   top1=%.4f top5=%.4f top10=%.4f",
        split_name,
        metrics["within_song_top1_chance"],
        metrics["within_song_top5_chance"],
        metrics["within_song_top10_chance"],
    )
    logging.info(
        "  %-5s          | no-negs  top1=%.4f top5=%.4f top10=%.4f",
        split_name,
        metrics["across_song_no_same_song_negatives_top1_chance"],
        metrics["across_song_no_same_song_negatives_top5_chance"],
        metrics["across_song_no_same_song_negatives_top10_chance"],
    )


def log_null_metrics(split_name, metrics):
    logging.info(
        "  %-5s null     | across   top1=%.4f±%.4f top5=%.4f±%.4f top10=%.4f±%.4f",
        split_name,
        metrics["across_song_top1_null_mean"],
        metrics["across_song_top1_null_std"],
        metrics["across_song_top5_null_mean"],
        metrics["across_song_top5_null_std"],
        metrics["across_song_top10_null_mean"],
        metrics["across_song_top10_null_std"],
    )
    logging.info(
        "  %-5s          | within   top1=%.4f±%.4f top5=%.4f±%.4f top10=%.4f±%.4f",
        split_name,
        metrics["within_song_top1_null_mean"],
        metrics["within_song_top1_null_std"],
        metrics["within_song_top5_null_mean"],
        metrics["within_song_top5_null_std"],
        metrics["within_song_top10_null_mean"],
        metrics["within_song_top10_null_std"],
    )
    logging.info(
        "  %-5s          | no-negs  top1=%.4f±%.4f top5=%.4f±%.4f top10=%.4f±%.4f",
        split_name,
        metrics["across_song_no_same_song_negatives_top1_null_mean"],
        metrics["across_song_no_same_song_negatives_top1_null_std"],
        metrics["across_song_no_same_song_negatives_top5_null_mean"],
        metrics["across_song_no_same_song_negatives_top5_null_std"],
        metrics["across_song_no_same_song_negatives_top10_null_mean"],
        metrics["across_song_no_same_song_negatives_top10_null_std"],
    )


def log_candidate_pool_metrics(split_name, metrics):
    logging.info(
        "  %-5s pools    | across   mean=%.1f min=%d max=%d",
        split_name,
        metrics["across_song_pool_size_mean"],
        metrics["across_song_pool_size_min"],
        metrics["across_song_pool_size_max"],
    )
    logging.info(
        "  %-5s          | within   mean=%.1f min=%d max=%d",
        split_name,
        metrics["within_song_pool_size_mean"],
        metrics["within_song_pool_size_min"],
        metrics["within_song_pool_size_max"],
    )
    logging.info(
        "  %-5s          | no-negs  mean=%.1f min=%d max=%d",
        split_name,
        metrics["across_song_no_same_song_negatives_pool_size_mean"],
        metrics["across_song_no_same_song_negatives_pool_size_min"],
        metrics["across_song_no_same_song_negatives_pool_size_max"],
    )


def log_lift_metrics(split_name, metrics):
    logging.info(
        "  %-5s lift     | across   top1=%.2fx top5=%.2fx top10=%.2fx",
        split_name,
        metrics["across_song_top1_chance_ratio"],
        metrics["across_song_top5_chance_ratio"],
        metrics["across_song_top10_chance_ratio"],
    )
    logging.info(
        "  %-5s          | within   top1=%.2fx top5=%.2fx top10=%.2fx",
        split_name,
        metrics["within_song_top1_chance_ratio"],
        metrics["within_song_top5_chance_ratio"],
        metrics["within_song_top10_chance_ratio"],
    )
    logging.info(
        "  %-5s          | no-negs  top1=%.2fx top5=%.2fx top10=%.2fx",
        split_name,
        metrics["across_song_no_same_song_negatives_top1_chance_ratio"],
        metrics["across_song_no_same_song_negatives_top5_chance_ratio"],
        metrics["across_song_no_same_song_negatives_top10_chance_ratio"],
    )


def collect_embeddings(dataloader, eeg_encoder, audio_projection, clip, device):
    eeg_encoder.eval()
    audio_projection.eval()
    clip.eval()

    total_loss = 0.0
    total_examples = 0
    all_eeg_embeds = []
    all_audio_embeds = []
    all_subject_ids = []
    all_song_ids = []
    all_window_idxs = []
    all_section_ids = []

    with torch.no_grad():
        for batch in dataloader:
            eeg = batch["eeg"].float().to(device)
            audio_features = batch["audio"].float().to(device)
            subject_ids = batch["subject_id"].long().to(device)
            song_ids = batch["song_id"].long().to(device)
            window_idxs = batch["window_idx"].long().to(device)

            eeg_embed = eeg_encoder(eeg, subject_ids)
            audio_embed = audio_projection(audio_features)
            aligned_eeg_embed, aligned_audio_embed = align_temporal_resolution(
                eeg_embed, audio_embed
            )

            loss = clip(
                aligned_eeg_embed,
                aligned_audio_embed,
                song_ids,
                window_idxs,
            )

            all_eeg_embeds.append(eeg_embed.cpu())
            all_audio_embeds.append(audio_embed.cpu())
            all_subject_ids.extend(batch["subject_id"].tolist())
            all_song_ids.extend(batch["song_id"].tolist())
            all_window_idxs.extend(batch["window_idx"].tolist())
            all_section_ids.extend(batch["section_id"].tolist())

            batch_size = eeg.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "eeg_embeds": torch.cat(all_eeg_embeds, dim=0),
        "audio_embeds": torch.cat(all_audio_embeds, dim=0),
        "subject_ids": all_subject_ids,
        "window_idxs": all_window_idxs,
        "song_ids": all_song_ids,
        "section_ids": all_section_ids,
    }
