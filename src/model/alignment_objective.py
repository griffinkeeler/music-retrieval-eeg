"""Configurable objectives for aligning EEG and frozen MERT representations."""

from torch import nn

from .clip_loss import CLIPLoss, sequence_similarity_logits


INFONCE_OBJECTIVE = "infonce"
COSINE_REGRESSION_OBJECTIVE = "cosine_regression"
SUPPORTED_ALIGNMENT_OBJECTIVES = (
    INFONCE_OBJECTIVE,
    COSINE_REGRESSION_OBJECTIVE,
)


class CosineRegressionLoss(nn.Module):
    """Regress each EEG representation directly onto its paired MERT target.

    Cosine similarity is computed jointly over feature and time dimensions, using
    the same implementation as retrieval scoring. Unlike InfoNCE, the loss has no
    dependence on unmatched examples in the batch.
    """

    def forward(self, eeg_embedding, audio_embedding, song_ids=None, window_idxs=None):
        del song_ids, window_idxs
        if eeg_embedding.shape[0] != audio_embedding.shape[0]:
            raise ValueError(
                "Cosine regression requires the same number of EEG and audio examples."
            )
        if eeg_embedding.shape[0] == 0:
            raise ValueError("Cosine regression requires at least one matched pair.")

        matched_cosines = sequence_similarity_logits(
            eeg_embedding,
            audio_embedding,
        ).diagonal()
        return 1.0 - matched_cosines.mean()


def _normalize_objective_name(objective_name):
    normalized = str(objective_name).strip().lower().replace("-", "_")
    aliases = {
        "clip": INFONCE_OBJECTIVE,
        "info_nce": INFONCE_OBJECTIVE,
        "paired_cosine": COSINE_REGRESSION_OBJECTIVE,
        "cosine": COSINE_REGRESSION_OBJECTIVE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_ALIGNMENT_OBJECTIVES:
        supported = ", ".join(SUPPORTED_ALIGNMENT_OBJECTIVES)
        raise ValueError(
            f"Unsupported alignment objective '{objective_name}'. "
            f"Choose one of: {supported}."
        )
    return normalized


def resolve_alignment_objective_name(config):
    """Return the normalized objective name, defaulting old configs to InfoNCE."""
    objective_config = config.get("objective") if config is not None else None
    if objective_config is None:
        return INFONCE_OBJECTIVE
    if isinstance(objective_config, str):
        return _normalize_objective_name(objective_config)
    return _normalize_objective_name(objective_config.get("type", INFONCE_OBJECTIVE))


def build_alignment_objective(config, objective_name=None):
    """Construct the configured alignment loss without changing model branches."""
    objective_name = _normalize_objective_name(
        objective_name
        if objective_name is not None
        else resolve_alignment_objective_name(config)
    )
    if objective_name == COSINE_REGRESSION_OBJECTIVE:
        return CosineRegressionLoss()

    clip_config = config.get("clip", {}) if config is not None else {}
    return CLIPLoss(temperature=float(clip_config.get("temperature", 0.07)))


def restore_alignment_objective(config, checkpoint):
    """Rebuild an objective and load new or legacy checkpoint state."""
    objective_name = checkpoint.get("objective_name")
    objective = build_alignment_objective(config, objective_name=objective_name)

    state_dict = checkpoint.get("objective_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("clip_state_dict")
    if state_dict is None:
        raise KeyError(
            "Checkpoint does not contain objective_state_dict or legacy "
            "clip_state_dict."
        )
    objective.load_state_dict(state_dict)
    return objective
