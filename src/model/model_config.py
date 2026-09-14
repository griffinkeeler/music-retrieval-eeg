def resolve_alignment_settings(config):
    ablations = config.get("ablations", {}) if config is not None else {}
    use_direct_alignment = bool(ablations.get("use_direct_alignment", False))

    if use_direct_alignment:
        return {
            "use_direct_alignment": True,
            "eeg_embedding_dim": 768,
            "use_projection_head": False,
            "audio_projection_mode": "identity",
            "audio_out_features": 768,
        }

    return {
        "use_direct_alignment": False,
        "eeg_embedding_dim": 512,
        "use_projection_head": True,
        "audio_projection_mode": "linear",
        "audio_out_features": 512,
    }
