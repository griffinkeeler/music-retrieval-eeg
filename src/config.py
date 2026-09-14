"""Load paper experiment configs with small, explicit YAML overrides."""

from pathlib import Path

from omegaconf import OmegaConf


def load_config(config_path, _loading=None):
    """Load a YAML config and recursively merge any relative ``extends`` files."""
    config_path = Path(config_path).resolve()
    loading = set() if _loading is None else set(_loading)
    if config_path in loading:
        chain = " -> ".join(str(path) for path in (*loading, config_path))
        raise ValueError(f"Config inheritance cycle detected: {chain}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    loading.add(config_path)
    config = OmegaConf.load(config_path)
    parent_value = config.pop("extends", None)
    if parent_value is None:
        return config

    parent_names = (
        list(parent_value)
        if OmegaConf.is_list(parent_value)
        else [str(parent_value)]
    )
    merged = OmegaConf.create()
    for parent_name in parent_names:
        parent_path = Path(parent_name)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        merged = OmegaConf.merge(
            merged,
            load_config(parent_path, _loading=loading),
        )
    return OmegaConf.merge(merged, config)


def resolve_split_directory(project_root, config):
    """Return the configured shared split directory, with legacy fallback."""
    configured = config.get("split_directory")
    if configured is None:
        configured = Path("runs") / str(config.run_name) / "splits"
    split_directory = Path(str(configured))
    if not split_directory.is_absolute():
        split_directory = Path(project_root) / split_directory
    return split_directory


def resolve_split_path(project_root, config, split_filename):
    """Resolve an absolute split path or a filename in the shared split folder."""
    split_path = Path(str(split_filename))
    if split_path.is_absolute():
        return split_path
    return resolve_split_directory(project_root, config) / split_path
