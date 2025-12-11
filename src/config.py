import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    """Load a YAML config file and return it as a dict."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    return cfg
