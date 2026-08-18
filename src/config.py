"""Config loading. Every entry point calls load_config()."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("paths", {})
    # make every path absolute relative to the project root
    for key, value in list(cfg["paths"].items()):
        cfg["paths"][key] = str((PROJECT_ROOT / value).resolve())
    return cfg
