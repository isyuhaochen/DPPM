"""Portable YAML configuration; environment overrides are never saved in checkpoints."""

import hashlib
import importlib
import os
import sys
from pathlib import Path

import yaml


def load_config(path):
    with open(path) as stream:
        config = yaml.safe_load(stream)
    for key, env in {
        "base_model": "DPPM_BASE_MODEL",
        "compiler_checkpoint": "DPPM_COMPILER_CHECKPOINT",
        "d2l_source": "DPPM_D2L_SOURCE",
        "device": "DPPM_DEVICE",
    }.items():
        if env in os.environ:
            config[key] = os.environ[env]
    return config


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_source(path, module):
    if path:
        path = Path(os.path.expandvars(str(path)))
        sys.path.insert(0, str(path / "src" if (path / "src").exists() else path))
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"Install {module}'s upstream package or set its source path in YAML"
        ) from exc
