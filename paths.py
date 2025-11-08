# Copyright (c) 2025 Vanargo
# Licensed under the MIT License. See LICENSE in the project root.

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

_MARKERS: tuple[str, ...] = (".git", "pyproject.toml", "README.md")


def _has_marker(p: Path, markers: Iterable[str]) -> bool:
    return any((p / m).exists() for m in markers)


def get_project_root(markers: Iterable[str] = _MARKERS) -> Path:
    """
    Walk upwards from the current file/directory
    until one of the markers is found.
    """
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        # if the module is placed in src/, go up to its root #
        base = parent if parent.is_dir() else parent.parent
        if _has_marker(base, markers):
            return base
    # fallback: current working directory #
    return Path.cwd()


# absolute paths from ROOT #
ROOT = get_project_root()

# data #
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INT_DIR = DATA_DIR / "interim"
PROC_DIR = DATA_DIR / "processed"

# source code and notebooks live in the project root #
SRC_DIR = ROOT / "src"
NB_DIR = ROOT / "notebooks"

# models, reports, and artifacts live in the project root #
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
ART_DIR = ROOT / "artifacts"
