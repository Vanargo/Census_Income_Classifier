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
    Идем вверх от текущего файла/каталога,
    пока не найдем один из маркеров.
    """
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        # если модуль помещен в src/, выходим на его корень #
        base = parent if parent.is_dir() else parent.parent
        if _has_marker(base, markers):
            return base
    # fallback: текущая рабочая директория #
    return Path.cwd()


# абсолютные пути от ROOT #
ROOT = get_project_root()

# данные #
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INT_DIR = DATA_DIR / "interim"
PROC_DIR = DATA_DIR / "processed"

# исходики и ноутбуки хранятся в корне проекта #
SRC_DIR = ROOT / "src"
NB_DIR = ROOT / "notebooks"

# модели, отчеты и артефакты - в корне проекта #
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
ART_DIR = ROOT / "artifacts"
