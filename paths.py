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
    Надежно находит корень проекта как родительскую папку,
    в которой есть любой из маркеров (_MARKERS).
    Работает как в .py, так и в Jupyter (где __file__ не определен).
    """
    # попытка через __file__ #
    try:
        here = Path(__file__).resolve().parent
        for p in [here, *here.parents]:
            if _has_marker(p, markers):
                return p
    except NameError:
        pass

    # fallback: старт от текущего рабочего каталога #
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if _has_marker(p, markers):
            return p

    return cwd


# подготовка удобных алиасов директорий проекта #
ROOT = get_project_root()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INT_DIR = DATA_DIR / "interim"
PROC_DIR = DATA_DIR / "processed"
NB_DIR = DATA_DIR / "notebooks"
SRC_DIR = DATA_DIR / "src"
ART_DIR = DATA_DIR / "artifacts"
REPORTS_DIR = DATA_DIR / "reports"
MODELS_DIR = DATA_DIR / "models"

# убеждаемся, что ключевые папки существуют #
for d in (DATA_DIR, RAW_DIR, INT_DIR, PROC_DIR, ART_DIR, REPORTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
