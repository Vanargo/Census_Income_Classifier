#!/usr/bin/env bash
# ------------------------------------------------------------------- #
# Purge heavy artifacts and reports from Git history
# ------------------------------------------------------------------- #

set -euo pipefail

echo "[purge] Starting filter-repo cleanup..."

# --- require tool --- #
if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "[purge] Installing git-filter-repo via pip..."
  pip install git-filter-repo
fi

# --- run cleanup --- #
git filter-repo --force \
  --path-glob '*.joblib' \
  --path-glob '*.npy' \
  --path-glob '*.html' \
  --path 'data/processed/adult_eda.csv' \
  --path 'data/raw/adult.data' \
  --path 'data/raw/adult.test' \
  --invert-paths

# --- compact repository --- #
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "[purge] Done."