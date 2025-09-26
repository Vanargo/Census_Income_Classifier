import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

MODEL = "src.models.infer"


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.smoke
def test_cli_inference(tmp_path: Path, small_raw_df: pd.DataFrame, model_path: Path):
    inp = tmp_path / "mini.csv"
    out = tmp_path / "preds.csv"
    small_raw_df.to_csv(inp, index=False)

    # запуск CLI #
    cmd = [
        sys.executable,
        "-m",
        MODEL,
        "--input",
        str(inp),
        "--output",
        str(out),
        "--model",
        str(model_path),
        "--proba",
        "--also-label",
    ]
    r = _run(cmd)
    assert r.returncode == 0, r.stdout + r.stderr
    df = pd.read_csv(out)
    assert set(df.columns) == {"proba", "label"}
    assert df["proba"].between(0.0, 1.0).all()
    assert df["label"].isin([0, 1]).all()
    assert not df.isna().any().any()


@pytest.mark.smoke
def test_cli_inference_thershold(tmp_path: Path, small_raw_df: pd.DataFrame, model_path: Path):
    inp = tmp_path / "mini.csv"
    out1 = tmp_path / "pred_t05.csv"
    out2 = tmp_path / "preds_t07.csv"
    small_raw_df.to_csv(inp, index=False)

    base = [
        sys.executable,
        "-m",
        MODEL,
        "--input",
        str(inp),
        "--model",
        str(model_path),
        "--proba",
        "--also-label",
    ]
    r1 = _run(base + ["--output", str(out1), "--threshold", "0.5"])
    r2 = _run(base + ["--output", str(out2), "--threshold", "0.7"])
    assert r1.returncode == 0 and r2.returncode == 0, r1.stdout + r1.stderr + r2.strout + r2.stderr

    d1 = pd.read_csv(out1)
    d2 = pd.read_csv(out2)
    # при более высоком пороге  обычно <= количество positive #
    assert d2["label"].sum() <= d1["label"].sum()


def test_cli_inference_missing_input(tmp_path: Path, model_path: Path):
    missing = tmp_path / "no.csv"
    out = tmp_path / "preds.csv"
    cmd = [
        sys.executable,
        "-m",
        MODEL,
        "--input",
        str(missing),
        "--output",
        str(out),
        "--model",
        str(model_path),
        "--proba",
    ]
    r = _run(cmd)
    assert r.returncode != 0
    assert ("not found" in (r.stdout + r.stderr).lower()) or (
        "cannot open" in (r.stdout + r.stderr).lower()
    )


def test_cli_inference_empty_csv(tmp_path: Path, model_path: Path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    out = tmp_path / "preds.csv"
    cmd = [
        sys.executable,
        "-m",
        MODEL,
        "--input",
        str(empty),
        "--output",
        str(out),
        "--model",
        str(model_path),
        "--proba",
    ]
    r = _run(cmd)
    assert r.returncode != 0
    msg = (r.stdout + r.stderr).lower()
    assert ("empty" in msg) or ("no columns" in msg)
