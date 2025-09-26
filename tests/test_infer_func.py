from pathlib import Path

from src.models.infer import run_inference


def test_run_inference_proba_and_label(small_raw_df, model_path: Path):
    out = run_inference(
        df=small_raw_df,
        model_path=model_path,
        proba=True,
        threshold=0.5,
        also_label=True,
    )

    # структура #
    assert set(out.columns) >= {
        "proba",
        "label",
    }, "Must have proba and label when proba+also_label."
    assert len(out) == len(small_raw_df) > 0, "Output rows must match input rows."

    # диапазоны #
    assert out["proba"].between(0.0, 1.0).all(), "Probabilities must be in [0, 1]."
    assert out["label"].isin([0, 1]).all(), "Labels must be binary 0/1."

    # NaN контроль #
    assert not out.isna().any().any(), "No NaNs expected in output."


def test_run_inference_labels_only(small_raw_df, model_path: Path):
    out = run_inference(
        df=small_raw_df,
        model_path=model_path,
        proba=False,
    )
    assert list(out.columns) == ["label"]
    assert out["label"].isin([0, 1]).all()
