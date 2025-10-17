# Copyright (c) 2025 Vanargo
# Licensed under the MIT License. See LICENSE in the project root.

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load


def _is_pipeline(obj) -> bool:
    """Heuristic: sklearn Pipeline has attribute 'named_steps' (dict-like)"""
    return hasattr(obj, "named_steps") and isinstance(obj.named_steps, dict)


def _has_predict_proba(model) -> bool:
    return hasattr(model, "predict_proba")


def _load_model(model_path: Path):
    """
    Гибкая загрузка артефактов модели.
    Возвращает пару (model_or_pipe, preproc_or_None), где:
    - если есть полноценный sklearn Pipeline -> (pipe, None);
    - если сохранен clf и (опционально) preproc -> (clf, preproc).
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    base_dir = model_path.parent

    def _maybe_load_pointer(x, depth=0):
        """
        Если x - строка/путь/директория - попытка распаковать:
        - если это файл - joblib.load;
        - если это директория - поиск типовых имен .jolib.
        Иначе - вернуть как есть.
        """
        if depth > 3:
            # защита от циклов
            return None

        if isinstance(x, str | Path):
            p = Path(x)
            if not p.is_absolute():
                p = base_dir / p

            if p.is_file():
                # попытка загрузить любой файл
                try:
                    return load(p)
                except Exception:
                    return x

            if p.is_dir():
                # поиск типовых pipeline/моделей
                candidates = [
                    "pipeline.joblib",
                    "model_pipeline.joblib",
                    "pipe.joblib",
                    "model.joblib",
                    "model_best.joblib",
                    "clf.joblib",
                    "estimator.joblib",
                ]
                candidates += [q.name for q in sorted(p.gloab("*.joblib"))]
                for name in candidates:
                    cand = p / name
                    if cand.exists():
                        try:
                            return load(cand)
                        except Exception:
                            continue
        return x

    def _extract_from(obj):
        # pointer - загрузка #
        obj = _maybe_load_pointer(obj)

        # если sklearn Pipeline #
        if _is_pipeline(obj):
            return obj, None

        # если оценщик (есть predict/predict_proba) - без препроцессора #
        if hasattr(obj, "predict") or hasattr(obj, "predict_proba"):
            return obj, None

        # контейнер-объект с атрибутами (SimpleNamespace и т.п.) #
        for attr in ("pipe", "pipeline"):
            if hasattr(obj, attr) and _is_pipeline(getattr(obj, attr)):
                return getattr(obj, attr), None
        # clf + preproc как атрибуты #
        if hasattr(obj, "__dict__"):
            clf = None
            preproc = None
            for attr in ("clf", "classifier", "model", "estimator"):
                v = getattr(obj, attr, None)
                v = _maybe_load_pointer(v)
                if v is not None and (hasattr(v, "predict") or hasattr(v, "predict_proba")):
                    clf = v
                    break
            for attr in ("preproc", "preprocessor", "transformer"):
                v = getattr(obj, attr, None)
                v = _maybe_load_pointer(v)
                if v is not None and hasattr(v, "transform"):
                    preproc = v
                    break
            if clf is not None:
                return clf, preproc

        # dict #
        if isinstance(obj, dict):
            # попытка найти pipeline
            for k in ("pipe", "pipeline", "model", "estimator"):
                if k in obj:
                    v = _maybe_load_pointer(obj[k])
                    if _is_pipeline(v):
                        return v, None
            # попытка найти clf + preproc
            clf = None
            preproc = None
            for k in ("clf", "classifier", "model", "estimator"):
                if k in obj:
                    v = _maybe_load_pointer(obj[k])
                    if v is not None and (hasattr(v, "predict") or hasattr(v, "predict_proba")):
                        clf = v
                        break
            for k in ("preproc", "preprocessor", "transformer"):
                if k in obj:
                    v = _maybe_load_pointer(obj[k])
                    if v is not None and hasattr(v, "transform"):
                        preproc = v
                        break
            if clf is not None:
                return clf, preproc

            path_like_keys = (
                "pipeline_path",
                "model_path",
                "pipe_path",
                "estamator_path",
                "path",
                "artifact",
                "artifact_path",
            )
            container_like_keys = ("artifacts", "models", "paths")

            # прямые путевые ключи
            for k in path_like_keys:
                if k in obj:
                    v = _maybe_load_pointer(obj[k], depth=1)
                    if v is not None:
                        extracted = _extract_from(v)
                        if extracted != (None, None):
                            return extracted

            # вложенные словари с путями
            for ck in container_like_keys:
                if ck in obj and isinstance(obj[k], dict):
                    for _, vv in obj[ck].items():
                        v = _maybe_load_pointer(vv, depth=1)
                        if v is not None:
                            extracted = _extract_from(v)
                            if extracted != (None, None):
                                return extracted

            # путевые ключи
            for k in ("pipeline_path", "model_path", "pipe_path", "estimator_path"):
                if k in obj:
                    v = _maybe_load_pointer(obj[k], depth=1)
                    if v is not None:
                        return _extract_from(v)

        # итерируемые контейнеры: список/кортеж/набор #
        if isinstance(obj, list | tuple | set):
            # попытка найти pipeline
            for v in obj:
                v = _maybe_load_pointer(v)
                if _is_pipeline(v):
                    return v, None
            # попытка найти clf + preproc
            clf = None
            preproc = None
            for v in obj:
                v = _maybe_load_pointer(v)
                if hasattr(v, "predict") or hasattr(v, "predict_proba"):
                    clf = v
                    break
            for v in obj:
                v = _maybe_load_pointer(v)
                if hasattr(v, "transform") and not _is_pipeline(v):
                    preproc = v
                    break
            if clf is not None:
                return clf, preproc

        # fallback: по типичным именам рядом с моделью #
        for name in (
            "pipeline.joblib",
            "model_pipeline.joblib",
            "pipe.joblib",
            "model.joblib",
            "clf.joblib",
            "estimator.joblib",
        ):
            cand = base_dir / name
            if cand.exists():
                v = load(cand)
                if _is_pipeline(v):
                    return v, None
                if hasattr(v, "predict") or hasattr(v, "predict_proba"):
                    return v, None

        return None, None

    obj = load(model_path)
    model, preproc = _extract_from(obj)
    if model is not None:
        return model, preproc

    if os.environ.get("INFER_DEBUG", "0") == "1":
        print("[debug] type(obj):", type(obj))
        try:
            if isinstance(obj, dict):
                print("[debug] dict_keys:", list(obj.keys())[:20])
        except Exception:
            pass
        try:
            attrs = [a for a in dir(obj) if not a.startswith("_")]
            print("[debug] attrs:", attrs[:30])
        except Exception:
            pass

    raise ValueError(
        f"Unrecognized model artifact structure in {model_path}. "
        f"Expected Pipeline, estimator, or dict/tuple with keys."
    )


def run_inference(
    df: pd.DataFrame,
    model_path: Path,
    proba: bool = True,
    threshold: float = 0.5,
    also_label: bool = True,
) -> pd.DataFrame:
    """
    Выполнить инференс по готовому DataFrame с сырыми фичами Adult.
    Возвращает DataFrame с колонками: ['proba', 'label'] или ['label'].
    """
    model, preproc = _load_model(model_path)

    # если pipeline - просто скормим ему df #
    if _is_pipeline(model):
        if proba and _has_predict_proba(model):
            preds = model.predict_proba(df)[:, 1]
        else:
            preds = model.predict(df)
    else:
        # не pipeline: clf (+/- preproc)
        if not hasattr(model, "predict") and not hasattr(model, "predict_proba"):
            raise TypeError("Loaded artifact is not a pipeline and has no predict/predict_proba.")

        X = preproc.transform(df) if preproc is not None else df

        if proba and _has_predict_proba(model):
            preds = model.predict_proba(X)[:, 1]
        else:
            preds = model.predict(X)

    # сборка выходного DataFrame #
    if proba:
        out = pd.DataFrame({"proba": np.asarray(preds, dtype=float)})
        if also_label:
            out["label"] = (out["proba"].values >= threshold).astype(int)
    else:
        out = pd.DataFrame({"label": np.asarray(preds, dtype=int)})
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference CLI for Census Income Classifier.")
    p.add_argument("--input", required=True, help="Path to input CSV with raw features.")
    p.add_argument(
        "--model",
        default="data/models/model_best.joblib",
        help="Path to model artifact (Pipeline preferred).",
    )
    p.add_argument("--output", required=True, help="Where to write predictions CSV.")
    p.add_argument("--proba", action="store_true", help='Output probabilities (column "proba").')
    p.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for label if --proba is set."
    )
    p.add_argument(
        "--also-label", action="store_true", help="When --proba used, also write hard label."
    )
    return p.parse_args()


def main():
    args = _parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    model_path = Path(args.model)

    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    try:
        if not in_path.exists():
            print(f"[error] Input file not found: {in_path}", file=sys.stderr)
            sys.exit(2)
        if in_path.suffix.lower() not in {".csv", ".json"}:
            print(
                f"[warn] Unrecognized extension {in_path.suffix}; trying to read as CSV.",
                file=sys.stderr,
            )

        if in_path.suffix.lower() == ".json":
            df = pd.read_csv(in_path)
        else:
            # пустой файл -> pandas поднимает ошибку #
            df = pd.read_csv(in_path)
        if df is None or df.shape[0] == 0 or df.shape[1] == 0:
            print("[error] Empty input data.", file=sys.stderr)
            sys.exit(3)
    except Exception as e:
        print(f"[error] Cannot read input: {e}", file=sys.stderr)
        sys.exit(4)

    out = run_inference(
        df=df,
        model_path=model_path,
        proba=args.proba,
        threshold=args.threshold,
        also_label=args.also_label,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[done] Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
