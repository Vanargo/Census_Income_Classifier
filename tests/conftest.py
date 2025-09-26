import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TESTS_DIR = PROJECT_ROOT / "tests"
FIXTURES_DIR = TESTS_DIR / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

RAW_COLS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]


def _micro_raw_df_fallback(n_rows: int = 8) -> pd.DataFrame:
    # минимальный встроенный датасет Adult-like (8 строк, 14 колонок) #
    rows = [
        dict(
            age=39,
            workclass="State-gov",
            fnlwgt=77516,
            education="Bachelors",
            education_num=13,
            **{"education-num": 13},
            **{"marital-status": "Never-married"},
            occupation="Adm-clerical",
            relationship="Not-in-family",
            race="White",
            sex="Male",
            **{"capital-gain": 2174},
            **{"capital-loss": 0},
            **{"hours-per-week": 40},
            **{"native-country": "United-States"},
        ),
        dict(
            age=50,
            workclass="Self-emp-not-inc",
            fnlwgt=83311,
            education="Bachelors",
            education_num=13,
            **{"education-num": 13},
            **{"marital-status": "Married-civ-spouse"},
            occupation="Exec-managerial",
            relationship="Husband",
            race="White",
            sex="Male",
            **{"capital-gain": 0},
            **{"capital-loss": 0},
            **{"hours-per-week": 13},
            **{"native-country": "United-States"},
        ),
        dict(
            age=38,
            workclass="Private",
            fnlwgt=215646,
            education="HS-grad",
            education_num=9,
            **{"education-num": 9},
            **{"marital-status": "Divorced"},
            occupation="Handlers-cleaners",
            relationship="Not-in-family",
            race="White",
            sex="Male",
            **{"capital-gain": 0},
            **{"capital-loss": 0},
            **{"hours-per-week": 40},
            **{"native-country": "United-States"},
        ),
        dict(
            age=53,
            workclass="Private",
            fnlwgt=234721,
            education="11th",
            education_num=7,
            **{"education-num": 7},
            **{"marital-status": "Married-civ-spouse"},
            occupation="Handlers-cleaners",
            relationship="Husband",
            race="Black",
            sex="Male",
            **{"capital-gain": 0},
            **{"capital-loss": 0},
            **{"hours-per-week": 40},
            **{"native-country": "United-States"},
        ),
        dict(
            age=28,
            workclass="Private",
            fnlwgt=338409,
            education="Bachelors",
            education_num=13,
            **{"education-num": 13},
            **{"marital-status": "Married-civ-spouse"},
            occupation="Prof-specialty",
            relationship="Wife",
            race="Black",
            sex="Female",
            **{"capital-gain": 0},
            **{"capital-loss": 0},
            **{"hours-per-week": 40},
            **{"native-country": "Cuba"},
        ),
        dict(
            age=37,
            workclass="Private",
            fnlwgt=284582,
            education="Masters",
            education_num=14,
            **{"education-num": 14},
            **{"marital-status": "Married-civ-spouse"},
            occupation="Exec-managerial",
            relationship="Wife",
            race="White",
            sex="Female",
            **{"capital-gain": 0},
            **{"capital-loss": 0},
            **{"hours-per-week": 40},
            **{"native-country": "United-States"},
        ),
        dict(
            age=49,
            workclass="Private",
            fnlwgt=160187,
            education="9th",
            education_num=5,
            **{"education-num": 5},
            **{"marital-status": "Married-spouse-absent"},
            occupation="Other-service",
            relationship="Not-in-family",
            race="Black",
            sex="Female",
            **{"capital-gain": 0},
            **{"capital-loss": 0},
            **{"hours-per-week": 16},
            **{"native-country": "Jamaica"},
        ),
        dict(
            age=52,
            workclass="Self-emp-not-inc",
            fnlwgt=209642,
            education="HS-grad",
            education_num=9,
            **{"education-num": 9},
            **{"marital-status": "Married-civ-spouse"},
            occupation="Exec-managerial",
            relationship="Husband",
            race="White",
            sex="Male",
            **{"capital-gain": 0},
            **{"capital-loss": 0},
            **{"hours-per-week": 45},
            **{"native-country": "United-States"},
        ),
    ]
    df = pd.DataFrame(rows)
    # нормализация имен #
    if "education_num" in df.columns and "education-num" not in df.columns:
        df = df.rename(columns={"education_num": "education-num"})
    return df[RAW_COLS]


@pytest.fixture(scope="session")
def small_raw_df() -> pd.DataFrame:
    """
    8 строк сырых фичей Adult для smoke-тестов препроцессинга/инференса.
    Приоритет: tests/fixtures/minidata.csv ->
    data/processed/*.csv -> data/processed/*.parquet -> встроенный DF.
    """
    # fixtures CSV (герметично для CI) #
    fx_csv = FIXTURES_DIR / "minidata.csv"
    if fx_csv.exists():
        df = pd.read_csv(fx_csv, sep=None, engine="python")
    else:
        # project CSV
        proj_csv = PROJECT_ROOT / "data" / "processed" / "adult_eda.csv"
        proj_par = PROJECT_ROOT / "data" / "processed" / "adult_eda.parquet"
        if proj_csv.exists():
            df = pd.read_csv(proj_csv)
        elif proj_par.exists():
            df = pd.read_parquet(proj_par)
        else:
            # встроенный fallback
            df = _micro_raw_df_fallback()

    # отброс целевой переменной #
    drop_cols = [c for c in ["income", "income_bin"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    have = [c for c in RAW_COLS if c in df.columns]
    df = df[have].head(8).copy()
    return df


def _find_first(*candidates: Path) -> Path | None:
    for p in candidates:
        if p and p.exists():
            return p
    return None


@pytest.fixture(scope="session")
def model_path() -> pd.DataFrame:
    """
    Возврат пути к pipeline/модели.
    Приоритет: tests/fixtures/micro_model.joblib ->
    data/models/model_best.joblib ->
    notebooks/... и т.д.
    Если ничего не найдено - помечаем тест как skipped (интеграционные артефакты).
    """
    fx = _find_first(FIXTURES_DIR / "micro_model.joblib")
    if fx:
        return fx

    candidates = [
        PROJECT_ROOT / "data" / "models" / "model_best.joblib",
        PROJECT_ROOT / "notebooks" / "data" / "models" / "model_best.joblib",
        PROJECT_ROOT / "data" / "models" / "lgb_best.joblib",
        PROJECT_ROOT / "notebooks" / "data" / "models" / "lgb_best.joblib",
    ]
    mp = _find_first(*candidates)
    if mp:
        return mp

    pytest.skip(
        "Model artifact not found - provide tests/fixtures/micro_model.joblib "
        "or build real artifacts."
    )
