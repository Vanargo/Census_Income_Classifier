from pathlib import Path

from joblib import dump
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
csv = HERE / "minidata.csv"

df = pd.read_csv(csv)

# модель для smoke: hours-per-week >= 40 -> 1, иначе 0 #
y = (df["hours-per-week"] >= 40).astype(int)

cat = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
num = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

preproc = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num),
    ]
)

pipe = Pipeline([("preproc", preproc), ("clf", LogisticRegression(max_iter=1000))])

pipe.fit(df, y)
dump(pipe, HERE / "micro_model.joblib")
print("[done] tests/fixtures/micro_model.joblib created.")
