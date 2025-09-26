import importlib

req = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "joblib",
    "lightgbm",
    "xgboost",
    "fairlearn",
    "shap",
    "pyarrow",
]

missing = []
for m in req:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, str(e)))

if missing:
    print("MISSING:", missing)
else:
    print("OK: all required imports available")
