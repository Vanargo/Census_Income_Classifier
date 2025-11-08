# Census Income Classifier

[![CI](https://github.com/Vanargo/Census_Income_Classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Vanargo/Census_Income_Classifier/actions/workflows/ci.yml)
[-> Changelog](./CHANGELOG.md)


## Overview
This project predicts whether an individual's annual income exceeds $50K based on census data (UCI Adult dataset).
The pipeline includes:
- exploratory Data Analysis (EDA);
- training and evaluation of multiple ML models;
- fairness assessment across sensitive groups;
- model explainability with SHAP.

The goal is to demonstrate a full Data Science workflow: from raw dataset to deployed model artifacts, with careful attention to fairness and transparency.


## Project Structure
- data/
    - raw/        # raw data (Adult dataset)
    - processed/  # processed data (train/test splits и пр.)
    - interim/    # prepared datasets for EDA and modeling
- notebooks/
    - 01_data_loading_and_eda.ipynb
    - 02_modeling.ipynb
    - 03_fairness_and_explainability.ipynb
- src/
    - models/
        - infer.py  # CLI interface (see section 'How to run inference')
        - __init__.py
    - __init__.py
- tests/
    - conftest.py
    - test_infer_cli.py
    - test_infer_func.py
    - test_preprocessing.py
    - fixtures/
        - minidata.csv
        - micro_model.joblib
- artifacts/  # training/evaluation artifacts (local or downloaded from GitHub Releases)
- models/     # trained models (local or downloaded from GitHub Releases)
- reports/
    - figures_01/ # key EDA figures
    - figures_02/ # key modeling figures
    - figures_03/ # fairness/explainability figures
- .github/  # CI configuration (GitHub Actions)
- requirements.txt
- requirements-dev.txt
- environment.yml
- README.md
- CHANGELOG.md


## Dataset
- **Source:** [UCI Machine Learning Repository - Adult (Census Income) dataset](https://archive.ics.uci.edu/dataset/2/adult)
- **Authors:** Barry Becker, Ronny Kohavi
- **License:** CC BY 4.0
- **DOI:** 10.24432/C5XW20
- **Files used:**
    - `data/raw/adult.data` (training data, ~32K rows)
    - `data/raw/adult.test` (test data, ~16K rows)

The dataset contains demographic and employment-related features, such as age, education, occupation, sex, and hours-per-week, with a binary label:
- '<=50K'
- '>50K'

If you use this dataset, please cite the UCI repository page above.


## Pipeline

### 1. Exploratory Data Analysis (EDA)
Notebook: `01_data_loading_and_eda.ipynb`
- loads raw dataset;
- cleans missing values and formats categorical variables;
- explores distributions, correlations, and potential outliers;
- saves processed dataset to `data/processed/adult_eda.csv` and `.parquet`.

### 2. Modeling
Notebook: `02_modeling.ipynb`
- splits data into train/test;
- applies unified preprocessing ('ColumnTransformer' with encoders/scalers);
- trains multiple models: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM;
- performs hyperparameter tuning (RandomizedSearch);
- compares models with AUC, F1, precision, recall, accuracy;
- saves the best model (LightGBM) and artifacts (features, metrics, predictions).

### 3. Fairness & Explainability
Notebook: `03_fairness_and_explainability.ipynb`
- loads best model and artifacts;
- defines sensitive features (sex, age, education);
- evaluates fairness metrics (Demographic Parity, Equalized Odds);
- explores thresholds trade-offs with Pareto curves;
- applies 'ThresholdOptimizer' for post-processing fairness mitigation;
- explains predictions using SHAP (global and local).


## Installation & Reproducibility

### 1. Create environment
```powershell
# conda
conda create -n census_ds2 python=3.10 -y
conda activate census_ds2
pip install -r requirements.txt
# optionally: enable pre-commit for local checks
pre-commit install
```
### 2. Code quality
```powershell
ruff check . --fix
ruff format .
black .
```

### 3. Tests
```powershell
# optionally: build a micro model for CLI/inference smoke tests
python tests/fixtures/build_micro_model.py
# run all tests
pytest -q
# smoke-tests for inference are skipped if the artifact
# `tests/fixtures/micro_model.joblib` is missing or no real model (`models/model_best.joblib`) is built.
```

### 4. Run notebooks
Execute in order:
1. `notebooks/01_data_loading_and_eda.ipynb`.
2. `notebooks/02_modeling.ipynb` (saves the model to `models/` and artifacts to `artifacts/`).
3. `notebooks/03_fairness_and_explainability.ipynb` (outputs and figures go to `reports/`).

### 5. Paths
The project uses the `paths.py` module with absolute paths from `ROOT`:
- data: `data/{raw,interim,processed}`;
- source code and notebooks: `src/`, `notebooks/`;
- models and artifacts: `models/`, `artifacts/`;
- reports: `reports/`.


## Results
### Model Performance
| Model     | AUC   | F1    | Precision | Recall | Accuracy |
|-----------|-------|-------|-----------|--------|----------|
| LGBM_best | 0.93  | 0.72  | 0.79      | 0.66   | 0.80     |
| XGB_es    | 0.93  | 0.72  | 0.79      | 0.66   | 0.88     |
| RF_best   | 0.92  | 0.69  | 0.81      | 0.59   | 0.81     |
| LogReg    | 0.91  | 0.67  | 0.74      | 0.61   | 0.74     |

### Fairness
We analyze model behavior across sensitive groups (e.g., `sex`, `race`, `age_group`).  
Below are 2–3 key plots showing how classification threshold choice affects both performance and inter-group disparity.

**Key visualizations:**
- **Pareto F1-DP:** trade-off between performance (F1) and demographic parity difference (DP).
    ![Pareto F1-DP](img_pareto)
- **Accuracy/F1 vs threshold:** how global metrics vary with threshold changes.
    ![Accuracy/F1 vs threshold](img_f1_thr)
- **DP vs threshold:** how the share difference of positive predictions between groups changes with the threshold.
    ![DP vs threshold](img_dp_thr)

*Additional:*
- **EOD vs threshold:** Equal Opportunity Difference behavior under threshold variation.
    ![EOD vs threshold](img_eod_thr)
- **Calibration by group:** reliability of predicted probabilities per sensitive group (example: `sex`).
    ![Calibration by sex](img_calib_sex)

**Brief analysis.**
1. The Pareto F1–DP plot shows that as F1 increases, the model loses demographic balance: higher accuracy comes with a greater gap in positive prediction rates between groups. This illustrates the inherent trade-off between model performance and fairness.
2. The DP vs threshold and EOD vs threshold curves show that threshold choice strongly influences group disparity: at low thresholds (t < 0.4), the model overpredicts positives for the dominant group; at high thresholds (t > 0.6), it underpredicts for underrepresented groups. The optimal zone is around t ≈ 0.5, where both F1 and EOD remain acceptable.
3. Together, these plots confirm that fairness correction (e.g., threshold tuning or reweighting) can substantially improve metric balance across sensitive groups without significant performance degradation.

The full set of figures and CSV files is available under **Assets & Releases** for release `taskB-2025-10-24`.

### Explainability
SHAP summary plots hightlight top features (education, occupation, hours-per-week, age).


## Testing

Unit tests are implemented in the `tests/` directory. They include:
- CLI smoke tests (`test_infer.py`);
- inference function smoke tests (`test_infer_func.py`);
- preprocessor smoke test (`test_preprocessing.py`).

Run all tests:
```bash
pytest -q
```


## Limitations
- sensitive attributes analyzed: sex, age, education only;
- some metric inconsistencies detected (accuracy = precision in some rows);
- fairness post-processing may reduce overall accuracy;
- inference script is implemented as a functional CLI (`src/models/infer.py`), and supports the following options `--proba`, `--threshold`, `also-label`.


## Usage

### Run Notebooks
Execute Jupyter notebooks in sequence:
1. `notebooks/01_data_loading_and_eda.ipynb`.
2. `notebooks/02_modeling.ipynb`.
3. `notebooks/03_fairness_and_explainability.ipynb`.

### Inference (after expanding `infer.py`)
Run predictions on new data:
```bash
# Inference from pre-encoded features (X_test_enc.npz) and a trained model:
python - <<'PY'
import numpy as np
import scipy.sparse as sp
from joblib import load
from pathlib import Path

# download artifacts from release `taskB-2025-10-24` to a local directory, then specify the model path for inference:
model_p = Path("data/artifacts/lgb_best.joblib") # after downloading from releases
x_enc_p = Path("data/artifacts/X_test_enc.npz") # after downloading from releases
out_p = Path("predictions.csv")

X = sp.load_npz(x_enc_p)
clf = load(model_p)
proba = clf.predict_proba(X)[:, 1]
np.savetxt(out_p, proba, delimiter=",", header="proba", comments="")
print(f"Saved: {out_p.resolve()}")
PY
```


## How to run inference

After training, the best pipeline is exported to `models/model_best.joblib`.
Use the pipeline artifact for inference on raw CSV (preferred).

### Example: Pipeline
Use raw CSV with original features:
```bash
python -m src.models.infer ^
    --model models/model_best.joblib ^
    --input data/processed/adult_eda.csv ^
    --output predictions/preds_pipeline.csv ^
    --proba --also-label --threshold 0.5
```
Note: Bare estimators without a bundled preprocessor are not supported by the CLI.
Use a Pipeline artifact (`model_best.joblib`).

### Example B: Bare estimator
Use encoded test set (.npz):
```bash
python -m src.models.infer \
    --model data/artifacts/lgb_best.joblib \  # файл скачан из Releases
    --input data/artifacts/X_test_enc.npz \   # файл скачан из Releases
    --output predictions/preds_lgbm.csv \
    --proba --also-label --threshold 0.5
```


## Artifacts & Releases

cally, artifacts and models are stored in `artifacts/` and `models/`.  
Large files are not committed (see `.gitignore`) but are published with GitHub Releases:
/releases/v1.0.0/
    - models/
        - LGBM_best.joblib
        - XGBoost_ES_best.joblib
    - artifacts/
        - X_test_enc.npy
        - fairness_threshold_scan.csv
    - reports_html/
    - MANIFEST.md
When rerunning the pipeline (`02_modeling.ipynb`), artifacts are overwritten locally.  
For reproducibility, it is recommended to use the versions provided in the Releases.

The latest builds are available on the repository’s Releases page.  
→ **Releases:** https://github.com/Vanargo/Census_Income_Classifier/releases

### What’s included in Releases
- Packages with ready-to-use inference artifacts:
    - models: `models/*.joblib`;
    - preprocessing artifacts: `artifacts/*.npy` (e.g., `X_test_enc.npy`);
- HTML reports generated from notebooks.

### How to reproduce locally without artifacts
1. Prepare the environment and data (see "Installation & Reproducibility").
2. Run the notebooks in order:
   - `01_data_loading_and_eda.ipynb` - data preparation;
   - `02_modeling.ipynb` - training and saving models to `models/` and artifacts to `artifacts/`;
   - `03_fairness_and_explainability.ipynb` - generating reports and plots in `reports/`.

> Note: raw Adult dataset files (UCI) are not committed. Положите их в `data/raw/` Place them in `data/raw/` or use the data download step in the `01_*` notebook.

[img_pareto]: https://github.com/Vanargo/Census_Income_Classifier/releases/download/taskB-2025-10-24/Pareto_f1_vs_dp.png
[img_f1_thr]: https://github.com/Vanargo/Census_Income_Classifier/releases/download/taskB-2025-10-24/accuracy_f1_vs_threshold.png
[img_dp_thr]: https://github.com/Vanargo/Census_Income_Classifier/releases/download/taskB-2025-10-24/dp_vs_threshold.png
[img_eod_thr]: https://github.com/Vanargo/Census_Income_Classifier/releases/download/taskB-2025-10-24/eod_vs_threshold.png
[img_calib_sex]: https://github.com/Vanargo/Census_Income_Classifier/releases/download/taskB-2025-10-24/calibration_sex.png

[csv_group_metrics]: https://github.com/Vanargo/Census_Income_Classifier/releases/download/taskB-2025-10-24/group_metrics_t_star.csv
[csv_select_rates]: https://github.com/Vanargo/Census_Income_Classifier/releases/download/taskB-2025-10-24/selection_rates_t_star.csv
[csv_thr_scan]: https://github.com/Vanargo/Census_Income_Classifier/releases/download/taskB-2025-10-24/fairness_threshold_scan.csv
[csv_preds]: https://github.com/Vanargo/Census_Income_Classifier/releases/download/taskB-2025-10-24/preds_pipeline.csv


## Versioning & Changelog

This project follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
All notable changes are documented in [`CHANGELOG.md`](./CHANGELOG.md).