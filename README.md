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
    - raw/ # Исходные данные (Adult dataset)
    - processed/ # Обработанные данные (train/test splits и пр.)
    - artifacts/ # Модели и артефакты обучения/тестирования
        - lgb_best.joblib # (estimator) лучший LGBM без препроцессинга
        - X_test_enc.npz # закодированный тест (для голых моделей)
        - fairness_thresold_scan.csv # экспорт скан-ия порогов (fariness)
    - models/
    - model_best.joblib # (pipeline) лучший пайплайн=препроцессинг+модель
- notebooks/
    - 01_data_loading_and_eda.ipynb
    - 02_modeling.ipynb
    - 03_fairness_and_explainability.ipynb
    - artifacts/ # Артефакты, сохраненные из ноутбуков
    - models/
- src/
    - models/
        - infer.py # CLI интерфейса (см. раздел 'How to run unference')
    - utils.py # вспомогательные функции
- tests/
    - conftest.py # фикстуры для unit-тестов
    - test_infer_cli.py # smoke-тесты CLI
    - test_infer_func.py # smoke-тесты функций инференса
    - test_preprocessing.py # smoke-тест препроцессора
    - fixtures/
        - minidata.csv
        - micro_model.joblib
- reports/
    - figures_01/ # ключевые графики EDA
        - eda_correlation_numeric.png
        - eda_numeric_distributions.png
    - figures_02/ # ключевые графики моделинга
        - roc_curve.png
        - pr_curve.png
        - calibration_curve.png
        - confusion_matrix.png
        - feature_importance_lgbm.png
        - feature_importance_xgb_gain.png
    - figures_03/ # fairness/explainability
        - Pareto_f1_vs_dp.png
- requirements.txt
- requirements-dev.txt
- environments.yml
- README.md


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
# опционально: pre-commit для локальных проверок
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
# опционально: собрать микромодель для смоук-тестов CLI/инференса
python tests/fixtures/build_micro_model.py
# запустить тесты
pytest -q
# smoke-тесты инференса пропускаются, если не предоставлен артефакт
# `tests/fixtures/micro_model.joblib` или не собрана реальная модель (`models/model_best.joblib`).
```

### 4. Run notebooks
Выполните по порядку:
1. `notebooks/01_data_loading_and_eda.ipynb`.
2. `notebooks/02_modeling.ipynb` (сохраняет модель в `models/` и артефакты в `artifacts/`).
3. `notebooks/03_fairness_and_explainability.ipynb` (выводы и графики в `reports/`).

### 5. Paths
Проект использует модуль paths.py с абсолютными путями от `ROOT`:
- данные: `data/{raw,interim,processed}`;
- программный код и ноутбуки: `src/`, `notebooks/`;
- модели и артефакты: `models/`, `artifacts/`;
- отчеты: `reports/`.


## Results
### Model Performance
| Model     | AUC   | F1    | Precision | Recall | Accuracy |
|-----------|-------|-------|-----------|--------|----------|
| LGBM_best | 0.93  | 0.72  | 0.79      | 0.66   | 0.80     |
| XGB_es    | 0.93  | 0.72  | 0.79      | 0.66   | 0.88     |
| RF_best   | 0.92  | 0.69  | 0.81      | 0.59   | 0.81     |
| LogReg    | 0.91  | 0.67  | 0.74      | 0.61   | 0.74     |

### Fairness
Мы анализируем поведение модели для чувствительных групп (например, `sex`, `race`, `age_group`). Ниже 2-3 ключевых графика, которые показывают влияние выбора порога классификации на качество и разницу показателей между группами.

**Ключевые визуализации:**
- **Pareto F1-DP:** компромисс между качеством (F1) и разницей по демографическому паритету (DP).
    ![Pareto F1-DP](img_pareto)
- **Accuracy/F1 vs threshold:** как глобальные метрики меняются при варьировании порога.
    ![Accuracy/F1 vs threshold](img_f1_thr)
- **DP vs threshold:** как меняется разница долей положительных предсказаний между группами при изменении порога.
    ![DP vs threshold](img_dp_thr)

*Дополнительно:*
- **EOD vs threshold:** поведение Equal Opportunity Difference при варьировании порога.
    ![EOD vs threshold](img_eod_thr)
- **Calibration by group:** надежность вероятностей по чувствительным группам (пример: `sex`).
    ![Calibration by sex](img_calib_sex)

**Краткий анализ.**
1. График Pareto F1-DP показывает, что при повышении F1 Score модель теряет баланс по Demographic Parity: улучшение точности сопровождается ростом разницы в положительных предсказаниях между группами. Это отражает неизбежный компромисс между общей производительностью и справедливостью классификатора.
2. Кривые DP vs threshold и EOD vs threshold демонстрируют, что выбор порога сильно влияет на разрыв между группами: при низких порогах (t < 0.4) модель переоценивает вероятность положительного исхода у доминирующей группы, а при высоких (t > 0.6) - занижает ее у менее представленных. Оптимальная зона находится около t ~ 0.5, где одновременно удерживаются приемлемые значения F1 и снижается EOD.
3. Совокупно эти визуализации подтверждают, что fairness-коррекция (например, подбор порога или reweighting) может существенно улучшить баланс метрик между чувствительными группами без серьезного ущерба по качеству модели.

Полный набор рисунков и CSV доступен в разделе **Assets & Releases** релиза `taskB-2025-10-24`.

### Explainability
SHAP summary plots hightlight top features (education, occupation, hours-per-week, age).


## Testing

Unit-тесты реализованы в папке `tests/`. Они включают:
- smoke-тесты CLI (`test_infer.py`);
- smoke-тесты функций инференса (`test_infer_func.py`);
- smoke-тест препроцессора (`test_preprocessing.py`).

Запуск всех тестов:
```bash
pytest -q
```


## Limitations
- sensitive attributes analyzed: sex, age, education only;
- some metric inconsistencies detected (accuracy = precision in some rows);
- fairness post-processing may reduce overall accuracy;
- inference script реализован как рабочий CLI (`src/models/infer.py`), поддерживает опции `--proba`, `--threshold`, `also-label`.


## Roadmap
- [ ] Fix metric bug in results table;
- [ ] fill `requirements.txt` with pinned versions;
- [x] expand `infer.py` into a full CLI tool;
- [x] save EDA and modeling plots into `reports/figures_01/` and `reports/figures_02/`;
- [x] add unit tests for preprocessing and inference;
- [ ] export HTML reports of notebooks.


## Usage

### Run Notebooks
Execute Jupyter notebooks in sequence:
1. `notebooks/01_data_loading_and_eda.ipynb`.
2. `notebooks/02_modeling.ipynb`.
3. `notebooks/03_fairness_and_explainability.ipynb`.

### Inference (after expanding `infer.py`)
Run predictions on new data:
```bash
# Инференс из уже закодированных признаков (X_test_enc.npz) и обученной модели:
python - <<'PY'
import numpy as np
import scipy.sparse as sp
from joblib import load
from pathlib import Path

# скачайте артефакты из релиза `taskB-2025-10-24` в локальный каталог, затем укажите путь: #
model_p = Path("data/artifacts/lgb_best.joblib") # после скачивания из releases
x_enc_p = Path("data/artifacts/X_test_enc.npz") # после скачивания из releases
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
pythion -m src.models.infer \
    --model data/artifacts/lgb_best.joblib \  # файл скачан из Releases
    --input data/artifacts/X_test_enc.npz \   # файл скачан из Releases
    --output predictions/preds_lgbm.csv \
    --proba --also-label --threshold 0.5
```


## Artifacts & Releases

Тяжелые артефакты не хранятся в Git-истории. Они исключены правилами `.gitignore`:
- `data/artifacts/**`, `data/models/**`
- `reports/**/*.html`, `reports/figures_*/**`
- бинарные массивы в `data/**/*.npy`, `data/**/*.npz`
- промежуточные `data/processed/*_eda.csv`

Актуальные выкладки доступны на странице Releases репозитория.  
-> **Releases:** https://github.com/Vanargo/Census_Income_Classifier/releases

### Что есть в Releases
- пакеты с готовыми артефактами инференса:
    - модели: `models/*.joblib`;
    - артефакты препроцессинга: `artifacts/*.npy` (например, `X_test_enc.npy`);
- HTML-отчеты ноутбуков.

### Как воспроизвести локально без артефактов
1. Подготовить окружение и данные (см. "Installation & Reproducibility").
2. Запустить ноутбуки по порядку:
   - `01_data_loading_and_eda.ipynb` - подготовка данных;
   - `02_modeling.ipynb` - обучение и сохранение моделей в `models/` и артефактов в `artifacts/`;
   - `03_fairness_and_explainability.ipynb` - генерация отчетов и графиков в `reports/`.

> Примечание: сырьевые файлы набора Adult (UCI) не коммитятся. Положите их в `data/raw/` или используйте шаг загрузки в `01_*` ноутбуке.

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