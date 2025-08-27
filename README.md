# Census Income Classifier

## Overview
This project predicts whether an individual's annual income exceeds $50K based on census data (UCI Adult dataset).
The pipeline includes:
- exploratory Data Analysis (EDA);
- training and evaluation of multiple ML models;
- fairness assessment across sensitive groups;
- model explainability with SHAP.

The goal is to demonstrate a full Data Science workflow: from raw dataset to deployed model artifacts, with careful attention to fairness and transparency.

---

## Project Structure
- data/ - Raw, interim, and processed datasets
    - raw/ - Original UCI Adult dataset files
    - processed/ - Cleaned & feature-engineered versions
- notebooks/ - Jupyter notebooks with full workflow
    - 01_data_loading_and_eda.ipynb
    - 02_modeling.ipynb
    - 03_fairness_and_explainability.ipynb
    - artifacts/ - Saved artifacts from trainig/testing
    - models/ - Serialized trained models
- reports/ - Figures and plots
    - figures_03/ - Fairness & explainability plots
- src/ - Source code
    - models/infer.py - Inference script (to be explained)
- models/ - Top-level folder for trained models (if any)
- requirements.txt - Python dependencies
- README.md - Project documentation
- .gitignore - Git ignore file

---

## Dataset
- **Source:** [UCI ML Repository - Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Files used:**
    - `data/raw/adult.data` (training data, ~32K rows)
    - `data/raw/adult.test` (test data, ~16K rows)

The dataset contains demographic and employment-related features, such as age, education, occupation, sex, and hours-per-week, with a binary label:
- '<=50K'
- '>50K'

---

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

---

## Installation
```bash
git clone <repo-url>
cd '1. Census Income Classifier'
python -m venv venv
# Activate:
# Linux/Mac: source venv/bin/activate
# Windows: venv/Scripts/activate
pip install -r requirements.txt
```

## Results
### Model Performance
| Model     | AUC   | F1    | Precision | Recall | Accuracy |
|-----------|-------|-------|-----------|--------|----------|
| LGBM_best | 0.93  | 0.72  | 0.79      | 0.66   | 0.80     |
| XGB_es    | 0.93  | 0.72  | 0.79      | 0.66   | 0.88     |
| RF_best   | 0.92  | 0.69  | 0.81      | 0.59   | 0.81     |
| LogReg    | 0.91  | 0.67  | 0.74      | 0.61   | 0.74     |

### Fairness
Pareto curve shows trade-off between F1 and Demographic Parity;
Example: ![Pareto plot](reports/figures_03/Pareto_f1_vs_dp.png)

### Explainability
SHAP summary plots hightlight top features (education, occupation, hours-per-week, age).

## Limitations
- sensitive attributes analyzed: sex, age, education only;
- some metric inconsistencies detected (accuracy = precision in some rows);
- fairness post-processing may reduce overall accuracy;
- inference script is currently a stub and must be extended.

## Roadmap
- [ ] Fix metric bug in results table;
- [ ] fill `requirements.txt` with pinned versions;
- [ ] expand `infer.py` into a full CLI tool;
- [ ] save EDA and modeling plots into `reports/figures_01/` and `reports/figures_02/`;
- [ ] add unit tests for preprocessing and inference;
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
python src/models/infer.py --input data/raw/sample.csv --output predictions.csv
```