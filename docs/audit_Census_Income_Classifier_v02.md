# Audit Report — 1. Census Income Classifier_v02

Generated: 2025-08-26 14:11:03

## Project Tree (abridged)

```
1. Census Income Classifier_v02/
  .gitignore  (0 B)
  README.md  (0 B)
  requirements.txt  (0 B)
  .git/
  data/
    interim/
    processed/
      adult_eda.csv  (6547337 B)
      adult_eda.parquet  (611000 B)
    raw/
      adult.data  (3974305 B)
      adult.test  (2003153 B)
  docs/
  models/
  notebooks/
    01_data_loading_and_eda.ipynb  (1665247 B)
    02_modeling.ipynb  (126498 B)
    03_fairness_and_explainability.ipynb  (1510458 B)
    .ipynb_checkpoints/
      01_data_loading_and_eda-checkpoint.ipynb  (3204890 B)
    artifacts/
      X_test_enc.npy  (9065760 B)
      X_test_sensitive.csv  (278291 B)
      feature_names_after_preproc.csv  (3207 B)
      results_df.csv  (748 B)
      threshold_scan.csv  (4996 B)
      y_pred_best.npy  (78280 B)
      y_proba_best.npy  (78280 B)
      y_true_test.np.npy  (78280 B)
      y_true_test.npy  (78280 B)
    models/
      LGBM_best.joblib  (1289199 B)
      threshold_optimizer__dp__sex.joblib  (101677 B)
  reports/
    figures_03/
      Pareto_f1_vs_dp.png  (42311 B)
  src/
    data/
    features/
    models/
      infer.py  (34 B)
    visualization/
```

## File Inventory (all files)

                                                                 path  size_bytes
                                                  .git/COMMIT_EDITMSG          33
                                                            .git/HEAD          21
                                                          .git/config         190
                                                     .git/description          73
                                     .git/hooks/applypatch-msg.sample         478
                                         .git/hooks/commit-msg.sample         896
                                 .git/hooks/fsmonitor-watchman.sample        4726
                                        .git/hooks/post-update.sample         189
                                     .git/hooks/pre-applypatch.sample         424
                                         .git/hooks/pre-commit.sample        1649
                                   .git/hooks/pre-merge-commit.sample         416
                                           .git/hooks/pre-push.sample        1374
                                         .git/hooks/pre-rebase.sample        4898
                                        .git/hooks/pre-receive.sample         544
                                 .git/hooks/prepare-commit-msg.sample        1492
                                   .git/hooks/push-to-checkout.sample        2783
                                 .git/hooks/sendemail-validate.sample        2308
                                             .git/hooks/update.sample        3650
                                                           .git/index         297
                                                    .git/info/exclude         240
                                                       .git/logs/HEAD         177
                                            .git/logs/refs/heads/main         177
               .git/objects/48/49e326bdb13560756b19dbee5f95295d60ce99          88
               .git/objects/b1/64c51b634108e6b4933edc2c782d33c4090853         138
               .git/objects/e6/9de29bb2d1d6434b8b29ae775ad8c2e48c5391          15
                                                 .git/refs/heads/main          41
                                                           .gitignore           0
                                                            README.md           0
                                         data/processed/adult_eda.csv     6547337
                                     data/processed/adult_eda.parquet      611000
                                                  data/raw/adult.data     3974305
                                                  data/raw/adult.test     2003153
notebooks/.ipynb_checkpoints/01_data_loading_and_eda-checkpoint.ipynb     3204890
                              notebooks/01_data_loading_and_eda.ipynb     1665247
                                          notebooks/02_modeling.ipynb      126498
                       notebooks/03_fairness_and_explainability.ipynb     1510458
                                   notebooks/artifacts/X_test_enc.npy     9065760
                             notebooks/artifacts/X_test_sensitive.csv      278291
                  notebooks/artifacts/feature_names_after_preproc.csv        3207
                                   notebooks/artifacts/results_df.csv         748
                               notebooks/artifacts/threshold_scan.csv        4996
                                  notebooks/artifacts/y_pred_best.npy       78280
                                 notebooks/artifacts/y_proba_best.npy       78280
                               notebooks/artifacts/y_true_test.np.npy       78280
                                  notebooks/artifacts/y_true_test.npy       78280
                                    notebooks/models/LGBM_best.joblib     1289199
                 notebooks/models/threshold_optimizer__dp__sex.joblib      101677
                               reports/figures_03/Pareto_f1_vs_dp.png       42311
                                                     requirements.txt           0
                                                  src/models/infer.py          34


## Notebooks Summary

### 01_data_loading_and_eda.ipynb
- Cells: 16 (code: 14, markdown: 2)
- Headings:
  - # --- 01. Data loading and EDA --- #
  - # --- Итоговые выводы по EDA --- #
- Imports: matplotlib, numpy, pandas, pathlib, random, scipy, seaborn, warnings
- Error outputs captured in JSON: 0

### 02_modeling.ipynb
- Cells: 31 (code: 18, markdown: 13)
- Headings:
  - # --- 02. Modeling: Baselines, Tuning, Evaluation --- #
  - # --- Data Split --- #
  - # --- Unified Preprocessing --- #
  - # --- Raw Copies for Analysis --- #
  - # --- Logistic Regression --- #
  - # --- Decision Tree --- #
  - # --- Random Forest --- #
  - # --- XGBoost (Early Stopping) --- #
  - # --- LightGBM (RandomizedSearch) --- #
  - # --- Unified Metrics Table --- #
  - # --- Model Comparison Chart --- #
  - # --- Saving Best Model & Artifacts --- #
  - # --- Export Artifacts for 03 (fairness & explainability) --- #
- Imports: joblib, lightgbm, matplotlib, numpy, pandas, pathlib, scipy, sklearn, xgboost
- Error outputs captured in JSON: 0

### 03_fairness_and_explainability.ipynb
- Cells: 27 (code: 16, markdown: 11)
- Headings:
  - # --- Fairness & Explainability: setup --- #
  - # --- Загрузка артефактов из 02_modeling.ipynb --- #
  - # --- sanity-check: размеры, NaN, распределение y_proba --- #
  - # --- Определение чувствительных признаков и групп --- #
  - # --- Базовая линия: метрики на общем пороге (0.5) --- #
  - # --- Fairness по группам при t = 0.5 --- #
  - # --- Пороговая кривая и Pareto: качество vs Demographic Parity --- #
  - # --- Post-Processing: ThresholdOptimizer (DemographicParity / EqualizedOdds) --- #
  - # --- Калибровка вероятностей по группам --- #
  - # --- Explainability: глобально (SHAP / альтернативы) --- #
  - # --- Риски, ограничения, рекомендации --- #
- Imports: fairlearn, inspect, joblib, matplotlib, numpy, os, pandas, pathlib, shap, sklearn, warnings
- Error outputs captured in JSON: 0

## Metrics & Artifacts Checks

### results_df.csv (head)
      model      auc       f1  precision   recall  accuracy
  LGBM_best 0.932112 0.722818   0.795175 0.662532  0.795175
     XGB_es 0.931339 0.721510   0.792627 0.662104  0.877674
    RF_best 0.923405 0.686405   0.811079 0.594953  0.811079
     LogReg 0.911697 0.669796   0.742070 0.610351  0.742070
RF_baseline 0.910776 0.683725   0.741500 0.634303  0.741500

### threshold_scan.csv (first 10 rows)

   t  accuracy     f1  precision  recall  roc_auc  dp_diff_max  eqodds_diff_max
0.05    0.6899 0.6021     0.4345  0.9803   0.9321       0.7069           0.6016
0.06    0.7088 0.6158     0.4500  0.9752   0.9321       0.6869           0.5694
0.07    0.7245 0.6281     0.4640  0.9718   0.9321       0.6692           0.5400
0.08    0.7372 0.6375     0.4759  0.9654   0.9321       0.6542           0.5179
0.09    0.7490 0.6470     0.4876  0.9611   0.9321       0.6425           0.5002
0.10    0.7583 0.6539     0.4974  0.9538   0.9321       0.6312           0.4830
0.11    0.7654 0.6591     0.5052  0.9478   0.9321       0.6211           0.4688
0.12    0.7724 0.6643     0.5134  0.9405   0.9321       0.6121           0.4547
0.13    0.7798 0.6698     0.5224  0.9333   0.9321       0.6008           0.4412
0.14    0.7858 0.6744     0.5299  0.9273   0.9321       0.5913           0.4272

### feature_names_after_preproc.csv (count)

- n_features: 115

### Observed Metric Anomalies

- Anomaly: 'accuracy' equals 'precision' for some rows -> likely mis-assigned column or metric bug.
        model  accuracy  precision
    LGBM_best  0.795175   0.795175
      RF_best  0.811079   0.811079
       LogReg  0.742070   0.742070
  RF_baseline  0.741500   0.741500
   DT_entropy  0.631060   0.631060
      DT_gini  0.605998   0.605998

## Key Issues & Gaps

1. README.md is empty — add project description, setup, usage, and results.
2. requirements.txt is empty — add pinned dependencies.
3. Hardcoded absolute Windows paths found in: 01_data_loading_and_eda.ipynb, 02_modeling.ipynb, 03_fairness_and_explainability.ipynb — switch to relative paths.
4. src/models/infer.py is only a stub — add a real CLI/script for inference on CSV/JSON.
5. .gitignore is empty — add ignores for data/raw, artifacts, .ipynb_checkpoints, etc.
6. No saved EDA figures (reports/figures_01) — save key plots.
7. No saved modeling figures (reports/figures_02) — save ROC/PR curves, calibration, confusions.

## Completion Assessment

Status: **Not yet ready to call "finished"**.

**Blocking items to finish:**
- Fill in `README.md` (project overview, dataset, pipeline, how to run, results, fairness, explainability, limitations).
- Fill `requirements.txt` with pinned versions; add `environment.yml` (optional) for conda.
- Remove hardcoded absolute paths from notebooks; use project‑relative paths via `Path(__file__).parents[...]` or `Path.cwd()`.
- Fix the **metrics table** bug where `accuracy` duplicates `precision` for several models; recompute metrics consistently.
- Save key plots to `reports/figures_01` and `reports/figures_02` (EDA, ROC/PR, calibration, confusion, feature importance).
- Expand `src/models/infer.py` into a working inference CLI (load joblib pipeline, read CSV, write predictions).
- Ensure all artifacts needed by 03 are produced automatically by 02 (and paths are consistent).

**Nice-to-have (for polish):**
- Add unit-style smoke tests for preprocessing and inference on 5–10 rows.
- Add `LICENSE`, `.gitignore` content, and a short `CHANGELOG.md` or "Results" section in README.
- Export an HTML report version of each notebook under `reports/`.