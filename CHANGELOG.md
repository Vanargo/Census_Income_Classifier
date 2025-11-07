# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adhares to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-11-07
### Added
- initial public release `v1.0.0` with reproducible DS pipeline for UCI Adult (Census Income);
- fairness analysis (selection rates, demographic parity, equalized odds, threshold scan);
- explainability via SHAP figures and calibration plots;
- installation & Reproducibilirt section in README.

### Changed
- cleaned and formatted notebooks (`01_data_loading_and_eda.ipynb`, `02_modeling.ipynb`, `03_fairness_and_explainability.ipynb`);
- consolidated project paths and artifact directories.

### Fixed
- updated README with run instructions and CI badge;
- introduced this changelog, linked from README.

### CI
- linting and formatting via `ruff` and `black` through pre-commit hooks.