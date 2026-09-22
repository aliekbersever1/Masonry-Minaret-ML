# Masonry Minaret Natural-Frequency Prediction

This repository contains the numerical database and Python scripts used to reproduce the machine-learning and explicit-regression analyses reported in the manuscript **Data-Driven Approaches for Predicting the Natural Frequencies of Masonry Minarets**.

## Repository structure

```text
.
├── data/
│   └── minaret_frequency_database.xlsx
├── ml_group_nested_cv.py
├── regression_formulas_group_cv.py
├── requirements.txt
└── results/                 # created automatically
    ├── ml/
    └── regression/
```

## Required database columns

The Excel database must contain the following column names exactly:

- `H_m` — total height, m
- `D0_m` — lower inner diameter, m
- `Dtop_m` — upper inner diameter, m
- `t_m` — wall thickness, m
- `B_kaide_m` — square base dimension, m
- `E (MPa)` — elastic modulus, MPa
- `Ro (kg/m3)` — density, kg/m³
- `f1 (Hz)` — first natural frequency, Hz
- `f2 (Hz)` — second natural frequency, Hz

The published dataset contains 1,146 finite-element model realizations and 191 unique geometry groups. Geometry groups are defined only by `H_m`, `D0_m`, `Dtop_m`, `t_m`, and `B_kaide_m`. The height parameter `Hc` / `H_silindir` is not used.

For data-driven modeling, the material predictor is computed as:

```text
sqrt_E_div_rho = sqrt(E / rho)
```

`E` and `rho` are therefore not supplied separately to the ML models or the explicit regression models.

## 1. Machine-learning analysis

Run:

```bash
python ml_group_nested_cv.py
```

or specify custom locations:

```bash
python ml_group_nested_cv.py --data /path/to/database.xlsx --output /path/to/results/ml
```

The script evaluates Random Forest, XGBoost, LightGBM, and CatBoost using geometry-grouped nested five-fold cross-validation. Hyperparameters are selected only within the inner GroupKFold loops, while held-out outer folds are used for performance evaluation. The script also exports pooled out-of-fold predictions, parity plots with ±20% error bounds, held-out permutation importance, SHAP plots, final hyperparameters, and fitted final models.

## 2. Explicit frequency equations

Run:

```bash
python regression_formulas_group_cv.py
```

or:

```bash
python regression_formulas_group_cv.py --data /path/to/database.xlsx --output /path/to/results/regression
```

This script evaluates **both**:

- simple linear regression using only `H`, `D0`, `Dtop`, `t`, `B`, and `sqrt(E/rho)`; and
- symbolic regression using PySR with the same six predictors.

No manually engineered geometric ratios are used in the final linear-regression baseline. Both approaches are evaluated using geometry-grouped five-fold cross-validation. Final equations are re-fitted using the complete database only after cross-validated performance evaluation.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

PySR may require additional Julia-related setup depending on the installed PySR version.

## Reproducibility note

The scripts use fixed random seeds where supported. Symbolic regression is stochastic and can be computationally intensive; small numerical or expression-level differences may occur across PySR/Julia versions or computing environments. Cross-validated performance should be used for generalization claims; full-data fits are provided only to obtain the final deployable equations/models.
