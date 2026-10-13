# R1C6 — mechanically scaled geometry-only baseline (Approach B)

This is the **sqrt(E/rho)-only** normalization. It does **not** use `H**2` scaling, beam-model ratios, or hyperparameter selection. It implements Reviewer 1, Comment 6 of *Data-Driven Approaches for Predicting the Natural Frequencies of Masonry Minarets*.

## Method

For each realization and mode `i`:

- `c = sqrt(1e6 * E_MPa / rho_kgm3)` in m/s, with E converted from MPa to Pa.
- `z_i = f_i / c` in m^-1.
- A linear geometry-only model predicts `z_i = intercept + b_H*H + b_D0*D0 + b_Dtop*Dtop + b_t*t + b_B*B`.
- All six material realizations per geometry are averaged **only for the regression target**; each independent geometry receives one regression row. The six normalized realizations are retained in the diagnostic/output tables.
- Five **original held-out geometry folds** are taken from `data/geometry_fold_assignments.xlsx`; regression coefficients are fitted only to training geometries in each outer iteration.
- Test predictions are returned to Hz as `f_hat = c * z_hat`; `R²`, MAE, RMSE, and SMAPE are calculated on the **held-out original-frequency values**.
- One final model for each mode is refitted on all 191 geometries only for export of final coefficients; final full-data fits are **not used** in cross-validation accuracy calculations.

## Repository placement and execution

Copy the script `mechanically_scaled_geometry_baseline.py` to the repository root, where the existing scripts are located. Ensure both files already exist:

```
data/minaret_frequency_database.xlsx
data/geometry_fold_assignments.xlsx
```

From the repository root run:

```
python mechanically_scaled_geometry_baseline.py
```

Or specify paths:

```
python mechanically_scaled_geometry_baseline.py --data data/minaret_frequency_database.xlsx --folds data/geometry_fold_assignments.xlsx --output results/mechanical_baseline
```

Dependencies: `numpy`, `scikit-learn`, and `matplotlib` (available through the project's environment). Excel files are read with the Python standard library; no `pandas` or `openpyxl` is needed for this script.

## Outputs in results/mechanical_baseline

- `R1C6_B_summary_metrics.csv`: five outer-fold mean and sample standard deviation, plus distinct pooled out-of-fold metrics.
- `R1C6_B_fold_metrics.csv`: training and held-out test performance separately for each outer fold.
- `R1C6_B_group_normalization.csv`: normalized-frequency means and within-group coefficients of variation for the 191 geometry groups, for f1 and f2.
- `R1C6_B_oof_predictions.csv`: all 1,146 held-out predictions for **each** frequency (2,292 rows total), including group/fold, FE target and percentage error.
- `R1C6_B_final_and_fold_coefficients.csv`: outer-training coefficients and clearly labeled final full-data coefficients.
- `R1C6_B_parity_figure.png`, `.svg`: combined two-panel figure matching the original manuscript's grouped-parity style, with color-coded folds, dashed 1:1 line, dotted ±20% limits, and a pooled-OOF metrics box.

## Verified numeric results

| Frequency | 5-fold mean test R² ± SD | 5-fold mean MAE ± SD (Hz) | 5-fold mean RMSE ± SD (Hz) | 5-fold mean SMAPE ± SD (%) |
|:--|:--|:--|:--|:--|
| f1 | 0.8222 ± 0.0293 | 0.1503 ± 0.0171 | 0.2114 ± 0.0217 | 24.31 ± 6.08 |
| f2 | 0.8217 ± 0.0286 | 0.1523 ± 0.0170 | 0.2137 ± 0.0210 | 24.45 ± 6.10 |

The within-geometry median normalized-frequency CV is 0.000786% (f1) and 0.000785% (f2). The highest within-group CV is 0.07921% (f1) and 0.05111% (f2).

**Important:** The figures use **pooled out-of-fold R²** (f1=0.8222065; f2=0.8217953), whereas the manuscript comparison table should use **mean ± SD of fold-specific R²**, matching the previous ML and regression tables. These are different summary estimands and should be labeled explicitly.

IMPORTANT LIMITATION: Because the OLS regression is unconstrained, it produced 12 nonphysical negative out-of-fold predictions for f1 and 12 for f2 (two held-out geometries). The plot shows them by extending the vertical axis below zero; **all of these predictions remain in the error metrics**. This should be disclosed in the manuscript.

These are surrogate errors against FE frequencies, not errors against in-situ measured frequencies. This benchmark intentionally tests the minimum mechanically motivated material normalization; previous exploratory `H**2`-based models are **not part of Approach B** and must not be reported as Approach B results.
