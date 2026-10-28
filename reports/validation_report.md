# Validation report

Design: `reports/validation_design.md`. All numbers below are read from saved artifacts.

## Feature-family ablation (stage B)
Rule: drop family iff mean MAE without it < with it AND better in >=2/3 folds. MAE with all families: 6.34725.

| family | mae_without | delta_vs_all | folds_better_without | dropped |
|---|---|---|---|---|
| book | 6.34558 | -0.00167 | 3 | True |
| auction | 6.35168 | 0.00443 | 0 | False |
| interact | 6.34496 | -0.00229 | 2 | True |
| temporal | 6.35299 | 0.00574 | 0 | False |
| xsec | 6.35109 | 0.00385 | 0 | False |
| stock | 6.34567 | -0.00157 | 3 | True |

Selected families: `['auction', 'temporal', 'xsec']`.

## LightGBM configuration search (stage C)
Rule: lowest mean dev MAE.

| config | mean_mae | fold1 | fold2 | fold3 |
|---|---|---|---|---|
| lgb_base | 6.34078 | 6.39060 | 6.29908 | 6.33266 |
| lgb_small | 6.34395 | 6.39386 | 6.30215 | 6.33582 |
| lgb_large | 6.33820 | 6.38841 | 6.29662 | 6.32957 |

Selected: `{'name': 'lgb_large', 'learning_rate': 0.05, 'num_leaves': 255, 'min_data_in_leaf': 300, 'feature_fraction': 0.6, 'bagging_fraction': 0.7, 'bagging_freq': 1, 'lambda_l2': 5.0, 'max_bin': 255}`.

## Frozen choices (before holdout)
Frozen at 2026-10-14T21:06:06, sha256 `07000e800d9b7e80…`.

* families: ['auction', 'temporal', 'xsec'] (61 columns)
* LightGBM params: `{'name': 'lgb_large', 'learning_rate': 0.05, 'num_leaves': 255, 'min_data_in_leaf': 300, 'feature_fraction': 0.6, 'bagging_fraction': 0.7, 'bagging_freq': 1, 'lambda_l2': 5.0, 'max_bin': 255}`; rounds = 410 (mean of dev best iterations {'fold1': 291.0, 'fold2': 467.0, 'fold3': 473.0})
* ridge partner: {'experiment': 'ridge_a10000', 'alpha': 10000.0}; blend weight on LightGBM: 0.9
* post-processing: demean=0.5, clip=None (pooled dev MAE, must beat 'none' in >=2/3 folds)
* backtest rule: {'cap': 0.02, 'quantile': 0.2, 'gross': 1.0, 'min_names': 10}

Blend grid (pooled dev OOF MAE, selected rows):

| w_lgb | mae |
|---|---|
| 0.00000 | 6.38261 |
| 0.20000 | 6.35817 |
| 0.40000 | 6.33975 |
| 0.60000 | 6.32742 |
| 0.80000 | 6.32124 |
| 1.00000 | 6.32129 |

Post-processing grid:

| demean | clip | mae | folds_better_than_none |
|---|---|---|---|
| 0.00000 |  | 6.32049 | 0.00000 |
| 0.00000 | 30.00000 | 6.32049 | 0.00000 |
| 0.00000 | 15.00000 | 6.32055 | 0.00000 |
| 0.50000 |  | 6.32017 | 2.00000 |
| 0.50000 | 30.00000 | 6.32017 | 2.00000 |
| 0.50000 | 15.00000 | 6.32023 | 2.00000 |
| 1.00000 |  | 6.32175 | 1.00000 |
| 1.00000 | 30.00000 | 6.32175 | 1.00000 |
| 1.00000 | 15.00000 | 6.32181 | 1.00000 |

## Development OOF (frozen pipeline, dates 301–420)

```json
{
 "n_rows_scored": 1319889,
 "mae": 6.3201721306498575,
 "rmse": 9.361114608781977,
 "dir_acc": 0.5651101031703133,
 "pearson_ic_mean": 0.20106886131065196,
 "spearman_ic_mean": 0.1936358401596583,
 "spearman_ic_tstat": 171.79336267803453,
 "n_ic_groups": 6600
}
```
Share of days with positive mean IC: 1.000.

## Final holdout (dates 421–480, evaluated once after freezing)

Evaluated at 2026-10-14T21:14:17 with frozen sha256 `07000e800d9b7e80…`. Rows scored: 659,999.

| model | mae | rmse | dir_acc | spearman_ic_mean |
|---|---|---|---|---|
| zero | 5.81528 | 8.67105 |  |  |
| train median (-0.0697) | 5.81534 |  |  |  |
| ridge only | 5.75337 | 8.59319 | 0.54690 | 0.14788 |
| LightGBM only (raw) | 5.69243 | 8.51101 | 0.56130 | 0.18725 |
| FINAL (frozen blend + post-processing) | 5.68835 | 8.50933 | 0.56242 | 0.18786 |

Improvement vs zero: 2.18%. Mean per-timestamp Spearman IC 0.1879 (t-stat over 3300 timestamps 115.7; timestamps are not independent, so treat the t-stat as descriptive).
Share of holdout days with positive mean IC: 1.000.

By auction stage:

| stage | mae | mae_zero |
|---|---|---|
| 0-290 s (no far/near) | 6.3139 | 6.4527 |
| 300-540 s | 4.9377 | 5.0503 |

Slices: `artifacts/full/predictions/holdout_mae_by_{date_id,stock_id,seconds_in_bucket}.csv`.

Context only: the Kaggle leaderboard (a different, later test period) is not comparable with these local numbers.

## Figures
* `reports/figures/full/fold_mae.png`
* `reports/figures/full/feature_importance.png`
* `reports/figures/full/pred_diagnostics_dev.png`, `pred_diagnostics_holdout.png`