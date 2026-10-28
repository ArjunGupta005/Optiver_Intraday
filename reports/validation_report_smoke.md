# Validation report

Design: `reports/validation_design.md`. All numbers below are read from saved artifacts.

## Feature-family ablation (stage B)
Rule: drop family iff mean MAE without it < with it AND better in >=2/3 folds. MAE with all families: 4.99312.

| family | mae_without | delta_vs_all | folds_better_without | dropped |
|---|---|---|---|---|
| book | 4.99479 | 0.00166 | 1 | False |
| auction | 5.00450 | 0.01138 | 0 | False |
| interact | 4.99437 | 0.00124 | 1 | False |
| temporal | 5.00295 | 0.00983 | 0 | False |
| xsec | 5.00701 | 0.01389 | 0 | False |
| stock | 4.99574 | 0.00262 | 1 | False |

Selected families: `['book', 'auction', 'interact', 'temporal', 'xsec', 'stock']`.

## LightGBM configuration search (stage C)
Rule: lowest mean dev MAE.

| config | mean_mae | fold1 | fold2 | fold3 |
|---|---|---|---|---|
| lgb_base | 4.98680 | 5.42697 | 4.90995 | 4.62348 |
| lgb_small | 4.99385 | 5.43163 | 4.92110 | 4.62882 |
| lgb_large | 4.98695 | 5.42117 | 4.91157 | 4.62811 |

Selected: `{'name': 'lgb_base', 'learning_rate': 0.05, 'num_leaves': 127, 'min_data_in_leaf': 500, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 1, 'lambda_l2': 1.0, 'max_bin': 255}`.

## Frozen choices (before holdout)
Frozen at 2026-10-14T21:24:31, sha256 `114f070eb81f8d0b…`.

* families: ['book', 'auction', 'interact', 'temporal', 'xsec', 'stock'] (80 columns)
* LightGBM params: `{'name': 'lgb_base', 'learning_rate': 0.05, 'num_leaves': 127, 'min_data_in_leaf': 500, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 1, 'lambda_l2': 1.0, 'max_bin': 255}`; rounds = 97 (mean of dev best iterations {'fold1': 130.0, 'fold2': 102.0, 'fold3': 58.0})
* ridge partner: {'experiment': 'ridge_a100', 'alpha': 100.0}; blend weight on LightGBM: 0.6
* post-processing: demean=1.0, clip=None (pooled dev MAE, must beat 'none' in >=2/3 folds)
* backtest rule: {'cap': 0.05, 'quantile': 0.2, 'gross': 1.0, 'min_names': 10}

Blend grid (pooled dev OOF MAE, selected rows):

| w_lgb | mae |
|---|---|
| 0.00000 | 4.97983 |
| 0.20000 | 4.97004 |
| 0.40000 | 4.96431 |
| 0.60000 | 4.96263 |
| 0.80000 | 4.96507 |
| 1.00000 | 4.97159 |

Post-processing grid:

| demean | clip | mae | folds_better_than_none |
|---|---|---|---|
| 0.00000 |  | 4.96263 | 0.00000 |
| 0.00000 | 30.00000 | 4.96263 | 0.00000 |
| 0.00000 | 15.00000 | 4.96263 | 0.00000 |
| 0.50000 |  | 4.96009 | 2.00000 |
| 0.50000 | 30.00000 | 4.96009 | 2.00000 |
| 0.50000 | 15.00000 | 4.96009 | 2.00000 |
| 1.00000 |  | 4.96001 | 2.00000 |
| 1.00000 | 30.00000 | 4.96001 | 2.00000 |
| 1.00000 | 15.00000 | 4.96001 | 2.00000 |

## Development OOF (frozen pipeline, dates 301–420)

```json
{
 "n_rows_scored": 59400,
 "mae": 4.9600070194311225,
 "rmse": 7.280039747997597,
 "dir_acc": 0.5781218410944133,
 "pearson_ic_mean": 0.2631051845879405,
 "spearman_ic_mean": 0.2380110672373047,
 "spearman_ic_tstat": 53.30428267990013,
 "n_ic_groups": 990
}
```
Share of days with positive mean IC: 1.000.

## Final holdout (dates 421–480, evaluated once after freezing)

Evaluated at 2026-10-14T21:24:44 with frozen sha256 `114f070eb81f8d0b…`. Rows scored: 33,000.

| model | mae | rmse | dir_acc | spearman_ic_mean |
|---|---|---|---|---|
| zero | 4.48195 | 6.22446 |  |  |
| train median (-0.0697) | 4.48299 |  |  |  |
| ridge only | 4.36946 | 6.06442 | 0.56440 | 0.20183 |
| LightGBM only (raw) | 4.37866 | 6.08408 | 0.56218 | 0.19650 |
| FINAL (frozen blend + post-processing) | 4.36007 | 6.06273 | 0.56798 | 0.20509 |

Improvement vs zero: 2.72%. Mean per-timestamp Spearman IC 0.2051 (t-stat over 550 timestamps 34.8; timestamps are not independent, so treat the t-stat as descriptive).
Share of holdout days with positive mean IC: 1.000.

By auction stage:

| stage | mae | mae_zero |
|---|---|---|
| 0-290 s (no far/near) | 4.5488 | 4.6685 |
| 300-540 s | 4.1336 | 4.2581 |

Slices: `artifacts/full/predictions/holdout_mae_by_{date_id,stock_id,seconds_in_bucket}.csv`.

Context only: the Kaggle leaderboard (a different, later test period) is not comparable with these local numbers.

## Figures
* `reports/figures/smoke/fold_mae.png`
* `reports/figures/smoke/feature_importance.png`
* `reports/figures/smoke/pred_diagnostics_dev.png`, `pred_diagnostics_holdout.png`