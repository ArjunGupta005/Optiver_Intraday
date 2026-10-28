# Validation design (fixed before modelling)

Written after the data audit and **before** any model was fitted. Boundaries live in
`configs/default.toml` (`[splits]`) and are copied into every experiment record.

## Facts from the audit that drive the design
* 481 trading days, `date_id` 0..480, contiguous; 200 stocks (191–200 present per day).
* 55 snapshots per stock-day at `seconds_in_bucket` 0, 10, ..., 540; no duplicate keys.
* `target` = 60 s forward stock-WAP move minus synthetic-index move, in bps. Each label uses
  WAP(t+60) of the **same auction on the same day**. No label spans two dates.
* 88 missing targets -> excluded from fitting and scoring.
* The example test files (dates 478–480) duplicate the last training dates; they are integration fixtures.

## Splits (whole `date_id`s only)

| split | train dates | validation dates | # train | # val |
|---|---|---|---|---|
| fold1 | 0–300 | 301–340 | 301 | 40 |
| fold2 | 0–340 | 341–380 | 341 | 40 |
| fold3 | 0–380 | 381–420 | 381 | 40 |
| **final holdout** | 0–420 | **421–480** | 421 | 60 |

* Expanding windows; initial history of 301 days (~63% of the data) so the first fold already has a
  realistic training size.
* The three development folds are used for feature selection, early stopping, hyperparameters,
  ensemble weights, clipping/demeaning choices and backtest rules. Every choice is written to
  `artifacts/full/frozen_choices.json` **before** the holdout is scored.
* The final model is refit on dates 0–420 with the number of boosting rounds chosen from the
  development folds (the mean of the three folds' early-stopping best iterations, no extrapolation; see `final.py`). Holdout targets
  are used once, for scoring.

## Label horizon at split boundaries
A label at (d, t) depends on WAP at (d, t+60) only, and t+60 <= 600 s, which is still inside the same
day's auction. Validation starts at the first snapshot of a new date, so **complete-date separation already
prevents any train label from overlapping a validation timestamp**; `embargo_days = 0`. No row-count purge
is applied (that would be wrong in a panel with interleaved stocks). Cross-day dependence (for example
persistent stock volatility) is not label leakage; it is legitimate history. `embargo_days` can still be
raised in the config as a sensitivity check.

## Fitted state
Everything fitted (per-stock medians, ridge imputation/clipping/standardisation, LightGBM, backtest rule
choices) uses training dates of the relevant split only. Features never read `target`; the primary pipeline
uses no target-derived features and no online retraining (the competition's revealed targets would only be
available for the previous day, at the start of a new day).

## Metrics
Primary: row-weighted MAE over all scored rows. Secondary: RMSE, directional accuracy (rows with
non-zero prediction and target), per-timestamp cross-sectional Pearson/Spearman IC (groups with < 10
names or zero variance are NaN and excluded), and MAE slices by fold, date, stock and `seconds_in_bucket`.
