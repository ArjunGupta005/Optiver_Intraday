# Optiver Trading at the Close — 60 s index-relative WAP forecasting

A reproducible research project on the Optiver *Trading at the Close* Kaggle data: forecast each stock's
60-second forward WAP move relative to Optiver's synthetic index (bps), serve the model through a causal
streaming predictor, and evaluate a **target-based research-proxy** long/short backtest.

## Results (measured; details in `reports/`)

**Forecasting** (MAE in bps; lower is better):

| | dev folds (mean of 3) | final holdout, dates 421–480 (scored once) |
|---|---|---|
| zero | 6.4716 | 5.8153 |
| training median | 6.4712 | 5.8153 |
| ridge | 6.3826 | 5.7534 |
| LightGBM L1 | 6.3213 | 5.6924 |
| **frozen final (0.9 LGBM + 0.1 ridge, 0.5 demean)** | 6.3202 (pooled dev OOF) | **5.6884 (−2.18% vs zero)** |

- **Signal quality:** holdout mean per-timestamp Spearman IC is 0.188, and the daily mean IC was positive on
  60 of 60 days. Directional accuracy is 56.2%.

**Streaming:**
- **Parity:** replaying all 3,300 holdout batches matched the batch path to within 1.8e-15.
- **Mock API:** the replay over the example files writes a valid 33,000-row `submission.csv`.
- **Latency:** p50 33 ms and p95 38 ms per 200-stock batch on a 4-core laptop.

**Target-based research-proxy backtest** (not a tradable strategy): long the top 20% and short the bottom 20%
every 60 s, dollar-neutral.

| holdout cost model | gross daily Sharpe |
|---|---|
| no costs | 74 |
| 2 bp per side | −22 |
| each name's own half quoted spread | −101 |

The WAP-marked edge does not survive realistic spread costs; see `reports/backtest_report.md` and
`reports/research_report.md` §7.

Key reports:
- `reports/research_report.md`
- `reports/validation_report.md`
- `reports/backtest_report.md`
- `reports/streaming_benchmark.md`
- `reports/data_audit.md`
- `reports/interview_guide.md`
- `reports/resume_bullets.md`

## What is in the data (from `reports/data_audit.md`)
5,237,980 rows; 200 stocks (191–200 per day); `date_id` 0–480 (481 days); 55 snapshots per stock-day at
`seconds_in_bucket` 0, 10, …, 540 of the closing auction. Top-of-book quotes plus auction state
(imbalance, matched size, reference/near/far prices) — **not** full order-book depth. Prices are normalised to the
stock's WAP at the auction start; sizes are dollar notionals. Ticker identities, calendar dates, executable prices
and the index weights are not given. `far_price`/`near_price` are missing before 300 s; 88 targets are NaN
(excluded, never imputed); 220 rows have no book at all.

Target: `10,000 x [WAP_stock(t+60)/WAP_stock(t) − WAP_index(t+60)/WAP_index(t)]`.

## Design in one paragraph
Splits are chronological by whole `date_id`: three expanding development folds (validate 301–340, 341–380, 381–420)
and an untouched final holdout (421–480). Labels only reach 60 s ahead within the same auction, so whole-date
separation already prevents label overlap (no purge needed; `embargo_days` is configurable). Features
(`src/optiver_tc/features.py`) are one shared polars implementation used by both the research pipeline and the
streaming predictor: top-of-book, auction, a few interactions, elapsed-time lags within stock-day (never row
shifts; reset every day), contemporaneous cross-sectional ranks/demeaning over the current batch, and per-stock
medians fitted on training dates only. Models: zero, training median, ridge (train-fitted imputation, clipping and
scaling) and LightGBM with an L1 objective. Every selection (feature families, LightGBM config, iterations,
ridge blend weight, post-processing, backtest rule) is made by a pre-declared rule on development folds and written
to `artifacts/full/frozen_choices.json` before the holdout is scored once.

## Setup

Windows PowerShell:
```powershell
cd Optiver_TC
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Linux / macOS:
```bash
cd Optiver_TC
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Python ≥ 3.10. Threads: `[resources].n_threads` in the config, overridable with `OPTIVER_THREADS=<n>`.
Peak memory of the full run is ~3–4 GB (float32 matrices, loaded column by column); `ablation_date_fraction`
controls the size of ablation runs.

## Commands
All commands take `--config configs/default.toml` (full research) or `--config configs/smoke.toml`
(dates 0–59, 60 stocks, tiny model budget — numbers from smoke runs are not research results).

```bash
# 1. safe extraction (validates every member path, raw files written read-only) + checksums + Parquet cache
optiver-tc extract --zip /path/to/optiver-trading-at-the-close.zip
# 2. data audit -> reports/data_audit.md
optiver-tc audit
# quick smoke run of the whole pipeline (after step 1; ~3 min on a 4-core laptop)
optiver-tc smoke --config configs/smoke.toml
# 3. development research (stages A-D, resumable) -> artifacts/full/experiments.jsonl, oof/
optiver-tc dev
# 4. freeze all choices from development folds only
optiver-tc freeze
# 5. final model on dates 0-420, holdout 421-480 scored once
optiver-tc final
# 6. target-based proxy backtest (+ WAP robustness, bootstrap)
optiver-tc backtest
# 7. streaming replay: mock API over the example files / full holdout parity check
optiver-tc replay --source example
optiver-tc replay --source holdout
# 8. latency benchmark and generated reports/figures
optiver-tc benchmark
optiver-tc report
# tests
python -m pytest
```
(`python -m optiver_tc.cli <command>` works without the console script.) The `final` step refuses to run before
`freeze`, and refuses to overwrite existing holdout predictions unless `OPTIVER_ALLOW_HOLDOUT_RERUN=1`.

## Kaggle / API compatibility
`data/raw/optiver2023/competition.cpython-310-x86_64-linux-gnu.so` is a Linux x86-64 CPython 3.10 extension and
cannot load on Windows/Python 3.11. `scripts/kaggle_inference.py` is the official-API entry point
(`optiver2023.make_env()` / `env.iter_test()` / `env.predict()`); it has **not** been executed here. Offline
runs use `optiver_tc.mock_api`, a repaired copy of the vendor `MockApi` (the vendor file is kept unmodified in
`data/raw/` and excluded from test discovery). Outputs of that path are labelled *offline mock replay*. The example
test files repeat dates 478–480 of the training data, so they are integration fixtures only.

## Repository layout
```
configs/            default.toml (full), smoke.toml
src/optiver_tc/     config, data (extraction/manifest/parquet), audit, splits, features (shared),
                    featurestore, models, experiments, research (stages A-D), final (freeze/holdout),
                    streaming (ModelBundle + StreamingPredictor + replay), mock_api, replay_cli,
                    backtest, benchmark, reporting, cli
scripts/            kaggle_inference.py
tests/              causality/parity, robustness, splits+backtest arithmetic, end-to-end
notebooks/          results_walkthrough.ipynb (reads saved artifacts via package code)
data/raw/           immutable extracted inputs (git-ignored)      data/processed/  parquet cache, feature store, manifest
artifacts/full/     experiments.jsonl, oof/, predictions/, model/, frozen_choices.json, metrics, logs
reports/            data_audit, validation_design, validation_report, experiments, backtest_report,
                    streaming_benchmark, research_report, interview_guide, resume_bullets, figures/
PROGRESS.md         resumable work log
```

## Limitations (read before quoting any number)
* The backtest is a **target-based research proxy**: it marks a dollar-neutral book at WAP using the competition
  label. It ignores fill prices, queue position, auction participation constraints, market impact and capacity, and
  the cost grid is hypothetical. It is not evidence of a live strategy.
* One holdout of 60 days: Sharpe and even MAE differences have wide uncertainty (see the bootstrap intervals).
* The local holdout is not the Kaggle leaderboard period; local MAE is not comparable to leaderboard scores
  (the screenshot's 5.4030 is historical context only).
* Tests demonstrate specific causal properties (future-perturbation invariance, label isolation, state resets,
  streaming/batch parity). They do not prove the absence of every possible look-ahead.
* The Kaggle entry point was not run (incompatible binary); only the mock replay was executed.
* Primary pipeline uses no revealed targets and no online retraining; those would be separate experiments.
