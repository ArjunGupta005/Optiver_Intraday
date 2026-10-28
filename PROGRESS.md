# PROGRESS

Persistent log so work can resume across sessions. Newest entries at the bottom of each section.

## Plan
1. Inspect ZIP + environment -> safe extraction -> audit -> parquet cache  [done]
2. Validation design recorded before modelling  [done: reports/validation_design.md]
3. Shared feature module (batch == streaming)  [done: src/optiver_tc/features.py]
4. Baselines + LightGBM on dev folds; ablations; bounded config search
5. Streaming predictor + offline replay + Kaggle entry point; tests
6. Freeze choices -> final holdout evaluation (once)
7. Target-based proxy backtest + WAP robustness + bootstrap
8. Latency benchmark, plots, reports, README, interview guide

## Environment (measured)
* Windows 11, Intel i7-1165G7 (4 cores / 8 threads), 16.9 GB RAM (often only 1.3–3.6 GB free), no GPU.
* Python 3.11.9 (Microsoft Store build). lightgbm 4.7.0, polars 1.44.1, pandas 2.2.2, numpy 1.26.4.
* The compiled `optiver2023` API is a Linux x86-64 CPython 3.10 ELF -> not importable here.

## Log
* Extraction: 7 members, all paths validated, written read-only to data/raw. Two copies of
  public_timeseries_testing_util.py (ZIP and Downloads) are byte-identical (sha256 24f42430...).
* Audit: 5,237,980 x 17; 200 stocks; 481 dates; 55 buckets/stock-day; 88 NaN targets; 220 rows with no book.
* NaN tokens in CSV converted to null in Parquet cache (polars otherwise treats them as float NaN).
* 19:03 started `optiver-tc dev` (full config, stages ABCD) in background; log artifacts/full/logs/dev.log.
* LightGBM + polars DLL clash on Windows fixed by importing lightgbm first (package __init__).
* Tests: 28 pass (causality/parity, robustness, splits/backtest, e2e synthetic). Clean .venv (polars 2.0,
  pandas 3.0, numpy 2.4) also passes.
* Smoke (configs/smoke.toml) completed end-to-end in 22 min under CPU contention; then restricted to 60 stocks.
  Smoke numbers are NOT research results.
* Resume: `python -m optiver_tc.cli dev` is resumable (skips (experiment, fold) pairs already in experiments.jsonl).
* 20:25 user asked for the finished project to live in `Desktop\Quant_Projects\Optiver_TC` (currently empty);
  decision (final): project stays in p1-137Optiver_TC; no move.
* Stage B result: keep auction, temporal, xsec; drop book, interact, stock (single-family drops, tiny effects).

## Status: COMPLETE (2026-10-14, ~21:45 local)
* Dev (stages A-D) finished 21:05. Frozen 21:06 (sha256 07000e80...). Holdout scored once 21:14:
  final MAE 5.6884 vs zero 5.8153 (-2.18%), IC 0.188.
* Backtest proxy: gross Sharpe 74 on holdout; negative at 2 bp/side; strongly negative with each name's
  own half-spread (~3.6 bp/side avg). Documented as non-tradable.
* Replay: holdout parity 1.8e-15 over 3,300 batches; mock-API submission 33,000 rows valid.
* Benchmark: warm p50 33 ms / p95 38 ms per batch.
* Added after holdout (reporting only, no re-selection): quoted-half-spread cost variant; fixed a
  cost-sensitivity chart bug (netted bars were shifted one cost slot).
* Smoke rerun: 188 s. Full test suite: 29 passed.
* Not done: Kaggle entry point never executed (binary incompatible); CatBoost/ensembles/revealed-target
  experiments not run (see research_report.md section 10).
