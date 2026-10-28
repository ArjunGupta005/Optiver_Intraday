# Research report — Optiver Trading at the Close

Every number here is read from saved artifacts:
- dev metrics: `artifacts/full/experiments.jsonl`
- holdout metrics: `holdout_metrics.json`
- backtest: `backtest_results.json`
- latency: `streaming_benchmark.json`

Supporting detail is in `validation_report.md`, `experiments.md`, `backtest_report.md`, `streaming_benchmark.md` and `data_audit.md`.

## 1. Question
Predict the 60-second forward move of each stock's WAP relative to Optiver's synthetic index, in bps, during the
Nasdaq closing auction (10-second snapshots, 0–540 s). The model must run as a causal streaming predictor. A
secondary question is whether the signal survives a simple, transparent portfolio proxy.

## 2. Validation protocol (fixed before modelling)
- **Splits:** whole-date chronological splits.
  - Development folds validate on dates 301–340, 341–380 and 381–420 (expanding window from date 0).
  - The final holdout is dates 421–480.
- **No label overlap:** labels never cross a day, so date separation already prevents overlap.
- **Selection:** every choice was made by a pre-declared rule on development folds only and written to
  `frozen_choices.json`. Those choices are feature families, LightGBM config, number of rounds, blend weight,
  demeaning/clipping and the backtest rule.
- **Single holdout read:** the holdout was then scored exactly once (21:14 local time, frozen sha256 `07000e80…`).

## 3. Hypotheses and feature families
| family | hypothesis | examples |
|---|---|---|
| book | displayed imbalance/microprice pressure predicts short-term WAP drift | spread bps, (bid−ask size)/total, WAP−mid |
| auction | unmatched auction interest pushes the clearing price; reference/near/far prices are the auction's own forecast | imbalance/matched, ref−WAP, near/far−ref, missingness flags |
| interact | pressure is stronger when liquidity is thin | imbalance × spread, flag × book imbalance |
| temporal | short-horizon reversal/continuation within stock-day | 10/30/60 s WAP returns, changes in imbalance/matched/spread, trailing 60 s volatility |
| xsec | the target is index-relative, so relative states matter | per-timestamp demeaned values and ranks of returns, imbalance, spread |
| stock | normalise by a stock's typical liquidity/volatility | train-only medians of spread, matched size, |10 s return| |

## 4. Results on the development folds (MAE, bps)
| model | fold1 | fold2 | fold3 | mean | vs zero |
|---|---|---|---|---|---|
| zero | 6.5127 | 6.4311 | 6.4711 | 6.4716 | – |
| training median | 6.5122 | 6.4306 | 6.4707 | 6.4712 | −0.01% |
| ridge (α = 1e4) | 6.4216 | 6.3513 | 6.3749 | 6.3826 | −1.38% |
| LightGBM, selected (`lgb_final_dev`) | 6.3704 | 6.2798 | 6.3136 | 6.3213 | −2.32% |

- **Stability:** the LightGBM gain over zero is similar in each fold (2.2–2.4%). The cross-sectional Spearman IC
  averages 0.193.

**Feature-family ablation (stage B).** Setup: leave-one-family-out, LightGBM at learning rate 0.1, on 50% of the
training dates.
- **Clear contributors:**
  - temporal: +0.0057 MAE when removed
  - auction: +0.0044
  - xsec: +0.0038
- **Dropped by the rule:** book, interact and stock. Removing each one *alone* lowered MAE slightly, by
  0.0016–0.0023.
- **Caveats:**
  - These are small effects.
  - The three removals were never tested jointly within stage B.
  - Much of the book information survives through the cross-sectional versions (`book_imb_dm` is the
    second-most important feature).

**Configuration search (stage C).** The larger tree configuration (`lgb_large`: 255 leaves, min 300 rows per leaf,
feature fraction 0.6, L2 = 5) won all three folds. The margin is small: 0.0026 MAE over `lgb_base`.

**Frozen post-processing.**
- **Blend:** 0.9 LightGBM + 0.1 ridge, chosen on pooled development OOF predictions (6.3205 vs 6.3213 for
  LightGBM alone).
- **Demeaning:** subtracting 0.5 × the per-timestamp mean prediction helped in 2 of 3 folds (6.3202).
  - Full demeaning (strength 1.0) hurt (6.3218), consistent with the synthetic index **not** being equal-weighted.
  - So no zero-sum constraint was imposed.
- **Clipping:** never helped.

**Importance (mean gain share across folds).**
- `stock_id`: 25%
- `book_imb_dm`: 8.8%
- `seconds_in_bucket`: 3.8%
- Then a long tail of reference-price, imbalance and spread features, mostly the cross-sectional versions.

## 5. Final holdout (dates 421–480, 659,999 scored rows)
| model | MAE | RMSE | directional acc. | mean Spearman IC |
|---|---|---|---|---|
| zero | 5.8153 | 8.6711 | – | – |
| training median | 5.8153 | – | – | – |
| ridge only | 5.7534 | 8.5932 | 0.547 | 0.148 |
| LightGBM only | 5.6924 | 8.5110 | 0.561 | 0.187 |
| **final (frozen)** | **5.6884** | **8.5093** | **0.562** | **0.188** |

- **Overall:** the final model is 2.18% better than zero in MAE, matching the development folds.
- **Every day was positive:** the mean IC was positive on all 60 holdout days.
- **By auction stage, the gain is similar in absolute size:**
  - 0–290 s: 6.314 vs 6.453
  - 300–540 s: 4.938 vs 5.050
- **Not comparable to Kaggle:** the holdout differs from the Kaggle leaderboard period, so these numbers do not
  map to a leaderboard rank.

## 6. Streaming inference
- **Parity:** `StreamingPredictor` replayed all 3,300 holdout batches. It matched the batch path to
  1.8e-15 at most, and every prediction was finite.
- **Mock API:** the vendor-style replay over the example files produced a 33,000-row `submission.csv`. Its row
  order and schema match `sample_submission.csv`.
  - This is an offline mock: the official compiled API cannot load on Windows/Python 3.11.
- **Latency on this laptop (one run):**
  - warm batch: p50 33 ms, p95 38 ms
  - cold start: ~1.8 s of imports plus a 43 ms first batch
  - model files: 10.8 MB

## 7. Signal backtest — target-based research proxy
Rule frozen on development data: every 60 s, go long the top 20% and short the bottom 20% of demeaned
predictions. Each side is 0.5 of capital, with a 2% cap per name, exactly net-zero.

| holdout (60 days) | mean daily (bps of capital) | Sharpe (daily, annualised) |
|---|---|---|
| no costs | +30.9 | 74 (block-bootstrap 95% CI 59–114) |
| 0.5 bp/side flat, open+close | +20.9 | 50 |
| 1 bp/side flat | +10.9 | 26 |
| 2 bp/side flat | −9.1 | −22 |
| **each name's own half-spread at entry and exit (avg ≈ 3.6 bp/side)** | **−41.0** | **−101 (negative every day)** |

The WAP-based robustness check uses observed WAP(t+60)/WAP(t) − 1 on the 0–480 s schedule. It gives the same
picture: gross Sharpe 69, and negative at 2 bp/side.

**Interpretation.** The gross proxy Sharpe is implausibly high for any real strategy, and that is the point to
explain, not to advertise:
1. **The proxy marks at WAP, not at fill prices.** WAP leans toward the thin side of the book, and much of the
   predictable 60-second WAP move is microprice/book-imbalance mechanics. Those mechanics cannot be captured by
   crossing the spread.
2. **The book turns over 20× capital per day.** At ~31 bps of gross edge per day, break-even is about
   1.5 bp per side when every interval is opened and closed, or about 2 bp if consecutive positions are netted.
3. **The signal concentrates in wide-spread names.** The traded names carry an average half-spread of ~3.6 bp,
   versus a median spread of 3.25 bp (half-spread ~1.6) across all holdout rows. Charging that cost turns every
   day negative.

So the project shows a stable, statistically clear forecasting signal (IC ≈ 0.19). It does **not** show a
tradable strategy, and the Sharpe ≈ 1.9 aspiration is neither met nor meaningfully testable with this proxy.

## 8. What worked, what did not
**Worked**
- Cross-sectional transforms (demeaned/ranked imbalance, reference-price deviation and returns).
- Elapsed-time temporal features.
- An L1 objective.
- Larger trees.
- Partial demeaning.
- One shared feature implementation, which gave exact streaming parity at low engineering cost.

**Did not help on dev folds**
- Per-stock training medians (stock_id as a categorical already captures this).
- Hand-made interactions.
- The raw book family (largely duplicated by its cross-sectional version).
- Clipping.
- Full (zero-sum) demeaning.
- More ridge regularisation, beyond a negligible amount.

**Engineering lessons**
- On Windows, importing LightGBM after polars/pyarrow crashed dataset construction; import order is pinned.
- The CSV holds `nan` tokens that polars reads as float NaN, not null; they are normalised in the Parquet cache.

## 9. Limitations
- **One holdout of 60 days.** The Sharpe standard error is about 2 at this length, but the cost assumption matters
  far more than sampling noise.
- **The backtest is a research proxy.** No auction-uncross price, queue, participation limit, impact or capacity
  is modelled. Costs are hypothetical, except the quoted-spread variant, which is still crude.
- **`stock_id` is a strong feature.** It ties the model to this 200-stock universe. Unseen stocks get valid but
  less informed predictions.
- **The tests are not a proof.** They demonstrate specific causal invariances; they do not prove the absence of
  all look-ahead.
- **The Kaggle entry point was not executed** (incompatible binary).
- **Not run:** CatBoost, multi-seed ensembles and deep models. Revealed-target features and online retraining
  were deliberately excluded.

## 10. Next steps (not done)
1. **Cost-aware signal:** predict net-of-half-spread edge, or trade only where |prediction| exceeds the name's
   half-spread. Select on development folds.
2. **Previous-day revealed targets** with explicit release timestamps, as a separate experiment.
3. **Index-weight estimation** from training labels to build a properly index-neutral portfolio.
4. **Joint re-ablation** of the dropped families and of `stock_id`; multi-seed LightGBM.
