# Backtest report — target-based research proxy

**This is a research proxy, not a demonstrated or executable strategy.** Returns are computed from the
competition label (index-relative WAP move, bps) on out-of-sample predictions. WAP is not a fill price,
the synthetic index weights are unknown, and the cost grid below is hypothetical, not measured.

Conventions are documented in `src/optiver_tc/backtest.py`. Summary: decision times every 60 s
([0, 60, 120, 180, 240, 300, 360, 420, 480, 540]), non-overlapping 60 s holds; demeaned signal; long and short books each
scaled to gross/2 under a per-stock cap (exactly net-zero); capital 1 per day, interval returns summed;
costs charged on entry and exit (`open_close`), with a `netted` sensitivity that charges only weight changes;
Sharpe = sqrt(252) x mean / std of DAILY returns.

## Rule selection (development OOF only)
Criterion: max dev Sharpe at 0.5 bps/side (open+close costs).

| cap | quantile | gross | min_names | sharpe_gross | sharpe_at_selection_cost | mean_daily_bps_gross | mean_daily_bps_at_cost |
|---|---|---|---|---|---|---|---|
| 0.0200 | 1.0000 | 1.0000 | 10.0000 | 87.5306 | 58.2496 | 29.8933 | 19.8933 |
| 0.0200 | 0.4000 | 1.0000 | 10.0000 | 87.4746 | 58.8054 | 30.5117 | 20.5117 |
| 0.0200 | 0.2000 | 1.0000 | 10.0000 | 85.3842 | 60.7774 | 34.6995 | 24.6995 |
| 0.0500 | 1.0000 | 1.0000 | 10.0000 | 83.9545 | 57.6485 | 31.9146 | 21.9146 |
| 0.0500 | 0.4000 | 1.0000 | 10.0000 | 83.9785 | 58.2793 | 32.6775 | 22.6775 |
| 0.0500 | 0.2000 | 1.0000 | 10.0000 | 81.7493 | 60.7645 | 38.9564 | 28.9564 |

Selected and frozen: `{'cap': 0.02, 'quantile': 0.2, 'gross': 1.0, 'min_names': 10}`.

## Development OOF (dates 301–420; the rule was chosen on these, so this is optimistic)

Days 120, intervals 1200 (active 1200), mean gross 1.000,
max |net| 2.64e-16, max |w| 0.0200, mean names long/short 40.0/40.0,
daily turnover 20.0x capital (open+close) vs 15.15x if netted.

| cost bps/side | turnover | mean daily bps | std daily bps | Sharpe | cum bps | max DD bps | up days | Sharpe 95% CI |
|---|---|---|---|---|---|---|---|---|
| 0.000 | open_close | 34.699 | 6.451 | 85.384 | 4163.935 | 0.000 | 1.000 | [74.82, 100.51] |
| 0.500 | open_close | 24.699 | 6.451 | 60.777 | 2963.935 | 0.000 | 1.000 | [53.62, 71.11] |
| 0.500 | netted | 27.125 | 6.437 | 66.895 | 3254.968 | 0.000 | 1.000 |  |
| 1.000 | open_close | 14.699 | 6.451 | 36.171 | 1763.935 | 0.000 | 1.000 |  |
| 1.000 | netted | 19.550 | 6.427 | 48.291 | 2346.001 | 0.000 | 1.000 |  |
| 2.000 | open_close | -5.301 | 6.451 | -13.043 | -636.065 | -626.465 | 0.158 |  |
| 2.000 | netted | 4.401 | 6.419 | 10.883 | 528.067 | -19.513 | 0.775 |  |

**Quoted-spread cost assumption** (replaces the flat grid; not added to it): each traded name pays half its
own quoted spread at entry and at exit (exit spread = entry spread at 540 s, where t+60 is unobserved).
Average round-trip cost 7.99 bps per unit of gross traded (~4.00 bps per side): mean daily -45.23 bps, Sharpe -112.25 (95% CI [-127.35, -99.93]), up days 0.00, cumulative -5428 bps, max DD -5384 bps.
Limitation: top-of-book spreads in normalised price units; ignores depth, auction-uncross pricing and impact.

## Final holdout (dates 421–480; rule frozen)

Days 60, intervals 600 (active 600), mean gross 1.000,
max |net| 2.29e-16, max |w| 0.0200, mean names long/short 40.0/40.0,
daily turnover 20.0x capital (open+close) vs 15.06x if netted.

| cost bps/side | turnover | mean daily bps | std daily bps | Sharpe | cum bps | max DD bps | up days | Sharpe 95% CI |
|---|---|---|---|---|---|---|---|---|
| 0.000 | open_close | 30.855 | 6.576 | 74.480 | 1851.274 | 0.000 | 1.000 | [58.55, 113.95] |
| 0.500 | open_close | 20.855 | 6.576 | 50.341 | 1251.274 | 0.000 | 1.000 | [40.18, 75.69] |
| 0.500 | netted | 23.322 | 6.559 | 56.444 | 1399.332 | 0.000 | 1.000 |  |
| 1.000 | open_close | 10.855 | 6.576 | 26.202 | 651.274 | 0.000 | 1.000 |  |
| 1.000 | netted | 15.790 | 6.547 | 38.287 | 947.391 | 0.000 | 1.000 |  |
| 2.000 | open_close | -9.145 | 6.576 | -22.076 | -548.726 | -536.623 | 0.050 |  |
| 2.000 | netted | 0.725 | 6.536 | 1.761 | 43.508 | -31.045 | 0.467 |  |

**Quoted-spread cost assumption** (replaces the flat grid; not added to it): each traded name pays half its
own quoted spread at entry and at exit (exit spread = entry spread at 540 s, where t+60 is unobserved).
Average round-trip cost 7.19 bps per unit of gross traded (~3.59 bps per side): mean daily -41.01 bps, Sharpe -100.66 (95% CI [-117.16, -90.04]), up days 0.00, cumulative -2461 bps, max DD -2412 bps.
Limitation: top-of-book spreads in normalised price units; ignores depth, auction-uncross pricing and impact.

### Robustness: observed WAP(t+60)/WAP(t) − 1, same rule

Decision times restricted to [0, 60, 120, 180, 240, 300, 360, 420, 480] because WAP(t+60) is not observed after 540 s
(the target at 540 s exists but its WAP(600) is not in the data, and nothing is extrapolated).
This marks the self-financing long/short book at raw stock WAP (not index-relative); net-zero
dollar exposure removes the equal-weighted common move but not the unknown index-weighted one.

Target proxy on the same restricted schedule, gross: Sharpe 68.732, mean 25.181 bps/day.

| cost bps/side | turnover | mean daily bps | std daily bps | Sharpe | cum bps | max DD bps | up days | Sharpe 95% CI |
|---|---|---|---|---|---|---|---|---|
| 0.000 | open_close | 25.179 | 5.813 | 68.758 | 1510.714 | 0.000 | 1.000 | [54.19, 99.42] |
| 0.500 | open_close | 16.179 | 5.813 | 44.181 | 970.714 | 0.000 | 1.000 | [35.40, 63.24] |
| 0.500 | netted | 18.360 | 5.787 | 50.364 | 1101.599 | 0.000 | 1.000 |  |
| 1.000 | open_close | 7.179 | 5.813 | 19.603 | 430.714 | -2.271 | 0.950 |  |
| 1.000 | netted | 11.541 | 5.765 | 31.782 | 692.484 | 0.000 | 1.000 |  |
| 2.000 | open_close | -10.821 | 5.813 | -29.551 | -649.286 | -635.654 | 0.050 |  |
| 2.000 | netted | -2.096 | 5.732 | -5.804 | -125.746 | -120.692 | 0.317 |  |

## Statistical power

The moving-block bootstrap (5-day blocks, 2,000 resamples) resamples days; it assumes dependence dies out
within ~5 days and that the sample period is representative. With 60 holdout days the Sharpe CI is wide
(a Sharpe estimate's standard error over n days is roughly sqrt(252/n) ≈ 2 for n = 60).

Figures: `reports/figures/full/backtest.png`, `reports/figures/full/holdout_daily_returns.png`.