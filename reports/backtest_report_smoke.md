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
| 0.0200 | 1.0000 | 1.0000 | 10.0000 | 43.6974 | 16.7913 | 16.1210 | 6.1899 |
| 0.0200 | 0.4000 | 1.0000 | 10.0000 | 42.6463 | 17.3942 | 16.1790 | 6.6079 |
| 0.0200 | 0.2000 | 1.0000 | 10.0000 | 49.1866 | 30.5289 | 12.6541 | 7.8541 |
| 0.0500 | 1.0000 | 1.0000 | 10.0000 | 45.2626 | 27.4307 | 25.3829 | 15.3829 |
| 0.0500 | 0.4000 | 1.0000 | 10.0000 | 44.8934 | 27.5474 | 25.8811 | 15.8811 |
| 0.0500 | 0.2000 | 1.0000 | 10.0000 | 46.9390 | 30.6443 | 28.8062 | 18.8062 |

Selected and frozen: `{'cap': 0.05, 'quantile': 0.2, 'gross': 1.0, 'min_names': 10}`.

## Development OOF (dates 301–420; the rule was chosen on these, so this is optimistic)

Days 18, intervals 180 (active 180), mean gross 1.000,
max |net| 1.25e-16, max |w| 0.0500, mean names long/short 12.0/12.0,
daily turnover 20.0x capital (open+close) vs 15.08x if netted.

| cost bps/side | turnover | mean daily bps | std daily bps | Sharpe | cum bps | max DD bps | up days | Sharpe 95% CI |
|---|---|---|---|---|---|---|---|---|
| 0.000 | open_close | 28.806 | 9.742 | 46.939 | 518.512 | 0.000 | 1.000 | [43.27, 66.22] |
| 0.500 | open_close | 18.806 | 9.742 | 30.644 | 338.512 | 0.000 | 1.000 | [28.28, 43.25] |
| 0.500 | netted | 21.268 | 9.581 | 35.239 | 382.815 | 0.000 | 1.000 |  |
| 1.000 | open_close | 8.806 | 9.742 | 14.350 | 158.512 | -2.813 | 0.833 |  |
| 1.000 | netted | 13.729 | 9.424 | 23.127 | 247.118 | 0.000 | 1.000 |  |
| 2.000 | open_close | -11.194 | 9.742 | -18.240 | -201.488 | -199.580 | 0.167 |  |
| 2.000 | netted | -1.349 | 9.125 | -2.346 | -24.275 | -49.838 | 0.333 |  |

**Quoted-spread cost assumption** (replaces the flat grid; not added to it): each traded name pays half its
own quoted spread at entry and at exit (exit spread = entry spread at 540 s, where t+60 is unobserved).
Average round-trip cost 6.79 bps per unit of gross traded (~3.39 bps per side): mean daily -39.06 bps, Sharpe -127.62 (95% CI [-197.84, -113.72]), up days 0.00, cumulative -703 bps, max DD -672 bps.
Limitation: top-of-book spreads in normalised price units; ignores depth, auction-uncross pricing and impact.

## Final holdout (dates 421–480; rule frozen)

Days 10, intervals 100 (active 100), mean gross 1.000,
max |net| 1.04e-16, max |w| 0.0500, mean names long/short 12.0/12.0,
daily turnover 20.0x capital (open+close) vs 15.28x if netted.

| cost bps/side | turnover | mean daily bps | std daily bps | Sharpe | cum bps | max DD bps | up days | Sharpe 95% CI |
|---|---|---|---|---|---|---|---|---|
| 0.000 | open_close | 24.223 | 4.900 | 78.474 | 242.234 | 0.000 | 1.000 | [59.15, 140.22] |
| 0.500 | open_close | 14.223 | 4.900 | 46.078 | 142.234 | 0.000 | 1.000 | [34.54, 81.32] |
| 0.500 | netted | 16.583 | 4.742 | 55.516 | 165.833 | 0.000 | 1.000 |  |
| 1.000 | open_close | 4.223 | 4.900 | 13.682 | 42.234 | -5.381 | 0.800 |  |
| 1.000 | netted | 8.943 | 4.594 | 30.905 | 89.431 | 0.000 | 1.000 |  |
| 2.000 | open_close | -15.777 | 4.900 | -51.110 | -157.766 | -139.482 | 0.000 |  |
| 2.000 | netted | -6.337 | 4.332 | -23.224 | -63.372 | -54.050 | 0.000 |  |

**Quoted-spread cost assumption** (replaces the flat grid; not added to it): each traded name pays half its
own quoted spread at entry and at exit (exit spread = entry spread at 540 s, where t+60 is unobserved).
Average round-trip cost 5.67 bps per unit of gross traded (~2.83 bps per side): mean daily -32.43 bps, Sharpe -120.66 (95% CI [-261.66, -89.06]), up days 0.00, cumulative -324 bps, max DD -287 bps.
Limitation: top-of-book spreads in normalised price units; ignores depth, auction-uncross pricing and impact.

### Robustness: observed WAP(t+60)/WAP(t) − 1, same rule

Decision times restricted to [0, 60, 120, 180, 240, 300, 360, 420, 480] because WAP(t+60) is not observed after 540 s
(the target at 540 s exists but its WAP(600) is not in the data, and nothing is extrapolated).
This marks the self-financing long/short book at raw stock WAP (not index-relative); net-zero
dollar exposure removes the equal-weighted common move but not the unknown index-weighted one.

Target proxy on the same restricted schedule, gross: Sharpe 73.586, mean 20.803 bps/day.

| cost bps/side | turnover | mean daily bps | std daily bps | Sharpe | cum bps | max DD bps | up days | Sharpe 95% CI |
|---|---|---|---|---|---|---|---|---|
| 0.000 | open_close | 20.805 | 4.487 | 73.601 | 208.055 | 0.000 | 1.000 | [58.87, 110.46] |
| 0.500 | open_close | 11.805 | 4.487 | 41.763 | 118.055 | 0.000 | 1.000 | [32.65, 64.73] |
| 0.500 | netted | 13.910 | 4.346 | 50.804 | 139.095 | 0.000 | 1.000 |  |
| 1.000 | open_close | 2.805 | 4.487 | 9.925 | 28.055 | -6.117 | 0.800 |  |
| 1.000 | netted | 7.014 | 4.217 | 26.402 | 70.136 | -1.043 | 0.900 |  |
| 2.000 | open_close | -15.195 | 4.487 | -53.752 | -151.945 | -134.081 | 0.000 |  |
| 2.000 | netted | -6.778 | 3.999 | -26.906 | -67.783 | -57.937 | 0.000 |  |

## Statistical power

The moving-block bootstrap (5-day blocks, 2,000 resamples) resamples days; it assumes dependence dies out
within ~5 days and that the sample period is representative. With 60 holdout days the Sharpe CI is wide
(a Sharpe estimate's standard error over n days is roughly sqrt(252/n) ≈ 2 for n = 60).

Figures: `reports/figures/full/backtest.png`, `reports/figures/full/holdout_daily_returns.png`.