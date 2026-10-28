"""Chronological splits, label-horizon boundaries, and synthetic backtest arithmetic."""
import copy

import numpy as np
import pandas as pd
import pytest

from optiver_tc.backtest import (Rule, block_bootstrap, capped_scale, daily_returns, portfolio_weights,
                                 sharpe, simulate, wap_forward_returns)
from optiver_tc.config import load_config
from optiver_tc.splits import Fold, dev_folds, final_split, validate_folds


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/default.toml")


def test_dev_folds_are_chronological_and_disjoint_from_holdout(cfg):
    folds = dev_folds(cfg)
    h = final_split(cfg)
    for f in folds:
        assert f.train_end < f.val_start
        assert f.val_end < h.val_start
    for a, b in zip(folds, folds[1:]):
        assert a.val_end < b.val_start and b.train_end >= a.val_end  # expanding window
    assert h.train_end == h.val_start - 1 and h.val_end == 480


def test_validate_folds_rejects_leaky_configs():
    with pytest.raises(ValueError):
        validate_folds([Fold("x", 0, 50, 40, 60)], 400)   # train overlaps val
    with pytest.raises(ValueError):
        validate_folds([Fold("x", 0, 50, 51, 430)], 421)  # touches holdout
    with pytest.raises(ValueError):
        validate_folds([Fold("a", 0, 50, 51, 60), Fold("b", 0, 55, 56, 70)], 421)  # overlapping val windows


def test_label_horizon_stays_within_date(cfg):
    """target(t) uses WAP(t+60); max t is 540 so t+60 <= 600 is the same auction: whole-date splits suffice."""
    assert max(cfg["backtest"]["decision_seconds"]) + 60 <= 600
    assert cfg["splits"]["embargo_days"] == 0


def test_embargo_shifts_train_end(cfg):
    c = copy.deepcopy(cfg)
    c.raw["splits"]["embargo_days"] = 2
    assert dev_folds(c)[0].train_end == dev_folds(cfg)[0].train_end - 2


# ------------------------------------------------------------------------- backtest
def test_capped_scale_respects_cap_and_total():
    w = capped_scale(np.array([10.0, 1, 1, 1, 1]), 0.5, 0.2)
    assert abs(w.sum() - 0.5) < 1e-12 and w.max() <= 0.2 + 1e-12
    w = capped_scale(np.array([1.0, 1.0]), 0.5, 0.1)      # infeasible -> all at cap
    assert np.allclose(w, 0.1)


def test_portfolio_weights_neutral_capped_and_signed():
    rng = np.random.default_rng(0)
    s = rng.normal(size=200)
    w = portfolio_weights(s, Rule(cap=0.02, quantile=1.0, gross=1.0, min_names=10))
    assert abs(w.sum()) < 1e-12 and abs(np.abs(w).sum() - 1.0) < 1e-9 and np.abs(w).max() <= 0.02 + 1e-12
    d = s - s.mean()
    assert (np.sign(w[w != 0]) == np.sign(d[w != 0])).all()
    wq = portfolio_weights(s, Rule(cap=0.05, quantile=0.2, min_names=10))
    assert (wq != 0).sum() <= 2 * 40 and abs(wq.sum()) < 1e-12


def test_portfolio_weights_degenerate_cases():
    r = Rule(min_names=10)
    assert not portfolio_weights(np.ones(100), r).any()             # zero signal dispersion
    assert not portfolio_weights(np.arange(15.0), r).any()          # too few names per side
    s = np.r_[np.random.default_rng(1).normal(size=50), [np.nan] * 5]
    w = portfolio_weights(s, Rule(cap=0.05, min_names=10))
    assert (w[-5:] == 0).all() and abs(w.sum()) < 1e-12
    # infeasible cap: 12 longs x 0.02 < 0.5 -> both sides shrink to 0.24, still neutral
    s2 = np.r_[np.ones(12), -np.ones(12)]
    w2 = portfolio_weights(s2, Rule(cap=0.02, min_names=10))
    assert abs(w2.sum()) < 1e-12 and abs(np.abs(w2).sum() - 0.48) < 1e-12


def _toy(n_dates=3, n=20, secs=(0, 60, 120), sign=1.0):
    rows = []
    for d in range(n_dates):
        for s in secs:
            for i in range(n):
                sig = i - (n - 1) / 2
                rows.append({"date_id": d, "seconds_in_bucket": s, "stock_id": i, "pred": sig,
                             "target": sign * 10.0 * np.sign(sig)})  # +/-10 bps, perfectly predicted
    return pd.DataFrame(rows)


def test_sign_bps_conversion_and_costs():
    df = _toy()
    rule = Rule(cap=1.0, min_names=5)
    iv = simulate(df, rule, [0, 60, 120])
    # gross 1: long half earns +10 bps, short half earns +10 bps -> 10 bps = 0.001 per interval
    assert np.allclose(iv["ret"], 1e-3)
    d0 = daily_returns(iv, 0.0)
    assert np.allclose(d0, 3e-3)
    d1 = daily_returns(iv, 1.0)                 # 1 bp/side, open+close, gross 1 -> 2 bps per interval
    assert np.allclose(d1, 3e-3 - 3 * 2e-4)
    bad = simulate(_toy(sign=-1.0), rule, [0, 60, 120])
    assert np.allclose(bad["ret"], -1e-3)


def test_index_component_cancels_when_net_zero():
    df = _toy()
    shifted = df.assign(target=df["target"] + 7.0)   # common index move added to every stock
    rule = Rule(cap=1.0, min_names=5)
    assert np.allclose(simulate(df, rule, [0, 60, 120])["ret"], simulate(shifted, rule, [0, 60, 120])["ret"])


def test_non_overlapping_schedule_and_missing_timestamps():
    df = _toy(secs=(0, 10, 60, 70, 120))
    iv = simulate(df, Rule(cap=1.0, min_names=5), [0, 60, 120, 180])
    assert iv["seconds_in_bucket"].tolist() == [0, 60, 120] * 3     # 180 absent -> skipped, never filled
    assert np.diff(sorted(set(iv["seconds_in_bucket"]))).min() >= 60


def test_netted_turnover_and_daily_sharpe():
    df = _toy()
    iv = simulate(df, Rule(cap=1.0, min_names=5), [0, 60, 120])
    # same weights each interval: open 1, no change, no change, final close 1 -> 2 per day
    assert np.allclose(iv.groupby("date_id")["netted_turnover"].sum(), 2.0)
    daily = pd.Series([0.001, 0.002, -0.0005, 0.0015])
    assert abs(sharpe(daily) - np.sqrt(252) * daily.mean() / daily.std(ddof=1)) < 1e-12
    dr = daily_returns(iv, 0.0, all_dates=[0, 1, 2, 3])
    assert dr.loc[3] == 0.0 and len(dr) == 4


def test_bootstrap_shapes():
    rng = np.random.default_rng(0)
    b = block_bootstrap(pd.Series(rng.normal(1e-4, 1e-3, 120)), reps=200, block=5, seed=0)
    assert b["sharpe_ci95"][0] < b["sharpe_ci95"][1] and 0 <= b["p_sharpe_le_0"] <= 1


def test_wap_forward_returns_no_extrapolation():
    raw = pd.DataFrame({"stock_id": 0, "date_id": 0, "seconds_in_bucket": [0, 60, 480, 540],
                        "wap": [1.0, 1.001, 1.002, 1.003]})
    r = wap_forward_returns(raw).set_index("seconds_in_bucket")["wap_ret"]
    assert abs(r[0] - 0.001) < 1e-12 and abs(r[480] - (1.003 / 1.002 - 1)) < 1e-12
    assert np.isnan(r[540]) and np.isnan(r[60])


def test_quoted_spread_cost_charged_once_per_side():
    from optiver_tc.backtest import half_spread_round_trip, spread_cost_summary
    df = _toy(n_dates=2, secs=(0, 60))
    sp = df[["stock_id", "date_id", "seconds_in_bucket"]].assign(spread_bps=4.0)
    rt = half_spread_round_trip(sp)
    assert np.allclose(rt["cost_rt_bps"], 4.0)          # 2 bps at entry + 2 bps at exit
    bt = {"bootstrap_reps": 20, "bootstrap_block_days": 1}
    res, net = spread_cost_summary(df, sp, Rule(cap=1.0, min_names=5), [0, 60], [0, 1], bt, 0)
    # gross 1: +10 bps per interval, minus 4 bps round trip, 2 intervals per day -> 12 bps per day
    assert np.allclose(net, 12e-4) and abs(res["avg_round_trip_cost_bps_per_unit_gross"] - 4.0) < 1e-9
