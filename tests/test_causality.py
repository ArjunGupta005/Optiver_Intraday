"""Parity, future-perturbation invariance, label isolation and state isolation."""
import numpy as np
import polars as pl

from optiver_tc import features as F
from optiver_tc.streaming import StreamingPredictor, replay

from conftest import make_raw

TOL = 1e-9


def _batch_pred(bundle, raw):
    feats = F.build_features(raw, bundle.stock_stats)
    return feats.select(["row_id"]).with_columns(pl.Series("pred_batch", bundle.predict_features(feats)))


def test_batch_streaming_prediction_parity(bundle, raw):
    b = _batch_pred(bundle, raw)
    s = replay(StreamingPredictor(bundle), raw)
    m = b.join(s, on="row_id")
    assert m.height == raw.height
    assert np.max(np.abs(m["pred_batch"].to_numpy() - m["pred"].to_numpy())) < TOL


def test_batch_streaming_feature_parity(bundle, raw):
    """Features computed from bounded history + current batch equal whole-day batch features."""
    full = F.build_features(raw, bundle.stock_stats).sort(["date_id", "seconds_in_bucket", "stock_id"])
    cols = bundle.columns
    pieces = []
    for d in raw["date_id"].unique().sort().to_list():
        day = raw.filter(pl.col("date_id") == d)
        for s in range(0, 550, 10):
            hist = day.filter(pl.col("seconds_in_bucket").is_between(s - F.HISTORY_SECONDS, s))
            f = F.build_features(hist, bundle.stock_stats).filter(pl.col("seconds_in_bucket") == s)
            pieces.append(f)
    stream = pl.concat(pieces).sort(["date_id", "seconds_in_bucket", "stock_id"])
    a = full.select(cols).to_numpy().astype(float)
    b = stream.select(cols).to_numpy().astype(float)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    m = ~np.isnan(a)
    assert np.max(np.abs(a[m] - b[m])) < 1e-9


def test_future_perturbation_invariance(bundle, raw):
    t0 = 250
    cut = (pl.col("date_id") > 1) | ((pl.col("date_id") == 1) & (pl.col("seconds_in_bucket") > t0))
    noise = np.random.default_rng(5).normal(0, 1, raw.height)
    pert = raw.with_columns([
        pl.when(cut).then(pl.col(c) * (1 + 0.01 * pl.Series(noise))).otherwise(pl.col(c)).alias(c)
        for c in ["wap", "bid_price", "ask_price", "reference_price", "matched_size", "imbalance_size", "bid_size"]
    ])
    keep = ~cut
    a = replay(StreamingPredictor(bundle), raw).filter(pl.col("row_id").is_in(raw.filter(keep)["row_id"].to_list()))
    b = replay(StreamingPredictor(bundle), pert).filter(pl.col("row_id").is_in(raw.filter(keep)["row_id"].to_list()))
    assert np.array_equal(a.sort("row_id")["pred"].to_numpy(), b.sort("row_id")["pred"].to_numpy())
    # batch features as well
    fa = F.build_features(raw, bundle.stock_stats).filter(keep).sort("row_id").select(bundle.columns).to_numpy()
    fb = F.build_features(pert, bundle.stock_stats).filter(keep).sort("row_id").select(bundle.columns).to_numpy()
    assert np.array_equal(np.nan_to_num(fa, nan=-999), np.nan_to_num(fb, nan=-999))


def test_label_isolation(bundle, raw):
    shuffled = raw.with_columns(pl.Series("target", np.random.default_rng(3).normal(0, 100, raw.height)))
    nolabel = raw.drop("target")
    a = replay(StreamingPredictor(bundle), raw)["pred"].to_numpy()
    b = replay(StreamingPredictor(bundle), shuffled)["pred"].to_numpy()
    c = replay(StreamingPredictor(bundle), nolabel)["pred"].to_numpy()
    assert np.array_equal(a, b) and np.array_equal(a, c)
    assert "target" not in bundle.columns and not any("target" in c for c in bundle.columns)


def test_no_state_bleed_across_dates(bundle, raw):
    """Predictions for date 2 do not depend on whether earlier dates were streamed first."""
    p_all = replay(StreamingPredictor(bundle), raw).filter(pl.col("date_id") == 2)
    p_one = replay(StreamingPredictor(bundle), raw.filter(pl.col("date_id") == 2))
    assert np.array_equal(p_all["pred"].to_numpy(), p_one["pred"].to_numpy())


def test_no_temporal_bleed_across_stocks(raw):
    """Perturbing stock 0's history leaves other stocks' temporal features unchanged."""
    pert = raw.with_columns(pl.when(pl.col("stock_id") == 0).then(pl.col("wap") * 1.05).otherwise(pl.col("wap")).alias("wap"))
    cols = F.TEMPORAL
    a = F.build_unfitted(raw).filter(pl.col("stock_id") != 0).sort("row_id").select(cols).to_numpy()
    b = F.build_unfitted(pert).filter(pl.col("stock_id") != 0).sort("row_id").select(cols).to_numpy()
    assert np.array_equal(np.nan_to_num(a, nan=-999), np.nan_to_num(b, nan=-999))


def test_temporal_state_resets_each_day():
    raw = make_raw(n_dates=2, n_stocks=4)
    f = F.build_unfitted(raw).filter(pl.col("seconds_in_bucket") == 0)
    for c in ["wap_ret_10", "wap_ret_60", "ref_chg_30", "wap_ret10_std_60"]:
        assert f[c].null_count() == f.height, c
    assert (f["n_hist_60"] == 0).all()
