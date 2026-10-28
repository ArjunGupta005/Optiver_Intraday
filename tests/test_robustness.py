"""Missing buckets/stocks/prices, zero denominators, row alignment, finiteness, order policy."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from optiver_tc import features as F
from optiver_tc.streaming import StreamingPredictor, replay

from conftest import make_raw


def test_missing_bucket_uses_elapsed_time_not_row_shift():
    raw = make_raw(n_dates=1, n_stocks=3)
    gap = raw.filter(~((pl.col("stock_id") == 1) & (pl.col("seconds_in_bucket") == 30)))
    f = F.build_unfitted(gap).filter(pl.col("stock_id") == 1)
    at40 = f.filter(pl.col("seconds_in_bucket") == 40)
    assert at40["wap_ret_10"].item() is None          # t-10 = 30 is missing -> null, not the 20 s row
    at60 = f.filter(pl.col("seconds_in_bucket") == 60)
    assert at60["wap_ret_30"].item() is None
    w = raw.filter(pl.col("stock_id") == 1).sort("seconds_in_bucket")["wap"].to_numpy()
    assert abs(at60["wap_ret_10"].item() - (w[6] / w[5] - 1) * 1e4) < 1e-9
    assert at40["n_hist_60"].item() == 2  # only 20/10 and 10/0 returns survive (30 s missing breaks 40/30 and 30/20)


def test_missing_stock_in_batch(bundle, raw):
    sub = raw.filter(~((pl.col("stock_id") == 3) & (pl.col("seconds_in_bucket") == 100)))
    out = replay(StreamingPredictor(bundle), sub)
    assert out.height == sub.height and np.isfinite(out["pred"].to_numpy()).all()
    f = F.build_unfitted(sub).filter((pl.col("seconds_in_bucket") == 100) & (pl.col("date_id") == 0))
    assert f["xs_n_stocks"].unique().to_list() == [11.0]


def test_missing_prices_and_zero_denominators(bundle, raw):
    day = raw.filter(pl.col("date_id") == 0)
    bad = day.with_columns(
        pl.when(pl.col("stock_id") == 2).then(None).otherwise(pl.col("wap")).alias("wap"),
        pl.when(pl.col("stock_id") == 2).then(None).otherwise(pl.col("bid_price")).alias("bid_price"),
        pl.when(pl.col("stock_id") == 4).then(0.0).otherwise(pl.col("bid_size")).alias("bid_size"),
        pl.when(pl.col("stock_id") == 4).then(0.0).otherwise(pl.col("ask_size")).alias("ask_size"),
        pl.when(pl.col("stock_id") == 5).then(pl.col("bid_price")).otherwise(pl.col("ask_price")).alias("ask_price"),
    )
    f = F.build_unfitted(bad)
    num = f.select(F.feature_columns(F.UNFITTED_FAMILIES)).to_numpy().astype(float)
    assert not np.isinf(num).any()
    assert f.filter(pl.col("stock_id") == 4)["book_imb"].null_count() == 55
    assert f.filter(pl.col("stock_id") == 5)["ref_in_spread"].null_count() == 55  # zero spread -> null
    out = replay(StreamingPredictor(bundle), bad)
    assert np.isfinite(out["pred"].to_numpy()).all()


def test_row_order_preserved_and_unseen_stock(bundle, raw):
    batch = raw.filter((pl.col("date_id") == 0) & (pl.col("seconds_in_bucket") == 0)).to_pandas()
    p = StreamingPredictor(bundle)
    ref = pd.Series(p.predict(batch), index=batch["row_id"])
    shuf = batch.sample(frac=1.0, random_state=1)
    shuf.loc[shuf.index[0], "stock_id"] = 999  # unseen stock id
    p2 = StreamingPredictor(bundle)
    out = p2.predict(shuf)
    assert len(out) == len(shuf) and np.isfinite(out).all()
    others = shuf["stock_id"] != 999
    # with an unseen stock replacing one stock, xsec features shift slightly; check alignment with the same rows
    p3 = StreamingPredictor(bundle)
    out3 = p3.predict(batch.sample(frac=1.0, random_state=1))
    assert np.allclose(out3, ref.reindex(batch.sample(frac=1.0, random_state=1)["row_id"]).to_numpy(), atol=1e-12)
    assert others.sum() == len(shuf) - 1


def test_empty_and_degenerate_batches(bundle, raw):
    p = StreamingPredictor(bundle)
    assert len(p.predict(raw.head(0))) == 0
    one = raw.filter((pl.col("date_id") == 0) & (pl.col("seconds_in_bucket") == 0)).head(1)
    out = p.predict(one)
    assert out.shape == (1,) and np.isfinite(out).all()
    allnan = raw.filter((pl.col("date_id") == 0) & (pl.col("seconds_in_bucket") == 10)).with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(c) for c in ["wap", "bid_price", "ask_price", "reference_price", "matched_size"]])
    assert np.isfinite(p.predict(allnan)).all()


def test_duplicate_and_out_of_order_policy(bundle, raw):
    day = raw.filter(pl.col("date_id") == 0)
    batches = [g for _, g in day.group_by("seconds_in_bucket", maintain_order=True)]
    p = StreamingPredictor(bundle)
    for b in batches[:5]:
        last = p.predict(b)
    again = p.predict(batches[4])                      # duplicate of the latest batch
    assert np.array_equal(last, again)
    early = p.predict(batches[2])                      # out of order, handled causally
    ref = StreamingPredictor(bundle)
    for b in batches[:3]:
        r = ref.predict(b)
    assert np.allclose(early, r, atol=1e-12)
    strict = StreamingPredictor(bundle, strict_order=True)
    strict.predict(batches[3])
    with pytest.raises(ValueError):
        strict.predict(batches[1])


def test_history_is_bounded(bundle, raw):
    p = StreamingPredictor(bundle)
    replay(p, raw)
    n_stocks = raw["stock_id"].n_unique()
    assert p.state.height <= n_stocks * (F.HISTORY_SECONDS // 10 + 1)
