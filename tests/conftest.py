import optiver_tc  # noqa: F401  (imports lightgbm first; see package __init__)

import numpy as np
import polars as pl
import pytest

from optiver_tc import features as F
from optiver_tc.data import SCHEMA
from optiver_tc.models import LGBMModel, RidgeModel
from optiver_tc.streaming import ModelBundle

# vendor templates are raw inputs, never test modules
collect_ignore_glob = ["../data/*"]


def make_raw(n_dates: int = 3, n_stocks: int = 12, seed: int = 0, date0: int = 0) -> pl.DataFrame:
    """Synthetic auction panel with the real schema, missing far/near before 300 s and a few gaps."""
    rng = np.random.default_rng(seed)
    rows = []
    tid = 0
    for d in range(date0, date0 + n_dates):
        wap = np.ones(n_stocks)
        for s in range(0, 550, 10):
            if s > 0:
                wap = wap * (1 + rng.normal(0, 5e-4, n_stocks))
            for i in range(n_stocks):
                spread = abs(rng.normal(3e-4, 1e-4)) + 1e-5
                bid, ask = wap[i] - spread / 2, wap[i] + spread / 2
                bs, as_ = rng.lognormal(9, 1), rng.lognormal(9, 1)
                matched = rng.lognormal(16, 1)
                imb = abs(rng.normal(0, 0.2)) * matched
                flag = int(rng.choice([-1, 0, 1]))
                rows.append({
                    "stock_id": i, "date_id": d, "seconds_in_bucket": s,
                    "imbalance_size": imb if flag else 0.0, "imbalance_buy_sell_flag": flag,
                    "reference_price": wap[i] * (1 + rng.normal(0, 2e-4)), "matched_size": matched,
                    "far_price": wap[i] * (1 + rng.normal(0, 1e-3)) if s >= 300 else None,
                    "near_price": wap[i] * (1 + rng.normal(0, 5e-4)) if s >= 300 else None,
                    "bid_price": bid, "bid_size": bs, "ask_price": ask, "ask_size": as_,
                    "wap": (bid * as_ + ask * bs) / (bs + as_),
                    "target": float(rng.normal(0, 8)), "time_id": tid, "row_id": f"{d}_{s}_{i}",
                })
            tid += 1
    return pl.DataFrame(rows, schema=SCHEMA)


@pytest.fixture(scope="session")
def raw() -> pl.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def bundle(tmp_path_factory, raw) -> ModelBundle:
    """Tiny fitted bundle: LightGBM + ridge blend, demeaning and clipping enabled (exercises every path)."""
    fams = list(F.FAMILIES)
    cols = F.feature_columns(fams)
    train = make_raw(n_dates=4, n_stocks=12, seed=1, date0=100)
    feats = F.build_unfitted(train)
    stats = F.fit_stock_stats(feats)
    feats = F.stock_features(feats, stats)
    X = feats.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    y = feats["target"].to_numpy()
    lgbm = LGBMModel({"objective": "l1", "num_leaves": 7, "min_data_in_leaf": 20, "learning_rate": 0.1},
                     categorical=["stock_id"], n_threads=1).fit(X, y, cols, num_boost_round=20)
    ridge_cols = [c for c in cols if c != "stock_id"]
    ridge = RidgeModel(alpha=10.0).fit(X[:, [cols.index(c) for c in ridge_cols]], y)
    schema = {"feature_columns": cols, "families": fams, "categorical": ["stock_id"],
              "models": [{"kind": "lgbm", "file": "lgbm.txt", "weight": 0.7, "columns": None},
                         {"kind": "ridge", "file": "ridge.json", "weight": 0.3, "columns": ridge_cols}],
              "postprocess": {"demean": True, "demean_strength": 0.5, "clip": 30.0},
              "fallback": 0.0, "history_seconds": F.HISTORY_SECONDS}
    d = tmp_path_factory.mktemp("bundle")
    ModelBundle.save(d, schema, [lgbm, ridge], stats)
    return ModelBundle.load(d, n_threads=1)
