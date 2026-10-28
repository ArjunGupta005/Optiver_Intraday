"""Shared feature implementation used by BOTH batch research and streaming inference.

Every function here is a pure transformation of a polars DataFrame. Batch research calls
`build_features` on whole dates; the streaming predictor calls the very same function on
(bounded intraday history + current batch) and keeps the rows of the current timestamp.

Causality contract (enforced by tests):
* row-wise features use only the row itself;
* temporal features use rows of the same (stock_id, date_id) at seconds_in_bucket - L exactly
  (elapsed-time joins, not row shifts; a missing bucket yields null, never interpolation);
* cross-sectional features use only rows sharing the same (date_id, seconds_in_bucket),
  i.e. exactly one API batch;
* stock statistics are fitted on training dates only (see `fit_stock_stats`).
No feature reads `target`, `row_id`, `time_id` or future rows.
"""
from __future__ import annotations

import polars as pl

KEYS = ["stock_id", "date_id", "seconds_in_bucket"]
TS_KEYS = ["date_id", "seconds_in_bucket"]
EPS_NONE = None  # zero/null denominators produce null (NaN), which LightGBM treats as missing

LAGS = (10, 30, 60)
ROLL_STEPS = 6          # rolling window = 6 x 10 s = 60 s, trailing, includes current timestamp
STEP = 10               # nominal cadence verified by the audit; lags are still joined by elapsed time
HISTORY_SECONDS = max(max(LAGS), ROLL_STEPS * STEP)  # bounded history the streaming state must keep


def _safe_div(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """a / b, null when b is null or exactly zero (documented zero-denominator policy)."""
    return pl.when(b.is_not_null() & (b != 0)).then(a / b).otherwise(None)


def _bps(ratio: pl.Expr) -> pl.Expr:
    return (ratio - 1.0) * 1e4


# --------------------------------------------------------------------------- row-wise families
BOOK = [
    "spread_bps", "book_imb", "log_bid_size", "log_ask_size", "log_quoted",
    "wap_mid_bps", "wap_open_bps",
]
AUCTION = [
    "imbalance_buy_sell_flag", "signed_imb_log", "log_matched", "imb_ratio", "imb_total_ratio",
    "matched_quoted_log", "imb_quoted_log", "ref_wap_bps", "ref_mid_bps", "ref_open_bps",
    "far_ref_bps", "near_ref_bps", "far_near_bps", "near_wap_bps",
    "far_missing", "near_missing", "ref_in_spread",
]
INTERACT = [
    "imb_x_spread", "flag_x_bookimb", "imb_x_refwap", "bookimb_x_spread", "bid_ask_minus_matched",
]


def row_features(df: pl.DataFrame) -> pl.DataFrame:
    c = pl.col
    mid = (c("bid_price") + c("ask_price")) / 2
    quoted = c("bid_size") + c("ask_size")
    signed_imb = c("imbalance_buy_sell_flag").cast(pl.Float64) * c("imbalance_size")
    out = df.with_columns(
        # --- top of book
        spread_bps=_safe_div(c("ask_price") - c("bid_price"), mid) * 1e4,
        book_imb=_safe_div(c("bid_size") - c("ask_size"), quoted),
        log_bid_size=c("bid_size").clip(lower_bound=0).log1p(),
        log_ask_size=c("ask_size").clip(lower_bound=0).log1p(),
        log_quoted=quoted.clip(lower_bound=0).log1p(),
        wap_mid_bps=_bps(_safe_div(c("wap"), mid)),
        wap_open_bps=(c("wap") - 1.0) * 1e4,  # WAP is normalised to 1 at the auction start
        # --- auction
        signed_imb_log=signed_imb.sign() * c("imbalance_size").clip(lower_bound=0).log1p(),
        log_matched=c("matched_size").clip(lower_bound=0).log1p(),
        imb_ratio=_safe_div(signed_imb, c("matched_size")),
        imb_total_ratio=_safe_div(signed_imb, c("matched_size") + c("imbalance_size")),
        matched_quoted_log=(c("matched_size").clip(lower_bound=0).log1p() - quoted.clip(lower_bound=0).log1p()),
        imb_quoted_log=signed_imb.sign() * (c("imbalance_size").clip(lower_bound=0).log1p() - quoted.clip(lower_bound=0).log1p()),
        ref_wap_bps=_bps(_safe_div(c("reference_price"), c("wap"))),
        ref_mid_bps=_bps(_safe_div(c("reference_price"), mid)),
        ref_open_bps=(c("reference_price") - 1.0) * 1e4,
        far_ref_bps=_bps(_safe_div(c("far_price"), c("reference_price"))),
        near_ref_bps=_bps(_safe_div(c("near_price"), c("reference_price"))),
        far_near_bps=_bps(_safe_div(c("far_price"), c("near_price"))),
        near_wap_bps=_bps(_safe_div(c("near_price"), c("wap"))),
        far_missing=c("far_price").is_null().cast(pl.Float64),
        near_missing=c("near_price").is_null().cast(pl.Float64),
        ref_in_spread=_safe_div(c("reference_price") - c("bid_price"), c("ask_price") - c("bid_price")),
    )
    out = out.with_columns(
        # --- interactions (few, each with a hypothesis; see reports/research_report.md)
        imb_x_spread=c("imb_ratio") * c("spread_bps"),
        flag_x_bookimb=c("imbalance_buy_sell_flag").cast(pl.Float64) * c("book_imb"),
        imb_x_refwap=c("imb_ratio") * c("ref_wap_bps"),
        bookimb_x_spread=c("book_imb") * c("spread_bps"),
        bid_ask_minus_matched=_safe_div(c("bid_size") - c("ask_size"), c("matched_size")),
    )
    return out


# --------------------------------------------------------------------------- temporal family
TEMPORAL_BASE = ["imb_ratio", "log_matched", "book_imb", "spread_bps", "ref_open_bps"]


def _temporal_names() -> list[str]:
    names = []
    for L in LAGS:
        names += [f"wap_ret_{L}", f"ref_chg_{L}", f"imb_ratio_chg_{L}", f"log_matched_chg_{L}",
                  f"book_imb_chg_{L}", f"spread_chg_{L}"]
    names += ["wap_ret10_mean_60", "wap_ret10_std_60", "n_hist_60"]
    return names


TEMPORAL = _temporal_names()


def _lag_block(keys: pl.DataFrame, df: pl.DataFrame, cols: list[str], lag: int) -> pl.DataFrame:
    """Values of `cols` at exactly (stock, date, seconds - lag); null where that row does not exist."""
    right = df.select(
        pl.col("stock_id"), pl.col("date_id"),
        (pl.col("seconds_in_bucket") + lag).alias("seconds_in_bucket"),
        *[pl.col(col).alias(f"{col}__l{lag}") for col in cols],
    )
    out = keys.join(right, on=KEYS, how="left", maintain_order="left")
    if out.height != keys.height:
        raise ValueError("duplicate (stock_id, date_id, seconds_in_bucket) keys in feature input")
    return out.drop(KEYS)


def temporal_features(df: pl.DataFrame, output_seconds: int | None = None) -> pl.DataFrame:
    """Elapsed-time lags within (stock_id, date_id). Requires row_features columns.

    `df` supplies history; rows are emitted for all of `df` (batch research) or only for
    seconds_in_bucket == output_seconds (streaming). The per-row arithmetic is identical.
    """
    c = pl.col
    left = df if output_seconds is None else df.filter(c("seconds_in_bucket") == output_seconds)
    keys = left.select(KEYS)
    wap_lags = sorted(set(LAGS) | {STEP * k for k in range(1, ROLL_STEPS + 1)})
    blocks = [_lag_block(keys, df, ["wap"] + (TEMPORAL_BASE if L in LAGS else []), L) for L in wap_lags]
    work = left.hstack([s for b in blocks for s in b.get_columns()])
    exprs = []
    for L in LAGS:
        exprs += [
            (_bps(_safe_div(c("wap"), c(f"wap__l{L}")))).alias(f"wap_ret_{L}"),
            (c("ref_open_bps") - c(f"ref_open_bps__l{L}")).alias(f"ref_chg_{L}"),
            (c("imb_ratio") - c(f"imb_ratio__l{L}")).alias(f"imb_ratio_chg_{L}"),
            (c("log_matched") - c(f"log_matched__l{L}")).alias(f"log_matched_chg_{L}"),
            (c("book_imb") - c(f"book_imb__l{L}")).alias(f"book_imb_chg_{L}"),
            (c("spread_bps") - c(f"spread_bps__l{L}")).alias(f"spread_chg_{L}"),
        ]
    # trailing 60 s window of 10 s WAP returns: r_k = wap(t-10(k-1)) / wap(t-10k) - 1, k = 1..6
    for k in range(1, ROLL_STEPS + 1):
        num = c("wap") if k == 1 else c(f"wap__l{STEP * (k - 1)}")
        exprs.append(_bps(_safe_div(num, c(f"wap__l{STEP * k}"))).alias(f"__r{k}"))
    work = work.with_columns(exprs)
    r = [c(f"__r{k}") for k in range(1, ROLL_STEPS + 1)]
    n = pl.sum_horizontal([x.is_not_null().cast(pl.Float64) for x in r])
    s = pl.sum_horizontal([x.fill_null(0.0) for x in r])
    ss = pl.sum_horizontal([(x * x).fill_null(0.0) for x in r])
    mean = _safe_div(s, n)
    var = pl.when(n >= 2).then((ss - n * mean * mean) / (n - 1)).otherwise(None)
    work = work.with_columns(
        wap_ret10_mean_60=mean,
        wap_ret10_std_60=var.clip(lower_bound=0).sqrt(),
        n_hist_60=n,
    )
    return work.drop([col for col in work.columns if "__" in col])


# --------------------------------------------------------------------------- cross-sectional family
XSEC_BASE = ["wap_ret_10", "wap_ret_60", "wap_open_bps", "imb_ratio", "book_imb", "ref_wap_bps",
             "spread_bps", "log_matched", "imb_ratio_chg_60"]
XSEC = [f"{b}_dm" for b in XSEC_BASE] + [f"{b}_rk" for b in XSEC_BASE] + \
       ["xs_wap_ret_60_mean", "xs_wap_ret_60_std", "xs_n_stocks"]


def xsec_features(df: pl.DataFrame) -> pl.DataFrame:
    """Contemporaneous cross-section = rows sharing (date_id, seconds_in_bucket) = one API batch."""
    c = pl.col
    exprs = []
    for b in XSEC_BASE:
        cnt = c(b).count().over(TS_KEYS).cast(pl.Float64)
        exprs.append((c(b) - c(b).mean().over(TS_KEYS)).alias(f"{b}_dm"))
        exprs.append(_safe_div(c(b).rank("average").over(TS_KEYS).cast(pl.Float64) - 0.5, cnt).alias(f"{b}_rk"))
    exprs += [
        c("wap_ret_60").mean().over(TS_KEYS).alias("xs_wap_ret_60_mean"),
        c("wap_ret_60").std().over(TS_KEYS).alias("xs_wap_ret_60_std"),
        c("stock_id").count().over(TS_KEYS).cast(pl.Float64).alias("xs_n_stocks"),
    ]
    return df.with_columns(exprs)


# --------------------------------------------------------------------------- fitted stock statistics
STOCK_STAT_COLS = {"log_matched": "st_med_log_matched", "spread_bps": "st_med_spread",
                   "log_quoted": "st_med_log_quoted", "abs_wap_ret_10": "st_med_absret10"}
STOCK = list(STOCK_STAT_COLS.values()) + ["log_matched_rel", "spread_rel", "wap_ret_10_volnorm"]


def fit_stock_stats(train_feats: pl.DataFrame) -> pl.DataFrame:
    """Per-stock medians of NON-target quantities, fitted on training rows only."""
    return (train_feats.lazy()
            .with_columns(abs_wap_ret_10=pl.col("wap_ret_10").abs())
            .group_by("stock_id")
            .agg([pl.col(k).median().alias(v) for k, v in STOCK_STAT_COLS.items()])
            .sort("stock_id")
            .collect())


def stock_features(df: pl.DataFrame, stock_stats: pl.DataFrame) -> pl.DataFrame:
    c = pl.col
    st = stock_stats.with_columns(pl.col("stock_id").cast(df.schema["stock_id"]))
    df = df.drop([v for v in STOCK if v in df.columns])
    out = df.join(st, on="stock_id", how="left", maintain_order="left")  # unseen stocks -> null
    return out.with_columns(
        log_matched_rel=c("log_matched") - c("st_med_log_matched"),
        spread_rel=_safe_div(c("spread_bps"), c("st_med_spread")),
        wap_ret_10_volnorm=_safe_div(c("wap_ret_10"), c("st_med_absret10")),
    )


# --------------------------------------------------------------------------- schema
META = ["seconds_in_bucket", "stock_id"]
FAMILIES = {"book": BOOK, "auction": AUCTION, "interact": INTERACT, "temporal": TEMPORAL,
            "xsec": XSEC, "stock": STOCK}
UNFITTED_FAMILIES = ["book", "auction", "interact", "temporal", "xsec"]
CATEGORICAL = ["stock_id"]


def feature_columns(families: list[str]) -> list[str]:
    cols = list(META)
    for f in families:
        cols += FAMILIES[f]
    return cols


def build_unfitted(df: pl.DataFrame, output_seconds: int | None = None) -> pl.DataFrame:
    """Row-wise + temporal + cross-sectional features (no fitted state).

    output_seconds=None: features for every row (batch research). An integer: features only for the
    rows at that seconds_in_bucket, with the other rows of `df` used as history (streaming). The
    cross-section of the emitted rows is complete either way, so outputs are identical.
    """
    return xsec_features(temporal_features(row_features(df), output_seconds))


def build_features(df: pl.DataFrame, stock_stats: pl.DataFrame | None, output_seconds: int | None = None) -> pl.DataFrame:
    out = build_unfitted(df, output_seconds)
    if stock_stats is not None:
        out = stock_features(out, stock_stats)
    return out
