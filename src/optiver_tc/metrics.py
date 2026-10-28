"""Forecast metrics. All metrics exclude rows whose target is missing (never imputed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _mask(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.isfinite(y) & np.isfinite(p)


def mae(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = _mask(y, p)
    return float(np.mean(np.abs(y[m] - p[m])))


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = _mask(y, p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def directional_accuracy(y, p) -> float:
    """Share of rows with y != 0 and p != 0 where sign(p) == sign(y). Rows with a zero on either side are excluded."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = _mask(y, p) & (y != 0) & (p != 0)
    return float(np.mean(np.sign(y[m]) == np.sign(p[m]))) if m.any() else float("nan")


def ic_by_timestamp(df: pd.DataFrame, pred: str = "pred", target: str = "target", min_names: int = 10) -> pd.DataFrame:
    """Cross-sectional Pearson and Spearman IC per (date_id, seconds_in_bucket).

    Degenerate groups (fewer than `min_names` finite pairs, or zero variance in either series)
    are returned with NaN IC and excluded from summary means.
    """
    d = df[["date_id", "seconds_in_bucket", pred, target]]
    d = d[np.isfinite(d[pred]) & np.isfinite(d[target])]
    out = []
    for (date, sec), g in d.groupby(["date_id", "seconds_in_bucket"], sort=True):
        x, y = g[pred].to_numpy(), g[target].to_numpy()
        if len(x) < min_names or np.std(x) == 0 or np.std(y) == 0:
            out.append((date, sec, len(x), np.nan, np.nan))
            continue
        pe = float(np.corrcoef(x, y)[0, 1])
        sp = float(stats.spearmanr(x, y).statistic)
        out.append((date, sec, len(x), pe, sp))
    return pd.DataFrame(out, columns=["date_id", "seconds_in_bucket", "n", "pearson_ic", "spearman_ic"])


def ic_fast(df: pd.DataFrame, pred: str = "pred", target: str = "target", min_names: int = 10) -> pd.DataFrame:
    """Vectorised equivalent of ic_by_timestamp (same degenerate-group policy)."""
    d = df[["date_id", "seconds_in_bucket", pred, target]].copy()
    d = d[np.isfinite(d[pred]) & np.isfinite(d[target])]
    g = d.groupby(["date_id", "seconds_in_bucket"], sort=True)
    d["rp"] = g[pred].rank()
    d["rt"] = g[target].rank()

    def corr(a: str, b: str) -> pd.Series:
        da = d[a] - g[a].transform("mean")
        db = d[b] - g[b].transform("mean")
        tmp = pd.DataFrame({"k1": d["date_id"], "k2": d["seconds_in_bucket"], "ab": da * db, "aa": da * da, "bb": db * db})
        s = tmp.groupby(["k1", "k2"], sort=True)[["ab", "aa", "bb"]].sum()
        den = np.sqrt(s["aa"] * s["bb"])
        return (s["ab"] / den.where(den > 0)).rename(None)

    n = g.size()
    res = pd.DataFrame({"n": n, "pearson_ic": corr(pred, target).values, "spearman_ic": corr("rp", "rt").values}, index=n.index)
    res.loc[res["n"] < min_names, ["pearson_ic", "spearman_ic"]] = np.nan
    res = res.reset_index()
    res.columns = ["date_id", "seconds_in_bucket", "n", "pearson_ic", "spearman_ic"]
    return res


def summary(df: pd.DataFrame, pred: str = "pred", target: str = "target") -> dict:
    y, p = df[target].to_numpy(float), df[pred].to_numpy(float)
    ic = ic_fast(df, pred, target)
    return {
        "n_rows_scored": int(_mask(y, p).sum()),
        "mae": mae(y, p),
        "rmse": rmse(y, p),
        "dir_acc": directional_accuracy(y, p),
        "pearson_ic_mean": float(ic["pearson_ic"].mean()),
        "spearman_ic_mean": float(ic["spearman_ic"].mean()),
        "spearman_ic_tstat": float(ic["spearman_ic"].mean() / ic["spearman_ic"].std() * np.sqrt(ic["spearman_ic"].notna().sum())),
        "n_ic_groups": int(ic["spearman_ic"].notna().sum()),
    }


def mae_by(df: pd.DataFrame, key: str, pred: str = "pred", target: str = "target") -> pd.DataFrame:
    d = df[np.isfinite(df[target]) & np.isfinite(df[pred])]
    err = (d[target] - d[pred]).abs()
    zero = d[target].abs()
    return pd.DataFrame({"mae": err.groupby(d[key]).mean(), "mae_zero": zero.groupby(d[key]).mean(),
                         "n": err.groupby(d[key]).size()}).reset_index()
