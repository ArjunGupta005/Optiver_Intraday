"""Target-based research-proxy backtest (NOT a demonstrated live strategy).

Conventions
-----------
* Decision times: a fixed schedule of seconds_in_bucket separated by exactly 60 s (default 0, 60, ..., 540),
  so consecutive 60 s label horizons tile the auction without overlap. A timestamp absent from the data is
  skipped (no filling from later observations).
* At each decision time the finite out-of-sample predictions of that cross-section are demeaned and split
  into a long book (positive demeaned signal) and a short book (negative). Each book is scaled to
  gross_exposure/2 of capital with a per-stock cap |w| <= cap (water-filling). If either side cannot reach
  gross/2 under the cap, both sides are scaled to the smaller feasible size, so net exposure is exactly 0.
  Fewer than `min_names_per_side` names on a side -> no position for that interval.
* Positions open at t and close at t+60. Interval proxy return = sum_i w_i * target_i / 1e4.
  With sum_i w_i = 0 the index term of the target cancels:
      sum w_i (r_i - r_idx) = sum w_i r_i - r_idx * sum w_i = sum w_i r_i,
  so the proxy is the stock-WAP P&L of a dollar-neutral book, marked at WAP (not executable prices).
* Capital: 1 unit per day. Every interval deploys gross exposure <= 1 unit; interval returns are summed into
  a daily return (no intraday compounding). Outside the ~10 auction minutes the capital is idle.
* Costs: c bps per side per unit of traded notional. Each interval is opened and closed, so cost per interval
  = 2 * c * gross / 1e4 (round trip = 2c). A 'netted' variant charging only |w_t - w_{t-1}| (plus the final
  close) is reported as a sensitivity. The cost grid is hypothetical, not measured.
* Sharpe = sqrt(252) * mean(daily) / std(daily, ddof=1). Never computed from pooled stock rows.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, write_json

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- weights
def capped_scale(x: np.ndarray, total: float, cap: float, iters: int = 100) -> np.ndarray:
    """Weights w = min(k*x, cap) with sum(w) = total for x >= 0 (water-filling). If n*cap < total, all at cap."""
    x = np.asarray(x, float)
    if total <= 0 or x.sum() <= 0:
        return np.zeros_like(x)
    if (x > 0).sum() * cap <= total + 1e-15:
        return np.where(x > 0, cap, 0.0)
    w = np.zeros_like(x)
    free = x > 0
    remaining = total
    for _ in range(iters):
        k = remaining / x[free].sum()
        trial = k * x[free]
        over = trial > cap
        if not over.any():
            w[free] = trial
            break
        idx = np.flatnonzero(free)[over]
        w[idx] = cap
        free[idx] = False
        remaining = total - w.sum()
        if not free.any() or remaining <= 0:
            break
    return w


@dataclass
class Rule:
    cap: float = 0.02
    quantile: float = 1.0          # 1.0 = all names; q<1 = top/bottom q fraction by demeaned signal
    gross: float = 1.0
    min_names: int = 10

    def to_dict(self) -> dict:
        return {"cap": self.cap, "quantile": self.quantile, "gross": self.gross, "min_names": self.min_names}


def portfolio_weights(signal: np.ndarray, rule: Rule) -> np.ndarray:
    s = np.asarray(signal, float)
    w = np.zeros_like(s)
    fin = np.isfinite(s)
    if fin.sum() < 2 * rule.min_names:
        return w
    d = np.where(fin, s - s[fin].mean(), 0.0)
    if np.allclose(d[fin], 0):
        return w
    long_m, short_m = fin & (d > 0), fin & (d < 0)
    if rule.quantile < 1.0:
        n_sel = max(rule.min_names, int(np.floor(rule.quantile * fin.sum())))
        order = np.argsort(np.where(fin, d, np.nan))  # NaN sorted last
        n_fin = fin.sum()
        bottom = order[:n_sel]
        top = order[n_fin - n_sel:n_fin]
        long_m = np.zeros_like(fin)
        short_m = np.zeros_like(fin)
        long_m[top] = d[top] > 0
        short_m[bottom] = d[bottom] < 0
    nl, ns = int(long_m.sum()), int(short_m.sum())
    if nl < rule.min_names or ns < rule.min_names:
        return w
    side = min(rule.gross / 2, nl * rule.cap, ns * rule.cap)
    w[long_m] = capped_scale(d[long_m], side, rule.cap)
    w[short_m] = -capped_scale(-d[short_m], side, rule.cap)
    return w


# ----------------------------------------------------------------------------- simulation
def simulate(df: pd.DataFrame, rule: Rule, decision_seconds: list[int], ret_col: str = "target",
             ret_scale: float = 1e-4, signal_col: str = "pred", cost_col: str | None = None) -> pd.DataFrame:
    """Per-interval results. df needs date_id, seconds_in_bucket, stock_id, signal_col, ret_col."""
    d = df[df["seconds_in_bucket"].isin(decision_seconds)]
    d = d[np.isfinite(d[signal_col])]
    rows = []
    prev_w: dict[int, pd.Series] = {}
    for (date, sec), g in d.groupby(["date_id", "seconds_in_bucket"], sort=True):
        w = portfolio_weights(g[signal_col].to_numpy(), rule)
        r = g[ret_col].to_numpy(float)
        valid = np.isfinite(r)
        # a held name whose return is unobservable contributes 0 and is reported
        pnl = float(np.sum(w[valid] * r[valid]) * ret_scale)
        ws = pd.Series(w, index=g["stock_id"].to_numpy())
        pw = prev_w.get(date)
        netted = float((ws.sub(pw, fill_value=0.0)).abs().sum()) if pw is not None else float(np.abs(w).sum())
        prev_w[date] = ws
        if cost_col is not None:
            # per-name round-trip cost in bps (e.g. half quoted spread at entry + at exit); a missing value
            # falls back to the cross-sectional median of that interval
            cb = g[cost_col].to_numpy(float)
            cb = np.where(np.isfinite(cb), cb, np.nanmedian(cb) if np.isfinite(cb).any() else 0.0)
            name_cost = float(np.sum(np.abs(w) * cb) * 1e-4)
        else:
            name_cost = 0.0
        rows.append({"name_cost": name_cost, "date_id": date, "seconds_in_bucket": sec, "ret": pnl, "gross": float(np.abs(w).sum()),
                     "net": float(w.sum()), "n_long": int((w > 0).sum()), "n_short": int((w < 0).sum()),
                     "max_abs_w": float(np.abs(w).max()) if len(w) else 0.0,
                     "missing_ret_weight": float(np.abs(w[~valid]).sum()), "netted_turnover": netted})
    out = pd.DataFrame(rows)
    if len(out):
        # final close of the day's last position for the netted variant
        last = out.groupby("date_id")["seconds_in_bucket"].transform("max") == out["seconds_in_bucket"]
        out.loc[last, "netted_turnover"] += out.loc[last, "gross"]
    return out


def daily_returns(intervals: pd.DataFrame, cost_bps_side: float, all_dates: list[int] | None = None,
                  netted: bool = False) -> pd.Series:
    traded = intervals["netted_turnover"] if netted else 2.0 * intervals["gross"]
    net = intervals["ret"] - cost_bps_side * 1e-4 * traded
    daily = net.groupby(intervals["date_id"]).sum()
    if all_dates is not None:
        daily = daily.reindex(all_dates, fill_value=0.0)  # a day without positions earns 0
    return daily


def sharpe(daily: pd.Series) -> float:
    sd = daily.std(ddof=1)
    return float(np.sqrt(252) * daily.mean() / sd) if sd > 0 and len(daily) > 1 else float("nan")


def max_drawdown(daily: pd.Series) -> float:
    cum = daily.cumsum()
    return float((cum - cum.cummax()).min()) if len(cum) else 0.0


def block_bootstrap(daily: pd.Series, reps: int, block: int, seed: int) -> dict:
    """Moving-block bootstrap over days (captures short serial dependence); percentile CIs."""
    x = daily.to_numpy(float)
    n = len(x)
    if n < 5:
        return {}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means, sharpes = np.empty(reps), np.empty(reps)
    for b in range(reps):
        starts = rng.integers(0, n - block + 1, nb)
        s = np.concatenate([x[i:i + block] for i in starts])[:n]
        means[b] = s.mean()
        sd = s.std(ddof=1)
        sharpes[b] = np.sqrt(252) * s.mean() / sd if sd > 0 else np.nan
    return {"reps": reps, "block_days": block,
            "mean_daily_ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "sharpe_ci95": [float(np.nanquantile(sharpes, 0.025)), float(np.nanquantile(sharpes, 0.975))],
            "p_sharpe_le_0": float(np.mean(sharpes <= 0))}


def summarize(intervals: pd.DataFrame, cost_grid: list[float], all_dates: list[int], cfg_bt: dict, seed: int) -> dict:
    res = {"n_days": len(all_dates), "n_intervals": int(len(intervals)),
           "n_active_intervals": int((intervals["gross"] > 0).sum()) if len(intervals) else 0,
           "mean_gross": float(intervals["gross"].mean()) if len(intervals) else 0.0,
           "max_abs_net": float(intervals["net"].abs().max()) if len(intervals) else 0.0,
           "max_abs_weight": float(intervals["max_abs_w"].max()) if len(intervals) else 0.0,
           "mean_names_long": float(intervals["n_long"].mean()) if len(intervals) else 0.0,
           "mean_names_short": float(intervals["n_short"].mean()) if len(intervals) else 0.0,
           "daily_turnover_open_close": float((2 * intervals["gross"]).groupby(intervals["date_id"]).sum().mean()) if len(intervals) else 0.0,
           "daily_turnover_netted": float(intervals["netted_turnover"].groupby(intervals["date_id"]).sum().mean()) if len(intervals) else 0.0,
           "by_cost": []}
    for c in cost_grid:
        for netted in (False, True):
            if netted and c == 0:
                continue
            d = daily_returns(intervals, c, all_dates, netted=netted)
            row = {"cost_bps_per_side": c, "turnover": "netted" if netted else "open_close",
                   "mean_daily_bps": float(d.mean() * 1e4), "std_daily_bps": float(d.std(ddof=1) * 1e4),
                   "sharpe": sharpe(d), "cum_return_bps": float(d.sum() * 1e4), "max_drawdown_bps": max_drawdown(d) * 1e4,
                   "hit_rate_days": float((d > 0).mean())}
            if not netted and c in (0.0, cfg_bt.get("selection_cost_bps", 0.5)):
                row["bootstrap"] = block_bootstrap(d, cfg_bt["bootstrap_reps"], cfg_bt["bootstrap_block_days"], seed)
            res["by_cost"].append(row)
    return res


def select_rule(oof: pd.DataFrame, cfg: Config) -> dict:
    """Grid over (cap, quantile) on DEVELOPMENT OOF predictions only; maximise Sharpe at selection cost."""
    bt = cfg["backtest"]
    dates = sorted(oof["date_id"].unique().tolist())
    grid = []
    for cap in bt["weight_caps"]:
        for q in bt["selection_quantiles"]:
            rule = Rule(cap=cap, quantile=q, gross=bt["gross_exposure"], min_names=bt["min_names_per_side"])
            iv = simulate(oof, rule, bt["decision_seconds"])
            d0 = daily_returns(iv, 0.0, dates)
            dc = daily_returns(iv, bt["selection_cost_bps"], dates)
            grid.append({**rule.to_dict(), "sharpe_gross": sharpe(d0), "sharpe_at_selection_cost": sharpe(dc),
                         "mean_daily_bps_gross": float(d0.mean() * 1e4), "mean_daily_bps_at_cost": float(dc.mean() * 1e4)})
    best = max(grid, key=lambda g: (np.nan_to_num(g["sharpe_at_selection_cost"], nan=-1e9)))
    return {"criterion": f"max dev Sharpe at {bt['selection_cost_bps']} bps/side (open+close costs)",
            "grid": grid, "selected": {k: best[k] for k in ("cap", "quantile", "gross", "min_names")}}


def wap_forward_returns(raw: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """Observed normalised WAP(t+h)/WAP(t) - 1 where both timestamps exist (no extrapolation past 540 s)."""
    a = raw[["stock_id", "date_id", "seconds_in_bucket", "wap"]]
    b = a.assign(seconds_in_bucket=a["seconds_in_bucket"] - horizon).rename(columns={"wap": "wap_fwd"})
    m = a.merge(b, on=["stock_id", "date_id", "seconds_in_bucket"], how="left")
    m["wap_ret"] = m["wap_fwd"] / m["wap"] - 1.0
    return m[["stock_id", "date_id", "seconds_in_bucket", "wap_ret"]]


def half_spread_round_trip(spreads: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """Round-trip quoted-spread cost per name in bps: spread_bps(t)/2 + spread_bps(t+60)/2.

    The exit spread is unobserved after 540 s; the entry half-spread is reused there (stated assumption).
    Spreads are normalised top-of-book quotes: this approximates crossing the displayed spread, not
    auction-uncross execution, and ignores size beyond the top of book.
    """
    a = spreads[["stock_id", "date_id", "seconds_in_bucket", "spread_bps"]]
    b = a.assign(seconds_in_bucket=a["seconds_in_bucket"] - horizon).rename(columns={"spread_bps": "spread_exit"})
    m = a.merge(b, on=["stock_id", "date_id", "seconds_in_bucket"], how="left")
    m["cost_rt_bps"] = m["spread_bps"] / 2 + m["spread_exit"].fillna(m["spread_bps"]) / 2
    return m[["stock_id", "date_id", "seconds_in_bucket", "cost_rt_bps"]]


def spread_cost_summary(preds: pd.DataFrame, spreads: pd.DataFrame, rule: Rule, decision_seconds: list[int],
                        all_dates: list[int], cfg_bt: dict, seed: int) -> tuple[dict, pd.Series]:
    m = preds.merge(half_spread_round_trip(spreads), on=["stock_id", "date_id", "seconds_in_bucket"], how="left")
    iv = simulate(m, rule, decision_seconds, cost_col="cost_rt_bps")
    net = (iv["ret"] - iv["name_cost"]).groupby(iv["date_id"]).sum().reindex(all_dates, fill_value=0.0)
    g = iv["gross"].sum()
    wavg = float(iv["name_cost"].sum() / g * 1e4) if g > 0 else float("nan")
    return {"assumption": "pay half the quoted spread of each traded name at entry and at exit (replaces the flat grid)",
            "avg_round_trip_cost_bps_per_unit_gross": wavg,
            "mean_daily_bps": float(net.mean() * 1e4), "std_daily_bps": float(net.std(ddof=1) * 1e4),
            "sharpe": sharpe(net), "cum_return_bps": float(net.sum() * 1e4), "max_drawdown_bps": max_drawdown(net) * 1e4,
            "hit_rate_days": float((net > 0).mean()),
            "bootstrap": block_bootstrap(net, cfg_bt["bootstrap_reps"], cfg_bt["bootstrap_block_days"], seed)}, net


def run_backtest(preds: pd.DataFrame, cfg: Config, rule: Rule, label: str, wap: pd.DataFrame | None = None,
                 spreads: pd.DataFrame | None = None) -> dict:
    bt = cfg["backtest"]
    dates = sorted(preds["date_id"].unique().tolist())
    iv = simulate(preds, rule, bt["decision_seconds"])
    out = {"label": label, "rule": rule.to_dict(), "decision_seconds": bt["decision_seconds"],
           "dates": [dates[0], dates[-1]] if dates else [],
           "target_proxy": summarize(iv, bt["cost_bps_per_side"], dates, bt, cfg["seed"])}
    daily = {"target_proxy_gross": daily_returns(iv, 0.0, dates)}
    for c in bt["cost_bps_per_side"]:
        daily[f"target_proxy_cost{c:g}"] = daily_returns(iv, c, dates)
    if spreads is not None:
        out["target_proxy"]["quoted_spread_cost"], daily["target_proxy_half_spread"] = spread_cost_summary(
            preds, spreads, rule, bt["decision_seconds"], dates, bt, cfg["seed"])
    if wap is not None:
        secs = [s for s in bt["decision_seconds"] if s + 60 <= 540]
        m = preds.merge(wap, on=["stock_id", "date_id", "seconds_in_bucket"], how="left")
        iv_w = simulate(m, rule, secs, ret_col="wap_ret", ret_scale=1.0)
        iv_t = simulate(m, rule, secs)  # same restricted schedule with the target, for a like-for-like comparison
        out["wap_robustness"] = {"decision_seconds": secs,
                                 "wap_proxy": summarize(iv_w, bt["cost_bps_per_side"], dates, bt, cfg["seed"]),
                                 "target_proxy_same_schedule": summarize(iv_t, [0.0], dates, bt, cfg["seed"])}
        daily["wap_proxy_gross"] = daily_returns(iv_w, 0.0, dates)
    return out, iv, pd.DataFrame(daily)


def run_backtest_reports(cfg: Config) -> None:
    from .reporting import backtest_outputs

    backtest_outputs(cfg)
