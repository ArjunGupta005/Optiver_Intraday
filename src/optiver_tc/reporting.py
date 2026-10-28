"""Generated tables, plots and markdown reports. Every number comes from saved artifacts."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics as M  # noqa: E402
from .backtest import Rule, run_backtest, wap_forward_returns  # noqa: E402
from .config import Config, read_json, write_json  # noqa: E402
from .experiments import load_records  # noqa: E402
from .featurestore import load_frame  # noqa: E402

log = logging.getLogger(__name__)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "lines.linewidth": 1.6, "legend.frameon": False,
})


def _fig_dir(cfg: Config) -> Path:
    d = cfg.reports_dir / "figures" / cfg["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- experiments
def experiment_table(cfg: Config) -> pd.DataFrame:
    rec = load_records(cfg).drop_duplicates(["experiment", "fold"], keep="last")
    zero = rec[rec["experiment"] == "zero"].set_index("fold")["mae"]
    piv = rec.pivot(index="experiment", columns="fold", values="mae")
    folds = list(piv.columns)
    t = piv.copy()
    t["mean_mae"] = piv.mean(1)
    t["std_mae"] = piv.std(1)
    t["impr_vs_zero_pct"] = 100 * (1 - piv.div(zero[folds], axis=1)).mean(1)
    meta = rec.groupby("experiment").agg(kind=("kind", "first"), n_features=("n_features", "first"),
                                         date_fraction=("date_fraction", "first"),
                                         mean_best_iter=("best_iteration", "mean"), fit_s_total=("fit_seconds", "sum"),
                                         spearman_ic=("spearman_ic_mean", "mean"), dir_acc=("dir_acc", "mean"),
                                         rmse=("rmse", "mean"))
    t = t.join(meta).reset_index().sort_values("mean_mae")
    t.to_csv(cfg.artifacts_dir / "experiment_table.csv", index=False)
    return t


def plot_folds(cfg: Config, t: pd.DataFrame) -> None:
    show = [e for e in ["zero", "median", t[t["experiment"].str.startswith("ridge")]["experiment"].iloc[0] if (t["experiment"].str.startswith("ridge")).any() else None,
                        "abl_all", "lgb_final_dev"] if e and e in set(t["experiment"])]
    folds = [c for c in t.columns if str(c).startswith("fold")]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cols = ["#9a9893", "#c3c2b7", C2, C3, C1]
    for i, e in enumerate(show):
        r = t.set_index("experiment").loc[e, folds].astype(float)
        ax.plot(folds, r.values, marker="o", markersize=5, color=cols[i % len(cols)], label=e)
    ax.set_ylabel("validation MAE (bps)")
    ax.set_title("Development-fold MAE by model")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, _fig_dir(cfg) / "fold_mae.png")


def plot_importance(cfg: Config, exp: str = "lgb_final_dev", top: int = 25) -> pd.DataFrame | None:
    files = sorted((cfg.artifacts_dir / "importance").glob(f"{exp}__fold*.csv"))
    if not files:
        return None
    imp = pd.concat([pd.read_csv(f) for f in files]).groupby("feature")["gain"].mean().sort_values(ascending=False)
    imp = (imp / imp.sum()).rename("gain_share").reset_index()
    imp.to_csv(cfg.artifacts_dir / f"importance_{exp}_mean.csv", index=False)
    s = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.24 * len(s) + 1))
    ax.barh(s["feature"], s["gain_share"], color=C1, height=0.7)
    ax.set_xlabel("share of total gain (mean over dev folds)")
    ax.set_title(f"LightGBM feature importance ({exp}, top {top})")
    ax.grid(axis="y", visible=False)
    _save(fig, _fig_dir(cfg) / "feature_importance.png")
    return imp


def prediction_diagnostics(cfg: Config, preds: pd.DataFrame, tag: str) -> dict:
    d = preds[np.isfinite(preds["target"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    # calibration: binned mean target vs mean prediction
    d["bin"] = pd.qcut(d["pred"], 20, duplicates="drop")
    cal = d.groupby("bin", observed=True).agg(p=("pred", "mean"), y=("target", "mean"))
    lim = float(np.abs(cal.values).max()) * 1.1
    axes[0].plot([-lim, lim], [-lim, lim], color=GRID, linewidth=1)
    axes[0].plot(cal["p"], cal["y"], marker="o", markersize=4, color=C1)
    axes[0].set_xlabel("mean prediction (bps), 20 quantile bins")
    axes[0].set_ylabel("mean realised target (bps)")
    axes[0].set_title("Calibration")
    sb = M.mae_by(d, "seconds_in_bucket")
    axes[1].plot(sb["seconds_in_bucket"], sb["mae_zero"], color="#9a9893", label="zero")
    axes[1].plot(sb["seconds_in_bucket"], sb["mae"], color=C1, label="model")
    axes[1].set_xlabel("seconds_in_bucket")
    axes[1].set_ylabel("MAE (bps)")
    axes[1].set_title("MAE by auction stage")
    axes[1].legend(fontsize=8)
    ic = M.ic_fast(d)
    daily_ic = ic.groupby("date_id")["spearman_ic"].mean()
    axes[2].bar(daily_ic.index, daily_ic.values, color=C3, width=0.8)
    axes[2].axhline(0, color=INK2, linewidth=0.8)
    axes[2].set_xlabel("date_id")
    axes[2].set_ylabel("mean Spearman IC")
    axes[2].set_title("Daily mean cross-sectional IC")
    _save(fig, _fig_dir(cfg) / f"pred_diagnostics_{tag}.png")
    bd = M.mae_by(d, "date_id")
    return {"ic_positive_day_share": float((daily_ic > 0).mean()), "mae_by_bucket": sb.to_dict("records"),
            "worst_dates": bd.assign(gap=bd["mae"] - bd["mae_zero"]).sort_values("gap", ascending=False).head(5).to_dict("records")}


# ----------------------------------------------------------------------------- backtest
def backtest_outputs(cfg: Config) -> dict:
    fz = read_json(cfg.artifacts_dir / "frozen_choices.json")
    rule = Rule(**fz["backtest"]["selected"])
    out = {"rule_selection": {k: v for k, v in fz["backtest"].items()}, "results": {}}
    dev = pd.read_parquet(cfg.artifacts_dir / "oof" / "frozen_dev_oof.parquet")
    sp_dev = load_frame(cfg, int(dev["date_id"].min()), int(dev["date_id"].max()),
                        ["stock_id", "date_id", "seconds_in_bucket", "spread_bps"]).to_pandas()
    r_dev, iv_dev, daily_dev = run_backtest(dev, cfg, rule, "development OOF (rule chosen here: in-sample for the rule)", spreads=sp_dev)
    out["results"]["dev"] = r_dev
    hp = cfg.artifacts_dir / "predictions" / "holdout_predictions.parquet"
    daily_h = None
    if hp.exists():
        ho = pd.read_parquet(hp)
        hs = fz["holdout"]
        raw = load_frame(cfg, hs["val_start"], hs["val_end"], ["stock_id", "date_id", "seconds_in_bucket", "wap", "spread_bps"]).to_pandas()
        r_h, iv_h, daily_h = run_backtest(ho, cfg, rule, "final holdout (rule frozen)", wap=wap_forward_returns(raw), spreads=raw)
        out["results"]["holdout"] = r_h
        iv_h.to_csv(cfg.artifacts_dir / "backtest_intervals_holdout.csv", index=False)
        daily_h.to_csv(cfg.artifacts_dir / "backtest_daily_holdout.csv")
    daily_dev.to_csv(cfg.artifacts_dir / "backtest_daily_dev.csv")
    write_json(out, cfg.artifacts_dir / "backtest_results.json")
    _plot_backtest(cfg, daily_dev, daily_h, out)
    _write_backtest_md(cfg, out)
    return out


def _plot_backtest(cfg, daily_dev, daily_h, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for daily, lab, col in [(daily_dev, "dev OOF", C2), (daily_h, "holdout", C1)]:
        if daily is None:
            continue
        axes[0].plot(daily.index, daily["target_proxy_gross"].cumsum() * 1e4, color=col, label=f"{lab}, 0 bps")
        k = [c for c in daily.columns if c.startswith("target_proxy_cost1")]
        if k:
            axes[0].plot(daily.index, daily[k[0]].cumsum() * 1e4, color=col, linestyle="--", label=f"{lab}, 1 bp/side")
        if "target_proxy_half_spread" in daily.columns:
            axes[0].plot(daily.index, daily["target_proxy_half_spread"].cumsum() * 1e4, color=col, linestyle=":",
                         label=f"{lab}, own half-spread")
    axes[0].axhline(0, color=INK2, linewidth=0.8)
    axes[0].set_xlabel("date_id")
    axes[0].set_ylabel("cumulative daily proxy return (bps of capital)")
    axes[0].set_title("Target-based research proxy: cumulative return")
    axes[0].legend(fontsize=7)
    res = out["results"].get("holdout") or out["results"]["dev"]
    bc = pd.DataFrame(res["target_proxy"]["by_cost"])
    costs = bc[bc["turnover"] == "open_close"]["cost_bps_per_side"].tolist()
    for name, col, off in [("open_close", C1, -0.18), ("netted", C3, 0.18)]:
        s = bc[bc["turnover"] == name]
        x = np.array([costs.index(c) for c in s["cost_bps_per_side"]])  # position by cost value
        axes[1].bar(x + off, s["sharpe"], width=0.34, color=col, label=name)
    axes[1].set_xticks(range(len(costs)))
    axes[1].set_xticklabels([f"{c:g}" for c in costs])
    axes[1].axhline(0, color=INK2, linewidth=0.8)
    axes[1].set_xlabel("hypothetical cost, bps per side")
    axes[1].set_ylabel("annualised Sharpe (daily)")
    axes[1].set_title(f"Cost sensitivity ({'holdout' if 'holdout' in out['results'] else 'dev'})")
    axes[1].legend(fontsize=8, title="turnover", title_fontsize=8)
    _save(fig, _fig_dir(cfg) / "backtest.png")
    if daily_h is not None:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(daily_h.index, daily_h["target_proxy_gross"] * 1e4, color=np.where(daily_h["target_proxy_gross"] >= 0, C1, C2), width=0.8)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xlabel("date_id")
        ax.set_ylabel("daily proxy return (bps), gross")
        ax.set_title("Holdout daily returns (target-based proxy, before costs)")
        _save(fig, _fig_dir(cfg) / "holdout_daily_returns.png")


def _cost_table(summary: dict) -> pd.DataFrame:
    rows = []
    for r in summary["by_cost"]:
        b = r.get("bootstrap") or {}
        rows.append({"cost bps/side": r["cost_bps_per_side"], "turnover": r["turnover"], "mean daily bps": r["mean_daily_bps"],
                     "std daily bps": r["std_daily_bps"], "Sharpe": r["sharpe"], "cum bps": r["cum_return_bps"],
                     "max DD bps": r["max_drawdown_bps"], "up days": r["hit_rate_days"],
                     "Sharpe 95% CI": f"[{b['sharpe_ci95'][0]:.2f}, {b['sharpe_ci95'][1]:.2f}]" if b else ""})
    return pd.DataFrame(rows)


def _write_backtest_md(cfg: Config, out: dict) -> None:
    sel = out["rule_selection"]
    L = ["# Backtest report — target-based research proxy", "",
         "**This is a research proxy, not a demonstrated or executable strategy.** Returns are computed from the",
         "competition label (index-relative WAP move, bps) on out-of-sample predictions. WAP is not a fill price,",
         "the synthetic index weights are unknown, and the cost grid below is hypothetical, not measured.", "",
         "Conventions are documented in `src/optiver_tc/backtest.py`. Summary: decision times every 60 s",
         f"({cfg['backtest']['decision_seconds']}), non-overlapping 60 s holds; demeaned signal; long and short books each",
         "scaled to gross/2 under a per-stock cap (exactly net-zero); capital 1 per day, interval returns summed;",
         "costs charged on entry and exit (`open_close`), with a `netted` sensitivity that charges only weight changes;",
         "Sharpe = sqrt(252) x mean / std of DAILY returns.", "",
         "## Rule selection (development OOF only)", f"Criterion: {sel['criterion']}.", "",
         _md(pd.DataFrame(sel["grid"])), "", f"Selected and frozen: `{sel['selected']}`.", ""]
    for key, title in [("dev", "Development OOF (dates 301–420; the rule was chosen on these, so this is optimistic)"),
                       ("holdout", "Final holdout (dates 421–480; rule frozen)")]:
        if key not in out["results"]:
            continue
        r = out["results"][key]
        s = r["target_proxy"]
        L += [f"## {title}", "",
              f"Days {s['n_days']}, intervals {s['n_intervals']} (active {s['n_active_intervals']}), mean gross {s['mean_gross']:.3f},",
              f"max |net| {s['max_abs_net']:.2e}, max |w| {s['max_abs_weight']:.4f}, mean names long/short {s['mean_names_long']:.1f}/{s['mean_names_short']:.1f},",
              f"daily turnover {s['daily_turnover_open_close']:.1f}x capital (open+close) vs {s['daily_turnover_netted']:.2f}x if netted.", "",
              _md(_cost_table(s), "{:.3f}"), ""]
        if "quoted_spread_cost" in s:
            q = s["quoted_spread_cost"]
            b = q.get("bootstrap") or {}
            ci = f" (95% CI [{b['sharpe_ci95'][0]:.2f}, {b['sharpe_ci95'][1]:.2f}])" if b else ""
            L += ["**Quoted-spread cost assumption** (replaces the flat grid; not added to it): each traded name pays half its",
                  "own quoted spread at entry and at exit (exit spread = entry spread at 540 s, where t+60 is unobserved).",
                  f"Average round-trip cost {q['avg_round_trip_cost_bps_per_unit_gross']:.2f} bps per unit of gross traded "
                  f"(~{q['avg_round_trip_cost_bps_per_unit_gross'] / 2:.2f} bps per side): mean daily {q['mean_daily_bps']:.2f} bps, "
                  f"Sharpe {q['sharpe']:.2f}{ci}, up days {q['hit_rate_days']:.2f}, cumulative {q['cum_return_bps']:.0f} bps, "
                  f"max DD {q['max_drawdown_bps']:.0f} bps.",
                  "Limitation: top-of-book spreads in normalised price units; ignores depth, auction-uncross pricing and impact.", ""]
        if "wap_robustness" in r:
            w = r["wap_robustness"]
            t0 = next(x for x in w["target_proxy_same_schedule"]["by_cost"] if x["cost_bps_per_side"] == 0)
            L += ["### Robustness: observed WAP(t+60)/WAP(t) − 1, same rule", "",
                  f"Decision times restricted to {w['decision_seconds']} because WAP(t+60) is not observed after 540 s",
                  "(the target at 540 s exists but its WAP(600) is not in the data, and nothing is extrapolated).",
                  "This marks the self-financing long/short book at raw stock WAP (not index-relative); net-zero",
                  "dollar exposure removes the equal-weighted common move but not the unknown index-weighted one.", "",
                  f"Target proxy on the same restricted schedule, gross: Sharpe {t0['sharpe']:.3f}, mean {t0['mean_daily_bps']:.3f} bps/day.", "",
                  _md(_cost_table(w["wap_proxy"]), "{:.3f}"), ""]
    L += ["## Statistical power", "",
          "The moving-block bootstrap (5-day blocks, 2,000 resamples) resamples days; it assumes dependence dies out",
          "within ~5 days and that the sample period is representative. With 60 holdout days the Sharpe CI is wide",
          "(a Sharpe estimate's standard error over n days is roughly sqrt(252/n) ≈ 2 for n = 60).", "",
          "Figures: `reports/figures/full/backtest.png`, `reports/figures/full/holdout_daily_returns.png`."]
    (cfg.reports_dir / f"backtest_report{'' if cfg['name'] == 'full' else '_' + cfg['name']}.md").write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------- validation report
def build_reports(cfg: Config) -> None:
    t = experiment_table(cfg)
    plot_folds(cfg, t)
    imp = plot_importance(cfg)
    sfx = "" if cfg["name"] == "full" else "_" + cfg["name"]
    folds = [c for c in t.columns if str(c).startswith("fold")]
    show = t[["experiment", "kind", "n_features", "date_fraction", *folds, "mean_mae", "std_mae", "impr_vs_zero_pct",
              "spearman_ic", "dir_acc", "rmse", "mean_best_iter", "fit_s_total"]]
    (cfg.reports_dir / f"experiments{sfx}.md").write_text(
        "# Experiment table (development folds only)\n\nGenerated from `artifacts/" + cfg["name"] +
        "/experiments.jsonl`. MAE in bps; `impr_vs_zero_pct` = mean over folds of 1 - MAE/MAE_zero.\n"
        "`date_fraction` < 1 means a seeded random subset of training dates (ablation/config-search budget).\n\n" +
        _md(show) + "\n", encoding="utf-8")
    L = ["# Validation report", "", "Design: `reports/validation_design.md`. All numbers below are read from saved artifacts.", ""]
    sel_dir = cfg.artifacts_dir / "selection"
    if (sel_dir / "stage_b_families.json").exists():
        b = read_json(sel_dir / "stage_b_families.json")
        L += ["## Feature-family ablation (stage B)", f"Rule: {b['rule']}. MAE with all families: {b['mae_all']:.5f}.", "",
              _md(pd.DataFrame(b["table"]), "{:.5f}"), "", f"Selected families: `{b['selected_families']}`.", ""]
    if (sel_dir / "stage_c_config.json").exists():
        c = read_json(sel_dir / "stage_c_config.json")
        L += ["## LightGBM configuration search (stage C)", f"Rule: {c['rule']}.", "",
              _md(pd.DataFrame([{"config": r["config"], "mean_mae": r["mean_mae"], **r["by_fold"]} for r in c["results"]]), "{:.5f}"),
              "", f"Selected: `{c['selected']}`.", ""]
    fz_p = cfg.artifacts_dir / "frozen_choices.json"
    if fz_p.exists():
        fz = read_json(fz_p)
        bl = pd.DataFrame(fz["blend"]["grid"])
        pp = pd.DataFrame([{k: v for k, v in g.items() if k != "by_fold"} for g in fz["postprocess"]["grid"]])
        L += ["## Frozen choices (before holdout)", f"Frozen at {fz['frozen_at']}, sha256 `{fz['sha256'][:16]}…`.", "",
              f"* families: {fz['families']} ({len(fz['feature_columns'])} columns)",
              f"* LightGBM params: `{fz['lgbm_params']}`; rounds = {fz['lgbm_num_boost_round']} (mean of dev best iterations {fz['lgbm_dev_best_iterations']})",
              f"* ridge partner: {fz['ridge']}; blend weight on LightGBM: {fz['blend']['w_lgb']}",
              f"* post-processing: demean={fz['postprocess']['demean']}, clip={fz['postprocess']['clip']} ({fz['postprocess']['rule']})",
              f"* backtest rule: {fz['backtest']['selected']}", "",
              "Blend grid (pooled dev OOF MAE, selected rows):", "",
              _md(bl.iloc[::4][["w_lgb", "mae"]], "{:.5f}"), "", "Post-processing grid:", "", _md(pp, "{:.5f}"), ""]
        dev = pd.read_parquet(cfg.artifacts_dir / "oof" / "frozen_dev_oof.parquet")
        dsum = M.summary(dev)
        diag = prediction_diagnostics(cfg, dev, "dev")
        L += ["## Development OOF (frozen pipeline, dates 301–420)", "", "```json", json.dumps(dsum, indent=1), "```",
              f"Share of days with positive mean IC: {diag['ic_positive_day_share']:.3f}.", ""]
    hm = cfg.artifacts_dir / "holdout_metrics.json"
    if hm.exists():
        h = read_json(hm)
        ho = pd.read_parquet(cfg.artifacts_dir / "predictions" / "holdout_predictions.parquet")
        diag = prediction_diagnostics(cfg, ho, "holdout")
        rows = [{"model": "zero", "mae": h["zero"]["mae"], "rmse": h["zero"]["rmse"]},
                {"model": f"train median ({h['train_median']['value']:.4f})", "mae": h["train_median"]["mae"]}]
        if "ridge_only" in h:
            rows.append({"model": "ridge only", **{k: h["ridge_only"][k] for k in ("mae", "rmse", "dir_acc", "spearman_ic_mean")}})
        rows.append({"model": "LightGBM only (raw)", **{k: h["lgbm_only_raw"][k] for k in ("mae", "rmse", "dir_acc", "spearman_ic_mean")}})
        rows.append({"model": "FINAL (frozen blend + post-processing)", **{k: h["final_model"][k] for k in ("mae", "rmse", "dir_acc", "spearman_ic_mean")}})
        sb = pd.DataFrame(diag["mae_by_bucket"])
        stage = sb.assign(stage=pd.cut(sb["seconds_in_bucket"], [-1, 290, 540], labels=["0-290 s (no far/near)", "300-540 s"]))
        st = stage.groupby("stage", observed=True).apply(lambda g: pd.Series({"mae": np.average(g["mae"], weights=g["n"]), "mae_zero": np.average(g["mae_zero"], weights=g["n"])}), include_groups=False).reset_index()
        L += ["## Final holdout (dates 421–480, evaluated once after freezing)", "",
              f"Evaluated at {h['evaluated_at']} with frozen sha256 `{h['frozen_sha256'][:16]}…`. Rows scored: {h['final_model']['n_rows_scored']:,}.", "",
              _md(pd.DataFrame(rows), "{:.5f}"), "",
              f"Improvement vs zero: {h['improvement_vs_zero_pct']:.2f}%. Mean per-timestamp Spearman IC {h['final_model']['spearman_ic_mean']:.4f} "
              f"(t-stat over {h['final_model']['n_ic_groups']} timestamps {h['final_model']['spearman_ic_tstat']:.1f}; timestamps are not independent, so treat the t-stat as descriptive).",
              f"Share of holdout days with positive mean IC: {diag['ic_positive_day_share']:.3f}.", "",
              "By auction stage:", "", _md(st, "{:.4f}"), "",
              "Slices: `artifacts/full/predictions/holdout_mae_by_{date_id,stock_id,seconds_in_bucket}.csv`.", "",
              "Context only: the Kaggle leaderboard (a different, later test period) is not comparable with these local numbers.", ""]
    L += ["## Figures", "* `reports/figures/" + cfg["name"] + "/fold_mae.png`", "* `reports/figures/" + cfg["name"] + "/feature_importance.png`",
          "* `reports/figures/" + cfg["name"] + "/pred_diagnostics_dev.png`, `pred_diagnostics_holdout.png`"]
    (cfg.reports_dir / f"validation_report{sfx}.md").write_text("\n".join(L), encoding="utf-8")
    log.info("reports written")
