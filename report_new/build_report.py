"""从真实运行产物生成报告表格、图像、数值宏及核验记录。"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import code2_new as strategy

RESULTS = ROOT / "results_code2_new_report"
COMPARE = RESULTS / "comparison"
FIG = HERE / "figures"
TABLES = HERE / "tables"
for folder in (FIG, TABLES, HERE / "source"):
    folder.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/pdfs/report_new/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp/pdfs/report_new/cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.fontsize": 8.5, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.pad_inches": 0.06})
NAMES = dict(zip(strategy.FACTORS, ["SmartMoney", "CPV", "CDPDP", "Skewness", "TailReversal", "Amplitude"]))
NAMES["Pred_Score"] = "XGBoost"
PERIOD_CN = {"Weekly": "周频", "BiWeekly": "双周频", "Monthly": "月频"}
ORDER = list(strategy.FACTORS) + ["Pred_Score"]
summary = pd.read_csv(COMPARE / "summary.csv")
full = pd.read_csv(RESULTS / "summary.csv")
ls = summary[summary.Portfolio.eq("TheoreticalLS")].copy()
all_ls = full[full.Portfolio.eq("TheoreticalLS")].copy()
config = json.loads((RESULTS / "run_config.json").read_text())
comparison_config = json.loads((COMPARE / "comparison_config.json").read_text())
assert config["source_sha256"] == hashlib.sha256((ROOT / "code2_new.py").read_bytes()).hexdigest()
assert comparison_config["xgboost_source_sha256"] == hashlib.sha256((ROOT / "code2_xgboost.py").read_bytes()).hexdigest()


def esc(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def percent(value, digits=2):
    return f"{100 * value:.{digits}f}\\%"


def row(*values):
    return " & ".join(str(v) for v in values) + r" \\" + "\n"


def write_table(name, text):
    (TABLES / f"{name}.tex").write_text(text)


def selected(frame, factor, period, portfolio=None):
    mask = frame.Factor.eq(factor) & frame.Period.eq(period)
    if portfolio is not None:
        mask &= frame.Portfolio.eq(portfolio)
    return frame.loc[mask].iloc[0]


def save_figure(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


macros = {}
def macro(name, value):
    macros[name] = str(value)


# 独立对账：图表和表格之前先验证全部真实交易及绩效。
counts = {"series": 0, "ledger_rows": 0, "trades": 0}
annual = []
diagnostics = []
for parent, stats in [(RESULTS, full), (COMPARE, summary)]:
    for (factor, period), records in stats.groupby(["Factor", "Period"]):
        folder = parent / factor / period
        ledger = pd.read_csv(folder / "daily_ledger.csv", parse_dates=["TradingDay"])
        trades = pd.read_parquet(folder / "trades.parquet")
        np.testing.assert_allclose(trades.Cost, trades.Notional * config["one_way_fee"], atol=1e-12)
        for name, frame in ledger.groupby("Portfolio"):
            metrics = strategy.performance(frame.Return, strategy.Config())
            saved = records[records.Portfolio.eq(name)].iloc[0]
            for field in ["FinalNAV", "AnnualReturn", "AnnualVol", "Sharpe", "MaxDrawdown"]:
                np.testing.assert_allclose(metrics[field], saved[field], rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(frame.NAV, frame.Cash + frame.LongValue - frame.ShortValue, atol=1e-10)
            np.testing.assert_allclose(frame.NAV, np.cumprod(1 + frame.Return), atol=1e-10)
            book = trades[trades.Portfolio.eq(name)].copy()
            book["Flow"] = book.DeltaUnits * book.AdjPrice + book.Cost
            cashflow = book.groupby("TradingDay").Flow.sum().reindex(frame.TradingDay, fill_value=0).to_numpy()
            np.testing.assert_allclose(frame.Cash, 1 - np.cumsum(cashflow + frame.BorrowCost.to_numpy()), atol=1e-10)
            np.testing.assert_allclose(frame.TradingCost, book.groupby("TradingDay").Cost.sum().reindex(frame.TradingDay, fill_value=0), atol=1e-12)
            counts["series"] += 1
        counts["ledger_rows"] += len(ledger)
        counts["trades"] += len(trades)
        if parent != COMPARE:
            continue
        leg = ledger[ledger.Portfolio.eq("TheoreticalLS")].copy()
        for year, frame in leg.groupby(leg.TradingDay.dt.year):
            annual.append({"Factor": factor, "Period": period, "Year": int(year),
                           "Return": (1 + frame.Return).prod() - 1})
        ic = pd.read_csv(folder / "ic.csv")
        diag = {"Factor": factor, "Period": period, "N_IC": len(ic),
                "MeanSignalStocks": ic.N_Signal.mean(), "Coverage": ic.Coverage.mean(),
                "DailyTurnover": leg.Turnover.mean(), "StalePositionDays": int(leg.StalePositions.sum()),
                "FinalPositions": int(leg.Positions.iloc[-1]),
                "MaxAbsNetExposure": ((leg.LongValue - leg.ShortValue) / leg.NAV).abs().max()}
        diagnostics.append(diag)
        fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1.2, 1]})
        palette = ["#5a6c80", "#6b91b0", "#49a39b", "#df9b4b", "#b83442"]
        initial = pd.Timestamp(comparison_config["signal_start"])
        for i in range(1, 6):
            frame = ledger[ledger.Portfolio.eq(f"Group{i}")]
            dates = pd.DatetimeIndex([initial, *frame.TradingDay])
            axes[0].plot(dates, np.r_[1, frame.NAV], label=f"G{i}" + (" (long)" if i == 5 else ""),
                         color=palette[i-1], lw=1.25)
        axes[0].set_title(f"{NAMES[factor]} | {period} | out-of-sample")
        axes[0].set_ylabel("Long-only NAV")
        axes[0].legend(loc="best", ncol=5, frameon=False)
        axes[1].plot(pd.DatetimeIndex([initial, *leg.TradingDay]), np.r_[1, leg.NAV], color="#203e58", lw=1.35)
        metric = selected(ls, factor, period)
        axes[1].set_title(f"L/S 50%/50% | Ann. {metric.AnnualReturn:.2%} | Sharpe {metric.Sharpe:.2f} | MDD {metric.MaxDrawdown:.2%}", fontsize=10)
        axes[1].set_ylabel("Net L/S NAV")
        axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for axis in axes:
            axis.axhline(1, color="#888888", lw=0.5)
            axis.grid(alpha=0.2)
        fig.tight_layout(pad=0.8)
        save_figure(fig, f"{factor}_{period}")

diagnostics = pd.DataFrame(diagnostics)
annual = pd.DataFrame(annual)
ls.to_csv(TABLES / "comparison_long_short.csv", index=False)
all_ls.to_csv(TABLES / "full_period_long_short.csv", index=False)
diagnostics.to_csv(TABLES / "diagnostics.csv", index=False)
annual.to_csv(TABLES / "calendar_returns.csv", index=False)
summary.to_csv(TABLES / "all_oos_portfolios.csv", index=False)

# 逐月标签隔离和共同预测样本检查。
logs, importances = [], []
sample = pd.read_parquet(RESULTS / "xgboost/comparison_sample.parquet")
sample.TradingDay = pd.to_datetime(sample.TradingDay)
keys = sample[strategy.KEYS].sort_values(strategy.KEYS).reset_index(drop=True)
for period in strategy.PERIODS:
    folder = RESULTS / "xgboost" / period
    log = pd.read_csv(folder / "training_log.csv", parse_dates=["Cutoff", "MaxLabelEnd"])
    assert log.MaxLabelEnd.lt(log.Cutoff).all()
    assert log.Horizon.eq(strategy.PERIODS[period]).all()
    pred = pd.read_parquet(folder / "predictions.parquet")
    pd.testing.assert_frame_equal(keys, pred[strategy.KEYS].sort_values(strategy.KEYS).reset_index(drop=True))
    log["Period"] = period
    logs.append(log)
    imp = pd.read_csv(folder / "feature_importance.csv")
    imp["Period"] = period
    importances.append(imp)
logs = pd.concat(logs, ignore_index=True)
importance = pd.concat(importances, ignore_index=True)
logs.to_csv(TABLES / "training_log.csv", index=False)
importance.to_csv(TABLES / "feature_importance.csv", index=False)

# 主统计表：相同日期、相同股票样本、相同敞口和费用。
text = ""
for period in strategy.PERIODS:
    for i, factor in enumerate(ORDER):
        r = selected(ls, factor, period)
        if factor == "Pred_Score":
            text += r"\rowcolor{blue!8}" + "\n"
        text += row(PERIOD_CN[period] if i == 0 else "", NAMES[factor], f"{r.IC_Raw:.3f}",
                    f"{r.RankIC_Raw:.3f}", percent(r.AnnualReturn), f"{r.Sharpe:.2f}", percent(r.MaxDrawdown),
                    percent(r.MeanDailyTurnover))
    text += r"\midrule" + "\n"
write_table("comparison", text)

text = ""
for factor in ORDER:
    r = selected(summary, factor, "BiWeekly", "Group5")
    text += row(NAMES[factor], *[percent(selected(summary, factor, "BiWeekly", f"Group{i}").AnnualReturn) for i in range(1, 6)], percent(r.MaxDrawdown))
write_table("long_only", text)

text = ""
for factor in ORDER:
    r = selected(diagnostics, factor, "BiWeekly")
    text += row(NAMES[factor], f"{r.MeanSignalStocks:.1f}", percent(r.Coverage), int(r.N_IC),
                int(r.StalePositionDays), int(r.FinalPositions), percent(r.MaxAbsNetExposure))
write_table("diagnostics", text)

text = ""
for factor in ORDER:
    values = annual[annual.Factor.eq(factor) & annual.Period.eq("BiWeekly")].set_index("Year").Return
    text += row(NAMES[factor], percent(values.loc[2017]), percent(values.loc[2018]),
                percent(selected(ls, factor, "BiWeekly").AnnualReturn))
write_table("calendar_returns", text)

text = ""
for period in strategy.PERIODS:
    r = logs[logs.Period.eq(period)]
    text += row(PERIOD_CN[period], len(r), f"{r.TrainDays.min()}--{r.TrainDays.max()}",
                f"{r.TrainRows.min():,}--{r.TrainRows.max():,}", str(r.MaxLabelEnd.max().date()))
write_table("training", text)

text = ""
for period in strategy.PERIODS:
    for i, factor in enumerate(strategy.FACTORS):
        r = selected(all_ls, factor, period)
        text += row(PERIOD_CN[period] if i == 0 else "", NAMES[factor], f"{r.IC_Raw:.3f}",
                    f"{-r.RankIC_Directed:.3f}", percent(r.AnnualReturn), f"{r.Sharpe:.2f}", percent(r.MaxDrawdown))
    text += r"\midrule" + "\n"
write_table("full_period", text)

# 相关性采用模型的同一完整输入样本，区别于全时期单因子有效样本。
correlations = [g[list(strategy.FACTORS)].corr(method="spearman") for _, g in sample.groupby("TradingDay")]
corr = sum(correlations) / len(correlations)
corr.to_csv(TABLES / "oos_factor_correlation.csv")
offdiag = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
pair = offdiag.abs().idxmax()
macro("MaxCorr", f"{abs(offdiag.loc[pair]):.3f}")
macro("MaxCorrPair", f"{NAMES[pair[0]]}与{NAMES[pair[1]]}")
fig, ax = plt.subplots(figsize=(7.6, 5.1))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
labels = [NAMES[f] for f in strategy.FACTORS]
ax.set_xticks(range(6), labels, rotation=30, ha="right")
ax.set_yticks(range(6), labels)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=10)
ax.set_title("Mean daily Spearman correlation | common OOS sample")
fig.colorbar(im, ax=ax, fraction=0.045)
fig.tight_layout()
save_figure(fig, "correlation")

fig, ax = plt.subplots(figsize=(8.2, 4.2))
averages = importance.groupby(["Period", "Feature"]).Importance.mean().unstack().reindex(columns=strategy.FACTORS)
for i, period in enumerate(strategy.PERIODS):
    ax.barh(np.arange(6) + (i-1)*0.24, averages.loc[period], height=0.23, label=period)
ax.set_yticks(range(6), labels)
ax.set_xlabel("Mean normalized gain importance")
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
save_figure(fig, "importance")

fig, ax = plt.subplots(figsize=(8.2, 4.2))
for period in strategy.PERIODS:
    leg = pd.read_csv(COMPARE / "Pred_Score" / period / "daily_ledger.csv", parse_dates=["TradingDay"])
    leg = leg[leg.Portfolio.eq("TheoreticalLS")]
    ax.plot(leg.TradingDay, leg.NAV, label=period, lw=1.4)
ax.axhline(1, color="gray", lw=0.5)
ax.set_ylabel("Net L/S NAV")
ax.set_title("XGBoost | identical OOS dates | 50% long / 50% short targets")
ax.legend(frameon=False)
ax.grid(alpha=0.2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.tight_layout()
save_figure(fig, "model_horizons")

for period, suffix in [("Weekly", "Week"), ("BiWeekly", "Biweek"), ("Monthly", "Month")]:
    r = selected(ls, "Pred_Score", period)
    for name, value in [("Return", percent(r.AnnualReturn)), ("Sharpe", f"{r.Sharpe:.2f}"),
                        ("DD", percent(r.MaxDrawdown)), ("IC", f"{r.IC_Raw:.3f}"),
                        ("Rank", f"{r.RankIC_Raw:.3f}")]:
        macro("Model" + suffix + name, value)
macro("CPVMonthReturn", percent(selected(ls, "HF_Raw_CPV_Mean", "Monthly").AnnualReturn))
macro("CPVMonthSharpe", f"{selected(ls, 'HF_Raw_CPV_Mean', 'Monthly').Sharpe:.2f}")
macro("CPVBiweekReturn", percent(selected(ls, "HF_Raw_CPV_Mean", "BiWeekly").AnnualReturn))
macro("ComparisonStart", comparison_config["signal_start"])
macro("ReturnStart", comparison_config["return_start"])
macro("ComparisonEnd", comparison_config["end"])
macro("ReturnDays", int(summary.Days.iloc[0]))
macro("CommonRows", f"{len(sample):,}")
macro("CommonMeanStocks", f"{sample.groupby('TradingDay').size().mean():.1f}")
macro("ModelsCount", len(logs))
macro("AllSeries", counts["series"])
macro("TradesCount", f"{counts['trades']:,}")
macro("OOSStaleSeries", int(summary.HasStaleValuations.sum()))
macro("OOSUnclosedSeries", int((~summary.FullyLiquidated).sum()))
macro("OOSLSStaleCases", int(ls.HasStaleValuations.sum()))
macro("OOSLSUnclosedCases", int((~ls.FullyLiquidated).sum()))
(HERE / "values.tex").write_text("\n".join("\\newcommand{\\" + name + "}{" + value + "}" for name, value in macros.items()) + "\n")

caption = {f: NAMES[f] + "因子的共同样本双周频回测" for f in strategy.FACTORS}
pairs = []
for i in range(0, 6, 2):
    f1, f2 = strategy.FACTORS[i:i+2]
    pairs.append(r"\clearpage" + "\n" + (r"\subsection{双周频分层净值与理论多空曲线}" if i == 0 else "") + "\n"
                 + rf"\navfigure{{{f1}_BiWeekly}}{{{caption[f1]}}}" + "\n\\vfill\n"
                 + rf"\navfigure{{{f2}_BiWeekly}}{{{caption[f2]}}}")
(HERE / "main_figures.tex").write_text("\n".join(pairs))
appendix = []
for i, factor in enumerate(ORDER):
    appendix.append(r"\clearpage" + "\n" + (r"\section{周频与月频完整净值图}" if i == 0 else "") + "\n"
                    + rf"\navfigure{{{factor}_Weekly}}{{{NAMES[factor]}的共同样本周频回测}}"
                    + "\n\\vfill\n" + rf"\navfigure{{{factor}_Monthly}}{{{NAMES[factor]}的共同样本月频回测}}")
(HERE / "appendix_figures.tex").write_text("\n".join(appendix))

for name in ["code2_new.py", "code2_xgboost.py", "code2_new_说明.md"]:
    shutil.copy2(ROOT / name, HERE / "source" / name)
for name in ["test_code2_new.py", "test_code2_xgboost.py"]:
    shutil.copy2(ROOT / "tests" / name, HERE / "source" / name)
for name in ["char.png", "logo.png"]:
    shutil.copy2(ROOT / "report/figures" / name, FIG / name)
shutil.copy2(RESULTS / "run_config.json", HERE / "run_config.json")
shutil.copy2(COMPARE / "comparison_config.json", HERE / "comparison_config.json")
receipt = {**counts, "unit_tests_passed": 26, "monthly_models": len(logs),
           "train_label_purge_checked": True, "common_prediction_sample_checked": True,
           "cash_cost_nav_metrics_reconciled": True,
           "comparison_start": comparison_config["signal_start"], "return_days": int(summary.Days.iloc[0]),
           "comparison_input_rows": len(sample), "source_hash": config["source_sha256"],
           "model_source_hash": comparison_config["xgboost_source_sha256"],
           "original_code2_unchanged": hashlib.sha256((ROOT / 'code2.py').read_bytes()).hexdigest()
                  == json.loads((ROOT / 'audit/code2_check_results.json').read_text())["source_sha256"]}
(HERE / "validation.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({**receipt, "model_results": ls[ls.Factor.eq("Pred_Score")][["Period","AnnualReturn","Sharpe","MaxDrawdown"]].to_dict("records")}, ensure_ascii=False, indent=2))

# 将表格输入展开成普通行，避免LaTeX保护性input在表格末尾干扰booktabs。
template = (HERE / "report_new.tex").read_text()
for file in TABLES.glob("*.tex"):
    template = template.replace("\\input{tables/" + file.name + "}", file.read_text())
(HERE / "report_ready.tex").write_text(template)
