"""按月滚动的 XGBoost 合成；与 code2_new 的持仓及成本账本共用口径。

每个持有期单独训练。目标为未来N日连续开盘收益的截面去均值值。
模型月初更新，只使用标签退出日严格早于该月首个交易日的样本；
固定参数，不用测试期挑参数或早停。单因子对照使用同一完整特征样本。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import code2_new as base


LOG = logging.getLogger("code2_new.xgboost")


def make_labels(features, market, days):
    """未来缺失仅影响历史训练标签；不用于过滤预测日股票池。"""
    labels = features[base.KEYS].copy()
    t = market.dates.get_indexer(labels.TradingDay)
    s = market.symbols.get_indexer(labels.SecuCode)
    if (t < 0).any() or (s < 0).any():
        raise ValueError("特征日期或股票未对齐行情。")
    valid = t + days + 1 < len(market.dates)
    buy = np.full(len(labels), np.nan)
    sell = buy.copy()
    buy[valid] = market.open[t[valid] + 1, s[valid]]
    sell[valid] = market.open[t[valid] + days + 1, s[valid]]
    good = valid & np.isfinite(buy) & np.isfinite(sell) & (buy > 0) & (sell > 0)
    labels["ForwardReturn"] = np.nan
    labels.loc[good, "ForwardReturn"] = sell[good] / buy[good] - 1
    labels["LabelEnd"] = pd.NaT
    labels.loc[valid, "LabelEnd"] = market.dates[t[valid] + days + 1].to_numpy()
    labels["Target"] = labels.ForwardReturn - labels.groupby("TradingDay").ForwardReturn.transform("mean")
    return labels


def training_sample(labeled, cutoff, window_months=12):
    """在月初之前已实现的标签；不能仅按信号日早于测试日筛样本。"""
    cutoff = pd.Timestamp(cutoff)
    start = cutoff - pd.DateOffset(months=window_months)
    mask = (labeled.TradingDay.ge(start) & labeled.TradingDay.lt(cutoff)
            & labeled.LabelEnd.lt(cutoff) & labeled.Target.notna())
    return labeled.loc[mask].copy()


def walk_forward(features, market, days, output, predict_start="2017-02-01", window_months=12):
    from xgboost import XGBRegressor
    import xgboost

    features = features.dropna(subset=list(base.FACTORS)).copy()
    labels = make_labels(features, market, days)
    labeled = features.merge(labels, on=base.KEYS, validate="one_to_one")
    months = sorted(pd.DatetimeIndex(market.dates[market.dates >= pd.Timestamp(predict_start)]).to_period("M").unique())
    output = Path(output)
    (output / "models").mkdir(parents=True, exist_ok=True)
    forecasts, logs, importance = [], [], []
    parameters = dict(max_depth=5, learning_rate=0.05, n_estimators=100,
                      n_jobs=4, random_state=42, objective="reg:squarederror", tree_method="hist")
    for month in months:
        dates = market.dates[market.dates.to_period("M") == month]
        cutoff = dates[0]
        train = training_sample(labeled, cutoff, window_months)
        test = features[features.TradingDay.dt.to_period("M").eq(month)
                        & features.TradingDay.ge(pd.Timestamp(predict_start))]
        if test.empty:
            continue
        if train.TradingDay.nunique() < 160 or len(train) < 10000:
            raise ValueError(f"{month} / {days}日模型历史样本不足；须延后预测起点或补充训练历史。")
        if train.LabelEnd.max() >= cutoff:
            raise AssertionError("训练标签尚未实现。")
        # 截面股票数不同，每个训练日期总权重相同；权重再归一化为均值1。
        counts = train.groupby("TradingDay").SecuCode.transform("count")
        weights = 1 / counts.to_numpy(float)
        weights /= weights.mean()
        model = XGBRegressor(**parameters)
        model.fit(train[list(base.FACTORS)], train.Target, sample_weight=weights)
        pred = test[base.KEYS].copy()
        pred["Pred_Score"] = model.predict(test[list(base.FACTORS)])
        forecasts.append(pred)
        model.save_model(output / "models" / f"{month}.json")
        logs.append({"PredictionMonth": str(month), "Cutoff": cutoff,
                     "TrainSignalStart": train.TradingDay.min(), "TrainSignalEnd": train.TradingDay.max(),
                     "MaxLabelEnd": train.LabelEnd.max(), "TrainDays": train.TradingDay.nunique(),
                     "TrainRows": len(train), "PredictionRows": len(test), "Horizon": days})
        importance.extend({"PredictionMonth": str(month), "Feature": factor, "Importance": float(value)}
                          for factor, value in zip(base.FACTORS, model.feature_importances_))
        LOG.info("XGBoost %s / %d日：训练 %d 行，标签最晚 %s，预测 %d 行", month, days,
                 len(train), train.LabelEnd.max().date(), len(test))
    if not forecasts:
        raise ValueError("没有生成样本外预测。")
    predictions = pd.concat(forecasts, ignore_index=True).sort_values(base.KEYS)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    pd.DataFrame(logs).to_csv(output / "training_log.csv", index=False)
    pd.DataFrame(importance).to_csv(output / "feature_importance.csv", index=False)
    (output / "model_config.json").write_text(json.dumps({"parameters": parameters,
        "xgboost_version": xgboost.__version__, "horizon": days, "window_months": window_months,
        "predict_start": predict_start, "features": list(base.FACTORS),
        "target": "cross-section demeaned adjusted open-to-open N-day return",
        "purge": "LabelEnd < first trading day of prediction month"}, ensure_ascii=False, indent=2))
    return predictions


def export_case(data, factor, period_name, days, market, cfg, destination, low_is_long):
    groups, schedule = base.signal_groups(data, factor, market.dates, days, cfg.min_cross_section, low_is_long)
    ic = base.calc_ic(groups, market, factor, days, low_is_long)
    ledger, trades = base.backtest(groups, schedule, market, cfg)
    destination.mkdir(parents=True, exist_ok=True)
    groups.to_parquet(destination / "signal_groups.parquet", index=False)
    ic.to_csv(destination / "ic.csv", index=False)
    ledger.to_csv(destination / "daily_ledger.csv", index=False)
    trades.to_parquet(destination / "trades.parquet", index=False)
    base.plot_result(ledger, destination / "nav.png", f"{factor} / {period_name} / OOS", cfg, market.dates[0])
    records = []
    for name, frame in ledger.groupby("Portfolio", sort=True):
        metrics = base.performance(frame.Return, cfg)
        mean, std = ic.IC_Directed.mean(), ic.IC_Directed.std()
        metrics.update({"Factor": factor, "Period": period_name, "Portfolio": name,
                        "IC_Raw": ic.IC_Raw.mean(), "RankIC_Raw": ic.RankIC_Raw.mean(),
                        "IC_Directed": mean, "RankIC_Directed": ic.RankIC_Directed.mean(),
                        "ICIR_PerPeriod": mean / std if std > 1e-12 else np.nan,
                        "IC_Coverage": ic.Coverage.mean(), "IC_Periods": len(ic),
                        "TotalTradingCost": frame.TradingCost.sum(), "TotalBorrowCost": frame.BorrowCost.sum(),
                        "MeanDailyTurnover": frame.Turnover.mean(),
                        "StalePositionDays": frame.StalePositions.sum(),
                        "HasStaleValuations": bool(frame.StalePositions.gt(0).any()),
                        "FinalPositions": frame.Positions.iloc[-1],
                        "FinalPendingReductions": frame.PendingReductions.iloc[-1],
                        "FullyLiquidated": bool(frame.Positions.iloc[-1] == 0)})
        records.append(metrics)
        if name in ("Group5", "TheoreticalLS"):
            LOG.info("OOS %s / %s / %s：年化 %.2f%%，Sharpe %.3f，回撤 %.2f%%", factor,
                     period_name, name, 100 * metrics["AnnualReturn"], metrics["Sharpe"],
                     100 * metrics["MaxDrawdown"])
    return records


def run_xgboost(cfg, predict_start="2017-02-01"):
    """先训练每一期限模型，再在相同日期/股票样本上重新回测所有对照。"""
    output = Path(cfg.output_dir)
    features = pd.read_parquet(output / "processed_features.parquet")
    features["TradingDay"] = pd.to_datetime(features.TradingDay)
    # 可预测样本只由当时六个输入是否齐全决定，不查看未来标签。
    complete = features.dropna(subset=list(base.FACTORS)).copy()
    loader = base.DataLoader(cfg)
    calendar = loader.calendar()
    universe = loader.universe(calendar)
    daily = loader.quotes(calendar, sorted(universe.SecuCode.unique()))
    market = base.Market.from_daily(daily, calendar)
    predictions = {}
    for name, days in base.PERIODS.items():
        predictions[name] = walk_forward(complete, market, days, output / "xgboost" / name, predict_start)
    comparison_dates = calendar[calendar >= pd.Timestamp(predict_start)]
    if len(comparison_dates) < 22:
        raise ValueError("样本外区间不足。")
    comparison_market = base.Market.from_daily(daily, comparison_dates)
    sample = complete[complete.TradingDay.isin(comparison_dates)].copy()
    # 所有期限都必须覆盖同一完整输入样本，不能因标签或训练筛选改变测试股票池。
    keys = sample[base.KEYS].sort_values(base.KEYS).reset_index(drop=True)
    for name, pred in predictions.items():
        pd.testing.assert_frame_equal(keys, pred[base.KEYS].sort_values(base.KEYS).reset_index(drop=True))
    sample.to_parquet(output / "xgboost" / "comparison_sample.parquet", index=False)
    result = []
    comparison = output / "comparison"
    for name, days in base.PERIODS.items():
        for factor in base.FACTORS:
            result.extend(export_case(sample, factor, name, days, comparison_market, cfg,
                                      comparison / factor / name, True))
        result.extend(export_case(predictions[name], "Pred_Score", name, days, comparison_market, cfg,
                                  comparison / "Pred_Score" / name, False))
    frame = pd.DataFrame(result)
    frame.to_csv(comparison / "summary.csv", index=False)
    manifest = {"prediction_start_requested": predict_start,
                "signal_start": str(comparison_dates[0].date()),
                "return_start": str(comparison_dates[1].date()),
                "end": str(comparison_dates[-1].date()), "comparison_rows": len(sample),
                "input_sample": "six finite contemporaneous factor values; same for all cases",
                "base_source_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                "xgboost_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (comparison / "comparison_config.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    LOG.info("XGBoost与单因子共同样本比较完成：%s", comparison)
    return frame
