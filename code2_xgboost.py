"""按月滚动的 XGBoost 合成；与 code2_new 的持仓及成本账本共用口径。

每个持有期单独训练。目标为未来N日连续开盘收益的截面去均值值。
模型月初更新，只使用标签退出日严格早于该月首个交易日的样本；
训练期内相关性去重和择优，季度执行两阶段参数搜索，月度重训。
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


def walk_forward(*args, **kwargs):
    from code2_tuning import walk_forward as tuned_walk_forward
    return tuned_walk_forward(*args, **kwargs)
