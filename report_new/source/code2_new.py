"""六个高频因子的研究和逐日持仓回测，保留原 code2.py 不变。

运行：python code2_new.py --help
验证：python -m unittest discover -s tests -p 'test_code2_new.py'

口径：T 收盘后信号，T+1 开盘执行，每 5/10/20 个交易日调仓。
用当日除权参考昨收重建向前连续价格；持仓数量使用该连续价格的单位。
研究采用可分割份额，不模拟整手、冲击成本、现金分红到账时点或税费明细。
日行情没有精确涨跌停价，默认以相对参考昨收的 4.9% 开盘跳空作为保守
方向性成交过滤（可配置），并非精确涨跌停撮合。无法买入的资金留现金，
无法减仓的头寸保留并逐日重试。未知行情按最近估值保留，并明确报告。
多空组合为理论诊断：调仓时多空目标各 50% 净资产，假定可借券；年借券
费率可配置。它与五个多头组均有各自的持仓、现金及实际成交费用账本。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from numba import njit
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


BASE = Path(__file__).resolve().parent
LOG = logging.getLogger("code2_new")
FACTORS = (
    "HF_SmartMoney", "HF_Raw_CPV_Mean", "HF_CDPDP",
    "HF_Skewness", "HF_Tail_Reversal", "HF_Amplitude",
)
PERIODS = {"Weekly": 5, "BiWeekly": 10, "Monthly": 20}
CONTROLS = ["Ctrl_Ret20", "Ctrl_Vol20", "Ctrl_Turnover20"]
KEYS = ["TradingDay", "SecuCode"]


@dataclass
class Config:
    data_dir: Path = BASE / "data"
    output_dir: Path = BASE / "results_code2_new"
    start: str = "2016-01-04"
    end: str = "2018-12-31"
    index: str = "500"
    factor_window: int = 20
    smart_window: int = 10
    annual_days: int = 250
    # 原代码双边千三，明确分解为每次买/卖各千一点五。
    one_way_fee: float = 0.0015
    open_gap_limit: float = 0.049
    annual_borrow_fee: float = 0.0
    annual_risk_free: float = 0.0
    min_cross_section: int = 20


def normalize_codes(values):
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


@njit(cache=True)
def subset_corr(changes, direction):
    x = changes[:-1]
    y = changes[1:]
    keep = (x > 0) & (y > 0) if direction > 0 else (x < 0) & (y < 0)
    a, b = x[keep], y[keep]
    if len(a) < 3:
        return np.nan
    a, b = a - np.mean(a), b - np.mean(b)
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return np.sum(a * b) / denom if denom > 1e-15 else np.nan


@njit(cache=True)
def intraday_metrics(open_, close, high, low, volume, amount):
    """按时间排序后的当日分钟数组；Smart Money 使用真实成交额。"""
    n = len(close)
    missing = np.full(6, np.nan)
    if n < 31:
        return missing
    if (np.any(~np.isfinite(open_)) or np.any(~np.isfinite(close))
            or np.any(~np.isfinite(high)) or np.any(~np.isfinite(low))
            or np.any(~np.isfinite(volume)) or np.any(~np.isfinite(amount))
            or np.any(open_ <= 0) or np.any(close <= 0)
            or np.any(high <= 0) or np.any(low <= 0)
            or np.any(volume < 0) or np.any(amount < 0)):
        return missing
    total_vol, total_amt = np.sum(volume), np.sum(amount)
    if total_vol <= 0 or total_amt <= 0:
        return missing
    scores = np.full(n, -np.inf)
    for i in range(n):
        previous = open_[0] if i == 0 else close[i - 1]
        if volume[i] > 0:
            scores[i] = abs(close[i] / previous - 1) / volume[i] ** 0.25
    remaining, smart_amt = total_vol * 0.2, 0.0
    for i in np.argsort(scores)[::-1]:
        if volume[i] <= 0:
            continue
        used = min(remaining, volume[i])
        smart_amt += amount[i] * used / volume[i]
        remaining -= used
        if remaining <= total_vol * 1e-12:
            break
    smart = (smart_amt / (total_vol * 0.2)) / (total_amt / total_vol)
    p, v = close - np.mean(close), volume - np.mean(volume)
    denominator = np.sqrt(np.sum(p * p) * np.sum(v * v))
    cpv = np.sum(p * v) / denominator if denominator > 1e-15 else np.nan
    changes = close[1:] - close[:-1]
    cdpdp = subset_corr(changes, 1) + subset_corr(changes, -1)
    returns = close[1:] / close[:-1] - 1
    centered = returns - np.mean(returns)
    variance = np.mean(centered ** 2)
    skew = np.mean(centered ** 3) / variance ** 1.5 if variance > 1e-15 else 0.0
    tail = close[-1] / close[-31] - 1  # 30 个收盘到收盘区间
    amplitude = (np.max(high) - np.min(low)) / open_[0]
    return np.array([smart, cpv, cdpdp, skew, tail, amplitude])


def prepare_daily(raw, calendar, cfg):
    """先在完整个股日历上计算连续价格及风格变量，再筛选股票池。

    scale[t]/scale[t-1] = 最后实际收盘价 / 当日除权参考昨收。
    因此连续收盘收益为 Close[t]/PrevClose[t]-1，不把除权当作亏损。
    当日开盘用同一个当日 scale，未将未来收益乘回当日价格。
    """
    raw = raw.copy()
    raw["TradingDay"] = pd.to_datetime(raw.TradingDay)
    raw["SecuCode"] = normalize_codes(raw.SecuCode)
    if raw.duplicated(KEYS).any():
        raise ValueError("日行情存在重复的股票/日期，不能直接回测。")
    pieces = []
    for symbol, group in raw.groupby("SecuCode", sort=True):
        g = group.set_index("TradingDay").reindex(calendar).copy()
        g.index.name = "TradingDay"
        g["SecuCode"] = symbol
        # 成交判断只能读取开盘时已知字段，不使用全天成交量或收盘结果。
        open_observed = g.OpenPrice.gt(0) & np.isfinite(g.OpenPrice) & g.PTFlag.eq(0)
        close_observed = g.ClosePrice.gt(0) & np.isfinite(g.ClosePrice) & g.PTFlag.eq(0)
        halted = g.PTFlag.eq(1)
        if (open_observed & ~close_observed).any():
            raise ValueError(f"{symbol} 有开盘却无有效收盘，无法可靠估值，须补全行情。")
        known_state = close_observed | halted
        observed_dates = g.index[close_observed]
        if len(observed_dates):
            interior = (g.index >= observed_dates[0]) & (g.index <= observed_dates[-1])
            if (interior & ~known_state).any():
                # 跨未知缺口无法区分价格涨跌和除权调整，禁止静默伪造复权收益。
                raise ValueError(f"{symbol} 历史行情存在未解释缺口，须补全行情或提供复权因子。")
        observed = open_observed | close_observed
        raw_close = g.ClosePrice.where(close_observed)
        prior_actual = raw_close.ffill().shift(1)
        if (observed & prior_actual.notna() & ~g.PrevClosePrice.gt(0)).any():
            raise ValueError(f"{symbol} 缺少有效除权参考昨收，无法建立连续价格。")
        step = (prior_actual / g.PrevClosePrice).where(observed & prior_actual.notna(), 1.0)
        if (~np.isfinite(step) | (step <= 0)).any():
            raise ValueError(f"{symbol} 价格调整因子无效。")
        scale = step.cumprod()
        g["AdjOpen"] = (g.OpenPrice * scale).where(open_observed)
        g["AdjClose"] = (g.ClosePrice * scale).where(close_observed).ffill()
        g["KnownState"] = known_state
        g["StaleQuote"] = ~g.KnownState
        gap = g.OpenPrice / g.PrevClosePrice - 1
        # 只看执行时已知的开盘，不使用当日最高/最低价推断成交。
        reference_ok = g.PrevClosePrice.gt(0)
        g["CanBuy"] = open_observed & reference_ok & gap.lt(cfg.open_gap_limit)
        g["CanSell"] = open_observed & reference_ok & gap.gt(-cfg.open_gap_limit)
        daily_ret = g.AdjClose.pct_change(fill_method=None)
        daily_ret = daily_ret.where(g.KnownState & g.KnownState.shift(1, fill_value=False))
        complete = g.KnownState.rolling(21, min_periods=21).sum().eq(21)
        g["Ctrl_Ret20"] = (g.AdjClose / g.AdjClose.shift(20) - 1).where(complete)
        g["Ctrl_Vol20"] = daily_ret.rolling(20, min_periods=20).std()
        floats = g.NonRestrictedShares.where(g.NonRestrictedShares.gt(0), g.AFloats)
        turnover = (g.Volume / floats).where(close_observed & floats.gt(0))
        turnover = turnover.mask(halted, 0.0)
        g["Ctrl_Turnover20"] = turnover.rolling(20, min_periods=20).mean()
        pieces.append(g[["SecuCode", "AdjOpen", "AdjClose", "CanBuy", "CanSell",
                         "StaleQuote"] + CONTROLS].reset_index())
    if not pieces:
        raise ValueError("筛选后没有日行情。")
    return pd.concat(pieces, ignore_index=True)


class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg.data_dir)

    def calendar(self):
        path = self.root / "交易日期/tradingdays.csv"
        dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(path).iloc[:, 0])).sort_values().unique()
        dates = dates[dates <= pd.Timestamp(self.cfg.end)]
        if not len(dates) or not (dates >= pd.Timestamp(self.cfg.start)).any():
            raise ValueError("指定区间没有交易日。")
        # 给风格变量及分钟滚动窗口留出历史，起点由交易日历决定。
        first = dates.searchsorted(pd.Timestamp(self.cfg.start))
        return dates[max(0, first - 60):]

    def universe(self, calendar):
        code = "000905" if self.cfg.index == "500" else "000300"
        path = self.root / f"指数权重/index_wt_{code}.parquet"
        univ = pd.read_parquet(path, columns=["date", "symbol"])
        univ = univ.rename(columns={"date": "TradingDay", "symbol": "SecuCode"})
        univ["TradingDay"] = pd.to_datetime(univ.TradingDay)
        univ["SecuCode"] = normalize_codes(univ.SecuCode)
        return univ[univ.TradingDay.isin(calendar)].drop_duplicates(KEYS)

    def quotes(self, calendar, symbols):
        fields = ["SecuCode", "TradingDay", "OpenPrice", "ClosePrice", "PrevClosePrice",
                  "PTFlag", "TradingVolumes", "NonRestrictedShares", "AFloats"]
        files = sorted((self.root / "日行情").glob("daily_quote_*.csv"))
        parts = []
        for path in files:
            year = int(path.stem.rsplit("_", 1)[1])
            if not calendar[0].year <= year <= calendar[-1].year:
                continue
            for chunk in pd.read_csv(path, usecols=fields, dtype={"SecuCode": str}, chunksize=250000):
                chunk["SecuCode"] = normalize_codes(chunk.SecuCode)
                chunk = chunk[chunk.SecuCode.isin(symbols)].copy()
                chunk["TradingDay"] = pd.to_datetime(chunk.TradingDay)
                parts.append(chunk[chunk.TradingDay.isin(calendar)])
        if not parts:
            raise FileNotFoundError("找不到对应年份日行情。")
        raw = pd.concat(parts, ignore_index=True).rename(columns={"TradingVolumes": "Volume"})
        return prepare_daily(raw, calendar, self.cfg)

    def industry_cap(self, calendar, symbols):
        industry = pd.read_csv(self.root / "股票行业分类/industry.csv", dtype={"SecuCode": str})
        industry["SecuCode"] = normalize_codes(industry.SecuCode)
        industry = industry[industry.SecuCode.isin(symbols)].copy()
        industry["InfoPublDate"] = pd.to_datetime(industry.InfoPublDate) + pd.Timedelta(days=1)
        industry = industry.sort_values("InfoPublDate")
        # Arrow 下推筛选，避免把所有股票所有年份市值读入内存。
        dataset = ds.dataset(self.root / "股票市值/stock_cap.parquet", format="parquet")
        mask = (ds.field("symbol").isin(list(symbols))
                & (ds.field("date") >= str(calendar[0].date()))
                & (ds.field("date") <= str(calendar[-1].date())))
        cap = dataset.to_table(columns=["date", "symbol", "tot_cap"], filter=mask).to_pandas()
        cap = cap.rename(columns={"date": "TradingDay", "symbol": "SecuCode"})
        cap["TradingDay"] = pd.to_datetime(cap.TradingDay)
        cap["SecuCode"] = normalize_codes(cap.SecuCode)
        cap["LnCap"] = np.log(cap.tot_cap.where(cap.tot_cap.gt(0)))
        if cap.duplicated(KEYS).any():
            raise ValueError("市值存在重复的股票/日期。")
        return industry, cap[KEYS + ["LnCap"]]


class FactorEngine:
    """先计算候选股票的完整历史，再按信号日动态成分选股。"""

    def __init__(self, cfg, calendar, symbols):
        self.cfg, self.calendar, self.symbols = cfg, calendar, set(symbols)
        self.cache = Path(cfg.output_dir) / "intraday_cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.definition_hash = hashlib.sha256(
            (inspect.getsource(intraday_metrics.py_func) + inspect.getsource(subset_corr.py_func)).encode()
        ).hexdigest()

    def run(self):
        all_files = sorted((Path(self.cfg.data_dir) / "bar_1m_2016_2018").glob("*.parquet"))
        if not all_files:
            raise FileNotFoundError("分钟数据目录没有 parquet 文件。")
        parts = []
        first_hf_date = pd.Timestamp(all_files[0].stem)
        files = [path for path in all_files if pd.Timestamp(path.stem) in self.calendar]
        for path in tqdm(files, desc="分钟因子（日缓存）"):
            day = pd.Timestamp(path.stem)
            metadata = {
                "definition": self.definition_hash, "source": str(path.resolve()),
                "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                "symbols": sorted(self.symbols),
            }
            cached = self.cache / path.name
            manifest = cached.with_suffix(".json")
            if cached.exists() and manifest.exists() and json.loads(manifest.read_text()) == metadata:
                day_factors = pd.read_parquet(cached)
            else:
                dataset = ds.dataset(path, format="parquet")
                columns = ["time", "symbol", "open", "close", "high", "low", "volume", "amount"]
                bars = dataset.to_table(columns=columns, filter=ds.field("symbol").isin(sorted(self.symbols))).to_pandas()
                bars = bars.sort_values(["symbol", "time"])
                rows = []
                for symbol, g in bars.groupby("symbol", sort=False):
                    if g.time.duplicated().any():
                        raise ValueError(f"{day.date()} {symbol} 分钟时间重复。")
                    metrics = intraday_metrics(*[
                        g[column].to_numpy(dtype=np.float64)
                        for column in ["open", "close", "high", "low", "volume", "amount"]
                    ])
                    rows.append([day, str(symbol).zfill(6), *metrics])
                day_factors = pd.DataFrame(rows, columns=KEYS + list(FACTORS))
                day_factors.to_parquet(cached, index=False)
                manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
            parts.append(day_factors)
        if not parts:
            raise ValueError("所选日期没有分钟数据。")
        raw = pd.concat(parts, ignore_index=True)
        return roll_factors(raw, self.calendar, self.cfg), first_hf_date


def roll_factors(raw, calendar, cfg):
    parts = []
    for symbol, group in raw.groupby("SecuCode", sort=True):
        grid = group.set_index("TradingDay")[list(FACTORS)].reindex(calendar)
        rolling = grid.rolling(cfg.factor_window, min_periods=cfg.factor_window).mean()
        rolling[FACTORS[0]] = grid[FACTORS[0]].rolling(cfg.smart_window, min_periods=cfg.smart_window).mean()
        # 任何因子都至少经过完整的公共预热期，缺失日不会压缩日历窗口。
        rolling.iloc[:cfg.factor_window - 1] = np.nan
        rolling["SecuCode"] = symbol
        rolling.index.name = "TradingDay"
        parts.append(rolling.reset_index())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEYS + list(FACTORS))


def process_features(factors, universe, daily, industry, cap, cfg):
    full = factors.merge(universe, on=KEYS, how="inner", validate="one_to_one")
    full = full.merge(daily[KEYS + CONTROLS], on=KEYS, how="left", validate="one_to_one")
    full = full.merge(cap, on=KEYS, how="left", validate="one_to_one")
    full = pd.merge_asof(
        full.sort_values("TradingDay"), industry[["SecuCode", "InfoPublDate", "tag"]],
        left_on="TradingDay", right_on="InfoPublDate", by="SecuCode", direction="backward",
    )
    full["Industry"] = full.tag.fillna("Unknown")
    processed = []
    for day, group in tqdm(full.groupby("TradingDay", sort=True), desc="截面中性化"):
        out = group[KEYS].copy()
        for factor in FACTORS:
            controls = CONTROLS if "CPV" in factor else []
            valid = group[[factor, "LnCap"] + controls].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            sub = group.loc[valid]
            out[factor] = np.nan
            if len(sub) < cfg.min_cross_section:
                continue
            y = sub[factor].to_numpy(dtype=float)
            median = np.median(y)
            mad = np.median(np.abs(y - median))
            if mad > 1e-12:
                y = np.clip(y, median - 5 * mad, median + 5 * mad)
            industries = pd.get_dummies(sub.Industry, drop_first=True, dtype=float)
            x = pd.concat([industries, sub[["LnCap"] + controls]], axis=1).astype(float)
            # 因子缺失不靠填0伪造信号；CPV控制变量缺失也不参与回归。
            fit = LinearRegression().fit(x, y)
            residual = y - fit.predict(x)
            std = np.std(residual)
            if std > 1e-12:
                out.loc[sub.index, factor] = (residual - np.mean(residual)) / std
        processed.append(out)
    if not processed:
        raise ValueError("没有可用因子截面。")
    return pd.concat(processed, ignore_index=True).sort_values(KEYS)


@dataclass
class Market:
    dates: pd.DatetimeIndex
    symbols: pd.Index
    open: np.ndarray
    close: np.ndarray
    buy: np.ndarray
    sell: np.ndarray
    stale: np.ndarray

    @classmethod
    def from_daily(cls, daily, dates):
        symbols = pd.Index(sorted(daily.SecuCode.unique()), name="SecuCode")

        def matrix(column):
            return daily.pivot(index="TradingDay", columns="SecuCode", values=column).reindex(index=dates, columns=symbols)

        return cls(pd.DatetimeIndex(dates), symbols, matrix("AdjOpen").to_numpy(float),
                   matrix("AdjClose").to_numpy(float), matrix("CanBuy").fillna(False).to_numpy(bool),
                   matrix("CanSell").fillna(False).to_numpy(bool), matrix("StaleQuote").fillna(True).to_numpy(bool))


def signal_groups(features, factor, calendar, period_days, min_stocks=20, low_is_long=True):
    """独立交易日历调仓；先按当时因子分组，不读取任何未来收益。"""
    schedule = pd.DatetimeIndex(calendar)[::period_days]
    frames = []
    for day in schedule:
        sub = features.loc[features.TradingDay.eq(day), KEYS + [factor]].dropna(subset=[factor]).copy()
        if len(sub) < min_stocks or sub[factor].nunique() < 5:
            continue  # 这个预定调仓日目标为现金，不把下一个有效日期提前。
        # Group5固定为看好的组；相同因子值用股票代码稳定打破平局。
        sub = sub.sort_values([factor, "SecuCode"], ascending=[not low_is_long, True])
        sub["Group"] = np.arange(len(sub)) * 5 // len(sub) + 1
        frames.append(sub)
    columns = KEYS + [factor, "Group"]
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)), schedule


def calc_ic(groups, market, factor, days, low_is_long=True):
    rows = []
    sign = -1 if low_is_long else 1
    for day, sub in groups.groupby("TradingDay", sort=True):
        t = market.dates.get_indexer([day])[0]
        if t < 0 or t + days + 1 >= len(market.dates):
            continue  # 所有股票统一省略数据尾部不完整期限；不改变交易分组。
        columns = market.symbols.get_indexer(sub.SecuCode)
        if (columns < 0).any():
            raise ValueError("因子股票没有对应行情列。")
        buy = market.open[t + 1, columns]
        sell = market.open[t + days + 1, columns]
        valid = np.isfinite(buy) & (buy > 0) & np.isfinite(sell) & (sell > 0)
        values = sub[factor].to_numpy(float)[valid]
        rets = sell[valid] / buy[valid] - 1
        pearson = rank = np.nan
        if len(rets) >= 10 and np.std(values) > 1e-12 and np.std(rets) > 1e-12:
            pearson = pd.Series(values).corr(pd.Series(rets))
            rank = pd.Series(values).corr(pd.Series(rets), method="spearman")
        rows.append({"SignalDate": day, "EntryDate": market.dates[t + 1],
                     "ExitDate": market.dates[t + days + 1], "N_Signal": len(sub),
                     "N_Paired": int(valid.sum()), "Coverage": valid.mean(),
                     "IC_Raw": pearson, "RankIC_Raw": rank,
                     "IC_Directed": sign * pearson, "RankIC_Directed": sign * rank})
    return pd.DataFrame(rows, columns=["SignalDate", "EntryDate", "ExitDate", "N_Signal", "N_Paired",
                                       "Coverage", "IC_Raw", "RankIC_Raw", "IC_Directed", "RankIC_Directed"])


class Portfolio:
    """连续价格单位下的现金/持仓账本，调仓外只重试未成交的减仓单。"""

    def __init__(self, symbols, cfg, name):
        self.symbols, self.cfg, self.name = symbols, cfg, name
        self.units = np.zeros(len(symbols))
        self.cash = 1.0
        self.pending = np.full(len(symbols), np.nan)
        self.trades = []

    def value(self, prices):
        held = np.abs(self.units) > 1e-14
        if np.any(held & (~np.isfinite(prices) | (prices <= 0))):
            raise ValueError("已有持仓缺少历史估值，不能凭空删除头寸。")
        return self.cash + np.sum(self.units[held] * prices[held])

    def trade(self, delta, prices, day):
        active = np.flatnonzero(np.abs(delta) > 1e-14)
        notionals = np.abs(delta[active] * prices[active])
        costs = notionals * self.cfg.one_way_fee
        self.cash -= np.sum(delta[active] * prices[active]) + costs.sum()
        self.units += delta
        self.units[np.abs(self.units) < 1e-14] = 0
        for i, notional, cost in zip(active, notionals, costs):
            self.trades.append({"TradingDay": day, "Portfolio": self.name, "SecuCode": self.symbols[i],
                                "DeltaUnits": delta[i], "AdjPrice": prices[i],
                                "Notional": notional, "Cost": cost})
        return notionals.sum(), costs.sum()

    def execute(self, prices, can_buy, can_sell, day, weights=None):
        equity = self.value(prices)
        if equity <= 0:
            raise ValueError(f"{day} {self.name} 净资产非正，停止回测。")
        if weights is not None:
            if np.sum(np.abs(weights)) > 1 + 1e-10:
                raise ValueError("目标总敞口不能超过净资产。")
            target = np.divide(weights * equity, prices, out=np.zeros_like(weights),
                               where=np.isfinite(prices) & (prices > 0))
            # 新调仓指令覆盖旧指令；目标没有行情的新买单不转移给其他股票。
            self.pending[:] = np.nan
        else:
            target = np.where(np.isfinite(self.pending), self.pending, self.units)
        opposite = self.units * target < 0
        reduction = (np.abs(target) < np.abs(self.units)) | opposite
        reduced_target = np.where(opposite, 0.0, target)
        delta = np.where(reduction, reduced_target - self.units, 0.0)
        permitted = np.where(delta > 0, can_buy, can_sell)
        blocked_reductions = reduction & ~permitted
        self.pending[reduction] = reduced_target[reduction]
        done = delta.copy()
        done[~permitted] = 0
        notional, cost = self.trade(done, prices, day)
        self.pending[reduction & permitted] = np.nan
        blocked = int(blocked_reductions.sum())
        if weights is not None:
            # 先释放旧风险，再按剩余净资产/现金预算开新仓。未卖出的旧仓占用预算。
            add = target - self.units
            same_sign = (self.units == 0) | (self.units * add > 0)
            add[~same_sign] = 0
            allowed = np.where(add > 0, can_buy, can_sell)
            blocked += int(((np.abs(add) > 1e-14) & ~allowed).sum())
            add[~allowed] = 0
            valid_prices = np.nan_to_num(prices, nan=0.0)
            gross = np.sum(np.abs(self.units * valid_prices))
            additional = np.sum(np.abs(add * valid_prices))
            room = max(self.value(prices) - gross, 0.0)
            scale = min(1.0, room / (additional * (1 + self.cfg.one_way_fee))) if additional > 0 else 0.0
            # 卖空所得可融资；此现金约束主要约束多头组。按全部增加单统一缩放。
            cash_needed = np.sum(add * valid_prices) + additional * self.cfg.one_way_fee
            if cash_needed > 0:
                scale = min(scale, max(self.cash, 0.0) / cash_needed)
            n, c = self.trade(add * scale, prices, day)
            notional, cost = notional + n, cost + c
        return {"TradedNotional": notional, "TradingCost": cost,
                "Turnover": notional / equity, "BlockedOrders": blocked}


def backtest(groups, schedule, market, cfg):
    n = len(market.symbols)
    books = [Portfolio(market.symbols, cfg, f"Group{i}") for i in range(1, 6)]
    books.append(Portfolio(market.symbols, cfg, "TheoreticalLS"))
    group_map = {day: sub for day, sub in groups.groupby("TradingDay")}
    dates = market.dates
    first_signal = dates.get_indexer([schedule[0]])[0]
    if first_signal < 0 or first_signal + 2 >= len(dates):
        raise ValueError("至少需要信号日及后续两个交易日。")
    execution = {dates[dates.get_loc(day) + 1]: day for day in schedule
                 if day in dates and dates.get_loc(day) + 1 < len(dates) - 1}
    rows = []
    previous_nav = {book.name: 1.0 for book in books}
    for t in range(first_signal + 1, len(dates)):
        day = dates[t]
        previous_close = market.close[t - 1]
        opening = np.where(np.isfinite(market.open[t]), market.open[t], previous_close)
        closing = np.where(np.isfinite(market.close[t]), market.close[t], previous_close)
        new_weights = None
        if day in execution:
            new_weights = np.zeros((6, n))
            sub = group_map.get(execution[day])
            if sub is not None:
                for group in range(1, 6):
                    columns = market.symbols.get_indexer(sub.loc[sub.Group.eq(group), "SecuCode"])
                    if (columns < 0).any():
                        raise ValueError("信号股票不在行情矩阵中。")
                    if len(columns):
                        new_weights[group - 1, columns] = 1 / len(columns)
                new_weights[5] = 0.5 * (new_weights[4] - new_weights[0])
        if t == len(dates) - 1:
            new_weights = np.zeros((6, n))  # 样本末日开盘尝试平仓；受阻的仓位保留。
        for b, book in enumerate(books):
            borrow = np.sum(np.maximum(-book.units, 0) * np.nan_to_num(previous_close))
            borrow *= cfg.annual_borrow_fee / cfg.annual_days
            book.cash -= borrow
            stats = book.execute(opening, market.buy[t], market.sell[t], day,
                                 None if new_weights is None else new_weights[b])
            nav = book.value(closing)
            if nav <= 0:
                raise ValueError(f"{day} {book.name} 日末净资产非正。")
            held = np.abs(book.units) > 1e-14
            values = book.units * np.nan_to_num(closing)
            rows.append({"TradingDay": day, "Portfolio": book.name, "NAV": nav,
                         "Return": nav / previous_nav[book.name] - 1, "Cash": book.cash,
                         "LongValue": np.maximum(values, 0).sum(),
                         "ShortValue": np.maximum(-values, 0).sum(),
                         "Positions": int(held.sum()), "StalePositions": int((held & market.stale[t]).sum()),
                         "PendingReductions": int(np.isfinite(book.pending).sum()), "BorrowCost": borrow, **stats})
            previous_nav[book.name] = nav
    columns = ["TradingDay", "Portfolio", "SecuCode", "DeltaUnits", "AdjPrice", "Notional", "Cost"]
    trades = pd.DataFrame([trade for book in books for trade in book.trades], columns=columns)
    return pd.DataFrame(rows), trades


def performance(returns, cfg):
    values = np.asarray(returns, dtype=float)
    if not len(values) or not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError("绩效计算需要有限且大于-100%的日收益。")
    nav = np.r_[1.0, np.cumprod(1 + values)]
    vol = np.std(values, ddof=1) * np.sqrt(cfg.annual_days) if len(values) > 1 else 0.0
    risk_free_daily = (1 + cfg.annual_risk_free) ** (1 / cfg.annual_days) - 1
    excess = values - risk_free_daily
    deviation = np.std(excess, ddof=1) if len(values) > 1 else 0.0
    sharpe = np.mean(excess) / deviation * np.sqrt(cfg.annual_days) if deviation > 1e-12 else np.nan
    return {"Days": len(values), "FinalNAV": nav[-1], "TotalReturn": nav[-1] - 1,
            "AnnualReturn": nav[-1] ** (cfg.annual_days / len(values)) - 1,
            "AnnualVol": vol, "Sharpe": sharpe,
            "MaxDrawdown": np.min(nav / np.maximum.accumulate(nav) - 1)}


def pyplot(cfg):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(cfg.output_dir) / "matplotlib_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(cfg.output_dir).resolve() / "system_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_factor_correlation(features, cfg):
    """逐日截面Spearman相关系数等权平均，避免混合日期引入时间效应。"""
    total = np.zeros((len(FACTORS), len(FACTORS)))
    count = np.zeros_like(total)
    for _, frame in features.groupby("TradingDay", sort=False):
        corr = frame[list(FACTORS)].corr(method="spearman", min_periods=cfg.min_cross_section).to_numpy()
        finite = np.isfinite(corr)
        total += np.where(finite, corr, 0)
        count += finite
    average = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    pd.DataFrame(average, index=FACTORS, columns=FACTORS).to_csv(Path(cfg.output_dir) / "factor_correlation.csv")
    plt = pyplot(cfg)
    fig, axis = plt.subplots(figsize=(9, 8))
    im = axis.imshow(average, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [factor.removeprefix("HF_") for factor in FACTORS]
    axis.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isfinite(average[i, j]):
                axis.text(j, i, f"{average[i, j]:.2f}", ha="center", va="center")
    axis.set_title("Mean daily cross-sectional Spearman correlation")
    fig.colorbar(im, ax=axis)
    fig.tight_layout()
    fig.savefig(Path(cfg.output_dir) / "factor_correlation.png", dpi=160)
    plt.close(fig)


def plot_result(ledger, destination, title, cfg, baseline_date):
    plt = pyplot(cfg)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for name, frame in ledger.groupby("Portfolio", sort=True):
        nav = pd.Series(frame.NAV.to_numpy(), index=pd.to_datetime(frame.TradingDay))
        nav = pd.concat([pd.Series([1.0], index=[baseline_date]), nav])
        axis = axes[1] if name == "TheoreticalLS" else axes[0]
        axis.plot(nav.index, nav.values, label=name)
    for axis in axes:
        axis.axhline(1, color="gray", linewidth=0.5)
        axis.set_ylabel("Net NAV")
        axis.legend()
        axis.grid(alpha=0.3)
    axes[0].set_title(title + " | long-only groups")
    axes[1].set_title("Theoretical L/S: 50% long / 50% short at rebalance")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run_pipeline(cfg, selected_factors=FACTORS, selected_periods=None):
    if cfg.one_way_fee < 0 or not 0 < cfg.open_gap_limit < 1:
        raise ValueError("费用须非负，开盘跳空阈值须在0和1之间。")
    if cfg.annual_borrow_fee < 0 or cfg.annual_risk_free <= -1:
        raise ValueError("借券费率须非负，无风险年收益须大于-1。")
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(cfg)
    calendar = loader.calendar()
    universe = loader.universe(calendar)
    if universe.empty:
        raise ValueError("区间内没有动态指数成分。")
    symbols = sorted(universe.SecuCode.unique())
    LOG.info("加载 %d 只历史候选股票；区间 %s 至 %s", len(symbols), calendar[0].date(), calendar[-1].date())
    daily = loader.quotes(calendar, symbols)
    factors, first_hf = FactorEngine(cfg, calendar, symbols).run()
    industry, cap = loader.industry_cap(calendar, symbols)
    features = process_features(factors, universe, daily, industry, cap, cfg)
    # 使用固定数据起点+窗口预热，而非按某因子的第一个有效样本决定调仓起点。
    hf_calendar = calendar[calendar >= first_hf]
    if len(hf_calendar) < cfg.factor_window:
        raise ValueError("分钟数据不足公共预热期。")
    first_signal = max(pd.Timestamp(cfg.start), hf_calendar[cfg.factor_window - 1])
    analysis_dates = calendar[calendar >= first_signal]
    market = Market.from_daily(daily, analysis_dates)
    features.to_parquet(output / "processed_features.parquet", index=False)
    plot_factor_correlation(features[features.TradingDay.isin(analysis_dates)], cfg)
    summary = []
    selected_periods = selected_periods or PERIODS
    for factor in selected_factors:
        for period_name, days in selected_periods.items():
            LOG.info("回测 %s / %s", factor, period_name)
            groups, schedule = signal_groups(features, factor, analysis_dates, days, cfg.min_cross_section)
            if groups.empty:
                LOG.warning("%s / %s 无有效截面，保留预定调仓日并以现金回测", factor, period_name)
            ic = calc_ic(groups, market, factor, days)
            ledger, trades = backtest(groups, schedule, market, cfg)
            destination = output / factor / period_name
            destination.mkdir(parents=True, exist_ok=True)
            groups.to_parquet(destination / "signal_groups.parquet", index=False)
            ic.to_csv(destination / "ic.csv", index=False)
            ledger.to_csv(destination / "daily_ledger.csv", index=False)
            trades.to_parquet(destination / "trades.parquet", index=False)
            for name, frame in ledger.groupby("Portfolio", sort=True):
                metrics = performance(frame.Return, cfg)
                average_ic = ic.IC_Directed.mean()
                deviation = ic.IC_Directed.std()
                metrics.update({"Factor": factor, "Period": period_name, "Portfolio": name,
                                "IC_Raw": ic.IC_Raw.mean(), "IC_Directed": average_ic,
                                "RankIC_Directed": ic.RankIC_Directed.mean(),
                                "IC_Coverage": ic.Coverage.mean(),
                                "ICIR_PerPeriod": average_ic / deviation if deviation > 1e-12 else np.nan,
                                "TotalTradingCost": frame.TradingCost.sum(),
                                "TotalBorrowCost": frame.BorrowCost.sum(),
                                "MeanDailyTurnover": frame.Turnover.mean(),
                                "StalePositionDays": frame.StalePositions.sum(),
                                "HasStaleValuations": bool(frame.StalePositions.gt(0).any()),
                                "FinalPositions": frame.Positions.iloc[-1],
                                "FinalPendingReductions": frame.PendingReductions.iloc[-1],
                                "FullyLiquidated": bool(frame.Positions.iloc[-1] == 0)})
                summary.append(metrics)
                if name in ("Group5", "TheoreticalLS"):
                    LOG.info("%s 年化 %.2f%% / Sharpe %.3f / 回撤 %.2f%% / 末日持仓 %d",
                             name, 100 * metrics["AnnualReturn"], metrics["Sharpe"],
                             100 * metrics["MaxDrawdown"], metrics["FinalPositions"])
                    if metrics["HasStaleValuations"]:
                        LOG.warning("%s 含未知行情持仓，绩效依赖最近已知价格估值假设，详见StalePositionDays。", name)
            plot_result(ledger, destination / "nav.png", f"{factor} / {period_name}", cfg, analysis_dates[0])
    result = pd.DataFrame(summary)
    result.to_csv(output / "summary.csv", index=False)
    settings = {name: str(value) if isinstance(value, Path) else value for name, value in vars(cfg).items()}
    settings.update({"factors": list(selected_factors), "periods": selected_periods,
                     "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "signal_start": str(analysis_dates[0].date()), "last_date": str(analysis_dates[-1].date())})
    (output / "run_config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2))
    (output / "说明.txt").write_text(
        "信号按固定交易日历每5/10/20日生成，次日开盘执行；尾部不足整期的仓位于末日开盘尝试平仓。\n"
        "IC使用完整N日连续开盘收益，缺失收益只影响IC配对，不影响信号分组。原始IC与方向调整IC均输出。\n"
        "五个多头组各自初始资金1；理论多空在调仓时目标多头50%、空头50%，非原代码双倍总敞口。\n"
        "费用按每次实际成交额计算，净值图和绩效均来自daily_ledger.csv的同一日收益。\n"
        f"每次买/卖费用率{cfg.one_way_fee:.6f}；年借券费率{cfg.annual_borrow_fee:.4%}；"
        f"开盘方向性跳空过滤阈值{cfg.open_gap_limit:.2%}。\n"
        "数据未提供精确涨跌停、成交队列、借券供应和退市清算；成交过滤是保守近似，理论空头不代表可借到券。\n"
        "连续价格单位相当于将除权权益即时再投资，使用可分割份额，不模拟分红到账、整手和市场冲击。\n"
        "无法增仓的订单撤销并留存现金；无法减仓的订单逐日重试，新调仓指令覆盖旧指令。\n"
        "未知行情沿用最近估值，不能成交；StalePositionDays记录此类敞口，末日受阻仓位仍保留并报告。\n"
        "若股票历史中间有未解释的行情缺口，无法从参考昨收区分除权和漏记涨跌，程序会停止并要求补全数据。\n"
        "Sharpe用算术日超额收益年化，年化收益用实际净值几何年化；最大回撤包含初始资金和每日估值。\n"
        "ICIR仅输出每期均值/标准差，不对序列独立性作未经检验的年化假设。\n"
    )
    LOG.info("输出目录：%s", output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=BASE / "data")
    parser.add_argument("--output-dir", type=Path, default=BASE / "results_code2_new")
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2018-12-31")
    parser.add_argument("--index", choices=["300", "500"], default="500")
    parser.add_argument("--factor", choices=FACTORS, action="append", help="可重复指定；默认全部六个")
    parser.add_argument("--period", choices=list(PERIODS), action="append", help="可重复指定；默认三种期限")
    parser.add_argument("--one-way-fee", type=float, default=0.0015)
    parser.add_argument("--open-gap-limit", type=float, default=0.049)
    parser.add_argument("--annual-borrow-fee", type=float, default=0.0)
    parser.add_argument("--annual-risk-free", type=float, default=0.0)
    parser.add_argument("--with-xgboost", action="store_true", help="增加按月滚动XGBoost及共同样本比较")
    parser.add_argument("--model-only", action="store_true", help="复用本输出目录的预处理因子，仅训练模型及共同样本回测")
    parser.add_argument("--predict-start", default="2017-02-01", help="XGBoost样本外预测起点")
    args = parser.parse_args()
    factors = args.factor or list(FACTORS)
    periods = {name: PERIODS[name] for name in args.period} if args.period else PERIODS
    config_args = {key: value for key, value in vars(args).items()
                   if key not in ("factor", "period", "with_xgboost", "model_only", "predict_start")}
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(**config_args)
    if not args.model_only:
        run_pipeline(cfg, factors, periods)
    if args.with_xgboost or args.model_only:
        from code2_xgboost import run_xgboost
        run_xgboost(cfg, args.predict_start)


if __name__ == "__main__":
    main()
