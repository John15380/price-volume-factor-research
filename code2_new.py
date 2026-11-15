"""量价多因子实验的唯一代码文件。

python code2_new.py --with-xgboost   # 九因子、训练期筛选/调参和十组回测
python code2_new.py --models-only   # 保留单因子账本，重跑全部滚动模型
python code2_new.py --self-test     # 因子、账本和时间隔离自动测试
python code2_new.py --report        # 核验当前结果并生成根目录 report.pdf

T日收盘信号，T+1开盘执行。手续费买卖各0.0015，完整往返0.003。
多头与理论多空分别记账；XGBoost仅使用标签已实现的历史训练样本。
各段在本文件中直接定义，不依赖任何本地Python模块。
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
# JIT编译缓存使用系统临时目录，不在作业目录生成__pycache__。
os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'quant-code2-numba'))


# 配置、行情与逐日持仓费用账本

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
LOG = logging.getLogger('code2_new')
_BASELINE_FACTORS = ('HF_SmartMoney', 'HF_Raw_CPV_Mean', 'HF_CDPDP', 'HF_Skewness', 'HF_Tail_Reversal', 'HF_Amplitude')
PERIODS = {'Weekly': 5, 'BiWeekly': 10, 'Monthly': 20}
CONTROLS = ['Ctrl_Ret20', 'Ctrl_Vol20', 'Ctrl_Turnover20']
KEYS = ['TradingDay', 'SecuCode']

@dataclass
class Config:
    data_dir: Path = BASE / 'data'
    output_dir: Path = BASE / 'results'
    start: str = '2016-01-04'
    end: str = '2018-12-31'
    index: str = '500'
    factor_window: int = 20
    smart_window: int = 10
    annual_days: int = 250
    one_way_fee: float = 0.0015
    open_gap_limit: float = 0.049
    annual_borrow_fee: float = 0.0
    annual_risk_free: float = 0.0
    min_cross_section: int = 20
    ls_leg_weight: float = 1.0
    n_groups: int = 10

def normalize_codes(values):
    return values.astype(str).str.replace('\\.0$', '', regex=True).str.zfill(6)

@njit(cache=True)
def subset_corr(changes, direction):
    x = changes[:-1]
    y = changes[1:]
    keep = (x > 0) & (y > 0) if direction > 0 else (x < 0) & (y < 0)
    a, b = (x[keep], y[keep])
    if len(a) < 3:
        return np.nan
    a, b = (a - np.mean(a), b - np.mean(b))
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return np.sum(a * b) / denom if denom > 1e-15 else np.nan

@njit(cache=True)
def intraday_metrics(open_, close, high, low, volume, amount):
    """按时间排序后的当日分钟数组；Smart Money 使用真实成交额。"""
    n = len(close)
    missing = np.full(6, np.nan)
    if n < 31:
        return missing
    if np.any(~np.isfinite(open_)) or np.any(~np.isfinite(close)) or np.any(~np.isfinite(high)) or np.any(~np.isfinite(low)) or np.any(~np.isfinite(volume)) or np.any(~np.isfinite(amount)) or np.any(open_ <= 0) or np.any(close <= 0) or np.any(high <= 0) or np.any(low <= 0) or np.any(volume < 0) or np.any(amount < 0):
        return missing
    total_vol, total_amt = (np.sum(volume), np.sum(amount))
    if total_vol <= 0 or total_amt <= 0:
        return missing
    scores = np.full(n, -np.inf)
    for i in range(n):
        previous = open_[0] if i == 0 else close[i - 1]
        if volume[i] > 0:
            scores[i] = abs(close[i] / previous - 1) / volume[i] ** 0.25
    remaining, smart_amt = (total_vol * 0.2, 0.0)
    for i in np.argsort(scores)[::-1]:
        if volume[i] <= 0:
            continue
        used = min(remaining, volume[i])
        smart_amt += amount[i] * used / volume[i]
        remaining -= used
        if remaining <= total_vol * 1e-12:
            break
    smart = smart_amt / (total_vol * 0.2) / (total_amt / total_vol)
    p, v = (close - np.mean(close), volume - np.mean(volume))
    denominator = np.sqrt(np.sum(p * p) * np.sum(v * v))
    cpv = np.sum(p * v) / denominator if denominator > 1e-15 else np.nan
    changes = close[1:] - close[:-1]
    cdpdp = subset_corr(changes, 1) + subset_corr(changes, -1)
    returns = close[1:] / close[:-1] - 1
    centered = returns - np.mean(returns)
    variance = np.mean(centered ** 2)
    skew = np.mean(centered ** 3) / variance ** 1.5 if variance > 1e-15 else 0.0
    tail = close[-1] / close[-31] - 1
    amplitude = (np.max(high) - np.min(low)) / open_[0]
    return np.array([smart, cpv, cdpdp, skew, tail, amplitude])

def prepare_daily(raw, calendar, cfg):
    """先在完整个股日历上计算连续价格及风格变量，再筛选股票池。

    scale[t]/scale[t-1] = 最后实际收盘价 / 当日除权参考昨收。
    因此连续收盘收益为 Close[t]/PrevClose[t]-1，不把除权当作亏损。
    当日开盘用同一个当日 scale，未将未来收益乘回当日价格。
    """
    raw = raw.copy()
    raw['TradingDay'] = pd.to_datetime(raw.TradingDay)
    raw['SecuCode'] = normalize_codes(raw.SecuCode)
    if raw.duplicated(KEYS).any():
        raise ValueError('日行情存在重复的股票/日期，不能直接回测。')
    pieces = []
    for symbol, group in raw.groupby('SecuCode', sort=True):
        g = group.set_index('TradingDay').reindex(calendar).copy()
        g.index.name = 'TradingDay'
        g['SecuCode'] = symbol
        open_observed = g.OpenPrice.gt(0) & np.isfinite(g.OpenPrice) & g.PTFlag.eq(0)
        close_observed = g.ClosePrice.gt(0) & np.isfinite(g.ClosePrice) & g.PTFlag.eq(0)
        halted = g.PTFlag.eq(1)
        if (open_observed & ~close_observed).any():
            raise ValueError(f'{symbol} 有开盘却无有效收盘，无法可靠估值，须补全行情。')
        known_state = close_observed | halted
        observed_dates = g.index[close_observed]
        if len(observed_dates):
            interior = (g.index >= observed_dates[0]) & (g.index <= observed_dates[-1])
            if (interior & ~known_state).any():
                raise ValueError(f'{symbol} 历史行情存在未解释缺口，须补全行情或提供复权因子。')
        observed = open_observed | close_observed
        raw_close = g.ClosePrice.where(close_observed)
        prior_actual = raw_close.ffill().shift(1)
        if (observed & prior_actual.notna() & ~g.PrevClosePrice.gt(0)).any():
            raise ValueError(f'{symbol} 缺少有效除权参考昨收，无法建立连续价格。')
        step = (prior_actual / g.PrevClosePrice).where(observed & prior_actual.notna(), 1.0)
        if (~np.isfinite(step) | (step <= 0)).any():
            raise ValueError(f'{symbol} 价格调整因子无效。')
        scale = step.cumprod()
        g['AdjOpen'] = (g.OpenPrice * scale).where(open_observed)
        g['AdjClose'] = (g.ClosePrice * scale).where(close_observed).ffill()
        g['KnownState'] = known_state
        g['StaleQuote'] = ~g.KnownState
        gap = g.OpenPrice / g.PrevClosePrice - 1
        reference_ok = g.PrevClosePrice.gt(0)
        g['CanBuy'] = open_observed & reference_ok & gap.lt(cfg.open_gap_limit)
        g['CanSell'] = open_observed & reference_ok & gap.gt(-cfg.open_gap_limit)
        daily_ret = g.AdjClose.pct_change(fill_method=None)
        daily_ret = daily_ret.where(g.KnownState & g.KnownState.shift(1, fill_value=False))
        complete = g.KnownState.rolling(21, min_periods=21).sum().eq(21)
        g['Ctrl_Ret20'] = (g.AdjClose / g.AdjClose.shift(20) - 1).where(complete)
        g['Ctrl_Vol20'] = daily_ret.rolling(20, min_periods=20).std()
        floats = g.NonRestrictedShares.where(g.NonRestrictedShares.gt(0), g.AFloats)
        turnover = (g.Volume / floats).where(close_observed & floats.gt(0))
        turnover = turnover.mask(halted, 0.0)
        g['Ctrl_Turnover20'] = turnover.rolling(20, min_periods=20).mean()
        pieces.append(g[['SecuCode', 'AdjOpen', 'AdjClose', 'CanBuy', 'CanSell', 'StaleQuote'] + CONTROLS].reset_index())
    if not pieces:
        raise ValueError('筛选后没有日行情。')
    return pd.concat(pieces, ignore_index=True)

class DataLoader:

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg.data_dir)

    def calendar(self):
        path = self.root / '交易日期/tradingdays.csv'
        dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(path).iloc[:, 0])).sort_values().unique()
        dates = dates[dates <= pd.Timestamp(self.cfg.end)]
        if not len(dates) or not (dates >= pd.Timestamp(self.cfg.start)).any():
            raise ValueError('指定区间没有交易日。')
        first = dates.searchsorted(pd.Timestamp(self.cfg.start))
        return dates[max(0, first - 60):]

    def universe(self, calendar):
        code = '000905' if self.cfg.index == '500' else '000300'
        path = self.root / f'指数权重/index_wt_{code}.parquet'
        univ = pd.read_parquet(path, columns=['date', 'symbol'])
        univ = univ.rename(columns={'date': 'TradingDay', 'symbol': 'SecuCode'})
        univ['TradingDay'] = pd.to_datetime(univ.TradingDay)
        univ['SecuCode'] = normalize_codes(univ.SecuCode)
        return univ[univ.TradingDay.isin(calendar)].drop_duplicates(KEYS)

    def quotes(self, calendar, symbols):
        fields = ['SecuCode', 'TradingDay', 'OpenPrice', 'ClosePrice', 'PrevClosePrice', 'PTFlag', 'TradingVolumes', 'NonRestrictedShares', 'AFloats']
        files = sorted((self.root / '日行情').glob('daily_quote_*.csv'))
        parts = []
        for path in files:
            year = int(path.stem.rsplit('_', 1)[1])
            if not calendar[0].year <= year <= calendar[-1].year:
                continue
            for chunk in pd.read_csv(path, usecols=fields, dtype={'SecuCode': str}, chunksize=250000):
                chunk['SecuCode'] = normalize_codes(chunk.SecuCode)
                chunk = chunk[chunk.SecuCode.isin(symbols)].copy()
                chunk['TradingDay'] = pd.to_datetime(chunk.TradingDay)
                parts.append(chunk[chunk.TradingDay.isin(calendar)])
        if not parts:
            raise FileNotFoundError('找不到对应年份日行情。')
        raw = pd.concat(parts, ignore_index=True).rename(columns={'TradingVolumes': 'Volume'})
        return prepare_daily(raw, calendar, self.cfg)

    def industry_cap(self, calendar, symbols):
        industry = pd.read_csv(self.root / '股票行业分类/industry.csv', dtype={'SecuCode': str})
        industry['SecuCode'] = normalize_codes(industry.SecuCode)
        industry = industry[industry.SecuCode.isin(symbols)].copy()
        industry['InfoPublDate'] = pd.to_datetime(industry.InfoPublDate) + pd.Timedelta(days=1)
        industry = industry.sort_values('InfoPublDate')
        dataset = ds.dataset(self.root / '股票市值/stock_cap.parquet', format='parquet')
        mask = ds.field('symbol').isin(list(symbols)) & (ds.field('date') >= str(calendar[0].date())) & (ds.field('date') <= str(calendar[-1].date()))
        cap = dataset.to_table(columns=['date', 'symbol', 'tot_cap'], filter=mask).to_pandas()
        cap = cap.rename(columns={'date': 'TradingDay', 'symbol': 'SecuCode'})
        cap['TradingDay'] = pd.to_datetime(cap.TradingDay)
        cap['SecuCode'] = normalize_codes(cap.SecuCode)
        cap['LnCap'] = np.log(cap.tot_cap.where(cap.tot_cap.gt(0)))
        if cap.duplicated(KEYS).any():
            raise ValueError('市值存在重复的股票/日期。')
        return (industry, cap[KEYS + ['LnCap']])

class FactorEngine:
    """先计算候选股票的完整历史，再按信号日动态成分选股。"""

    def __init__(self, cfg, calendar, symbols):
        self.cfg, self.calendar, self.symbols = (cfg, calendar, set(symbols))
        self.cache = Path(cfg.output_dir) / 'intraday_cache'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.definition_hash = hashlib.sha256((inspect.getsource(intraday_metrics.py_func) + inspect.getsource(subset_corr.py_func)).encode()).hexdigest()

    def run(self):
        all_files = sorted((Path(self.cfg.data_dir) / 'bar_1m_2016_2018').glob('*.parquet'))
        if not all_files:
            raise FileNotFoundError('分钟数据目录没有 parquet 文件。')
        parts = []
        first_hf_date = pd.Timestamp(all_files[0].stem)
        files = [path for path in all_files if pd.Timestamp(path.stem) in self.calendar]
        for path in tqdm(files, desc='分钟因子（日缓存）'):
            day = pd.Timestamp(path.stem)
            metadata = {'definition': self.definition_hash, 'source': str(path.resolve()), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns, 'symbols': sorted(self.symbols)}
            cached = self.cache / path.name
            manifest = cached.with_suffix('.json')
            if cached.exists() and manifest.exists() and (json.loads(manifest.read_text()) == metadata):
                day_factors = pd.read_parquet(cached)
            else:
                dataset = ds.dataset(path, format='parquet')
                columns = ['time', 'symbol', 'open', 'close', 'high', 'low', 'volume', 'amount']
                bars = dataset.to_table(columns=columns, filter=ds.field('symbol').isin(sorted(self.symbols))).to_pandas()
                bars = bars.sort_values(['symbol', 'time'])
                rows = []
                for symbol, g in bars.groupby('symbol', sort=False):
                    if g.time.duplicated().any():
                        raise ValueError(f'{day.date()} {symbol} 分钟时间重复。')
                    metrics = intraday_metrics(*[g[column].to_numpy(dtype=np.float64) for column in ['open', 'close', 'high', 'low', 'volume', 'amount']])
                    rows.append([day, str(symbol).zfill(6), *metrics])
                day_factors = pd.DataFrame(rows, columns=KEYS + list(_BASELINE_FACTORS))
                day_factors.to_parquet(cached, index=False)
                manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
            parts.append(day_factors)
        if not parts:
            raise ValueError('所选日期没有分钟数据。')
        raw = pd.concat(parts, ignore_index=True)
        return (roll_factors(raw, self.calendar, self.cfg), first_hf_date)

def roll_factors(raw, calendar, cfg):
    parts = []
    for symbol, group in raw.groupby('SecuCode', sort=True):
        grid = group.set_index('TradingDay')[list(_BASELINE_FACTORS)].reindex(calendar)
        rolling = grid.rolling(cfg.factor_window, min_periods=cfg.factor_window).mean()
        rolling[_BASELINE_FACTORS[0]] = grid[_BASELINE_FACTORS[0]].rolling(cfg.smart_window, min_periods=cfg.smart_window).mean()
        rolling.iloc[:cfg.factor_window - 1] = np.nan
        rolling['SecuCode'] = symbol
        rolling.index.name = 'TradingDay'
        parts.append(rolling.reset_index())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEYS + list(_BASELINE_FACTORS))

def process_features(factors, universe, daily, industry, cap, cfg):
    full = factors.merge(universe, on=KEYS, how='inner', validate='one_to_one')
    full = full.merge(daily[KEYS + CONTROLS], on=KEYS, how='left', validate='one_to_one')
    full = full.merge(cap, on=KEYS, how='left', validate='one_to_one')
    full = pd.merge_asof(full.sort_values('TradingDay'), industry[['SecuCode', 'InfoPublDate', 'tag']], left_on='TradingDay', right_on='InfoPublDate', by='SecuCode', direction='backward')
    full['Industry'] = full.tag.fillna('Unknown')
    processed = []
    for day, group in tqdm(full.groupby('TradingDay', sort=True), desc='截面中性化'):
        out = group[KEYS].copy()
        for factor in _BASELINE_FACTORS:
            controls = CONTROLS if 'CPV' in factor else []
            valid = group[[factor, 'LnCap'] + controls].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
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
            x = pd.concat([industries, sub[['LnCap'] + controls]], axis=1).astype(float)
            fit = LinearRegression().fit(x, y)
            residual = y - fit.predict(x)
            std = np.std(residual)
            if std > 1e-12:
                out.loc[sub.index, factor] = (residual - np.mean(residual)) / std
        processed.append(out)
    if not processed:
        raise ValueError('没有可用因子截面。')
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
    increase: np.ndarray | None = None

    @classmethod
    def from_daily(cls, daily, dates):
        symbols = pd.Index(sorted(daily.SecuCode.unique()), name='SecuCode')

        def matrix(column):
            return daily.pivot(index='TradingDay', columns='SecuCode', values=column).reindex(index=dates, columns=symbols)
        return cls(pd.DatetimeIndex(dates), symbols, matrix('AdjOpen').to_numpy(float), matrix('AdjClose').to_numpy(float), matrix('CanBuy').fillna(False).to_numpy(bool), matrix('CanSell').fillna(False).to_numpy(bool), matrix('StaleQuote').fillna(True).to_numpy(bool), matrix('CanIncrease').fillna(False).to_numpy(bool) if 'CanIncrease' in daily else None)

def signal_groups(features, factor, calendar, period_days, min_stocks=20, low_is_long=True, n_groups=10):
    """独立交易日历调仓；先按当时因子分组，不读取任何未来收益。"""
    schedule = pd.DatetimeIndex(calendar)[::period_days]
    frames = []
    for day in schedule:
        sub = features.loc[features.TradingDay.eq(day), KEYS + [factor]].dropna(subset=[factor]).copy()
        if len(sub) < max(min_stocks, n_groups) or sub[factor].nunique() < n_groups:
            continue
        sub = sub.sort_values([factor, 'SecuCode'], ascending=[not low_is_long, True])
        sub['Group'] = np.arange(len(sub)) * n_groups // len(sub) + 1
        frames.append(sub)
    columns = KEYS + [factor, 'Group']
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns), schedule)

def calc_ic(groups, market, factor, days, low_is_long=True):
    rows = []
    sign = -1 if low_is_long else 1
    for day, sub in groups.groupby('TradingDay', sort=True):
        t = market.dates.get_indexer([day])[0]
        if t < 0 or t + days + 1 >= len(market.dates):
            continue
        columns = market.symbols.get_indexer(sub.SecuCode)
        if (columns < 0).any():
            raise ValueError('因子股票没有对应行情列。')
        buy = market.open[t + 1, columns]
        sell = market.open[t + days + 1, columns]
        valid = np.isfinite(buy) & (buy > 0) & np.isfinite(sell) & (sell > 0)
        values = sub[factor].to_numpy(float)[valid]
        rets = sell[valid] / buy[valid] - 1
        pearson = rank = np.nan
        if len(rets) >= 10 and np.std(values) > 1e-12 and (np.std(rets) > 1e-12):
            pearson = pd.Series(values).corr(pd.Series(rets))
            rank = pd.Series(values).corr(pd.Series(rets), method='spearman')
        rows.append({'SignalDate': day, 'EntryDate': market.dates[t + 1], 'ExitDate': market.dates[t + days + 1], 'N_Signal': len(sub), 'N_Paired': int(valid.sum()), 'Coverage': valid.mean(), 'IC_Raw': pearson, 'RankIC_Raw': rank, 'IC_Directed': sign * pearson, 'RankIC_Directed': sign * rank})
    return pd.DataFrame(rows, columns=['SignalDate', 'EntryDate', 'ExitDate', 'N_Signal', 'N_Paired', 'Coverage', 'IC_Raw', 'RankIC_Raw', 'IC_Directed', 'RankIC_Directed'])

class Portfolio:
    """连续价格单位下的现金/持仓账本，调仓外只重试未成交的减仓单。"""

    def __init__(self, symbols, cfg, name):
        self.symbols, self.cfg, self.name = (symbols, cfg, name)
        self.units = np.zeros(len(symbols))
        self.cash = 1.0
        self.pending = np.full(len(symbols), np.nan)
        self.trades = []

    def value(self, prices):
        held = np.abs(self.units) > 1e-14
        if np.any(held & (~np.isfinite(prices) | (prices <= 0))):
            raise ValueError('已有持仓缺少历史估值，不能凭空删除头寸。')
        return self.cash + np.sum(self.units[held] * prices[held])

    def trade(self, delta, prices, day):
        active = np.flatnonzero(np.abs(delta) > 1e-14)
        notionals = np.abs(delta[active] * prices[active])
        costs = notionals * self.cfg.one_way_fee
        self.cash -= np.sum(delta[active] * prices[active]) + costs.sum()
        self.units += delta
        self.units[np.abs(self.units) < 1e-14] = 0
        for i, notional, cost in zip(active, notionals, costs):
            self.trades.append({'TradingDay': day, 'Portfolio': self.name, 'SecuCode': self.symbols[i], 'DeltaUnits': delta[i], 'AdjPrice': prices[i], 'Notional': notional, 'Cost': cost})
        return (notionals.sum(), costs.sum())

    def execute(self, prices, can_buy, can_sell, day, weights=None, can_increase=None):
        equity = self.value(prices)
        gross_limit = 2 * self.cfg.ls_leg_weight if self.name == 'TheoreticalLS' else 1.0
        if equity <= 0:
            raise ValueError(f'{day} {self.name} 净资产非正，停止回测。')
        if weights is not None:
            if np.sum(np.abs(weights)) > gross_limit + 1e-10:
                raise ValueError('目标总敞口超过当前组合约定上限。')
            target = np.divide(weights * equity, prices, out=np.zeros_like(weights), where=np.isfinite(prices) & (prices > 0))
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
        admission_blocked = 0
        if weights is not None:
            add = target - self.units
            same_sign = (self.units == 0) | (self.units * add > 0)
            add[~same_sign] = 0
            admission = np.ones(len(add), dtype=bool) if can_increase is None else can_increase
            admission_blocked = int(((np.abs(add) > 1e-14) & ~admission).sum())
            allowed = np.where(add > 0, can_buy, can_sell) & admission
            blocked += int(((np.abs(add) > 1e-14) & ~allowed).sum())
            add[~allowed] = 0
            valid_prices = np.nan_to_num(prices, nan=0.0)
            gross = np.sum(np.abs(self.units * valid_prices))
            additional = np.sum(np.abs(add * valid_prices))
            room = max(self.value(prices) * gross_limit - gross, 0.0)
            scale = min(1.0, room / (additional * (1 + gross_limit * self.cfg.one_way_fee))) if additional > 0 else 0.0
            cash_needed = np.sum(add * valid_prices) + additional * self.cfg.one_way_fee
            if cash_needed > 0:
                scale = min(scale, max(self.cash, 0.0) / cash_needed)
            n, c = self.trade(add * scale, prices, day)
            notional, cost = (notional + n, cost + c)
        return {'TradedNotional': notional, 'TradingCost': cost, 'Turnover': notional / equity, 'BlockedOrders': blocked, 'AdmissionBlockedOrders': admission_blocked}

def backtest(groups, schedule, market, cfg):
    n = len(market.symbols)
    if cfg.n_groups < 2:
        raise ValueError('分组数至少为2。')
    books = [Portfolio(market.symbols, cfg, f'Group{i}') for i in range(1, cfg.n_groups + 1)]
    books.append(Portfolio(market.symbols, cfg, 'TheoreticalLS'))
    group_map = {day: sub for day, sub in groups.groupby('TradingDay')}
    dates = market.dates
    first_signal = dates.get_indexer([schedule[0]])[0]
    if first_signal < 0 or first_signal + 2 >= len(dates):
        raise ValueError('至少需要信号日及后续两个交易日。')
    execution = {dates[dates.get_loc(day) + 1]: day for day in schedule if day in dates and dates.get_loc(day) + 1 < len(dates) - 1}
    rows = []
    previous_nav = {book.name: 1.0 for book in books}
    for t in range(first_signal + 1, len(dates)):
        day = dates[t]
        previous_close = market.close[t - 1]
        opening = np.where(np.isfinite(market.open[t]), market.open[t], previous_close)
        closing = np.where(np.isfinite(market.close[t]), market.close[t], previous_close)
        new_weights = None
        if day in execution:
            new_weights = np.zeros((cfg.n_groups + 1, n))
            sub = group_map.get(execution[day])
            if sub is not None:
                for group in range(1, cfg.n_groups + 1):
                    columns = market.symbols.get_indexer(sub.loc[sub.Group.eq(group), 'SecuCode'])
                    if (columns < 0).any():
                        raise ValueError('信号股票不在行情矩阵中。')
                    if len(columns):
                        new_weights[group - 1, columns] = 1 / len(columns)
                new_weights[cfg.n_groups] = cfg.ls_leg_weight * (new_weights[cfg.n_groups - 1] - new_weights[0])
        if t == len(dates) - 1:
            new_weights = np.zeros((cfg.n_groups + 1, n))
        for b, book in enumerate(books):
            borrow = np.sum(np.maximum(-book.units, 0) * np.nan_to_num(previous_close))
            borrow *= cfg.annual_borrow_fee / cfg.annual_days
            book.cash -= borrow
            stats = book.execute(opening, market.buy[t], market.sell[t], day, None if new_weights is None else new_weights[b], None if market.increase is None else market.increase[t])
            nav = book.value(closing)
            if nav <= 0:
                raise ValueError(f'{day} {book.name} 日末净资产非正。')
            held = np.abs(book.units) > 1e-14
            values = book.units * np.nan_to_num(closing)
            rows.append({'TradingDay': day, 'Portfolio': book.name, 'NAV': nav, 'Return': nav / previous_nav[book.name] - 1, 'Cash': book.cash, 'LongValue': np.maximum(values, 0).sum(), 'ShortValue': np.maximum(-values, 0).sum(), 'Positions': int(held.sum()), 'StalePositions': int((held & market.stale[t]).sum()), 'PendingReductions': int(np.isfinite(book.pending).sum()), 'BorrowCost': borrow, **stats})
            previous_nav[book.name] = nav
    columns = ['TradingDay', 'Portfolio', 'SecuCode', 'DeltaUnits', 'AdjPrice', 'Notional', 'Cost']
    trades = pd.DataFrame([trade for book in books for trade in book.trades], columns=columns)
    return (pd.DataFrame(rows), trades)

def performance(returns, cfg):
    values = np.asarray(returns, dtype=float)
    if not len(values) or not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError('绩效计算需要有限且大于-100%的日收益。')
    nav = np.r_[1.0, np.cumprod(1 + values)]
    vol = np.std(values, ddof=1) * np.sqrt(cfg.annual_days) if len(values) > 1 else 0.0
    risk_free_daily = (1 + cfg.annual_risk_free) ** (1 / cfg.annual_days) - 1
    excess = values - risk_free_daily
    deviation = np.std(excess, ddof=1) if len(values) > 1 else 0.0
    sharpe = np.mean(excess) / deviation * np.sqrt(cfg.annual_days) if deviation > 1e-12 else np.nan
    return {'Days': len(values), 'FinalNAV': nav[-1], 'TotalReturn': nav[-1] - 1, 'AnnualReturn': nav[-1] ** (cfg.annual_days / len(values)) - 1, 'AnnualVol': vol, 'Sharpe': sharpe, 'MaxDrawdown': np.min(nav / np.maximum.accumulate(nav) - 1)}

def pyplot(cfg):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(cfg.output_dir) / 'matplotlib_cache'))
    os.environ.setdefault('XDG_CACHE_HOME', str(Path(cfg.output_dir).resolve() / 'system_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt

def plot_factor_correlation(features, cfg):
    """逐日截面Spearman相关系数等权平均，避免混合日期引入时间效应。"""
    total = np.zeros((len(_BASELINE_FACTORS), len(_BASELINE_FACTORS)))
    count = np.zeros_like(total)
    for _, frame in features.groupby('TradingDay', sort=False):
        corr = frame[list(_BASELINE_FACTORS)].corr(method='spearman', min_periods=cfg.min_cross_section).to_numpy()
        finite = np.isfinite(corr)
        total += np.where(finite, corr, 0)
        count += finite
    average = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    pd.DataFrame(average, index=_BASELINE_FACTORS, columns=_BASELINE_FACTORS).to_csv(Path(cfg.output_dir) / 'factor_correlation.csv')
    plt = pyplot(cfg)
    fig, axis = plt.subplots(figsize=(9, 8))
    im = axis.imshow(average, vmin=-1, vmax=1, cmap='coolwarm')
    labels = [factor.removeprefix('HF_') for factor in _BASELINE_FACTORS]
    axis.set_xticks(range(len(labels)), labels, rotation=40, ha='right')
    axis.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isfinite(average[i, j]):
                axis.text(j, i, f'{average[i, j]:.2f}', ha='center', va='center')
    axis.set_title('Mean daily cross-sectional Spearman correlation')
    fig.colorbar(im, ax=axis)
    fig.tight_layout()
    fig.savefig(Path(cfg.output_dir) / 'factor_correlation.png', dpi=160)
    plt.close(fig)

def plot_result(ledger, destination, title, cfg, baseline_date):
    plt = pyplot(cfg)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for name, frame in ledger.groupby('Portfolio', sort=True):
        nav = pd.Series(frame.NAV.to_numpy(), index=pd.to_datetime(frame.TradingDay))
        nav = pd.concat([pd.Series([1.0], index=[baseline_date]), nav])
        axis = axes[1] if name == 'TheoreticalLS' else axes[0]
        axis.plot(nav.index, nav.values, label=name)
    for axis in axes:
        axis.axhline(1, color='gray', linewidth=0.5)
        axis.set_ylabel('Net NAV')
        axis.legend()
        axis.grid(alpha=0.3)
    axes[0].set_title(title + ' | long-only groups')
    axes[1].set_title(f'Theoretical L/S: {cfg.ls_leg_weight:.0%} long / {cfg.ls_leg_weight:.0%} short targets')
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)

# 历史交易状态、证券代码和官方复权

from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
TRADING_ALIASES = {'601313': '601360', '000022': '001872'}
TRADING_CODE_CHANGES = {'601360': ('601313', '2018-02-28'), '001872': ('000022', '2018-12-26')}
TRADING_ALIAS_SOURCES = {'601360': 'https://static.cninfo.com.cn/finalpage/2018-02-28/1204434951.PDF', '001872': 'https://static.cninfo.com.cn/finalpage/2018-12-26/1205690369.PDF'}

def trading_canonical(values):
    return normalize_codes(values).replace(TRADING_ALIASES)

def trading_raw_symbols(symbols):
    return sorted(set(symbols) | {old for old, new in TRADING_ALIASES.items() if new in set(symbols)})

def trading_trading_code(dates, symbols):
    result = pd.Series(symbols).astype(str).copy()
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    result = result.reset_index(drop=True)
    for new, (old, start) in TRADING_CODE_CHANGES.items():
        result.loc[result.eq(new) & dates.lt(pd.Timestamp(start))] = old
    return result.to_numpy()

def trading_suspension_at_open(timing):
    if pd.isna(timing) or not str(timing).strip():
        return True
    intervals = re.findall('(\\d{1,2}):(\\d{2})\\s*-\\s*(\\d{1,2}):(\\d{2})', str(timing))
    if not intervals:
        raise ValueError(f'无法识别停牌时段：{timing}')
    opening = 9 * 60 + 30
    return any((int(h1) * 60 + int(m1) <= opening < int(h2) * 60 + int(m2) for h1, m1, h2, m2 in intervals))

def trading_deduplicate(frame, keys, value_columns, name, absolute_tolerance=1e-08):
    """代码更名历史回填可能产生同一证券两条相同记录，冲突时停止。"""
    if frame.duplicated(keys).any():
        duplicate = frame[frame.duplicated(keys, keep=False)]
        for column in value_columns:
            spread = duplicate.groupby(keys)[column].agg(['min', 'max'])
            if pd.api.types.is_numeric_dtype(frame[column]):
                conflicts = (spread['max'] - spread['min']).abs() > absolute_tolerance
            else:
                conflicts = spread['min'].ne(spread['max'])
            if conflicts.any():
                raise ValueError(f'{name}代码归并后有冲突记录：{column}')
        frame = frame.drop_duplicates(keys, keep='last')
    return frame

def trading_input_manifest(data_dir):
    root = Path(data_dir)
    paths = [root / 'ST情况/is_st.csv', root / '停复牌情况/is_suspend.csv']
    paths += sorted((root / '涨跌停价').glob('*.csv'))
    paths += sorted((root / '复权因子').glob('*.csv'))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    records = [{'path': str(path.resolve()), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns} for path in paths]
    return {'files': records, 'signature': hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}

def trading_load_tables(data_dir, calendar, symbols):
    root = Path(data_dir)
    wanted = set(symbols)

    def read(path):
        frame = pd.read_csv(path, dtype={'symbol': str})
        frame = frame.rename(columns={'symbol': 'SecuCode', 'trade_date': 'TradingDay'})
        original = normalize_codes(frame.SecuCode)
        frame.SecuCode = trading_canonical(frame.SecuCode)
        frame['_CanonicalSource'] = original.eq(frame.SecuCode)
        frame.TradingDay = pd.to_datetime(frame.TradingDay)
        return frame[frame.SecuCode.isin(wanted) & frame.TradingDay.isin(calendar)].copy()
    st = read(root / 'ST情况/is_st.csv')
    st = st[KEYS].drop_duplicates().assign(IsST=True)
    suspend = read(root / '停复牌情况/is_suspend.csv')
    suspend['SuspendAtOpen'] = suspend.suspend_timing.map(trading_suspension_at_open)
    suspend['SuspendAny'] = True
    suspend['SuspendFullDay'] = suspend.suspend_timing.isna() | suspend.suspend_timing.astype(str).str.strip().eq('')
    suspend = suspend.groupby(KEYS, as_index=False)[['SuspendAtOpen', 'SuspendAny', 'SuspendFullDay']].max()
    limits, factors = ([], [])
    for year in range(calendar[0].year, calendar[-1].year + 1):
        lp = root / f'涨跌停价/A_share_limit_{year}.csv'
        ap = root / f'复权因子/adj_factor_{year}.csv'
        if not lp.exists() or not ap.exists():
            raise FileNotFoundError(f'{year}年实际涨跌停价或复权因子缺失。')
        limits.append(read(lp))
        factors.append(read(ap))
    limits = pd.concat(limits, ignore_index=True).sort_values('_CanonicalSource')
    factors = pd.concat(factors, ignore_index=True).sort_values('_CanonicalSource')
    limits = trading_deduplicate(limits, KEYS, ['up_limit', 'down_limit'], '涨跌停价').drop(columns='_CanonicalSource')
    factors = trading_deduplicate(factors, KEYS, ['adj_factor'], '复权因子', absolute_tolerance=0.0005001).drop(columns='_CanonicalSource')
    if (factors.adj_factor.le(0) | ~np.isfinite(factors.adj_factor)).any():
        raise ValueError('官方复权因子必须为正且有限。')
    limits['InvalidLimitSource'] = (limits.up_limit < limits.down_limit) | limits.up_limit.le(0) | limits.down_limit.le(0) | ~np.isfinite(limits.up_limit) | ~np.isfinite(limits.down_limit)
    return (st, suspend, limits, factors)

def trading_prepare_daily(raw, calendar, cfg, first_seen, all_calendar, tables):
    """官方因子构造连续价格；信号池与开盘执行状态分别生成。"""
    raw = raw.copy()
    raw.SecuCode = trading_canonical(raw.SecuCode)
    raw.TradingDay = pd.to_datetime(raw.TradingDay)
    raw = trading_deduplicate(raw, KEYS, ['OpenPrice', 'ClosePrice', 'PrevClosePrice', 'Volume', 'Amount'], '日行情')
    st, suspend, limits, factors = tables
    merged = raw.merge(st, on=KEYS, how='left', validate='one_to_one')
    merged = merged.merge(suspend, on=KEYS, how='left', validate='one_to_one')
    merged = merged.merge(limits, on=KEYS, how='left', validate='one_to_one')
    merged = merged.merge(factors, on=KEYS, how='left', validate='one_to_one')
    for column in ['IsST', 'SuspendAtOpen', 'SuspendAny', 'SuspendFullDay']:
        merged[column] = merged[column].eq(True)
    state_indexes = [(table.set_index(['SecuCode', 'TradingDay']).sort_index(), columns, set(table.SecuCode)) for table, columns in [(st, ['IsST']), (suspend, ['SuspendAtOpen', 'SuspendAny', 'SuspendFullDay'])]]
    factor_index = factors.set_index(['SecuCode', 'TradingDay']).adj_factor.sort_index()
    factor_symbols = set(factors.SecuCode)
    parts = []
    for symbol, group in merged.groupby('SecuCode', sort=True):
        g = group.set_index('TradingDay').reindex(calendar).copy()
        g.index.name = 'TradingDay'
        for indexed, columns, known_symbols in state_indexes:
            t = indexed.loc[symbol] if symbol in known_symbols else pd.DataFrame(columns=columns, index=pd.DatetimeIndex([]))
            for column in columns:
                g[column] = t[column].reindex(calendar).eq(True)
        factor = factor_index.loc[symbol].reindex(calendar) if symbol in factor_symbols else pd.Series(np.nan, index=calendar)
        observed_close = g.ClosePrice.gt(0) & np.isfinite(g.ClosePrice) & g.PTFlag.eq(0)
        if (observed_close & factor.isna()).any():
            bad = g.index[observed_close & factor.isna()][0]
            raise ValueError(f'{symbol} {bad.date()}正常报价缺少官方复权因子。')
        factor_asof = factor.ffill()
        observed_factors = factor.dropna()
        scale = factor_asof / observed_factors.iloc[0] if len(observed_factors) else factor_asof
        g['Scale'] = scale
        g['AdjFactor'] = factor_asof
        g['AdjFactorExact'] = factor.notna()
        full_halt = g.SuspendFullDay | g.PTFlag.eq(1)
        halt_open = g.SuspendAtOpen | g.PTFlag.eq(1)
        known = observed_close | full_halt
        quoted_open = g.OpenPrice.gt(0) & np.isfinite(g.OpenPrice) & g.PTFlag.eq(0) & ~halt_open
        g['AdjOpen'] = (g.OpenPrice * scale).where(quoted_open)
        g['AdjClose'] = (g.ClosePrice * scale).where(observed_close).ffill()
        g['StaleQuote'] = ~known
        bounds = g.up_limit.gt(0) & g.down_limit.gt(0) & np.isfinite(g.up_limit) & np.isfinite(g.down_limit) & g.up_limit.ge(g.down_limit)
        g['LimitKnown'] = bounds
        in_range = bounds & (g.OpenPrice >= g.down_limit - 1e-06) & (g.OpenPrice <= g.up_limit + 1e-06)
        g['OpenWithinBounds'] = in_range
        g['CanBuy'] = quoted_open & in_range & (g.OpenPrice < g.up_limit - 1e-06)
        g['CanSell'] = quoted_open & in_range & (g.OpenPrice > g.down_limit + 1e-06)
        g['CanIncrease'] = ~g.IsST
        positions = all_calendar.get_indexer(calendar)
        first_position = all_calendar.searchsorted(first_seen[symbol])
        g['Age'] = positions - first_position + 1
        g['SignalEligible'] = observed_close & ~g.IsST & ~g.SuspendAny & ~g.PTFlag.eq(1) & g.Age.ge(60)
        returns = g.AdjClose.pct_change(fill_method=None).where(known & known.shift(1, fill_value=False))
        complete = known.rolling(21, min_periods=21).sum().eq(21)
        g['Ctrl_Ret20'] = (g.AdjClose / g.AdjClose.shift(20) - 1).where(complete)
        g['Ctrl_Vol20'] = returns.rolling(20, min_periods=20).std()
        floats = g.NonRestrictedShares.where(g.NonRestrictedShares.gt(0), g.AFloats)
        turnover = (g.Volume / floats).where(observed_close & floats.gt(0)).mask(full_halt, 0.0)
        g['Ctrl_Turnover20'] = turnover.rolling(20, min_periods=20).mean()
        g['DailyReturn'] = returns
        g['DailyAmplitude'] = (g.HighPrice / g.LowPrice - 1).where(g.LowPrice.gt(0) & observed_close)
        g['SecuCode'] = symbol
        g['TradingCode'] = trading_trading_code(calendar, pd.Series(symbol, index=calendar))
        parts.append(g.reset_index())
    daily = pd.concat(parts, ignore_index=True)
    active = daily.PTFlag.eq(0) & daily.OpenPrice.gt(0) & daily.ClosePrice.gt(0)
    diagnostics = {'rows': len(daily), 'normal_quoted_rows': int(active.sum()), 'st_stock_days': int(daily.IsST.sum()), 'suspend_stock_days': int(daily.SuspendAny.sum()), 'intraday_only_suspend_stock_days': int((daily.SuspendAny & ~daily.SuspendFullDay).sum()), 'open_suspended_stock_days': int(daily.SuspendAtOpen.sum()), 'normal_quotes_missing_limits': int((active & ~daily.LimitKnown).sum()), 'invalid_limit_source_rows': int(limits.InvalidLimitSource.sum()) if 'InvalidLimitSource' in limits else 0, 'normal_quotes_missing_adjustment': int((active & ~daily.AdjFactorExact).sum()), 'normal_quotes_outside_supplied_limits': int((active & daily.LimitKnown & ~daily.OpenWithinBounds).sum()), 'open_at_upper': int((active & daily.LimitKnown & (daily.OpenPrice >= daily.up_limit - 1e-06)).sum()), 'open_at_lower': int((active & daily.LimitKnown & (daily.OpenPrice <= daily.down_limit + 1e-06)).sum()), 'suspend_records_with_positive_open': int((daily.SuspendAtOpen & active).sum()), 'security_aliases': TRADING_ALIASES, 'alias_sources': TRADING_ALIAS_SOURCES, 'input_files': trading_input_manifest(cfg.data_dir)}
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'trading_data_diagnostics.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    daily.loc[active & ~daily.LimitKnown, KEYS + ['OpenPrice', 'IsST']].to_csv(output / 'missing_limits.csv', index=False)
    daily.loc[active & daily.LimitKnown & ~daily.OpenWithinBounds, KEYS + ['OpenPrice', 'up_limit', 'down_limit', 'Age']].to_csv(output / 'inconsistent_limits.csv', index=False)
    return daily

def trading_industry_cap(loader, calendar, symbols):
    industry, cap = loader.industry_cap(calendar, trading_raw_symbols(symbols))
    industry.SecuCode = trading_canonical(industry.SecuCode)
    cap.SecuCode = trading_canonical(cap.SecuCode)
    cap = trading_deduplicate(cap, KEYS, ['LnCap'], '市值')
    industry = industry.drop_duplicates(['SecuCode', 'InfoPublDate', 'tag']).sort_values('InfoPublDate')
    return (industry, cap)

# 研报因子与截面预处理

from collections import defaultdict, deque
from dataclasses import replace
import hashlib
import inspect
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from numba import njit
from tqdm import tqdm
RESEARCH_LOG = logging.getLogger('code2_research')
RESEARCH_PRIMARY = ('HF_SmartMoney', 'HF_CPV', 'HF_CDPDP', 'HF_IdealAmplitude', 'HF_IdealReversal', 'HF_CDPP')
RESEARCH_ADDITIONAL = ('HF_CDPDP4', 'HF_LowAmpMomentum', 'HF_Skewness')
RESEARCH_FACTORS = RESEARCH_PRIMARY + RESEARCH_ADDITIONAL
RESEARCH_LOW_IS_LONG = {factor: factor != 'HF_LowAmpMomentum' for factor in RESEARCH_FACTORS}
RESEARCH_DAY_FIELDS = ('PV', 'PP', 'PM', 'MP', 'MM', 'DP_Pos_P', 'DP_Neg_P', 'Skew', 'Tail', 'OrdinaryAmp', 'SmartDaily', 'AvgTrade')

@njit(cache=True)
def research_correlation(x, y):
    if len(x) < 3:
        return 0.0
    a, b = (x - np.mean(x), y - np.mean(y))
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return np.sum(a * b) / denom if denom > 1e-14 else 0.0

@njit(cache=True)
def research_smart_ratio(scores, volume, amount):
    total_v, total_a = (volume.sum(), amount.sum())
    if total_v <= 0 or total_a <= 0:
        return np.nan
    target, used, selected_amount = (0.2 * total_v, 0.0, 0.0)
    for i in np.argsort(scores)[::-1]:
        if volume[i] <= 0:
            continue
        take = min(volume[i], target - used)
        selected_amount += amount[i] * take / volume[i]
        used += take
        if used >= target - total_v * 1e-12:
            break
    return selected_amount / target / (total_a / total_v)

@njit(cache=True)
def research_day_metrics(open_, close, high, low, volume, amount, trades, scale):
    n = len(close)
    if n < 31 or volume.sum() <= 0 or amount.sum() <= 0:
        return (np.full(len(RESEARCH_DAY_FIELDS), np.nan), np.empty(0), np.empty(0), np.empty(0))
    adjusted_volume = volume / scale
    score = np.full(n, -np.inf)
    for i in range(n):
        prior = open_[0] if i == 0 else close[i - 1]
        if adjusted_volume[i] > 0:
            score[i] = abs(close[i] / prior - 1) / adjusted_volume[i] ** 0.25
    changes = close[1:] - close[:-1]
    x, y = (changes[:-1], changes[1:])
    pp, pm = ((x > 0) & (y > 0), (x > 0) & (y < 0))
    mp, mm = ((x < 0) & (y > 0), (x < 0) & (y < 0))
    next_price = close[2:]
    positive, negative = (x > 0, x < 0)
    returns = close[1:] / close[:-1] - 1
    centered = returns - returns.mean()
    variance = np.mean(centered ** 2)
    skew = np.mean(centered ** 3) / variance ** 1.5 if variance > 1e-15 else 0.0
    count = trades.sum()
    values = np.array([research_correlation(close, volume), research_correlation(x[pp], y[pp]), research_correlation(x[pm], y[pm]), research_correlation(x[mp], y[mp]), research_correlation(x[mm], y[mm]), research_correlation(x[positive], next_price[positive]), research_correlation(x[negative], next_price[negative]), skew, close[-1] / close[-31] - 1, (high.max() - low.min()) / open_[0], research_smart_ratio(score, adjusted_volume, amount), amount.sum() / count if count > 0 else np.nan])
    return (values, score, adjusted_volume, amount)

@njit(cache=True)
def research_conditional_factors(prices, amplitude, returns, avg_trade, ideal_valid=None):
    n = len(prices)
    out = np.full((n, 3), np.nan)
    for t in range(19, n):
        a, p = (amplitude[t - 19:t + 1], prices[t - 19:t + 1])
        r, v = (returns[t - 19:t + 1], avg_trade[t - 19:t + 1])
        valid = np.isfinite(a) & np.isfinite(p) & (p > 0) & (a >= 0)
        if ideal_valid is not None:
            valid &= ideal_valid[t - 19:t + 1]
        count = valid.sum()
        if count >= 10:
            valid_a, valid_p = (a[valid], p[valid])
            order = np.argsort(valid_p)
            k = count // 4
            out[t, 0] = np.mean(valid_a[order[-k:]]) - np.mean(valid_a[order[:k]])
        if np.isfinite(r).all() and np.isfinite(v).all():
            order = np.argsort(v)
            out[t, 1] = np.sum(r[order[-10:]]) - np.sum(r[order[:10]])
        if t >= 159:
            a160, r160 = (amplitude[t - 159:t + 1], returns[t - 159:t + 1])
            if np.isfinite(a160).all() and np.isfinite(r160).all():
                order = np.argsort(a160)
                out[t, 2] = np.sum(r160[order[:112]])
    return out

def research_ideal_valid_days(quotes):
    """信号形成前可知的有效日，独立于次日订单能否成交。"""
    high, low, close = (quotes.HighPrice, quotes.LowPrice, quotes.ClosePrice)
    upper, lower = (quotes.up_limit, quotes.down_limit)
    same_price = (high - low).abs().le(1e-06)
    one_price_limit = same_price & ((close - upper).abs().le(1e-06) | (close - lower).abs().le(1e-06))
    valid = np.isfinite(high) & np.isfinite(low) & np.isfinite(close) & low.gt(0) & high.ge(low) & close.gt(0) & quotes.Volume.gt(0) & quotes.PTFlag.eq(0) & ~quotes.SuspendAny & ~one_price_limit
    return valid

def research_load_data(cfg, universe_name):
    loader = DataLoader(cfg)
    all_calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(Path(cfg.data_dir) / '交易日期/tradingdays.csv').TradingDay)).sort_values()
    all_calendar = all_calendar[all_calendar <= pd.Timestamp(cfg.end)]
    first = all_calendar.searchsorted(pd.Timestamp(cfg.start))
    calendar = all_calendar[max(0, first - 200):]
    universe = loader.universe(calendar) if universe_name == '500' else None
    if universe is not None:
        universe.SecuCode = trading_canonical(universe.SecuCode)
        universe = universe.drop_duplicates(KEYS)
    wanted = set(universe.SecuCode.unique()) if universe is not None else None
    first_seen, parts = ({}, [])
    fields = ['SecuCode', 'TradingDay', 'OpenPrice', 'ClosePrice', 'PrevClosePrice', 'HighPrice', 'LowPrice', 'PTFlag', 'TradingVolumes', 'TurnoverValue', 'NonRestrictedShares', 'AFloats']
    for path in sorted((Path(cfg.data_dir) / '日行情').glob('daily_quote_*.csv')):
        year = int(path.stem.rsplit('_', 1)[-1])
        if year > calendar[-1].year:
            continue
        columns = fields if year >= calendar[0].year else KEYS + ['ClosePrice']
        for chunk in pd.read_csv(path, usecols=columns, dtype={'SecuCode': str}, chunksize=250000):
            chunk.SecuCode = trading_canonical(chunk.SecuCode)
            if wanted is not None:
                chunk = chunk[chunk.SecuCode.isin(wanted)].copy()
            else:
                chunk = chunk[chunk.SecuCode.str.startswith(('000', '001', '002', '003', '300', '600', '601', '603', '605', '688'))].copy()
            chunk.TradingDay = pd.to_datetime(chunk.TradingDay)
            for symbol, date in chunk.loc[chunk.ClosePrice.gt(0)].groupby('SecuCode').TradingDay.min().items():
                first_seen[symbol] = min(first_seen.get(symbol, date), date)
            if year >= calendar[0].year:
                parts.append(chunk[chunk.TradingDay.isin(calendar)])
    raw = pd.concat(parts, ignore_index=True).rename(columns={'TradingVolumes': 'Volume', 'TurnoverValue': 'Amount'})
    symbols = sorted(raw.SecuCode.unique())
    tables = trading_load_tables(cfg.data_dir, calendar, symbols)
    daily = trading_prepare_daily(raw, calendar, cfg, first_seen, all_calendar, tables)
    daily.to_parquet(Path(cfg.output_dir) / 'daily_market.parquet', index=False)
    eligible = daily.loc[daily.SignalEligible, KEYS].copy()
    if universe is not None:
        eligible = eligible.merge(universe, on=KEYS, validate='one_to_one')
    industry, cap = trading_industry_cap(loader, calendar, symbols)
    return (calendar, symbols, daily, eligible, industry, cap)

def research_calculate_factors(cfg, calendar, symbols, daily):
    cache = Path(cfg.output_dir) / 'research_cache'
    cache.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256((inspect.getsource(research_day_metrics.py_func) + inspect.getsource(research_correlation.py_func) + inspect.getsource(research_smart_ratio.py_func)).encode()).hexdigest()
    dependency = trading_input_manifest(cfg.data_dir)['signature']
    scale = daily.set_index(KEYS).Scale
    histories = defaultdict(deque)
    files = sorted((Path(cfg.data_dir) / 'bar_1m_2016_2018').glob('*.parquet'))
    parts = []
    date_index = {d: i for i, d in enumerate(calendar)}
    for path in tqdm([p for p in files if pd.Timestamp(p.stem) in date_index], desc='研报分钟因子'):
        day = pd.Timestamp(path.stem)
        position = date_index[day]
        cached = cache / path.name
        meta = cached.with_suffix('.json')
        key = {'source': str(path.resolve()), 'mtime_ns': path.stat().st_mtime_ns, 'size': path.stat().st_size, 'definition': fingerprint, 'symbols': symbols, 'pooled_window': 10, 'trading_inputs': dependency, 'aliases': TRADING_ALIASES}
        if cached.exists() and meta.exists() and (json.loads(meta.read_text()) == key):
            parts.append(pd.read_parquet(cached))
            continue
        if not histories and parts:
            raise ValueError('分钟缓存不完整：请使用新的output-dir完整重算，避免丢失10日池化历史。')
        columns = ['time', 'symbol', 'open', 'close', 'high', 'low', 'volume', 'amount', 'trade_num']
        bars = ds.dataset(path, format='parquet').to_table(columns=columns, filter=ds.field('symbol').isin(trading_raw_symbols(symbols))).to_pandas()
        bars.symbol = trading_canonical(bars.symbol)
        bars = bars.sort_values(['symbol', 'time'])
        rows = []
        for symbol, frame in bars.groupby('symbol', sort=False):
            valid = np.isfinite(frame[['open', 'close', 'high', 'low', 'volume', 'amount']]).all(axis=1)
            valid &= frame[['open', 'close', 'high', 'low']].gt(0).all(axis=1) & frame.volume.ge(0) & frame.amount.ge(0)
            frame = frame.loc[valid]
            a = scale.get((day, symbol), np.nan)
            if not np.isfinite(a) or a <= 0:
                continue
            values, score, volume, amount = research_day_metrics(*[frame[c].to_numpy(float) for c in ['open', 'close', 'high', 'low', 'volume', 'amount', 'trade_num']], a)
            history = histories[symbol]
            while history and history[0][0] <= position - 10:
                history.popleft()
            if len(score):
                history.append((position, score, volume, amount))
            pooled = np.nan
            if len(history) == 10 and history[-1][0] - history[0][0] == 9:
                pooled = research_smart_ratio(np.concatenate([x[1] for x in history]), np.concatenate([x[2] for x in history]), np.concatenate([x[3] for x in history]))
            rows.append([day, symbol, *values, pooled])
        part = pd.DataFrame(rows, columns=KEYS + list(RESEARCH_DAY_FIELDS) + ['SmartPooled'])
        part.to_parquet(cached, index=False)
        meta.write_text(json.dumps(key, ensure_ascii=False))
        parts.append(part)
    raw = pd.concat(parts, ignore_index=True)
    raw.to_parquet(Path(cfg.output_dir) / 'daily_research_metrics.parquet', index=False)
    rolled = []
    daily_index = daily.set_index(['SecuCode', 'TradingDay'])[['AdjClose', 'DailyAmplitude', 'DailyReturn', 'HighPrice', 'LowPrice', 'ClosePrice', 'up_limit', 'down_limit', 'Volume', 'PTFlag', 'SuspendAny']].sort_index()
    for symbol, frame in tqdm(raw.groupby('SecuCode'), desc='研报时间窗口'):
        grid = frame.set_index('TradingDay').reindex(calendar)
        rolling = grid[list(RESEARCH_DAY_FIELDS)].rolling(20, min_periods=20).mean()
        rolling['PVStd'] = grid.PV.rolling(20, min_periods=20).std(ddof=1)
        weights = np.arange(20, dtype=float) - 9.5
        rolling['PVTrend'] = grid.PV.rolling(20, min_periods=20).apply(lambda x: np.dot(x, weights) / np.dot(weights, weights), raw=True)
        rolling['SmartPooled'] = grid.SmartPooled
        rolling['SmartDaily'] = grid.SmartDaily.rolling(10, min_periods=10).mean()
        quotes = daily_index.loc[symbol].reindex(calendar)
        selected = research_conditional_factors(quotes.AdjClose.to_numpy(float), quotes.DailyAmplitude.to_numpy(float), quotes.DailyReturn.to_numpy(float), grid.AvgTrade.to_numpy(float), research_ideal_valid_days(quotes).to_numpy(bool))
        rolling['IdealAmp'], rolling['IdealRev'], rolling['LowAmpMom'] = (selected[:, 0], selected[:, 1], selected[:, 2])
        rolling['SecuCode'] = symbol
        rolling.index.name = 'TradingDay'
        rolled.append(rolling.reset_index())
    result = pd.concat(rolled, ignore_index=True)
    result.to_parquet(Path(cfg.output_dir) / 'rolling_components.parquet', index=False)
    return result

def research_zscore(y):
    deviation = y.std(ddof=0)
    return (y - y.mean()) / deviation if deviation > 1e-12 else y * 0

def research_winsor(y):
    med = y.median()
    mad = (y - med).abs().median()
    return y.clip(med - 5 * mad, med + 5 * mad) if mad > 1e-12 else y

def research_residual(y, controls):
    output = pd.Series(np.nan, index=y.index)
    valid = y.notna() & controls.notna().all(axis=1)
    if valid.sum() >= max(20, controls.shape[1] + 3):
        x = np.column_stack([np.ones(valid.sum()), controls.loc[valid].to_numpy(float)])
        v = y.loc[valid].to_numpy(float)
        output.loc[valid] = v - x @ np.linalg.lstsq(x, v, rcond=None)[0]
    return output

def research_preprocess(rolling, eligible, daily, cap):
    full = rolling.merge(eligible, on=KEYS, validate='one_to_one')
    full = full.merge(cap, on=KEYS, how='left', validate='one_to_one')
    full = full.merge(daily[KEYS + CONTROLS], on=KEYS, how='left', validate='one_to_one')
    outputs = []
    for _, day in tqdm(full.groupby('TradingDay'), desc='研报分阶段合成'):
        out = day[KEYS].copy()
        avg = research_residual(research_winsor(day.PV), day[['LnCap']])
        std = research_residual(research_winsor(day.PVStd), day[['LnCap']])
        avg_ret = research_residual(avg, day[['Ctrl_Ret20']])
        std_ret = research_residual(std, day[['Ctrl_Ret20']])
        deret = research_zscore(avg_ret) + research_zscore(std_ret)
        trend = research_residual(research_winsor(day.PVTrend), day[['LnCap'] + CONTROLS])
        out['HF_CPV'] = research_zscore(research_zscore(deret) + research_zscore(trend))
        pieces = {key: research_zscore(research_residual(research_winsor(day[key]), day[['LnCap']])) for key in ['PP', 'PM', 'MP', 'MM', 'DP_Pos_P', 'DP_Neg_P']}
        out['HF_CDPDP'] = research_zscore(pieces['PP'] + pieces['MM'])
        out['HF_CDPDP4'] = research_zscore(pieces['PP'] - pieces['PM'] - pieces['MP'] + pieces['MM'])
        out['HF_CDPP'] = research_zscore(pieces['DP_Pos_P'] - pieces['DP_Neg_P'])
        for target, source in [('HF_SmartMoney', 'SmartPooled'), ('HF_IdealAmplitude', 'IdealAmp'), ('HF_IdealReversal', 'IdealRev'), ('HF_LowAmpMomentum', 'LowAmpMom'), ('HF_Skewness', 'Skew')]:
            out[target] = research_zscore(research_winsor(day[source]))
        outputs.append(out)
    result = pd.concat(outputs, ignore_index=True).sort_values(KEYS)
    return result

def research_month_schedule(calendar):
    frame = pd.Series(pd.DatetimeIndex(calendar), index=pd.DatetimeIndex(calendar))
    return pd.DatetimeIndex(frame.groupby(frame.index.to_period('M')).max())

def research_scheduled_groups(features, factor, schedule, low=True, n_groups=10):
    pieces = []
    for date in schedule:
        sub = features.loc[features.TradingDay.eq(date), KEYS + [factor]].dropna().copy()
        if len(sub) < max(20, n_groups) or sub[factor].nunique() < n_groups:
            continue
        sub = sub.sort_values([factor, 'SecuCode'], ascending=[not low, True])
        sub['Group'] = np.arange(len(sub)) * n_groups // len(sub) + 1
        pieces.append(sub)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=KEYS + [factor, 'Group'])

def research_calendar_ic(groups, market, factor, schedule, low=True):
    rows = []
    for start, end in zip(schedule[:-1], schedule[1:]):
        sub = groups[groups.TradingDay.eq(start)]
        entry, exit_ = (market.dates.get_loc(start) + 1, market.dates.get_loc(end) + 1)
        if sub.empty or exit_ >= len(market.dates):
            continue
        columns = market.symbols.get_indexer(sub.SecuCode)
        buy, sell = (market.open[entry, columns], market.open[exit_, columns])
        valid = np.isfinite(buy) & np.isfinite(sell) & (buy > 0) & (sell > 0)
        x = pd.Series(sub[factor].to_numpy()[valid])
        r = pd.Series(sell[valid] / buy[valid] - 1)
        ic = x.corr(r) if len(x) >= 10 and x.std() > 1e-12 and (r.std() > 1e-12) else np.nan
        rank = x.corr(r, method='spearman') if np.isfinite(ic) else np.nan
        rows.append({'SignalDate': start, 'EntryDate': market.dates[entry], 'ExitDate': market.dates[exit_], 'HoldingDays': exit_ - entry, 'N_Signal': len(sub), 'N_Paired': int(valid.sum()), 'Coverage': valid.mean(), 'IC_Raw': ic, 'RankIC_Raw': rank, 'IC_Directed': (-1 if low else 1) * ic, 'RankIC_Directed': (-1 if low else 1) * rank})
    return pd.DataFrame(rows)

def research_export_case(features, factor, name, market, cfg, folder, low=True, with_gross=False):
    if name == 'MonthEnd':
        schedule = research_month_schedule(market.dates)
        groups = research_scheduled_groups(features, factor, schedule, low, cfg.n_groups)
        ic = research_calendar_ic(groups, market, factor, schedule, low)
    else:
        days = PERIODS[name]
        groups, schedule = signal_groups(features, factor, market.dates, days, cfg.min_cross_section, low, cfg.n_groups)
        ic = calc_ic(groups, market, factor, days, low)
    ledger, trades = backtest(groups, schedule, market, cfg)
    folder.mkdir(parents=True, exist_ok=True)
    groups.to_parquet(folder / 'signal_groups.parquet', index=False)
    ledger.to_csv(folder / 'daily_ledger.csv', index=False)
    trades.to_parquet(folder / 'trades.parquet', index=False)
    if len(trades):
        trades['TradingCode'] = trading_trading_code(trades.TradingDay, trades.SecuCode)
        trades.to_parquet(folder / 'trades.parquet', index=False)
    ic.to_csv(folder / 'ic.csv', index=False)
    plot_result(ledger, folder / 'nav.png', f'{factor} / {name} / paper definitions', cfg, schedule[0])
    records = []
    for portfolio, frame in ledger.groupby('Portfolio'):
        record = performance(frame.Return, cfg)
        record.update({'Factor': factor, 'Period': name, 'Portfolio': portfolio, 'Cost': 'Net', 'LSLegWeight': cfg.ls_leg_weight, 'IC_Raw': ic.IC_Raw.mean(), 'RankIC_Raw': ic.RankIC_Raw.mean(), 'IC_Coverage': ic.Coverage.mean(), 'MeanSignalStocks': ic.N_Signal.mean(), 'MeanDailyTurnover': frame.Turnover.mean(), 'TotalTradingCost': frame.TradingCost.sum(), 'StalePositionDays': frame.StalePositions.sum(), 'FinalPositions': frame.Positions.iloc[-1], 'BlockedOrders': frame.BlockedOrders.sum(), 'AdmissionBlockedOrders': frame.AdmissionBlockedOrders.sum()})
        records.append(record)
    if with_gross:
        gross, _ = backtest(groups, schedule, market, replace(cfg, one_way_fee=0))
        gross.to_csv(folder / 'daily_ledger_gross.csv', index=False)
        for portfolio, frame in gross.groupby('Portfolio'):
            record = performance(frame.Return, cfg)
            record.update({'Factor': factor, 'Period': name, 'Portfolio': portfolio, 'Cost': 'Gross', 'LSLegWeight': cfg.ls_leg_weight})
            records.append(record)
    return records

def research_icir_scores(features, market, days, output, predict_start):
    """开源2021脚注3的12期最大ICIR方法移植，当前六个可实现因子。

    只用已实现的非重叠历史IC；协方差使用固定10%对角收缩以处理12样本/6维
    的不稳定性。原报告未给出数值求逆细节，此处是公开的数值约定。
    """
    features = features.dropna(subset=list(RESEARCH_PRIMARY)).sort_values(KEYS)
    dates = market.dates
    first = dates.searchsorted(pd.Timestamp(predict_start))
    anchors = np.arange(first % days, len(dates) - days - 1, days)
    feature_map = {date: frame for date, frame in features.groupby('TradingDay')}
    historic = []
    for t in anchors:
        if dates[t] not in feature_map:
            continue
        frame = feature_map[dates[t]]
        columns = market.symbols.get_indexer(frame.SecuCode)
        buy, sell = (market.open[t + 1, columns], market.open[t + days + 1, columns])
        valid = np.isfinite(buy) & np.isfinite(sell) & (buy > 0) & (sell > 0)
        returns = pd.Series(sell[valid] / buy[valid] - 1)
        values = []
        for factor in RESEARCH_PRIMARY:
            x = pd.Series(-frame[factor].to_numpy()[valid])
            values.append(x.corr(returns) if len(x) >= 10 and x.std() > 1e-12 and (returns.std() > 1e-12) else np.nan)
        if np.isfinite(values).all():
            historic.append((dates[t], dates[t + days + 1], values))
    predictions = []
    weights = []
    for date, frame in feature_map.items():
        if date < pd.Timestamp(predict_start):
            continue
        past = [item for item in historic if item[1] < date][-12:]
        pred = frame[KEYS].copy()
        pred['ICIR_12'] = np.nan
        if len(past) == 12:
            sample = np.array([item[2] for item in past])
            mean = sample.mean(axis=0)
            covariance = np.cov(sample, rowvar=False)
            scale = max(np.trace(covariance) / len(RESEARCH_PRIMARY), 1e-08)
            covariance = 0.9 * covariance + 0.1 * scale * np.eye(len(RESEARCH_PRIMARY))
            w = np.linalg.solve(covariance, mean)
            if np.sum(np.abs(w)) > 1e-12:
                w /= np.sum(np.abs(w))
                pred['ICIR_12'] = -frame[list(RESEARCH_PRIMARY)].to_numpy() @ w
                weights.append({'TradingDay': date, 'MaxLabelEnd': past[-1][1], 'ICPeriods': 12, **{factor: float(value) for factor, value in zip(RESEARCH_PRIMARY, w)}})
        predictions.append(pred)
    result = pd.concat(predictions, ignore_index=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output / 'predictions.parquet', index=False)
    pd.DataFrame(weights).to_csv(output / 'weights.csv', index=False)
    return result

# 持有期标签与历史训练样本

import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
MODEL_LOG = logging.getLogger('code2_new.xgboost')

def model_make_labels(features, market, days):
    """未来缺失仅影响历史训练标签；不用于过滤预测日股票池。"""
    labels = features[KEYS].copy()
    t = market.dates.get_indexer(labels.TradingDay)
    s = market.symbols.get_indexer(labels.SecuCode)
    if (t < 0).any() or (s < 0).any():
        raise ValueError('特征日期或股票未对齐行情。')
    valid = t + days + 1 < len(market.dates)
    buy = np.full(len(labels), np.nan)
    sell = buy.copy()
    buy[valid] = market.open[t[valid] + 1, s[valid]]
    sell[valid] = market.open[t[valid] + days + 1, s[valid]]
    good = valid & np.isfinite(buy) & np.isfinite(sell) & (buy > 0) & (sell > 0)
    labels['ForwardReturn'] = np.nan
    labels.loc[good, 'ForwardReturn'] = sell[good] / buy[good] - 1
    labels['LabelEnd'] = pd.NaT
    labels.loc[valid, 'LabelEnd'] = market.dates[t[valid] + days + 1].to_numpy()
    labels['Target'] = labels.ForwardReturn - labels.groupby('TradingDay').ForwardReturn.transform('mean')
    return labels

def model_training_sample(labeled, cutoff, window_months=12):
    """在月初之前已实现的标签；不能仅按信号日早于测试日筛样本。"""
    cutoff = pd.Timestamp(cutoff)
    start = cutoff - pd.DateOffset(months=window_months)
    mask = labeled.TradingDay.ge(start) & labeled.TradingDay.lt(cutoff) & labeled.LabelEnd.lt(cutoff) & labeled.Target.notna()
    return labeled.loc[mask].copy()

# 训练期去重、时间验证调参与滚动模型

from pathlib import Path
import json
import logging
import time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import xgboost
TUNING_LOG = logging.getLogger('tuning')
TUNING_SEED = 20261115

def tuning_rank_ic_by_day(frame, prediction):
    f = pd.DataFrame({'TradingDay': frame.TradingDay.to_numpy(), 'prediction': prediction, 'target': frame.Target.to_numpy()})
    ranked = f.groupby('TradingDay')[['prediction', 'target']].rank()
    centered = ranked - ranked.groupby(f.TradingDay).transform('mean')
    product = (centered.prediction * centered.target).groupby(f.TradingDay).sum()
    var1 = centered.prediction.pow(2).groupby(f.TradingDay).sum()
    var2 = centered.target.pow(2).groupby(f.TradingDay).sum()
    return product / np.sqrt(var1 * var2).replace(0, np.nan)

def tuning_selector_statistics(train, feature_cols):
    """截面相关逐日等权；质量为训练期日RankIC绝对均值/标准差。"""
    corr = train.groupby('TradingDay')[feature_cols + ['Target']].corr(method='spearman', min_periods=20)
    average = corr.groupby(level=1).mean().reindex(index=feature_cols, columns=feature_cols)
    stats = []
    for factor in feature_cols:
        values = corr.xs(factor, level=1).Target.dropna()
        mean, sd = (values.mean(), values.std(ddof=1))
        quality = abs(mean) / max(sd, 0.01) if np.isfinite(mean) and np.isfinite(sd) else -np.inf
        stats.append({'Feature': factor, 'MeanRankIC': mean, 'RankICStd': sd, 'Quality': quality, 'ICDays': len(values), 'Coverage': float(train[factor].notna().mean())})
    return (pd.DataFrame(stats), average)

def tuning_choose_features(stats, corr, threshold=0.8, max_features=6):
    ranked = stats.sort_values(['Quality', 'Feature'], ascending=[False, True])
    selected = []
    records = []
    for row in ranked.to_dict('records'):
        factor = row['Feature']
        reason = 'selected'
        duplicate = ''
        if row['Coverage'] < 0.5 or row['ICDays'] < 20 or (not np.isfinite(row['Quality'])):
            reason = 'insufficient_history'
        elif len(selected) >= max_features:
            reason = 'feature_budget'
        else:
            for kept in selected:
                value = corr.loc[factor, kept]
                if not np.isfinite(value) or abs(value) >= threshold:
                    reason = 'correlated' if np.isfinite(value) else 'unknown_correlation'
                    duplicate = kept
                    break
        if reason == 'selected':
            selected.append(factor)
        records.append({**row, 'Selected': reason == 'selected', 'Reason': reason, 'Representative': duplicate, 'CorrelationThreshold': threshold, 'MaxFeatures': max_features})
    if not selected:
        raise ValueError('训练历史中没有可用的候选因子。')
    return (selected, pd.DataFrame(records))

def tuning_purged_folds(train, n_splits=3, validation_days=21):
    dates = pd.DatetimeIndex(sorted(train.TradingDay.unique()))
    if len(dates) < n_splits * validation_days + 80:
        raise ValueError('时间验证历史不足。')
    folds = []
    for i in range(n_splits):
        start = len(dates) - (n_splits - i) * validation_days
        val_dates = dates[start:start + validation_days]
        tr = train[train.TradingDay.lt(val_dates[0]) & train.LabelEnd.lt(val_dates[0])].copy()
        va = train[train.TradingDay.isin(val_dates)].copy()
        if tr.TradingDay.nunique() < 60:
            raise ValueError('purge后折内训练期不足60天。')
        assert tr.LabelEnd.max() < va.TradingDay.min()
        folds.append((tr, va))
    return folds

def tuning_candidate_parameters(count, seed):
    if count < 4:
        raise ValueError('完整搜索至少包含4个候选。')
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        out.append({'max_depth': int(rng.choice([2, 3, 4, 5, 6])), 'learning_rate': float(np.exp(rng.uniform(np.log(0.015), np.log(0.2)))), 'min_child_weight': float(rng.choice([1, 3, 10, 30, 60, 100])), 'subsample': float(rng.uniform(0.55, 1.0)), 'colsample_bytree': float(rng.uniform(0.55, 1.0)), 'reg_alpha': 0.0 if rng.random() < 0.25 else float(10 ** rng.uniform(-4, 0.3)), 'reg_lambda': float(10 ** rng.uniform(-1, 2)), 'gamma': 0.0 if rng.random() < 0.5 else float(10 ** rng.uniform(-5, -1)), 'max_bin': int(rng.choice([128, 256])), 'corr_threshold': float(rng.choice([0.7, 0.8, 0.9])), 'max_features': int(rng.choice([4, 6, 9]))})
    for i, (depth, rate) in enumerate([(2, 0.03), (3, 0.05), (4, 0.08), (5, 0.04)]):
        out[i].update(max_depth=depth, learning_rate=rate, min_child_weight=10.0, reg_alpha=0.0, reg_lambda=5.0, gamma=0.0, subsample=0.85, colsample_bytree=0.9, corr_threshold=0.8, max_features=6)
    return out

def tuning_sample_weights(frame):
    w = 1 / frame.groupby('TradingDay').SecuCode.transform('count').to_numpy(float)
    return w / w.mean()

def tuning_fitted_trial(tr, va, cols, parameters, max_trees, patience, trial_id):
    params = {k: v for k, v in parameters.items() if k not in ('corr_threshold', 'max_features')}
    model = XGBRegressor(**params, n_estimators=max_trees, early_stopping_rounds=patience, eval_metric='rmse', objective='reg:squarederror', tree_method='hist', n_jobs=2, random_state=TUNING_SEED + trial_id, verbosity=0)
    model.fit(tr[cols].astype('float32'), tr.Target, sample_weight=tuning_sample_weights(tr), eval_set=[(va[cols].astype('float32'), va.Target)], sample_weight_eval_set=[tuning_sample_weights(va)], verbose=False)
    values = tuning_rank_ic_by_day(va, model.predict(va[cols].astype('float32')))
    score = float(values.mean()) if values.notna().any() else -1.0
    return (score, int(model.best_iteration + 1))

def tuning_tune(train, feature_cols, destination, trials=96, finalists=16, seed=TUNING_SEED):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    folds = tuning_purged_folds(train)
    candidates = tuning_candidate_parameters(trials, seed)
    stats = [tuning_selector_statistics(tr, feature_cols) for tr, _ in folds]
    selections = {}
    for fold in range(3):
        for threshold in [0.7, 0.8, 0.9]:
            for budget in [4, 6, 9]:
                selections[fold, threshold, budget] = tuning_choose_features(*stats[fold], threshold, budget)[0]
    for i, (tr, va) in enumerate(folds):
        assert tr.LabelEnd.max() < va.TradingDay.min()
    split_records = [{'Fold': i, 'TrainSignalStart': tr.TradingDay.min(), 'TrainSignalEnd': tr.TradingDay.max(), 'TrainMaxLabelEnd': tr.LabelEnd.max(), 'ValidationStart': va.TradingDay.min(), 'ValidationEnd': va.TradingDay.max(), 'ValidationMaxLabelEnd': va.LabelEnd.max(), 'TrainRows': len(tr), 'ValidationRows': len(va)} for i, (tr, va) in enumerate(folds)]
    pd.DataFrame(split_records).to_csv(destination / 'splits.csv', index=False)
    coarse = []
    for i, params in enumerate(candidates):
        tr, va = folds[-1]
        cols = selections[2, params['corr_threshold'], params['max_features']]
        score, trees = tuning_fitted_trial(tr, va, cols, params, 300, 25, i)
        coarse.append({'Trial': i, 'Stage': 'coarse', 'Fold': 2, 'RankIC': score, 'BestTrees': trees, 'Features': '|'.join(cols), **params})
        if (i + 1) % 24 == 0:
            TUNING_LOG.info('%s 初筛 %d/%d', destination.name, i + 1, trials)
    pd.DataFrame(coarse).to_csv(destination / 'coarse_trials.csv', index=False)
    promoted = sorted(coarse, key=lambda q: (-q['RankIC'], q['Trial']))[:min(finalists, trials)]
    fine = []
    summary = []
    for record in promoted:
        i = record['Trial']
        params = candidates[i]
        scores = []
        trees = []
        for fold, (tr, va) in enumerate(folds):
            cols = selections[fold, params['corr_threshold'], params['max_features']]
            score, best = tuning_fitted_trial(tr, va, cols, params, 1200, 50, i)
            scores.append(score)
            trees.append(best)
            fine.append({'Trial': i, 'Stage': 'fine', 'Fold': fold, 'RankIC': score, 'BestTrees': best, 'Features': '|'.join(cols), **params})
        mean = float(np.mean(scores))
        sd = float(np.std(scores))
        summary.append({'Trial': i, 'MeanRankIC': mean, 'StdRankIC': sd, 'SelectionScore': mean - 0.1 * sd, 'Trees': max(1, int(np.median(trees))), **params})
    pd.DataFrame(fine).to_csv(destination / 'fine_trials.csv', index=False)
    ranking = pd.DataFrame(summary).sort_values(['SelectionScore', 'MeanRankIC', 'Trial'], ascending=[False, False, True])
    ranking.to_csv(destination / 'search_summary.csv', index=False)
    winner = ranking.iloc[0]
    chosen = candidates[int(winner.Trial)].copy()
    chosen['n_estimators'] = int(winner.Trees)
    result = {'parameters': chosen, 'winner_trial': int(winner.Trial), 'inner_mean_rank_ic': float(winner.MeanRankIC), 'inner_rank_ic_std': float(winner.StdRankIC), 'selection_score': float(winner.SelectionScore), 'candidates': trials, 'finalists': len(promoted), 'cv_fits': trials + len(promoted) * 3, 'max_training_label_end': str(train.LabelEnd.max()), 'validation_folds': 3, 'screening_tree_cap': 300, 'final_tree_cap': 1200, 'early_stop_metric': 'validation weighted RMSE', 'search_metric': 'mean validation daily RankIC - 0.1 * std across 3 chronological folds'}
    (destination / 'best_parameters.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result

def tuning_walk_forward(features, market, days, output, predict_start='2017-02-01', window_months=12, feature_cols=None, trials=96, finalists=16):
    feature_cols = list(RESEARCH_FACTORS if feature_cols is None else feature_cols)
    if any(('Tail' in f or 'Legacy' in f for f in feature_cols)):
        raise ValueError('已剔除的因子不可进入模型。')
    features = features[KEYS + feature_cols].copy()
    features = features[features[feature_cols].notna().sum(axis=1) >= 3].sort_values(KEYS)
    labels = model_make_labels(features, market, days)
    labeled = features.merge(labels, on=KEYS, validate='one_to_one')
    months = sorted(pd.DatetimeIndex(market.dates[market.dates >= pd.Timestamp(predict_start)]).to_period('M').unique())
    output = Path(output)
    (output / 'models').mkdir(parents=True, exist_ok=True)
    forecasts = []
    logs = []
    selection_logs = []
    importance = []
    active_quarter = None
    best = None
    t0 = time.monotonic()
    for month in months:
        dates = market.dates[market.dates.to_period('M') == month]
        cutoff = dates[0]
        train = model_training_sample(labeled, cutoff, window_months)
        test = features[features.TradingDay.dt.to_period('M').eq(month) & features.TradingDay.ge(pd.Timestamp(predict_start))]
        if test.empty:
            continue
        if train.TradingDay.nunique() < 160:
            raise ValueError('外层训练数据不足160天。')
        assert train.LabelEnd.max() < cutoff
        quarter = month.asfreq('Q')
        if quarter != active_quarter:
            TUNING_LOG.info('%d日 %s 开始训练期调参：%d候选、%d晋级、3个purge时间折', days, quarter, trials, finalists)
            best = tuning_tune(train, feature_cols, output / 'tuning' / str(quarter), trials, finalists, TUNING_SEED + days * 100 + int(quarter.ordinal))
            active_quarter = quarter
            tuning_cutoff = cutoff
        params = best['parameters'].copy()
        threshold = params.pop('corr_threshold')
        budget = params.pop('max_features')
        stats, corr = tuning_selector_statistics(train, feature_cols)
        selected, decisions = tuning_choose_features(stats, corr, threshold, budget)
        decisions['PredictionMonth'] = str(month)
        decisions['Cutoff'] = cutoff
        decisions['MaxLabelEnd'] = train.LabelEnd.max()
        selection_logs.append(decisions)
        select_dir = output / 'selection'
        select_dir.mkdir(exist_ok=True)
        corr.to_csv(select_dir / f'{month}_correlation.csv')
        model = XGBRegressor(**params, n_jobs=2, random_state=TUNING_SEED, objective='reg:squarederror', tree_method='hist', verbosity=0)
        model.fit(train[selected].astype('float32'), train.Target, sample_weight=tuning_sample_weights(train))
        pred = test[KEYS].copy()
        pred['Pred_Score'] = model.predict(test[selected].astype('float32'))
        pred.loc[test[selected].notna().sum(axis=1).to_numpy() == 0, 'Pred_Score'] = np.nan
        forecasts.append(pred)
        model.save_model(output / 'models' / f'{month}.json')
        logs.append({'PredictionMonth': str(month), 'Cutoff': cutoff, 'MaxLabelEnd': train.LabelEnd.max(), 'TrainSignalStart': train.TradingDay.min(), 'TrainSignalEnd': train.TradingDay.max(), 'TrainRows': len(train), 'TrainDays': train.TradingDay.nunique(), 'PredictionRows': len(test), 'SelectedFeatures': '|'.join(selected), 'FeatureCount': len(selected), 'CorrelationThreshold': threshold, 'MaxFeatures': budget, 'TuningQuarter': str(active_quarter), 'TuningCutoff': tuning_cutoff, 'Trees': params['n_estimators'], 'InnerRankIC': best['inner_mean_rank_ic'], 'Horizon': days})
        importance.extend(({'PredictionMonth': str(month), 'Feature': f, 'Importance': float(v)} for f, v in zip(selected, model.feature_importances_)))
        TUNING_LOG.info('%d日 %s 训练%d行，入选%s，树%d；累计%.1f秒', days, month, len(train), ','.join(selected), params['n_estimators'], time.monotonic() - t0)
    predictions = pd.concat(forecasts, ignore_index=True).sort_values(KEYS)
    predictions.to_parquet(output / 'predictions.parquet', index=False)
    pd.DataFrame(logs).to_csv(output / 'training_log.csv', index=False)
    pd.concat(selection_logs, ignore_index=True).to_csv(output / 'feature_selection.csv', index=False)
    pd.DataFrame(importance).to_csv(output / 'feature_importance.csv', index=False)
    (output / 'model_config.json').write_text(json.dumps({'xgboost_version': xgboost.__version__, 'candidate_features': feature_cols, 'horizon': days, 'predict_start': predict_start, 'training_window_months': window_months, 'trials': trials, 'finalists': finalists, 'quarterly_tuning': True, 'monthly_selection_and_refit': True, 'seed': TUNING_SEED, 'purge': 'training LabelEnd < validation start / outer prediction month start', 'native_missing_values': True, 'prediction_admission': 'at least 3 contemporaneous available candidate factors', 'correlation': 'training daily cross-sectional Spearman, equally weighted dates; greedy representative by |mean RankIC|/std', 'target': 'cross-section demeaned adjusted N-day open return'}, ensure_ascii=False, indent=2))
    return predictions

# 完整实验流程

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import hashlib
import inspect
import json
import logging
import numpy as np
import pandas as pd
PIPELINE_LOG = logging.getLogger('pipeline')
PIPELINE_CANDIDATES = tuple(RESEARCH_FACTORS)

def pipeline_factor_signature():
    functions = [research_day_metrics.py_func, research_correlation.py_func, research_smart_ratio.py_func, research_conditional_factors.py_func, research_ideal_valid_days, research_preprocess, research_winsor, research_zscore, research_residual]
    return hashlib.sha256((''.join((inspect.getsource(f) for f in functions)) + str(PIPELINE_CANDIDATES)).encode()).hexdigest()

def pipeline_file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def pipeline_write_input_manifest(cfg, origin):
    folder = Path(cfg.output_dir) / 'input'
    meta = {'origin': origin, 'factors': PIPELINE_CANDIDATES, 'factor_signature': pipeline_factor_signature(), 'trading_input_signature': trading_input_manifest(cfg.data_dir)['signature'], 'files': {name: pipeline_file_hash(folder / name) for name in ['features.parquet', 'daily_market.parquet']}}
    (folder / 'input_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))

def pipeline_prepare_inputs(cfg):
    folder = Path(cfg.output_dir) / 'input'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'input_manifest.json'
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        if meta['factor_signature'] != pipeline_factor_signature():
            raise ValueError('因子定义已经变化，需重新生成输入缓存。')
        if meta['trading_input_signature'] != trading_input_manifest(cfg.data_dir)['signature']:
            raise ValueError('交易数据已变化，需重新生成输入缓存。')
        for name, digest in meta['files'].items():
            if pipeline_file_hash(folder / name) != digest:
                raise ValueError(f'{name}缓存校验失败。')
        PIPELINE_LOG.info('已验证并复用9因子及实际交易状态输入；不复用模型预测。')
    else:
        inner = replace(cfg, output_dir=folder)
        calendar, symbols, daily, eligible, industry, cap = research_load_data(inner, '500')
        rolling = research_calculate_factors(inner, calendar, symbols, daily)
        features = research_preprocess(rolling, eligible, daily, cap)[KEYS + list(PIPELINE_CANDIDATES)]
        features.to_parquet(folder / 'features.parquet', index=False)
        pipeline_write_input_manifest(cfg, 'fresh raw-minute calculation with current factor definitions')
    features = pd.read_parquet(folder / 'features.parquet')
    daily = pd.read_parquet(folder / 'daily_market.parquet')
    assert list(features.columns) == KEYS + list(PIPELINE_CANDIDATES)
    assert not any(('Tail' in c or 'Legacy' in c for c in features.columns))
    return (features, daily)

def pipeline_export(features, factor, period, market, cfg, destination, low):
    rows = research_export_case(features, factor, period, market, cfg, destination, low, False)
    for row in rows:
        row['Groups'] = cfg.n_groups
        row['Role'] = 'Long' if row['Portfolio'] == f'Group{cfg.n_groups}' else 'LongShort' if row['Portfolio'] == 'TheoreticalLS' else 'Decile'
    return rows

def pipeline_run(cfg, with_models=True, models_only=False, prepare_only=False, predict_start='2017-02-01', trials=96, finalists=16):
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if cfg.n_groups != 10:
        raise ValueError('当前提交版固定为10分组。')
    if not np.isclose(cfg.one_way_fee, 0.0015):
        raise ValueError('当前提交版固定每次买卖各0.0015，完整往返0.003。')
    features, daily = pipeline_prepare_inputs(cfg)
    if prepare_only:
        return
    calendar = pd.DatetimeIndex(sorted(daily.TradingDay.unique()))
    full_dates = calendar[calendar >= pd.Timestamp('2016-01-29')]
    full_market = Market.from_daily(daily, full_dates)
    feature_coverage = features.groupby('TradingDay')[list(PIPELINE_CANDIDATES)].count()
    feature_coverage.to_csv(output / 'feature_coverage.csv')
    pool = None
    jobs = {}
    if with_models:
        full_history = Market.from_daily(daily, calendar)
        pool = ThreadPoolExecutor(max_workers=3)
        for name, days in PERIODS.items():
            isolated = replace(full_history, dates=pd.DatetimeIndex(full_history.dates.to_numpy(copy=True)), symbols=pd.Index(full_history.symbols.to_numpy(copy=True)))
            isolated_features = features.copy(deep=True)
            jobs[name] = pool.submit(tuning_walk_forward, isolated_features, isolated, days, output / 'models' / name, predict_start, 12, list(PIPELINE_CANDIDATES), trials, finalists)
    records = []
    if not models_only:
        for factor in PIPELINE_CANDIDATES:
            for period in list(PERIODS) + ['MonthEnd']:
                PIPELINE_LOG.info('10分组单因子 %s / %s', factor, period)
                records += pipeline_export(features, factor, period, full_market, cfg, output / 'factors' / factor / period, RESEARCH_LOW_IS_LONG[factor])
        pd.DataFrame(records).to_csv(output / 'factor_summary.csv', index=False)
    if with_models:
        comparison = []
        test_dates = calendar[calendar >= pd.Timestamp(predict_start)]
        market = Market.from_daily(daily, test_dates)
        try:
            for name, future in jobs.items():
                pred = future.result()
                keys = pred.loc[pred.Pred_Score.notna(), KEYS]
                common = features.merge(keys, on=KEYS, validate='one_to_one')
                for factor in PIPELINE_CANDIDATES:
                    comparison += pipeline_export(common, factor, name, market, cfg, output / 'oos' / factor / name, RESEARCH_LOW_IS_LONG[factor])
                comparison += pipeline_export(pred, 'Pred_Score', name, market, cfg, output / 'oos/Pred_Score' / name, False)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        pd.DataFrame(comparison).to_csv(output / 'oos_summary.csv', index=False)
    source_names = [Path(__file__).name]
    manifest = {'universe': 'point-in-time CSI500', 'candidate_factors': PIPELINE_CANDIDATES, 'groups': 10, 'one_way_fee': cfg.one_way_fee, 'round_trip_fee': 2 * cfg.one_way_fee, 'ls_leg_weight': cfg.ls_leg_weight, 'signal_start': '2016-01-29', 'predict_start': predict_start, 'end': str(full_dates[-1].date()), 'parameter_candidates_per_search': trials, 'finalists_per_search': finalists, 'tuning_frequency': 'quarterly', 'refit_frequency': 'monthly', 'inner_folds': 3, 'seed': TUNING_SEED, 'cache_manifest': json.loads((output / 'input/input_manifest.json').read_text()), 'sources': {name: pipeline_file_hash(Path(__file__).parent / name) for name in source_names}}
    (output / 'run_config.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    PIPELINE_LOG.info('完整实验完成：%s', output)

# 对外公开的当前九因子目录
FACTORS = RESEARCH_FACTORS


def run_self_tests(output_dir=BASE / "results"):
    """运行与合并前一致的65项因子、账本、训练边界测试。"""
    import io
    import unittest
    import numpy as np
    import pandas as pd

    class CorrectedPipelineTests(unittest.TestCase):

        def setUp(self):
            self.cfg = Config(one_way_fee=0)

        def raw_quotes(self, dates, prices, reference=None):
            reference = reference if reference is not None else np.r_[prices[0], prices[:-1]]
            return pd.DataFrame({'TradingDay': dates, 'SecuCode': '000001', 'OpenPrice': prices, 'ClosePrice': prices, 'PrevClosePrice': reference, 'PTFlag': 0, 'Volume': 1000, 'NonRestrictedShares': 1000000, 'AFloats': 1000000})

        def market(self, periods=12, stocks=20):
            dates = pd.bdate_range('2017-01-02', periods=periods)
            symbols = pd.Index([f'{i:06d}' for i in range(stocks)])
            prices = np.ones((periods, stocks))
            allowed = np.ones_like(prices, dtype=bool)
            return Market(dates, symbols, prices.copy(), prices.copy(), allowed.copy(), allowed.copy(), ~allowed)

        def features(self, market):
            return pd.DataFrame([{'TradingDay': day, 'SecuCode': symbol, 'factor': float(i)} for day in market.dates for i, symbol in enumerate(market.symbols)])

        def test_ex_rights_continuous_prices_and_no_future_feedback(self):
            dates = pd.bdate_range('2017-01-02', periods=4)
            raw = self.raw_quotes(dates, [100.0, 50.0, 55.0, 60.0], [100.0, 50.0, 50.0, 55.0])
            daily = prepare_daily(raw, dates, self.cfg)
            np.testing.assert_allclose(daily.AdjClose, [100, 100, 110, 120])
            np.testing.assert_allclose(daily.AdjOpen, [100, 100, 110, 120])
            modified = raw.copy()
            modified.loc[3, 'ClosePrice'] = 80
            changed = prepare_daily(modified, dates, self.cfg)
            np.testing.assert_allclose(daily.AdjClose[:3], changed.AdjClose[:3])
            self.assertTrue(daily.CanBuy.iloc[1])

        def test_suspension_keeps_value_and_does_not_become_minus_100_percent(self):
            dates = pd.bdate_range('2017-01-02', periods=4)
            raw = self.raw_quotes(dates, [10.0, 0.0, 0.0, 11.0], [10.0, 10.0, 10.0, 10.0])
            raw.loc[[1, 2], 'PTFlag'] = 1
            raw.loc[[1, 2], 'Volume'] = 0
            daily = prepare_daily(raw, dates, self.cfg)
            np.testing.assert_allclose(daily.AdjClose, [10, 10, 10, 11])
            self.assertTrue(daily.AdjOpen.iloc[1:3].isna().all())
            self.assertFalse(daily.CanSell.iloc[1])
            self.assertFalse(daily.StaleQuote.iloc[1])

        def test_unknown_quote_is_marked_stale(self):
            dates = pd.bdate_range('2017-01-02', periods=4)
            raw = self.raw_quotes(dates, [10.0, 10.0, 10.0, 10.0]).drop(index=3)
            daily = prepare_daily(raw, dates, self.cfg)
            self.assertTrue(daily.StaleQuote.iloc[3])
            self.assertEqual(daily.AdjClose.iloc[3], 10)
            self.assertFalse(daily.CanBuy.iloc[3])

        def test_unexplained_interior_gap_requires_data_repair(self):
            dates = pd.bdate_range('2017-01-02', periods=4)
            raw = self.raw_quotes(dates, [10.0, 10.0, 10.0, 10.0]).drop(index=1)
            with self.assertRaisesRegex(ValueError, '未解释缺口'):
                prepare_daily(raw, dates, self.cfg)

        def test_execution_does_not_use_full_day_volume(self):
            dates = pd.bdate_range('2017-01-02', periods=4)
            raw = self.raw_quotes(dates, [10.0, 10.0, 10.0, 10.0])
            baseline = prepare_daily(raw, dates, self.cfg)
            raw.loc[1, 'Volume'] = 0
            changed = prepare_daily(raw, dates, self.cfg)
            self.assertEqual(baseline.CanBuy.iloc[1], changed.CanBuy.iloc[1])
            self.assertEqual(baseline.CanSell.iloc[1], changed.CanSell.iloc[1])

        def test_style_windows_use_complete_history(self):
            dates = pd.bdate_range('2017-01-02', periods=25)
            raw = self.raw_quotes(dates, np.linspace(10, 12.4, 25))
            daily = prepare_daily(raw, dates, self.cfg)
            self.assertAlmostEqual(daily.Ctrl_Ret20.iloc[20], 0.2)
            raw = raw.drop(index=10)
            with self.assertRaisesRegex(ValueError, '未解释缺口'):
                prepare_daily(raw, dates, self.cfg)

        def test_intraday_smart_money_uses_real_amount(self):
            n = 60
            close = 10 + np.sin(np.arange(n)) * 0.1
            volume = 100 + np.arange(n, dtype=float)
            amount = volume * (10 + np.cos(np.arange(n)) * 0.3)
            values = intraday_metrics(close, close, close + 0.01, close - 0.01, volume, amount)
            ret = np.r_[0, close[1:] / close[:-1] - 1]
            order = np.argsort(np.abs(ret) / volume ** 0.25)[::-1]
            target = volume.sum() * 0.2
            remaining, smart_amount = (target, 0.0)
            for i in order:
                used = min(volume[i], remaining)
                smart_amount += amount[i] * used / volume[i]
                remaining -= used
                if remaining <= 1e-10:
                    break
            expected = smart_amount / target / (amount.sum() / volume.sum())
            self.assertAlmostEqual(values[0], expected)
            self.assertAlmostEqual(values[4], close[-1] / close[-31] - 1)

        def test_rolling_missing_day_does_not_compress_window(self):
            dates = pd.bdate_range('2017-01-02', periods=25)
            raw = pd.DataFrame({'TradingDay': dates, 'SecuCode': '000001', **{f: 1.0 for f in _BASELINE_FACTORS}})
            raw = raw.drop(index=10)
            rolled = roll_factors(raw, dates, self.cfg)
            self.assertTrue(rolled.HF_Amplitude.iloc[20:].isna().all())

        def test_fixed_schedule_and_low_value_long_direction(self):
            market = self.market()
            features = self.features(market)
            features = features[~features.TradingDay.eq(market.dates[5])]
            groups, schedule = signal_groups(features, 'factor', market.dates, 5)
            self.assertEqual(list(schedule), list(market.dates[::5]))
            self.assertNotIn(market.dates[6], set(groups.TradingDay))
            first = groups[groups.TradingDay.eq(market.dates[0])]
            self.assertEqual(set(first[first.Group.eq(10)].SecuCode), set(market.symbols[:2]))

        def test_ic_uses_matching_n_day_open_horizon(self):
            market = self.market(periods=25)
            for t in range(len(market.dates)):
                market.open[t] = 1 + t / 100 * np.arange(1, 21)
            features = self.features(market)
            for days in (5, 10, 20):
                groups, _ = signal_groups(features, 'factor', market.dates, days)
                result = calc_ic(groups, market, 'factor', days)
                self.assertEqual(result.ExitDate.iloc[0], market.dates[days + 1])
                self.assertEqual(result.EntryDate.iloc[0], market.dates[1])
                expected_returns = market.open[days + 1] / market.open[1] - 1
                expected = pd.Series(np.arange(20.0)).corr(pd.Series(expected_returns))
                self.assertAlmostEqual(result.IC_Raw.iloc[0], expected)

        def test_future_missing_price_changes_ic_coverage_not_signal_groups(self):
            market = self.market()
            features = self.features(market)
            before, _ = signal_groups(features, 'factor', market.dates, 5)
            market.open[6, 0] = np.nan
            after, _ = signal_groups(features, 'factor', market.dates, 5)
            pd.testing.assert_frame_equal(before, after)
            result = calc_ic(after, market, 'factor', 5)
            self.assertEqual(result.N_Signal.iloc[0], 20)
            self.assertEqual(result.N_Paired.iloc[0], 19)

        def test_initial_and_actual_turnover_costs(self):
            cfg = Config(one_way_fee=0.0015)
            book = Portfolio(pd.Index(['a', 'b']), cfg, 'test')
            all_ = np.ones(2, dtype=bool)
            initial = book.execute(np.ones(2), all_, all_, 'day1', np.array([0.5, 0.5]))
            self.assertGreater(initial['TradingCost'], 0)
            self.assertAlmostEqual(book.value(np.ones(2)), 1 / 1.0015)
            drift = book.execute(np.array([1.2, 0.8]), all_, all_, 'day2', np.array([0.5, 0.5]))
            self.assertGreater(drift['TradingCost'], 0)
            self.assertGreaterEqual(book.cash, -1e-12)

        def test_rejected_buy_keeps_cash_without_reranking(self):
            book = Portfolio(pd.Index(['a', 'b']), self.cfg, 'test')
            book.execute(np.ones(2), np.array([False, True]), np.ones(2, bool), 'day1', np.array([0.5, 0.5]))
            np.testing.assert_allclose(book.units, [0, 0.5])
            self.assertAlmostEqual(book.cash, 0.5)

        def test_blocked_exit_keeps_position_and_retries(self):
            book = Portfolio(pd.Index(['a']), self.cfg, 'test')
            yes, no = (np.array([True]), np.array([False]))
            book.execute(np.array([1.0]), yes, yes, 'day1', np.array([1.0]))
            book.execute(np.array([1.0]), no, no, 'day2', np.array([0.0]))
            self.assertEqual(book.units[0], 1)
            self.assertEqual(book.pending[0], 0)
            book.execute(np.array([1.2]), yes, yes, 'day3')
            self.assertEqual(book.units[0], 0)
            self.assertAlmostEqual(book.cash, 1.2)

        def test_future_rank_changes_cannot_change_past_fees_or_nav(self):
            market = self.market()
            features = self.features(market)
            groups, schedule = signal_groups(features, 'factor', market.dates, 3)
            baseline, _ = backtest(groups, schedule, market, Config())
            features.loc[features.TradingDay.eq(market.dates[9]), 'factor'] *= -1
            changed, schedule = signal_groups(features, 'factor', market.dates, 3)
            modified, _ = backtest(changed, schedule, market, Config())
            pd.testing.assert_frame_equal(baseline[baseline.TradingDay < market.dates[10]], modified[modified.TradingDay < market.dates[10]])

        def test_non_rebalance_signals_do_not_change_nav(self):
            market = self.market()
            features = self.features(market)
            groups, schedule = signal_groups(features, 'factor', market.dates, 5)
            baseline, _ = backtest(groups, schedule, market, self.cfg)
            features.loc[~features.TradingDay.isin(schedule), 'factor'] *= -1
            changed, schedule = signal_groups(features, 'factor', market.dates, 5)
            result, _ = backtest(changed, schedule, market, self.cfg)
            pd.testing.assert_frame_equal(baseline, result)

        def test_final_blocked_positions_remain_reported(self):
            market = self.market()
            market.sell[-1] = False
            groups, schedule = signal_groups(self.features(market), 'factor', market.dates, 5)
            ledger, _ = backtest(groups, schedule, market, self.cfg)
            final = ledger[(ledger.TradingDay == market.dates[-1]) & (ledger.Portfolio == 'Group10')].iloc[0]
            self.assertEqual(final.Positions, 2)
            self.assertEqual(final.PendingReductions, 2)
            self.assertAlmostEqual(final.NAV, 1)

        def test_ledger_cash_positions_and_trades_reconcile(self):
            market = self.market()
            market.close[1:, :4] *= 1.05
            market.open[2:, :4] *= 1.05
            cfg = Config()
            groups, schedule = signal_groups(self.features(market), 'factor', market.dates, 5)
            ledger, trades = backtest(groups, schedule, market, cfg)
            for name, frame in ledger.groupby('Portfolio'):
                np.testing.assert_allclose(frame.NAV, frame.Cash + frame.LongValue - frame.ShortValue)
                metrics = performance(frame.Return, cfg)
                self.assertAlmostEqual(metrics['FinalNAV'], frame.NAV.iloc[-1])
                self.assertAlmostEqual(frame.TradingCost.sum(), trades.loc[trades.Portfolio.eq(name), 'Cost'].sum())

        def test_drawdown_includes_initial_capital_and_daily_path(self):
            metrics = performance([-0.1, 0.01, 0.01], self.cfg)
            self.assertAlmostEqual(metrics['MaxDrawdown'], -0.1)
            metrics = performance([-0.2, 0.25], self.cfg)
            self.assertAlmostEqual(metrics['TotalReturn'], 0)
            self.assertAlmostEqual(metrics['MaxDrawdown'], -0.2)

        def test_geometric_annual_return_and_arithmetic_sharpe(self):
            returns = np.array([-0.1, 0.1, 0.02])
            metrics = performance(returns, self.cfg)
            expected = np.prod(1 + returns) ** (250 / 3) - 1
            self.assertAlmostEqual(metrics['AnnualReturn'], expected)
            self.assertAlmostEqual(metrics['Sharpe'], returns.mean() / returns.std(ddof=1) * np.sqrt(250))

    import unittest
    import tempfile
    import numpy as np
    import pandas as pd

    class PaperConstructionTests(unittest.TestCase):

        def test_smart_money_is_pooled_not_daily_average(self):
            scores = [np.array([9.0, 1.0]), np.array([1.0, 0.0])]
            volume = [np.array([1.0, 9.0]), np.array([90.0, 10.0])]
            amount = [np.array([20.0, 90.0]), np.array([900.0, 100.0])]
            pooled = research_smart_ratio(np.concatenate(scores), np.concatenate(volume), np.concatenate(amount))
            mean = np.mean([research_smart_ratio(s, v, a) for s, v, a in zip(scores, volume, amount)])
            self.assertNotAlmostEqual(pooled, mean)
            self.assertAlmostEqual(pooled, 230 / 22 / (1110 / 110))

        def test_ideal_amplitude_conditions_on_price_rank(self):
            price = np.arange(1.0, 21.0)
            amp = np.arange(20.0) / 100
            result = research_conditional_factors(price, amp, np.ones(20) * 0.01, np.arange(20.0) + 1)
            self.assertAlmostEqual(result[-1, 0], amp[-5:].mean() - amp[:5].mean())
            self.assertNotAlmostEqual(result[-1, 0], amp.mean())

        def test_ideal_amplitude_selects_valid_days_instead_of_requiring_twenty(self):
            price = np.arange(1.0, 21.0)
            amp = np.arange(20.0) / 100
            amp[8] = np.nan
            result = research_conditional_factors(price, amp, np.ones(20) * 0.01, np.ones(20))
            self.assertAlmostEqual(result[-1, 0], amp[-4:].mean() - amp[:4].mean())

        def test_ideal_amplitude_requires_ten_effective_days(self):
            price = np.arange(1.0, 21.0)
            amp = np.arange(20.0) / 100
            valid = np.arange(20) < 10
            result = research_conditional_factors(price, amp, np.ones(20) * 0.01, np.ones(20), valid)
            self.assertAlmostEqual(result[-1, 0], amp[8:10].mean() - amp[:2].mean())
            valid[9] = False
            result = research_conditional_factors(price, amp, np.ones(20) * 0.01, np.ones(20), valid)
            self.assertTrue(np.isnan(result[-1, 0]))

        def test_ideal_effective_days_exclude_halts_and_one_price_limits(self):
            q = pd.DataFrame({'HighPrice': [11.0, 9.0, 10.0, 11.0, 11.0], 'LowPrice': [11.0, 9.0, 10.0, 10.0, 10.0], 'ClosePrice': [11.0, 9.0, 10.0, 10.5, 10.5], 'up_limit': [11.0] * 5, 'down_limit': [9.0] * 5, 'Volume': [100.0] * 5, 'PTFlag': [0] * 5, 'SuspendAny': [False, False, False, False, True]})
            self.assertEqual(research_ideal_valid_days(q).tolist(), [False, False, True, True, False])

        def test_ideal_effective_mask_does_not_change_long_momentum_definition(self):
            price = np.ones(160)
            amp = np.arange(160.0) / 100
            ret = np.arange(160.0) / 10000
            a = research_conditional_factors(price, amp, ret, np.ones(160))
            b = research_conditional_factors(price, amp, ret, np.ones(160), np.zeros(160, dtype=bool))
            self.assertAlmostEqual(a[-1, 2], b[-1, 2])
            self.assertTrue(np.isnan(b[-1, 0]))

        def test_ideal_reversal_conditions_on_average_trade_amount(self):
            price = np.ones(20)
            ret = np.arange(20.0) / 1000
            avg = np.arange(20.0)[::-1] + 1
            result = research_conditional_factors(price, np.ones(20) * 0.1, ret, avg)
            self.assertAlmostEqual(result[-1, 1], ret[:10].sum() - ret[-10:].sum())

        def test_long_momentum_selects_low_amplitude_seventy_percent(self):
            price = np.ones(160)
            amp = np.arange(160.0)[::-1]
            ret = np.arange(160.0) / 10000
            result = research_conditional_factors(price, amp, ret, np.ones(160))
            self.assertAlmostEqual(result[-1, 2], ret[-112:].sum())
            self.assertTrue(np.isnan(result[158, 2]))

        def test_zero_variance_correlation_neutral_not_entire_stock_missing(self):
            self.assertEqual(research_correlation(np.ones(20), np.arange(20.0)), 0)
            self.assertEqual(research_correlation(np.ones(2), np.ones(2)), 0)
            price = np.ones(60) * 10
            values, *_ = research_day_metrics(price, price, price, price, np.ones(60), np.ones(60) * 10, np.ones(60), 1.0)
            self.assertTrue(np.isfinite(values[:7]).all())
            self.assertTrue((values[:7] == 0).all())

        def synthetic(self):
            rng = np.random.default_rng(123)
            n = 80
            day = pd.Timestamp('2017-03-31')
            keys = pd.DataFrame({'TradingDay': day, 'SecuCode': [f'{i:06d}' for i in range(n)]})
            rolling = keys.copy()
            for name in list(RESEARCH_DAY_FIELDS) + ['PVStd', 'PVTrend', 'SmartPooled', 'IdealAmp', 'IdealRev', 'LowAmpMom']:
                rolling[name] = rng.normal(size=n)
            daily = keys.copy()
            for c in CONTROLS:
                daily[c] = rng.normal(size=n)
            cap = keys.assign(LnCap=rng.normal(size=n))
            return (rolling, keys, daily, cap)

        def test_cdpdp_standardizes_subfactors_before_combining(self):
            rolling, keys, daily, cap = self.synthetic()
            a = research_preprocess(rolling, keys, daily, cap)
            rolling.PP *= 100
            b = research_preprocess(rolling, keys, daily, cap)
            np.testing.assert_allclose(a.HF_CDPDP, b.HF_CDPDP, atol=1e-10)

        def test_cpv_final_factor_contains_variation_and_trend(self):
            rolling, keys, daily, cap = self.synthetic()
            a = research_preprocess(rolling, keys, daily, cap)
            rolling.PVTrend = rolling.PVTrend[::-1].to_numpy()
            b = research_preprocess(rolling, keys, daily, cap)
            self.assertLess(a.HF_CPV.corr(b.HF_CPV), 0.99)
            self.assertAlmostEqual(a.HF_CDPDP.corr(b.HF_CDPDP), 1.0)

        def test_calendar_month_uses_true_variable_holding_length(self):
            dates = pd.bdate_range('2016-02-01', '2016-06-03')
            symbols = pd.Index([f'{i:06d}' for i in range(20)])
            prices = np.ones((len(dates), 20))
            flag = np.ones_like(prices, bool)
            market = Market(dates, symbols, prices, prices, flag, flag, ~flag)
            schedule = research_month_schedule(dates)
            self.assertEqual(schedule[0], pd.Timestamp('2016-02-29'))
            features = pd.DataFrame([{'TradingDay': d, 'SecuCode': s, 'f': float(i)} for d in schedule for i, s in enumerate(symbols)])
            groups = research_scheduled_groups(features, 'f', schedule)
            ic = research_calendar_ic(groups, market, 'f', schedule)
            self.assertEqual(ic.EntryDate.iloc[0], pd.Timestamp('2016-03-01'))
            self.assertEqual(ic.ExitDate.iloc[0], pd.Timestamp('2016-04-01'))
            self.assertEqual(ic.HoldingDays.iloc[0], 23)

        def test_leverage_convention_is_explicit(self):
            allowed = np.ones(2, bool)
            book1 = Portfolio(pd.Index(['a', 'b']), Config(one_way_fee=0, ls_leg_weight=0.5), 'TheoreticalLS')
            book2 = Portfolio(pd.Index(['a', 'b']), Config(one_way_fee=0, ls_leg_weight=1.0), 'TheoreticalLS')
            book1.execute(np.ones(2), allowed, allowed, 'entry', np.array([0.5, -0.5]))
            book2.execute(np.ones(2), allowed, allowed, 'entry', np.array([1.0, -1.0]))
            self.assertAlmostEqual(book2.value(np.array([1.1, 1])) - 1, 2 * (book1.value(np.array([1.1, 1])) - 1))

        def test_long_and_short_both_pay_actual_fees(self):
            cfg = Config(one_way_fee=0.0015, ls_leg_weight=1.0)
            book = Portfolio(pd.Index(['a', 'b']), cfg, 'TheoreticalLS')
            allowed = np.ones(2, bool)
            stats = book.execute(np.ones(2), allowed, allowed, 'entry', np.array([1.0, -1.0]))
            self.assertAlmostEqual(stats['TradingCost'], 0.003 / (1 + 0.003))
            self.assertAlmostEqual(book.value(np.ones(2)), 1 / (1 + 0.003))
            stats = book.execute(np.ones(2), allowed, allowed, 'exit', np.zeros(2))
            self.assertGreater(stats['TradingCost'], 0)
            self.assertLess(book.value(np.ones(2)), 1)

        def test_icir_weight_window_uses_only_realized_twelve_periods(self):
            dates = pd.bdate_range('2016-01-04', periods=100)
            symbols = pd.Index([f'{i:06d}' for i in range(20)])
            t = np.arange(100)[:, None]
            s = np.arange(20)[None, :]
            prices = 10 + np.sin(t / 8 + s / 3) + 0.005 * t * s
            flags = np.ones_like(prices, bool)
            market = Market(dates, symbols, prices.copy(), prices.copy(), flags, flags, ~flags)
            rng = np.random.default_rng(88)
            features = pd.DataFrame([{'TradingDay': date, 'SecuCode': symbol, **{f: float(rng.normal()) for f in RESEARCH_PRIMARY}} for date in dates for symbol in symbols])
            with tempfile.TemporaryDirectory() as folder:
                result = research_icir_scores(features, market, 5, folder, str(dates[80].date()))
                w = pd.read_csv(__import__('pathlib').Path(folder) / 'weights.csv', parse_dates=['TradingDay', 'MaxLabelEnd'])
                self.assertTrue(w.ICPeriods.eq(12).all())
                self.assertTrue(w.MaxLabelEnd.lt(w.TradingDay).all())
                self.assertTrue(result[result.TradingDay.eq(dates[80])].ICIR_12.notna().all())

        def test_future_prices_cannot_change_first_icir_prediction(self):
            dates = pd.bdate_range('2016-01-04', periods=100)
            symbols = pd.Index([f'{i:06d}' for i in range(20)])
            t = np.arange(100)[:, None]
            s = np.arange(20)[None, :]
            prices = 10 + np.sin(t / 8 + s / 3) + 0.005 * t * s
            flags = np.ones_like(prices, bool)
            market = Market(dates, symbols, prices.copy(), prices.copy(), flags, flags, ~flags)
            rng = np.random.default_rng(88)
            features = pd.DataFrame([{'TradingDay': date, 'SecuCode': symbol, **{f: float(rng.normal()) for f in RESEARCH_PRIMARY}} for date in dates for symbol in symbols])
            with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                before = research_icir_scores(features, market, 5, a, str(dates[80].date()))
                market.open[80:] *= np.linspace(1, 3, 20)
                after = research_icir_scores(features, market, 5, b, str(dates[80].date()))
                np.testing.assert_allclose(before[before.TradingDay.eq(dates[80])].ICIR_12, after[after.TradingDay.eq(dates[80])].ICIR_12)

    import tempfile
    from pathlib import Path
    import unittest
    import numpy as np
    import pandas as pd

    class HistoricalTradingDataTests(unittest.TestCase):

        def setUp(self):
            self.dates = pd.bdate_range('2017-03-01', periods=4)
            self.all_dates = pd.bdate_range('2016-01-01', '2019-01-01')
            self.raw = pd.DataFrame({'TradingDay': self.dates, 'SecuCode': '000001', 'OpenPrice': 10.0, 'ClosePrice': 10.0, 'PrevClosePrice': 10.0, 'HighPrice': 10.1, 'LowPrice': 9.9, 'PTFlag': 0, 'Volume': 1000.0, 'Amount': 10000.0, 'NonRestrictedShares': 1000000.0, 'AFloats': 1000000.0})
            self.st = pd.DataFrame(columns=KEYS + ['IsST'])
            self.suspend = pd.DataFrame(columns=KEYS + ['SuspendAtOpen', 'SuspendAny', 'SuspendFullDay'])
            self.limits = self.raw[KEYS].assign(up_limit=11.0, down_limit=9.0)
            self.factors = self.raw[KEYS].assign(adj_factor=1.0)
            self.temp = tempfile.TemporaryDirectory()
            self.cfg = Config(output_dir=Path(self.temp.name), one_way_fee=0)

        def tearDown(self):
            self.temp.cleanup()

        def prepare(self):
            symbols = trading_canonical(self.raw.SecuCode).unique()
            seen = {symbol: pd.Timestamp('2016-01-01') for symbol in symbols}
            return trading_prepare_daily(self.raw, self.dates, self.cfg, seen, self.all_dates, (self.st, self.suspend, self.limits, self.factors))

        def test_actual_upper_limit_blocks_buys_allows_sells(self):
            self.raw.loc[0, 'OpenPrice'] = 11.0
            day = self.prepare().iloc[0]
            self.assertFalse(day.CanBuy)
            self.assertTrue(day.CanSell)

        def test_actual_lower_limit_blocks_sells_allows_buys(self):
            self.raw.loc[0, 'OpenPrice'] = 9.0
            day = self.prepare().iloc[0]
            self.assertTrue(day.CanBuy)
            self.assertFalse(day.CanSell)

        def test_intraday_suspend_timing_is_evaluated_at_open(self):
            self.assertFalse(trading_suspension_at_open('13:00-15:00'))
            self.assertFalse(trading_suspension_at_open('09:31-10:01'))
            self.assertTrue(trading_suspension_at_open('09:30-10:00'))
            self.assertTrue(trading_suspension_at_open('09:30-10:31,10:31-14:57'))
            self.assertTrue(trading_suspension_at_open(np.nan))

        def test_afternoon_halt_does_not_block_morning_trade(self):
            self.suspend = pd.DataFrame([{'TradingDay': self.dates[0], 'SecuCode': '000001', 'SuspendAtOpen': False, 'SuspendAny': True, 'SuspendFullDay': False}])
            day = self.prepare().iloc[0]
            self.assertTrue(day.CanBuy)
            self.assertTrue(day.CanSell)
            self.assertFalse(day.SignalEligible)

        def test_open_halt_does_not_use_later_opening_price(self):
            self.suspend = pd.DataFrame([{'TradingDay': self.dates[1], 'SecuCode': '000001', 'SuspendAtOpen': True, 'SuspendAny': True, 'SuspendFullDay': False}])
            self.raw.loc[1, 'OpenPrice'] = 10.5
            day = self.prepare().iloc[1]
            self.assertTrue(np.isnan(day.AdjOpen))
            self.assertFalse(day.CanBuy)
            self.assertFalse(day.CanSell)

        def test_st_excluded_from_signal_and_new_risk_not_exits(self):
            self.st = pd.DataFrame([{'TradingDay': self.dates[0], 'SecuCode': '000001', 'IsST': True}])
            day = self.prepare().iloc[0]
            self.assertFalse(day.SignalEligible)
            self.assertFalse(day.CanIncrease)
            self.assertTrue(day.CanBuy)
            self.assertTrue(day.CanSell)

        def test_st_long_reduction_is_not_blocked_by_admission_rule(self):
            book = Portfolio(pd.Index(['000001']), self.cfg, 'Group10')
            yes = np.array([True])
            no = np.array([False])
            book.execute(np.array([10.0]), yes, yes, self.dates[0], np.array([1.0]), yes)
            book.execute(np.array([10.0]), yes, yes, self.dates[1], np.array([0.0]), no)
            self.assertEqual(book.units[0], 0)
            self.assertAlmostEqual(book.cash, 1.0)

        def test_st_short_cover_is_not_blocked_by_admission_rule(self):
            book = Portfolio(pd.Index(['000001']), self.cfg, 'TheoreticalLS')
            yes = np.array([True])
            no = np.array([False])
            book.execute(np.array([10.0]), yes, yes, self.dates[0], np.array([-1.0]), yes)
            book.execute(np.array([10.0]), yes, yes, self.dates[1], np.array([0.0]), no)
            self.assertEqual(book.units[0], 0)
            self.assertAlmostEqual(book.cash, 1.0)

        def test_future_st_or_limits_do_not_remove_current_signal(self):
            before = self.prepare()
            self.st = pd.DataFrame([{'TradingDay': self.dates[1], 'SecuCode': '000001', 'IsST': True}])
            self.limits.loc[1, 'up_limit'] = 10.0
            after = self.prepare()
            self.assertEqual(before.SignalEligible.iloc[0], after.SignalEligible.iloc[0])
            self.assertEqual(before.CanBuy.iloc[0], after.CanBuy.iloc[0])
            self.assertFalse(after.CanIncrease.iloc[1])
            self.assertFalse(after.CanBuy.iloc[1])

        def test_official_adjustment_and_future_invariance(self):
            self.raw.loc[1:, 'OpenPrice'] = 5.0
            self.raw.loc[1:, 'ClosePrice'] = 5.0
            self.factors.loc[1:, 'adj_factor'] = 2.0
            before = self.prepare()
            np.testing.assert_allclose(before.AdjClose, [10, 10, 10, 10])
            self.factors.loc[3, 'adj_factor'] = 3.0
            after = self.prepare()
            np.testing.assert_allclose(before.AdjClose.iloc[:3], after.AdjClose.iloc[:3])

        def test_unknown_limit_does_not_assume_unlimited_trading(self):
            self.limits = self.limits.drop(index=1)
            day = self.prepare().iloc[1]
            self.assertFalse(day.CanBuy)
            self.assertFalse(day.CanSell)
            self.assertTrue(day.SignalEligible)

        def test_conflicting_open_and_limit_bounds_block_orders_not_signal(self):
            self.raw.loc[0, 'OpenPrice'] = 12.0
            day = self.prepare().iloc[0]
            self.assertFalse(day.CanBuy)
            self.assertFalse(day.CanSell)
            self.assertTrue(day.SignalEligible)

        def test_missing_official_factor_fails_instead_of_dropping_stock(self):
            self.factors = self.factors.drop(index=1)
            with self.assertRaisesRegex(ValueError, '缺少官方复权因子'):
                self.prepare()

        def test_code_changes_have_one_continuous_security_identity(self):
            self.dates = pd.to_datetime(['2018-12-19', '2018-12-20', '2018-12-26', '2018-12-27'])
            self.raw.TradingDay = self.dates
            self.raw.SecuCode = ['000022', '000022', '001872', '001872']
            self.limits = self.raw[KEYS].copy()
            self.limits.SecuCode = '001872'
            self.limits = self.limits.assign(up_limit=11.0, down_limit=9.0)
            self.factors = self.raw[KEYS].copy()
            self.factors.SecuCode = '001872'
            self.factors = self.factors.assign(adj_factor=1.0)
            result = self.prepare()
            self.assertEqual(result.SecuCode.nunique(), 1)
            self.assertEqual(result.SecuCode.iloc[0], '001872')
            self.assertEqual(result.TradingCode.tolist(), ['000022', '000022', '001872', '001872'])
            np.testing.assert_allclose(result.AdjClose, 10)

        def test_duplicate_alias_values_cannot_silently_conflict(self):
            frame = pd.DataFrame({'TradingDay': [self.dates[0]] * 2, 'SecuCode': ['000022', '001872'], 'value': [1.0, 2.0]})
            frame.SecuCode = trading_canonical(frame.SecuCode)
            with self.assertRaisesRegex(ValueError, '冲突'):
                trading_deduplicate(frame, KEYS, ['value'], 'test')

        def test_new_st_at_execution_keeps_cash_without_reranking(self):
            dates = pd.bdate_range('2017-01-02', periods=8)
            symbols = pd.Index([f'{i:06d}' for i in range(20)])
            price = np.ones((8, 20))
            yes = np.ones_like(price, bool)
            entry = yes.copy()
            entry[1, 0] = False
            market = Market(dates, symbols, price, price, yes, yes, ~yes, entry)
            features = pd.DataFrame([{'TradingDay': d, 'SecuCode': s, 'f': float(i)} for d in dates for i, s in enumerate(symbols)])
            groups, schedule = signal_groups(features, 'f', dates, 5)
            self.assertIn('000000', set(groups[groups.TradingDay.eq(dates[0])].SecuCode))
            ledger, trades = backtest(groups, schedule, market, self.cfg)
            first = ledger[(ledger.TradingDay == dates[1]) & ledger.Portfolio.eq('Group10')].iloc[0]
            self.assertAlmostEqual(first.Cash, 0.5)
            self.assertEqual(first.Positions, 1)
            self.assertEqual(first.AdmissionBlockedOrders, 1)

    import unittest
    import numpy as np
    import pandas as pd

    class TunedModelTests(unittest.TestCase):

        def data(self):
            rng = np.random.default_rng(42)
            days = pd.bdate_range('2016-01-04', periods=220)
            data = []
            for i, day in enumerate(days):
                for s in range(30):
                    signal = float(rng.normal())
                    noise = float(rng.normal())
                    data.append({'TradingDay': day, 'SecuCode': f'{s:06d}', 'Target': signal + 0.3 * noise, 'a': signal, 'duplicate': signal, 'noise': noise, 'LabelEnd': days[min(i + 10, 219)]})
            return pd.DataFrame(data)

        def test_correlated_pair_keeps_better_representative(self):
            f = self.data()
            stats, corr = tuning_selector_statistics(f, ['a', 'duplicate', 'noise'])
            stats.loc[stats.Feature.eq('a'), 'Quality'] += 0.01
            chosen, log = tuning_choose_features(stats, corr, 0.8, 9)
            self.assertIn('a', chosen)
            self.assertNotIn('duplicate', chosen)
            self.assertEqual(log.loc[log.Feature.eq('duplicate'), 'Representative'].iloc[0], 'a')
            for a in chosen:
                for b in chosen:
                    if a != b:
                        self.assertLess(abs(corr.loc[a, b]), 0.8)

        def test_purged_validation_excludes_unrealized_training_labels(self):
            f = self.data()
            for train, valid in tuning_purged_folds(f):
                self.assertLess(train.LabelEnd.max(), valid.TradingDay.min())
                self.assertLess(train.TradingDay.max(), valid.TradingDay.min())

        def test_future_data_cannot_change_training_selection(self):
            f = self.data()
            cutoff = sorted(f.TradingDay.unique())[160]
            train = f[f.LabelEnd < cutoff]
            a = tuning_choose_features(*tuning_selector_statistics(train, ['a', 'duplicate', 'noise']))[0]
            f.loc[f.TradingDay >= cutoff, ['a', 'Target']] *= -100
            b = tuning_choose_features(*tuning_selector_statistics(f[f.LabelEnd < cutoff], ['a', 'duplicate', 'noise']))[0]
            self.assertEqual(a, b)

        def test_ten_groups_have_correct_size_and_best_direction(self):
            dates = pd.bdate_range('2017-01-02', periods=8)
            symbols = [f'{i:06d}' for i in range(103)]
            features = pd.DataFrame([{'TradingDay': d, 'SecuCode': s, 'f': i} for d in dates for i, s in enumerate(symbols)])
            groups, _ = signal_groups(features, 'f', dates, 5, n_groups=10)
            first = groups[groups.TradingDay.eq(dates[0])]
            self.assertEqual(first.Group.nunique(), 10)
            self.assertLessEqual(first.groupby('Group').size().max() - first.groupby('Group').size().min(), 1)
            self.assertEqual(set(first[first.Group.eq(10)].SecuCode), set(symbols[:10]))

        def test_round_trip_is_three_per_thousand(self):
            cfg = Config()
            self.assertAlmostEqual(cfg.one_way_fee * 2, 0.003)
            book = Portfolio(pd.Index(['stock']), cfg, 'Group10')
            yes = np.ones(1, bool)
            book.execute(np.ones(1), yes, yes, 'buy', np.ones(1))
            book.execute(np.ones(1), yes, yes, 'sell', np.zeros(1))
            for trade in book.trades:
                self.assertAlmostEqual(trade['Cost'] / trade['Notional'], 0.0015)
            self.assertAlmostEqual(sum((q['Cost'] for q in book.trades)) / book.trades[0]['Notional'], 0.003)

        def test_removed_factors_cannot_enter_active_catalog(self):
            self.assertEqual(len(RESEARCH_FACTORS), 9)
            self.assertFalse(any(('Tail' in f or 'Legacy' in f for f in RESEARCH_FACTORS)))

        def test_search_is_reproducible_and_covers_parameters(self):
            a = tuning_candidate_parameters(96, 42)
            b = tuning_candidate_parameters(96, 42)
            self.assertEqual(a, b)
            self.assertEqual(len(set((q['max_depth'] for q in a))), 5)
            self.assertEqual(set((q['corr_threshold'] for q in a)), {0.7, 0.8, 0.9})

    import unittest
    import numpy as np
    import pandas as pd

    class ModelTimingTests(unittest.TestCase):

        def setUp(self):
            self.dates = pd.bdate_range('2016-01-04', periods=100)
            self.symbols = pd.Index([f'{i:06d}' for i in range(20)])
            prices = 10 + np.arange(100)[:, None] * np.arange(1, 21)[None, :] / 100
            flags = np.ones_like(prices, dtype=bool)
            self.market = Market(self.dates, self.symbols, prices.copy(), prices.copy(), flags, flags, ~flags)
            self.features = pd.DataFrame([{'TradingDay': day, 'SecuCode': symbol, **{factor: float(i) for factor in _BASELINE_FACTORS}} for day in self.dates for i, symbol in enumerate(self.symbols)])

        def test_label_horizons_and_centering(self):
            for days in (5, 10, 20):
                labels = model_make_labels(self.features, self.market, days)
                first = labels[labels.TradingDay.eq(self.dates[0])]
                np.testing.assert_allclose(first.ForwardReturn, self.market.open[days + 1] / self.market.open[1] - 1)
                self.assertTrue(first.LabelEnd.eq(self.dates[days + 1]).all())
                self.assertAlmostEqual(first.Target.mean(), 0)

        def test_purge_uses_realization_date(self):
            labels = model_make_labels(self.features, self.market, 20)
            cutoff = self.dates[60]
            selected = model_training_sample(labels, cutoff)
            self.assertTrue(selected.LabelEnd.lt(cutoff).all())
            self.assertEqual(selected.TradingDay.max(), self.dates[38])
            self.assertNotIn(self.dates[50], set(selected.TradingDay))

        def test_window_excludes_old_signal_dates(self):
            labels = model_make_labels(self.features, self.market, 5)
            cutoff = self.dates[-1]
            selected = model_training_sample(labels, cutoff, window_months=1)
            self.assertTrue(selected.TradingDay.ge(cutoff - pd.DateOffset(months=1)).all())

        def test_future_prices_do_not_change_available_training_labels(self):
            cutoff = self.dates[60]
            before = model_training_sample(model_make_labels(self.features, self.market, 20), cutoff)
            self.market.open[60:] *= np.linspace(2, 3, 20)
            after = model_training_sample(model_make_labels(self.features, self.market, 20), cutoff)
            pd.testing.assert_frame_equal(before, after)

        def test_missing_future_price_preserves_prediction_input_rows(self):
            self.market.open[21, 0] = np.nan
            labels = model_make_labels(self.features, self.market, 20)
            self.assertEqual(len(labels), len(self.features))
            first = labels[(labels.TradingDay == self.dates[0]) & (labels.SecuCode == self.symbols[0])]
            self.assertTrue(first.Target.isna().all())
            self.assertEqual(len(self.features[self.features.TradingDay.eq(self.dates[0])]), 20)

        def test_prediction_score_direction_is_high_value_long(self):
            features = self.features.rename(columns={_BASELINE_FACTORS[0]: 'Pred_Score'})
            groups, _ = signal_groups(features, 'Pred_Score', self.dates, 5, low_is_long=False)
            first = groups[groups.TradingDay.eq(self.dates[0])]
            self.assertEqual(set(first[first.Group.eq(10)].SecuCode), set(self.symbols[-2:]))
    suite = unittest.TestSuite()
    for case in [CorrectedPipelineTests, PaperConstructionTests, HistoricalTradingDataTests, TunedModelTests, ModelTimingTests]:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    log = stream.getvalue()
    folder = Path(output_dir) / 'logs'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'tests.log').write_text(log)
    print(log, end='')
    if not result.wasSuccessful():
        raise RuntimeError('自动测试未通过，详见 results/logs/tests.log')
    return result.testsRun


# 报告图表及当前实验结果核验
def _render_report_contents():
    """从当前唯一results目录生成报告表格/图像，并逐项核对账本与时间隔离。"""
    from pathlib import Path
    import hashlib
    import json
    import os
    import re
    import sys
    import numpy as np
    import pandas as pd
    HERE = BASE / 'report'
    ROOT = HERE.parent
    R = ROOT / 'results'
    for p in [HERE / 'tables', HERE / 'figures']:
        p.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'legend.fontsize': 7, 'pdf.fonttype': 42, 'axes.spines.right': False, 'axes.spines.top': False})
    NAMES = {'HF_SmartMoney': 'SmartMoney', 'HF_CPV': 'CPV', 'HF_CDPDP': 'CDPDP-prime', 'HF_IdealAmplitude': 'IdealAmplitude', 'HF_IdealReversal': 'IdealReversal', 'HF_CDPP': 'CDPP', 'HF_CDPDP4': 'CDPDP-4', 'HF_LowAmpMomentum': 'LowAmpMomentum', 'HF_Skewness': 'Skewness', 'Pred_Score': 'XGBoost'}
    TEX = {**NAMES, 'HF_CDPDP': 'CDPDP$^{\\prime}$'}
    FREQ = {'Weekly': '5日', 'BiWeekly': '10日', 'Monthly': '20日', 'MonthEnd': '自然月'}
    full = pd.read_csv(R / 'factor_summary.csv')
    oos = pd.read_csv(R / 'oos_summary.csv')
    config = json.loads((R / 'run_config.json').read_text())
    assert config['groups'] == 10 and config['round_trip_fee'] == 0.003
    for name, digest in config['sources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    assert set(full.Factor) == set(RESEARCH_FACTORS) and len(RESEARCH_FACTORS) == 9
    assert not any(('Tail' in f or 'Legacy' in f for f in full.Factor.unique()))

    def pick(frame, factor, period, portfolio):
        return frame[frame.Factor.eq(factor) & frame.Period.eq(period) & frame.Portfolio.eq(portfolio)].iloc[0]

    def pct(x):
        return f'{x * 100:.2f}\\%'

    def line(*values):
        return ' & '.join((str(v) for v in values)) + ' \\\\' + '\n'

    def table(name, text):
        (HERE / 'tables' / f'{name}.tex').write_text(text)

    def save(fig, name):
        fig.savefig(HERE / 'figures' / f'{name}.pdf', bbox_inches='tight')
        fig.savefig(HERE / 'figures' / f'{name}.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
    counts = {'series': 0, 'trades': 0, 'ledger_rows': 0}
    for parent, summary in [(R / 'factors', full), (R / 'oos', oos)]:
        for factor, period in summary[['Factor', 'Period']].drop_duplicates().itertuples(index=False, name=None):
            folder = parent / factor / period
            trades = pd.read_parquet(folder / 'trades.parquet')
            ledger = pd.read_csv(folder / 'daily_ledger.csv', parse_dates=['TradingDay'])
            assert ledger.Portfolio.nunique() == 11
            assert set(ledger.Portfolio) == {*[f'Group{i}' for i in range(1, 11)], 'TheoreticalLS'}
            np.testing.assert_allclose(trades.Cost, trades.Notional * 0.0015, atol=1e-12)
            groups = pd.read_parquet(folder / 'signal_groups.parquet')
            if len(groups):
                assert set(groups.Group.unique()) == set(range(1, 11))
            for portfolio, f in ledger.groupby('Portfolio'):
                np.testing.assert_allclose(f.NAV, f.Cash + f.LongValue - f.ShortValue, atol=1e-10)
                np.testing.assert_allclose(f.NAV, np.cumprod(1 + f.Return), atol=1e-10)
                b = trades[trades.Portfolio.eq(portfolio)].copy()
                b['Flow'] = b.DeltaUnits * b.AdjPrice + b.Cost
                flow = b.groupby('TradingDay').Flow.sum().reindex(f.TradingDay, fill_value=0).to_numpy()
                np.testing.assert_allclose(f.Cash, 1 - np.cumsum(flow + f.BorrowCost.to_numpy()), atol=1e-10)
                metrics = performance(f.Return, Config())
                record = pick(summary, factor, period, portfolio)
                for key in ['FinalNAV', 'AnnualReturn', 'Sharpe', 'MaxDrawdown']:
                    np.testing.assert_allclose(record[key], metrics[key], rtol=1e-09, atol=1e-12)
                counts['series'] += 1
            counts['ledger_rows'] += len(ledger)
            counts['trades'] += len(trades)
    selection = []
    training = []
    best_rows = []
    coarse_total = fine_total = 0
    searches = 0
    for name, days in PERIODS.items():
        parent = R / 'models' / name
        log = pd.read_csv(parent / 'training_log.csv', parse_dates=['Cutoff', 'MaxLabelEnd', 'TuningCutoff'])
        assert len(log) == 23 and log.MaxLabelEnd.lt(log.Cutoff).all() and log.TuningCutoff.le(log.Cutoff).all()
        log['Period'] = name
        training.append(log)
        decisions = pd.read_csv(parent / 'feature_selection.csv', parse_dates=['Cutoff', 'MaxLabelEnd'])
        assert decisions.MaxLabelEnd.lt(decisions.Cutoff).all()
        assert not any(('Tail' in f or 'Legacy' in f for f in decisions.Feature.unique()))
        decisions['Period'] = name
        selection.append(decisions)
        for month, chosen in decisions[decisions.Selected].groupby('PredictionMonth'):
            corr = pd.read_csv(parent / 'selection' / f'{month}_correlation.csv', index_col=0)
            features = chosen.Feature.tolist()
            threshold = chosen.CorrelationThreshold.iloc[0]
            assert len(features) <= chosen.MaxFeatures.iloc[0]
            for a in features:
                for b in features:
                    if a != b:
                        assert abs(corr.loc[a, b]) < threshold + 1e-12
        for folder in sorted((parent / 'tuning').iterdir()):
            if not folder.is_dir():
                continue
            coarse = pd.read_csv(folder / 'coarse_trials.csv')
            fine = pd.read_csv(folder / 'fine_trials.csv')
            best = json.loads((folder / 'best_parameters.json').read_text())
            assert len(coarse) == 96 and len(fine) == 48 and (fine.Fold.nunique() == 3)
            cutoff = log.loc[log.TuningQuarter.eq(folder.name), 'TuningCutoff'].min()
            assert pd.Timestamp(best['max_training_label_end']) < cutoff
            splits = pd.read_csv(folder / 'splits.csv', parse_dates=['TrainMaxLabelEnd', 'ValidationStart', 'ValidationEnd', 'ValidationMaxLabelEnd'])
            assert splits.TrainMaxLabelEnd.lt(splits.ValidationStart).all()
            assert splits.ValidationMaxLabelEnd.lt(cutoff).all()
            best_rows.append({'Quarter': folder.name, 'Period': name, 'Days': days, 'MeanRankIC': best['inner_mean_rank_ic'], **best['parameters']})
            searches += 1
            coarse_total += len(coarse)
            fine_total += len(fine)
    selection = pd.concat(selection, ignore_index=True)
    training = pd.concat(training, ignore_index=True)
    best_frame = pd.DataFrame(best_rows).sort_values(['Quarter', 'Days'])
    assert searches == 24 and coarse_total == 2304 and (fine_total == 1152)
    training.to_csv(R / 'training_summary.csv', index=False)
    selection.to_csv(R / 'feature_selection_summary.csv', index=False)
    best_frame.to_csv(R / 'best_parameters.csv', index=False)
    text = ''
    for factor in RESEARCH_FACTORS:
        for i, period in enumerate(list(PERIODS) + ['MonthEnd']):
            long = pick(full, factor, period, 'Group10')
            ls = pick(full, factor, period, 'TheoreticalLS')
            text += line(TEX[factor] if i == 0 else '', FREQ[period], pct(long.AnnualReturn), f'{long.Sharpe:.2f}', pct(ls.AnnualReturn), f'{ls.Sharpe:.2f}')
        text += '\\midrule' + '\n' if factor != RESEARCH_FACTORS[-1] else ''
    table('factors', text)
    text = ''
    for factor in RESEARCH_FACTORS:
        numbers = [int(selection[selection.Period.eq(period) & selection.Feature.eq(factor)].Selected.sum()) for period in PERIODS]
        text += line(TEX[factor], *numbers)
    table('selection', text)
    text = ''
    for row in best_frame.itertuples(index=False):
        text += line(row.Quarter, FREQ[row.Period], row.max_depth, f'{row.learning_rate:.3f}', row.n_estimators, f'{row.corr_threshold:.2f}', row.max_features, f'{row.MeanRankIC:.3f}')
    table('tuning', text)
    text = ''
    for period in PERIODS:
        long = pick(oos, 'Pred_Score', period, 'Group10')
        ls = pick(oos, 'Pred_Score', period, 'TheoreticalLS')
        text += line(FREQ[period], pct(long.AnnualReturn), f'{long.Sharpe:.2f}', pct(ls.AnnualReturn), f'{ls.Sharpe:.2f}')
    table('model', text)
    all_performance = pd.concat([full.assign(Study='FullPeriod'), oos.assign(Study='OOS')], ignore_index=True)
    all_performance[all_performance.Portfolio.isin(['Group10', 'TheoreticalLS'])].to_csv(R / 'long_short_summary.csv', index=False)
    matrix = pd.read_csv(R / 'models/BiWeekly/selection/2017-02_correlation.csv', index_col=0)
    matrix = matrix.reindex(index=RESEARCH_FACTORS, columns=RESEARCH_FACTORS)
    matrix.to_csv(R / 'training_factor_correlation.csv')
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap='coolwarm')
    labels = [NAMES[f] for f in RESEARCH_FACTORS]
    ax.set_xticks(range(9), labels, rotation=45, ha='right')
    ax.set_yticks(range(9), labels)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f'{matrix.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    save(fig, 'correlation')
    calendar = pd.DatetimeIndex(sorted(pd.read_parquet(R / 'input/daily_market.parquet', columns=['TradingDay']).TradingDay.unique()))

    def nav_figure(folder, factor, period, filename):
        ledger = pd.read_csv(folder / factor / period / 'daily_ledger.csv', parse_dates=['TradingDay'])
        fig, axes = plt.subplots(2, 1, figsize=(8.1, 4.7), sharex=True, gridspec_kw={'height_ratios': [1.12, 1]})
        baseline = calendar[calendar.get_loc(ledger.TradingDay.min()) - 1]
        colors = plt.get_cmap('tab10').colors
        for i in range(1, 11):
            x = ledger[ledger.Portfolio.eq(f'Group{i}')]
            axes[0].plot([baseline, *x.TradingDay], [1.0, *x.NAV], color='black' if i == 10 else colors[i - 1], lw=1.6 if i == 10 else 0.7, label=f'G{i}', alpha=1 if i == 10 else 0.8)
        axes[0].legend(frameon=False, ncol=5, loc='lower left', fontsize=6.5)
        metrics = performance(ledger[ledger.Portfolio.eq('Group10')].Return, Config())
        axes[0].set_title(f"{NAMES[factor]} | {period} | G10 CAGR {metrics['AnnualReturn']:.2%} | Sharpe {metrics['Sharpe']:.2f}", fontsize=9)
        axes[0].set_ylabel('Long-only NAV')
        x = ledger[ledger.Portfolio.eq('TheoreticalLS')]
        axes[1].plot([baseline, *x.TradingDay], [1.0, *x.NAV], color='#234b68', lw=1.2)
        metrics = performance(x.Return, Config())
        axes[1].set_title(f"L/S 100%/100% | CAGR {metrics['AnnualReturn']:.2%} | Sharpe {metrics['Sharpe']:.2f}", fontsize=9)
        axes[1].set_ylabel('Net L/S NAV')
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.axhline(1, color='gray', lw=0.5)
        fig.tight_layout()
        save(fig, filename)
    for period in PERIODS:
        nav_figure(R / 'oos', 'Pred_Score', period, 'model_' + period)
    for factor in RESEARCH_FACTORS:
        nav_figure(R / 'factors', factor, 'MonthEnd', factor + '_MonthEnd')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for period in PERIODS:
        d = pd.read_csv(R / 'oos/Pred_Score' / period / 'daily_ledger.csv', parse_dates=['TradingDay'])
        for ax, portfolio in zip(axes, ['Group10', 'TheoreticalLS']):
            f = d[d.Portfolio.eq(portfolio)]
            ax.plot(f.TradingDay, f.NAV, label=period, lw=1)
    for ax, title in zip(axes, ['G10 long-only', 'L/S 100% / 100%']):
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        ax.set_ylabel('Net NAV')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    save(fig, 'model_horizons')
    appendix = ''
    for i in range(0, 9, 2):
        appendix += '\\clearpage' + '\n'
        if i == 0:
            appendix += '\\section{九因子自然月末十组净值}' + '\n'
        for j, factor in enumerate(RESEARCH_FACTORS[i:i + 2]):
            if j:
                appendix += '\\vfill' + '\n'
            appendix += f'\\resultfigure{{{factor}_MonthEnd}}{{{TEX[factor]}自然月末回测}}' + '\n'
    (HERE / 'appendix_figures.tex').write_text(appendix)
    test_log = (R / 'logs/tests.log').read_text()
    tests = int(re.search('Ran (\\d+) tests', test_log).group(1))
    assert '\nOK' in test_log
    macros = {'Searches': searches, 'CoarseFits': coarse_total, 'FineFits': fine_total, 'CVFits': coarse_total + fine_total, 'Models': len(training), 'Series': counts['series'], 'Trades': f"{counts['trades']:,}", 'Tests': tests}
    for period, suffix in [('Weekly', 'Week'), ('BiWeekly', 'Biweek'), ('Monthly', 'Month')]:
        macros['Long' + suffix] = pct(pick(oos, 'Pred_Score', period, 'Group10').AnnualReturn)
        macros['LS' + suffix] = pct(pick(oos, 'Pred_Score', period, 'TheoreticalLS').AnnualReturn)
    (HERE / 'values.tex').write_text('\n'.join(('\\newcommand{\\' + k + '}{' + str(v) + '}' for k, v in macros.items())) + '\n')
    template = (HERE / 'template.tex').read_text()
    for path in (HERE / 'tables').glob('*.tex'):
        template = template.replace('\\input{tables/' + path.name + '}', path.read_text())
    (HERE / 'report_ready.tex').write_text(template)
    receipt = {**counts, 'tests_passed': tests, 'groups': 10, 'candidates': list(RESEARCH_FACTORS), 'monthly_models': len(training), 'searches': searches, 'cv_fits': coarse_total + fine_total, 'round_trip_fee': 0.003, 'training_label_time_checked': True, 'inner_purge_checked': True, 'selected_pair_correlations_checked': True, 'source_hashes': config['sources']}
    (R / 'validation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def build_report():
    """系统临时目录编译两次，作业根目录只保留最终report.pdf。"""
    import shutil
    import subprocess
    import tempfile
    if shutil.which('xelatex') is None:
        raise RuntimeError('生成报告需要XeLaTeX，请先安装TeX Live或MacTeX。')
    run_self_tests()
    with tempfile.TemporaryDirectory(prefix='code2-report-') as temporary:
        work = Path(temporary)
        os.environ['MPLCONFIGDIR'] = str(work / 'matplotlib')
        _render_report_contents()
        for iteration in [1, 2]:
            command = ['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                       '-jobname=report', '-output-directory=' + str(work), 'report_ready.tex']
            completed = subprocess.run(command, cwd=BASE / 'report', capture_output=True, text=True)
            (BASE / 'results/logs' / f'xelatex_pass{iteration}.log').write_text(completed.stdout + completed.stderr)
            if completed.returncode:
                raise RuntimeError(f'XeLaTeX编译失败，详见results/logs/xelatex_pass{iteration}.log')
        shutil.copy2(work / 'report.pdf', BASE / 'report.pdf')
    LOG.info('报告已生成：%s', BASE / 'report.pdf')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=BASE / 'data')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'results')
    parser.add_argument('--with-xgboost', action='store_true', help='完整执行训练期筛选和调参')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--models-only', action='store_true')
    parser.add_argument('--self-test', action='store_true', help='运行内置65项自动测试')
    parser.add_argument('--report', action='store_true', help='核验结果并生成根目录report.pdf')
    parser.add_argument('--predict-start', default='2017-02-01')
    parser.add_argument('--trials', type=int, default=96)
    parser.add_argument('--finalists', type=int, default=16)
    parser.add_argument('--groups', type=int, default=10)
    parser.add_argument('--one-way-fee', type=float, default=.0015)
    parser.add_argument('--ls-leg-weight', type=float, default=1.)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.report:
        build_report()
        return
    if args.self_test:
        run_self_tests(args.output_dir)
        return
    cfg = Config(data_dir=args.data_dir, output_dir=args.output_dir, n_groups=args.groups,
                 one_way_fee=args.one_way_fee, ls_leg_weight=args.ls_leg_weight)
    pipeline_run(cfg, with_models=args.with_xgboost or args.models_only, models_only=args.models_only,
                 prepare_only=args.prepare_only, predict_start=args.predict_start,
                 trials=args.trials, finalists=args.finalists)


if __name__ == '__main__':
    main()
