"""历史ST、停复牌、实际涨跌停价和官方复权因子的时点对齐。

稀疏ST/停牌文件的记录表示该日处于该状态，未列出的日期视为无该状态。
空停牌时段按全天停牌处理；已给出时段时仅依据09:30当时的状态限制开盘交易。
开盘等于涨停价时不买（含空头回补），等于跌停价时不卖（含新开空头）。
不根据盘中高低价判断早盘成交，不按T+1状态事先删除T日信号。
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

import code2_new as base


ALIASES = {"601313": "601360", "000022": "001872"}
CODE_CHANGES = {"601360": ("601313", "2018-02-28"), "001872": ("000022", "2018-12-26")}
ALIAS_SOURCES = {
    "601360": "https://static.cninfo.com.cn/finalpage/2018-02-28/1204434951.PDF",
    "001872": "https://static.cninfo.com.cn/finalpage/2018-12-26/1205690369.PDF",
}


def canonical(values):
    return base.normalize_codes(values).replace(ALIASES)


def raw_symbols(symbols):
    return sorted(set(symbols) | {old for old,new in ALIASES.items() if new in set(symbols)})


def trading_code(dates, symbols):
    result = pd.Series(symbols).astype(str).copy()
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    result = result.reset_index(drop=True)
    for new,(old,start) in CODE_CHANGES.items():
        result.loc[result.eq(new) & dates.lt(pd.Timestamp(start))] = old
    return result.to_numpy()


def suspension_at_open(timing):
    if pd.isna(timing) or not str(timing).strip():
        return True
    intervals = re.findall(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", str(timing))
    if not intervals:
        raise ValueError(f"无法识别停牌时段：{timing}")
    opening = 9*60+30
    return any(int(h1)*60+int(m1) <= opening < int(h2)*60+int(m2) for h1,m1,h2,m2 in intervals)


def deduplicate(frame, keys, value_columns, name, absolute_tolerance=1e-8):
    """代码更名历史回填可能产生同一证券两条相同记录，冲突时停止。"""
    if frame.duplicated(keys).any():
        duplicate = frame[frame.duplicated(keys,keep=False)]
        for column in value_columns:
            spread = duplicate.groupby(keys)[column].agg(["min","max"])
            if pd.api.types.is_numeric_dtype(frame[column]):
                conflicts = (spread["max"]-spread["min"]).abs() > absolute_tolerance
            else:
                conflicts = spread["min"].ne(spread["max"])
            if conflicts.any():
                raise ValueError(f"{name}代码归并后有冲突记录：{column}")
        frame = frame.drop_duplicates(keys,keep="last")
    return frame


def input_manifest(data_dir):
    root = Path(data_dir)
    paths = [root/"ST情况/is_st.csv", root/"停复牌情况/is_suspend.csv"]
    paths += sorted((root/"涨跌停价").glob("*.csv"))
    paths += sorted((root/"复权因子").glob("*.csv"))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    records = [{"path":str(path.resolve()),"size":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns} for path in paths]
    return {"files":records,"signature":hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()}


def load_tables(data_dir, calendar, symbols):
    root = Path(data_dir)
    wanted = set(symbols)

    def read(path):
        frame = pd.read_csv(path,dtype={"symbol":str})
        frame = frame.rename(columns={"symbol":"SecuCode","trade_date":"TradingDay"})
        original = base.normalize_codes(frame.SecuCode)
        frame.SecuCode = canonical(frame.SecuCode)
        frame["_CanonicalSource"] = original.eq(frame.SecuCode)
        frame.TradingDay = pd.to_datetime(frame.TradingDay)
        return frame[frame.SecuCode.isin(wanted) & frame.TradingDay.isin(calendar)].copy()

    st = read(root/"ST情况/is_st.csv")
    st = st[base.KEYS].drop_duplicates().assign(IsST=True)
    suspend = read(root/"停复牌情况/is_suspend.csv")
    suspend["SuspendAtOpen"] = suspend.suspend_timing.map(suspension_at_open)
    suspend["SuspendAny"] = True
    suspend["SuspendFullDay"] = suspend.suspend_timing.isna() | suspend.suspend_timing.astype(str).str.strip().eq("")
    # 双代码历史回填按停牌区间并集处理，不能用一条局部状态覆盖另一条全天状态。
    suspend = suspend.groupby(base.KEYS,as_index=False)[["SuspendAtOpen","SuspendAny","SuspendFullDay"]].max()
    limits, factors = [], []
    for year in range(calendar[0].year,calendar[-1].year+1):
        lp = root/f"涨跌停价/A_share_limit_{year}.csv"
        ap = root/f"复权因子/adj_factor_{year}.csv"
        if not lp.exists() or not ap.exists():
            raise FileNotFoundError(f"{year}年实际涨跌停价或复权因子缺失。")
        limits.append(read(lp))
        factors.append(read(ap))
    limits = pd.concat(limits,ignore_index=True).sort_values("_CanonicalSource")
    factors = pd.concat(factors,ignore_index=True).sort_values("_CanonicalSource")
    limits = deduplicate(limits,base.KEYS,["up_limit","down_limit"],"涨跌停价").drop(columns="_CanonicalSource")
    # 000022历史因子有3位小数，001872对应历史有4位；仅容许半个旧精度单位，优先当前代码原始记录。
    factors = deduplicate(factors,base.KEYS,["adj_factor"],"复权因子",absolute_tolerance=0.0005001).drop(columns="_CanonicalSource")
    if (factors.adj_factor.le(0) | ~np.isfinite(factors.adj_factor)).any():
        raise ValueError("官方复权因子必须为正且有限。")
    limits["InvalidLimitSource"] = ((limits.up_limit < limits.down_limit) | limits.up_limit.le(0)
        | limits.down_limit.le(0) | ~np.isfinite(limits.up_limit) | ~np.isfinite(limits.down_limit))
    return st,suspend,limits,factors


def prepare_daily(raw, calendar, cfg, first_seen, all_calendar, tables):
    """官方因子构造连续价格；信号池与开盘执行状态分别生成。"""
    raw = raw.copy()
    raw.SecuCode = canonical(raw.SecuCode)
    raw.TradingDay = pd.to_datetime(raw.TradingDay)
    raw = deduplicate(raw,base.KEYS,["OpenPrice","ClosePrice","PrevClosePrice","Volume","Amount"],"日行情")
    st,suspend,limits,factors = tables
    merged = raw.merge(st,on=base.KEYS,how="left",validate="one_to_one")
    merged = merged.merge(suspend,on=base.KEYS,how="left",validate="one_to_one")
    merged = merged.merge(limits,on=base.KEYS,how="left",validate="one_to_one")
    merged = merged.merge(factors,on=base.KEYS,how="left",validate="one_to_one")
    for column in ["IsST","SuspendAtOpen","SuspendAny","SuspendFullDay"]:
        merged[column] = merged[column].eq(True)
    state_indexes = [(table.set_index(["SecuCode","TradingDay"]).sort_index(),columns,set(table.SecuCode))
                     for table,columns in [(st,["IsST"]),(suspend,["SuspendAtOpen","SuspendAny","SuspendFullDay"])]]
    factor_index = factors.set_index(["SecuCode","TradingDay"]).adj_factor.sort_index()
    factor_symbols = set(factors.SecuCode)
    parts = []
    for symbol,group in merged.groupby("SecuCode",sort=True):
        g = group.set_index("TradingDay").reindex(calendar).copy()
        g.index.name = "TradingDay"
        # 稀疏状态也须合入无报价日，否则会漏掉停牌记录和停牌期间的ST变更。
        for indexed,columns,known_symbols in state_indexes:
            t = indexed.loc[symbol] if symbol in known_symbols else pd.DataFrame(columns=columns,index=pd.DatetimeIndex([]))
            for column in columns:
                g[column] = t[column].reindex(calendar).eq(True)
        factor = factor_index.loc[symbol].reindex(calendar) if symbol in factor_symbols else pd.Series(np.nan,index=calendar)
        observed_close = g.ClosePrice.gt(0) & np.isfinite(g.ClosePrice) & g.PTFlag.eq(0)
        if (observed_close & factor.isna()).any():
            bad = g.index[observed_close & factor.isna()][0]
            raise ValueError(f"{symbol} {bad.date()}正常报价缺少官方复权因子。")
        factor_asof = factor.ffill()
        observed_factors = factor.dropna()
        scale = factor_asof / observed_factors.iloc[0] if len(observed_factors) else factor_asof
        g["Scale"] = scale
        g["AdjFactor"] = factor_asof
        g["AdjFactorExact"] = factor.notna()
        full_halt = g.SuspendFullDay | g.PTFlag.eq(1)
        halt_open = g.SuspendAtOpen | g.PTFlag.eq(1)
        known = observed_close | full_halt
        # 开盘停牌而10点复牌时，不用10点后的首笔价格为09:30持仓估值或定仓。
        quoted_open = g.OpenPrice.gt(0) & np.isfinite(g.OpenPrice) & g.PTFlag.eq(0) & ~halt_open
        g["AdjOpen"] = (g.OpenPrice*scale).where(quoted_open)
        g["AdjClose"] = (g.ClosePrice*scale).where(observed_close).ffill()
        g["StaleQuote"] = ~known
        bounds = (g.up_limit.gt(0) & g.down_limit.gt(0) & np.isfinite(g.up_limit)
                  & np.isfinite(g.down_limit) & g.up_limit.ge(g.down_limit))
        g["LimitKnown"] = bounds
        in_range = bounds & (g.OpenPrice>=g.down_limit-1e-6) & (g.OpenPrice<=g.up_limit+1e-6)
        g["OpenWithinBounds"] = in_range
        # 1厘浮点容差小于A股一分报价单位；相等就保守视为不可在该方向成交。
        g["CanBuy"] = quoted_open & in_range & (g.OpenPrice < g.up_limit-1e-6)
        g["CanSell"] = quoted_open & in_range & (g.OpenPrice > g.down_limit+1e-6)
        g["CanIncrease"] = ~g.IsST
        positions = all_calendar.get_indexer(calendar)
        first_position = all_calendar.searchsorted(first_seen[symbol])
        g["Age"] = positions-first_position+1
        g["SignalEligible"] = observed_close & ~g.IsST & ~g.SuspendAny & ~g.PTFlag.eq(1) & g.Age.ge(60)
        returns = g.AdjClose.pct_change(fill_method=None).where(known & known.shift(1,fill_value=False))
        complete = known.rolling(21,min_periods=21).sum().eq(21)
        g["Ctrl_Ret20"] = (g.AdjClose/g.AdjClose.shift(20)-1).where(complete)
        g["Ctrl_Vol20"] = returns.rolling(20,min_periods=20).std()
        floats = g.NonRestrictedShares.where(g.NonRestrictedShares.gt(0),g.AFloats)
        turnover = (g.Volume/floats).where(observed_close & floats.gt(0)).mask(full_halt,0.)
        g["Ctrl_Turnover20"] = turnover.rolling(20,min_periods=20).mean()
        g["DailyReturn"] = returns
        g["DailyAmplitude"] = (g.HighPrice/g.LowPrice-1).where(g.LowPrice.gt(0)&observed_close)
        g["SecuCode"] = symbol
        g["TradingCode"] = trading_code(calendar,pd.Series(symbol,index=calendar))
        parts.append(g.reset_index())
    daily = pd.concat(parts,ignore_index=True)
    active = daily.PTFlag.eq(0) & daily.OpenPrice.gt(0) & daily.ClosePrice.gt(0)
    diagnostics = {
        "rows":len(daily), "normal_quoted_rows":int(active.sum()),
        "st_stock_days":int(daily.IsST.sum()),"suspend_stock_days":int(daily.SuspendAny.sum()),
        "intraday_only_suspend_stock_days":int((daily.SuspendAny&~daily.SuspendFullDay).sum()),
        "open_suspended_stock_days":int(daily.SuspendAtOpen.sum()),
        "normal_quotes_missing_limits":int((active&~daily.LimitKnown).sum()),
        "invalid_limit_source_rows":int(limits.InvalidLimitSource.sum()) if "InvalidLimitSource" in limits else 0,
        "normal_quotes_missing_adjustment":int((active&~daily.AdjFactorExact).sum()),
        "normal_quotes_outside_supplied_limits":int((active&daily.LimitKnown&~daily.OpenWithinBounds).sum()),
        "open_at_upper":int((active&daily.LimitKnown&(daily.OpenPrice>=daily.up_limit-1e-6)).sum()),
        "open_at_lower":int((active&daily.LimitKnown&(daily.OpenPrice<=daily.down_limit+1e-6)).sum()),
        "suspend_records_with_positive_open":int((daily.SuspendAtOpen&active).sum()),
        "security_aliases":ALIASES,"alias_sources":ALIAS_SOURCES,
        "input_files":input_manifest(cfg.data_dir),
    }
    output = Path(cfg.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    (output/"trading_data_diagnostics.json").write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2))
    daily.loc[active&~daily.LimitKnown,base.KEYS+["OpenPrice","IsST"]].to_csv(output/"missing_limits.csv",index=False)
    daily.loc[active&daily.LimitKnown&~daily.OpenWithinBounds,
              base.KEYS+["OpenPrice","up_limit","down_limit","Age"]].to_csv(output/"inconsistent_limits.csv",index=False)
    return daily


def industry_cap(loader,calendar,symbols):
    industry,cap = loader.industry_cap(calendar,raw_symbols(symbols))
    industry.SecuCode = canonical(industry.SecuCode)
    cap.SecuCode = canonical(cap.SecuCode)
    cap = deduplicate(cap,base.KEYS,["LnCap"],"市值")
    industry = industry.drop_duplicates(["SecuCode","InfoPublDate","tag"]).sort_values("InfoPublDate")
    return industry,cap
