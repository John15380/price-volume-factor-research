"""逐篇核对本地研报后的因子实现。原code2.py保持不变。

来源：东吴20200223 pp.5/9/11-12；东吴20210301 pp.9-13/19；
开源20210924 pp.4/7-8。20221218依赖主动/非主动大小单字段，现有分钟bar
无法识别，明确不冒充复现。已接入历史ST、停复牌、实际涨跌停价和官方复权
因子；60日上市年龄仍以历史首次有效报价日衡量。
"""
from __future__ import annotations

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

import code2_new as base
import code2_xgboost as boosting
import code2_trading_data as trading

LOG = logging.getLogger("code2_research")
PRIMARY = ("HF_SmartMoney", "HF_CPV", "HF_CDPDP", "HF_IdealAmplitude", "HF_IdealReversal", "HF_CDPP")
ADDITIONAL = ("HF_CDPDP4", "HF_LowAmpMomentum", "HF_Skewness")
FACTORS = PRIMARY + ADDITIONAL
LOW_IS_LONG = {factor: factor != "HF_LowAmpMomentum" for factor in FACTORS}
DAY_FIELDS = ("PV", "PP", "PM", "MP", "MM", "DP_Pos_P", "DP_Neg_P", "Skew", "Tail", "OrdinaryAmp", "SmartDaily", "AvgTrade")


@njit(cache=True)
def correlation(x, y):
    # 原code2中的中性约定：零方差/不足3对的子序列记0；不是整个停牌日填0。
    if len(x) < 3:
        return 0.0
    a, b = x - np.mean(x), y - np.mean(y)
    denom = np.sqrt(np.sum(a*a) * np.sum(b*b))
    return np.sum(a*b) / denom if denom > 1e-14 else 0.0


@njit(cache=True)
def smart_ratio(scores, volume, amount):
    total_v, total_a = volume.sum(), amount.sum()
    if total_v <= 0 or total_a <= 0:
        return np.nan
    target, used, selected_amount = 0.2 * total_v, 0.0, 0.0
    for i in np.argsort(scores)[::-1]:
        if volume[i] <= 0:
            continue
        take = min(volume[i], target - used)
        selected_amount += amount[i] * take / volume[i]
        used += take
        if used >= target - total_v * 1e-12:
            break
    return (selected_amount / target) / (total_a / total_v)


@njit(cache=True)
def day_metrics(open_, close, high, low, volume, amount, trades, scale):
    n = len(close)
    if n < 31 or volume.sum() <= 0 or amount.sum() <= 0:
        return np.full(len(DAY_FIELDS), np.nan), np.empty(0), np.empty(0), np.empty(0)
    # 连续价格对应的份额单位；amount保持实际货币金额。
    adjusted_volume = volume / scale
    score = np.full(n, -np.inf)
    for i in range(n):
        prior = open_[0] if i == 0 else close[i-1]
        if adjusted_volume[i] > 0:
            score[i] = abs(close[i]/prior-1) / adjusted_volume[i] ** 0.25
    changes = close[1:] - close[:-1]
    x, y = changes[:-1], changes[1:]
    pp, pm = (x > 0) & (y > 0), (x > 0) & (y < 0)
    mp, mm = (x < 0) & (y > 0), (x < 0) & (y < 0)
    # 单序列差分：DeltaP_t 与 P_(t+1)，先按DeltaP的正负分组。
    next_price = close[2:]
    positive, negative = x > 0, x < 0
    returns = close[1:] / close[:-1] - 1
    centered = returns - returns.mean()
    variance = np.mean(centered**2)
    skew = np.mean(centered**3) / variance**1.5 if variance > 1e-15 else 0.0
    count = trades.sum()
    values = np.array([
        correlation(close, volume), correlation(x[pp], y[pp]), correlation(x[pm], y[pm]),
        correlation(x[mp], y[mp]), correlation(x[mm], y[mm]),
        correlation(x[positive], next_price[positive]), correlation(x[negative], next_price[negative]),
        skew, close[-1]/close[-31]-1, (high.max()-low.min())/open_[0],
        smart_ratio(score, adjusted_volume, amount), amount.sum()/count if count > 0 else np.nan])
    return values, score, adjusted_volume, amount


@njit(cache=True)
def conditional_factors(prices, amplitude, returns, avg_trade, ideal_valid=None):
    n = len(prices)
    out = np.full((n, 3), np.nan)
    for t in range(19, n):
        a, p = amplitude[t-19:t+1], prices[t-19:t+1]
        r, v = returns[t-19:t+1], avg_trade[t-19:t+1]
        valid = np.isfinite(a) & np.isfinite(p) & (p > 0) & (a >= 0)
        if ideal_valid is not None:
            valid &= ideal_valid[t-19:t+1]
        # 开源20200516 p.5：剔除停牌和一字涨跌停，有效日不足10日才置空。
        # 开源20210924 p.8：高/低价各取25%。非整数日数向下取整，原文未指定舍入。
        count = valid.sum()
        if count >= 10:
            valid_a, valid_p = a[valid], p[valid]
            order = np.argsort(valid_p)
            k = count // 4
            out[t, 0] = np.mean(valid_a[order[-k:]]) - np.mean(valid_a[order[:k]])
        if np.isfinite(r).all() and np.isfinite(v).all():
            order = np.argsort(v)
            out[t, 1] = np.sum(r[order[-10:]]) - np.sum(r[order[:10]])
        if t >= 159:
            a160, r160 = amplitude[t-159:t+1], returns[t-159:t+1]
            if np.isfinite(a160).all() and np.isfinite(r160).all():
                order = np.argsort(a160)
                out[t, 2] = np.sum(r160[order[:112]])
    return out


def ideal_valid_days(quotes):
    """信号形成前可知的有效日，独立于次日订单能否成交。"""
    high, low, close = quotes.HighPrice, quotes.LowPrice, quotes.ClosePrice
    upper, lower = quotes.up_limit, quotes.down_limit
    same_price = (high-low).abs().le(1e-6)
    one_price_limit = same_price & ((close-upper).abs().le(1e-6) | (close-lower).abs().le(1e-6))
    valid = (np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
             & low.gt(0) & high.ge(low) & close.gt(0) & quotes.Volume.gt(0)
             & quotes.PTFlag.eq(0) & ~quotes.SuspendAny & ~one_price_limit)
    return valid


def load_data(cfg, universe_name):
    loader = base.DataLoader(cfg)
    all_calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(Path(cfg.data_dir)/"交易日期/tradingdays.csv").TradingDay)).sort_values()
    all_calendar = all_calendar[all_calendar <= pd.Timestamp(cfg.end)]
    first = all_calendar.searchsorted(pd.Timestamp(cfg.start))
    calendar = all_calendar[max(0, first-200):]
    universe = loader.universe(calendar) if universe_name == "500" else None
    if universe is not None:
        universe.SecuCode = trading.canonical(universe.SecuCode)
        universe = universe.drop_duplicates(base.KEYS)
    wanted = set(universe.SecuCode.unique()) if universe is not None else None
    first_seen, parts = {}, []
    fields = ["SecuCode", "TradingDay", "OpenPrice", "ClosePrice", "PrevClosePrice", "HighPrice", "LowPrice",
              "PTFlag", "TradingVolumes", "TurnoverValue", "NonRestrictedShares", "AFloats"]
    for path in sorted((Path(cfg.data_dir)/"日行情").glob("daily_quote_*.csv")):
        year = int(path.stem.rsplit("_", 1)[-1])
        if year > calendar[-1].year:
            continue
        columns = fields if year >= calendar[0].year else base.KEYS+["ClosePrice"]
        for chunk in pd.read_csv(path, usecols=columns, dtype={"SecuCode": str}, chunksize=250000):
            chunk.SecuCode = trading.canonical(chunk.SecuCode)
            if wanted is not None:
                chunk = chunk[chunk.SecuCode.isin(wanted)].copy()
            else:
                chunk = chunk[chunk.SecuCode.str.startswith(("000", "001", "002", "003", "300", "600", "601", "603", "605", "688"))].copy()
            chunk.TradingDay = pd.to_datetime(chunk.TradingDay)
            for symbol, date in chunk.loc[chunk.ClosePrice.gt(0)].groupby("SecuCode").TradingDay.min().items():
                first_seen[symbol] = min(first_seen.get(symbol, date), date)
            if year >= calendar[0].year:
                parts.append(chunk[chunk.TradingDay.isin(calendar)])
    raw = pd.concat(parts, ignore_index=True).rename(columns={"TradingVolumes":"Volume", "TurnoverValue":"Amount"})
    symbols = sorted(raw.SecuCode.unique())
    tables = trading.load_tables(cfg.data_dir,calendar,symbols)
    daily = trading.prepare_daily(raw,calendar,cfg,first_seen,all_calendar,tables)
    daily.to_parquet(Path(cfg.output_dir)/"daily_market.parquet",index=False)
    eligible = daily.loc[daily.SignalEligible,base.KEYS].copy()
    if universe is not None:
        eligible = eligible.merge(universe, on=base.KEYS, validate="one_to_one")
    industry, cap = trading.industry_cap(loader,calendar,symbols)
    return calendar, symbols, daily, eligible, industry, cap


def calculate_factors(cfg, calendar, symbols, daily):
    cache = Path(cfg.output_dir)/"research_cache"
    cache.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256((inspect.getsource(day_metrics.py_func)+inspect.getsource(correlation.py_func)
        +inspect.getsource(smart_ratio.py_func)).encode()).hexdigest()
    dependency = trading.input_manifest(cfg.data_dir)["signature"]
    scale = daily.set_index(base.KEYS).Scale
    histories = defaultdict(deque)
    files = sorted((Path(cfg.data_dir)/"bar_1m_2016_2018").glob("*.parquet"))
    parts = []
    date_index = {d:i for i,d in enumerate(calendar)}
    for path in tqdm([p for p in files if pd.Timestamp(p.stem) in date_index], desc="研报分钟因子"):
        day = pd.Timestamp(path.stem)
        position = date_index[day]
        cached = cache/path.name
        meta = cached.with_suffix(".json")
        key = {"source":str(path.resolve()),"mtime_ns":path.stat().st_mtime_ns,"size":path.stat().st_size,
               "definition":fingerprint,"symbols":symbols,"pooled_window":10,"trading_inputs":dependency,
               "aliases":trading.ALIASES}
        # 缓存直接保存已完成的10日池化结果。重算时从当前输出首日顺序恢复历史。
        if cached.exists() and meta.exists() and json.loads(meta.read_text()) == key:
            parts.append(pd.read_parquet(cached))
            continue
        # 若需要从缓存中间重新开始，先回读前9日分钟，仅恢复池化历史，不输出旧结果。
        if not histories and parts:
            raise ValueError("分钟缓存不完整：请使用新的output-dir完整重算，避免丢失10日池化历史。")
        columns = ["time","symbol","open","close","high","low","volume","amount","trade_num"]
        bars = ds.dataset(path,format="parquet").to_table(columns=columns,filter=ds.field("symbol").isin(trading.raw_symbols(symbols))).to_pandas()
        bars.symbol = trading.canonical(bars.symbol)
        bars = bars.sort_values(["symbol","time"])
        rows = []
        for symbol, frame in bars.groupby("symbol",sort=False):
            valid = np.isfinite(frame[["open","close","high","low","volume","amount"]]).all(axis=1)
            valid &= frame[["open","close","high","low"]].gt(0).all(axis=1) & frame.volume.ge(0) & frame.amount.ge(0)
            frame = frame.loc[valid]
            a = scale.get((day, symbol), np.nan)
            if not np.isfinite(a) or a <= 0:
                continue
            values, score, volume, amount = day_metrics(*[frame[c].to_numpy(float) for c in
                ["open","close","high","low","volume","amount","trade_num"]], a)
            history = histories[symbol]
            while history and history[0][0] <= position-10:
                history.popleft()
            if len(score):
                history.append((position,score,volume,amount))
            pooled = np.nan
            if len(history) == 10 and history[-1][0]-history[0][0] == 9:
                pooled = smart_ratio(np.concatenate([x[1] for x in history]),
                                     np.concatenate([x[2] for x in history]),np.concatenate([x[3] for x in history]))
            rows.append([day,symbol,*values,pooled])
        part = pd.DataFrame(rows,columns=base.KEYS+list(DAY_FIELDS)+["SmartPooled"])
        part.to_parquet(cached,index=False)
        meta.write_text(json.dumps(key,ensure_ascii=False))
        parts.append(part)
    raw = pd.concat(parts,ignore_index=True)
    raw.to_parquet(Path(cfg.output_dir)/"daily_research_metrics.parquet",index=False)
    rolled = []
    daily_index = daily.set_index(["SecuCode","TradingDay"])[[
        "AdjClose","DailyAmplitude","DailyReturn","HighPrice","LowPrice","ClosePrice",
        "up_limit","down_limit","Volume","PTFlag","SuspendAny"]].sort_index()
    for symbol, frame in tqdm(raw.groupby("SecuCode"),desc="研报时间窗口"):
        grid = frame.set_index("TradingDay").reindex(calendar)
        rolling = grid[list(DAY_FIELDS)].rolling(20,min_periods=20).mean()
        rolling["PVStd"] = grid.PV.rolling(20,min_periods=20).std(ddof=1)
        weights = np.arange(20,dtype=float)-9.5
        rolling["PVTrend"] = grid.PV.rolling(20,min_periods=20).apply(lambda x: np.dot(x,weights)/np.dot(weights,weights),raw=True)
        rolling["SmartPooled"] = grid.SmartPooled
        rolling["SmartDaily"] = grid.SmartDaily.rolling(10,min_periods=10).mean()
        quotes = daily_index.loc[symbol].reindex(calendar)
        selected = conditional_factors(quotes.AdjClose.to_numpy(float),quotes.DailyAmplitude.to_numpy(float),
                                      quotes.DailyReturn.to_numpy(float),grid.AvgTrade.to_numpy(float),
                                      ideal_valid_days(quotes).to_numpy(bool))
        rolling["IdealAmp"],rolling["IdealRev"],rolling["LowAmpMom"] = selected[:,0],selected[:,1],selected[:,2]
        rolling["SecuCode"] = symbol
        rolling.index.name = "TradingDay"
        rolled.append(rolling.reset_index())
    result = pd.concat(rolled,ignore_index=True)
    result.to_parquet(Path(cfg.output_dir)/"rolling_components.parquet",index=False)
    return result


def zscore(y):
    deviation = y.std(ddof=0)
    return (y-y.mean())/deviation if deviation > 1e-12 else y*0


def winsor(y):
    med = y.median()
    mad = (y-med).abs().median()
    return y.clip(med-5*mad,med+5*mad) if mad > 1e-12 else y


def residual(y, controls):
    output = pd.Series(np.nan,index=y.index)
    valid = y.notna() & controls.notna().all(axis=1)
    if valid.sum() >= max(20, controls.shape[1]+3):
        x = np.column_stack([np.ones(valid.sum()),controls.loc[valid].to_numpy(float)])
        v = y.loc[valid].to_numpy(float)
        output.loc[valid] = v-x@np.linalg.lstsq(x,v,rcond=None)[0]
    return output


def preprocess(rolling, eligible, daily, cap):
    full = rolling.merge(eligible,on=base.KEYS,validate="one_to_one")
    full = full.merge(cap,on=base.KEYS,how="left",validate="one_to_one")
    full = full.merge(daily[base.KEYS+base.CONTROLS],on=base.KEYS,how="left",validate="one_to_one")
    outputs = []
    for _, day in tqdm(full.groupby("TradingDay"),desc="研报分阶段合成"):
        out = day[base.KEYS].copy()
        # 东吴2020：均值和标准差分别市值中性化，再分别剔除反转、Z并相加；
        # 趋势单独剔市值/反转/换手/波动，再与前一合成项分别Z后相加。
        avg = residual(winsor(day.PV),day[["LnCap"]])
        std = residual(winsor(day.PVStd),day[["LnCap"]])
        avg_ret = residual(avg,day[["Ctrl_Ret20"]])
        std_ret = residual(std,day[["Ctrl_Ret20"]])
        deret = zscore(avg_ret)+zscore(std_ret)
        trend = residual(winsor(day.PVTrend),day[["LnCap"]+base.CONTROLS])
        out["HF_CPV"] = zscore(zscore(deret)+zscore(trend))
        # 东吴2021：四个子项先各自处理及Z；prime保留同向两项。
        pieces = {key:zscore(residual(winsor(day[key]),day[["LnCap"]])) for key in ["PP","PM","MP","MM","DP_Pos_P","DP_Neg_P"]}
        out["HF_CDPDP"] = zscore(pieces["PP"]+pieces["MM"])
        out["HF_CDPDP4"] = zscore(pieces["PP"]-pieces["PM"]-pieces["MP"]+pieces["MM"])
        out["HF_CDPP"] = zscore(pieces["DP_Pos_P"]-pieces["DP_Neg_P"])
        # 开源2021仅声明截面去极值/标准化，没有把所有因子统一行业正交化。
        for target,source in [("HF_SmartMoney","SmartPooled"),("HF_IdealAmplitude","IdealAmp"),
                              ("HF_IdealReversal","IdealRev"),("HF_LowAmpMomentum","LowAmpMom"),
                              ("HF_Skewness","Skew")]:
            out[target] = zscore(winsor(day[source]))
        outputs.append(out)
    result = pd.concat(outputs,ignore_index=True).sort_values(base.KEYS)
    return result


def month_schedule(calendar):
    frame = pd.Series(pd.DatetimeIndex(calendar),index=pd.DatetimeIndex(calendar))
    return pd.DatetimeIndex(frame.groupby(frame.index.to_period("M")).max())


def scheduled_groups(features,factor,schedule,low=True,n_groups=10):
    pieces = []
    for date in schedule:
        sub = features.loc[features.TradingDay.eq(date),base.KEYS+[factor]].dropna().copy()
        if len(sub)<max(20,n_groups) or sub[factor].nunique()<n_groups:
            continue
        sub = sub.sort_values([factor,"SecuCode"],ascending=[not low,True])
        sub["Group"] = np.arange(len(sub))*n_groups//len(sub)+1
        pieces.append(sub)
    return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=base.KEYS+[factor,"Group"])


def calendar_ic(groups,market,factor,schedule,low=True):
    rows=[]
    for start,end in zip(schedule[:-1],schedule[1:]):
        sub=groups[groups.TradingDay.eq(start)]
        entry,exit_=market.dates.get_loc(start)+1,market.dates.get_loc(end)+1
        if sub.empty or exit_>=len(market.dates):
            continue
        columns=market.symbols.get_indexer(sub.SecuCode)
        buy,sell=market.open[entry,columns],market.open[exit_,columns]
        valid=np.isfinite(buy)&np.isfinite(sell)&(buy>0)&(sell>0)
        x=pd.Series(sub[factor].to_numpy()[valid]);r=pd.Series(sell[valid]/buy[valid]-1)
        ic=x.corr(r) if len(x)>=10 and x.std()>1e-12 and r.std()>1e-12 else np.nan
        rank=x.corr(r,method="spearman") if np.isfinite(ic) else np.nan
        rows.append({"SignalDate":start,"EntryDate":market.dates[entry],"ExitDate":market.dates[exit_],
                     "HoldingDays":exit_-entry,"N_Signal":len(sub),"N_Paired":int(valid.sum()),"Coverage":valid.mean(),
                     "IC_Raw":ic,"RankIC_Raw":rank,"IC_Directed":(-1 if low else 1)*ic,"RankIC_Directed":(-1 if low else 1)*rank})
    return pd.DataFrame(rows)


def export_case(features,factor,name,market,cfg,folder,low=True,with_gross=False):
    if name=="MonthEnd":
        schedule=month_schedule(market.dates)
        # 避免把首个不完整月的最后一天提前至样本起点；正常从月底形成信号。
        groups=scheduled_groups(features,factor,schedule,low,cfg.n_groups)
        ic=calendar_ic(groups,market,factor,schedule,low)
    else:
        days=base.PERIODS[name]
        groups,schedule=base.signal_groups(features,factor,market.dates,days,cfg.min_cross_section,low,cfg.n_groups)
        ic=base.calc_ic(groups,market,factor,days,low)
    ledger,trades=base.backtest(groups,schedule,market,cfg)
    folder.mkdir(parents=True,exist_ok=True)
    groups.to_parquet(folder/"signal_groups.parquet",index=False)
    ledger.to_csv(folder/"daily_ledger.csv",index=False)
    trades.to_parquet(folder/"trades.parquet",index=False)
    if len(trades):
        trades["TradingCode"] = trading.trading_code(trades.TradingDay,trades.SecuCode)
        trades.to_parquet(folder/"trades.parquet",index=False)
    ic.to_csv(folder/"ic.csv",index=False)
    base.plot_result(ledger,folder/"nav.png",f"{factor} / {name} / paper definitions",cfg,schedule[0])
    records=[]
    for portfolio,frame in ledger.groupby("Portfolio"):
        record=base.performance(frame.Return,cfg)
        record.update({"Factor":factor,"Period":name,"Portfolio":portfolio,"Cost":"Net",
                       "LSLegWeight":cfg.ls_leg_weight,"IC_Raw":ic.IC_Raw.mean(),"RankIC_Raw":ic.RankIC_Raw.mean(),
                       "IC_Coverage":ic.Coverage.mean(),"MeanSignalStocks":ic.N_Signal.mean(),
                       "MeanDailyTurnover":frame.Turnover.mean(),"TotalTradingCost":frame.TradingCost.sum(),
                       "StalePositionDays":frame.StalePositions.sum(),"FinalPositions":frame.Positions.iloc[-1],
                       "BlockedOrders":frame.BlockedOrders.sum(),"AdmissionBlockedOrders":frame.AdmissionBlockedOrders.sum()})
        records.append(record)
    if with_gross:
        gross,_=base.backtest(groups,schedule,market,replace(cfg,one_way_fee=0))
        gross.to_csv(folder/"daily_ledger_gross.csv",index=False)
        for portfolio,frame in gross.groupby("Portfolio"):
            record=base.performance(frame.Return,cfg)
            record.update({"Factor":factor,"Period":name,"Portfolio":portfolio,"Cost":"Gross","LSLegWeight":cfg.ls_leg_weight})
            records.append(record)
    return records


def icir_scores(features,market,days,output,predict_start):
    """开源2021脚注3的12期最大ICIR方法移植，当前六个可实现因子。

    只用已实现的非重叠历史IC；协方差使用固定10%对角收缩以处理12样本/6维
    的不稳定性。原报告未给出数值求逆细节，此处是公开的数值约定。
    """
    features=features.dropna(subset=list(PRIMARY)).sort_values(base.KEYS)
    dates=market.dates
    first=dates.searchsorted(pd.Timestamp(predict_start))
    anchors=np.arange(first % days,len(dates)-days-1,days)
    feature_map={date:frame for date,frame in features.groupby("TradingDay")}
    historic=[]
    for t in anchors:
        if dates[t] not in feature_map:
            continue
        frame=feature_map[dates[t]]
        columns=market.symbols.get_indexer(frame.SecuCode)
        buy,sell=market.open[t+1,columns],market.open[t+days+1,columns]
        valid=np.isfinite(buy)&np.isfinite(sell)&(buy>0)&(sell>0)
        returns=pd.Series(sell[valid]/buy[valid]-1)
        values=[]
        for factor in PRIMARY:
            x=pd.Series(-frame[factor].to_numpy()[valid])
            values.append(x.corr(returns) if len(x)>=10 and x.std()>1e-12 and returns.std()>1e-12 else np.nan)
        if np.isfinite(values).all():
            historic.append((dates[t],dates[t+days+1],values))
    predictions=[];weights=[]
    for date,frame in feature_map.items():
        if date < pd.Timestamp(predict_start):
            continue
        past=[item for item in historic if item[1]<date][-12:]
        pred=frame[base.KEYS].copy();pred["ICIR_12"]=np.nan
        if len(past)==12:
            sample=np.array([item[2] for item in past]);mean=sample.mean(axis=0)
            covariance=np.cov(sample,rowvar=False)
            scale=max(np.trace(covariance)/len(PRIMARY),1e-8)
            covariance=.9*covariance+.1*scale*np.eye(len(PRIMARY))
            w=np.linalg.solve(covariance,mean)
            if np.sum(np.abs(w))>1e-12:
                w/=np.sum(np.abs(w))
                pred["ICIR_12"]=-frame[list(PRIMARY)].to_numpy()@w
                weights.append({"TradingDay":date,"MaxLabelEnd":past[-1][1],"ICPeriods":12,
                                **{factor:float(value) for factor,value in zip(PRIMARY,w)}})
        predictions.append(pred)
    result=pd.concat(predictions,ignore_index=True)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result.to_parquet(output/"predictions.parquet",index=False)
    pd.DataFrame(weights).to_csv(output/"weights.csv",index=False)
    return result


def run_research(cfg,universe_name='500',with_xgboost=True,predict_start='2017-02-01',model_only=False):
    from code2_pipeline import run
    if universe_name != '500':
        raise ValueError('当前整理版以动态中证500为主样本。')
    return run(cfg,with_models=with_xgboost,models_only=model_only,predict_start=predict_start)
