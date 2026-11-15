"""按原始研报复核定义，分离有效日规则、股票池和费用影响。"""
from pathlib import Path
import hashlib
import json
import logging
import sys
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import code2_new as base
import code2_research as r

OUT = ROOT / 'results_code2_definition_audit'
VARIANTS = ('IA_Strict20', 'IA_ValidWithoutLimitExclusion', 'HF_IdealAmplitude')


@njit(cache=True)
def amplitude_variants(price, amp, no_halt, effective):
    result = np.full((len(price), 3), np.nan)
    for t in range(19, len(price)):
        p, a = price[t-19:t+1], amp[t-19:t+1]
        if np.isfinite(p).all() and np.isfinite(a).all():
            order = np.argsort(p)
            result[t, 0] = a[order[-5:]].mean()-a[order[:5]].mean()
        for column in (1, 2):
            valid = (np.isfinite(p) & np.isfinite(a) & (p > 0) & (a >= 0)
                     & (no_halt[t-19:t+1] if column == 1 else effective[t-19:t+1]))
            if valid.sum() >= 10:
                x, y = p[valid], a[valid]
                order = np.argsort(x)
                k = len(x)//4
                result[t, column] = y[order[-k:]].mean()-y[order[:k]].mean()
    return result


def numeric_reference_checks(daily):
    """独立numpy逐项算术核对，不将测试期收益用于选公式。"""
    dates = pd.DatetimeIndex(sorted(daily.TradingDay.unique()))
    dates = dates[(dates >= '2017-03-20') & (dates <= '2017-03-31')]
    symbols = ['000001', '000100', '600000']
    # 从可用中证500证券中固定选3只，避免检查完全未进当前日行情的股票。
    symbols = sorted(daily.loc[daily.TradingDay.eq(dates[-1]) & daily.SignalEligible, 'SecuCode'].unique())[:3]
    scale = daily.set_index(base.KEYS).Scale
    pooled = {s: [] for s in symbols}
    daily_errors, pooled_errors, correlation_errors = [], [], []
    for day in dates:
        path = ROOT / 'data/bar_1m_2016_2018' / f'{day.date()}.parquet'
        bars = ds.dataset(path).to_table(filter=ds.field('symbol').isin(symbols)).to_pandas()
        for symbol, g in bars.sort_values(['symbol', 'time']).groupby('symbol'):
            if len(g) < 31:
                continue
            p = g.close.to_numpy(float)
            v = g.volume.to_numpy(float)/scale.loc[(day, symbol)]
            a = g.amount.to_numpy(float)
            prev = np.r_[g.open.iloc[0], p[:-1]]
            scores = np.full(len(p), -np.inf)
            scores[v > 0] = np.abs(p[v > 0]/prev[v > 0]-1)/v[v > 0]**.25
            values, cs, cv, ca = r.day_metrics(*[g[c].to_numpy(float) for c in
                ['open', 'close', 'high', 'low', 'volume', 'amount', 'trade_num']], scale.loc[(day, symbol)])
            np.testing.assert_allclose(scores, cs, atol=1e-14)
            delta = np.diff(p); x, y = delta[:-1], delta[1:]
            masks = [(x > 0)&(y > 0), (x > 0)&(y < 0), (x < 0)&(y > 0), (x < 0)&(y < 0)]
            expected = []
            for mask in masks:
                xx, yy = x[mask], y[mask]
                expected.append(np.corrcoef(xx, yy)[0, 1] if len(xx) >= 3 and np.std(xx)*np.std(yy) > 1e-14 else 0.)
            correlation_errors.extend(np.abs(np.asarray(expected)-values[1:5]).tolist())
            order = np.argsort(scores)[::-1]
            take = np.clip(v.sum()*.2-(np.cumsum(v[order])-v[order]), 0, v[order])
            ref = np.sum(np.divide(a[order]*take, v[order], out=np.zeros(len(v)), where=v[order] > 0))/(v.sum()*.2)/(a.sum()/v.sum())
            daily_errors.append(abs(ref-values[10]))
            pooled[symbol].append((scores, v, a))
    for symbol, parts in pooled.items():
        if len(parts) != 10:
            continue
        scores, v, a = [np.concatenate([part[k] for part in parts]) for k in range(3)]
        order = np.argsort(scores)[::-1]
        take = np.clip(v.sum()*.2-(np.cumsum(v[order])-v[order]), 0, v[order])
        reference = np.sum(np.divide(a[order]*take, v[order], out=np.zeros(len(v)), where=v[order] > 0))/(v.sum()*.2)/(a.sum()/v.sum())
        pooled_errors.append(abs(reference-r.smart_ratio(scores, v, a)))
    assert len(pooled_errors) == 3
    assert max(daily_errors+pooled_errors+correlation_errors) < 1e-10
    return {'stock_days_checked': len(daily_errors), 'pooled_10_day_windows': len(pooled_errors),
            'max_smart_daily_error': max(daily_errors), 'max_smart_pooled_error': max(pooled_errors),
            'max_quadrant_correlation_error': max(correlation_errors), 'symbols': symbols,
            'scope': 'arithmetic check of stated conventions, not proof that unspecified vendor conventions match broker data'}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    stats, records = [], []
    for universe, source in [('CSI500', 'results_code2_constraints'), ('AllA', 'results_research_all_a_constraints')]:
        logging.info('%s: amplitude definition recheck', universe)
        daily = pd.read_parquet(ROOT/source/'daily_market.parquet')
        if universe == 'CSI500':
            reference = numeric_reference_checks(daily)
            (OUT/'numeric_reference_checks.json').write_text(json.dumps(reference, ensure_ascii=False, indent=2))
        all_dates = pd.DatetimeIndex(sorted(daily.TradingDay.unique()))
        eligible = daily.loc[daily.SignalEligible, base.KEYS]
        if universe == 'CSI500':
            available = pd.read_parquet(ROOT/source/'processed_features.parquet', columns=base.KEYS)
            eligible = eligible.merge(available, on=base.KEYS, validate='one_to_one')
        parts = []
        counts = {'Universe': universe, 'EligibleStockDays': 0, 'Strict20': 0, 'PaperEffective': 0,
                  'EligibleDaysWithHistoricalOnePriceLimit': 0, 'EffectiveDaysLessThan10': 0}
        for symbol, g in daily.groupby('SecuCode', sort=False):
            g = g.sort_values('TradingDay').set_index('TradingDay').reindex(all_dates)
            valid = r.ideal_valid_days(g).to_numpy(bool)
            no_halt = (~g.SuspendAny & g.PTFlag.eq(0) & g.Volume.gt(0)).to_numpy(bool)
            values = amplitude_variants(g.AdjClose.to_numpy(float), g.DailyAmplitude.to_numpy(float), no_halt, valid)
            # 交叉核对生产实现，研究版本仅改变有效日规则，禁止以收益选择版本。
            check = r.conditional_factors(g.AdjClose.to_numpy(float), g.DailyAmplitude.to_numpy(float),
                                         g.DailyReturn.to_numpy(float), np.full(len(g), np.nan), valid)
            np.testing.assert_allclose(values[:, 2], check[:, 0], equal_nan=True, atol=1e-14)
            sub = pd.DataFrame(values, columns=VARIANTS, index=all_dates)
            sub['ValidDays20'] = pd.Series(valid, index=all_dates).rolling(20, min_periods=20).sum()
            sub['ExcludedLimitDays20'] = pd.Series(no_halt & ~valid & g.DailyAmplitude.notna().to_numpy(), index=all_dates).rolling(20, min_periods=20).sum()
            sub['SecuCode'] = symbol
            sub.index.name = 'TradingDay'
            parts.append(sub.reset_index())
        raw = pd.concat(parts, ignore_index=True).merge(eligible, on=base.KEYS, validate='one_to_one')
        for factor in VARIANTS:
            raw[factor] = raw.groupby('TradingDay')[factor].transform(lambda x: r.zscore(r.winsor(x)))
        raw.to_parquet(OUT/f'{universe}_features.parquet', index=False)
        sample = raw[raw.TradingDay >= '2016-01-29']
        counts.update(EligibleStockDays=len(sample), Strict20=int(sample[VARIANTS[0]].notna().sum()),
                      PaperEffective=int(sample[VARIANTS[2]].notna().sum()),
                      EligibleDaysWithHistoricalOnePriceLimit=int(sample.ExcludedLimitDays20.gt(0).sum()),
                      EffectiveDaysLessThan10=int(sample.ValidDays20.lt(10).sum()))
        stats.append(counts)
        dates = all_dates[all_dates >= '2016-01-29']
        market = base.Market.from_daily(daily, dates)
        cfg = base.Config(output_dir=OUT)
        for factor in VARIANTS:
            for period in list(base.PERIODS)+['MonthEnd']:
                logging.info('%s %s %s', universe, factor, period)
                case = r.export_case(raw, factor, period, market, cfg, OUT/universe/factor/period, True, True)
                records.extend([{**record, 'Universe': universe, 'Sample': 'Available'} for record in case])
        common = raw.dropna(subset=[VARIANTS[0], VARIANTS[2]])
        for factor in [VARIANTS[0], VARIANTS[2]]:
            case = r.export_case(common, factor, 'MonthEnd', market, cfg, OUT/universe/'paired'/factor, True, True)
            records.extend([{**record, 'Universe': universe, 'Sample': 'Paired'} for record in case])
    pd.DataFrame(records).to_csv(OUT/'summary.csv', index=False)
    pd.DataFrame(stats).to_csv(OUT/'coverage.csv', index=False)
    config = {'source_primary_20200516': 'audit/source_reports/20200516.pdf pp.5-7',
              'source_recap_20210924': '价量多因子模型/20210924*.pdf p.8',
              'rules_chosen_before_backtest': True,
              'rounding_not_specified_in_report': 'floor(n_valid*0.25); daily price ties follow deterministic argsort',
              'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [Path(__file__), Path(r.__file__), Path(base.__file__)]}}
    (OUT/'audit_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(pd.DataFrame(records).query("Portfolio == 'TheoreticalLS' and Cost == 'Net'")[
        ['Universe', 'Sample', 'Factor', 'Period', 'AnnualReturn', 'Sharpe', 'MeanSignalStocks']].to_string(index=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run()
