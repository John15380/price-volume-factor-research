"""在因子不变时，分离名义敞口、费用和成交代理的影响。"""
from pathlib import Path
from dataclasses import replace
import sys
import logging
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import code2_new as base
import code2_research as r


def run():
    cfg=base.Config(output_dir=ROOT/'results_code2_research',ls_leg_weight=1.)
    calendar,symbols,daily,*_=r.load_data(cfg,'500')
    dates=calendar[calendar>=pd.Timestamp('2016-01-29')]
    market=base.Market.from_daily(daily,dates)
    schedule=r.month_schedule(dates)
    raw=[]
    for year in [2015,2016,2017,2018]:
        path=cfg.data_dir/f'日行情/daily_quote_{year}.csv'
        f=pd.read_csv(path,usecols=['TradingDay','SecuCode','OpenPrice','PrevClosePrice'],dtype={'SecuCode':str})
        f=f[f.SecuCode.isin(symbols)].copy();f.TradingDay=pd.to_datetime(f.TradingDay)
        raw.append(f)
    quotes=pd.concat(raw,ignore_index=True)
    gap=(quotes.OpenPrice/quotes.PrevClosePrice-1)
    quotes['Gap']=gap
    gap=quotes.pivot(index='TradingDay',columns='SecuCode',values='Gap').reindex(index=dates,columns=market.symbols).to_numpy()
    stress=base.Market(dates,market.symbols,market.open,market.close,market.buy & (gap<.049),market.sell & (gap>-.049),market.stale)
    records=[]
    for factor in ['HF_SmartMoney','HF_CPV','HF_CDPDP','HF_IdealAmplitude']:
        groups=pd.read_parquet(cfg.output_dir/factor/'MonthEnd/signal_groups.parquet')
        for leg in [.5,1.]:
            for mode,execution in [('OpenQuote',market),('Gap4.9Stress',stress)]:
                for fee in [0.,.0015]:
                    c=replace(cfg,ls_leg_weight=leg,one_way_fee=fee)
                    ledger,_=base.backtest(groups,schedule,execution,c)
                    ls=ledger[ledger.Portfolio.eq('TheoreticalLS')]
                    record=base.performance(ls.Return,c)
                    record.update({'Factor':factor,'Period':'MonthEnd','LegWeight':leg,'Execution':mode,'OneWayFee':fee,
                                   'TotalTradingCost':ls.TradingCost.sum(),'MeanDailyTurnover':ls.Turnover.mean()})
                    records.append(record)
    test_dates=calendar[calendar>=pd.Timestamp('2017-02-01')]
    test_market=base.Market.from_daily(daily,test_dates)
    gap=quotes.pivot(index='TradingDay',columns='SecuCode',values='Gap').reindex(index=test_dates,columns=test_market.symbols).to_numpy()
    stress=base.Market(test_dates,test_market.symbols,test_market.open,test_market.close,
                       test_market.buy&(gap<.049),test_market.sell&(gap>-.049),test_market.stale)
    for name,days in base.PERIODS.items():
        groups=pd.read_parquet(cfg.output_dir/'comparison/Pred_Score'/name/'signal_groups.parquet')
        schedule=test_dates[::days]
        for leg in [.5,1.]:
            for mode,execution in [('OpenQuote',test_market),('Gap4.9Stress',stress)]:
                for fee in [0.,.0015]:
                    c=replace(cfg,ls_leg_weight=leg,one_way_fee=fee)
                    ledger,_=base.backtest(groups,schedule,execution,c)
                    ls=ledger[ledger.Portfolio.eq('TheoreticalLS')]
                    record=base.performance(ls.Return,c)
                    record.update({'Factor':'Pred_Score','Period':name,'LegWeight':leg,'Execution':mode,'OneWayFee':fee,
                                   'TotalTradingCost':ls.TradingCost.sum(),'MeanDailyTurnover':ls.Turnover.mean()})
                    records.append(record)
    out=pd.DataFrame(records)
    out.to_csv(cfg.output_dir/'sensitivity.csv',index=False)
    print(out[['Factor','LegWeight','Execution','OneWayFee','AnnualReturn','Sharpe','MaxDrawdown']].to_string(index=False))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    run()
