"""同一信号样本下分离成交约束、费用与多空敞口。"""
from pathlib import Path
from dataclasses import replace
import sys
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import code2_new as base
import code2_research as r


def run():
    output=ROOT/'results_code2_constraints'
    daily=pd.read_parquet(output/'daily_market.parquet')
    calendar=pd.DatetimeIndex(sorted(daily.TradingDay.unique()))
    cfg=base.Config(output_dir=output,ls_leg_weight=1.)
    records=[]
    cases=[(factor,'MonthEnd',output/factor/'MonthEnd') for factor in
           ['HF_SmartMoney','HF_CPV','HF_CDPDP','HF_IdealAmplitude']]
    cases += [('Pred_Score',name,output/'comparison/Pred_Score'/name) for name in base.PERIODS]
    for factor,name,folder in cases:
        start='2017-02-01' if factor=='Pred_Score' else '2016-01-29'
        dates=calendar[calendar>=pd.Timestamp(start)]
        actual=base.Market.from_daily(daily,dates)
        # 保留真实开盘停牌：没有09:30价格就不准用稍后复牌首笔价。仅去掉方向价格约束与ST增仓限制。
        quoted=np.isfinite(actual.open)&(actual.open>0)
        benchmark=base.Market(dates,actual.symbols,actual.open,actual.close,quoted,quoted,actual.stale,np.ones_like(quoted))
        schedule=r.month_schedule(dates) if name=='MonthEnd' else dates[::base.PERIODS[name]]
        groups=pd.read_parquet(folder/'signal_groups.parquet')
        for leg in [.5,1.]:
            for mode,market in [('Actual',actual),('QuotedOpenBenchmark',benchmark)]:
                for fee in [0.,.0015]:
                    c=replace(cfg,ls_leg_weight=leg,one_way_fee=fee)
                    ledger,_=base.backtest(groups,schedule,market,c)
                    frame=ledger[ledger.Portfolio.eq('TheoreticalLS')]
                    metrics=base.performance(frame.Return,c)
                    metrics.update({'Factor':factor,'Period':name,'LegWeight':leg,'Execution':mode,'OneWayFee':fee,
                                    'TotalTradingCost':frame.TradingCost.sum(),'BlockedOrders':frame.BlockedOrders.sum(),
                                    'AdmissionBlockedOrders':frame.AdmissionBlockedOrders.sum()})
                    records.append(metrics)
    result=pd.DataFrame(records)
    result.to_csv(output/'sensitivity.csv',index=False)
    print(result[result.Execution.eq('Actual')&result.OneWayFee.eq(.0015)][
        ['Factor','Period','LegWeight','AnnualReturn','Sharpe','MaxDrawdown']].to_string(index=False))


if __name__=='__main__':run()
