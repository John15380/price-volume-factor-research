"""额外检查：全A可用样本的东吴价格相关因子，不用分钟bar冒充资金流。

使用批量Numba聚合避免逐股票DataFrame开销。主研究的聪明钱等因子仍由
code2_research完整计算；此脚本仅比较股票池改变对CPV/CDPP/CDPDP的影响。
"""
from pathlib import Path
import sys
import json
import logging
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from numba import njit
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import code2_new as base
import code2_research as r
import code2_trading_data as trading


@njit(cache=True)
def batch_stats(close,volume,starts,ends):
    output=np.full((len(starts),7),np.nan)
    for i in range(len(starts)):
        p=close[starts[i]:ends[i]]
        v=volume[starts[i]:ends[i]]
        valid=np.isfinite(p)&np.isfinite(v)&(p>0)&(v>=0)
        p,v=p[valid],v[valid]
        if len(p)<31 or v.sum()<=0:
            continue
        delta=p[1:]-p[:-1]
        x,y=delta[:-1],delta[1:]
        pp,pm=(x>0)&(y>0),(x>0)&(y<0)
        mp,mm=(x<0)&(y>0),(x<0)&(y<0)
        positive,negative=x>0,x<0
        output[i]=np.array([r.correlation(p,v),r.correlation(x[pp],y[pp]),r.correlation(x[pm],y[pm]),
                            r.correlation(x[mp],y[mp]),r.correlation(x[mm],y[mm]),
                            r.correlation(x[positive],p[2:][positive]),r.correlation(x[negative],p[2:][negative])])
    return output


def run():
    cfg=base.Config(output_dir=ROOT/'results_research_all_a_constraints',ls_leg_weight=1.)
    cfg.output_dir.mkdir(parents=True,exist_ok=True)
    calendar,symbols,daily,eligible,industry,cap=r.load_data(cfg,'all')
    LOG=logging.getLogger('all_a_check')
    LOG.info('全A可用样本%d只；含历史ST、停牌、实际涨跌停及官方复权',len(symbols))
    parts=[]
    cols=['PV','PP','PM','MP','MM','DP_Pos_P','DP_Neg_P']
    cache=cfg.output_dir/'daily_price_metrics.parquet'
    if cache.exists():
        raw=pd.read_parquet(cache)
    else:
        files=[p for p in sorted((cfg.data_dir/'bar_1m_2016_2018').glob('*.parquet')) if pd.Timestamp(p.stem) in calendar]
        for path in tqdm(files,desc='全A价格相关性'):
            bars=ds.dataset(path,format='parquet').to_table(columns=['symbol','time','close','volume'],filter=ds.field('symbol').isin(trading.raw_symbols(symbols))).to_pandas()
            bars.symbol=trading.canonical(bars.symbol)
            bars=bars.sort_values(['symbol','time'])
            codes=bars.symbol.to_numpy()
            starts=np.r_[0,np.flatnonzero(codes[1:]!=codes[:-1])+1]
            ends=np.r_[starts[1:],len(codes)]
            values=batch_stats(bars.close.to_numpy(float),bars.volume.to_numpy(float),starts,ends)
            frame=pd.DataFrame(values,columns=cols)
            frame['SecuCode']=codes[starts]
            frame['TradingDay']=pd.Timestamp(path.stem)
            parts.append(frame)
        raw=pd.concat(parts,ignore_index=True)
        raw.to_parquet(cache,index=False)
    parts=[]
    weights=np.arange(20.)-9.5
    for symbol,frame in tqdm(raw.groupby('SecuCode'),desc='全A窗口'):
        grid=frame.set_index('TradingDay')[cols].reindex(calendar)
        roll=grid.rolling(20,min_periods=20).mean()
        roll['PVStd']=grid.PV.rolling(20,min_periods=20).std()
        roll['PVTrend']=grid.PV.rolling(20,min_periods=20).apply(lambda x:np.dot(x,weights)/np.dot(weights,weights),raw=True)
        for name in r.DAY_FIELDS+('SmartPooled','IdealAmp','IdealRev','LowAmpMom'):
            if name not in roll:roll[name]=np.nan
        roll['SecuCode']=symbol;roll.index.name='TradingDay'
        parts.append(roll.reset_index())
    rolling=pd.concat(parts,ignore_index=True)
    features=r.preprocess(rolling,eligible,daily,cap)
    features.to_parquet(cfg.output_dir/'processed_features.parquet',index=False)
    dates=calendar[calendar>=pd.Timestamp('2016-01-29')]
    market=base.Market.from_daily(daily,dates)
    records=[]
    for factor in ['HF_CPV','HF_CDPP','HF_CDPDP','HF_CDPDP4','Legacy_CPVMean','Legacy_CDPDPRawSum']:
        LOG.info('全A自然月回测 %s',factor)
        records+=r.export_case(features,factor,'MonthEnd',market,cfg,cfg.output_dir/factor/'MonthEnd',True,True)
    pd.DataFrame(records).to_csv(cfg.output_dir/'summary.csv',index=False)
    (cfg.output_dir/'limitations.json').write_text(json.dumps({'universe':'available mainland A shares',
      'candidates':len(symbols),'historical_ST_flag_available':True,'listing_age':'first positive quote + trading calendar, >=60 days',
      'execution':'actual limit bounds and 09:30 halt state; ST admission restrictions','ls_leg_weight':1.0,'one_way_fee':.0015,
      'purpose':'stock-universe sensitivity for price-correlation families; not exact broker-data reproduction'},ensure_ascii=False,indent=2))
    LOG.info('全A对照完成')


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    run()
