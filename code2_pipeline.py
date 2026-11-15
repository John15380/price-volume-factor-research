"""当前唯一实验入口：9候选因子、10分组、训练期筛选与调参。"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import hashlib
import inspect
import json
import logging
import numpy as np
import pandas as pd

import code2_new as base
import code2_research as research
import code2_trading_data as trading
import code2_tuning as tuning

LOG=logging.getLogger('pipeline')
CANDIDATES=tuple(research.FACTORS)


def factor_signature():
    functions=[research.day_metrics.py_func,research.correlation.py_func,research.smart_ratio.py_func,
               research.conditional_factors.py_func,research.ideal_valid_days,research.preprocess,
               research.winsor,research.zscore,research.residual]
    return hashlib.sha256((''.join(inspect.getsource(f) for f in functions)+str(CANDIDATES)).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def write_input_manifest(cfg, origin):
    folder=Path(cfg.output_dir)/'input'
    meta={'origin':origin,'factors':CANDIDATES,'factor_signature':factor_signature(),
          'trading_input_signature':trading.input_manifest(cfg.data_dir)['signature'],
          'files':{name:file_hash(folder/name) for name in ['features.parquet','daily_market.parquet']}}
    (folder/'input_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))


def prepare_inputs(cfg):
    folder=Path(cfg.output_dir)/'input';folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'input_manifest.json'
    if manifest.exists():
        meta=json.loads(manifest.read_text())
        if meta['factor_signature']!=factor_signature():raise ValueError('因子定义已经变化，需重新生成输入缓存。')
        if meta['trading_input_signature']!=trading.input_manifest(cfg.data_dir)['signature']:
            raise ValueError('交易数据已变化，需重新生成输入缓存。')
        for name,digest in meta['files'].items():
            if file_hash(folder/name)!=digest:raise ValueError(f'{name}缓存校验失败。')
        LOG.info('已验证并复用9因子及实际交易状态输入；不复用模型预测。')
    else:
        inner=replace(cfg,output_dir=folder)
        calendar,symbols,daily,eligible,industry,cap=research.load_data(inner,'500')
        rolling=research.calculate_factors(inner,calendar,symbols,daily)
        features=research.preprocess(rolling,eligible,daily,cap)[base.KEYS+list(CANDIDATES)]
        features.to_parquet(folder/'features.parquet',index=False)
        write_input_manifest(cfg,'fresh raw-minute calculation with current factor definitions')
    features=pd.read_parquet(folder/'features.parquet')
    daily=pd.read_parquet(folder/'daily_market.parquet')
    assert list(features.columns)==base.KEYS+list(CANDIDATES)
    assert not any('Tail' in c or 'Legacy' in c for c in features.columns)
    return features,daily


def export(features,factor,period,market,cfg,destination,low):
    rows=research.export_case(features,factor,period,market,cfg,destination,low,False)
    for row in rows:
        row['Groups']=cfg.n_groups
        row['Role']='Long' if row['Portfolio']==f'Group{cfg.n_groups}' else ('LongShort' if row['Portfolio']=='TheoreticalLS' else 'Decile')
    return rows


def run(cfg,with_models=True,models_only=False,prepare_only=False,predict_start='2017-02-01',trials=96,finalists=16):
    output=Path(cfg.output_dir);output.mkdir(parents=True,exist_ok=True)
    if cfg.n_groups!=10:raise ValueError('当前提交版固定为10分组。')
    if not np.isclose(cfg.one_way_fee,.0015):raise ValueError('当前提交版固定每次买卖各0.0015，完整往返0.003。')
    features,daily=prepare_inputs(cfg)
    if prepare_only:return
    calendar=pd.DatetimeIndex(sorted(daily.TradingDay.unique()))
    full_dates=calendar[calendar>=pd.Timestamp('2016-01-29')]
    full_market=base.Market.from_daily(daily,full_dates)
    feature_coverage=features.groupby('TradingDay')[list(CANDIDATES)].count()
    feature_coverage.to_csv(output/'feature_coverage.csv')
    pool=None;jobs={}
    if with_models:
        full_history=base.Market.from_daily(daily,calendar)
        pool=ThreadPoolExecutor(max_workers=3)
        # pandas的Index哈希引擎会惰性构建，不在线程之间共享可变缓存。
        for name,days in base.PERIODS.items():
            isolated=replace(full_history,
                dates=pd.DatetimeIndex(full_history.dates.to_numpy(copy=True)),
                symbols=pd.Index(full_history.symbols.to_numpy(copy=True)))
            isolated_features=features.copy(deep=True)
            jobs[name]=pool.submit(tuning.walk_forward,isolated_features,isolated,days,output/'models'/name,
                                   predict_start,12,list(CANDIDATES),trials,finalists)
    records=[]
    if not models_only:
        for factor in CANDIDATES:
            for period in list(base.PERIODS)+['MonthEnd']:
                LOG.info('10分组单因子 %s / %s',factor,period)
                records+=export(features,factor,period,full_market,cfg,output/'factors'/factor/period,
                                research.LOW_IS_LONG[factor])
        pd.DataFrame(records).to_csv(output/'factor_summary.csv',index=False)
    if with_models:
        comparison=[];test_dates=calendar[calendar>=pd.Timestamp(predict_start)]
        market=base.Market.from_daily(daily,test_dates)
        try:
            for name,future in jobs.items():
                pred=future.result()
                keys=pred.loc[pred.Pred_Score.notna(),base.KEYS]
                common=features.merge(keys,on=base.KEYS,validate='one_to_one')
                for factor in CANDIDATES:
                    comparison+=export(common,factor,name,market,cfg,output/'oos'/factor/name,
                                       research.LOW_IS_LONG[factor])
                comparison+=export(pred,'Pred_Score',name,market,cfg,output/'oos/Pred_Score'/name,False)
        finally:pool.shutdown(wait=True,cancel_futures=True)
        pd.DataFrame(comparison).to_csv(output/'oos_summary.csv',index=False)
    source_names=['code2_new.py','code2_research.py','code2_trading_data.py','code2_xgboost.py','code2_tuning.py','code2_pipeline.py']
    manifest={'universe':'point-in-time CSI500','candidate_factors':CANDIDATES,'groups':10,
              'one_way_fee':cfg.one_way_fee,'round_trip_fee':2*cfg.one_way_fee,'ls_leg_weight':cfg.ls_leg_weight,
              'signal_start':'2016-01-29','predict_start':predict_start,'end':str(full_dates[-1].date()),
              'parameter_candidates_per_search':trials,'finalists_per_search':finalists,'tuning_frequency':'quarterly',
              'refit_frequency':'monthly','inner_folds':3,'seed':tuning.SEED,
              'cache_manifest':json.loads((output/'input/input_manifest.json').read_text()),
              'sources':{name:file_hash(Path(base.__file__).parent/name) for name in source_names}}
    (output/'run_config.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    LOG.info('完整实验完成：%s',output)
