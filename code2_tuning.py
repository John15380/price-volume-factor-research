"""只读训练历史的相关性筛选、分阶段时间验证搜索与滚动XGBoost。"""
from pathlib import Path
import json
import logging
import time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import xgboost

import code2_new as base
from code2_xgboost import make_labels, training_sample

LOG=logging.getLogger('tuning')
SEED=20261115


def rank_ic_by_day(frame, prediction):
    f=pd.DataFrame({'TradingDay':frame.TradingDay.to_numpy(),'prediction':prediction,
                    'target':frame.Target.to_numpy()})
    ranked=f.groupby('TradingDay')[['prediction','target']].rank()
    centered=ranked-ranked.groupby(f.TradingDay).transform('mean')
    product=(centered.prediction*centered.target).groupby(f.TradingDay).sum()
    var1=centered.prediction.pow(2).groupby(f.TradingDay).sum()
    var2=centered.target.pow(2).groupby(f.TradingDay).sum()
    return product/np.sqrt(var1*var2).replace(0,np.nan)


def selector_statistics(train, feature_cols):
    """截面相关逐日等权；质量为训练期日RankIC绝对均值/标准差。"""
    corr=train.groupby('TradingDay')[feature_cols+['Target']].corr(method='spearman',min_periods=20)
    average=corr.groupby(level=1).mean().reindex(index=feature_cols,columns=feature_cols)
    stats=[]
    for factor in feature_cols:
        values=corr.xs(factor,level=1).Target.dropna()
        mean,sd=values.mean(),values.std(ddof=1)
        quality=abs(mean)/max(sd,.01) if np.isfinite(mean) and np.isfinite(sd) else -np.inf
        stats.append({'Feature':factor,'MeanRankIC':mean,'RankICStd':sd,'Quality':quality,
                      'ICDays':len(values),'Coverage':float(train[factor].notna().mean())})
    return pd.DataFrame(stats),average


def choose_features(stats, corr, threshold=.8, max_features=6):
    ranked=stats.sort_values(['Quality','Feature'],ascending=[False,True])
    selected=[];records=[]
    for row in ranked.to_dict('records'):
        factor=row['Feature'];reason='selected';duplicate=''
        if row['Coverage']<.5 or row['ICDays']<20 or not np.isfinite(row['Quality']):
            reason='insufficient_history'
        elif len(selected)>=max_features:
            reason='feature_budget'
        else:
            for kept in selected:
                value=corr.loc[factor,kept]
                if not np.isfinite(value) or abs(value)>=threshold:
                    reason='correlated' if np.isfinite(value) else 'unknown_correlation'
                    duplicate=kept;break
        if reason=='selected':selected.append(factor)
        records.append({**row,'Selected':reason=='selected','Reason':reason,'Representative':duplicate,
                        'CorrelationThreshold':threshold,'MaxFeatures':max_features})
    if not selected:raise ValueError('训练历史中没有可用的候选因子。')
    return selected,pd.DataFrame(records)


def purged_folds(train, n_splits=3, validation_days=21):
    dates=pd.DatetimeIndex(sorted(train.TradingDay.unique()))
    if len(dates)<n_splits*validation_days+80:raise ValueError('时间验证历史不足。')
    folds=[]
    for i in range(n_splits):
        start=len(dates)-(n_splits-i)*validation_days
        val_dates=dates[start:start+validation_days]
        tr=train[train.TradingDay.lt(val_dates[0])&train.LabelEnd.lt(val_dates[0])].copy()
        va=train[train.TradingDay.isin(val_dates)].copy()
        if tr.TradingDay.nunique()<60:raise ValueError('purge后折内训练期不足60天。')
        assert tr.LabelEnd.max()<va.TradingDay.min()
        folds.append((tr,va))
    return folds


def candidate_parameters(count, seed):
    if count<4:raise ValueError('完整搜索至少包含4个候选。')
    rng=np.random.default_rng(seed);out=[]
    for i in range(count):
        out.append({'max_depth':int(rng.choice([2,3,4,5,6])),
                    'learning_rate':float(np.exp(rng.uniform(np.log(.015),np.log(.2)))),
                    'min_child_weight':float(rng.choice([1,3,10,30,60,100])),
                    'subsample':float(rng.uniform(.55,1.)),
                    'colsample_bytree':float(rng.uniform(.55,1.)),
                    'reg_alpha':0. if rng.random()<.25 else float(10**rng.uniform(-4,.3)),
                    'reg_lambda':float(10**rng.uniform(-1,2)),
                    'gamma':0. if rng.random()<.5 else float(10**rng.uniform(-5,-1)),
                    'max_bin':int(rng.choice([128,256])),
                    'corr_threshold':float(rng.choice([.7,.8,.9])),
                    'max_features':int(rng.choice([4,6,9]))})
    # 固定的常用区域也纳入搜索；不是测试期选择的基线。
    for i,(depth,rate) in enumerate([(2,.03),(3,.05),(4,.08),(5,.04)]):
        out[i].update(max_depth=depth,learning_rate=rate,min_child_weight=10.,reg_alpha=0.,
                      reg_lambda=5.,gamma=0.,subsample=.85,colsample_bytree=.9,corr_threshold=.8,max_features=6)
    return out


def sample_weights(frame):
    w=1/frame.groupby('TradingDay').SecuCode.transform('count').to_numpy(float)
    return w/w.mean()


def fitted_trial(tr,va,cols,parameters,max_trees,patience,trial_id):
    params={k:v for k,v in parameters.items() if k not in ('corr_threshold','max_features')}
    model=XGBRegressor(**params,n_estimators=max_trees,early_stopping_rounds=patience,
        eval_metric='rmse',objective='reg:squarederror',tree_method='hist',
        n_jobs=2,random_state=SEED+trial_id,verbosity=0)
    model.fit(tr[cols].astype('float32'),tr.Target,sample_weight=sample_weights(tr),
              eval_set=[(va[cols].astype('float32'),va.Target)],
              sample_weight_eval_set=[sample_weights(va)],verbose=False)
    values=rank_ic_by_day(va,model.predict(va[cols].astype('float32')))
    score=float(values.mean()) if values.notna().any() else -1.
    return score,int(model.best_iteration+1)


def tune(train, feature_cols, destination, trials=96, finalists=16, seed=SEED):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    folds=purged_folds(train)
    candidates=candidate_parameters(trials,seed)
    stats=[selector_statistics(tr,feature_cols) for tr,_ in folds]
    selections={}
    for fold in range(3):
        for threshold in [.7,.8,.9]:
            for budget in [4,6,9]:
                selections[(fold,threshold,budget)]=choose_features(*stats[fold],threshold,budget)[0]
    for i,(tr,va) in enumerate(folds):
        assert tr.LabelEnd.max()<va.TradingDay.min()
    split_records=[{'Fold':i,'TrainSignalStart':tr.TradingDay.min(),'TrainSignalEnd':tr.TradingDay.max(),
                    'TrainMaxLabelEnd':tr.LabelEnd.max(),'ValidationStart':va.TradingDay.min(),
                    'ValidationEnd':va.TradingDay.max(),'ValidationMaxLabelEnd':va.LabelEnd.max(),
                    'TrainRows':len(tr),'ValidationRows':len(va)} for i,(tr,va) in enumerate(folds)]
    pd.DataFrame(split_records).to_csv(destination/'splits.csv',index=False)
    coarse=[]
    for i,params in enumerate(candidates):
        tr,va=folds[-1];cols=selections[(2,params['corr_threshold'],params['max_features'])]
        score,trees=fitted_trial(tr,va,cols,params,300,25,i)
        coarse.append({'Trial':i,'Stage':'coarse','Fold':2,'RankIC':score,'BestTrees':trees,
                       'Features':'|'.join(cols),**params})
        if (i+1)%24==0:LOG.info('%s 初筛 %d/%d',destination.name,i+1,trials)
    pd.DataFrame(coarse).to_csv(destination/'coarse_trials.csv',index=False)
    promoted=sorted(coarse,key=lambda q:(-q['RankIC'],q['Trial']))[:min(finalists,trials)]
    fine=[];summary=[]
    for record in promoted:
        i=record['Trial'];params=candidates[i];scores=[];trees=[]
        for fold,(tr,va) in enumerate(folds):
            cols=selections[(fold,params['corr_threshold'],params['max_features'])]
            score,best=fitted_trial(tr,va,cols,params,1200,50,i)
            scores.append(score);trees.append(best)
            fine.append({'Trial':i,'Stage':'fine','Fold':fold,'RankIC':score,'BestTrees':best,
                         'Features':'|'.join(cols),**params})
        mean=float(np.mean(scores));sd=float(np.std(scores))
        summary.append({'Trial':i,'MeanRankIC':mean,'StdRankIC':sd,'SelectionScore':mean-.1*sd,
                        'Trees':max(1,int(np.median(trees))),**params})
    pd.DataFrame(fine).to_csv(destination/'fine_trials.csv',index=False)
    ranking=pd.DataFrame(summary).sort_values(['SelectionScore','MeanRankIC','Trial'],ascending=[False,False,True])
    ranking.to_csv(destination/'search_summary.csv',index=False)
    winner=ranking.iloc[0];chosen=candidates[int(winner.Trial)].copy()
    chosen['n_estimators']=int(winner.Trees)
    result={'parameters':chosen,'winner_trial':int(winner.Trial),'inner_mean_rank_ic':float(winner.MeanRankIC),
            'inner_rank_ic_std':float(winner.StdRankIC),'selection_score':float(winner.SelectionScore),
            'candidates':trials,'finalists':len(promoted),'cv_fits':trials+len(promoted)*3,
            'max_training_label_end':str(train.LabelEnd.max()),'validation_folds':3,
            'screening_tree_cap':300,'final_tree_cap':1200,'early_stop_metric':'validation weighted RMSE',
            'search_metric':'mean validation daily RankIC - 0.1 * std across 3 chronological folds'}
    (destination/'best_parameters.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def walk_forward(features,market,days,output,predict_start='2017-02-01',window_months=12,
                 feature_cols=None,trials=96,finalists=16):
    from code2_research import FACTORS
    feature_cols=list(FACTORS if feature_cols is None else feature_cols)
    if any('Tail' in f or 'Legacy' in f for f in feature_cols):raise ValueError('已剔除的因子不可进入模型。')
    features=features[base.KEYS+feature_cols].copy()
    features=features[features[feature_cols].notna().sum(axis=1)>=3].sort_values(base.KEYS)
    labels=make_labels(features,market,days)
    labeled=features.merge(labels,on=base.KEYS,validate='one_to_one')
    months=sorted(pd.DatetimeIndex(market.dates[market.dates>=pd.Timestamp(predict_start)]).to_period('M').unique())
    output=Path(output);(output/'models').mkdir(parents=True,exist_ok=True)
    forecasts=[];logs=[];selection_logs=[];importance=[];active_quarter=None;best=None
    t0=time.monotonic()
    for month in months:
        dates=market.dates[market.dates.to_period('M')==month];cutoff=dates[0]
        train=training_sample(labeled,cutoff,window_months)
        test=features[features.TradingDay.dt.to_period('M').eq(month)&features.TradingDay.ge(pd.Timestamp(predict_start))]
        if test.empty:continue
        if train.TradingDay.nunique()<160:raise ValueError('外层训练数据不足160天。')
        assert train.LabelEnd.max()<cutoff
        quarter=month.asfreq('Q')
        if quarter!=active_quarter:
            LOG.info('%d日 %s 开始训练期调参：%d候选、%d晋级、3个purge时间折',days,quarter,trials,finalists)
            best=tune(train,feature_cols,output/'tuning'/str(quarter),trials,finalists,
                      SEED+days*100+int(quarter.ordinal))
            active_quarter=quarter;tuning_cutoff=cutoff
        params=best['parameters'].copy();threshold=params.pop('corr_threshold');budget=params.pop('max_features')
        stats,corr=selector_statistics(train,feature_cols)
        selected,decisions=choose_features(stats,corr,threshold,budget)
        decisions['PredictionMonth']=str(month);decisions['Cutoff']=cutoff
        decisions['MaxLabelEnd']=train.LabelEnd.max();selection_logs.append(decisions)
        select_dir=output/'selection';select_dir.mkdir(exist_ok=True)
        corr.to_csv(select_dir/f'{month}_correlation.csv')
        model=XGBRegressor(**params,n_jobs=2,random_state=SEED,objective='reg:squarederror',tree_method='hist',verbosity=0)
        model.fit(train[selected].astype('float32'),train.Target,sample_weight=sample_weights(train))
        pred=test[base.KEYS].copy();pred['Pred_Score']=model.predict(test[selected].astype('float32'))
        pred.loc[test[selected].notna().sum(axis=1).to_numpy()==0,'Pred_Score']=np.nan
        forecasts.append(pred);model.save_model(output/'models'/f'{month}.json')
        logs.append({'PredictionMonth':str(month),'Cutoff':cutoff,'MaxLabelEnd':train.LabelEnd.max(),
                     'TrainSignalStart':train.TradingDay.min(),'TrainSignalEnd':train.TradingDay.max(),
                     'TrainRows':len(train),'TrainDays':train.TradingDay.nunique(),'PredictionRows':len(test),
                     'SelectedFeatures':'|'.join(selected),'FeatureCount':len(selected),
                     'CorrelationThreshold':threshold,'MaxFeatures':budget,'TuningQuarter':str(active_quarter),
                     'TuningCutoff':tuning_cutoff,'Trees':params['n_estimators'],
                     'InnerRankIC':best['inner_mean_rank_ic'],'Horizon':days})
        importance.extend({'PredictionMonth':str(month),'Feature':f,'Importance':float(v)} for f,v in zip(selected,model.feature_importances_))
        LOG.info('%d日 %s 训练%d行，入选%s，树%d；累计%.1f秒',days,month,len(train),','.join(selected),params['n_estimators'],time.monotonic()-t0)
    predictions=pd.concat(forecasts,ignore_index=True).sort_values(base.KEYS)
    predictions.to_parquet(output/'predictions.parquet',index=False)
    pd.DataFrame(logs).to_csv(output/'training_log.csv',index=False)
    pd.concat(selection_logs,ignore_index=True).to_csv(output/'feature_selection.csv',index=False)
    pd.DataFrame(importance).to_csv(output/'feature_importance.csv',index=False)
    (output/'model_config.json').write_text(json.dumps({'xgboost_version':xgboost.__version__,'candidate_features':feature_cols,
        'horizon':days,'predict_start':predict_start,'training_window_months':window_months,'trials':trials,
        'finalists':finalists,'quarterly_tuning':True,'monthly_selection_and_refit':True,'seed':SEED,
        'purge':'training LabelEnd < validation start / outer prediction month start',
        'native_missing_values':True,'prediction_admission':'at least 3 contemporaneous available candidate factors',
        'correlation':'training daily cross-sectional Spearman, equally weighted dates; greedy representative by |mean RankIC|/std',
        'target':'cross-section demeaned adjusted N-day open return'},ensure_ascii=False,indent=2))
    return predictions
