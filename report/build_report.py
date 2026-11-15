"""从当前唯一results目录生成报告表格/图像，并逐项核对账本与时间隔离。"""
from pathlib import Path
import hashlib
import json
import os
import re
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import code2_new as base
import code2_research as research

R=ROOT/'results'
for p in [HERE/'tables',HERE/'figures',R/'audit']:
    p.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/pdfs/current/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'legend.fontsize':7,'pdf.fonttype':42,
                     'axes.spines.right':False,'axes.spines.top':False})

NAMES={'HF_SmartMoney':'SmartMoney','HF_CPV':'CPV','HF_CDPDP':'CDPDP-prime',
       'HF_IdealAmplitude':'IdealAmplitude','HF_IdealReversal':'IdealReversal','HF_CDPP':'CDPP',
       'HF_CDPDP4':'CDPDP-4','HF_LowAmpMomentum':'LowAmpMomentum','HF_Skewness':'Skewness',
       'Pred_Score':'XGBoost'}
TEX={**NAMES,'HF_CDPDP':r'CDPDP$^{\prime}$'}
FREQ={'Weekly':'5日','BiWeekly':'10日','Monthly':'20日','MonthEnd':'自然月'}
full=pd.read_csv(R/'factor_summary.csv')
oos=pd.read_csv(R/'oos_summary.csv')
config=json.loads((R/'run_config.json').read_text())
assert config['groups']==10 and config['round_trip_fee']==.003
for name,digest in config['sources'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
assert set(full.Factor)==set(research.FACTORS) and len(research.FACTORS)==9
assert not any('Tail' in f or 'Legacy' in f for f in full.Factor.unique())


def pick(frame,factor,period,portfolio):
    return frame[frame.Factor.eq(factor)&frame.Period.eq(period)&frame.Portfolio.eq(portfolio)].iloc[0]


def pct(x):return f'{x*100:.2f}\\%'
def line(*values):return ' & '.join(str(v) for v in values)+r' \\'+'\n'
def table(name,text):(HERE/'tables'/f'{name}.tex').write_text(text)
def save(fig,name):
    fig.savefig(HERE/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(HERE/'figures'/f'{name}.png',bbox_inches='tight',dpi=150)
    plt.close(fig)


counts={'series':0,'trades':0,'ledger_rows':0}
for parent,summary in [(R/'factors',full),(R/'oos',oos)]:
    for factor,period in summary[['Factor','Period']].drop_duplicates().itertuples(index=False,name=None):
        folder=parent/factor/period
        trades=pd.read_parquet(folder/'trades.parquet')
        ledger=pd.read_csv(folder/'daily_ledger.csv',parse_dates=['TradingDay'])
        assert ledger.Portfolio.nunique()==11
        assert set(ledger.Portfolio)=={*[f'Group{i}' for i in range(1,11)],'TheoreticalLS'}
        np.testing.assert_allclose(trades.Cost,trades.Notional*.0015,atol=1e-12)
        groups=pd.read_parquet(folder/'signal_groups.parquet')
        if len(groups):assert set(groups.Group.unique())==set(range(1,11))
        for portfolio,f in ledger.groupby('Portfolio'):
            np.testing.assert_allclose(f.NAV,f.Cash+f.LongValue-f.ShortValue,atol=1e-10)
            np.testing.assert_allclose(f.NAV,np.cumprod(1+f.Return),atol=1e-10)
            b=trades[trades.Portfolio.eq(portfolio)].copy()
            b['Flow']=b.DeltaUnits*b.AdjPrice+b.Cost
            flow=b.groupby('TradingDay').Flow.sum().reindex(f.TradingDay,fill_value=0).to_numpy()
            np.testing.assert_allclose(f.Cash,1-np.cumsum(flow+f.BorrowCost.to_numpy()),atol=1e-10)
            metrics=base.performance(f.Return,base.Config())
            record=pick(summary,factor,period,portfolio)
            for key in ['FinalNAV','AnnualReturn','Sharpe','MaxDrawdown']:
                np.testing.assert_allclose(record[key],metrics[key],rtol=1e-9,atol=1e-12)
            counts['series']+=1
        counts['ledger_rows']+=len(ledger);counts['trades']+=len(trades)

selection=[];training=[];best_rows=[];coarse_total=fine_total=0;searches=0
for name,days in base.PERIODS.items():
    parent=R/'models'/name
    log=pd.read_csv(parent/'training_log.csv',parse_dates=['Cutoff','MaxLabelEnd','TuningCutoff'])
    assert len(log)==23 and log.MaxLabelEnd.lt(log.Cutoff).all() and log.TuningCutoff.le(log.Cutoff).all()
    log['Period']=name;training.append(log)
    decisions=pd.read_csv(parent/'feature_selection.csv',parse_dates=['Cutoff','MaxLabelEnd'])
    assert decisions.MaxLabelEnd.lt(decisions.Cutoff).all()
    assert not any('Tail' in f or 'Legacy' in f for f in decisions.Feature.unique())
    decisions['Period']=name;selection.append(decisions)
    for month,chosen in decisions[decisions.Selected].groupby('PredictionMonth'):
        corr=pd.read_csv(parent/'selection'/f'{month}_correlation.csv',index_col=0)
        features=chosen.Feature.tolist();threshold=chosen.CorrelationThreshold.iloc[0]
        assert len(features)<=chosen.MaxFeatures.iloc[0]
        for a in features:
            for b in features:
                if a!=b:assert abs(corr.loc[a,b])<threshold+1e-12
    for folder in sorted((parent/'tuning').iterdir()):
        if not folder.is_dir():continue
        coarse=pd.read_csv(folder/'coarse_trials.csv');fine=pd.read_csv(folder/'fine_trials.csv')
        best=json.loads((folder/'best_parameters.json').read_text())
        assert len(coarse)==96 and len(fine)==48 and fine.Fold.nunique()==3
        cutoff=log.loc[log.TuningQuarter.eq(folder.name),'TuningCutoff'].min()
        assert pd.Timestamp(best['max_training_label_end'])<cutoff
        splits=pd.read_csv(folder/'splits.csv',parse_dates=['TrainMaxLabelEnd','ValidationStart','ValidationEnd','ValidationMaxLabelEnd'])
        assert splits.TrainMaxLabelEnd.lt(splits.ValidationStart).all()
        assert splits.ValidationMaxLabelEnd.lt(cutoff).all()
        best_rows.append({'Quarter':folder.name,'Period':name,'Days':days,
                          'MeanRankIC':best['inner_mean_rank_ic'],**best['parameters']})
        searches+=1;coarse_total+=len(coarse);fine_total+=len(fine)
selection=pd.concat(selection,ignore_index=True)
training=pd.concat(training,ignore_index=True)
best_frame=pd.DataFrame(best_rows).sort_values(['Quarter','Days'])
assert searches==24 and coarse_total==2304 and fine_total==1152
training.to_csv(R/'training_summary.csv',index=False)
selection.to_csv(R/'feature_selection_summary.csv',index=False)
best_frame.to_csv(R/'best_parameters.csv',index=False)

text=''
for factor in research.FACTORS:
    for i,period in enumerate(list(base.PERIODS)+['MonthEnd']):
        long=pick(full,factor,period,'Group10');ls=pick(full,factor,period,'TheoreticalLS')
        text+=line(TEX[factor] if i==0 else '',FREQ[period],pct(long.AnnualReturn),f'{long.Sharpe:.2f}',pct(ls.AnnualReturn),f'{ls.Sharpe:.2f}')
    text+=r'\midrule'+'\n' if factor!=research.FACTORS[-1] else ''
table('factors',text)

text=''
for factor in research.FACTORS:
    numbers=[int(selection[selection.Period.eq(period)&selection.Feature.eq(factor)].Selected.sum()) for period in base.PERIODS]
    text+=line(TEX[factor],*numbers)
table('selection',text)

text=''
for row in best_frame.itertuples(index=False):
    text+=line(row.Quarter,FREQ[row.Period],row.max_depth,f'{row.learning_rate:.3f}',row.n_estimators,
               f'{row.corr_threshold:.2f}',row.max_features,f'{row.MeanRankIC:.3f}')
table('tuning',text)

text=''
for period in base.PERIODS:
    long=pick(oos,'Pred_Score',period,'Group10');ls=pick(oos,'Pred_Score',period,'TheoreticalLS')
    text+=line(FREQ[period],pct(long.AnnualReturn),f'{long.Sharpe:.2f}',pct(ls.AnnualReturn),f'{ls.Sharpe:.2f}')
table('model',text)

all_performance=pd.concat([full.assign(Study='FullPeriod'),oos.assign(Study='OOS')],ignore_index=True)
all_performance[all_performance.Portfolio.isin(['Group10','TheoreticalLS'])].to_csv(R/'long_short_summary.csv',index=False)

matrix=pd.read_csv(R/'models/BiWeekly/selection/2017-02_correlation.csv',index_col=0)
matrix=matrix.reindex(index=research.FACTORS,columns=research.FACTORS)
matrix.to_csv(R/'training_factor_correlation.csv')
fig,ax=plt.subplots(figsize=(8,7))
im=ax.imshow(matrix.to_numpy(),vmin=-1,vmax=1,cmap='coolwarm')
labels=[NAMES[f] for f in research.FACTORS]
ax.set_xticks(range(9),labels,rotation=45,ha='right');ax.set_yticks(range(9),labels)
for i in range(9):
    for j in range(9):ax.text(j,i,f'{matrix.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7)
fig.colorbar(im,ax=ax,shrink=.75);fig.tight_layout();save(fig,'correlation')

calendar=pd.DatetimeIndex(sorted(pd.read_parquet(R/'input/daily_market.parquet',columns=['TradingDay']).TradingDay.unique()))


def nav_figure(folder,factor,period,filename):
    ledger=pd.read_csv(folder/factor/period/'daily_ledger.csv',parse_dates=['TradingDay'])
    fig,axes=plt.subplots(2,1,figsize=(8.1,4.7),sharex=True,gridspec_kw={'height_ratios':[1.12,1]})
    baseline=calendar[calendar.get_loc(ledger.TradingDay.min())-1]
    colors=plt.get_cmap('tab10').colors
    for i in range(1,11):
        x=ledger[ledger.Portfolio.eq(f'Group{i}')]
        axes[0].plot([baseline,*x.TradingDay],[1.,*x.NAV],color='black' if i==10 else colors[i-1],
                     lw=1.6 if i==10 else .7,label=f'G{i}',alpha=1 if i==10 else .8)
    axes[0].legend(frameon=False,ncol=5,loc='lower left',fontsize=6.5)
    metrics=base.performance(ledger[ledger.Portfolio.eq('Group10')].Return,base.Config())
    axes[0].set_title(f'{NAMES[factor]} | {period} | G10 CAGR {metrics["AnnualReturn"]:.2%} | Sharpe {metrics["Sharpe"]:.2f}',fontsize=9)
    axes[0].set_ylabel('Long-only NAV')
    x=ledger[ledger.Portfolio.eq('TheoreticalLS')]
    axes[1].plot([baseline,*x.TradingDay],[1.,*x.NAV],color='#234b68',lw=1.2)
    metrics=base.performance(x.Return,base.Config())
    axes[1].set_title(f'L/S 100%/100% | CAGR {metrics["AnnualReturn"]:.2%} | Sharpe {metrics["Sharpe"]:.2f}',fontsize=9)
    axes[1].set_ylabel('Net L/S NAV');axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for ax in axes:ax.grid(alpha=.2);ax.axhline(1,color='gray',lw=.5)
    fig.tight_layout();save(fig,filename)


for period in base.PERIODS:nav_figure(R/'oos','Pred_Score',period,'model_'+period)
for factor in research.FACTORS:nav_figure(R/'factors',factor,'MonthEnd',factor+'_MonthEnd')
fig,axes=plt.subplots(1,2,figsize=(9,3.3))
for period in base.PERIODS:
    d=pd.read_csv(R/'oos/Pred_Score'/period/'daily_ledger.csv',parse_dates=['TradingDay'])
    for ax,portfolio in zip(axes,['Group10','TheoreticalLS']):
        f=d[d.Portfolio.eq(portfolio)];ax.plot(f.TradingDay,f.NAV,label=period,lw=1)
for ax,title in zip(axes,['G10 long-only','L/S 100% / 100%']):
    ax.set_title(title);ax.grid(alpha=.2);ax.legend(frameon=False);ax.set_ylabel('Net NAV')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));ax.tick_params(axis='x',rotation=25)
fig.tight_layout();save(fig,'model_horizons')

appendix=''
for i in range(0,9,2):
    appendix+=r'\clearpage'+'\n'
    if i==0:appendix+=r'\section{九因子自然月末十组净值}'+'\n'
    for j,factor in enumerate(research.FACTORS[i:i+2]):
        if j:appendix+=r'\vfill'+'\n'
        appendix+=rf'\resultfigure{{{factor}_MonthEnd}}{{{TEX[factor]}自然月末回测}}'+'\n'
(HERE/'appendix_figures.tex').write_text(appendix)

test_log=(R/'logs/tests.log').read_text()
tests=int(re.search(r'Ran (\d+) tests',test_log).group(1));assert '\nOK' in test_log
macros={'Searches':searches,'CoarseFits':coarse_total,'FineFits':fine_total,'CVFits':coarse_total+fine_total,
        'Models':len(training),'Series':counts['series'],'Trades':f'{counts["trades"]:,}','Tests':tests}
for period,suffix in [('Weekly','Week'),('BiWeekly','Biweek'),('Monthly','Month')]:
    macros['Long'+suffix]=pct(pick(oos,'Pred_Score',period,'Group10').AnnualReturn)
    macros['LS'+suffix]=pct(pick(oos,'Pred_Score',period,'TheoreticalLS').AnnualReturn)
(HERE/'values.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items())+'\n')
template=(HERE/'template.tex').read_text()
for path in (HERE/'tables').glob('*.tex'):template=template.replace('\\input{tables/'+path.name+'}',path.read_text())
(HERE/'report_ready.tex').write_text(template)
receipt={**counts,'tests_passed':tests,'groups':10,'candidates':list(research.FACTORS),'monthly_models':len(training),
         'searches':searches,'cv_fits':coarse_total+fine_total,'round_trip_fee':.003,
         'training_label_time_checked':True,'inner_purge_checked':True,'selected_pair_correlations_checked':True,
         'source_hashes':config['sources']}
(R/'audit/validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False,indent=2))
