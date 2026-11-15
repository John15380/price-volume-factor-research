"""从第二轮研报修订的真实产物生成对照报告；不手工调整收益。"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import code2_new as base
import code2_research as paper
import code2_trading_data as trading

R=ROOT/'results_code2_constraints'
A=ROOT/'results_research_all_a_constraints'
for p in [HERE/'figures',HERE/'tables',HERE/'source',HERE/'source/tests',HERE/'source/audit',HERE/'provenance',HERE/'logs']:
    p.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/pdfs/research_report/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'tmp/pdfs/research_report/cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams.update({'font.size':10.5,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8.5,
                     'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

NAMES={'HF_SmartMoney':'SmartMoney','HF_CPV':'CPV','HF_CDPDP':'CDPDP-prime',
       'HF_IdealAmplitude':'IdealAmplitude','HF_IdealReversal':'IdealReversal','HF_CDPP':'CDPP',
       'HF_CDPDP4':'CDPDP-4','HF_LowAmpMomentum':'LowAmpMomentum','HF_Skewness':'Skewness',
       'HF_Tail_Reversal':'TailReturn','Legacy_SmartDailyMean':'DailyMean-Q',
       'Legacy_CPVMean':'CPV-Mean','Legacy_CDPDPRawSum':'RawSum-CDPDP','Legacy_Amplitude':'OrdinaryAmp',
       'Pred_Score':'XGBoost','ICIR_12':'ICIR-12'}
FREQ={'Weekly':'周频5日','BiWeekly':'双周10日','Monthly':'固定20日','MonthEnd':'自然月末'}
full=pd.read_csv(R/'summary.csv')
compare=pd.read_csv(R/'comparison/summary.csv')
all_a=pd.read_csv(A/'summary.csv')
paired=pd.read_csv(R/'paired_summary.csv')
sensitivity=pd.read_csv(R/'sensitivity.csv')
config=json.loads((R/'research_config.json').read_text())
for name,digest in config['sources'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name


def pct(x):return f'{x*100:.2f}\\%'
def line(*values):return ' & '.join(str(x) for x in values)+r' \\'+'\n'
def table(name,text):(HERE/'tables'/f'{name}.tex').write_text(text)
def pick(frame,factor,period='MonthEnd',cost='Net'):
    return frame[frame.Factor.eq(factor)&frame.Period.eq(period)&frame.Portfolio.eq('TheoreticalLS')&frame.Cost.eq(cost)].iloc[0]
def save(fig,name):
    fig.savefig(HERE/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(HERE/'figures'/f'{name}.png',bbox_inches='tight',dpi=150)
    plt.close(fig)


counts={'series':0,'trades':0,'rows':0}
def verify(parent,summary,paired_mode=False):
    if paired_mode:
        cases=[]
        for group in (parent).iterdir():
            if group.is_dir():
                for case in group.iterdir():
                    if case.is_dir():cases.append(case)
    else:
        cases=[parent/factor/period for factor,period in summary[['Factor','Period']].drop_duplicates().itertuples(index=False,name=None)]
    for folder in cases:
        trades=pd.read_parquet(folder/'trades.parquet')
        ledger=pd.read_csv(folder/'daily_ledger.csv',parse_dates=['TradingDay'])
        np.testing.assert_allclose(trades.Cost,trades.Notional*.0015,atol=1e-12)
        for portfolio,f in ledger.groupby('Portfolio'):
            np.testing.assert_allclose(f.NAV,f.Cash+f.LongValue-f.ShortValue,atol=1e-10)
            np.testing.assert_allclose(f.NAV,np.cumprod(1+f.Return),atol=1e-10)
            b=trades[trades.Portfolio.eq(portfolio)].copy()
            b['Flow']=b.DeltaUnits*b.AdjPrice+b.Cost
            flow=b.groupby('TradingDay').Flow.sum().reindex(f.TradingDay,fill_value=0).to_numpy()
            np.testing.assert_allclose(f.Cash,1-np.cumsum(flow+f.BorrowCost.to_numpy()),atol=1e-10)
            if not paired_mode:
                record=summary[summary.Factor.eq(folder.parent.name)&summary.Period.eq(folder.name)&summary.Portfolio.eq(portfolio)&summary.Cost.eq('Net')].iloc[0]
                metric=base.performance(f.Return,base.Config())
                for key in ['FinalNAV','AnnualReturn','Sharpe','MaxDrawdown']:
                    np.testing.assert_allclose(record[key],metric[key],rtol=1e-9,atol=1e-12)
            counts['series']+=1
        counts['rows']+=len(ledger);counts['trades']+=len(trades)


verify(R,full);verify(R/'comparison',compare);verify(A,all_a);verify(R/'paired',paired,True)
for name in base.PERIODS:
    log=pd.read_csv(R/'xgboost'/name/'training_log.csv',parse_dates=['Cutoff','MaxLabelEnd'])
    assert log.MaxLabelEnd.lt(log.Cutoff).all()
    weight=pd.read_csv(R/'icir'/name/'weights.csv',parse_dates=['TradingDay','MaxLabelEnd'])
    assert weight.MaxLabelEnd.lt(weight.TradingDay).all() and weight.ICPeriods.eq(12).all()

text=''
for f in paper.PRIMARY+paper.ADDITIONAL:
    net,gross=pick(full,f),pick(full,f,cost='Gross')
    text+=line(NAMES[f],pct(gross.AnnualReturn),pct(net.AnnualReturn),f'{net.Sharpe:.2f}',pct(net.MaxDrawdown),f'{net.RankIC_Raw:.3f}',f'{net.MeanSignalStocks:.0f}')
table('monthly',text)

text=''
for f in ['HF_CPV','HF_CDPP','HF_CDPDP','HF_CDPDP4']:
    s,a=pick(full,f),pick(all_a,f)
    text+=line(NAMES[f],pct(s.AnnualReturn),f'{s.Sharpe:.2f}',pct(a.AnnualReturn),f'{a.Sharpe:.2f}',pct(a.MaxDrawdown),f'{a.MeanSignalStocks:.0f}')
table('universe',text)

text=''
for f,old in [('HF_SmartMoney','Legacy_SmartDailyMean'),('HF_CPV','Legacy_CPVMean'),('HF_CDPDP','Legacy_CDPDPRawSum'),('HF_IdealAmplitude','Legacy_Amplitude')]:
    n,o=pick(paired,f),pick(paired,old)
    text+=line(NAMES[f],pct(o.AnnualReturn),pct(n.AnnualReturn),f'{100*(n.AnnualReturn-o.AnnualReturn):+.2f}',f'{o.Sharpe:.2f}',f'{n.Sharpe:.2f}')
table('paired',text)

text=''
for f in ['HF_SmartMoney','HF_CPV','HF_CDPDP','HF_IdealAmplitude']:
    subset=sensitivity[sensitivity.Factor.eq(f)]
    values=[]
    for leg,execution,fee in [(.5,'Actual',.0015),(1.,'Actual',.0015),(1.,'QuotedOpenBenchmark',.0015),(1.,'Actual',0.)]:
        x=subset[subset.LegWeight.eq(leg)&subset.Execution.eq(execution)&subset.OneWayFee.eq(fee)].iloc[0]
        values.append(pct(x.AnnualReturn))
    text+=line(NAMES[f],*values)
table('sensitivity',text)

text=''
for name in base.PERIODS:
    for i,f in enumerate(list(paper.PRIMARY)+['ICIR_12','Pred_Score']):
        q=pick(compare,f,name)
        if f=='Pred_Score':text+=r'\rowcolor{blue!7}'+'\n'
        text+=line(FREQ[name] if i==0 else '',NAMES[f],f'{q.IC_Raw:.3f}',pct(q.AnnualReturn),f'{q.Sharpe:.2f}',pct(q.MaxDrawdown),f'{q.MeanSignalStocks:.0f}')
    text+=r'\midrule'+'\n'
table('comparison',text)

text=''
for name in base.PERIODS:
    q=pick(compare,'Pred_Score',name)
    p=sensitivity[sensitivity.Factor.eq('Pred_Score')&sensitivity.Period.eq(name)&sensitivity.LegWeight.eq(.5)&sensitivity.Execution.eq('Actual')&sensitivity.OneWayFee.eq(.0015)].iloc[0]
    text+=line(FREQ[name],pct(q.AnnualReturn),f'{q.Sharpe:.2f}',pct(q.MaxDrawdown),pct(p.AnnualReturn),f'{p.Sharpe:.2f}',pct(p.MaxDrawdown))
table('model_exposure',text)

# 配对提升图：负变化也原样保留，避免把“更忠实”误写成“每个都更赚钱”。
fig,ax=plt.subplots(figsize=(8,3.8))
labels=[];old=[];new=[]
for f,o in [('HF_SmartMoney','Legacy_SmartDailyMean'),('HF_CPV','Legacy_CPVMean'),('HF_CDPDP','Legacy_CDPDPRawSum'),('HF_IdealAmplitude','Legacy_Amplitude')]:
    labels.append(NAMES[f]);old.append(pick(paired,o).AnnualReturn*100);new.append(pick(paired,f).AnnualReturn*100)
x=np.arange(4)
ax.bar(x-.18,old,width=.35,label='Simplified construction',color='#8397a8')
ax.bar(x+.18,new,width=.35,label='Paper construction',color='#b23a48')
ax.set_xticks(x,labels);ax.set_ylabel('Annual return (%)');ax.legend(frameon=False,loc='upper center',ncol=2,bbox_to_anchor=(.5,1.15));ax.grid(axis='y',alpha=.2)
fig.tight_layout();save(fig,'paired')

def plot_nav(parent,factor,period,filename):
    ledger=pd.read_csv(parent/factor/period/'daily_ledger.csv',parse_dates=['TradingDay'])
    fig,axes=plt.subplots(2,1,figsize=(8.2,5.1),sharex=True,gridspec_kw={'height_ratios':[1.1,1]})
    colors=['#61748b','#5c91ad','#57a195','#d7a154','#b63d45']
    for i in range(1,6):
        f=ledger[ledger.Portfolio.eq(f'Group{i}')]
        axes[0].plot(f.TradingDay,f.NAV,label=f'G{i}',color=colors[i-1],lw=1.1)
    axes[0].legend(frameon=False,ncol=5,loc='best')
    axes[0].set_title(NAMES[factor]+' | '+period+' | '+('CSI500' if parent==R else 'OOS'))
    axes[0].set_ylabel('Long-only NAV')
    f=ledger[ledger.Portfolio.eq('TheoreticalLS')]
    axes[1].plot(f.TradingDay,f.NAV,color='#244c68',lw=1.3)
    metrics=base.performance(f.Return,base.Config())
    axes[1].set_title(f"L/S 100%/100% targets | CAGR {metrics['AnnualReturn']:.2%} | Sharpe {metrics['Sharpe']:.2f} | MDD {metrics['MaxDrawdown']:.2%}",fontsize=10)
    axes[1].set_ylabel('Net L/S NAV');axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for a in axes:a.axhline(1,color='gray',lw=.5);a.grid(alpha=.2)
    fig.tight_layout();save(fig,filename)

for factor in paper.PRIMARY:
    plot_nav(R,factor,'MonthEnd',factor+'_MonthEnd')
plot_nav(R/'comparison','Pred_Score','BiWeekly','model_biweekly')
fig,ax=plt.subplots(figsize=(8.2,3.8))
for name in base.PERIODS:
    d=pd.read_csv(R/'comparison/Pred_Score'/name/'daily_ledger.csv',parse_dates=['TradingDay'])
    d=d[d.Portfolio.eq('TheoreticalLS')]
    ax.plot(d.TradingDay,d.NAV,label=name)
ax.legend(frameon=False);ax.set_ylabel('Net L/S NAV (100%/100%)');ax.grid(alpha=.2);ax.axhline(1,color='gray',lw=.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.tight_layout();save(fig,'model_horizons')

text=''
for i in range(0,6,2):
    f,g=paper.PRIMARY[i:i+2]
    text+=r'\clearpage'+'\n'
    if i==0:text+=r'\section{研报因子自然月末净值图}'+'\n'
    text+=rf'\navfigure{{{f}_MonthEnd}}{{{NAMES[f]}自然月末回测}}'+'\n\\vfill\n'
    text+=rf'\navfigure{{{g}_MonthEnd}}{{{NAMES[g]}自然月末回测}}'+'\n'
(HERE/'figures_appendix.tex').write_text(text)

macros={'Trades':f"{counts['trades']:,}",'Series':str(counts['series']),
        'CPVCSI':pct(pick(full,'HF_CPV').AnnualReturn),'CDPPCSI':pct(pick(full,'HF_CDPP').AnnualReturn),
        'CPVAll':pct(pick(all_a,'HF_CPV').AnnualReturn),'CDPDPAll':pct(pick(all_a,'HF_CDPDP').AnnualReturn)}
macros['CandidateCount']=str(config['candidates'])
macros['PrimaryMeanStocks']=f"{pick(full,'HF_CPV').MeanSignalStocks:.1f}"
pred_sample=pd.read_parquet(R/'xgboost/Weekly/predictions.parquet')
macros['ModelMeanStocks']=f"{pred_sample.groupby('TradingDay').size().mean():.1f}"
macros['ModelRows']=f"{len(pred_sample):,}"
for period,suffix in [('Weekly','Week'),('BiWeekly','Biweek'),('Monthly','Month')]:
    q=pick(compare,'Pred_Score',period)
    macros['XGB'+suffix]=pct(q.AnnualReturn)
    macros['XGB'+suffix+'Sharpe']=f'{q.Sharpe:.2f}'
(HERE/'values.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
for name in ['summary.csv','paired_summary.csv','sensitivity.csv','feature_coverage.csv']:
    shutil.copy2(R/name,HERE/'tables'/name)
shutil.copy2(R/'comparison/summary.csv',HERE/'tables/oos_summary.csv')
shutil.copy2(A/'summary.csv',HERE/'tables/all_a_summary.csv')
for name in ['code2_new.py','code2_research.py','code2_xgboost.py','code2_trading_data.py','code2_new_说明.md']:
    shutil.copy2(ROOT/name,HERE/'source'/name)
for name in ['full_a_price_factors.py','constraints_sensitivity.py','研报逐项核对.md']:
    shutil.copy2(ROOT/'audit'/name,HERE/'source/audit'/name)
for path in (ROOT/'tests').glob('test_code2*.py'):
    shutil.copy2(path,HERE/'source/tests'/path.name)
for parent,prefix,names in [
        (R,'csi500',['research_config.json','trading_data_diagnostics.json','missing_limits.csv','inconsistent_limits.csv']),
        (A,'all_a',['limitations.json','trading_data_diagnostics.json','missing_limits.csv','inconsistent_limits.csv'])]:
    for name in names:shutil.copy2(parent/name,HERE/'provenance'/f'{prefix}_{name}')
for name in ['code2_constraints_final_run.log','full_a_constraints_run.log','constraints_sensitivity.log','constraints_tests_final.log']:
    shutil.copy2(Path('/private/tmp')/name,HERE/'logs'/name)
for name in ['char.png','logo.png']:shutil.copy2(ROOT/'report/figures'/name,HERE/'figures'/name)
receipt={**counts,'tests_passed':54,'sensitivity_cases':len(sensitivity),'training_label_time_checked':True,'icir_12_realized_periods_checked':True,
         'research_source_hashes':config['sources'],'original_code2_unchanged':hashlib.sha256((ROOT/'code2.py').read_bytes()).hexdigest()==json.loads((ROOT/'audit/code2_check_results.json').read_text())['source_sha256']}
(HERE/'validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
template=(HERE/'report_template.tex').read_text()
for path in (HERE/'tables').glob('*.tex'):template=template.replace('\\input{tables/'+path.name+'}',path.read_text())
(HERE/'report_ready.tex').write_text(template)
print(json.dumps(receipt,ensure_ascii=False,indent=2))
