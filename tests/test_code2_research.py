import unittest
import tempfile
import numpy as np
import pandas as pd

import code2_new as base
import code2_research as r


class PaperConstructionTests(unittest.TestCase):
    def test_smart_money_is_pooled_not_daily_average(self):
        scores=[np.array([9.,1.]),np.array([1.,0.])]
        volume=[np.array([1.,9.]),np.array([90.,10.])]
        amount=[np.array([20.,90.]),np.array([900.,100.])]
        pooled=r.smart_ratio(np.concatenate(scores),np.concatenate(volume),np.concatenate(amount))
        mean=np.mean([r.smart_ratio(s,v,a) for s,v,a in zip(scores,volume,amount)])
        self.assertNotAlmostEqual(pooled,mean)
        # 排序分数9的一股先入，再从分数1的分钟中取21股；其价格均为10。
        self.assertAlmostEqual(pooled,(230/22)/(1110/110))

    def test_ideal_amplitude_conditions_on_price_rank(self):
        price=np.arange(1.,21.)
        amp=np.arange(20.)/100
        result=r.conditional_factors(price,amp,np.ones(20)*.01,np.arange(20.)+1)
        self.assertAlmostEqual(result[-1,0],amp[-5:].mean()-amp[:5].mean())
        self.assertNotAlmostEqual(result[-1,0],amp.mean())

    def test_ideal_amplitude_selects_valid_days_instead_of_requiring_twenty(self):
        price=np.arange(1.,21.);amp=np.arange(20.)/100
        amp[8]=np.nan
        result=r.conditional_factors(price,amp,np.ones(20)*.01,np.ones(20))
        self.assertAlmostEqual(result[-1,0],amp[-4:].mean()-amp[:4].mean())

    def test_ideal_amplitude_requires_ten_effective_days(self):
        price=np.arange(1.,21.);amp=np.arange(20.)/100
        valid=np.arange(20)<10
        result=r.conditional_factors(price,amp,np.ones(20)*.01,np.ones(20),valid)
        self.assertAlmostEqual(result[-1,0],amp[8:10].mean()-amp[:2].mean())
        valid[9]=False
        result=r.conditional_factors(price,amp,np.ones(20)*.01,np.ones(20),valid)
        self.assertTrue(np.isnan(result[-1,0]))

    def test_ideal_effective_days_exclude_halts_and_one_price_limits(self):
        q=pd.DataFrame({'HighPrice':[11.,9.,10.,11.,11.], 'LowPrice':[11.,9.,10.,10.,10.],
            'ClosePrice':[11.,9.,10.,10.5,10.5], 'up_limit':[11.]*5,'down_limit':[9.]*5,
            'Volume':[100.]*5,'PTFlag':[0]*5,'SuspendAny':[False,False,False,False,True]})
        self.assertEqual(r.ideal_valid_days(q).tolist(),[False,False,True,True,False])

    def test_ideal_effective_mask_does_not_change_long_momentum_definition(self):
        price=np.ones(160);amp=np.arange(160.)/100;ret=np.arange(160.)/10000
        a=r.conditional_factors(price,amp,ret,np.ones(160))
        b=r.conditional_factors(price,amp,ret,np.ones(160),np.zeros(160,dtype=bool))
        self.assertAlmostEqual(a[-1,2],b[-1,2])
        self.assertTrue(np.isnan(b[-1,0]))

    def test_ideal_reversal_conditions_on_average_trade_amount(self):
        price=np.ones(20)
        ret=np.arange(20.)/1000
        avg=np.arange(20.)[::-1]+1
        result=r.conditional_factors(price,np.ones(20)*.1,ret,avg)
        self.assertAlmostEqual(result[-1,1],ret[:10].sum()-ret[-10:].sum())

    def test_long_momentum_selects_low_amplitude_seventy_percent(self):
        price=np.ones(160)
        amp=np.arange(160.)[::-1]
        ret=np.arange(160.)/10000
        result=r.conditional_factors(price,amp,ret,np.ones(160))
        self.assertAlmostEqual(result[-1,2],ret[-112:].sum())
        self.assertTrue(np.isnan(result[158,2]))

    def test_zero_variance_correlation_neutral_not_entire_stock_missing(self):
        self.assertEqual(r.correlation(np.ones(20),np.arange(20.)),0)
        self.assertEqual(r.correlation(np.ones(2),np.ones(2)),0)
        price=np.ones(60)*10
        values,*_=r.day_metrics(price,price,price,price,np.ones(60),np.ones(60)*10,np.ones(60),1.)
        self.assertTrue(np.isfinite(values[:7]).all())
        self.assertTrue((values[:7]==0).all())

    def synthetic(self):
        rng=np.random.default_rng(123)
        n=80;day=pd.Timestamp('2017-03-31')
        keys=pd.DataFrame({'TradingDay':day,'SecuCode':[f'{i:06d}' for i in range(n)]})
        rolling=keys.copy()
        for name in list(r.DAY_FIELDS)+['PVStd','PVTrend','SmartPooled','IdealAmp','IdealRev','LowAmpMom']:
            rolling[name]=rng.normal(size=n)
        daily=keys.copy()
        for c in base.CONTROLS: daily[c]=rng.normal(size=n)
        cap=keys.assign(LnCap=rng.normal(size=n))
        return rolling,keys,daily,cap

    def test_cdpdp_standardizes_subfactors_before_combining(self):
        rolling,keys,daily,cap=self.synthetic()
        a=r.preprocess(rolling,keys,daily,cap)
        rolling.PP*=100
        b=r.preprocess(rolling,keys,daily,cap)
        np.testing.assert_allclose(a.HF_CDPDP,b.HF_CDPDP,atol=1e-10)

    def test_cpv_final_factor_contains_variation_and_trend(self):
        rolling,keys,daily,cap=self.synthetic()
        a=r.preprocess(rolling,keys,daily,cap)
        rolling.PVTrend=rolling.PVTrend[::-1].to_numpy()
        b=r.preprocess(rolling,keys,daily,cap)
        self.assertLess(a.HF_CPV.corr(b.HF_CPV),.99)
        self.assertAlmostEqual(a.HF_CDPDP.corr(b.HF_CDPDP),1.)

    def test_calendar_month_uses_true_variable_holding_length(self):
        dates=pd.bdate_range('2016-02-01','2016-06-03')
        symbols=pd.Index([f'{i:06d}' for i in range(20)])
        prices=np.ones((len(dates),20));flag=np.ones_like(prices,bool)
        market=base.Market(dates,symbols,prices,prices,flag,flag,~flag)
        schedule=r.month_schedule(dates)
        self.assertEqual(schedule[0],pd.Timestamp('2016-02-29'))
        features=pd.DataFrame([{'TradingDay':d,'SecuCode':s,'f':float(i)} for d in schedule for i,s in enumerate(symbols)])
        groups=r.scheduled_groups(features,'f',schedule)
        ic=r.calendar_ic(groups,market,'f',schedule)
        self.assertEqual(ic.EntryDate.iloc[0],pd.Timestamp('2016-03-01'))
        self.assertEqual(ic.ExitDate.iloc[0],pd.Timestamp('2016-04-01'))
        self.assertEqual(ic.HoldingDays.iloc[0],23)

    def test_leverage_convention_is_explicit(self):
        allowed=np.ones(2,bool)
        book1=base.Portfolio(pd.Index(['a','b']),base.Config(one_way_fee=0,ls_leg_weight=.5),'TheoreticalLS')
        book2=base.Portfolio(pd.Index(['a','b']),base.Config(one_way_fee=0,ls_leg_weight=1.),'TheoreticalLS')
        book1.execute(np.ones(2),allowed,allowed,'entry',np.array([.5,-.5]))
        book2.execute(np.ones(2),allowed,allowed,'entry',np.array([1.,-1.]))
        self.assertAlmostEqual(book2.value(np.array([1.1,1]))-1,2*(book1.value(np.array([1.1,1]))-1))

    def test_long_and_short_both_pay_actual_fees(self):
        cfg=base.Config(one_way_fee=.0015,ls_leg_weight=1.)
        book=base.Portfolio(pd.Index(['a','b']),cfg,'TheoreticalLS')
        allowed=np.ones(2,bool)
        stats=book.execute(np.ones(2),allowed,allowed,'entry',np.array([1.,-1.]))
        self.assertAlmostEqual(stats['TradingCost'],.003/(1+.003))
        self.assertAlmostEqual(book.value(np.ones(2)),1/(1+.003))
        stats=book.execute(np.ones(2),allowed,allowed,'exit',np.zeros(2))
        self.assertGreater(stats['TradingCost'],0)
        self.assertLess(book.value(np.ones(2)),1)

    def test_icir_weight_window_uses_only_realized_twelve_periods(self):
        dates=pd.bdate_range('2016-01-04',periods=100)
        symbols=pd.Index([f'{i:06d}' for i in range(20)])
        t=np.arange(100)[:,None];s=np.arange(20)[None,:]
        prices=10+np.sin(t/8+s/3)+.005*t*s
        flags=np.ones_like(prices,bool)
        market=base.Market(dates,symbols,prices.copy(),prices.copy(),flags,flags,~flags)
        rng=np.random.default_rng(88)
        features=pd.DataFrame([{'TradingDay':date,'SecuCode':symbol,
            **{f:float(rng.normal()) for f in r.PRIMARY}} for date in dates for symbol in symbols])
        with tempfile.TemporaryDirectory() as folder:
            result=r.icir_scores(features,market,5,folder,str(dates[80].date()))
            w=pd.read_csv(__import__('pathlib').Path(folder)/'weights.csv',parse_dates=['TradingDay','MaxLabelEnd'])
            self.assertTrue(w.ICPeriods.eq(12).all())
            self.assertTrue(w.MaxLabelEnd.lt(w.TradingDay).all())
            self.assertTrue(result[result.TradingDay.eq(dates[80])].ICIR_12.notna().all())

    def test_future_prices_cannot_change_first_icir_prediction(self):
        dates=pd.bdate_range('2016-01-04',periods=100)
        symbols=pd.Index([f'{i:06d}' for i in range(20)])
        t=np.arange(100)[:,None];s=np.arange(20)[None,:]
        prices=10+np.sin(t/8+s/3)+.005*t*s;flags=np.ones_like(prices,bool)
        market=base.Market(dates,symbols,prices.copy(),prices.copy(),flags,flags,~flags)
        rng=np.random.default_rng(88)
        features=pd.DataFrame([{'TradingDay':date,'SecuCode':symbol,
            **{f:float(rng.normal()) for f in r.PRIMARY}} for date in dates for symbol in symbols])
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            before=r.icir_scores(features,market,5,a,str(dates[80].date()))
            market.open[80:]*=np.linspace(1,3,20)
            after=r.icir_scores(features,market,5,b,str(dates[80].date()))
            np.testing.assert_allclose(before[before.TradingDay.eq(dates[80])].ICIR_12,
                                       after[after.TradingDay.eq(dates[80])].ICIR_12)


if __name__=='__main__': unittest.main()
