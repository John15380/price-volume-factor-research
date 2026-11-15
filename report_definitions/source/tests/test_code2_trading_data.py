import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import code2_new as base
import code2_trading_data as trading


class HistoricalTradingDataTests(unittest.TestCase):
    def setUp(self):
        self.dates=pd.bdate_range('2017-03-01',periods=4)
        self.all_dates=pd.bdate_range('2016-01-01','2019-01-01')
        self.raw=pd.DataFrame({'TradingDay':self.dates,'SecuCode':'000001','OpenPrice':10.,'ClosePrice':10.,
            'PrevClosePrice':10.,'HighPrice':10.1,'LowPrice':9.9,'PTFlag':0,'Volume':1000.,'Amount':10000.,
            'NonRestrictedShares':1000000.,'AFloats':1000000.})
        self.st=pd.DataFrame(columns=base.KEYS+['IsST'])
        self.suspend=pd.DataFrame(columns=base.KEYS+['SuspendAtOpen','SuspendAny','SuspendFullDay'])
        self.limits=self.raw[base.KEYS].assign(up_limit=11.,down_limit=9.)
        self.factors=self.raw[base.KEYS].assign(adj_factor=1.)
        self.temp=tempfile.TemporaryDirectory()
        self.cfg=base.Config(output_dir=Path(self.temp.name),one_way_fee=0)

    def tearDown(self):self.temp.cleanup()

    def prepare(self):
        symbols=trading.canonical(self.raw.SecuCode).unique()
        seen={symbol:pd.Timestamp('2016-01-01') for symbol in symbols}
        return trading.prepare_daily(self.raw,self.dates,self.cfg,seen,self.all_dates,
                                     (self.st,self.suspend,self.limits,self.factors))

    def test_actual_upper_limit_blocks_buys_allows_sells(self):
        self.raw.loc[0,'OpenPrice']=11.
        day=self.prepare().iloc[0]
        self.assertFalse(day.CanBuy)
        self.assertTrue(day.CanSell)

    def test_actual_lower_limit_blocks_sells_allows_buys(self):
        self.raw.loc[0,'OpenPrice']=9.
        day=self.prepare().iloc[0]
        self.assertTrue(day.CanBuy)
        self.assertFalse(day.CanSell)

    def test_intraday_suspend_timing_is_evaluated_at_open(self):
        self.assertFalse(trading.suspension_at_open('13:00-15:00'))
        self.assertFalse(trading.suspension_at_open('09:31-10:01'))
        self.assertTrue(trading.suspension_at_open('09:30-10:00'))
        self.assertTrue(trading.suspension_at_open('09:30-10:31,10:31-14:57'))
        self.assertTrue(trading.suspension_at_open(np.nan))

    def test_afternoon_halt_does_not_block_morning_trade(self):
        self.suspend=pd.DataFrame([{'TradingDay':self.dates[0],'SecuCode':'000001',
            'SuspendAtOpen':False,'SuspendAny':True,'SuspendFullDay':False}])
        day=self.prepare().iloc[0]
        self.assertTrue(day.CanBuy)
        self.assertTrue(day.CanSell)
        self.assertFalse(day.SignalEligible)

    def test_open_halt_does_not_use_later_opening_price(self):
        self.suspend=pd.DataFrame([{'TradingDay':self.dates[1],'SecuCode':'000001',
            'SuspendAtOpen':True,'SuspendAny':True,'SuspendFullDay':False}])
        self.raw.loc[1,'OpenPrice']=10.5
        day=self.prepare().iloc[1]
        self.assertTrue(np.isnan(day.AdjOpen))
        self.assertFalse(day.CanBuy)
        self.assertFalse(day.CanSell)

    def test_st_excluded_from_signal_and_new_risk_not_exits(self):
        self.st=pd.DataFrame([{'TradingDay':self.dates[0],'SecuCode':'000001','IsST':True}])
        day=self.prepare().iloc[0]
        self.assertFalse(day.SignalEligible)
        self.assertFalse(day.CanIncrease)
        self.assertTrue(day.CanBuy)
        self.assertTrue(day.CanSell)

    def test_st_long_reduction_is_not_blocked_by_admission_rule(self):
        book=base.Portfolio(pd.Index(['000001']),self.cfg,'Group5')
        yes=np.array([True]);no=np.array([False])
        book.execute(np.array([10.]),yes,yes,self.dates[0],np.array([1.]),yes)
        book.execute(np.array([10.]),yes,yes,self.dates[1],np.array([0.]),no)
        self.assertEqual(book.units[0],0)
        self.assertAlmostEqual(book.cash,1.)

    def test_st_short_cover_is_not_blocked_by_admission_rule(self):
        book=base.Portfolio(pd.Index(['000001']),self.cfg,'TheoreticalLS')
        yes=np.array([True]);no=np.array([False])
        book.execute(np.array([10.]),yes,yes,self.dates[0],np.array([-1.]),yes)
        book.execute(np.array([10.]),yes,yes,self.dates[1],np.array([0.]),no)
        self.assertEqual(book.units[0],0)
        self.assertAlmostEqual(book.cash,1.)

    def test_future_st_or_limits_do_not_remove_current_signal(self):
        before=self.prepare()
        self.st=pd.DataFrame([{'TradingDay':self.dates[1],'SecuCode':'000001','IsST':True}])
        self.limits.loc[1,'up_limit']=10.
        after=self.prepare()
        self.assertEqual(before.SignalEligible.iloc[0],after.SignalEligible.iloc[0])
        self.assertEqual(before.CanBuy.iloc[0],after.CanBuy.iloc[0])
        self.assertFalse(after.CanIncrease.iloc[1])
        self.assertFalse(after.CanBuy.iloc[1])

    def test_official_adjustment_and_future_invariance(self):
        self.raw.loc[1:,'OpenPrice']=5.
        self.raw.loc[1:,'ClosePrice']=5.
        self.factors.loc[1:,'adj_factor']=2.
        before=self.prepare()
        np.testing.assert_allclose(before.AdjClose,[10,10,10,10])
        self.factors.loc[3,'adj_factor']=3.
        after=self.prepare()
        np.testing.assert_allclose(before.AdjClose.iloc[:3],after.AdjClose.iloc[:3])

    def test_unknown_limit_does_not_assume_unlimited_trading(self):
        self.limits=self.limits.drop(index=1)
        day=self.prepare().iloc[1]
        self.assertFalse(day.CanBuy)
        self.assertFalse(day.CanSell)
        self.assertTrue(day.SignalEligible)

    def test_conflicting_open_and_limit_bounds_block_orders_not_signal(self):
        self.raw.loc[0,'OpenPrice']=12.
        day=self.prepare().iloc[0]
        self.assertFalse(day.CanBuy)
        self.assertFalse(day.CanSell)
        self.assertTrue(day.SignalEligible)

    def test_missing_official_factor_fails_instead_of_dropping_stock(self):
        self.factors=self.factors.drop(index=1)
        with self.assertRaisesRegex(ValueError,'缺少官方复权因子'):
            self.prepare()

    def test_code_changes_have_one_continuous_security_identity(self):
        self.dates=pd.to_datetime(['2018-12-19','2018-12-20','2018-12-26','2018-12-27'])
        self.raw.TradingDay=self.dates
        self.raw.SecuCode=['000022','000022','001872','001872']
        self.limits=self.raw[base.KEYS].copy();self.limits.SecuCode='001872'
        self.limits=self.limits.assign(up_limit=11.,down_limit=9.)
        self.factors=self.raw[base.KEYS].copy();self.factors.SecuCode='001872'
        self.factors=self.factors.assign(adj_factor=1.)
        result=self.prepare()
        self.assertEqual(result.SecuCode.nunique(),1)
        self.assertEqual(result.SecuCode.iloc[0],'001872')
        self.assertEqual(result.TradingCode.tolist(),['000022','000022','001872','001872'])
        np.testing.assert_allclose(result.AdjClose,10)

    def test_duplicate_alias_values_cannot_silently_conflict(self):
        frame=pd.DataFrame({'TradingDay':[self.dates[0]]*2,'SecuCode':['000022','001872'],'value':[1.,2.]})
        frame.SecuCode=trading.canonical(frame.SecuCode)
        with self.assertRaisesRegex(ValueError,'冲突'):
            trading.deduplicate(frame,base.KEYS,['value'],'test')

    def test_new_st_at_execution_keeps_cash_without_reranking(self):
        dates=pd.bdate_range('2017-01-02',periods=8)
        symbols=pd.Index([f'{i:06d}' for i in range(20)])
        price=np.ones((8,20));yes=np.ones_like(price,bool);entry=yes.copy();entry[1,0]=False
        market=base.Market(dates,symbols,price,price,yes,yes,~yes,entry)
        features=pd.DataFrame([{'TradingDay':d,'SecuCode':s,'f':float(i)} for d in dates for i,s in enumerate(symbols)])
        groups,schedule=base.signal_groups(features,'f',dates,5)
        self.assertIn('000000',set(groups[groups.TradingDay.eq(dates[0])].SecuCode))
        ledger,trades=base.backtest(groups,schedule,market,self.cfg)
        first=ledger[(ledger.TradingDay==dates[1])&ledger.Portfolio.eq('Group5')].iloc[0]
        self.assertAlmostEqual(first.Cash,.25)
        self.assertEqual(first.Positions,3)
        self.assertEqual(first.AdmissionBlockedOrders,1)


if __name__=='__main__':unittest.main()
