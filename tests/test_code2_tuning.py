import unittest
import numpy as np
import pandas as pd
import code2_new as base
import code2_research as r
import code2_tuning as t


class TunedModelTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(42)
        days=pd.bdate_range('2016-01-04',periods=220)
        data=[]
        for i,day in enumerate(days):
            for s in range(30):
                signal=float(rng.normal());noise=float(rng.normal())
                data.append({'TradingDay':day,'SecuCode':f'{s:06d}','Target':signal+.3*noise,
                             'a':signal,'duplicate':signal,'noise':noise,'LabelEnd':days[min(i+10,219)]})
        return pd.DataFrame(data)

    def test_correlated_pair_keeps_better_representative(self):
        f=self.data();stats,corr=t.selector_statistics(f,['a','duplicate','noise'])
        stats.loc[stats.Feature.eq('a'),'Quality']+=.01
        chosen,log=t.choose_features(stats,corr,.8,9)
        self.assertIn('a',chosen);self.assertNotIn('duplicate',chosen)
        self.assertEqual(log.loc[log.Feature.eq('duplicate'),'Representative'].iloc[0],'a')
        for a in chosen:
            for b in chosen:
                if a!=b:self.assertLess(abs(corr.loc[a,b]),.8)

    def test_purged_validation_excludes_unrealized_training_labels(self):
        f=self.data()
        for train,valid in t.purged_folds(f):
            self.assertLess(train.LabelEnd.max(),valid.TradingDay.min())
            self.assertLess(train.TradingDay.max(),valid.TradingDay.min())

    def test_future_data_cannot_change_training_selection(self):
        f=self.data();cutoff=sorted(f.TradingDay.unique())[160]
        train=f[f.LabelEnd<cutoff]
        a=t.choose_features(*t.selector_statistics(train,['a','duplicate','noise']))[0]
        f.loc[f.TradingDay>=cutoff,['a','Target']]*=-100
        b=t.choose_features(*t.selector_statistics(f[f.LabelEnd<cutoff],['a','duplicate','noise']))[0]
        self.assertEqual(a,b)

    def test_ten_groups_have_correct_size_and_best_direction(self):
        dates=pd.bdate_range('2017-01-02',periods=8);symbols=[f'{i:06d}' for i in range(103)]
        features=pd.DataFrame([{'TradingDay':d,'SecuCode':s,'f':i} for d in dates for i,s in enumerate(symbols)])
        groups,_=base.signal_groups(features,'f',dates,5,n_groups=10)
        first=groups[groups.TradingDay.eq(dates[0])]
        self.assertEqual(first.Group.nunique(),10)
        self.assertLessEqual(first.groupby('Group').size().max()-first.groupby('Group').size().min(),1)
        self.assertEqual(set(first[first.Group.eq(10)].SecuCode),set(symbols[:10]))

    def test_round_trip_is_three_per_thousand(self):
        cfg=base.Config();self.assertAlmostEqual(cfg.one_way_fee*2,.003)
        book=base.Portfolio(pd.Index(['stock']),cfg,'Group10');yes=np.ones(1,bool)
        book.execute(np.ones(1),yes,yes,'buy',np.ones(1))
        book.execute(np.ones(1),yes,yes,'sell',np.zeros(1))
        for trade in book.trades:self.assertAlmostEqual(trade['Cost']/trade['Notional'],.0015)
        self.assertAlmostEqual(sum(q['Cost'] for q in book.trades)/book.trades[0]['Notional'],.003)

    def test_removed_factors_cannot_enter_active_catalog(self):
        self.assertEqual(len(r.FACTORS),9)
        self.assertFalse(any('Tail' in f or 'Legacy' in f for f in r.FACTORS))

    def test_search_is_reproducible_and_covers_parameters(self):
        a=t.candidate_parameters(96,42);b=t.candidate_parameters(96,42)
        self.assertEqual(a,b)
        self.assertEqual(len(set(q['max_depth'] for q in a)),5)
        self.assertEqual(set(q['corr_threshold'] for q in a),{.7,.8,.9})


if __name__=='__main__':unittest.main()
