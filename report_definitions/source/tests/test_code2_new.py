import unittest

import numpy as np
import pandas as pd

import code2_new as m


class CorrectedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = m.Config(one_way_fee=0)

    def raw_quotes(self, dates, prices, reference=None):
        reference = reference if reference is not None else np.r_[prices[0], prices[:-1]]
        return pd.DataFrame({"TradingDay": dates, "SecuCode": "000001", "OpenPrice": prices,
                             "ClosePrice": prices, "PrevClosePrice": reference, "PTFlag": 0,
                             "Volume": 1000, "NonRestrictedShares": 1000000, "AFloats": 1000000})

    def market(self, periods=12, stocks=20):
        dates = pd.bdate_range("2017-01-02", periods=periods)
        symbols = pd.Index([f"{i:06d}" for i in range(stocks)])
        prices = np.ones((periods, stocks))
        allowed = np.ones_like(prices, dtype=bool)
        return m.Market(dates, symbols, prices.copy(), prices.copy(), allowed.copy(), allowed.copy(), ~allowed)

    def features(self, market):
        return pd.DataFrame([
            {"TradingDay": day, "SecuCode": symbol, "factor": float(i)}
            for day in market.dates for i, symbol in enumerate(market.symbols)
        ])

    def test_ex_rights_continuous_prices_and_no_future_feedback(self):
        dates = pd.bdate_range("2017-01-02", periods=4)
        raw = self.raw_quotes(dates, [100., 50., 55., 60.], [100., 50., 50., 55.])
        daily = m.prepare_daily(raw, dates, self.cfg)
        np.testing.assert_allclose(daily.AdjClose, [100, 100, 110, 120])
        np.testing.assert_allclose(daily.AdjOpen, [100, 100, 110, 120])
        modified = raw.copy()
        modified.loc[3, "ClosePrice"] = 80
        changed = m.prepare_daily(modified, dates, self.cfg)
        np.testing.assert_allclose(daily.AdjClose[:3], changed.AdjClose[:3])
        self.assertTrue(daily.CanBuy.iloc[1])  # 除权后的参考昨收是50，不是上一日100。

    def test_suspension_keeps_value_and_does_not_become_minus_100_percent(self):
        dates = pd.bdate_range("2017-01-02", periods=4)
        raw = self.raw_quotes(dates, [10., 0., 0., 11.], [10., 10., 10., 10.])
        raw.loc[[1, 2], "PTFlag"] = 1
        raw.loc[[1, 2], "Volume"] = 0
        daily = m.prepare_daily(raw, dates, self.cfg)
        np.testing.assert_allclose(daily.AdjClose, [10, 10, 10, 11])
        self.assertTrue(daily.AdjOpen.iloc[1:3].isna().all())
        self.assertFalse(daily.CanSell.iloc[1])
        self.assertFalse(daily.StaleQuote.iloc[1])  # 已知停牌与未知行情分开记录。

    def test_unknown_quote_is_marked_stale(self):
        dates = pd.bdate_range("2017-01-02", periods=4)
        raw = self.raw_quotes(dates, [10., 10., 10., 10.]).drop(index=3)
        daily = m.prepare_daily(raw, dates, self.cfg)
        self.assertTrue(daily.StaleQuote.iloc[3])
        self.assertEqual(daily.AdjClose.iloc[3], 10)
        self.assertFalse(daily.CanBuy.iloc[3])

    def test_unexplained_interior_gap_requires_data_repair(self):
        dates = pd.bdate_range("2017-01-02", periods=4)
        raw = self.raw_quotes(dates, [10., 10., 10., 10.]).drop(index=1)
        with self.assertRaisesRegex(ValueError, "未解释缺口"):
            m.prepare_daily(raw, dates, self.cfg)

    def test_execution_does_not_use_full_day_volume(self):
        dates = pd.bdate_range("2017-01-02", periods=4)
        raw = self.raw_quotes(dates, [10., 10., 10., 10.])
        baseline = m.prepare_daily(raw, dates, self.cfg)
        raw.loc[1, "Volume"] = 0
        changed = m.prepare_daily(raw, dates, self.cfg)
        self.assertEqual(baseline.CanBuy.iloc[1], changed.CanBuy.iloc[1])
        self.assertEqual(baseline.CanSell.iloc[1], changed.CanSell.iloc[1])

    def test_style_windows_use_complete_history(self):
        dates = pd.bdate_range("2017-01-02", periods=25)
        raw = self.raw_quotes(dates, np.linspace(10, 12.4, 25))
        daily = m.prepare_daily(raw, dates, self.cfg)
        self.assertAlmostEqual(daily.Ctrl_Ret20.iloc[20], 0.2)
        raw = raw.drop(index=10)
        with self.assertRaisesRegex(ValueError, "未解释缺口"):
            m.prepare_daily(raw, dates, self.cfg)

    def test_intraday_smart_money_uses_real_amount(self):
        n = 60
        close = 10 + np.sin(np.arange(n)) * 0.1
        volume = 100 + np.arange(n, dtype=float)
        amount = volume * (10 + np.cos(np.arange(n)) * 0.3)
        values = m.intraday_metrics(close, close, close + 0.01, close - 0.01, volume, amount)
        ret = np.r_[0, close[1:] / close[:-1] - 1]
        order = np.argsort(np.abs(ret) / volume ** 0.25)[::-1]
        target = volume.sum() * 0.2
        remaining, smart_amount = target, 0.
        for i in order:
            used = min(volume[i], remaining)
            smart_amount += amount[i] * used / volume[i]
            remaining -= used
            if remaining <= 1e-10:
                break
        expected = (smart_amount / target) / (amount.sum() / volume.sum())
        self.assertAlmostEqual(values[0], expected)
        self.assertAlmostEqual(values[4], close[-1] / close[-31] - 1)

    def test_rolling_missing_day_does_not_compress_window(self):
        dates = pd.bdate_range("2017-01-02", periods=25)
        raw = pd.DataFrame({"TradingDay": dates, "SecuCode": "000001", **{f: 1. for f in m.FACTORS}})
        raw = raw.drop(index=10)
        rolled = m.roll_factors(raw, dates, self.cfg)
        self.assertTrue(rolled.HF_Amplitude.iloc[20:].isna().all())

    def test_fixed_schedule_and_low_value_long_direction(self):
        market = self.market()
        features = self.features(market)
        # 缺失预定信号日，不应把下一天替换成调仓日。
        features = features[~features.TradingDay.eq(market.dates[5])]
        groups, schedule = m.signal_groups(features, "factor", market.dates, 5)
        self.assertEqual(list(schedule), list(market.dates[::5]))
        self.assertNotIn(market.dates[6], set(groups.TradingDay))
        first = groups[groups.TradingDay.eq(market.dates[0])]
        self.assertEqual(set(first[first.Group.eq(5)].SecuCode), set(market.symbols[:4]))

    def test_ic_uses_matching_n_day_open_horizon(self):
        market = self.market(periods=25)
        for t in range(len(market.dates)):
            market.open[t] = 1 + (t / 100) * np.arange(1, 21)
        features = self.features(market)
        for days in (5, 10, 20):
            groups, _ = m.signal_groups(features, "factor", market.dates, days)
            result = m.calc_ic(groups, market, "factor", days)
            self.assertEqual(result.ExitDate.iloc[0], market.dates[days + 1])
            self.assertEqual(result.EntryDate.iloc[0], market.dates[1])
            expected_returns = market.open[days + 1] / market.open[1] - 1
            expected = pd.Series(np.arange(20.)).corr(pd.Series(expected_returns))
            self.assertAlmostEqual(result.IC_Raw.iloc[0], expected)

    def test_future_missing_price_changes_ic_coverage_not_signal_groups(self):
        market = self.market()
        features = self.features(market)
        before, _ = m.signal_groups(features, "factor", market.dates, 5)
        market.open[6, 0] = np.nan
        after, _ = m.signal_groups(features, "factor", market.dates, 5)
        pd.testing.assert_frame_equal(before, after)
        result = m.calc_ic(after, market, "factor", 5)
        self.assertEqual(result.N_Signal.iloc[0], 20)
        self.assertEqual(result.N_Paired.iloc[0], 19)

    def test_initial_and_actual_turnover_costs(self):
        cfg = m.Config(one_way_fee=0.0015)
        book = m.Portfolio(pd.Index(["a", "b"]), cfg, "test")
        all_ = np.ones(2, dtype=bool)
        initial = book.execute(np.ones(2), all_, all_, "day1", np.array([0.5, 0.5]))
        self.assertGreater(initial["TradingCost"], 0)
        self.assertAlmostEqual(book.value(np.ones(2)), 1 / 1.0015)
        drift = book.execute(np.array([1.2, 0.8]), all_, all_, "day2", np.array([0.5, 0.5]))
        self.assertGreater(drift["TradingCost"], 0)  # 集合不变，权重漂移仍有交易。
        self.assertGreaterEqual(book.cash, -1e-12)

    def test_rejected_buy_keeps_cash_without_reranking(self):
        book = m.Portfolio(pd.Index(["a", "b"]), self.cfg, "test")
        book.execute(np.ones(2), np.array([False, True]), np.ones(2, bool), "day1", np.array([0.5, 0.5]))
        np.testing.assert_allclose(book.units, [0, 0.5])
        self.assertAlmostEqual(book.cash, 0.5)

    def test_blocked_exit_keeps_position_and_retries(self):
        book = m.Portfolio(pd.Index(["a"]), self.cfg, "test")
        yes, no = np.array([True]), np.array([False])
        book.execute(np.array([1.]), yes, yes, "day1", np.array([1.]))
        book.execute(np.array([1.]), no, no, "day2", np.array([0.]))
        self.assertEqual(book.units[0], 1)
        self.assertEqual(book.pending[0], 0)
        book.execute(np.array([1.2]), yes, yes, "day3")
        self.assertEqual(book.units[0], 0)
        self.assertAlmostEqual(book.cash, 1.2)

    def test_future_rank_changes_cannot_change_past_fees_or_nav(self):
        market = self.market()
        features = self.features(market)
        groups, schedule = m.signal_groups(features, "factor", market.dates, 3)
        baseline, _ = m.backtest(groups, schedule, market, m.Config())
        features.loc[features.TradingDay.eq(market.dates[9]), "factor"] *= -1
        changed, schedule = m.signal_groups(features, "factor", market.dates, 3)
        modified, _ = m.backtest(changed, schedule, market, m.Config())
        pd.testing.assert_frame_equal(baseline[baseline.TradingDay < market.dates[10]],
                                      modified[modified.TradingDay < market.dates[10]])

    def test_non_rebalance_signals_do_not_change_nav(self):
        market = self.market()
        features = self.features(market)
        groups, schedule = m.signal_groups(features, "factor", market.dates, 5)
        baseline, _ = m.backtest(groups, schedule, market, self.cfg)
        features.loc[~features.TradingDay.isin(schedule), "factor"] *= -1
        changed, schedule = m.signal_groups(features, "factor", market.dates, 5)
        result, _ = m.backtest(changed, schedule, market, self.cfg)
        pd.testing.assert_frame_equal(baseline, result)

    def test_final_blocked_positions_remain_reported(self):
        market = self.market()
        market.sell[-1] = False
        groups, schedule = m.signal_groups(self.features(market), "factor", market.dates, 5)
        ledger, _ = m.backtest(groups, schedule, market, self.cfg)
        final = ledger[(ledger.TradingDay == market.dates[-1]) & (ledger.Portfolio == "Group5")].iloc[0]
        self.assertEqual(final.Positions, 4)
        self.assertEqual(final.PendingReductions, 4)
        self.assertAlmostEqual(final.NAV, 1)

    def test_ledger_cash_positions_and_trades_reconcile(self):
        market = self.market()
        market.close[1:, :4] *= 1.05
        market.open[2:, :4] *= 1.05
        cfg = m.Config()
        groups, schedule = m.signal_groups(self.features(market), "factor", market.dates, 5)
        ledger, trades = m.backtest(groups, schedule, market, cfg)
        for name, frame in ledger.groupby("Portfolio"):
            np.testing.assert_allclose(frame.NAV, frame.Cash + frame.LongValue - frame.ShortValue)
            metrics = m.performance(frame.Return, cfg)
            self.assertAlmostEqual(metrics["FinalNAV"], frame.NAV.iloc[-1])
            self.assertAlmostEqual(frame.TradingCost.sum(), trades.loc[trades.Portfolio.eq(name), "Cost"].sum())

    def test_drawdown_includes_initial_capital_and_daily_path(self):
        metrics = m.performance([-0.1, 0.01, 0.01], self.cfg)
        self.assertAlmostEqual(metrics["MaxDrawdown"], -0.1)
        metrics = m.performance([-0.2, 0.25], self.cfg)
        self.assertAlmostEqual(metrics["TotalReturn"], 0)
        self.assertAlmostEqual(metrics["MaxDrawdown"], -0.2)

    def test_geometric_annual_return_and_arithmetic_sharpe(self):
        returns = np.array([-0.1, 0.1, 0.02])
        metrics = m.performance(returns, self.cfg)
        expected = np.prod(1 + returns) ** (250 / 3) - 1
        self.assertAlmostEqual(metrics["AnnualReturn"], expected)
        self.assertAlmostEqual(metrics["Sharpe"], returns.mean() / returns.std(ddof=1) * np.sqrt(250))


if __name__ == "__main__":
    unittest.main()
