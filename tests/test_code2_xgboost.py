import unittest

import numpy as np
import pandas as pd

import code2_new as base
import code2_xgboost as model


class ModelTimingTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2016-01-04", periods=100)
        self.symbols = pd.Index([f"{i:06d}" for i in range(20)])
        prices = 10 + np.arange(100)[:, None] * np.arange(1, 21)[None, :] / 100
        flags = np.ones_like(prices, dtype=bool)
        self.market = base.Market(self.dates, self.symbols, prices.copy(), prices.copy(), flags, flags, ~flags)
        self.features = pd.DataFrame([
            {"TradingDay": day, "SecuCode": symbol, **{factor: float(i) for factor in base.FACTORS}}
            for day in self.dates for i, symbol in enumerate(self.symbols)
        ])

    def test_label_horizons_and_centering(self):
        for days in (5, 10, 20):
            labels = model.make_labels(self.features, self.market, days)
            first = labels[labels.TradingDay.eq(self.dates[0])]
            np.testing.assert_allclose(first.ForwardReturn, self.market.open[days+1] / self.market.open[1] - 1)
            self.assertTrue(first.LabelEnd.eq(self.dates[days+1]).all())
            self.assertAlmostEqual(first.Target.mean(), 0)

    def test_purge_uses_realization_date(self):
        labels = model.make_labels(self.features, self.market, 20)
        cutoff = self.dates[60]
        selected = model.training_sample(labels, cutoff)
        self.assertTrue(selected.LabelEnd.lt(cutoff).all())
        self.assertEqual(selected.TradingDay.max(), self.dates[38])
        self.assertNotIn(self.dates[50], set(selected.TradingDay))

    def test_window_excludes_old_signal_dates(self):
        labels = model.make_labels(self.features, self.market, 5)
        cutoff = self.dates[-1]
        selected = model.training_sample(labels, cutoff, window_months=1)
        self.assertTrue(selected.TradingDay.ge(cutoff-pd.DateOffset(months=1)).all())

    def test_future_prices_do_not_change_available_training_labels(self):
        cutoff = self.dates[60]
        before = model.training_sample(model.make_labels(self.features, self.market, 20), cutoff)
        self.market.open[60:] *= np.linspace(2, 3, 20)
        after = model.training_sample(model.make_labels(self.features, self.market, 20), cutoff)
        pd.testing.assert_frame_equal(before, after)

    def test_missing_future_price_preserves_prediction_input_rows(self):
        self.market.open[21, 0] = np.nan
        labels = model.make_labels(self.features, self.market, 20)
        self.assertEqual(len(labels), len(self.features))
        first = labels[(labels.TradingDay == self.dates[0]) & (labels.SecuCode == self.symbols[0])]
        self.assertTrue(first.Target.isna().all())
        self.assertEqual(len(self.features[self.features.TradingDay.eq(self.dates[0])]), 20)

    def test_prediction_score_direction_is_high_value_long(self):
        features = self.features.rename(columns={base.FACTORS[0]: "Pred_Score"})
        groups, _ = base.signal_groups(features, "Pred_Score", self.dates, 5, low_is_long=False)
        first = groups[groups.TradingDay.eq(self.dates[0])]
        self.assertEqual(set(first[first.Group.eq(10)].SecuCode), set(self.symbols[-2:]))


if __name__ == "__main__":
    unittest.main()
