"""Check code2.py in isolation without running its pipeline or changing caches.

Run from the project root:
  /opt/homebrew/Caskroom/miniconda/base/envs/quant/bin/python audit/check_code2.py
"""

import ast
import contextlib
import hashlib
import io
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "code2.py"


def load_analyzer():
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FactorAnalyzer")
    namespace = {"pd": pd, "np": np, "Config": SimpleNamespace(COST_RATE_TOTAL=0.003)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["FactorAnalyzer"]


def analyze(analyzer, panel, ascending=True):
    """Observe intermediate original-function results without changing its code."""
    function = analyzer.analyze_single_period
    observed = {}

    def trace(frame, event, arg):
        if frame.f_code is function.__code__:
            if event == "return":
                observed.update(frame.f_locals)
            return trace
        return None

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        result = function(panel, "factor", "ret_5", "Weekly", 5, ascending)
    finally:
        sys.settrace(previous)
    assert result is not None
    return result, observed


def run():
    analyzer = load_analyzer()
    results = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()}

    dates = pd.bdate_range("2017-03-01", periods=25)
    prices = np.arange(100.0, 125.0)
    quotes = pd.DataFrame({"TradingDay": dates, "SecuCode": "000001", "OpenPrice": prices})
    forward = analyzer.calc_forward_returns(quotes, {"Weekly": 5, "BiWeekly": 10, "Monthly": 20})
    results["holding_horizons"] = {}
    for name, days in [("Weekly", 5), ("BiWeekly", 10), ("Monthly", 20)]:
        actual = forward[name].iloc[0][f"Ret_{name}"]
        expected = prices[days + 1] / prices[1] - 1
        assert np.isclose(actual, expected)
        results["holding_horizons"][name] = {"days": days, "return": actual}

    dates = pd.bdate_range("2017-03-01", periods=11)
    panel = pd.DataFrame([
        {"TradingDay": date, "SecuCode": f"{i:06d}", "factor": float(i), "ret_5": 0.02 if i >= 16 else 0.0}
        for date in dates for i in range(20)
    ])
    baseline, baseline_locals = analyze(analyzer, panel)
    changed = panel.copy()
    changed.loc[~changed.TradingDay.isin(dates[::5]) & (changed.factor >= 16), "ret_5"] = 0.5
    modified, modified_locals = analyze(analyzer, changed)
    assert baseline["cli_msg"] == modified["cli_msg"]
    assert baseline["metrics_str"] == modified["metrics_str"]
    assert not np.isclose(baseline["ls_nav"].iloc[-1], modified["ls_nav"].iloc[-1])
    results["plot_statistics_disagreement"] = {
        "change": "only non-rebalance signal forward returns changed",
        "all_reported_metrics_unchanged": True,
        "plot_final_nav_before": baseline["ls_nav"].iloc[-1],
        "plot_final_nav_after": modified["ls_nav"].iloc[-1],
        "period_ledger_final_nav": (1 + baseline_locals["ls_ret_N_net"]).prod(),
    }

    loss_panel = panel.copy()
    loss_panel.loc[loss_panel.factor >= 16, "ret_5"] = 0.01
    loss_panel.loc[(loss_panel.TradingDay == dates[0]) & (loss_panel.factor >= 16), "ret_5"] = -0.1
    _, observed = analyze(analyzer, loss_panel)
    nav = np.r_[1.0, (1 + observed["ls_ret_N_net"]).cumprod().to_numpy()]
    correct_dd = np.min(nav / np.maximum.accumulate(nav) - 1)
    assert np.isclose(observed["max_dd"], 0)
    assert np.isclose(correct_dd, -0.1)
    results["initial_drawdown_omitted"] = {
        "reported_max_drawdown": observed["max_dd"],
        "correct_endpoint_drawdown_including_initial_capital": correct_dd,
    }

    _, before = analyze(analyzer, panel, ascending=False)
    missing = panel.copy()
    missing.loc[(missing.TradingDay == dates[0]) & (missing.SecuCode == "000000"), "ret_5"] = np.nan
    _, after = analyze(analyzer, missing, ascending=False)
    long_before = sorted(before["daily_long_stocks"].iloc[0])
    long_after = sorted(after["daily_long_stocks"].iloc[0])
    assert long_before == ["000000", "000001", "000002", "000003"]
    assert long_after == ["000001", "000002", "000003", "000004"]
    zero_quotes = pd.DataFrame({"TradingDay": dates[:8], "SecuCode": "000000", "OpenPrice": 10.0})
    zero_quotes.loc[zero_quotes.TradingDay == dates[6], "OpenPrice"] = 0
    zero_forward = analyzer.calc_forward_returns(zero_quotes, {"Weekly": 5})["Weekly"]
    assert dates[0] not in set(zero_forward.TradingDay)
    results["future_missing_return_changes_signal_portfolio"] = {
        "signal_day_factors_identical": True,
        "long_before": long_before,
        "long_after_only_future_return_becomes_missing": long_after,
        "zero_exit_open_removes_signal_day_return": True,
    }

    future_ranks = panel.copy()
    final_day = future_ranks.TradingDay == dates[-1]
    future_ranks.loc[final_day, "factor"] = 19 - future_ranks.loc[final_day, "factor"]
    _, future_locals = analyze(analyzer, future_ranks)
    assert np.isclose(baseline_locals["ls_ret_N_resampled"].iloc[0], future_locals["ls_ret_N_resampled"].iloc[0])
    assert np.isclose(baseline_locals["avg_turnover"], 0)
    assert np.isclose(future_locals["avg_turnover"], 0.5)
    results["future_turnover_changes_past_cost"] = {
        "change": "only final rebalance factor rankings reversed",
        "first_period_gross_before": baseline_locals["ls_ret_N_resampled"].iloc[0],
        "first_period_gross_after": future_locals["ls_ret_N_resampled"].iloc[0],
        "first_period_net_before": baseline_locals["ls_ret_N_net"].iloc[0],
        "first_period_net_after": future_locals["ls_ret_N_net"].iloc[0],
        "unchanged_constituents_initial_transaction_cost": baseline_locals["cost_per_period"] * baseline_locals["avg_turnover"],
    }

    volatile = panel.copy()
    period_returns = [-0.1, 0.1, 0.02]
    for date, ret in zip(dates[::5], period_returns):
        volatile.loc[(volatile.TradingDay == date) & (volatile.factor >= 16), "ret_5"] = ret
    _, observed = analyze(analyzer, volatile)
    net = observed["ls_ret_N_net"]
    geometric = (1 + net).prod() ** (50 / len(net)) - 1
    arithmetic_sharpe = net.mean() / net.std() * np.sqrt(50)
    assert not np.isclose(observed["ann_ls_ret"], geometric)
    assert not np.isclose(observed["ls_sharpe"], arithmetic_sharpe)
    results["annualization_definitions"] = {
        "period_returns": period_returns,
        "reported_annual_return": observed["ann_ls_ret"],
        "geometric_annual_return_same_250_day_convention": geometric,
        "reported_return_volatility_ratio": observed["ls_sharpe"],
        "arithmetic_sharpe_zero_risk_free": arithmetic_sharpe,
    }

    columns = ["SecuCode", "TradingDay", "OpenPrice", "ClosePrice", "PrevClosePrice", "ret"]
    daily = pd.read_csv(ROOT / "data/日行情/daily_quote_2017.csv", usecols=columns, dtype={"SecuCode": str})
    stock = daily.loc[daily.SecuCode == "603528"].sort_values("TradingDay").copy()
    stock["TradingDay"] = pd.to_datetime(stock.TradingDay)
    stock["prior_actual_close"] = stock.ClosePrice.shift()
    event = stock.loc[stock.TradingDay == "2017-06-07"].iloc[0]
    raw_ret = event.ClosePrice / event.prior_actual_close - 1
    assert raw_ret < -0.6 and event.ret > 0.1
    actual_forward = analyzer.calc_forward_returns(stock, {"Weekly": 5})["Weekly"]
    example = actual_forward.loc[actual_forward.TradingDay == "2017-06-05"].iloc[0]
    results["unadjusted_real_prices"] = {
        "symbol": "603528", "event_date": "2017-06-07",
        "prior_actual_close": event.prior_actual_close,
        "ex_rights_reference_previous_close": event.PrevClosePrice,
        "close": event.ClosePrice,
        "raw_close_return": raw_ret,
        "provided_daily_return": event.ret,
        "original_function_weekly_return_signal_2017_06_05": example.Ret_Weekly,
    }
    return results


if __name__ == "__main__":
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        results = run()
    text = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    (ROOT / "audit/code2_check_results.json").write_text(text)
    print(text, end="")
