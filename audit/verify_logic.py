"""Reproduce review findings without running or changing the trading pipelines.

Run from the project root with the existing quant environment:
  /opt/homebrew/Caskroom/miniconda/base/envs/quant/bin/python audit/verify_logic.py

Only selected original definitions are compiled from their ASTs. This avoids
imports, cache deletion, model training, and plot generation in the pipelines.
"""

import ast
import contextlib
import io
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def original_class(filename, name):
    source = ROOT / filename
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = {
        "pd": pd,
        "np": np,
        "Config": SimpleNamespace(
            PATH_TRADING_DAYS=str(ROOT / "data/交易日期/tradingdays.csv"),
            COST_RATE_TOTAL=0.003,
        ),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name], namespace


def run():
    results = {}
    analyzer1, _ = original_class("code.py", "FactorAnalyzer")
    analyzer2, _ = original_class("code2.py", "FactorAnalyzer")
    loader, _ = original_class("code.py", "DataLoader")

    dates = pd.bdate_range("2017-03-01", periods=8)
    quotes = pd.DataFrame(
        {
            "TradingDay": dates,
            "SecuCode": "000001",
            "ClosePrice": [100, 101, 102, 103, 104, 120, 121, 122],
            "OpenPrice": [100, 200, 150, 130, 120, 110, 100, 90],
        }
    )
    first1 = analyzer1.calc_forward_returns(quotes, {"Weekly": 5})["Weekly"].iloc[0]
    first2 = analyzer2.calc_forward_returns(quotes, {"Weekly": 5})["Weekly"].iloc[0]
    correct = quotes.iloc[6].OpenPrice / quotes.iloc[1].OpenPrice - 1
    assert np.isclose(first1.Ret_Weekly, 0.2)
    assert np.isclose(first2.Ret_Weekly, correct)
    results["execution_timing"] = {
        "code_py_weekly_return": float(first1.Ret_Weekly),
        "code2_py_weekly_return": float(first2.Ret_Weekly),
        "next_open_to_next_open_plus_5_return": float(correct),
    }

    dates = pd.to_datetime(["2017-03-01", "2017-03-02", "2017-03-03", "2017-03-06", "2017-03-07"])
    quotes = pd.DataFrame(
        {
            "TradingDay": dates,
            "SecuCode": "000001",
            "ClosePrice": 10.0,
            "OpenPrice": [10.0, 10.0, 0.0, 10.0, 10.0],
            "PTFlag": [0, 0, 1, 0, 0],
        }
    )
    labeled = loader.construct_label(quotes)
    invalid_exit_label = labeled.loc[labeled.TradingDay == dates[0], "Ret_Hold"].iloc[0]
    assert np.isclose(invalid_exit_label, -0.2)
    results["suspended_exit"] = {
        "entry_open": 10.0,
        "exit_open": 0.0,
        "exit_is_suspended": True,
        "original_training_label": float(invalid_exit_label),
    }

    baseline = quotes.assign(OpenPrice=10.0, PTFlag=0)
    future_up = baseline.copy()
    future_up.loc[future_up.TradingDay == dates[1], "OpenPrice"] = 11.0
    before = loader.construct_label(baseline)
    after = loader.construct_label(future_up)
    assert dates[0] in set(before.TradingDay)
    assert dates[0] not in set(after.TradingDay)
    results["future_conditioned_signal_universe"] = {
        "identical_signal_day_inputs": True,
        "signal_row_with_normal_next_open": True,
        "signal_row_with_limit_up_next_open": False,
    }

    dates = pd.bdate_range("2017-03-01", periods=11)
    panel = pd.DataFrame(
        [
            {
                "TradingDay": d,
                "SecuCode": f"{i:06d}",
                "factor": float(i),
                "ret_5": 0.02 if i >= 16 else 0.0,
            }
            for d in dates
            for i in range(20)
        ]
    )
    other_signals = panel.copy()
    other_signals.loc[
        ~other_signals.TradingDay.isin(dates[::5]) & (other_signals.factor >= 16),
        "ret_5",
    ] = 0.50
    for filename, analyzer in [("code.py", analyzer1), ("code2.py", analyzer2)]:
        res1 = analyzer.analyze_single_period(panel, "factor", "ret_5", "Weekly", 5, True)
        res2 = analyzer.analyze_single_period(other_signals, "factor", "ret_5", "Weekly", 5, True)
        assert res1["metrics_str"] == res2["metrics_str"]
        assert not np.isclose(res1["ls_nav"].iloc[-1], res2["ls_nav"].iloc[-1])
        results[f"{filename}_plot_vs_statistics"] = {
            "all_reported_metrics_unchanged": True,
            "changed_only_non_rebalance_signal_returns": True,
            "original_plot_final_nav": float(res1["ls_nav"].iloc[-1]),
            "modified_plot_final_nav": float(res2["ls_nav"].iloc[-1]),
        }

    loss_panel = panel.copy()
    loss_panel.loc[loss_panel.factor >= 16, "ret_5"] = 0.01
    loss_panel.loc[(loss_panel.TradingDay == dates[0]) & (loss_panel.factor >= 16), "ret_5"] = -0.10
    loss_res = analyzer1.analyze_single_period(loss_panel, "factor", "ret_5", "Weekly", 5, True)
    assert "Max DD:       0.00%" in loss_res["metrics_str"]
    results["initial_nav_missing_in_drawdown"] = {
        "reported_max_drawdown": 0.0,
        "first_period_net_return": -0.103,
        "correct_max_drawdown_including_initial_capital": -0.103,
    }

    cols = ["SecuCode", "TradingDay", "OpenPrice", "ClosePrice", "PrevClosePrice", "ret", "PTFlag"]
    daily = pd.concat(
        [
            pd.read_csv(ROOT / f"data/日行情/daily_quote_{year}.csv", usecols=cols, dtype={"SecuCode": str})
            for year in [2016, 2017, 2018]
        ],
        ignore_index=True,
    )
    daily["TradingDay"] = pd.to_datetime(daily.TradingDay)
    daily = daily.sort_values(["SecuCode", "TradingDay"])
    group = daily.groupby("SecuCode")
    daily["prior_close"] = group.ClosePrice.shift()
    daily["raw_close_ret"] = daily.ClosePrice / daily.prior_close - 1
    daily["difference"] = daily.raw_close_ret - daily.ret
    for col, offset, name in [
        ("OpenPrice", -1, "next_open"),
        ("OpenPrice", -2, "exit_open"),
        ("PTFlag", -1, "next_pt"),
        ("PTFlag", -2, "exit_pt"),
    ]:
        daily[name] = group[col].shift(offset)
    daily["open_chg"] = daily.next_open / daily.ClosePrice - 1
    universe = pd.read_parquet(ROOT / "data/指数权重/index_wt_000905.parquet", columns=["date", "symbol"])
    universe = universe.rename(columns={"date": "TradingDay", "symbol": "SecuCode"})
    universe["TradingDay"] = pd.to_datetime(universe.TradingDay)
    universe["SecuCode"] = universe.SecuCode.astype(str).str.zfill(6)
    pool = daily.merge(universe, on=["TradingDay", "SecuCode"], how="inner")
    discrepancies = pool.loc[
        (pool.PTFlag == 0) & (pool.prior_close > 0) & pool.ret.notna() & (pool.difference.abs() > 0.05)
    ]
    example = discrepancies.sort_values("difference").iloc[0]
    assert len(discrepancies) == 220
    results["unadjusted_prices_in_real_data"] = {
        "scope": "2016-2018 daily observations in contemporaneous CSI500 membership; before factor warmup and other filters",
        "absolute_return_discrepancies_over_5_percentage_points": int(len(discrepancies)),
        "example": {
            "code": example.SecuCode,
            "date": str(example.TradingDay.date()),
            "previous_actual_close": float(example.prior_close),
            "ex_rights_reference_previous_close": float(example.PrevClosePrice),
            "current_close": float(example.ClosePrice),
            "raw_price_change": float(example.raw_close_ret),
            "provided_daily_return": float(example.ret),
        },
    }
    valid_entry = (pool.next_pt == 0) & (pool.open_chg < 0.095) & pool.ClosePrice.notna()
    invalid_exit = valid_entry & (pool.exit_pt == 1)
    assert int(invalid_exit.sum()) == 615
    results["real_data_suspended_exits"] = {
        "scope": "2016-2018 contemporaneous CSI500 daily observations; before factor warmup and other filters",
        "observations_passing_entry_filter_but_suspended_at_exit": int(invalid_exit.sum()),
    }
    suspended = pool.loc[invalid_exit].iloc[0]
    stock = daily.loc[daily.SecuCode == suspended.SecuCode, cols].copy()
    labels = loader.construct_label(stock)
    actual_label = labels.loc[labels.TradingDay == suspended.TradingDay, "Ret_Hold"].iloc[0]
    assert np.isclose(actual_label, -0.2)
    results["real_data_suspended_exits"]["original_function_example"] = {
        "code": suspended.SecuCode,
        "signal_date": str(suspended.TradingDay.date()),
        "entry_open": float(suspended.next_open),
        "suspended_exit_open": float(suspended.exit_open),
        "training_label": float(actual_label),
    }

    output = ROOT / "audit/verification_results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    return results


if __name__ == "__main__":
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", FutureWarning)
        results = run()
    print(json.dumps(results, ensure_ascii=False, indent=2))
