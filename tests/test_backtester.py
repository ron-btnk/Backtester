"""Tests for the engine. They use made-up prices, so they need no internet. Run with: python -m pytest"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import backtester as bt

START, END = "2015-01-01", "2019-01-01"
LONG_SHORT = {"buy_rule": "BUY IF PRICE > MA50", "sell_rule": "SELL IF PRICE < MA50",
              "short_rule": "SHORT IF PRICE < MA50", "cover_rule": "COVER IF PRICE > MA50", "initial": 10000}


def make_prices(close):
    close = np.asarray(close, dtype=float)
    opens = np.concatenate([[close[0]], close[:-1]])  # each day opens where the last one closed
    return pd.DataFrame({"Open": opens, "High": np.maximum(opens, close), "Low": np.minimum(opens, close),
                         "Close": close}, index=pd.bdate_range("2013-01-01", periods=len(close)))


@pytest.fixture
def prices():
    steps = np.random.default_rng(1).normal(0, 0.01, 1800)
    return make_prices(100 * np.exp(np.cumsum(steps)))


# ---- Rules and formulas ----

def test_formula_gives_the_same_numbers_as_pandas(prices):
    close = prices["Close"]
    expected = (close - close.rolling(20).mean()) / close.rolling(20).std()
    pd.testing.assert_series_equal(bt.evaluate("(PRICE - MA20) / STD20", close), expected, check_names=False)
    pd.testing.assert_series_equal(bt.evaluate("PRICE[3]", close), close.shift(3), check_names=False)
    pd.testing.assert_series_equal(bt.evaluate("SQRT(PRICE ^ 2)", close), close, check_names=False)


def test_and_binds_tighter_than_or(prices):
    close = prices["Close"]
    signal = bt.build_signal("BUY IF PRICE > MA50 AND RSI14 < 60 OR ZSCORE20 < -2", close)
    expected = ((close > bt.get_indicator("MA50", close)) & (bt.get_indicator("RSI14", close) < 60)
                | (bt.get_indicator("ZSCORE20", close) < -2))
    assert signal.equals(expected) and signal.any() and not signal.all()


def test_crossing_fires_only_on_the_day_of_the_cross():
    close = pd.Series([1, 1, 1, 3, 3, 3, 1, 1], index=pd.bdate_range("2020-01-01", periods=8), dtype=float)
    assert bt.build_signal("BUY IF PRICE CROSSES_ABOVE 2", close).tolist() == [False] * 3 + [True] + [False] * 4
    assert bt.build_signal("SELL IF PRICE CROSSES_BELOW 2", close).tolist() == [False] * 6 + [True, False]


@pytest.mark.parametrize("text", [
    "BUY IF __import__('os').system('echo hacked') > 0",
    "BUY IF PRICE.__class__ > 0",
    "BUY IF open('secrets.txt') > 0",
    "BUY IF (lambda: 1)() > 0",
    "BUY IF EVAL(PRICE) > 0",
    "BUY IF PRICE",
    "BUY IF MA1 > 0",
    "BUY IF NOT_AN_INDICATOR > 0",
])
def test_anything_outside_the_rule_language_is_rejected(text):
    with pytest.raises(ValueError):
        bt.validate_rule(text)


def test_default_cover_is_the_short_rule_reversed():
    assert bt.default_cover("short if price < ma200") == "COVER IF PRICE > MA200"
    assert bt.default_cover("SHORT IF MA20 CROSSES_BELOW MA50") == "COVER IF MA20 CROSSES_ABOVE MA50"


# ---- Backtest engine ----

def test_no_lookahead(prices):
    # scrambling the future must not change anything that happened before it
    cut = prices.index.get_loc(pd.Timestamp("2017-06-01"))
    scrambled = prices.copy()
    scrambled.iloc[cut:] *= np.random.default_rng(2).uniform(0.5, 1.5, (len(prices) - cut, 1))
    a = bt.run_backtest(prices, START, END, **LONG_SHORT)
    b = bt.run_backtest(scrambled, START, END, **LONG_SHORT)
    before = a["strategy"].index < prices.index[cut]
    assert before.sum() > 500
    assert np.allclose(a["strategy"][before], b["strategy"][before])
    assert np.array_equal(a["weights"][before], b["weights"][before])


def test_signal_at_the_close_fills_at_the_next_open():
    # flat at 100, then one jump to 110. The rule first sees 110 at that day's close, so it has to
    # pay 110 at the next open and makes nothing. Getting in at 100 would be trading on the future
    prices = make_prices([100.0] * 300 + [110.0] * 100)
    result = bt.run_backtest(prices, "2013-06-01", "2015-01-01", "BUY IF PRICE > 105", "SELL IF NONE",
                             initial=10000, cost_pct=0.0)
    assert result["trades"].iloc[0]["entry_date"] == prices.index[301].date()
    assert result["metrics"]["total_return"] == pytest.approx(0.0)


def test_trade_profits_add_up_to_the_change_in_the_account(prices):
    result = bt.run_backtest(prices, START, END, **LONG_SHORT, cost_pct=0.1, short_fee_pct=2.0,
                             stop_loss_pct=4, take_profit_pct=8, target_vol_pct=12)
    trades = result["trades"]
    assert {"long", "short"} <= set(trades["side"]) and len(trades) > 20
    assert trades["profit"].sum() == pytest.approx(result["metrics"]["final"] - 10000, abs=0.01 * len(trades))


def test_always_long_matches_buy_and_hold(prices):
    result = bt.run_backtest(prices, START, END, "BUY IF PRICE > 0", "SELL IF NONE", initial=10000, cost_pct=0.0)
    # the rule needs one close before it can buy, so it starts a day later and then moves in step
    assert np.allclose(result["strategy"].pct_change().iloc[2:], result["buy_hold"].pct_change().iloc[2:])
    assert result["stats"]["time_long"] == pytest.approx(100, abs=0.2)


def test_costs_and_borrow_fees_only_ever_hurt(prices):
    free = bt.run_backtest(prices, START, END, **LONG_SHORT, cost_pct=0.0)["metrics"]["final"]
    costly = bt.run_backtest(prices, START, END, **LONG_SHORT, cost_pct=0.2)["metrics"]["final"]
    with_fee = bt.run_backtest(prices, START, END, **LONG_SHORT, cost_pct=0.2, short_fee_pct=3.0)["metrics"]["final"]
    assert free > costly > with_fee


def test_a_short_gains_what_the_price_loses():
    prices = make_prices(np.concatenate([np.full(300, 100.0), np.linspace(100, 80, 100)]))
    result = bt.run_backtest(prices, "2013-06-01", "2015-01-01", "BUY IF NONE", "SELL IF NONE",
                             initial=10000, cost_pct=0.0, rebalance_pct=100,
                             short_rule="SHORT IF PRICE > 0", cover_rule="COVER IF NONE")
    assert result["metrics"]["total_return"] == pytest.approx(20.0)
    assert result["stats"]["time_short"] > 99


def test_stop_loss_closes_the_trade():
    prices = make_prices(np.concatenate([np.full(300, 100.0), np.linspace(100, 70, 60), np.full(40, 70.0)]))
    result = bt.run_backtest(prices, "2013-06-01", "2015-01-01", "BUY IF PRICE[1] >= 100", "SELL IF NONE",
                             initial=10000, cost_pct=0.0, stop_loss_pct=5)
    last = result["trades"].iloc[-1]
    assert last["exit_reason"] == "stop loss" and -7 < last["return_pct"] <= -5
    assert result["weights"].iloc[-1] == 0


def test_a_wiped_out_account_stops_trading():
    # a short that triples overnight loses more than the account holds
    prices = make_prices(np.concatenate([np.full(300, 100.0), np.full(100, 300.0)]))
    result = bt.run_backtest(prices, "2013-06-01", "2015-01-01", "BUY IF PRICE > 200", "SELL IF NONE",
                             initial=10000, cost_pct=0.0, short_rule="SHORT IF PRICE < 200", cover_rule="")
    assert len(result["trades"]) == 1 and result["trades"].iloc[0]["exit_reason"] == "account wiped out"
    assert result["weights"].iloc[-1] == 0


# ---- Robustness checks ----

def test_random_benchmark_tells_skill_from_luck(prices):
    # holding only on days the price is about to rise is real timing skill (by cheating),
    # so it should beat every strategy that holds for the same stretches at random dates
    close = prices["Close"][START:END]
    weights = (close.pct_change().shift(-1) > 0).astype(float)
    cheat = bt.random_benchmark({"weights": weights, "close": close}, cost_pct=0.0)
    assert cheat["sharpe_beaten_pct"] == 100
    assert sum(length for length, _ in bt._holding_blocks(weights)) == int((weights != 0).sum())


def test_random_benchmark_is_repeatable(prices):
    result = bt.run_backtest(prices, START, END, **LONG_SHORT)
    a, b = bt.random_benchmark(result, 0.02), bt.random_benchmark(result, 0.02)
    assert a["sharpe_beaten_pct"] == b["sharpe_beaten_pct"] and 0 <= a["sharpe_beaten_pct"] <= 100


def test_only_real_settings_are_found_in_a_rule():
    # the [1] in PRICE[1], the power in ^ 2 and the whole-number factor in 2 * are not settings
    found = bt.find_parameters("BUY IF (PRICE[1] - MA60[1]) / STD60[1] < -1.5 AND RETURN_5D ^ 2 > 2 * 0.05")
    assert [(p["kind"], p["value"]) for p in found] == [("window", 60), ("window", 60), ("window", 5),
                                                         ("threshold", -1.5), ("threshold", 0.05)]


def test_one_dial_changes_the_same_number_everywhere():
    settings = {"buy_rule": "BUY IF PRICE > MA60", "sell_rule": "SELL IF PRICE < MA60 - 2 * STD60"}
    dials = bt.rule_dials(settings)
    assert list(dials) == [("window", 60)]
    assert bt.rules_with(settings, dials, {("window", 60): 45}) == {
        "buy_rule": "BUY IF PRICE > MA45", "sell_rule": "SELL IF PRICE < MA45 - 2 * STD45"}


def test_full_analysis_runs_and_gives_a_verdict(prices):
    settings = {**bt.QUICK_DEFAULTS, **LONG_SHORT, "cost_pct": 0.02}
    r = bt.analyse_ticker(prices, START, END, settings, split_date="2017-01-01")
    assert r["label"] in ("promising", "mixed", "doesn't hold up")
    assert r["train"]["end"] < r["test"]["start"]
    assert r["sens"]["tested"] == 4 and r["random"]["runs"] == bt.RANDOM_RUNS
