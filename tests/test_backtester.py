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


def make_prices(close, first_day="2013-01-01"):
    close = np.asarray(close, dtype=float)
    opens = np.concatenate([[close[0]], close[:-1]])  # each day opens where the last one closed
    return pd.DataFrame({"Open": opens, "High": np.maximum(opens, close), "Low": np.minimum(opens, close),
                         "Close": close}, index=pd.bdate_range(first_day, periods=len(close)))


def flat_prices(days=400, special=None):
    # 100 all day, every day, except the days given as {day number: (open, high, low, close)}
    prices = make_prices(np.full(days, 100.0))
    for day, bar in (special or {}).items():
        prices.iloc[day, :4] = bar
    return prices


def random_prices(seed, days=1800, first_day="2013-01-01"):
    steps = np.random.default_rng(seed).normal(0, 0.01, days)
    return make_prices(100 * np.exp(np.cumsum(steps)), first_day)


FLAT = ("2013-06-03", "2014-06-02")  # a window inside flat_prices, after the warm-up
ALWAYS_LONG = {"buy_rule": "BUY IF PRICE > 0", "sell_rule": "SELL IF NONE", "initial": 10000, "cost_pct": 0.0}
ALWAYS_SHORT = {"buy_rule": "BUY IF NONE", "sell_rule": "SELL IF NONE", "short_rule": "SHORT IF PRICE > 0",
                "cover_rule": "COVER IF NONE", "initial": 10000, "cost_pct": 0.0, "rebalance_pct": 100}


@pytest.fixture
def prices():
    return random_prices(1)


# ---- Data ----

def yahoo_frame(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"Open": close * 1.001, "High": close * 1.004, "Low": close * 0.996, "Close": close,
                         "Dividends": 0.0}, index=pd.bdate_range("2020-01-01", periods=len(close)))


def test_a_bad_forex_quote_is_removed_and_the_bars_are_rebuilt_from_closes():
    close = 1.1 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.003, 60)))
    close[30] *= 1.08  # one quote 8% away from the days on either side
    raw = yahoo_frame(close)
    clean = bt.clean_prices(raw, "EURUSD=X")
    assert raw.index[30] not in clean.index and len(clean) == 59
    assert clean.attrs["notes"][0].startswith("Removed 1 bad quote")
    assert np.allclose(clean["Open"].iloc[1:], clean["Close"].iloc[:-1])
    assert (clean["High"] == clean[["Open", "Close"]].max(axis=1)).all()
    # a stock is traded on an exchange, so a jump like that is taken as real
    assert len(bt.clean_prices(raw, "SPY")) == 60 and bt.clean_prices(raw, "SPY").attrs["notes"] == []


def test_a_big_move_in_a_wild_month_is_not_called_a_bad_quote():
    close = 1.1 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.03, 60)))
    close[30], close[31] = close[29] * 1.06, close[29] * 1.001  # 6% up and straight back, among 3% days
    assert len(bt.clean_prices(yahoo_frame(close), "EURUSD=X")) == 60


def test_fund_payouts_count_as_dividends_and_prices_are_kept_inside_the_days_range():
    raw = yahoo_frame(np.full(10, 50.0))
    raw["Capital Gains"] = 0.0
    raw.iloc[4, raw.columns.get_loc("Dividends")] = 0.5
    raw.iloc[4, raw.columns.get_loc("Capital Gains")] = 0.25
    raw.iloc[6, raw.columns.get_loc("High")] = 49.0  # below its own open and close
    clean = bt.clean_prices(raw, "SPY")
    assert clean["Dividends"].iloc[4] == 0.75 and clean["Dividends"].sum() == 0.75
    assert (clean["High"] >= clean[["Open", "Close"]].max(axis=1)).all()


def test_ready_made_lists_stand_for_their_tickers():
    assert len(bt.FOREX_ALL) == 28 and len(set(bt.FOREX_ALL)) == 28
    assert bt.FOREX_MAJORS == ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X", "USDJPY=X"]
    stocks = bt.universe("SP500")
    assert 450 < len(stocks) < 550 and len(set(stocks)) == len(stocks) and {"AAPL", "BRK-B"} <= set(stocks)
    assert bt.as_tickers("spy, forex, eurusd=x, ") == ["SPY"] + bt.FOREX_MAJORS
    assert set(bt.UNIVERSES) == {"FOREX", "FOREX28", "SP500"} and all(bt.universe(name) for name in bt.UNIVERSES)


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


# ---- Stops, take profits and delay ----

def only_trade(prices, **settings):
    result = bt.run_backtest(prices, *FLAT, **settings)
    assert len(result["trades"]) == 1
    return result["trades"].iloc[0], result["metrics"]["final"]


def test_a_stop_fills_at_its_level_on_the_day_it_is_touched():
    # bought at 100 with a 5% stop. The price dips to 93 during day 200 and closes at 99
    prices = flat_prices(special={200: (100, 100, 93, 99)})
    trade, final = only_trade(prices, **ALWAYS_LONG, stop_loss_pct=5)
    assert trade["exit_reason"] == "stop loss" and trade["exit_date"] == prices.index[200].date()
    assert trade["return_pct"] == pytest.approx(-5.0) and final == pytest.approx(9500)


def test_a_price_that_jumps_past_the_stop_overnight_fills_at_the_open():
    trade, final = only_trade(flat_prices(special={200: (90, 95, 90, 95)}), **ALWAYS_LONG, stop_loss_pct=5)
    assert trade["exit_reason"] == "stop loss" and final == pytest.approx(9000)


def test_a_take_profit_fills_at_its_level_or_at_a_better_open():
    trade, final = only_trade(flat_prices(special={200: (100, 112, 100, 101)}), **ALWAYS_LONG, take_profit_pct=10)
    assert trade["exit_reason"] == "take profit" and final == pytest.approx(11000)
    _, final = only_trade(flat_prices(special={200: (115, 115, 115, 115)}), **ALWAYS_LONG, take_profit_pct=10)
    assert final == pytest.approx(11500)


def test_when_one_day_touches_both_levels_the_stop_is_assumed_to_come_first():
    trade, final = only_trade(flat_prices(special={200: (100, 112, 93, 100)}), **ALWAYS_LONG,
                              stop_loss_pct=5, take_profit_pct=10)
    assert trade["exit_reason"] == "stop loss" and final == pytest.approx(9500)


def test_a_short_is_stopped_when_the_price_rises():
    trade, final = only_trade(flat_prices(special={200: (100, 106, 100, 100)}), **ALWAYS_SHORT, stop_loss_pct=5)
    assert trade["side"] == "short" and trade["exit_reason"] == "stop loss" and final == pytest.approx(9500)


def test_after_a_stop_the_rule_must_switch_off_and_on_before_it_enters_again():
    rule = {**ALWAYS_LONG, "buy_rule": "BUY IF PRICE >= 100", "stop_loss_pct": 5}
    still_on = flat_prices(special={200: (100, 100, 93, 100)})  # closes back at 100: the rule never switched off
    reset = flat_prices(special={200: (100, 100, 93, 99)})  # closes at 99: off, and on again the day after
    assert len(bt.run_backtest(still_on, *FLAT, **rule)["trades"]) == 1
    trades = bt.run_backtest(reset, *FLAT, **rule)["trades"]
    assert len(trades) == 2 and trades.iloc[1]["entry_date"] == reset.index[202].date()


def test_no_lookahead_with_stops_or_in_a_portfolio(prices):
    cut = prices.index.get_loc(pd.Timestamp("2017-06-01"))
    other = random_prices(7)
    scrambled, other_scrambled = prices.copy(), other.copy()
    noise = np.random.default_rng(2).uniform(0.5, 1.5, (len(prices) - cut, 1))
    scrambled.iloc[cut:] *= noise
    other_scrambled.iloc[cut:] *= noise
    settings = {**LONG_SHORT, "stop_loss_pct": 3, "take_profit_pct": 6}
    before = slice(None, prices.index[cut - 1])
    a = bt.run_backtest(prices, START, END, **settings)
    b = bt.run_backtest(scrambled, START, END, **settings)
    assert {"stop loss", "take profit"} <= set(a["trades"]["exit_reason"])
    assert np.allclose(a["strategy"][before], b["strategy"][before])
    a = bt.run_portfolio({"A": prices, "B": other}, START, END, **settings)
    b = bt.run_portfolio({"A": scrambled, "B": other_scrambled}, START, END, **settings)
    assert len(a["strategy"][before]) > 500 and np.allclose(a["strategy"][before], b["strategy"][before])


def test_a_day_of_delay_moves_the_fills_a_day_later(prices):
    now = bt.run_backtest(prices, START, END, **LONG_SHORT)["trades"]
    late = bt.run_backtest(prices, START, END, **LONG_SHORT, delay_days=1)["trades"]
    on_time = set(pd.to_datetime(now["entry_date"]))
    for entry in pd.to_datetime(late["entry_date"])[5:40]:  # the first few differ: the window starts mid-signal
        assert prices.index[prices.index.get_loc(entry) - 1] in on_time


# ---- Dividends and interest ----

def test_dividends_are_paid_to_longs_and_owed_by_shorts():
    prices = flat_prices()
    prices["Dividends"] = 0.0
    prices.iloc[200, prices.columns.get_loc("Dividends")] = 1.0  # 1 a share on a 100 share position
    long = bt.run_backtest(prices, *FLAT, **ALWAYS_LONG)
    assert long["metrics"]["final"] == pytest.approx(10100)  # paid in and put back into the shares
    assert long["bh_metrics"]["final"] == pytest.approx(10100)
    assert bt.run_backtest(prices, *FLAT, **ALWAYS_SHORT)["metrics"]["final"] == pytest.approx(9900)


def test_carry_is_earned_long_and_paid_short_for_every_night_held():
    long = bt.run_backtest(flat_prices(), *FLAT, **ALWAYS_LONG, carry_pct=3.65)  # 0.01% a night
    days = long["strategy"].index
    nights = (days[-1] - days[1]).days  # in the market from the second day, weekends included
    assert nights > 300
    assert long["metrics"]["final"] == pytest.approx(10000 * (1 + 0.0001 * nights))
    short = bt.run_backtest(flat_prices(), *FLAT, **ALWAYS_SHORT, carry_pct=3.65)
    assert short["metrics"]["final"] == pytest.approx(10000 * (1 - 0.0001 * nights))


def test_idle_cash_earns_interest_and_invested_money_does_not():
    idle = bt.run_backtest(flat_prices(), *FLAT, "BUY IF NONE", "SELL IF NONE", initial=10000, cash_rate_pct=3.65)
    days = idle["strategy"].index
    nights = (days[-1] - days[0]).days
    final = idle["metrics"]["final"]
    assert 10000 * (1 + 0.0001 * nights) < final <= 10000 * 1.0001 ** nights  # interest on the interest too
    assert idle["cash_interest"] == pytest.approx(final - 10000)
    busy = bt.run_backtest(flat_prices(), *FLAT, **ALWAYS_LONG, cash_rate_pct=3.65)
    assert busy["metrics"]["final"] == pytest.approx(10001)  # only the first night, before it was invested


def test_sharpe_counts_only_the_return_above_cash(prices):
    values = prices["Close"]
    assert bt.performance(values, 100, cash_rate_pct=3)["sharpe"] < bt.performance(values, 100)["sharpe"]


# ---- Portfolios and many tickers ----

LONG_ONLY = {"buy_rule": "BUY IF PRICE > MA50", "sell_rule": "SELL IF PRICE < MA50", "initial": 10000, "cost_pct": 0.1}


def test_a_portfolio_of_one_ticker_is_the_same_as_a_single_backtest(prices):
    settings = {**LONG_SHORT, "stop_loss_pct": 4, "short_fee_pct": 1.0}
    one = bt.run_backtest(prices, START, END, **settings)
    many = bt.run_portfolio({"A": prices}, START, END, **settings)
    assert np.allclose(one["strategy"], many["strategy"]) and np.allclose(one["buy_hold"], many["buy_hold"])
    assert many["tickers"] == ["A"] and list(many["trades"]["ticker"].unique()) == ["A"]


def test_a_portfolio_trades_every_ticker_out_of_one_account():
    data = {f"T{seed}": random_prices(seed) for seed in (1, 2, 3, 4)}
    data["LATE"] = random_prices(9, days=900, first_day="2016-01-01")  # listed a year into the test
    data["TINY"] = random_prices(8, days=10, first_day="2016-01-01")  # too few days to test
    result = bt.run_portfolio(data, START, END, **LONG_ONLY)
    trades = result["trades"]
    assert result["tickers"] == ["T1", "T2", "T3", "T4", "LATE"] and [t for t, _ in result["skipped"]] == ["TINY"]
    assert set(trades["ticker"]) == set(result["tickers"])
    assert trades.loc[trades["ticker"] == "LATE", "entry_date"].min() >= pd.Timestamp("2016-01-01").date()
    # nothing is borrowed: never more than the account invested (a little drift inside the rebalance band aside)
    assert result["weights"].min() >= 0 and result["gross"].max() < 1.02 and result["positions"].max() <= 5
    change = result["metrics"]["final"] - 10000
    assert trades["profit"].sum() == pytest.approx(change, abs=0.01 * len(trades))
    assert result["per_ticker"]["Profit"].sum() == pytest.approx(change, abs=0.01 * len(trades))
    # buy & hold splits the money equally and holds, with LATE's share in cash until it lists
    assert result["buy_hold"].iloc[0] == pytest.approx(10000, rel=0.02)


def test_money_is_split_into_equal_slices_or_spread_over_open_positions():
    # the same four tickers each time, and only UP is above 150, so only UP is ever bought
    data = {"UP": flat_prices() * 2, "A": flat_prices(), "B": flat_prices(), "C": flat_prices()}
    rule = {**ALWAYS_LONG, "buy_rule": "BUY IF PRICE > 150"}
    invested = lambda **how: bt.run_portfolio(data, *FLAT, **rule, **how)["weights"].iloc[-1]
    assert invested() == pytest.approx(0.25)  # an equal slice each, the other three stay in cash
    assert invested(max_weight_pct=60) == pytest.approx(0.60)  # all of it to the open position, up to the cap
    assert invested(max_weight_pct=100) == pytest.approx(1.0)


def test_scan_gives_one_row_per_ticker_and_names_the_ones_it_could_not_test():
    data = {"A": random_prices(1), "B": random_prices(2), "TINY": random_prices(3, days=10, first_day="2016-01-01")}
    settings = {**bt.QUICK_DEFAULTS, **LONG_SHORT, "cost_pct": 0.02}
    table, failed = bt.scan_tickers(data, START, END, settings)
    assert list(table["Ticker"]) == ["A", "B"] and [ticker for ticker, _ in failed] == ["TINY"]
    summary = bt.scan_summary(table)
    assert summary["tickers"] == 2 and 0 <= summary["beat_return"] <= 2
    single = bt.run_backtest(data["A"], START, END, **settings)
    assert table["Return %"].iloc[0] == round(single["metrics"]["total_return"], 1)


def test_portfolio_analysis_runs_and_gives_a_verdict():
    data = {f"T{seed}": random_prices(seed) for seed in (1, 2, 3)}
    settings = {**bt.QUICK_DEFAULTS, **LONG_ONLY}
    r = bt.analyse_portfolio(data, START, END, settings, split_date="2017-01-01", max_weight_pct=50)
    assert r["portfolio"] and r["label"] in ("promising", "mixed", "doesn't hold up")
    assert r["train"]["end"] < r["test"]["start"] and r["random"] is None and r["sens"] is None
    assert any(note.startswith("Filled a day late") for note in r["notes"])


def test_sp500_comparison_holds_the_fund_over_the_same_days(prices):
    settings = {**bt.QUICK_DEFAULTS, **LONG_SHORT, "cost_pct": 0.02}
    # with the ticker itself as the "fund", the comparison has to be its own buy & hold
    r = bt.analyse_ticker(prices, START, END, settings, advanced=False, sp500_fund=prices)
    assert np.allclose(r["sp500"]["values"], r["full"]["buy_hold"]) and "the S&P 500" in r["notes"][-1]
    listed_later = random_prices(5, days=300, first_day="2018-01-01")
    assert bt.analyse_ticker(prices, START, END, settings, advanced=False, sp500_fund=listed_later)["sp500"] is None
    assert bt.analyse_ticker(prices, START, END, settings, advanced=False)["sp500"] is None


# ---- Robustness checks ----

def test_random_benchmark_tells_skill_from_luck(prices):
    # holding only on days the price is about to rise is real timing skill (by cheating),
    # so it should beat every strategy that holds for the same stretches at random dates
    close = prices["Close"][START:END]
    weights = (close.pct_change().shift(-1) > 0).astype(float)
    cheat = bt.random_benchmark({"weights": weights, "buy_hold": close}, cost_pct=0.0)
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
