"""Streamlit front end for the backtester. Run with: streamlit run app.py"""

import os
import time
from contextlib import ExitStack, contextmanager
from io import BytesIO
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import backtester as bt

st.set_page_config(page_title="Backtester", layout="wide")

RULE_LABELS = {"buy_rule": "Buy rule", "sell_rule": "Sell rule", "short_rule": "Short rule",
               "cover_rule": "Cover rule"}

# keeps the page a readable width with the sidebar open or closed, and stops numbers and
# equations being cut off on narrow screens
STYLE = """
<style>
[data-testid="stMainBlockContainer"] {
    max-width: 1100px; margin: 0 auto; padding: 3rem clamp(1rem, 4vw, 3rem) 2rem;
    container-type: inline-size;
}
[data-testid="stMetricValue"] {font-size: clamp(1.25rem, 4.5cqw, 2.25rem);}
[data-testid="stMetricValue"] > div {overflow: visible; text-overflow: clip; white-space: nowrap;}
.katex-display {overflow-x: auto; overflow-y: hidden; padding: 0.25rem 0;}
[data-testid="stBaseButton-primary"]:active {transform: scale(0.97);}
.skeleton {display: flex; flex-direction: column; gap: 1rem; min-height: 75vh; padding-top: 0.5rem;}
.skeleton .cards {display: flex; gap: 1rem;}
.skeleton .bar {
    flex: 1; height: 5.5rem; border-radius: 0.5rem; background-size: 200% 100%;
    background-image: linear-gradient(90deg, rgba(128, 128, 128, 0.14) 25%, rgba(128, 128, 128, 0.28) 50%,
                                      rgba(128, 128, 128, 0.14) 75%);
    animation: skeleton-shimmer 1.4s linear infinite;
}
.skeleton .bar.title {flex: none; height: 2.25rem; width: 45%;}
.skeleton .bar.chart {min-height: 16rem;}
@keyframes skeleton-shimmer {from {background-position: 200% 0;} to {background-position: -200% 0;}}
@media (prefers-reduced-motion: reduce) {.skeleton .bar {animation: none;}}
</style>
"""

# grey placeholder in the shape of the results (title, three metric cards, verdict, chart), shown while a
# backtest runs. It is tall enough that the spinner above it can be scrolled to the top of the screen
SKELETON = """
<div class="skeleton">
  <div class="bar title"></div>
  <div class="cards"><div class="bar"></div><div class="bar"></div><div class="bar"></div></div>
  <div class="bar"></div>
  <div class="bar chart"></div>
</div>
"""

GUIDE = """
A rule is a keyword, `IF`, then one or more comparisons.

```
BUY IF PRICE > MA200
SELL IF RSI14 > 70
BUY IF (PRICE - MA20) / STD20 < -2 AND PRICE > MA200
SELL IF MA20 CROSSES_BELOW MA50 OR RSI14 > 75
SHORT IF ZSCORE20 > 2
COVER IF ZSCORE20 < 0
```

- **Indicators** (add the window, e.g. `MA60`): `PRICE`, `MA`, `EMA`, `STD`, `RSI`, `ZSCORE`, `VOL`,
  `HIGH`, `LOW`, `RETURN_5D`, `DIST_MA`, `DIST_HIGH`, `DIST_LOW`, plus `MACD`, `MACD_SIGNAL`, `DRAWDOWN`
- **Compare** with `>`, `<`, `>=`, `<=`, `CROSSES_ABOVE`, `CROSSES_BELOW` and combine with `AND` / `OR`
- **Earlier days**: `PRICE[1]` is yesterday's price, `PRICE[2]` two days ago
- **Formulas**: `+ - * / ^`, brackets, and `SIN COS TAN ASIN ACOS ATAN ABS SQRT LOG EXP`
- **Shorting** is optional. Leave the short rule blank to stay long-only. Leave the cover rule blank
  to use the opposite of the short rule
- **Sizing** (advanced): `SIZE BY RSI14 FROM 40 TO 20` holds 0% at RSI 40 and 100% at RSI 20
- **Stop loss and take profit** (advanced) are measured from the price the trade was opened at. They
  fill during the day at their level, or at the open if the price jumped past it overnight. After
  one of them closes a trade, the rule has to switch off and on again before it re-enters
- **Several tickers** are tested one by one, each with the full starting money, or as one portfolio
  that shares it. A portfolio's money is split by the app: more goes to the tickers where the rule's
  earlier signals paid off more often, less to positions that move together

Signals use the close and trades happen at the next day's open. Dividends are paid into the account.
"""

CODE = f"{bt.__version__} {os.path.getmtime(bt.__file__)} {os.path.getmtime(__file__)}"
OWN_TICKERS = "My own tickers"
SEPARATE, SHARED = "Each ticker on its own", "One shared portfolio"
SPLITS = {"Decided by the app": "smart", "An equal slice per ticker": "equal",
          "Spread over open positions": "spread"}
HOW_SPLIT = {"smart": "split by the app, with more where the rule's record says a profit is likelier",
             "equal": "split into an equal slice each", "spread": "spread equally over the open positions"}
SMART_SPLIT = f"""
The app decides how much of the account each ticker gets, from what it knew on each day:

1. **How likely is a profit?** For every ticker it looks at how the rule's earlier signals ended: how
   often they won, and how big the wins and losses were. A ticker's own record is short, so it is mixed
   with the record of all the tickers together. A ticker needs {bt.KELLY_PRIOR} signals of its own before
   its record counts as much as everyone's.
2. **How much to bet?** The Kelly formula turns that into the share of the money that grows it fastest:
   *chance of a win ÷ average loss − chance of a loss ÷ average win*. The app bets half of it, because
   the inputs are estimates. When the wins don't pay for the losses the answer is negative and the ticker
   gets nothing. Its signals are still followed on paper, so it can earn its way back.
3. **Is it the same bet twice?** Positions that moved together over the last {bt.OVERLAP_DAYS} days are
   cut, so that two copies of one bet don't get double the money.
4. **Is there enough money?** If the shares add up to more than the account they are scaled down to fit.
   Nothing is borrowed, and no ticker gets more than twice an equal slice unless you set another limit.

Until {bt.KELLY_MIN_SIGNALS} signals have closed there is no record, and every ticker gets the same.

This rests on one assumption: that a rule which has worked on a ticker will keep working there. That is
often not true, so the result always shows what equal slices would have made.
"""
SURVIVORS = ("These are today's S&P 500 members. Companies that went bust or were dropped along the way are "
             "missing, so every number here, buy & hold included, looks better than it could have been in "
             "real time. Compare the rule with holding the same stocks, not with the index.")


@st.cache_resource(show_spinner=False, ttl=3600, max_entries=60)
def load_chunk(tickers, start, end):
    # shared between reruns and never changed after loading, so it is cached without copying
    return bt.load_many(list(tickers), start, end)


def load_data(tickers, start, end, status=None):
    data, failed = {}, []
    for i in range(0, len(tickers), bt.DOWNLOAD_CHUNK):
        loaded, missing = load_chunk(tuple(tickers[i:i + bt.DOWNLOAD_CHUNK]), start, end)
        data.update(loaded)
        failed += missing
        if status and len(tickers) > bt.DOWNLOAD_CHUNK:
            status("Loading prices", min(i + bt.DOWNLOAD_CHUNK, len(tickers)), len(tickers))
    return data, failed


# the tutorial at the top of the page. Each step has the same number as its group in the sidebar
STEPS = {
    False: [
        ("1. Pick what to test",
         "Type a ticker such as `SPY` or `EURUSD=X`, or choose a ready-made list. Then set the dates."),
        ("2. Write the rules",
         "Say when to buy and when to sell, for example `BUY IF PRICE > MA200`. The short rules are optional."),
        ("3. Set the cost and run",
         "Enter what one trade costs and press **Run backtest**. The result and a verdict appear below."),
    ],
    True: [
        ("1. Pick what to test",
         "Type a ticker such as `SPY` or `EURUSD=X`, or choose a ready-made list. Then set the dates."),
        ("2. Write the rules",
         "Say when to buy and when to sell, for example `BUY IF PRICE > MA200`. The short rules are optional."),
        ("3. Set costs and a comparison",
         "Enter what one trade costs. Tick the box to see the S&P 500 next to your result."),
        ("4. Split the period in two",
         "Pick a split date. The rules are judged before and after it, which catches rules fitted to the past."),
        ("5. Adjust the details",
         "Optional: starting money, position size, stop loss, take profit, interest, and how a portfolio "
         "splits its money."),
        ("6. Run it and read the checks",
         "Press **Run backtest**. Advanced adds a random benchmark, a sensitivity check and a search for "
         "better numbers."),
    ],
}

YEAR_TIP = ("To jump to another year, click the year in the box and type it. The calendar's own year "
            "list only reaches 10 years each way.")


def sidebar_inputs():
    # the sidebar reads top to bottom as the steps of a backtest. Advanced mode adds a fourth group
    # of settings, folded away so the Run button stays in reach
    d = bt.CUSTOM_DEFAULTS
    sb = st.sidebar
    sb.header("Your backtest")
    advanced = sb.radio("Mode", ["Quick", "Advanced"], horizontal=True,
                        help="Quick asks for the essentials. Advanced adds every setting, a train/test "
                             "split, a random benchmark, a sensitivity check and a search for better "
                             "numbers. It takes longer.") == "Advanced"

    sb.markdown("##### 1. What to test")
    lists = {label: key for key, label in bt.UNIVERSES.items()}
    choice = sb.selectbox("Tickers from", [OWN_TICKERS, *lists],
                          help="Your own tickers, or a ready-made list to see whether a rule works "
                               "across a whole market and not just on one ticker.")
    universe = lists.get(choice)
    tickers = ""
    if universe is None:
        tickers = sb.text_input("Tickers", ", ".join(d["tickers"]),
                                help="Yahoo Finance symbols, separated by commas. E.g. EURUSD=X, SPY, AAPL. "
                                     "Enter two or more to compare them or to run them as one portfolio.")
        count = len(bt.as_tickers(tickers))
    else:
        count = len(bt.universe(universe))
        if universe == "SP500":
            sb.caption(f"{count} stocks. Loading and testing them takes a few minutes. It is best run on "
                       "your own computer, see the README.")

    # always on show, so it is clear that your own tickers can be run as a portfolio too
    portfolio = sb.radio("With several tickers", [SEPARATE, SHARED],
                         help="On its own: every ticker gets the full starting money and its own "
                              "result. Shared: one account trades all of them, so the starting "
                              "money is split between them. A portfolio needs two or more tickers.") == SHARED
    if not portfolio and count > bt.MAX_DETAILED:
        sb.caption(f"With more than {bt.MAX_DETAILED} tickers each gets one quick backtest. "
                   "Pick any of them afterwards for the full checks.")
    if portfolio and not advanced:
        sb.caption("The app decides how much each ticker gets, from how the rule has done on it so far. "
                   "Advanced mode has other ways to split the money.")

    left, right = sb.columns(2)
    # without max_value Streamlit stops the picker 10 years after the default date
    start = left.date_input("From", pd.Timestamp(d["start"]).date(), min_value=date(1970, 1, 1),
                            max_value=date.today(), help=YEAR_TIP)
    end = right.date_input("To", date.today(), min_value=date(1970, 1, 1), help=YEAR_TIP)

    sb.markdown("##### 2. The rules")
    settings = {
        "buy_rule": sb.text_input("Buy rule", d["buy_rule"], help="When to open a long position."),
        "sell_rule": sb.text_input("Sell rule", d["sell_rule"], help="When to close it."),
        "short_rule": sb.text_input("Short rule", d["short_rule"],
                                    help="Optional: when to bet on a fall. Leave blank for long-only."),
        "cover_rule": sb.text_input("Cover rule", d["cover_rule"],
                                    help="When to close the short. Leave blank to use the opposite of the short rule."),
        **bt.QUICK_DEFAULTS,
    }
    sb.caption("How to write rules is explained on the right.")

    sb.markdown("##### 3. Costs and comparison" if advanced else "##### 3. Cost, then run")
    settings["cost_pct"] = sb.number_input("Cost per trade (%)", min_value=0.0, value=float(d["cost_pct"]),
                                           step=0.01, format="%.2f", help="About 0.02 for FX, 0.1 for stocks.")
    compare_sp500 = sb.checkbox("Compare to the S&P 500", value=False,
                                help="Adds what the same money would have made in an S&P 500 fund "
                                     f"({bt.SP500_FUND}, dividends reinvested) to the numbers and charts.")

    split_date = max_weight_pct = None
    allocation = "smart"
    if advanced:
        sb.markdown("##### 4. Train/test split")
        if sb.checkbox("Judge the rules on two periods", value=True,
                       help="Rules are judged separately before and after the split date. A rule that "
                            "only works before it was probably fitted to the past."):
            split_date = sb.date_input("Split date", pd.Timestamp(d["split_date"]).date(),
                                       min_value=date(1970, 1, 1), max_value=date.today(),
                                       help="Must be after the start date and before the end date. " + YEAR_TIP)

        sb.markdown("##### 5. Details (optional)")
        if portfolio:
            with sb.expander("How the portfolio splits its money", expanded=True):
                allocation = SPLITS[st.radio(
                    "Split the money", list(SPLITS),
                    help="Decided by the app: more to the tickers where the rule's earlier signals paid off "
                         "more often, less to positions that move together. Equal slices: every ticker may "
                         "use the same share and the rest waits in cash. Spread: shared equally by the "
                         "tickers that have a position open.")]
                if allocation != "equal":
                    max_weight_pct = st.number_input(
                        "Most in one ticker (%)", min_value=1.0, max_value=100.0, step=5.0,
                        value=float(d["max_weight_pct"]) if allocation == "spread" else None,
                        placeholder="automatic", help="Left empty, no ticker gets more than twice an equal slice.")
        with sb.expander("Money and position size"):
            settings["initial"] = st.number_input("Starting money", min_value=1.0, value=float(d["initial"]),
                                                  step=1000.0, format="%.0f")
            settings["position_pct"] = st.number_input(
                "Max invested (%)", min_value=1.0, max_value=100.0, value=float(d["position_pct"]), step=5.0,
                help="The most of the money a position may use.")
            settings["size_rule"] = st.text_input(
                "Sizing rule", "", placeholder="SIZE BY RSI14 FROM 40 TO 20",
                help="Optional. How much to hold while the buy rule is active: 0% at the first number, "
                     "100% at the second.")
            settings["target_vol_pct"] = st.number_input(
                "Target volatility (%)", min_value=0.1, value=None, placeholder="off",
                help="Hold less when the asset is jumpy. E.g. 15.")
            settings["rebalance_pct"] = st.number_input(
                "Rebalance band (%)", min_value=0.0, value=float(d["rebalance_pct"]), step=1.0,
                help="Only trade when the position is this far from its target.")
        with sb.expander("Stop loss and take profit"):
            settings["stop_loss_pct"] = st.number_input(
                "Stop loss (%)", min_value=0.1, value=None, placeholder="off",
                help="Closes the trade when it is this far below its opening price. Fills during the day.")
            settings["take_profit_pct"] = st.number_input(
                "Take profit (%)", min_value=0.1, value=None, placeholder="off",
                help="Closes the trade when it is this far above its opening price. Fills during the day.")
        with sb.expander("Interest and fees"):
            settings["short_fee_pct"] = st.number_input(
                "Borrow fee (% per year)", min_value=0.0, value=float(d["short_fee_pct"]), step=0.1,
                help="Paid on short positions. About 0 for FX, 0.5 for stocks.")
            settings["carry_pct"] = st.number_input(
                "Carry (% per year)", value=float(d["carry_pct"]), step=0.25,
                help="Interest for holding the position overnight: earned when long, paid when short. "
                     "For a currency pair it is the first currency's interest rate minus the second's, "
                     "so it can be negative. Leave at 0 for stocks.")
            settings["cash_rate_pct"] = st.number_input(
                "Interest on cash (% per year)", min_value=0.0, value=float(d["cash_rate_pct"]), step=0.25,
                help="Earned on money that is not invested. Sharpe then counts only the return above it.")
        sb.markdown("##### 6. Run")
    else:
        sb.caption("Quick mode uses 10,000 starting money, fully invested, no stops.")

    run = sb.button("Run backtest", type="primary", width="stretch")
    return run, {"tickers": tickers, "universe": universe, "start": start, "end": end, "settings": settings,
                 "split_date": split_date, "advanced": advanced, "portfolio": portfolio,
                 "allocation": allocation, "max_weight_pct": max_weight_pct, "compare_sp500": compare_sp500}


def has_rule(text):
    try:
        return bt.parse_rule(text) is not None
    except ValueError:
        return True  # unreadable, which is reported as its own problem


def check_inputs(raw):
    """Returns (cleaned inputs, list of problems to show the user)."""
    problems = []
    start, end, split_date = raw["start"], raw["end"], raw["split_date"]
    tickers = bt.universe(raw["universe"]) if raw["universe"] else bt.as_tickers(raw["tickers"])
    if not tickers:
        problems.append("**Tickers**: enter at least one ticker, e.g. `EURUSD=X` or `SPY`.")
    if raw["portfolio"] and len(tickers) == 1:
        problems.append(f"**Portfolio**: a portfolio needs at least two tickers. Add more after `{tickers[0]}`, "
                        f"separated by commas, or choose *{SEPARATE}*.")
    if start >= end:
        problems.append("**Dates**: the start date must be before the end date.")

    settings = dict(raw["settings"])
    for key in RULE_LABELS:
        settings[key] = settings[key].strip()
    try:
        if bt.parse_rule(settings["short_rule"]) is None:
            settings["short_rule"] = settings["cover_rule"] = ""
        elif not settings["cover_rule"]:
            settings["cover_rule"] = bt.default_cover(settings["short_rule"])
    except ValueError:
        pass  # reported below with the other rules

    for key, label in RULE_LABELS.items():
        try:
            bt.validate_rule(settings[key])
        except Exception as e:
            problems.append(f"**{label}**: {e}")
    if not has_rule(settings["buy_rule"]) and not has_rule(settings["short_rule"]):
        problems.append("**Rules**: add a buy rule or a short rule, otherwise nothing is ever traded.")
    try:
        bt.validate_size(settings["size_rule"])
    except Exception as e:
        problems.append(f"**Sizing rule**: {e}")

    if split_date and start < end and not (start < split_date < end):
        problems.append(f"**Split date**: {split_date} is not between the start date ({start}) and the end date "
                        f"({end}). Move the split date, or untick *Train/test split* to use any start date.")

    iso = lambda day: day.isoformat() if day else None
    return {**raw, "tickers": tickers, "start": iso(start), "end": iso(end), "settings": settings,
            "split_date": iso(split_date)}, problems


def analyse(inputs, status=None):
    # a study is one of three kinds. "tickers": a few tickers, each with every check. "scan": many
    # tickers with one quick backtest each. "portfolio": all of them trading out of one account
    start, end, settings = inputs["start"], inputs["end"], inputs["settings"]
    study = {**inputs, "kind": "tickers", "results": {}, "details": {}, "fund": None, "code": CODE}
    try:
        study["data"], study["failed"] = load_data(inputs["tickers"], start, end, status)
    except Exception as e:
        study["data"], study["failed"] = {}, [("Prices", str(e))]
    data = study["data"]
    if inputs["compare_sp500"] and data:
        try:
            study["fund"] = load_data([bt.SP500_FUND], start, end)[0][bt.SP500_FUND]
        except Exception:
            study["failed"].append(("S&P 500 comparison", "no data found for " + bt.SP500_FUND))

    if inputs["portfolio"] and len(data) > 1:
        study["kind"] = "portfolio"
        if status and len(data) > bt.MAX_DETAILED:
            status("Trading the portfolio", 0, len(data))
        try:
            study["result"] = bt.analyse_portfolio(data, start, end, settings, inputs["split_date"],
                                                   inputs["advanced"], inputs["allocation"],
                                                   inputs["max_weight_pct"], study["fund"])
        except Exception as e:
            study["result"] = None
            study["failed"].append(("Portfolio", str(e)))
    elif len(data) > bt.MAX_DETAILED:
        study["kind"] = "scan"
        progress = (lambda done, total: status("Testing each ticker", done, total)) if status else None
        study["table"], skipped = bt.scan_tickers(data, start, end, settings, progress)
        study["failed"] += skipped
    else:
        for ticker, prices in data.items():
            try:
                study["results"][ticker] = bt.analyse_ticker(prices, start, end, settings, inputs["split_date"],
                                                             inputs["advanced"], study["fund"])
            except Exception as e:
                study["failed"].append((ticker, str(e)))
    return study


@st.cache_data(show_spinner=False)
def pendulum_study(end, code):
    # code changes whenever the app is updated, so a result worked out by older code is not reused
    d = bt.DEFAULT_STUDY
    return analyse({"tickers": d["tickers"], "universe": None, "start": d["start"], "end": end,
                    "settings": dict(d["settings"]), "split_date": d["split_date"], "advanced": True,
                    "portfolio": False, "allocation": "smart", "max_weight_pct": None, "compare_sp500": False})


def pct(x, digits=1):
    return f"{x:.{digits}f}%"


def show_metrics(r):
    full = r["full"]
    s, b = full["metrics"], full["bh_metrics"]
    sp = r["sp500"]["metrics"] if r["sp500"] else None
    held = "Holding them all" if r["portfolio"] else "Buy & hold"
    rows = [("Total return", "total_return", pct, "{:+.1f} pts", "What the starting money gained or lost."),
            ("Sharpe", "sharpe", "{:.2f}".format, "{:+.2f}",
             "Return per unit of risk, per year. Above 1 is very good, around 0 is nothing."),
            ("Max drawdown", "max_drawdown", pct, "{:+.1f} pts", "The worst fall from a peak along the way.")]
    for card, (label, key, show, delta, meaning) in zip(st.columns(3), rows):
        with card.container(border=True):
            st.metric(label, show(s[key]), delta=delta.format(s[key] - b[key]) + f" vs {held.lower()}", help=meaning)
            st.caption(f"{held}: {show(b[key])}" + (f"  \nS&P 500: {show(sp[key])}" if sp else ""))


VERDICTS = {"promising": "it passed most of the checks", "mixed": "it passed some checks and failed others",
            "doesn't hold up": "it failed most of the checks"}
MARKS = {"+": ":green[**✓**]", "-": ":red[**✗**]", "": ":gray[–]"}


def show_verdict(r, settings):
    trades = r["full"]["trades"]
    notes = list(zip(r.get("marks") or [""] * len(r["notes"]), r["notes"]))  # a result from before marks existed
    if trades.empty:
        notes.insert(0, ("-", "No trades: the rules never opened a position"))
    else:
        if bt.shorting_on(settings) and not (trades["side"] == "short").any():
            notes.append(("", "The short rule never triggered"))
        if r["test"] is not None and r["test"]["trades"].empty:
            notes.append(("-", "No trades in the test period"))
    with st.container(border=True):
        st.markdown(f"**Verdict: {r['label']}**, {VERDICTS[r['label']]}")
        st.markdown("  \n".join(f"{MARKS[mark]} {note}" for mark, note in notes))
        st.caption("✓ speaks for the rules, ✗ against, – is information. The tabs below show the detail.")
    if r["data_notes"]:
        more = f", and {len(r['data_notes']) - 4} more repairs" if len(r["data_notes"]) > 4 else ""
        st.caption("Data: " + "; ".join(r["data_notes"][:4]) + more + ".")


def show_figure(study, name, draw):
    # Streamlit reruns the whole page on every click, and drawing a chart takes seconds. Each
    # chart of a study is therefore drawn once and kept as a picture
    pictures = study.setdefault("pictures", {})
    if name not in pictures:
        fig = draw()
        pictures[name] = None
        if fig is not None:
            picture = BytesIO()
            fig.savefig(picture, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            pictures[name] = picture.getvalue()
    if pictures[name]:
        st.image(pictures[name], width="stretch")


def trades_tab(full):
    st_, trades = full["stats"], full["trades"]
    if trades.empty:
        st.info("The rules never opened a position, so there are no trades to show.")
        return
    pf = "∞" if st_["profit_factor"] == float("inf") else f"{st_['profit_factor']:.2f}"
    cols = st.columns(4)
    cols[0].metric("Closed trades", st_["trades"], help=f"{st_['long_trades']} long, {st_['short_trades']} short")
    cols[1].metric("Win rate", pct(st_["win_rate"]),
                   help=f"Average win {st_['avg_win']:.2f}%, average loss {st_['avg_loss']:.2f}%")
    cols[2].metric("Profit factor", pf, help="Total profit of winning trades divided by total loss of losing trades.")
    cols[3].metric("Avg days per trade", f"{st_['avg_days']:.1f}",
                   help=f"Long {st_['time_long']:.0f}% of days, short {st_['time_short']:.0f}%, "
                        f"out {100 - st_['time_in_market']:.0f}%")

    table = trades.rename(columns={
        "ticker": "Ticker", "side": "Side", "entry_date": "Entry", "exit_date": "Exit", "days_held": "Days",
        "size": "Size", "profit": "Profit", "return_pct": "Return %", "orders": "Orders",
        "exit_reason": "Exit reason", "status": "Status"})
    st.dataframe(table, hide_index=True, width="stretch", height=380)

    closed = trades[trades["status"] == "closed"]
    left, right = st.columns(2)
    if not closed.empty:
        by_reason = (closed.groupby(["side", "exit_reason"])
                     .agg(trades=("profit", "size"), avg_return=("return_pct", "mean"),
                          total_profit=("profit", "sum"))
                     .round(2).reset_index())
        by_reason.columns = ["Side", "Exit reason", "Trades", "Avg return %", "Total profit"]
        left.caption("Exits by reason")
        left.dataframe(by_reason, hide_index=True, width="stretch")
    yearly = bt.period_returns(full["strategy"], full["buy_hold"], "Y").rename(columns={
        "strategy_%": "Strategy %", "buy_hold_%": "Buy & hold %", "difference_%": "Difference"})
    yearly.index = yearly.index.astype(str)
    yearly.index.name = "Year"
    right.caption("By year")
    right.dataframe(yearly, width="stretch")
    if full["cash_interest"]:
        st.caption(f"Interest earned on cash: {full['cash_interest']:,.2f}")


def train_test_tab(r, split_date):
    if r["train"] is None:
        st.info("Switch to advanced mode and set a train/test split date to see how the rules did "
                "on each period separately.")
        return
    st.caption(f"Split at {split_date}. A rule that only works before the split was probably fitted to the past.")
    rows = []
    for name, part in (("Train", r["train"]), ("Test", r["test"])):
        m, bm = part["metrics"], part["bh_metrics"]
        rows.append({"Period": f"{name}: {part['start']} to {part['end']}",
                     "Strategy %": round(m["total_return"], 2), "Buy & hold %": round(bm["total_return"], 2),
                     "Strategy Sharpe": round(m["sharpe"], 2), "Buy & hold Sharpe": round(bm["sharpe"], 2),
                     "Max drawdown %": round(m["max_drawdown"], 2), "Trades": part["stats"]["trades"],
                     "Beat buy & hold": "yes" if m["total_return"] > bm["total_return"] else "no"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def timing_and_costs(r):
    # the two reruns that come with every backtest
    full, free, late = r["full"]["metrics"], r["no_cost"]["metrics"], r["late"]["metrics"]
    st.markdown("##### Costs and timing")
    st.dataframe(pd.DataFrame({
        "As tested": [pct(full["total_return"]), f"{full['sharpe']:.2f}"],
        "Without costs": [pct(free["total_return"]), f"{free['sharpe']:.2f}"],
        "Filled a day late": [pct(late["total_return"]), f"{late['sharpe']:.2f}"],
    }, index=["Total return", "Sharpe"]), width="stretch")
    st.caption("A result that needs free trading, or needs every order filled the moment the signal "
               "appears, would not survive real trading.")


def robustness_tab(ticker, r, study):
    timing_and_costs(r)
    if r["portfolio"]:
        st.info("The random benchmark, the sensitivity check and the search for better numbers look at "
                "one price series. To run them, test a ticker on its own.")
        return
    if not r["advanced"]:
        st.info("Switch to advanced mode to compare the rules with random strategies, test nearby "
                "settings and search for better numbers.")
        return
    rnd, sens, variants = r["random"], r["sens"], r["variants"]

    st.markdown("##### Random benchmark")
    if rnd is None:
        st.write("Not available: the rules never opened a position.")
    else:
        st.write(f"Beat **{rnd['sharpe_beaten_pct']:.0f}%** of {rnd['runs']} random strategies that spent "
                 f"the same time in the market, ranked by Sharpe (median random return "
                 f"{rnd['random_median_return']:.2f}%).")

    st.markdown("##### Sensitivity")
    if sens is None:
        st.write("Not available: the rules have no numbers to vary.")
    else:
        st.write(f"**{sens['stable_pct']:.0f}%** of {sens['tested']} nearby settings beat buy & hold "
                 f"(Sharpe {sens['bh_sharpe']:.2f}). A real edge shouldn't depend on one exact number.")
    show_figure(study, f"checks {ticker}", lambda: bt.plot_checks(ticker, r))
    if sens is not None:
        rows = {}
        for row in sens["rows"]:
            cells = []
            for v, sh in zip(row["values"], row["sharpes"]):
                mark = " (current)" if v == row["original"] else ""
                cells.append(f"{v:g}{mark}: " + ("n/a" if np.isnan(sh) else f"{sh:.2f}"))
            rows[row["label"]] = cells
        width = max(len(cells) for cells in rows.values())
        table = pd.DataFrame.from_dict({k: v + [""] * (width - len(v)) for k, v in rows.items()}, orient="index",
                                       columns=[f"Setting {i + 1}" for i in range(width)])
        table.index.name = "Number varied"
        with st.expander("Sharpe for each number, varied one at a time"):
            st.dataframe(table, width="stretch")

    st.markdown("##### Better numbers")
    if variants is None:
        st.write("Not available: the rules have no numbers to tune.")
    elif not variants["better"]:
        st.write("No better numbers found nearby.")
    else:
        period = "the train period" if variants["split"] else "the full period"
        st.write(f"A search on {period} found numbers with a higher Sharpe:")
        st.code("\n".join(variants["rules"].values()), language=None, wrap_lines=True)
        yours, imp = variants["yours"], variants["improved"]
        ym, im, bm = yours["metrics"], imp["metrics"], imp["bh_metrics"]
        compare = pd.DataFrame({
            "Yours": [pct(ym["total_return"]), f"{ym['sharpe']:.2f}", pct(ym["max_drawdown"]),
                      str(yours["stats"]["trades"])],
            "Improved": [pct(im["total_return"]), f"{im['sharpe']:.2f}", pct(im["max_drawdown"]),
                         str(imp["stats"]["trades"])],
            "Buy & hold": [pct(bm["total_return"]), f"{bm['sharpe']:.2f}", pct(bm["max_drawdown"]), "-"],
        }, index=["Total return", "Sharpe", "Max drawdown", "Trades"])
        st.dataframe(compare, width="stretch")
        if variants["split"]:
            it, yt = variants["improved_test"]["metrics"], variants["your_test"]["metrics"]
            st.write(f"On the test period only: Sharpe **{it['sharpe']:.2f}** improved vs **{yt['sharpe']:.2f}** "
                     "yours. If the improvement disappears here, the search was just fitting the past.")


def show_ticker(ticker, r, study):
    full = r["full"]
    st.subheader(f"{ticker}, {full['start']} to {full['end']}")
    show_metrics(r)
    show_verdict(r, study["settings"])

    charts, trades, train_test, robustness = st.tabs(["Charts", "Trades", "Train vs test", "Robustness checks"])
    with charts:
        show_figure(study, f"results {ticker}",
                    lambda: bt.plot_results(ticker, r, study["split_date"], compact=not r["advanced"]))
    with trades:
        trades_tab(full)
    with train_test:
        train_test_tab(r, study["split_date"])
    with robustness:
        robustness_tab(ticker, r, study)


def split_tab(r):
    full, equal = r["full"], r["equal"]
    if equal is not None:
        ours, theirs = full["metrics"], equal["metrics"]
        st.dataframe(pd.DataFrame({
            "This split": [pct(ours["total_return"]), f"{ours['sharpe']:.2f}", pct(ours["max_drawdown"]),
                           pct(full["stats"]["avg_invested"], 0)],
            "Equal slices": [pct(theirs["total_return"]), f"{theirs['sharpe']:.2f}", pct(theirs["max_drawdown"]),
                             pct(equal["stats"]["avg_invested"], 0)],
        }, index=["Total return", "Sharpe", "Max drawdown", "Invested on average"]), width="stretch")
        st.caption("The same rules and the same tickers, with only the split changed. If this split is not "
                   "better than equal slices, it was not worth its complexity here.")
    table = full["shares"]
    if full["allocation"] == "smart":
        if "Kelly share %" in table.columns:
            st.caption("What the split is based on at the end of the test. Win chance and the average win and "
                       "loss come from how the rule's signals ended on each ticker, mixed with the record of "
                       "all tickers. Kelly share is what the formula would bet before it is halved, cut for "
                       "overlap and scaled to fit the account. Negative means no money.")
        else:
            st.caption(f"Fewer than {bt.KELLY_MIN_SIGNALS} signals closed during the test, so there was no "
                       "record to go on and every ticker got the same.")
    st.dataframe(table, hide_index=True, width="stretch", height=420)
    if full["allocation"] == "smart":
        with st.expander("How the app decides"):
            st.markdown(SMART_SPLIT)


def show_portfolio(study):
    r = study["result"]
    if r is None:
        return
    full = r["full"]
    how = HOW_SPLIT[full["allocation"]]
    if study["max_weight_pct"] and full["allocation"] != "equal":
        how += f", at most {study['max_weight_pct']:g}% in one ticker"
    st.subheader(f"Portfolio of {len(full['tickers'])} tickers, {full['start']} to {full['end']}")
    st.caption(f"One account of {full['initial']:,.0f} trades every ticker with the same rules. The money is "
               f"{how}. The result is compared with putting the money equally into the same tickers and "
               "never selling.")
    for ticker, reason in full["skipped"]:
        st.caption(f"Left out {ticker}: {reason}.")
    show_metrics(r)
    show_verdict(r, study["settings"])

    charts, split, by_ticker, trades, train_test, robustness = st.tabs(
        ["Charts", "How the money was split", "By ticker", "Trades", "Train vs test", "Robustness checks"])
    with charts:
        show_figure(study, "portfolio", lambda: bt.plot_portfolio(r, study["split_date"]))
    with split:
        split_tab(r)
    with by_ticker:
        st.caption("Where the money was made. Buy & hold profit is what simply holding that ticker's "
                   "equal share of the money made.")
        st.dataframe(full["per_ticker"], hide_index=True, width="stretch", height=420)
    with trades:
        trades_tab(full)
    with train_test:
        train_test_tab(r, study["split_date"])
    with robustness:
        robustness_tab("Portfolio", r, study)


def show_scan(study):
    table = study["table"]
    if table.empty:
        return
    s = bt.scan_summary(table)
    n = s["tickers"]
    st.subheader(f"{n} tickers, each on its own, {table['From'].min()} to {study['end']}")
    st.caption("Every ticker was tested separately with the full starting money. A rule with a real edge "
               "should beat buy & hold on most of them, not on a lucky few.")
    cards = [("Beat buy & hold", f"{s['beat_return']} of {n}", f"{s['beat_return'] / n:.0%} of tickers, by return"),
             ("Better Sharpe", f"{s['beat_sharpe']} of {n}", f"{s['beat_sharpe'] / n:.0%} of tickers"),
             ("Median return", pct(s["median_return"]), f"Buy & hold: {pct(s['median_bh'])}"),
             ("Median Sharpe", f"{s['median_sharpe']:.2f}", f"Buy & hold: {s['median_bh_sharpe']:.2f}")]
    for card, (label, value, note) in zip(st.columns(4), cards):
        with card.container(border=True):
            st.metric(label, value)
            st.caption(note)
    if s["no_trades"]:
        st.caption(f"The rules never traded on {s['no_trades']} of the tickers.")

    overview, everything, detail = st.tabs(["Overview", "All tickers", "One ticker in detail"])
    with overview:
        show_figure(study, "scan", lambda: bt.plot_scan(table))
    with everything:
        st.caption("Click a column to sort. Difference is the rule's return minus buy & hold's, in points.")
        st.dataframe(table.sort_values("Difference", ascending=False), hide_index=True, width="stretch",
                     height=520)
    with detail:
        ticker = st.selectbox("Ticker", sorted(table["Ticker"]), index=None,
                              placeholder="Pick a ticker to run every check on it")
        if ticker:
            if ticker not in study["details"]:
                with st.spinner(f"Running the checks on {ticker}...", show_time=True):
                    study["details"][ticker] = bt.analyse_ticker(
                        study["data"][ticker], study["start"], study["end"], study["settings"],
                        study["split_date"], study["advanced"], study["fund"])
            show_ticker(ticker, study["details"][ticker], study)


def show_study(study):
    st.divider()
    st.header(study["title"])
    for ticker, reason in study["failed"][:10]:
        st.error(f"**{ticker}** was skipped: {reason}. Check the ticker symbol on Yahoo Finance and the dates.")
    if len(study["failed"]) > 10:
        st.error(f"{len(study['failed']) - 10} more tickers were skipped.")
    if study["universe"] == "SP500":
        st.warning(SURVIVORS)

    if study["kind"] == "portfolio":
        show_portfolio(study)
    elif study["kind"] == "scan":
        show_scan(study)
    elif study["results"]:
        results = study["results"]
        if len(results) > 1:
            summary = pd.DataFrame([bt.summary_row(t, r) for t, r in results.items()]).dropna(axis=1, how="all")
            st.dataframe(summary, hide_index=True, width="stretch")
            ticker = st.radio("Show details for", list(results), horizontal=True)
        else:
            ticker = next(iter(results))
        show_ticker(ticker, results[ticker], study)


def pendulum_card():
    d = bt.DEFAULT_STUDY
    with st.container(border=True):
        st.subheader("Example: does EUR/USD move like a pendulum?")
        st.write("My own study, ready to run. The 60-day average is the bottom of the swing and the price is "
                 "the pendulum: buy when a swing bottoms out, sell when it reaches the other side. The rules "
                 "were set on 2005-2015 and tested on 2016 onwards.")
        with st.expander("The idea and the maths"):
            st.caption("Equation of motion, for small swings")
            st.latex(r"\frac{d^2\theta}{dt^2} = -\omega^2\,\theta")
            st.caption(r"Its energy ½v² + ½ω²θ² tells you how far the swing will go")
            st.latex(r"\text{amplitude} = \sqrt{\theta^2 + \frac{v^2}{\omega^2}}")
            st.markdown(
                "- **θ**: how far price is from its 60-day average, in standard deviations\n"
                "- **v**: how much θ changed since yesterday\n"
                "- **ω²**: 0.01, how strongly price is pulled back (a swing of about 63 days)\n\n"
                "Markets aren't a perfect pendulum, so friction lets the swing keep only 80% of that, and news "
                "can knock it off course completely.\n\n"
                "Buy when a swing bottoms out below -1.5 and sell when it reaches 80% of the predicted "
                "amplitude on the other side. If price runs past 3 standard deviations, the average has "
                "probably moved, so get out. Shorts are the mirror image.")
        with st.expander("The rules, written out in full"):
            st.code("\n".join(d["settings"][key] for _, key in bt.RULE_SIDES), language=None, wrap_lines=True)
        return st.button("Run the pendulum study", type="primary")


def scroll_to_results():
    st.html(f'<script>/* {time.time()} */'
            'document.getElementById("results")?.scrollIntoView({behavior: "smooth"})</script>',
            unsafe_allow_javascript=True)


@contextmanager
def running(message):
    # the results sit below the study card, so bring that spot into view straight away: the spinner
    # then shows where the results will appear, and a second one sits under the sidebar's Run button.
    # The skeleton below the spinner shows the shape of what is coming. Long runs report how far
    # they are through the function this yields
    with ExitStack() as spinners:
        with st.sidebar:
            spinners.enter_context(st.spinner("Running...", show_time=True))
        spinners.enter_context(st.spinner(message, show_time=True))
        progress = st.empty()
        skeleton = st.empty()
        skeleton.html(SKELETON)
        spinners.callback(skeleton.empty)
        spinners.callback(progress.empty)
        scroll_to_results()

        def status(text, done, total):
            progress.progress(min(done / max(total, 1), 1.0),
                              text=f"{text}: {done} of {total}" if done else f"{text}...")
        yield status


def run_message(inputs):
    count = len(inputs["tickers"])
    if count > bt.DOWNLOAD_CHUNK:
        return f"Loading and testing {count} tickers. This takes a few minutes..."
    if inputs["portfolio"]:
        return f"Running the portfolio of {count} tickers..."
    if count > bt.MAX_DETAILED:
        return f"Testing {count} tickers..."
    if inputs["advanced"]:
        return "Running the backtest and the robustness checks. This can take a minute..."
    return "Running the backtest..."


def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("Backtester")
    st.write("Test a trading rule on past prices and find out whether it had an edge or was just lucky.")
    run_custom, raw_inputs = sidebar_inputs()
    steps = STEPS[raw_inputs["advanced"]]
    for row in range(0, len(steps), 3):
        for column, (title, text) in zip(st.columns(3), steps[row:row + 3]):
            # "stretch" makes the boxes of a row as tall as the tallest, whatever the length of their text
            with column.container(border=True, height="stretch"):
                st.markdown(f"**{title}**")
                st.caption(text)
    st.caption(f"The {len(steps)} steps are in the sidebar on the left, under the same numbers. "
               + ("Quick mode leaves out the settings most tests do not need. " if not raw_inputs["advanced"] else "")
               + "Or try the example below first.")
    with st.expander("How to write rules"):
        st.markdown(GUIDE)
    run_default = pendulum_card()

    # the margin keeps the spot clear of Streamlit's top bar when it is scrolled into view
    st.html('<div id="results" style="scroll-margin-top: 4.5rem"></div>')
    if run_default:
        st.session_state["problems"] = []
        with running("Running the pendulum study. This can take a minute..."):
            study = pendulum_study(date.today().isoformat(), CODE)
        if not study["results"]:
            pendulum_study.clear()  # e.g. Yahoo was unreachable: try again next time
        st.session_state["study"] = {**study, "title": "Pendulum study"}
    elif run_custom:
        inputs, problems = check_inputs(raw_inputs)
        st.session_state["problems"] = problems
        st.session_state.pop("study", None)
        if not problems:
            with running(run_message(inputs)) as status:
                st.session_state["study"] = {**analyse(inputs, status), "title": "Your backtest"}

    for problem in st.session_state.get("problems", []):
        st.error(problem)
    if st.session_state.get("study", {}).get("code", CODE) != CODE:
        st.session_state.pop("study")  # worked out before the app was updated: its shape may no longer fit
        st.info("The app was updated since that result was made. Run it again to see it.")
    if "study" in st.session_state:
        show_study(st.session_state["study"])
    if run_default or run_custom:
        scroll_to_results()  # again once the results are drawn, in case the page moved
    st.caption(f"Backtester v{bt.__version__}. Daily data from Yahoo Finance. Not financial advice.")


main()
