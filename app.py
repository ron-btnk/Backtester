"""Streamlit front end for the backtester. Run with: streamlit run app.py"""

import time
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
</style>
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

Signals use the close and trades happen at the next day's open.
"""


@st.cache_data(show_spinner=False, ttl=3600)
def load_prices(ticker, start, end):
    return bt.load_prices(ticker, start, end)


def sidebar_inputs():
    d = bt.CUSTOM_DEFAULTS
    sb = st.sidebar
    sb.header("Your backtest")
    advanced = sb.radio("Mode", ["Quick", "Advanced"], horizontal=True,
                        help="Advanced adds a train/test split, a random benchmark, a sensitivity "
                             "check and a search for better numbers. It takes longer.") == "Advanced"
    tickers = sb.text_input("Tickers", ", ".join(d["tickers"]),
                            help="Yahoo Finance symbols, separated by commas. E.g. EURUSD=X, SPY, AAPL")
    left, right = sb.columns(2)
    start = left.date_input("Start date", pd.Timestamp(d["start"]).date(), min_value=date(1970, 1, 1))
    end = right.date_input("End date", date.today(), min_value=date(1970, 1, 1))

    settings = {
        "buy_rule": sb.text_input("Buy rule", d["buy_rule"]),
        "sell_rule": sb.text_input("Sell rule", d["sell_rule"]),
        "short_rule": sb.text_input("Short rule", d["short_rule"], help="Optional. Leave blank for long-only."),
        "cover_rule": sb.text_input("Cover rule", d["cover_rule"],
                                    help="Leave blank to use the opposite of the short rule."),
        "cost_pct": sb.number_input("Cost per trade (%)", min_value=0.0, value=float(d["cost_pct"]),
                                    step=0.01, format="%.2f", help="About 0.02 for FX, 0.1 for stocks."),
        **bt.QUICK_DEFAULTS,
    }

    split_date = None
    if advanced:
        settings["short_fee_pct"] = sb.number_input(
            "Borrow fee (% per year)", min_value=0.0, value=float(d["short_fee_pct"]), step=0.1,
            help="Paid on short positions. About 0 for FX, 0.5 for stocks.")
        settings["size_rule"] = sb.text_input(
            "Sizing rule", "", placeholder="SIZE BY RSI14 FROM 40 TO 20",
            help="Optional. How much to hold while the buy rule is active.")
        settings["initial"] = sb.number_input("Starting money", min_value=1.0, value=float(d["initial"]),
                                              step=1000.0, format="%.0f")
        settings["position_pct"] = sb.number_input("Max invested (%)", min_value=1.0, max_value=100.0,
                                                   value=float(d["position_pct"]), step=5.0)
        settings["target_vol_pct"] = sb.number_input(
            "Target volatility (%)", min_value=0.1, value=None, placeholder="off",
            help="Hold less when the asset is jumpy. E.g. 15.")
        settings["rebalance_pct"] = sb.number_input(
            "Rebalance band (%)", min_value=0.0, value=float(d["rebalance_pct"]), step=1.0,
            help="Only trade when the position is this far from its target.")
        settings["stop_loss_pct"] = sb.number_input("Stop loss (%)", min_value=0.1, value=None,
                                                    placeholder="off")
        settings["take_profit_pct"] = sb.number_input("Take profit (%)", min_value=0.1, value=None,
                                                      placeholder="off")
        if sb.checkbox("Train/test split", value=True,
                       help="Rules are judged separately before and after this date."):
            split_date = sb.date_input("Split date", pd.Timestamp(d["split_date"]).date(),
                                       min_value=date(1970, 1, 1))
    else:
        sb.caption("Quick mode uses 10,000 starting money, fully invested, no stops.")

    run = sb.button("Run backtest", type="primary", width="stretch")
    return run, advanced, tickers, start, end, settings, split_date


def has_rule(text):
    try:
        return bt.parse_rule(text) is not None
    except ValueError:
        return True  # unreadable, which is reported as its own problem


def check_inputs(tickers, start, end, settings, split_date):
    """Returns (cleaned inputs, list of problems to show the user)."""
    problems = []
    tickers = bt.as_tickers(tickers)
    if not tickers:
        problems.append("**Tickers**: enter at least one ticker, e.g. `EURUSD=X` or `SPY`.")
    if start >= end:
        problems.append("**Dates**: the start date must be before the end date.")

    settings = dict(settings)
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

    if split_date and not (start < split_date < end):
        problems.append("**Split date**: it must fall between the start and end date.")

    iso = lambda day: day.isoformat() if day else None
    return (tickers, iso(start), iso(end), settings, iso(split_date)), problems


def analyse(tickers, start, end, settings, split_date, advanced):
    results, failed = {}, []
    for ticker in tickers:
        try:
            prices = load_prices(ticker, start, end)
            results[ticker] = bt.analyse_ticker(prices, start, end, settings, split_date, advanced)
        except Exception as e:
            failed.append((ticker, str(e)))
    return {"results": results, "failed": failed, "settings": settings, "split_date": split_date,
            "advanced": advanced}


@st.cache_data(show_spinner=False)
def pendulum_study(end):
    d = bt.DEFAULT_STUDY
    return analyse(d["tickers"], d["start"], end, dict(d["settings"]), d["split_date"], advanced=True)


def pct(x, digits=1):
    return f"{x:.{digits}f}%"


def show_metrics(full):
    s, b = full["metrics"], full["bh_metrics"]
    rows = [("Total return", pct(s["total_return"]), pct(b["total_return"]),
             f"{s['total_return'] - b['total_return']:+.1f} pts"),
            ("Sharpe", f"{s['sharpe']:.2f}", f"{b['sharpe']:.2f}", f"{s['sharpe'] - b['sharpe']:+.2f}"),
            ("Max drawdown", pct(s["max_drawdown"]), pct(b["max_drawdown"]),
             f"{s['max_drawdown'] - b['max_drawdown']:+.1f} pts")]
    for card, (label, value, bh_value, delta) in zip(st.columns(3), rows):
        with card.container(border=True):
            st.metric(label, value, delta=delta)
            st.caption(f"Buy & hold: {bh_value}")


def show_verdict(r, settings):
    trades = r["full"]["trades"]
    notes = list(r["notes"])
    if trades.empty:
        notes.insert(0, "No trades: the rules never opened a position")
    else:
        if bt.shorting_on(settings) and not (trades["side"] == "short").any():
            notes.append("The short rule never triggered")
        if r["test"] is not None and r["test"]["trades"].empty:
            notes.append("No trades in the test period")
    with st.container(border=True):
        st.markdown(f"**Verdict: {r['label']}**\n\n" + "\n".join(f"- {note}" for note in notes))


def show_figure(fig):
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)


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
        "side": "Side", "entry_date": "Entry", "exit_date": "Exit", "days_held": "Days", "size": "Size",
        "profit": "Profit", "return_pct": "Return %", "orders": "Orders", "exit_reason": "Exit reason",
        "status": "Status"})
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
    yearly = full["yearly"].rename(columns={"strategy_%": "Strategy %", "buy_hold_%": "Buy & hold %",
                                            "difference_%": "Difference"})
    yearly.index = yearly.index.astype(str)
    yearly.index.name = "Year"
    right.caption("By year")
    right.dataframe(yearly, width="stretch")


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


def robustness_tab(ticker, r):
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
                 f"the same time in the market (median random return {rnd['random_median_return']:.2f}%).")

    st.markdown("##### Sensitivity")
    if sens is None:
        st.write("Not available: the rules have no numbers to vary.")
    else:
        st.write(f"**{sens['stable_pct']:.0f}%** of {sens['tested']} nearby settings beat buy & hold "
                 f"(Sharpe {sens['bh_sharpe']:.2f}). A real edge shouldn't depend on one exact number.")
    show_figure(bt.plot_checks(ticker, r))
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


def show_study(study):
    st.divider()
    st.header(study["title"])
    for ticker, reason in study["failed"]:
        st.error(f"**{ticker}** was skipped: {reason}. Check the ticker symbol on Yahoo Finance and the dates.")
    results = study["results"]
    if not results:
        return

    if len(results) > 1:
        summary = pd.DataFrame([bt.summary_row(t, r) for t, r in results.items()]).dropna(axis=1, how="all")
        st.dataframe(summary, hide_index=True, width="stretch")
        ticker = st.radio("Show details for", list(results), horizontal=True)
    else:
        ticker = next(iter(results))
    r = results[ticker]
    full = r["full"]

    st.subheader(f"{ticker}, {full['start']} to {full['end']}")
    show_metrics(full)
    show_verdict(r, study["settings"])

    charts, trades, train_test, robustness = st.tabs(["Charts", "Trades", "Train vs test", "Robustness checks"])
    with charts:
        show_figure(bt.plot_results(ticker, r, study["split_date"], compact=not r["advanced"]))
    with trades:
        trades_tab(full)
    with train_test:
        train_test_tab(r, study["split_date"])
    with robustness:
        robustness_tab(ticker, r)


def pendulum_card():
    d = bt.DEFAULT_STUDY
    with st.container(border=True):
        st.subheader("Does EUR/USD move like a pendulum?")
        st.write("My default study. The 60-day average is treated as the bottom of a pendulum's swing and "
                 "price as the pendulum. Rules were set on 2005-2015 and tested on 2016 onwards.")
        st.caption("Equation of motion")
        st.latex(r"\frac{d^2\theta}{dt^2} = -\frac{g}{L}\,\sin\theta")
        st.caption("Energy is conserved, so a swing turns at")
        st.latex(r"\theta = \arccos\!\left(\cos\theta - \frac{v^2}{2\,g/L}\right)")
        st.markdown(
            "- **θ**: how far price is from its 60-day average, in standard deviations\n"
            "- **v**: how much θ changed since yesterday\n"
            "- **g/L**: 0.05, how strongly price is pulled back to the average\n\n"
            "Buy when a swing bottoms out below -1.5 and sell where the formula says it turns on the "
            "other side. If no turning point exists (the swing would go over the top), the average has "
            "probably moved, so get out. Shorts are the mirror image.")
        with st.expander("The rules, written out in full"):
            st.code("\n".join(d["settings"][key] for _, key in bt.RULE_SIDES), language=None, wrap_lines=True)
        return st.button("Run the pendulum study", type="primary")


def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("Backtester")
    st.write("Write trading rules as text and test them on any ticker from Yahoo Finance. "
             "Run my default study below, or use the sidebar on the left to test your own rules.")

    run_custom, advanced, tickers, start, end, settings, split_date = sidebar_inputs()
    run_default = pendulum_card()
    with st.expander("How to write rules"):
        st.markdown(GUIDE)

    if run_default:
        st.session_state["problems"] = []
        with st.spinner("Running the pendulum study. This can take a minute..."):
            study = pendulum_study(date.today().isoformat())
        if not study["results"]:
            pendulum_study.clear()  # e.g. Yahoo was unreachable: try again next time
        st.session_state["study"] = {**study, "title": "Pendulum study"}
    elif run_custom:
        inputs, problems = check_inputs(tickers, start, end, settings, split_date)
        st.session_state["problems"] = problems
        st.session_state.pop("study", None)
        if not problems:
            with st.spinner("Running the backtest and the robustness checks. This can take a minute..."
                            if advanced else "Running the backtest..."):
                st.session_state["study"] = {**analyse(*inputs, advanced), "title": "Your backtest"}

    st.html('<div id="results"></div>')
    for problem in st.session_state.get("problems", []):
        st.error(problem)
    if "study" in st.session_state:
        show_study(st.session_state["study"])
    if run_default or run_custom:
        # the results sit below the study card, so bring them into view after a run
        st.html(f'<script>/* {time.time()} */'
                'document.getElementById("results")?.scrollIntoView({behavior: "smooth"})</script>',
                unsafe_allow_javascript=True)
    st.caption("Daily data only. Not financial advice.")


main()
