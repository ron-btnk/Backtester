"""The EUR/USD study behind the top of the README and the app.

Run with: python study.py
It downloads EUR/USD, reruns the 200-day trend rule with the real engine (full period, train and
test, 300 random strategies) and writes docs/headline.png. The numbers it prints are the ones the
README quotes.
"""

import matplotlib.pyplot as plt
import pandas as pd

import backtester as bt

TICKER = "EURUSD=X"
START = "2005-01-01"
SPLIT = "2016-01-01"
END = "2026-10-16"  # fixed, so the README and the app show the same numbers

TREND = {
    "buy_rule": "BUY IF PRICE > MA200",
    "sell_rule": "SELL IF PRICE < MA200",
    "short_rule": "SHORT IF PRICE < MA200",
    "cover_rule": "COVER IF PRICE > MA200",
    "cost_pct": 0.02,
    **bt.QUICK_DEFAULTS,
}

# (name, result) for the 8 ideas, all designed on 2005-2015. Only the 200-day rule and the pendulum
# are in this repository. The other results come from earlier runs whose rules were not kept, so they
# can't be rerun; NOT_RERUN marks them.
NOT_RERUN = "*"
IDEAS = [
    ("200-day trend", "held up"),
    ("Rally then dip" + NOT_RERUN, "held up"),
    ("Break-even vs momentum" + NOT_RERUN, "mixed"),
    ("Pendulum", "failed"),
    ("Short-term reversal" + NOT_RERUN, "failed"),
    ("Failed breakouts" + NOT_RERUN, "failed"),
    ("Equilibrium mean reversion" + NOT_RERUN, "failed"),
    ("Break-even sellers" + NOT_RERUN, "failed"),
]

STRATEGY_COLOR = "#2a78d6"
REFERENCE_COLOR = "#8f8e89"
INK, MUTED = "#0b0b0b", "#52514e"


def run(prices=None):
    """The 200-day rule on the study period: full, train, test and the random benchmark."""
    if prices is None:
        prices = bt.load_prices(TICKER, START, END)
    full = bt.run_backtest(prices, START, END, **TREND)
    train = bt.run_backtest(prices, START, SPLIT, **TREND)
    test = bt.run_backtest(prices, SPLIT, END, **TREND)
    random = bt.random_benchmark(full, TREND["cost_pct"], TREND["short_fee_pct"])
    return {"full": full, "train": train, "test": test, "random": random}


def plot(result):
    """Ideas marked pass/fail on the left, the 200-day rule's portfolio value on the right."""
    with plt.rc_context({"font.family": "serif"}):
        fig, (list_ax, ax) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 2.1]})
        _draw_ideas(list_ax)
        _draw_equity(ax, result["full"])
        fig.tight_layout()
    return fig


def plot_equity(result):
    """The equity panel alone, for the app, which lists the ideas next to it as text."""
    with plt.rc_context({"font.family": "serif"}):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        _draw_equity(ax, result["full"])
        fig.tight_layout()
    return fig


def _draw_ideas(ax):
    ax.axis("off")
    ax.text(0, 1, "8 ideas, designed on 2005-2015", fontsize=12, color=INK, weight="bold",
            transform=ax.transAxes, va="top")
    for i, (name, outcome) in enumerate(IDEAS):
        y = 0.86 - i * 0.105
        strong = outcome == "held up"
        style = {"fontsize": 11, "color": INK if strong else MUTED, "weight": "bold" if strong else "normal",
                 "transform": ax.transAxes, "va": "top"}
        ax.text(0, y, name, **style)
        ax.text(1, y, outcome, ha="right", **style)
    ax.text(0, 0.86 - len(IDEAS) * 0.105, f"{NOT_RERUN} earlier result, rules not kept",
            fontsize=9, color=MUTED, transform=ax.transAxes, va="top")


def _draw_equity(ax, full):
    strategy, buy_hold = full["strategy"], full["buy_hold"]
    ax.plot(strategy.index, strategy.values, color=STRATEGY_COLOR, linewidth=1.3)
    ax.plot(buy_hold.index, buy_hold.values, color=REFERENCE_COLOR, linewidth=1.1, linestyle="--")
    ax.text(strategy.index[-1], strategy.iloc[-1], " 200-day trend", color=INK, va="center", fontsize=10,
            clip_on=False)
    ax.text(buy_hold.index[-1], buy_hold.iloc[-1], " buy & hold", color=MUTED, va="center", fontsize=10,
            clip_on=False)

    split = pd.Timestamp(SPLIT)
    ax.axvline(split, color=MUTED, linewidth=1)
    ax.text(split, 1.02, "train  |  test", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=10, color=INK)
    ax.set_title(f"EUR/USD {full['start'].year}-{full['end'].year}, value of 10,000 after 0.02% costs",
                 fontsize=11, color=INK, pad=22)
    ax.set_xlim(strategy.index[0], strategy.index[-1])
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, labelsize=9)


if __name__ == "__main__":
    r = run()
    f, rnd = r["full"], r["random"]
    print(f"{TICKER} {f['start']} to {f['end']}")
    print(f"200-day trend: {f['metrics']['total_return']:+.1f}%, buy & hold {f['bh_metrics']['total_return']:+.1f}%")
    print(f"train {r['train']['metrics']['total_return']:+.1f}% (buy & hold "
          f"{r['train']['bh_metrics']['total_return']:+.1f}%), test {r['test']['metrics']['total_return']:+.1f}% "
          f"(buy & hold {r['test']['bh_metrics']['total_return']:+.1f}%)")
    print(f"beat {rnd['sharpe_beaten_pct']:.1f}% of {rnd['runs']} random strategies by Sharpe "
          f"({rnd['return_beaten_pct']:.1f}% by return)")
    print(f"largest single trade: {f['stats']['top_trade_share']:.0f}% of the profit")
    plot(r).savefig("docs/headline.png", dpi=150, facecolor="white")
    print("wrote docs/headline.png")
