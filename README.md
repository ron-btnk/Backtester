# Backtester

A backtesting engine where you write trading rules as text, including formulas, and test them on any ticker from Yahoo Finance.

I mostly used it on EUR/USD to test my own ideas. All rules were designed on 2005-2015. I didn't touch 2016-2026 until the end, so that each idea got one honest test on data it had never seen.

**Result:** I tested 8 hypotheses and 2 survived: the 200-day trend rule and my "rally then dip" rule. Both made money in both periods while buy & hold lost about 17%. The pendulum idea, which is the default study, didn't.

## Default study: does EUR/USD move like a pendulum?

Run all cells and press Enter to reproduce this.

The idea: treat the 60-day moving average as the bottom of a pendulum and price as the pendulum swinging around it.

- θ = (price - MA60) / STD60, how far price is from the average
- v = θ today - θ yesterday
- Equation of motion: d²θ/dt² = -(g/L) sin θ
- Energy is conserved, so the swing turns where θ = arccos(cos θ - v² / (2g/L))

It buys at the bottom of a swing (below -1.5) and sells where the formula says the swing turns on the other side. If there is no turning point (the pendulum would go over the top), I take that as the equilibrium having moved and close the trade. Shorts are the mirror image.

The formulas go straight into the rules:

```
BUY   IF (PRICE[1] - MA60[1]) / STD60[1] < (PRICE[2] - MA60[2]) / STD60[2]
      AND (PRICE - MA60) / STD60 > (PRICE[1] - MA60[1]) / STD60[1]
      AND (PRICE - MA60) / STD60 < -1.5
SELL  IF ACOS(COS((PRICE - MA60) / STD60) - ((PRICE - MA60) / STD60 - (PRICE[1] - MA60[1]) / STD60[1]) ^ 2 / (2 * 0.05)) - (PRICE - MA60) / STD60 < 0.2
      OR COS((PRICE - MA60) / STD60) - ((PRICE - MA60) / STD60 - (PRICE[1] - MA60[1]) / STD60[1]) ^ 2 / (2 * 0.05) < -1
```

Before using real data I checked it on fake prices. On prices that actually swing like a pendulum it got a Sharpe of about 2.75. On random prices it got nothing. So if EUR/USD were a pendulum, the test would have picked it up.

Results on EUR/USD:

| | Strategy | Buy & hold |
|---|---|---|
| Total return | -23.1% | -16.6% |
| Sharpe | -0.23 | -0.02 |
| Max drawdown | -26.6% | -40.0% |
| Trades | 349 | - |

- 2005-2015: -13.9% vs -19.5% for buy & hold, so it beat it
- 2016-2026: -10.7% vs +4.3%, so it lost
- Costs took 11.5 points. It only beat buy & hold before costs
- It beat 25% of 300 random strategies, so worse than random timing
- Almost no nearby settings worked. There's one green square in the heatmap, in a corner on its own, which is a lucky spike rather than a real edge

The automatic search found "better" numbers on 2005-2015 (54-day window, gravity 0.062) that made +16.2% over the full period. But on 2016-2026 they still lost money. It was just fitting the past.

So EUR/USD isn't a pendulum, or at least not one stable enough to trade.

## Everything I tested

All on EUR/USD. Buy & hold lost about 17% over the full period.

| Hypothesis | Idea | Result |
|---|---|---|
| Short-term reversal | Buy after a down day, sell after an up day | Failed, about -48%. Trades almost every day so costs kill it |
| 200-day trend | Long above the 200-day average, short below | Worked, about +26%, positive in both periods |
| Equilibrium mean reversion | Bet on a return to the average, exit if overstretched | Failed |
| Break-even sellers | Old highs act as resistance, old lows as support | Failed |
| Failed breakouts | Bet against breakouts that reverse within days | Failed, about -42%, every nearby setting lost too |
| Break-even vs momentum | Size of the drop decides who wins at the old high | Mixed, roughly flat, lost in training and gained in the test |
| Rally then dip | Profit-takers cause a dip, latecomers buy it | Worked, about +31%, positive in both periods |
| Pendulum | Physics predicts where a swing turns | Failed |

The 200-day rule is the standard one from research on currency markets, so I picked it before seeing any data, which makes it less likely to be luck. Rally then dip was my own idea, and its whole heatmap is green, so it doesn't depend on one exact setting.

2 out of 8 could still partly be luck. The next step is running both, unchanged, on other currency pairs. For two of the ideas (break-even vs momentum and the pendulum) I did see the full period while building them, but I didn't change the rules after that.

## How the tests stay realistic

- Signals use the close, trades happen at the next day's open, so no lookahead
- Every trade pays a cost, shorts pay a borrow fee
- Rules are chosen on one period and tested on another
- Each strategy is compared to 300 random ones with the same time in the market
- Nearby settings get tested too, to check the result isn't a lucky spike

## Custom backtest

Type C at the start to test your own rules on any ticker.

- Quick mode: tickers, dates, rules and cost
- Advanced mode: every setting, plus the train/test split, random benchmark, sensitivity check and a search for better numbers

## Writing rules

```
BUY IF PRICE > MA200
SELL IF RSI14 > 70
BUY IF (PRICE - MA20) / STD20 < -2 AND PRICE > MA200
SELL IF MA20 CROSSES_BELOW MA50
SHORT IF ZSCORE20 > 2
COVER IF ZSCORE20 < 0
```

- Indicators: PRICE, MA, EMA, STD, RSI, ZSCORE, VOL, HIGH, LOW, RETURN_ND, DIST_MA, DIST_HIGH, DIST_LOW, MACD, DRAWDOWN (add the window, e.g. MA60)
- `PRICE[1]` means yesterday's price, `[2]` two days ago and so on
- You can use `+ - * / ^`, brackets and SIN, COS, TAN, ASIN, ACOS, ATAN, ABS, SQRT, LOG, EXP
- Combine conditions with AND / OR
- Optional: position sizing, volatility target, stop loss, take profit

It prints a results table, a breakdown of the trades and a verdict, then shows the charts.

## How to run

```
git clone https://github.com/ron-btnk/Backtester
cd Backtester
pip install -r requirements.txt
jupyter notebook Backtester.ipynb
```

Run all cells, then press Enter for the default study or type C for your own rules.

## Limitations

- Daily data only
- Yahoo's FX open prices aren't exact
- Interest rates are ignored. Holding euros vs dollars earns different rates, which matters when shorting FX
- Costs are a flat estimate, not real spreads
- Only one currency pair so far

Not financial advice.
