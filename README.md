# Backtester

Tests trading rules on past prices. You write a rule as text, such as `BUY IF PRICE > MA200`, and it shows what the rule would have made against buying and holding, then runs checks to tell an edge from luck.

**Web app:** https://backtester-rbtnk.streamlit.app/

**Locally:**

```
git clone https://github.com/ron-btnk/Backtester
cd Backtester
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The hosted app sleeps when unused, so the first load can take a minute. I recommend running the S&P 500 list locally: it loads and tests about 500 stocks, which takes a few minutes and is too heavy for the free server.

## Using it

Quick mode has three steps, all in the sidebar:

1. **Pick what to test.** A ticker (`SPY`, `EURUSD=X`), several, or a ready-made list: 7 or 28 forex pairs, or the S&P 500 stocks. Set the dates.
2. **Write the rules.** When to buy and sell. Short and cover rules are optional.
3. **Set the cost and run.**

Advanced mode adds a train/test split, optional settings (position size, stops, interest) and three heavier checks.

### Rules

```
BUY   IF PRICE > MA200
SELL  IF PRICE < MA200
SHORT IF PRICE < MA200
COVER IF PRICE > MA200

BUY  IF (PRICE - MA20) / STD20 < -2 AND PRICE > MA200
SELL IF MA20 CROSSES_BELOW MA50 OR RSI14 > 75
BUY  IF PRICE < PRICE[1]
```

| | |
|---|---|
| Indicators | `PRICE`, `MA`, `EMA`, `STD`, `RSI`, `ZSCORE`, `VOL`, `HIGH`, `LOW`, `RETURN_ND`, `DIST_MA`, `DIST_HIGH`, `DIST_LOW`, `MACD`, `MACD_SIGNAL`, `DRAWDOWN`. Add a window: `MA60` |
| Comparisons | `>` `<` `>=` `<=` `CROSSES_ABOVE` `CROSSES_BELOW`, joined with `AND` / `OR` |
| Maths | `+ - * / ^`, brackets, `SIN COS TAN ASIN ACOS ATAN ABS SQRT LOG EXP` |
| Earlier days | `PRICE[1]` is yesterday's price |

### Several tickers

| Choice | What happens |
|---|---|
| Each ticker on its own | Every ticker gets the full starting money and its own result. With more than 5, each gets one backtest and the app counts on how many the rule beat buy and hold |
| One shared portfolio | One account trades all the tickers with the same rules and one pot of money |

In a portfolio the app decides how much each ticker gets:

1. It estimates the chance of a profit from how the rule's earlier signals ended on that ticker, mixed with the record of all tickers.
2. The Kelly formula turns that into a bet size, f = p/L - (1 - p)/W, where p is the chance of a win, W the average win and L the average loss. The app bets half of f. A ticker with a negative f gets nothing.
3. Positions that moved together over the last 60 days are cut.
4. Shares are scaled to fit the account. Nothing is borrowed.

Advanced mode can use equal slices instead. Every portfolio result shows what equal slices would have made.

## How a backtest works

- **Data:** daily prices from Yahoo Finance, not adjusted for dividends. Dividends are paid into the account.
- **Timing:** signals are read at the close and orders fill at the next open, so a rule never uses a price it could not have known.
- **Costs:** every trade pays a percentage of its value.
- **Stops and take profits:** measured from the entry price. They fill during the day at their level, or at the open after a gap. The rule must switch off and on before it re-enters.
- **Interest:** optional borrow fee on shorts, carry on positions, and interest on idle cash.
- **Forex:** Yahoo's forex history has wrong quotes and, since about 2011, lists a day's open, high and low with the wrong day. Wrong quotes are removed and reported, each day opens at the previous close, and stops are checked on closes.

## The checks

| Check | Question | Runs |
|---|---|---|
| Costs | How much of the return did costs take? | Always |
| A day late | Does it survive every order being filled a day later? | Always |
| Train/test split | Does it work on a later period it never saw? | Advanced |
| Random benchmark | Does it beat 300 strategies trading at random times? | Advanced, one ticker |
| Sensitivity | Does it survive changing every number in the rules? | Advanced, one ticker |
| Number search | Do better numbers from the first period help in the second? | Advanced, one ticker |
| Across tickers | Does it work on most tickers of a list? | Lists |

The verdict is *promising*, *mixed* or *doesn't hold up*. A strategy that lost money is marked down even if the market lost more.

## Results

I tested 8 strategies on EUR/USD from 2005 to 2026. Each was designed on 2005-2015 and tested once on 2016-2026, with 0.02% costs. Buy and hold lost 17%.

| Strategy | Result |
|---|---|
| 200-day trend | +77% on EUR/USD. Made money on only 3 of 28 currency pairs |
| Pendulum | Lost 4.9%. Beat 54% of random strategies |
| Rally then dip | About +31% (version 1) |
| Break-even vs momentum | Roughly flat (version 1) |
| Short-term reversal | About -48% (version 1) |
| Failed breakouts | About -42% (version 1) |
| Equilibrium mean reversion | Failed in training (version 1) |
| Break-even sellers | Failed in training (version 1) |

The first two were rerun with the current engine. The rules for the other six are not in this repository, so those numbers come from version 1.

**The trend rule was the one lucky pair.** On all 28 pairs its median return was -38%, against -7% for buy and hold, and EUR/USD was the best of them. Its +77% also falls to +30.5% if every order fills one day later.

**The pendulum has no edge.** It lost less than buy and hold, but it lost money and did no better than random timing.

**The app's portfolio split did not beat equal slices.** Total return, Sharpe ratio and max drawdown, 2005-2026:

| | App's split | Equal slices |
|---|---|---|
| Trend, 28 forex pairs | -9.3%, -0.21, -13.4% | -24.1%, -0.22, -33.5% |
| Pendulum, 28 forex pairs | -11.0%, -0.12, -23.5% | +6.8%, 0.09, -16.7% |
| Trend, S&P 500 stocks | +467%, 0.84, -27.3% | +410%, 0.94, -19.1% |

## My thinking

**The goal is to avoid fooling myself.** A backtest will show a profit for almost any rule if you try enough rules. So the checks matter more than the return: a result only counts if it survives costs, a later period, random timing, different numbers and other tickers.

**Why EUR/USD.** It is the most traded currency pair, costs are low, and its price is not moved by earnings, dividends or stock splits. Many of its largest traders, such as central banks and companies hedging payments, are not trading for profit, which is where an edge would have to come from.

**The pendulum.** The 60-day average is the bottom of the swing and the price is the pendulum. With θ the distance from the average in standard deviations and v its daily change, a small swing has constant energy, which gives the amplitude it will reach:

$$\frac{d^2\theta}{dt^2} \approx -\omega^2\theta \qquad A = \sqrt{\theta^2 + \frac{v^2}{\omega^2}}$$

The rule buys at the bottom of a swing below θ = -1.5 and sells at 0.8A on the other side, with 0.8 for friction. My first version used the full equation with sin and arccos, which treats θ as a real angle although it is measured in standard deviations. I rewrote it because the physics was wrong, not to get a better result. On simulated prices that do swing like a pendulum the rule beats every random strategy, so the test can find a pendulum when there is one. EUR/USD is not one.

**Why most results are probably luck.** If a useless rule has a 50% chance of beating buy and hold in each of two periods, about 2 out of 8 useless rules will beat it in both. I tested 8 and found 2. That is why I ran the best one on 27 other pairs, and why it failing there is the most informative result here.

**Why I rebuilt the engine.** Checking version 1 against the raw data showed it was kinder than real trading: wrong forex quotes created false signals, forex orders were filled a day late for most of the history, and stops did almost nothing because the rule bought straight back. Version 2 fixes these, and reruns every backtest a day late, because the fill timing alone moved the trend rule's result by more than half. [CHANGELOG.md](CHANGELOG.md) lists every change.

## Limitations

- Daily data only. Within a day the order of the high and low is unknown.
- Forex is simulated on closes, so a stop touched during the day that recovered is missed.
- Carry and interest on cash are one fixed rate for the whole test, and 0 unless set.
- The S&P 500 list is today's members. Companies that failed or were dropped are missing, so results on it look too good: holding today's members made 2,052% while the index made 849%.
- Costs are a flat percentage, not real bid/ask spreads.

## Files

```
backtester.py              engine: data, rules, simulation, checks, charts
app.py                     web app (built with Claude Code)
Backtester.ipynb           the same code as a notebook
data/sp500.csv             S&P 500 members on 8 October 2026, from Wikipedia
tests/test_backtester.py   tests on made-up prices: python -m pytest
```

Not financial advice. MIT license.
