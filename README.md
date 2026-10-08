# Backtester

A tool for testing trading rules on past prices. You write a rule as plain text, such as `BUY IF PRICE > MA200`, and it shows what would have happened, compared with simply buying and holding. It is built to answer one question honestly: did the rule have an edge, or was it luck?

[Open the web app](https://backtester-rbtnk.streamlit.app/), or run it on your own computer:

```
git clone https://github.com/ron-btnk/Backtester
cd Backtester
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The hosted app runs on a small free server and sleeps when unused, so the first load can take a minute. **I recommend running the S&P 500 list locally.** It downloads and tests about 500 stocks, which takes a few minutes on a normal computer and is too heavy for the free server.

## What I did

1. **Built the backtester.** A rule language with formulas, a day-by-day simulation with costs, and a set of checks that try to catch results that only worked by luck.
2. **Tested 8 strategies on EUR/USD, 2005-2026.** Each was designed on 2005-2015 and tested once on 2016-2026. One of them models the exchange rate as a pendulum.
3. **Found that almost nothing holds up.** The pendulum lost money. The best result, a 200-day trend rule, made +77% on EUR/USD but made money on only 3 of the 28 currency pairs I then ran it on. [Results](#results)
4. **Rebuilt the engine as version 2** after finding that the first one was not realistic enough: bad prices in the data, orders filled a day late, stops that did almost nothing. [What changed](#what-version-2-changed)

## How the app works

1. **Pick what to test.** One ticker, a few, or a ready-made list: 7 major forex pairs, all 28 pairs of the 8 main currencies, or the S&P 500 stocks.
2. **Write the rules.** Up to four: when to buy, sell, short and cover.
3. **Choose how several tickers share the money.** Each on its own, or one shared portfolio.
4. **Run it.** The app loads daily prices from Yahoo Finance, simulates every day, and runs the checks.
5. **Read the verdict.** It lists what passed and what failed, with charts, every trade, and the numbers behind each check.

Quick mode asks only for tickers, dates, rules and cost. Advanced mode adds every setting and the heavier checks.

### Writing rules

```
BUY   IF PRICE > MA200                                 # long above the 200-day average
SELL  IF PRICE < MA200
SHORT IF PRICE < MA200                                 # optional: bet on a fall
COVER IF PRICE > MA200

BUY  IF (PRICE - MA20) / STD20 < -2 AND PRICE > MA200  # formulas, combined with AND / OR
SELL IF MA20 CROSSES_BELOW MA50 OR RSI14 > 75
BUY  IF PRICE < PRICE[1]                               # [1] is yesterday's value
```

| | |
|---|---|
| Indicators | `PRICE`, `MA`, `EMA`, `STD`, `RSI`, `ZSCORE`, `VOL`, `HIGH`, `LOW`, `RETURN_ND`, `DIST_MA`, `DIST_HIGH`, `DIST_LOW`, `MACD`, `MACD_SIGNAL`, `DRAWDOWN`, with a window, e.g. `MA60` |
| Comparisons | `>` `<` `>=` `<=` `CROSSES_ABOVE` `CROSSES_BELOW` |
| Maths | `+ - * / ^`, brackets, `SIN COS TAN ASIN ACOS ATAN ABS SQRT LOG EXP` |
| Settings | Cost per trade, stop loss, take profit, position sizing, volatility target, rebalance band, borrow fee, carry, interest on cash |

### One ticker, many tickers, or a portfolio

| | What happens |
|---|---|
| One ticker, or up to 5 | Each gets the full starting money and, in advanced mode, every check |
| More than 5, each on its own | Each gets one backtest. The result shows on how many the rule beat buy and hold, and any ticker can be opened for the full checks |
| One shared portfolio | One account trades all the tickers with the same rules and one pot of money |

Running a rule on a whole list is a check in itself. A real edge should show on most tickers of the same kind, not only on the one the rule was designed on.

A tick box adds the S&P 500 (the SPY fund, dividends reinvested) to the numbers and charts of any test.

### How a portfolio splits its money

By default the app decides, each evening, how much of the account each ticker gets:

1. **How likely is a profit?** It looks at how the rule's earlier signals ended on that ticker: how often they won and how big the wins and losses were. One ticker's record is short, so it is mixed with the record of all the tickers together.
2. **How much to bet?** The Kelly formula gives the share that grows the money fastest, f = p/L - (1 - p)/W, with p the chance of a win, W the average win and L the average loss. The app bets half of it. A ticker whose wins do not pay for its losses gets nothing, and is still followed on paper so it can earn its way back.
3. **Is it the same bet twice?** Positions that moved together over the last 60 days are cut. Long EUR/USD and long GBP/USD are close to one bet on the dollar.
4. **Limits.** Shares are scaled down to fit the account, nothing is borrowed, and no ticker gets more than twice an equal slice.

Advanced mode can instead give every ticker an equal slice, or spread the money equally over the open positions.

This assumes a rule that has worked on a ticker will keep working there, which is often not true. So every portfolio result also shows what equal slices would have made. In my tests the app's split did not beat them:

| Rule and tickers, 2005-2026 | App's split | Equal slices |
|---|---|---|
| 200-day trend, 7 major forex pairs | -9.3%, -0.06, -21.6% | -8.1%, -0.03, -26.4% |
| 200-day trend, 28 forex pairs | -9.3%, -0.21, -13.4% | -24.1%, -0.22, -33.5% |
| Pendulum, 7 major forex pairs | -14.7%, -0.13, -30.7% | +3.6%, 0.06, -23.4% |
| Pendulum, 28 forex pairs | -11.0%, -0.12, -23.5% | +6.8%, 0.09, -16.7% |
| 200-day trend, 10 large US stocks | +918%, 0.96, -33.2% | +492%, 0.98, -19.9% |
| 200-day trend, S&P 500 stocks | +467%, 0.84, -27.3% | +410%, 0.94, -19.1% |

Each cell is total return, Sharpe ratio and max drawdown. The higher stock returns come from having more of the money invested, with deeper drawdowns, so compare the Sharpe ratios.

## How a backtest runs

- **Prices** are daily, from Yahoo Finance, and not adjusted for dividends. Dividends are paid into the account instead: a long position gets them, a short position owes them.
- **Signals** are read at each day's close and **orders fill at the next open**, so a rule never uses a price it could not have known. Every trade pays a cost.
- **Stop loss and take profit** are measured from the price the trade opened at and fill during the day at their level, or at the open if the price jumped past it overnight. After one closes a trade, the rule has to switch off and on again before it re-enters.
- **Interest**: shorts pay a borrow fee, positions can earn or pay carry, and idle cash can earn interest. All are charged per night.
- **Forex** is treated differently, because Yahoo's forex history has faults. Some quotes are plainly wrong, and since about 2011 a bar's open, high and low are listed with the wrong day. So bad quotes are removed and listed, each day opens at the previous close, and stops are checked on closes only.

## The checks

| Check | What it asks |
|---|---|
| Costs | How much of the return did costs take? |
| A day late | Does it still work if every order is filled a day later? |
| Train/test split | Does it work on a later period the rules never saw? |
| Random benchmark | Does it beat 300 strategies that held for the same stretches at random times? |
| Sensitivity | Does it survive moving every number in the rules up and down? |
| Number search | Do better numbers found on the first period still help on the second? |
| Across tickers | Does it work on most tickers of a list? |

The first two run on every backtest. The train/test split needs advanced mode. The random benchmark, sensitivity and number search need advanced mode and a single ticker. Each check adds or removes points and the verdict is *promising*, *mixed* or *doesn't hold up*. A strategy that lost money is marked down even if the market lost more.

## Results

### Eight strategies on EUR/USD

Costs were 0.02% per trade. Buy and hold lost 17% over the period. The first two rows were rerun with version 2. The rules for the other six are not in this repository, so their results are from version 1.

| Strategy | Idea | Result |
|---|---|---|
| 200-day trend | Long above the 200-day average, short below | +77% on EUR/USD, positive in both periods. Failed on other pairs, see below |
| Pendulum | Price swings around its average like a pendulum | Failed: lost 4.9% and beat only 54% of random strategies |
| Rally then dip | Profit-takers cause a dip, latecomers buy it | About +31%, positive in both periods (v1) |
| Break-even vs momentum | Size of the drop decides who wins at the old high | Roughly flat (v1) |
| Short-term reversal | Buy after a down day, sell after an up day | About -48%, costs dominate (v1) |
| Failed breakouts | Bet against breakouts that reverse within days | About -42% (v1) |
| Equilibrium mean reversion | Bet on a return to the average | Failed on 2005-2015 (v1) |
| Break-even sellers | Old highs act as resistance | Failed on 2005-2015 (v1) |

### The trend rule did not carry over

The 200-day rule looked like the strongest result, so I ran it unchanged on all 28 currency pairs:

| 200-day trend, 2005-2026 | Result |
|---|---|
| Made money | 3 of 28 pairs |
| Beat buy and hold | 7 of 28 pairs |
| Median return | -38.0%, against -7.2% for buy and hold |

EUR/USD was the best of the 28. It was most likely the one lucky pair.

That is what testing many strategies does. If a useless rule has a 50% chance of beating buy and hold in each of two periods, then out of 8 useless rules about 2 will beat it in both. My study found 2.

### The pendulum model

The model treats the 60-day average as the bottom of a pendulum's swing and the price as the pendulum. The angle is the distance from the average in standard deviations, and the velocity is its change since yesterday:

$$\theta = \frac{P - \mathrm{MA}_{60}}{\mathrm{STD}_{60}} \qquad v = \theta_t - \theta_{t-1}$$

For small swings the energy is constant, which gives how far the swing will go before it turns:

$$\frac{d^2\theta}{dt^2} \approx -\omega^2\theta \qquad A = \sqrt{\theta^2 + \frac{v^2}{\omega^2}}$$

With ω² = 0.01 a full swing takes about 63 days. The strategy buys at the bottom of a swing below θ = -1.5 and sells when θ reaches 0.8A on the other side, the 0.8 standing for friction. If θ passes 3 the average has probably moved and the trade is closed. Shorts are the mirror image.

| EUR/USD, 2005-2026 | Pendulum | Buy and hold |
|---|---|---|
| Total return | -4.9% | -17.1% |
| Sharpe | 0.01 | -0.04 |
| Max drawdown | -42.9% | -40.0% |

It did better than buy and hold in both periods, but over the whole test it lost money, and it beat only 54% of strategies that traded at random times. Losing less than a falling market is not an edge. On simulated prices that really do swing like a pendulum the same rules beat all 300 random strategies, so the test can find a pendulum when there is one. On all 28 pairs it made money on 13 and lost on 15.

This is the second version of the model. The first used the full equation with sin and arccos, which treats θ as a real angle although it is measured in standard deviations. I rewrote it because the physics was wrong, not to improve the result.

## What version 2 changed

| | Version 1 | Version 2 |
|---|---|---|
| Bad prices | Used as they came | Bad forex quotes removed and reported |
| Forex fills | A day late after 2010 | At the close the signal came from |
| Stops | Checked on the close, filled next day, re-entered next day | Fill during the day, then wait for the rule to reset |
| Dividends | Hidden in adjusted prices | Paid into the account |
| Fill timing | Not tested | Every backtest rerun a day late |
| Several tickers | One by one | Lists, and shared portfolios |
| Benchmark | Buy and hold | Also the S&P 500 |

The effect on the two rules I could rerun:

| EUR/USD, 2005-2026 | Version 1 | Version 2 | Version 2, a day late |
|---|---|---|---|
| 200-day trend | +26.7% | +77.0% | +30.5% |
| Pendulum | -15.5% | -4.9% | -9.2% |

More than half of the trend rule's return depends on trading at the very close that gave the signal. That is possible in a market open around the clock, and it is also the most favourable assumption a backtest can make.

[CHANGELOG.md](CHANGELOG.md) has the full list.

## Limitations

- Daily data only. Within a day the backtester knows the open, high, low and close, not their order.
- Forex is simulated on closes, so a stop touched during the day that recovered is missed.
- Carry and interest on cash are one fixed rate for the whole test, and 0 unless you set them.
- The S&P 500 list is today's members. Companies that went bust or were dropped are missing, so every result on it, buy and hold included, looks better than it could have been. Holding today's 502 members made 2,052% while the index made 849%.
- Costs are a flat percentage, not real bid/ask spreads.
- The portfolio split judges a ticker by the rule's past signals on it, which with few signals is mostly luck.

## Project structure

```
backtester.py              the engine: data, rules, simulation, checks, charts
app.py                     the web app (built with Claude Code)
Backtester.ipynb           the same code as a notebook, with the pendulum study's output
data/sp500.csv             S&P 500 members on 8 October 2026, from Wikipedia
tests/test_backtester.py   tests on made-up prices, run with python -m pytest
CHANGELOG.md               what changed between versions
```

Not financial advice. MIT license.
