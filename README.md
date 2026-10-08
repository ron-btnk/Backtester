# Backtester

A backtester for trading rules written as plain text, including full formulas. Test a rule on one ticker, on a whole list of them (28 forex pairs, the S&P 500 stocks) or as one portfolio that shares the same money. Every backtest runs with trading costs, a rerun with every order filled a day late, an out-of-sample test period, a comparison against 300 random strategies and a parameter sensitivity check, so a result that only worked by luck is easy to spot.

[Open the web app](https://backtester-rbtnk.streamlit.app/) to try it without installing anything.

The repository includes a study of 8 strategies on EUR/USD from 2005 to 2026, one of which models the exchange rate as a damped pendulum. The pendulum did not hold up. A 200-day trend rule did on EUR/USD, and version 2 shows it does not carry over to other currency pairs. [See the study](#study-eurusd-2005-2026).

## What is new in version 2

Version 2 makes the simulation closer to real trading and adds testing on many tickers. [CHANGELOG.md](CHANGELOG.md) has the full list.

| | Version 1 | Version 2 |
|---|---|---|
| Bad prices | Used as they came from Yahoo | Bad forex quotes are removed and reported |
| Forex fills | Yahoo's open, which is a day late after 2010 | The close the signal was read from |
| Stop loss, take profit | Checked on the close, filled the next day | Fill during the day at their level |
| After a stop | Bought back the next day | Waits for the rule to switch off and on |
| Dividends | Hidden inside adjusted prices | Paid into the account |
| Interest | Borrow fee on shorts | Also carry on positions and interest on cash |
| Fill timing | Not tested | Every backtest is rerun a day late |
| Several tickers | One by one | One by one, or one shared portfolio |
| Ready-made lists | None | Forex pairs, S&P 500 stocks |
| Benchmark | Buy and hold | Also the S&P 500, on request |

## Quick start

Online: https://backtester-rbtnk.streamlit.app/

Locally, as a web app:

```
cd ~\Desktop
git clone https://github.com/ron-btnk/Backtester
cd Backtester
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Locally, as a notebook: run the first four lines above, then `python -m notebook Backtester.ipynb`. Run all cells, press Enter for the pendulum study or type C to enter your own rules.

The hosted app runs on a small free server. It is slower on the heavy checks and sleeps after 12 hours without visitors, so the first load can take a minute. For many backtests, and for the S&P 500 list, run it locally.

## Writing rules

A strategy is up to four rules: when to buy, sell, short and cover. Each rule compares two formulas.

```
# Trend following: long above the 200-day average, short below
BUY   IF PRICE > MA200
SELL  IF PRICE < MA200
SHORT IF PRICE < MA200
COVER IF PRICE > MA200

# Formulas on both sides, combined with AND / OR
BUY  IF (PRICE - MA20) / STD20 < -2 AND PRICE > MA200
SELL IF MA20 CROSSES_BELOW MA50 OR RSI14 > 75

# [n] is the value n days ago: price fell today
BUY IF PRICE < PRICE[1]
```

| | |
|---|---|
| Indicators | `PRICE`, `MA`, `EMA`, `STD`, `RSI`, `ZSCORE`, `VOL`, `HIGH`, `LOW`, `RETURN_ND`, `DIST_MA`, `DIST_HIGH`, `DIST_LOW`, `MACD`, `MACD_SIGNAL`, `DRAWDOWN`, with a window, e.g. `MA60` |
| Comparisons | `>` `<` `>=` `<=` `CROSSES_ABOVE` `CROSSES_BELOW` |
| Arithmetic | `+ - * / ^` and brackets |
| Functions | `SIN` `COS` `TAN` `ASIN` `ACOS` `ATAN` `ABS` `SQRT` `LOG` `EXP` |
| Past values | `PRICE[1]` is yesterday, `MA60[2]` is the 60-day average two days ago |

Optional settings: position sizing by an indicator, volatility targeting, a rebalance band, stop loss, take profit, a borrow fee for shorts, carry and interest on cash.

## What you can test

| | |
|---|---|
| One ticker | Any Yahoo Finance symbol, e.g. `EURUSD=X`, `SPY`, `AAPL`. In advanced mode it gets every check |
| A few tickers | Up to 5 are each tested like a single ticker, with a summary table |
| A ready-made list | 7 major forex pairs, all 28 pairs of the 8 main currencies, or the S&P 500 stocks. Each ticker gets one backtest, and the result shows on how many of them the rule beat buy and hold. Any ticker can then be opened for the full checks |
| A portfolio | The same tickers trading out of one account with one pot of starting money |
| Against the S&P 500 | A tick box adds what the same money would have made in an S&P 500 fund (SPY, dividends reinvested) to the numbers and the charts |

Testing a rule on a whole list is a check in itself. A rule with a real edge should work on most tickers of the same kind, not only on the one it was designed on.

### How the money is split

With several tickers there are two ways to run them.

**Each on its own.** Every ticker gets the full starting money and its own result. The tickers do not affect each other.

**One shared portfolio.** One account trades all of them with the same rules, so the starting money is shared. The portfolio is compared with splitting the money equally over the same tickers and never selling. There are two ways to share:

- *An equal slice per ticker.* With 10 tickers each may use 10% of the account. A ticker with no signal leaves its slice in cash, so the portfolio is often only partly invested.
- *Spread over open positions.* The money is shared by the tickers that have a position open, up to a limit per ticker. With 2 signals on and a 20% limit, 40% is invested. With 10 signals on, each gets 10%.

Positions are resized when they drift further from their share than the rebalance band. Sells are done before buys, and a buy never spends more cash than the account holds.

## How a backtest runs

**Prices.** Daily Open, High, Low and Close from Yahoo Finance, with 500 extra days loaded before the start date so long averages are ready on day one. Prices are not adjusted for dividends. They are the prices that were really quoted, so a rule like `PRICE > 105` means what it says, and dividends are paid into the account on the day the share starts trading without them. A long position gets them and puts them back into the shares. A short position owes them.

**Signals and fills.** Signals are calculated on each day's close and orders fill at the next day's open, so a rule never uses a price it could not have known. Every trade pays a cost as a percentage of the traded value. If a short signal arrives while the strategy is long, the position flips directly.

**Stop loss and take profit.** Both are measured from the price the trade was opened at, and both rest in the market as orders. They fill during the day at their own level, or at the open if the price jumped past the level overnight. When one day's range contains both levels, the stop is assumed to come first. After a stop or a take profit closes a trade, the rule has to switch off and on again before it enters in that direction again. Otherwise a rule like `PRICE > MA200` would buy straight back the next morning.

**Interest.** Short positions pay a borrow fee. Carry is interest for holding a position overnight, earned when long and paid when short. For a currency pair it is the first currency's interest rate minus the second's. Money that is not invested can earn interest, and the Sharpe ratio then counts only the return above it. All three are charged per night, so a weekend costs three.

**Forex.** Yahoo's forex history has two faults. Some closes are plainly wrong: EUR/USD shows 1.49 on 8 December 2008, between two days at 1.27 and 1.29. And since about 2011 the open, high and low of a bar belong to the day after the close they are listed with, so "the next open" is really a day late. Only the closes line up over the whole history. For tickers ending in `=X` the backtester therefore:

- removes a close that jumps more than 8 times the usual daily move and is back the next day, and lists what it removed. "Usual" is measured on the days around it, so the real jumps of a crisis are kept.
- rebuilds each bar from closes. A currency trades around the clock, so the last close is the price you can trade at next, and each day opens there.
- checks stops against closes only. A stop still fills at its own level, but a dip that recovered before the close is not seen.

Stock prices come from an exchange and are used as they are.

## Checks

| Check | What it does |
|---|---|
| Train/test split | Rules are chosen on one period and judged on a later one the rules never saw |
| Random benchmark | 300 strategies with the same holding periods, placed at random times. A real edge should beat most of them |
| Sensitivity | Every number in the rules is moved to 0.5x, 0.75x, 1.25x and 1.5x. A real edge survives small changes, a lucky one shows up as a single good cell in a heatmap |
| Number search | Looks for better numbers on the training period only, then reports how they do on the test period |
| Cost check | Reruns without costs to show how much of the return costs took |
| A day late | Reruns with every signal acted on one day later. A result that needs perfect timing would not survive real trading |
| Across tickers | Runs the same rule on a whole list and counts how often it beat buy and hold |

Each backtest ends with a verdict that lists what passed and what failed. A strategy that lost money is marked down even if the market lost more. The cost check and the day-late check run on every backtest. The random benchmark, the sensitivity check and the number search look at one price series, so they run on single tickers and not on portfolios.

## Study: EUR/USD 2005-2026

Every rule was designed on 2005-2015 and tested once on 2016-2026. Costs were 0.02% per trade. Buy and hold lost about 17% over the full period.

The two rules whose text is in this repository were rerun with version 2. The other six results are from version 1 and have not been rerun.

| Strategy | Idea | Result | Engine |
|---|---|---|---|
| 200-day trend | Long above the 200-day average, short below | Held up on EUR/USD. +77%, positive in both periods. [Did not carry over to other pairs](#what-version-2-changed) | v2 |
| Pendulum | Price swings around its average like a pendulum | Failed. [Details](#the-pendulum-model) | v2 |
| Rally then dip | Profit-takers cause a dip, latecomers buy it | Held up. About +31%, positive in both periods | v1 |
| Break-even vs momentum | Size of the drop decides who wins at the old high | Mixed. Roughly flat, lost in training, gained in the test | v1 |
| Short-term reversal | Buy after a down day, sell after an up day | Failed. About -48%, trades almost daily so costs dominate | v1 |
| Failed breakouts | Bet against breakouts that reverse within days | Failed. About -42%, every nearby setting lost too | v1 |
| Equilibrium mean reversion | Bet on a return to the average | Failed on 2005-2015, not tested further | v1 |
| Break-even sellers | Old highs act as resistance, old lows as support | Failed on 2005-2015, not tested further | v1 |

The 200-day rule is a classic trend rule, and research found trend rules profitable on currencies in the 1980s and 90s. It was fixed before looking at any data. Rally then dip was designed for this study, and its sensitivity heatmap was green for every nearby setting.

EUR/USD was chosen because it is the most traded currency pair, costs are low, and its price is not affected by earnings, dividends or stock splits. Many large traders in currencies, such as central banks and companies hedging payments, trade for reasons other than profit, which is where an edge would have to come from.

### What version 2 changed

The same two rules on the same dates, with the old engine and the new one:

| EUR/USD, 2005-2026 | Version 1 | Version 2 | Version 2, filled a day late |
|---|---|---|---|
| 200-day trend | +26.7% | +77.0% | +30.5% |
| Pendulum | -15.5% | -4.9% | -9.2% |

Two things moved the numbers. Five bad quotes in 2008 were removed, each of which had caused false signals. And orders now fill at the close the signal was read from, where version 1 used Yahoo's open, which after 2010 is a full day later.

The last column is the important one. The trend rule earns 77% if it trades at the close that gave the signal and 30% one day later. It beats buy and hold either way, but more than half of the return depends on trading at the very close that gave the signal. That is possible in a market open around the clock, and it is also the most favourable assumption a backtest can make.

Version 1 ended with "the next test is running both, unchanged, on other currency pairs". Version 2 can do that in one run. The 200-day rule, unchanged, on all 28 pairs of the 8 main currencies:

| 200-day trend, 2005-2026 | Result |
|---|---|
| Made money | 3 of 28 pairs |
| Beat buy and hold | 7 of 28 pairs |
| Median return | -38.0%, against -7.2% for buy and hold |
| As one portfolio of 28 | -24.1%, against -0.9% for holding all 28 |
| As one portfolio of the 7 majors | -8.1%, against -4.6% for holding all 7 |

EUR/USD was the best of the 28. The rule that looked like the strongest result of the study was most likely the one lucky pair, which is what the multiple testing section below warns about.

### The pendulum model

The pendulum model failed. On EUR/USD it lost money and did no better than random timing.

| | Pendulum | Buy and hold |
|---|---|---|
| Total return | -4.9% | -17.1% |
| Sharpe | 0.01 | -0.04 |
| Max drawdown | -42.9% | -40.0% |
| Trades | 96 | - |

The model treats the 60-day average as the bottom of a pendulum's swing and the price as the pendulum. The angle is the distance from the average in standard deviations, and the velocity is its change since yesterday:

$$\theta = \frac{P - \mathrm{MA}_{60}}{\mathrm{STD}_{60}} \qquad v = \theta_t - \theta_{t-1}$$

For small swings the pendulum equation becomes linear and the energy is constant:

$$\frac{d^2\theta}{dt^2} = -\frac{g}{L}\sin\theta \approx -\omega^2\theta \qquad \frac{1}{2}v^2 + \frac{1}{2}\omega^2\theta^2 = \text{constant}$$

So the amplitude, how far the swing goes before it turns, can be calculated from today's position and velocity:

$$A = \sqrt{\theta^2 + \frac{v^2}{\omega^2}}$$

With ω² = 0.01 a full swing takes about 63 days. The strategy buys at the bottom of a swing below θ = -1.5 and sells when θ reaches 0.8A on the other side. The factor 0.8 accounts for friction. If θ passes 3, the average has probably moved and the trade is closed. Shorts are the mirror image. The rules contain the formulas directly:

```
BUY  IF (PRICE[1] - MA60[1]) / STD60[1] < (PRICE[2] - MA60[2]) / STD60[2]
     AND (PRICE - MA60) / STD60 > (PRICE[1] - MA60[1]) / STD60[1]
     AND (PRICE - MA60) / STD60 < -1.5
SELL IF (PRICE - MA60) / STD60 > 0.8 * SQRT(((PRICE - MA60) / STD60) ^ 2
        + ((PRICE - MA60) / STD60 - (PRICE[1] - MA60[1]) / STD60[1]) ^ 2 / 0.01)
     OR (PRICE - MA60) / STD60 < -3
```

On simulated prices that swing like a pendulum, the same rules beat all 300 random strategies with a Sharpe ratio of about 4. On simulated random prices they found nothing. The test can therefore detect a pendulum when one exists.

On EUR/USD the verdict was:

```
Verdict: mixed
  Beat buy & hold (-4.9% vs -17.1%)
  Better Sharpe (0.01 vs -0.04)
  Lost money (-4.9%)
  Costs took 3.8 points of return
  Filled a day late it makes -9.2% instead of -4.9%
  Beat buy & hold in both the train and test period
  Beat 54% of 300 random strategies
  71% of 28 nearby settings beat buy & hold
```

It lost less than buy and hold in 2005-2015 (-15.9% against -19.4%) and made more in 2016-2026 (+13.0% against +2.9%). That reads better than it is. Over 21 years it lost money, its Sharpe ratio is 0.01, and it beat only 54% of strategies that held for the same stretches at random times. Losing less than a falling market is not an edge. The worst trades were longs held for 4 to 8 months through long declines. In a slow decline the average falls with the price, so the price probably never looks far enough from equilibrium to trigger an exit. The number search found a 48-day window that made +27.7% over the full period, and on 2016-2026 alone it was about level with the original (Sharpe 0.23 against 0.20).

On all 28 currency pairs the pendulum made money on 13 and lost on 15, with a median return of -1.4%. As one portfolio of 28 it made +6.8% in 21 years, a Sharpe ratio of 0.09.

This is the second version of the model. The first used the full equation with sin and arccos, which treats θ as a real angle even though it is measured in standard deviations, and it assumed the market conserves energy. With the version 1 engine it lost 23.1% against 16.6% for buy and hold and beat 25% of random strategies. The model was rewritten because the physics was wrong, not to improve the result, so the second version also had a single test.

### Multiple testing

Testing many strategies on the same data produces winners by luck. If a useless rule has a 50% chance of beating buy and hold in a period, and the two periods are independent:

$$P(\text{at least one of } n \text{ wins one period}) = 1 - 0.5^n \qquad P(\text{at least one of } n \text{ wins both periods}) = 1 - 0.75^n$$

| Rules tested | At least one wins one period | At least one wins both periods | Expected winners in both |
|---|---|---|---|
| 1 | 50% | 25% | 0.25 |
| 8 | 99.6% | 90.0% | 2.0 |
| 20 | >99.99% | 99.7% | 5.0 |

Two periods reduce the problem but do not solve it. Eight useless rules would produce two winners in both periods on average, the same number this study found. That is why the random benchmark, the sensitivity check and the fixed 200-day rule matter more than the split itself, and why testing on currency pairs that were not used to design the rules mattered most of all: it is the test the 200-day rule failed.

## Example: the S&P 500 list

The 200-day rule, long only, on today's S&P 500 stocks from 2005 to 2026, with 0.05% costs:

| | Result |
|---|---|
| Beat buy and hold on return | 51 of 502 stocks |
| Better Sharpe than buy and hold | 87 of 502 |
| Smaller max drawdown | 406 of 502 |
| As one portfolio, equal slices | +410%, Sharpe 0.94, max drawdown -19% |
| Holding all 502 equally | +2,052%, Sharpe 0.86, max drawdown -45% |
| S&P 500 fund (SPY) | +849%, Sharpe 0.65, max drawdown -55% |

The rule gives up most of the return for a much smaller drawdown. The middle row needs care: holding today's 502 members made 2,052% while the index itself made 849%. The gap is survivorship. The list holds the companies that are in the index today, which are the ones that did well, and leaves out those that went bust or were dropped. Every number computed on this list is better than it could have been in real time, so compare a rule with holding the same stocks, not with the index.

## Project structure

```
backtester.py              engine: data cleaning, indicators, rule parser, backtest, checks, charts
app.py                     Streamlit web app (built with Claude Code)
Backtester.ipynb           notebook version, with the full pendulum study and its output
data/sp500.csv             S&P 500 members on 8 October 2026, from Wikipedia
tests/test_backtester.py   engine tests on made-up prices, run with python -m pytest
CHANGELOG.md               what changed between versions
requirements.txt
```

## Limitations

- Daily data only. Within a day the backtester knows the open, high, low and close, not the order they came in.
- Forex is simulated on closes only, so a stop that was touched during the day and recovered is missed. Forex results also depend strongly on fill timing, which is why the day-late check exists.
- Carry and the interest on cash are one fixed rate for the whole test. Real interest rates moved between 0% and 5% over 2005-2026. Both are 0 unless you set them, so forex results leave out interest rate differences by default.
- The S&P 500 list is today's members, so results on it are too good. See the example above.
- Costs are a flat percentage, not real bid/ask spreads, and there is no limit on how much can be traded at the quoted price.
- The bad quote filter only runs on forex. A wrong price in a stock's history would be used as it is.
- A portfolio runs the same rules on every ticker. It does not rank tickers against each other.

Not financial advice. MIT license.
