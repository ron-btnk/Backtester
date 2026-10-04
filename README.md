# Backtester

I built a backtesting engine where you write trading rules as text and test them on any ticker from Yahoo Finance. Try it in your browser: https://backtester-rbtnk.streamlit.app/

I tested some of my own strategies on EUR/USD. [Why EUR/USD?](#why-i-chose-to-focus-on-forexeurusdx)

I tested 8 strategies, 2 of which worked. [How I dealt with the multiple testing problem](#how-i-dealt-with-the-multiple-testing-problem)

More on the strategies:

| Strategy | Idea | Result |
|---|---|---|
| 200-day trend | Long above the 200-day average, short below | Worked. About +27%, positive in both periods |
| Rally then dip | Profit-takers cause a dip, latecomers buy it | Worked. About +31%, positive in both periods |
| Break-even vs momentum | Size of the drop decides who wins at the old high | Mixed. Roughly flat, lost in training, gained in the test |
| Pendulum | Price swings around its average like a pendulum | Failed. [Details](#the-pendulum-model) |
| Short-term reversal | Buy after a down day, sell after an up day | Failed. About -48%, trades almost daily so costs kill it |
| Failed breakouts | Bet against breakouts that reverse within days | Failed. About -42%, every nearby setting lost too |
| Equilibrium mean reversion | Bet on a return to the average | Failed on 2005-2015 |
| Break-even sellers | Old highs act as resistance, old lows as support | Failed on 2005-2015 |

I designed every rule on 2005-2015 and tested it on 2016-2026. Buy and hold lost about 17% over the full 2005-2026 period. The last two failed in 2005-2015, so they never got to the test. [More on the two that worked](#the-two-that-worked)

## The pendulum model

It failed. I treated EUR/USD like a pendulum swinging around its 60-day average and wrote the physics into the trading rules as formulas. It did about as well as buy and hold and no better than random timing. [Why a pendulum?](#why-a-pendulum)

| | Pendulum | Buy and hold |
|---|---|---|
| Total return | -15.8% | -17.1% |
| Sharpe | -0.03 | -0.02 |
| Max drawdown | -41.8% | -40.0% |
| Trades | 96 | - |

The angle is how far price is from its 60-day average in standard deviations, and the velocity is how much it changed since yesterday:

$$\theta = \frac{P - \mathrm{MA}_{60}}{\mathrm{STD}_{60}} \qquad v = \theta_t - \theta_{t-1}$$

For small swings the pendulum equation simplifies, and its energy stays constant:

$$\frac{d^2\theta}{dt^2} = -\frac{g}{L}\sin\theta \approx -\omega^2\theta \qquad \frac{1}{2}v^2 + \frac{1}{2}\omega^2\theta^2 = \text{constant}$$

That gives the amplitude A, how far the swing goes before it turns:

$$A = \sqrt{\theta^2 + \frac{v^2}{\omega^2}}$$

I set ω² = 0.01, a swing of about 63 days. The rule buys at the bottom of a swing (θ below -1.5) and sells when θ reaches 0.8A on the other side. The 0.8 is friction. If θ goes past 3 the average has probably moved, so the trade closes. Shorts are the mirror image. [The full rules and what went wrong](#the-pendulum-in-detail)

## The pendulum in detail

The formulas are typed straight into the rules:

```
BUY   IF (PRICE[1] - MA60[1]) / STD60[1] < (PRICE[2] - MA60[2]) / STD60[2]
      AND (PRICE - MA60) / STD60 > (PRICE[1] - MA60[1]) / STD60[1]
      AND (PRICE - MA60) / STD60 < -1.5
SELL  IF (PRICE - MA60) / STD60 > 0.8 * SQRT(((PRICE - MA60) / STD60) ^ 2
         + ((PRICE - MA60) / STD60 - (PRICE[1] - MA60[1]) / STD60[1]) ^ 2 / 0.01)
      OR (PRICE - MA60) / STD60 < -3
```

On fake prices that swing like a pendulum, this rule beat all 300 random strategies with a Sharpe ratio of about 4. On random prices it found nothing, so the test can tell a pendulum from noise. On EUR/USD it beat 44% of 300 random strategies, and 29% of 28 nearby settings beat buy and hold.

It lost to buy and hold in 2005-2015 (Sharpe -0.13 against -0.07) and beat it in 2016-2026 (0.16 against 0.08). A rule that wins in one period and loses in the other looks like noise. It won 55% of its trades but lost money overall, with small wins and a few large losses. The worst trades were longs held for 4 to 8 months through long EUR/USD declines. In a slow decline the average falls with the price, so the pendulum probably never looks far enough from equilibrium to exit.

The automatic search found a 48-day window that made +17.9% over the full period. On 2016-2026 alone it did worse than the original (Sharpe -0.03 against 0.16), so it was fitting the past.

This is the second version. The first used the full equation with sin and arccos, which treats θ as an angle, but θ is measured in standard deviations, so that was an arbitrary choice. It also assumed the market conserves energy. That version lost 23.1% against 16.6% for buy and hold and beat 25% of random strategies. I rewrote it after the physics was criticised, not to improve the result, so the new version had one test of its own. Neither held up.

## The two that worked

The 200-day rule is a classic trend rule, and studies of currencies in the 1980s and 90s found trend rules profitable. I fixed it before looking at any data, so it counts as one test and not as part of a search. Rally then dip is my own idea. Its sensitivity heatmap is green for every nearby setting, so it does not depend on exact numbers. For break-even vs momentum I saw the full period while building it, but I did not change the rules afterwards. Next I will run both survivors, unchanged, on other currency pairs.

## How the tests are built

Signals use the close and trades happen at the next day's open, so there is no lookahead. Every trade pays a cost, and shorts pay a borrow fee. Rules are chosen on one period and tested on another. Each strategy is compared with 300 random ones that spend the same time long and short. Nearby settings are tested too, so a lucky spike shows up as one green square in a red heatmap.

## Why I chose to focus on Forex/EURUSD=X

EUR/USD is the most traded currency pair in the world. Trading it costs very little, so I assumed 0.02% per trade. There are 21 years of daily data and no earnings, dividends or stock splits, so a rule depends on price only. Central banks and companies hedging future payments trade currencies for reasons other than profit. If an edge exists it would come from them, which gives every idea a question to answer: who is on the other side of the trade, and why do they lose? Research also found that trend rules worked on currencies in the 1980s and 90s, so I could test a classic one, the 200-day rule, as a check on my own ideas.

## Why a pendulum

Looking at EUR/USD charts, price seemed to swing around a level and then go about as far on the other side. A pendulum does the same around its lowest point, and its equations predict where a swing turns. So I used them as the trading rule, with the 60-day average as the lowest point.

## How I dealt with the multiple testing problem

If you test enough strategies on the same data, some will work by luck. This is the multiple testing problem, also called data snooping.

Say a useless rule has a 50% chance of beating buy and hold in one period, and the two periods are independent. This is a simplification, so the numbers show the size of the effect and not exact odds.

$$P(\text{at least one of } n \text{ useless rules wins one period}) = 1 - 0.5^n$$

$$P(\text{one useless rule wins both periods}) = 0.5 \times 0.5 = 0.25$$

$$P(\text{at least one of } n \text{ useless rules wins both periods}) = 1 - 0.75^n$$

| Rules tested | At least one wins one period | At least one wins both periods | Expected winners in both |
|---|---|---|---|
| 1 | 50% | 25% | 0.25 |
| 8 | 99.6% | 90.0% | 2.0 |
| 10 | 99.9% | 94.4% | 2.5 |
| 20 | >99.99% | 99.7% | 5.0 |

Splitting the data into two periods helps a bit. With 8 useless rules, at least one still passes both periods 90% of the time. The expected number is 8 x 0.25 = 2, and I found 2. Getting 2 or more out of 8 happens about 63% of the time by luck, so passing both periods is not proof on its own.

On top of the split I used the random benchmark and the sensitivity heatmap ([how the tests are built](#how-the-tests-are-built)). The 200-day rule was fixed in advance, so its chance of passing both periods by luck is 25%, not 90%. The strongest test still to do is other currency pairs, because a real effect should show up on data I did not design the rule on.

## Write your own rules

Quick mode asks for tickers, dates, rules and cost. Advanced mode adds every setting, plus the train/test split, random benchmark, sensitivity check and a search for better numbers.

```
BUY IF PRICE > MA200
SELL IF RSI14 > 70
BUY IF (PRICE - MA20) / STD20 < -2 AND PRICE > MA200
SELL IF MA20 CROSSES_BELOW MA50
SHORT IF ZSCORE20 > 2
COVER IF ZSCORE20 < 0
```

Indicators are PRICE, MA, EMA, STD, RSI, ZSCORE, VOL, HIGH, LOW, RETURN_ND, DIST_MA, DIST_HIGH and DIST_LOW, with a window added, for example MA60. MACD, MACD_SIGNAL and DRAWDOWN take no window. `PRICE[1]` is yesterday's price and `PRICE[2]` the day before. Formulas can use `+ - * / ^`, brackets, and SIN, COS, TAN, ASIN, ACOS, ATAN, ABS, SQRT, LOG and EXP. Conditions combine with AND / OR. Position sizing, a volatility target, stop loss and take profit are optional.

## Run it yourself

**Online:** https://backtester-rbtnk.streamlit.app/. Nothing to install, works on a phone. It runs on a small free server, so the heavy checks are slower, and it sleeps when nobody uses it, so the first load can take a minute.

**Web app on your computer:** faster, and it never sleeps.

```
cd ~\Desktop
git clone https://github.com/ron-btnk/Backtester
cd Backtester
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**Notebook:** same first four lines, then `python -m notebook Backtester.ipynb`. Run all cells, then press Enter for the pendulum or type C for your own rules. All the code is visible and editable.

The engine is in `backtester.py` and the web page in `app.py`. I built the web front end with Claude Code.

## Limitations

Daily data only. Yahoo's FX open prices are not exact. Interest rates are ignored, and holding euros against dollars earns different rates, which matters when shorting FX. Costs are a flat estimate and not real spreads. Only one currency pair so far.

Not financial advice.
