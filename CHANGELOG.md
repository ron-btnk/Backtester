# Changelog

Versions are numbered major.minor.patch. The first number changes when results change or old settings stop working, the second when something is added, the third for fixes.

## 2.0.0 - 2026-10-08

Version 2 makes the simulation closer to real trading and adds testing on many tickers. Results differ from version 1, mostly on forex and wherever a stop loss or take profit is used. On clean prices with no stops and no dividends, version 2 gives exactly the same account as version 1.

### Realism

- **Bad forex quotes are removed.** A close that jumps more than 8 times the usual daily move and is back the next day never traded. EUR/USD had five of these in 2008, and the 28 forex pairs had 23 between them. What was removed is listed with the result.
- **Forex orders fill at the close the signal was read from.** Since about 2011 Yahoo lists a forex bar's open, high and low with the wrong day, so "the next open" was a full day late for half the history and on time for the other half. Forex bars are now rebuilt from closes.
- **Stops and take profits fill during the day.** They rest in the market as orders and fill at their own level, or at the open if the price jumped past it overnight. Before, they were checked on the close and filled the next morning, so a 5% stop could lose 7%.
- **No buying straight back after a stop.** After a stop or take profit, the rule has to switch off and on again before it enters in that direction again. Before, a rule like `PRICE > MA200` re-entered the next day and the stop did almost nothing.
- **Stops are measured from the price the trade was opened at**, not the average price after resizing.
- **Dividends are paid into the account.** Prices are no longer adjusted for dividends, so `PRICE > 105` compares against the price that was really quoted. Longs receive dividends and reinvest them, shorts owe them, and buy and hold reinvests them.
- **Carry and interest on cash.** Two new settings, both 0 by default. Carry is earned on long positions and paid on short ones. Idle cash can earn interest, and Sharpe then counts only the return above it.
- **Fees are charged per night**, so a weekend costs three nights. Before, every bar cost one trading day.
- **Every backtest is rerun with fills a day late**, and the verdict reports it. A result that needs perfect timing is marked down.
- **The verdict marks down a strategy that lost money**, even when buy and hold lost more.

### Added

- **Portfolios.** Several tickers can trade out of one account that shares the starting money. The portfolio is compared with holding all its tickers equally.
- **The app decides the split.** By default a portfolio gives more to the tickers where the rule's earlier signals paid off more often (the Kelly formula at half strength, with each ticker's record mixed with the record of all of them), less to positions that move together, and never more than the account holds. Equal slices and an equal spread over open positions are still there in advanced mode, and every result shows what equal slices would have made.
- **Ready-made lists**: 7 major forex pairs, all 28 pairs of the 8 main currencies, and the S&P 500 stocks. With more than 5 tickers each gets one quick backtest, the result shows on how many the rule beat buy and hold, and any ticker can be opened for the full checks.
- **Compare to the S&P 500.** A tick box adds an S&P 500 fund (SPY, dividends reinvested) to the numbers and charts.
- Prices for many tickers load in batches with a progress bar.
- `__version__`, shown in the app and the notebook.

### Changed for anyone using `backtester.py` directly

- `run_backtest` is now two steps: `trade_plan` (what the rules want to hold) and `simulate` (the money). It takes the same arguments as before, plus `carry_pct`, `cash_rate_pct` and `delay_days`.
- New: `run_portfolio`, `analyse_portfolio`, `scan_tickers`, `load_many`, `clean_prices`, `compare_sp500`, `universe`, `expected_edge`. `run_portfolio` and `simulate` take `allocation="smart"`, `"equal"` or `"spread"`.
- `load_prices` returns unadjusted prices with a `Dividends` column.
- `random_benchmark` reads the asset's returns from `result["buy_hold"]`, so dividends are included.

## 1.0.0 - 2026-10-08

The backtester as it was before version 2, tagged `v1.0.0`: one ticker at a time, fills at the next open, stops checked on the close, dividend-adjusted prices.
