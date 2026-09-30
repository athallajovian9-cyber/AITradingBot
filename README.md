# AI Trading Bot — Advanced Honest Backtester

A trading research tool that tells you the truth, built to replace a version that
claimed to be an AI and was not.

## The one rule

**Nothing here predicts the future. It measures the past.**

Every result is shown next to buy-and-hold on the same capital, the same bar and the
same costs. Costs are charged on every fill. Where a result cannot be trusted, the
tool says so rather than dressing it up.

## Two halves

This is the split the project was built around: **research in Python, the hot path in
Rust**, and a browser tool for looking at the whole thing.

```
research/train.py      trains the models (numpy only) and exports them
engine/                Rust: ingestion, features, inference, order execution
AITradingBot.html      the browser research tool - ten assets, six tabs
```

The two halves meet at two files: `engine/models/<SYM>.json` (the trained model) and
`engine/models/parity.json` (the proof that both compute the same numbers).

## Play (the browser tool)

Double-click **`AITradingBot.exe`**. No installer, no setup, works offline.

Or run **`LAUNCHER.bat`**, which finds Chrome or Edge and opens the same page in a
borderless window. Either way there are no external requests — the page is one file
with its data and its engine inside it.

**Ten real assets, full OHLCV** — S&P 500, Nasdaq 100, Apple, Microsoft, NVIDIA,
Tesla, Gold, 20-year Treasuries, Bitcoin, Ethereum. Two years of daily bars, fetched
at build time and embedded. TLT fell 20% over the window; NVDA rose 87%.

**Real indicators** — Wilder RSI(14), Wilder ATR(14), MACD(12,26,9), Bollinger %B,
realised volatility, volume z-scores, trend spreads. RSI is checked in the test suite
against the published Wilder worked example.

**Real models** — logistic regression trained online with L2 regularisation, linear
regression, and **a neural net with a hidden layer and backpropagation** whose weights
are shown as they update and which is tested on XOR, which a linear model provably
cannot solve.

**Real risk controls** — full capital, volatility targeting, or half-Kelly sizing; ATR
stop-loss, ATR trailing stop, ATR take-profit, and a maximum holding period.

**Real costs** — commission, spread, slippage, and optional market impact that scales
with order size.

### The six tabs

**Backtest** — one asset, one configuration. Twenty metrics including Sharpe, Sortino,
Calmar, beta, annualised alpha, information ratio, profit factor, exposure, VaR 95%,
max drawdown and how long it lasted.

**Walk-forward** — the honest test. Repeatedly fit on a window of the past, then trade
the *next* window untouched, with an **embargo** of at least the forecast horizon so no
label leaks across the boundary. Everything on the resulting curve is out-of-sample.

**Portfolio** — run the strategy across several assets at once, capital split equally,
against an equal-weight buy-and-hold benchmark.

**Optimise** — grid-search every setting in-sample, then re-run each one
out-of-sample. **The gap between the two is the overfitting, and it is the headline.**

**Monte Carlo** — resample the strategy's own daily returns thousands of times to show
how much of a result is just the order things happened in. Seed-fixed, so it reproduces.

**Models** — logistic vs linear vs neural net on identical data and costs. A more
complex model is not automatically better, and this is where you see that.

## The Rust engine

High-speed data ingestion, feature engineering, model inference and order execution.

```
crates/core    domain types: bars, orders, fills, positions, account, cash settlement
crates/feed    ingestion: CSV reader, deterministic replay, external-command feed
crates/model   indicators, feature engineering, model inference (logistic/linear/MLP)
crates/exec    risk checks, order lifecycle, matching engine, latency measurement
crates/cli     the `bot` binary
```

### Build and run

```bash
cd engine
cargo build --release
cargo test --release --workspace

./target/release/bot ingest   data/SPY.csv
./target/release/bot features data/SPY.csv models/SPY.json
./target/release/bot backtest data/SPY.csv models/SPY.json --thresh 0.52
./target/release/bot bench    data/SPY.csv models/SPY.json
```

First build downloads serde. Everything else is the standard library. No TLS stack, no
async runtime, no HTTP client.

### Measured performance

Not claimed — printed by `bot bench`. Reproduce it yourself.

```
ingestion        2.2 - 3.0 M bars/s       120 - 164 MB/s
features         ~5.5 microseconds/bar    ~175 K bars/s
inference        46 ns/prediction         ~22 M predictions/s
order handling   87 ns mean   p50 100 ns  p99 200 ns    7.36 M orders/s
end to end       5.3 microseconds/bar
```

One honest caveat: the latency maximum is ~1.7 ms, from the first call. Every
percentile figure is from the steady state.

### Two rules the engine is built around

**No lookahead.** A market order sent after a bar closes fills at the OPEN of a later
bar — `latency_bars` later, default 1 — never at a price that was already visible when
the strategy decided. This is tested directly.

**Latency is measured, not asserted.** `ExecutionEngine::send` times the work it owns —
risk check plus venue submit — and reports real percentiles from samples. Network time
is not included and is not claimed.

### The Python/Rust parity problem

Feature engineering exists twice, in two languages. If they drift, Rust feeds the model
inputs it was never trained on: every prediction becomes wrong and every test still
passes. `crates/model/tests/parity.rs` is the guard — the trainer exports the bars it
used, the features it computed and the probabilities it produced, and Rust replays the
identical stream and fails on any disagreement.

```
parity: ETH-USD on 95 rows across 731 bars
        worst feature delta      4.686e-11
        worst probability delta  6.301e-13
```

Two things make this exact rather than approximate: the streaming window is longer than
any series (1024 bars), so the EMA seeding matches the trainer's rather than
approximating it; and the parity file carries the whole series from bar 0, so both
sides seed from the same point.

### A performance bug worth recording

`macd_hist` was first written as `ema(&x[..i], ..)` inside a loop, recomputing the
entire EMA for every prefix. That is O(n²) per call, and it is called once per bar. It
measured **112 microseconds per bar** — not high-speed by any definition. Rewriting it
as a single pass made it **5.0 microseconds: 22.6× faster**, with the parity delta
unchanged at 4.686e-11, which is what proves the rewrite is mathematically identical
rather than merely close.

### Live venues

`Exchange` is a trait; `SimExchange` implements it deterministically and is what the
tests exercise. For live data, `CommandFeed` runs an external process and parses its
stdout as bars:

```bash
bot ingest <(curl -s "https://venue/api/klines?symbol=BTCUSDT&interval=1m")
```

That keeps TLS out of the engine and makes the venue-specific part one command line
rather than code. **No live venue adapter ships**, because Binance and Coinbase are both
unreachable from the machine this was built on, and an untested adapter would be worse
than none.

### Real-money safety

Going live is a deliberate act, not a default:

- `RiskLimits::kill_switch` refuses every order when set; `ExecutionEngine::kill()` sets it
- pre-trade checks cover order size, resulting position, gross exposure, insufficient
  cash, a fat-finger band on limit prices, and a daily-loss circuit breaker
- `SimConfig` defaults to 1 bar of latency, 5 bps commission and 2 bps slippage, so a
  result cannot be produced without costs unless you deliberately set them to zero
- nothing here places a real order. Wiring a venue means implementing `Exchange`

## What it actually finds

The same conclusion from both halves, reached independently:

```
browser tool   walk-forward beat buy-and-hold in a minority of assets
               portfolios track their benchmark more closely than any single
               asset tracks its own
               the best in-sample setting shows a large negative gap out-of-sample
               the spread across resampled paths is wide enough that a single
               good-looking run proves very little

rust engine    symbol    strategy    buy & hold    excess
               SPY         +0.67%      +27.09%    -26.42%
               AAPL       -48.87%      +27.58%    -76.44%
               NVDA       +42.52%      +62.04%    -19.52%
               BTC-USD    -36.45%      -13.94%    -22.51%
               TLT        -10.92%      -10.97%     +0.05%

trainer        out-of-sample accuracy over ten symbols: -2.5 points against a
               coin flip on average, with in-sample accuracy far higher. That
               gap is overfitting, reported rather than hidden.
```

On the earlier four-asset set, the horizon sweep is worth keeping on record:

```
horizon  1   beats buy-and-hold 0/4   mean excess -24.9%
horizon  5   beats buy-and-hold 0/4   mean excess -14.4%
horizon 10   beats buy-and-hold 1/4   mean excess -10.8%
horizon 20   beats buy-and-hold 0/4   mean excess -31.5%
horizon 40   beats buy-and-hold 0/4   mean excess -31.7%
```

There **is** a real statistical signal at a 10–20 bar horizon (correlation ≈ 0.12
between the model's output and realised forward returns). It is still not enough: the
edge is smaller than the round-trip costs plus the drift given up while out of the
market. One bar ahead the signal is slightly **anti**-correlated — and that is the
horizon the original version traded on.

None of that is a bug. It is what happens when a simple technical strategy is measured
honestly, and it is exactly what the version this replaces was hiding.

## Why the old one was not an AI

For the record, since this replaces it:

```
"Primary Signal Indicator" (RSI / MACD / MA)
    read into a variable used ONLY inside a log string.
    All three options were the same code path.

"Learning Rate (AI Weighting)"
    appeared once, as Math.sin(newPrice * lr).
    A sine frequency. No weights, no epochs, no training.

The "signal"
    Math.sin(price x lr) + random()
```

It reported a 48% hit rate, beat buy-and-hold in 20 of 400 runs, and looked profitable
only because its price generator had a built-in upward drift — any long-only strategy
ends green on a rising series, and the bot took the credit.

## Verify it yourself

```bash
# the browser engine
cd assets
node test_engine2.js   # 74 checks: indicators vs published values, XOR, no lookahead,
                       #            costs, sizing, stops, metrics, Monte Carlo, portfolio
node test_page2.js     # 35 checks: the built page runs the tested engine, every tab
                       #            works on every asset, and the wording never overclaims
node report2.js        # walk-forward, portfolio, overfit gap and Monte Carlo on real data

# the rust engine
cd ../engine
cargo test --release --workspace    # 60 checks, including the python parity test

# retrain from scratch
python3 ../research/train.py        # needs numpy only
```

The decisive test in every suite is the lookahead check: truncate the price series and
confirm that earlier results do not change when the future exists. A backtest that
fails it is fiction.

## Sources

All assets: Yahoo Finance, daily bars with full OHLCV, fetched at build time and
embedded. Reported in the UI. To refresh, re-fetch `assets/market2.json` and rebuild
with `python3 assets/build_app2.py`.
