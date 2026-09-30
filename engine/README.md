# Trading engine (Rust)

High-speed data ingestion, feature engineering, model inference and order execution.

Training lives in Python (`../research/train.py`). Everything on the hot path lives
here. The two halves meet at two files: `models/<SYM>.json` (the trained model) and
`models/parity.json` (the proof that both halves compute the same numbers).

## Layout

```
crates/core    domain types: bars, orders, fills, positions, account, cash settlement
crates/feed    ingestion: CSV reader, deterministic replay, external-command feed
crates/model   indicators, feature engineering, model inference (logistic/linear/MLP)
crates/exec    risk checks, order lifecycle, matching engine, latency measurement
crates/cli     the `bot` binary
```

## Build

```bash
cd engine
cargo build --release
cargo test --release --workspace
```

First build downloads serde. Everything else is the standard library. No TLS stack,
no async runtime, no HTTP client.

## Run

```bash
./target/release/bot ingest   data/SPY.csv
./target/release/bot features data/SPY.csv models/SPY.json
./target/release/bot backtest data/SPY.csv models/SPY.json --thresh 0.52
./target/release/bot bench    data/SPY.csv models/SPY.json
```

## Measured performance

Not claimed - printed by `bot bench` on this machine. Reproduce it yourself.

```
ingestion        2.2 - 3.0 M bars/s       120 - 164 MB/s
features         ~5.5 microseconds/bar    ~175 K bars/s
inference        46 ns/prediction         ~22 M predictions/s
order handling   87 ns mean   p50 100 ns  p99 200 ns    7.36 M orders/s
end to end       5.3 microseconds/bar
```

One honest caveat: the latency maximum is ~1.7 ms, from the first call. Every
percentile figure is from the steady state.

## Two rules the engine is built around

**No lookahead.** A market order sent after a bar closes fills at the OPEN of a later
bar - `latency_bars` later, default 1 - never at a price that was already visible when
the strategy decided. `fills_at_the_next_bar_open_not_the_deciding_close` tests this
directly.

**Latency is measured, not asserted.** `ExecutionEngine::send` times the work it owns -
risk check plus venue submit - and `Metrics` reports real percentiles from samples.
Network time is not included and is not claimed.

## The Python/Rust parity problem

Feature engineering exists twice, in two languages. If they drift, Rust feeds the model
inputs it was never trained on: every prediction becomes wrong and every test still
passes.

`crates/model/tests/parity.rs` is the guard. The trainer exports the bars it used, the
features it computed and the probabilities it produced; Rust replays the identical
stream and fails on any disagreement.

```
parity: ETH-USD on 95 rows across 731 bars
        worst feature delta      4.686e-11
        worst probability delta  6.301e-13
```

Two things make this exact rather than approximate:

- **The streaming window is longer than any series** (1024 bars). EMA-based indicators
  are seeded with a simple average of their first `n` values and that seed decays only
  as `(1-k)^bars`. With a short window the seed still influenced the result at the 1e-2
  level, which is far too much to agree with a trainer that saw the full history.
- **The parity file carries the whole series from bar 0**, so both sides seed their
  EMAs from the same point.

## A performance bug worth recording

`macd_hist` was first written as `ema(&x[..i], ..)` inside a loop, recomputing the
entire EMA for every prefix. That is O(n^2) per call, and it is called once per bar. It
measured **112 microseconds per bar** - not high-speed by any definition.

Rewriting it as a single pass made it **5.0 microseconds per bar: 22.6x faster**, with
the parity delta unchanged at 4.686e-11, which is what proves the rewrite is
mathematically identical and not merely close.

## Live venues

`Exchange` is a trait; `SimExchange` implements it deterministically and is what the
tests exercise. For live data, `CommandFeed` runs an external process and parses its
stdout as bars:

```bash
bot ingest <(curl -s "https://venue/api/klines?symbol=BTCUSDT&interval=1m")
```

That keeps TLS out of the engine and makes the venue-specific part one command line
rather than code. **No live venue adapter is shipped**, because Binance and Coinbase
are both unreachable from the machine this was built on, and an untested adapter would
be worse than none.

## Real-money safety

The engine is built so that going live is a deliberate act, not a default:

- `RiskLimits::kill_switch` refuses every order when set; `ExecutionEngine::kill()`
  sets it.
- Pre-trade checks cover order size, resulting position, gross exposure, an
  insufficient-cash check, a fat-finger band on limit prices, and a daily-loss circuit
  breaker.
- `SimConfig` defaults to 1 bar of latency, 5 bps commission and 2 bps slippage, so a
  result cannot be produced without costs unless you deliberately set them to zero.
- Nothing here places a real order. Wiring a venue means implementing `Exchange`.

## What the engine finds

The same conclusion as the browser backtester, reached independently:

```
symbol    strategy    buy & hold    excess
SPY          +0.67%      +27.09%    -26.42%
AAPL        -48.87%      +27.58%    -76.44%
NVDA        +42.52%      +62.04%    -19.52%
BTC-USD     -36.45%      -13.94%    -22.51%
TLT         -10.92%      -10.97%     +0.05%
```

And the trainer's own out-of-sample accuracy, over ten symbols: **-2.5 points against a
coin flip** on average, with in-sample accuracy far higher than out-of-sample. That gap
is overfitting, and it is reported rather than hidden.
