//! Command-line front end for the engine.
//!
//! Four things it can do:
//!
//!   ingest   <csv>                     parse bars, report throughput
//!   features <csv> [model.json]        stream bars, print the latest feature vector
//!   backtest <csv> <model.json>        full loop: features -> signal -> order -> fill
//!   bench    <csv> <model.json>        measure ingestion, features and order latency
//!
//! No argument-parsing crate: the surface is small and a hand-written parser keeps the
//! dependency tree to serde and nothing else.

use std::io::BufReader;
use std::time::Instant;

use botcore::{Account, Bar, OrderType, Side, SymbolId, TimeInForce};
use exec::{ExecutionEngine, RiskEngine, RiskLimits, SimConfig, SimExchange};
use feed::{feed_from_read, Feed};
use model::{FeatureEngine, ModelFile};

const SYM: SymbolId = SymbolId(1);

fn usage() -> ! {
    eprintln!(
        "usage:\n  \
         ingest   <csv>\n  \
         features <csv> [model.json]\n  \
         backtest <csv> <model.json> [--thresh 0.55] [--fee-bps 5] [--slip-bps 2]\n  \
         bench    <csv> <model.json>"
    );
    std::process::exit(2);
}

fn open_feed(csv: &str) -> feed::CsvFeed<BufReader<std::fs::File>> {
    match std::fs::File::open(csv) {
        Ok(f) => feed_from_read(f, SYM),
        Err(e) => {
            eprintln!("cannot open {}: {}", csv, e);
            std::process::exit(1);
        }
    }
}

fn load_model(path: &str) -> ModelFile {
    let p = std::path::Path::new(path);
    match ModelFile::load(p) {
        Ok(m) => {
            if let Err(e) = m.check_feature_names() {
                eprintln!("model/feature mismatch: {}", e);
                std::process::exit(1);
            }
            m
        }
        Err(e) => {
            eprintln!("cannot load model {}: {}", path, e);
            std::process::exit(1);
        }
    }
}

fn flag(args: &[String], name: &str, default: f64) -> f64 {
    for i in 0..args.len() {
        if args[i] == name && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse::<f64>() {
                return v;
            }
        }
    }
    default
}

/* --------------------------------------------------------------- ingest */
fn cmd_ingest(csv: &str) {
    let t0 = Instant::now();
    let mut f = open_feed(csv);
    let bars = match f.read_all(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ingest failed: {}", e);
            std::process::exit(1);
        }
    };
    let wall = t0.elapsed();
    let s = f.stats();

    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut vol = 0.0;
    for b in &bars {
        lo = lo.min(b.low);
        hi = hi.max(b.high);
        vol += b.volume;
    }
    let span = if bars.len() > 1 {
        (bars[bars.len() - 1].ts - bars[0].ts) as f64 / 86_400_000_000_000.0
    } else {
        0.0
    };

    println!("  source        {}", f.source_name());
    println!("  bars          {}", bars.len());
    println!("  bytes         {}", s.bytes);
    println!("  bad lines     {}", s.bad_lines);
    println!("  calendar days {:.0}", span);
    println!("  price range   {:.2} .. {:.2}", lo, hi);
    println!("  total volume  {:.0}", vol);
    println!("  wall time     {:.3} ms", wall.as_secs_f64() * 1e3);
    println!(
        "  throughput    {:.0} bars/s   {:.1} MB/s",
        s.bars_per_sec(),
        s.mb_per_sec()
    );
}

/* ------------------------------------------------------------- features */
fn cmd_features(csv: &str, model: Option<&str>) {
    let mut f = open_feed(csv);
    let bars = f.read_all(0).expect("read");
    let mut fe = FeatureEngine::new();
    let mut last: Option<[f64; 10]> = None;
    let t0 = Instant::now();
    for b in &bars {
        if let Some(x) = fe.on_bar(b) {
            last = Some(x);
        }
    }
    let el = t0.elapsed();
    let Some(x) = last else {
        eprintln!("not enough bars: {} seen", fe.bars_seen());
        std::process::exit(1);
    };

    println!("  bars seen     {}", fe.bars_seen());
    println!("  window        {} bars", model::MAX_WINDOW);
    println!(
        "  feature time  {:.3} ms total, {:.0} ns/bar",
        el.as_secs_f64() * 1e3,
        el.as_nanos() as f64 / fe.bars_seen().max(1) as f64
    );
    println!("  features:");
    for (i, name) in model::FEATURE_NAMES.iter().enumerate() {
        println!("    {:<22} {:+.6}", name, x[i]);
    }
    if let Some(mp) = model {
        let mf = load_model(mp);
        let p = mf.model.predict(&x);
        println!("  model         {} ({})", mf.model.kind(), mf.trained_on);
        println!("  prediction    {:.6}", p);
        println!("  influence:");
        for (i, v) in mf.model.influence().iter().enumerate() {
            println!("    {:<22} {:+.6}", model::FEATURE_NAMES[i], v);
        }
    }
}

/* ------------------------------------------------------------- backtest */
struct BtResult {
    equity: Vec<f64>,
    bench: Vec<f64>,
    fills: u64,
    rejected: u64,
    fees: f64,
    exposure_bars: u64,
    bars: u64,
    latency_p50: u64,
    latency_p99: u64,
    orders: u64,
}

fn run_backtest(csv: &str, model: &str, thresh: f64, fee_bps: f64, slip_bps: f64) -> BtResult {
    let mf = load_model(model);
    let mut f = open_feed(csv);
    let bars = f.read_all(0).expect("read");

    let cfg = SimConfig {
        fee_bps,
        slippage_bps: slip_bps,
        latency_bars: 1,
        fill_ratio: 1.0,
    };
    let acct = Account::new(100_000.0, &[SYM]);
    let limits = RiskLimits {
        max_gross_notional: 200_000.0,
        ..Default::default()
    };
    let mut eng = ExecutionEngine::new(
        SimExchange::new(cfg),
        RiskEngine::new(limits, 100_000.0),
        acct,
    );

    let mut fe = FeatureEngine::new();
    let mut equity = Vec::with_capacity(bars.len());
    let mut bench = Vec::with_capacity(bars.len());
    let mut exposure = 0u64;

    let bench_entry =
        100_000.0 / (bars.first().map(|b| b.close).unwrap_or(1.0) * (1.0 + fee_bps / 10_000.0));

    for b in &bars {
        // The model sees only what the bar stream has produced so far.
        let x = fe.on_bar(b);

        // Decide, then advance the venue. The fill lands on a LATER bar, so the
        // decision can never be traded at a price this bar revealed.
        if let Some(x) = x {
            let p = mf.model.predict(&x);
            let pos = eng.account.position_qty(SYM);
            let flat = pos.abs() < 1e-9;

            if flat && p > thresh {
                let cash = eng.account.cash;
                let px = b.close;
                let qty = (cash * 0.95 / px).floor().max(0.0);
                if qty > 0.0 {
                    let _ = eng.send(
                        SYM,
                        Side::Buy,
                        qty,
                        OrderType::Market,
                        TimeInForce::Gtc,
                        b.ts,
                    );
                }
            } else if pos > 0.0 && p < 1.0 - thresh {
                let _ = eng.send(
                    SYM,
                    Side::Sell,
                    pos,
                    OrderType::Market,
                    TimeInForce::Gtc,
                    b.ts,
                );
            }
        }

        eng.on_bar(b);
        if eng.account.position_qty(SYM).abs() > 1e-9 {
            exposure += 1;
        }
        equity.push(eng.equity());
        bench.push(bench_entry * b.close);
    }

    let l = &eng.metrics.latency;
    BtResult {
        equity,
        bench,
        fills: eng.metrics.fills,
        rejected: eng.metrics.orders_rejected,
        fees: eng.account.fees_paid,
        exposure_bars: exposure,
        bars: bars.len() as u64,
        latency_p50: l.percentile_ns(50.0),
        latency_p99: l.percentile_ns(99.0),
        orders: eng.metrics.orders_sent,
    }
}

fn metrics(eq: &[f64], warm: usize) -> (f64, f64, f64, f64) {
    let start = eq[warm.min(eq.len() - 1)];
    let final_v = eq[eq.len() - 1];
    let total = final_v / start - 1.0;

    let mut rets = Vec::with_capacity(eq.len());
    for i in (warm + 1)..eq.len() {
        if eq[i - 1] > 0.0 {
            rets.push(eq[i] / eq[i - 1] - 1.0);
        }
    }
    let mean = if rets.is_empty() {
        0.0
    } else {
        rets.iter().sum::<f64>() / rets.len() as f64
    };
    let sd = if rets.len() < 2 {
        0.0
    } else {
        let v = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (rets.len() - 1) as f64;
        v.sqrt()
    };
    let sharpe = if sd > 1e-12 {
        (mean / sd) * 252f64.sqrt()
    } else {
        0.0
    };

    let mut peak = start;
    let mut mdd = 0.0;
    for v in eq.iter().skip(warm) {
        if *v > peak {
            peak = *v;
        }
        let d = v / peak - 1.0;
        if d < mdd {
            mdd = d;
        }
    }
    (total, sharpe, mdd, final_v)
}

fn cmd_backtest(csv: &str, model: &str, args: &[String]) {
    let thresh = flag(args, "--thresh", 0.55);
    let fee = flag(args, "--fee-bps", 5.0);
    let slip = flag(args, "--slip-bps", 2.0);

    let t0 = Instant::now();
    let r = run_backtest(csv, model, thresh, fee, slip);
    let el = t0.elapsed();

    let warm = model::WARMUP;
    let (total, sharpe, mdd, final_v) = metrics(&r.equity, warm);
    let (btotal, bsharpe, bmdd, bfinal) = metrics(&r.bench, warm);

    println!("  backtest      {} bars, model {}", r.bars, model);
    println!(
        "  threshold     {:.2}   costs {:.1}+{:.1} bps",
        thresh, fee, slip
    );
    println!();
    println!("  {:<16}{:>14}{:>14}", "", "strategy", "buy & hold");
    println!(
        "  {:<16}{:>14}{:>14}",
        "final equity",
        format!("{:.0}", final_v),
        format!("{:.0}", bfinal)
    );
    println!(
        "  {:<16}{:>13.2}%{:>13.2}%",
        "total return",
        total * 100.0,
        btotal * 100.0
    );
    println!("  {:<16}{:>14.2}{:>14.2}", "sharpe", sharpe, bsharpe);
    println!(
        "  {:<16}{:>13.2}%{:>13.2}%",
        "max drawdown",
        mdd * 100.0,
        bmdd * 100.0
    );
    println!();
    println!("  excess vs hold  {:.2}%", (total - btotal) * 100.0);
    println!("  orders sent     {}", r.orders);
    println!("  fills           {}", r.fills);
    println!("  rejected        {}", r.rejected);
    println!("  fees paid       {:.2}", r.fees);
    println!(
        "  exposure        {} bars ({:.0}%)",
        r.exposure_bars,
        r.exposure_bars as f64 / r.bars as f64 * 100.0
    );
    println!(
        "  order latency   p50 {} ns   p99 {} ns",
        r.latency_p50, r.latency_p99
    );
    println!("  wall time       {:.2} ms", el.as_secs_f64() * 1e3);
}

/* ----------------------------------------------------------------- bench */
fn cmd_bench(csv: &str, model: &str) {
    println!("  === ingestion ===");
    for run in 0..3 {
        let mut f = open_feed(csv);
        let t0 = Instant::now();
        let bars = f.read_all(0).expect("read");
        let el = t0.elapsed();
        let s = f.stats();
        println!(
            "    run {}  {:>6} bars in {:>7.3} ms   {:>12.0} bars/s   {:>6.1} MB/s  ({} bad)",
            run + 1,
            bars.len(),
            el.as_secs_f64() * 1e3,
            s.bars_per_sec(),
            s.mb_per_sec(),
            s.bad_lines
        );
    }

    let mut f = open_feed(csv);
    let bars: Vec<Bar> = f.read_all(0).expect("read");
    let mf = load_model(model);

    println!();
    println!("  === features ===");
    for run in 0..3 {
        let mut fe = FeatureEngine::new();
        let t0 = Instant::now();
        let mut n = 0u64;
        for b in &bars {
            if fe.on_bar(b).is_some() {
                n += 1;
            }
        }
        let el = t0.elapsed();
        println!(
            "    run {}  {:>6} vectors in {:>7.3} ms   {:>10.0} ns/bar   {:>12.0} bars/s",
            run + 1,
            n,
            el.as_secs_f64() * 1e3,
            el.as_nanos() as f64 / bars.len().max(1) as f64,
            bars.len() as f64 / el.as_secs_f64()
        );
    }

    println!();
    println!("  === inference ===");
    let mut fe = FeatureEngine::new();
    let mut xs: Vec<[f64; 10]> = Vec::new();
    for b in &bars {
        if let Some(x) = fe.on_bar(b) {
            xs.push(x);
        }
    }
    let t0 = Instant::now();
    let mut acc = 0.0;
    for x in &xs {
        acc += mf.model.predict(x);
    }
    let el = t0.elapsed();
    println!(
        "    {} predictions in {:.3} ms   {:.0} ns/prediction   (sum {:.3})",
        xs.len(),
        el.as_secs_f64() * 1e3,
        el.as_nanos() as f64 / xs.len().max(1) as f64,
        acc
    );

    println!();
    println!("  === order handling (risk check + venue submit) ===");
    let acct = Account::new(100_000_000.0, &[SYM]);
    let mut eng = ExecutionEngine::new(
        SimExchange::new(SimConfig::default()),
        RiskEngine::new(RiskLimits::default(), 100_000_000.0),
        acct,
    );
    eng.on_bar(&bars[0]);
    let n = 200_000;
    let t0 = Instant::now();
    for i in 0..n {
        let _ = eng.send(
            SYM,
            if i % 2 == 0 { Side::Buy } else { Side::Sell },
            1.0,
            OrderType::Limit(bars[0].close * 0.999),
            TimeInForce::Gtc,
            i as i64,
        );
    }
    let el = t0.elapsed();
    let l = &eng.metrics.latency;
    println!(
        "    {} orders in {:.3} ms   {:.0} ns/order mean   p50 {} ns   p99 {} ns   max {} ns",
        n,
        el.as_secs_f64() * 1e3,
        l.mean_ns(),
        l.percentile_ns(50.0),
        l.percentile_ns(99.0),
        l.max_ns()
    );
    println!(
        "    throughput  {:.0} orders/s   sent {}  rejected {}",
        n as f64 / el.as_secs_f64(),
        eng.metrics.orders_sent,
        eng.metrics.orders_rejected
    );

    println!();
    println!("  === end-to-end through the venue ===");
    let mut eng = ExecutionEngine::new(
        SimExchange::new(SimConfig::default()),
        RiskEngine::new(RiskLimits::default(), 100_000.0),
        Account::new(100_000.0, &[SYM]),
    );
    let mut fe = FeatureEngine::new();
    let t0 = Instant::now();
    let mut signals = 0u64;
    for b in &bars {
        if let Some(x) = fe.on_bar(b) {
            if mf.model.predict(&x) > 0.55 && eng.account.position_qty(SYM).abs() < 1e-9 {
                signals += 1;
                let qty = (eng.account.cash * 0.95 / b.close).floor().max(0.0);
                if qty > 0.0 {
                    let _ = eng.send(
                        SYM,
                        Side::Buy,
                        qty,
                        OrderType::Market,
                        TimeInForce::Gtc,
                        b.ts,
                    );
                }
            }
        }
        eng.on_bar(b);
    }
    let el = t0.elapsed();
    println!(
        "    {} bars, {} signals, {} fills in {:.3} ms  ({:.0} ns/bar end to end)",
        bars.len(),
        signals,
        eng.metrics.fills,
        el.as_secs_f64() * 1e3,
        el.as_nanos() as f64 / bars.len().max(1) as f64
    );
    println!("    final equity {:.0}", eng.equity());
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    match args[0].as_str() {
        "ingest" => {
            if args.len() < 2 {
                usage();
            }
            cmd_ingest(&args[1]);
        }
        "features" => {
            if args.len() < 2 {
                usage();
            }
            cmd_features(&args[1], args.get(2).map(|s| s.as_str()));
        }
        "backtest" => {
            if args.len() < 3 {
                usage();
            }
            cmd_backtest(&args[1], &args[2], &args);
        }
        "bench" => {
            if args.len() < 3 {
                usage();
            }
            cmd_bench(&args[1], &args[2]);
        }
        _ => usage(),
    }
}
