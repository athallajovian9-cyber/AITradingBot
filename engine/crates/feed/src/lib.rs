//! Market data ingestion.
//!
//! # Why a command feed instead of an HTTP client
//!
//! Rust's standard library has no TLS, so an HTTPS client means pulling in a large
//! dependency tree. More importantly the exchange endpoints are unreachable from the
//! machine this was built on (Binance and Coinbase both fail to connect), so any
//! hand-rolled client would be unverifiable code pretending to be tested.
//!
//! Instead a feed is anything that can hand over bars, and the live path is
//! `CommandFeed`, which runs an external process and parses its stdout as CSV:
//!
//! ```text
//! curl -s https://venue/api/klines?symbol=BTCUSDT&interval=1m
//! ```
//!
//! That is testable - a synthetic command proves the parsing, buffering and error
//! handling - and it is how real deployments wire a venue without linking TLS into
//! the hot path. The venue-specific part is one command line, not code.

use std::io::{BufRead, BufReader, Read};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::time::Instant;

pub use botcore::{Bar, SymbolId};

#[derive(Debug)]
pub enum FeedError {
    Io(String),
    Parse { line: u64, why: String },
    Spawn(String),
    Exhausted,
}

impl std::fmt::Display for FeedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeedError::Io(e) => write!(f, "io: {}", e),
            FeedError::Parse { line, why } => write!(f, "parse error on line {}: {}", line, why),
            FeedError::Spawn(e) => write!(f, "spawn: {}", e),
            FeedError::Exhausted => write!(f, "feed exhausted"),
        }
    }
}
impl std::error::Error for FeedError {}

/// What every data source implements.
pub trait Feed {
    /// Fill `out` with up to `max` bars. Returns how many were actually produced.
    /// `Ok(0)` means the source has no more data right now.
    fn poll_batch(&mut self, max: usize, out: &mut Vec<Bar>) -> Result<usize, FeedError>;
    fn source_name(&self) -> &str;

    /// Drain the feed completely, up to `limit` bars (0 = no limit).
    fn read_all(&mut self, limit: usize) -> Result<Vec<Bar>, FeedError> {
        let mut all = Vec::new();
        let mut buf = Vec::with_capacity(4096);
        loop {
            buf.clear();
            let n = self.poll_batch(4096, &mut buf)?;
            if n == 0 {
                break;
            }
            all.append(&mut buf);
            if limit > 0 && all.len() >= limit {
                all.truncate(limit);
                break;
            }
        }
        Ok(all)
    }
}

/// Counters for a feed, so throughput claims can be measured rather than asserted.
#[derive(Debug, Clone, Copy, Default)]
pub struct FeedStats {
    pub bars: u64,
    pub bytes: u64,
    pub bad_lines: u64,
    pub elapsed_ns: u128,
}

impl FeedStats {
    pub fn bars_per_sec(&self) -> f64 {
        if self.elapsed_ns == 0 {
            0.0
        } else {
            self.bars as f64 / (self.elapsed_ns as f64 / 1e9)
        }
    }
    pub fn mb_per_sec(&self) -> f64 {
        if self.elapsed_ns == 0 {
            0.0
        } else {
            (self.bytes as f64 / 1_048_576.0) / (self.elapsed_ns as f64 / 1e9)
        }
    }
}

/// Days since the Unix epoch for a proleptic Gregorian date.
/// (Howard Hinnant's `days_from_civil`.) Dependency-free and exact.
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Parse `YYYY-MM-DD` (or a bare epoch-seconds / epoch-nanos integer) into ns.
pub fn parse_ts(s: &str) -> Option<i64> {
    let s = s.trim();
    if let Some((y, rest)) = s.split_once('-') {
        let (m, d) = rest.split_once('-')?;
        // tolerate a trailing time component
        let d = d.split(['T', ' ']).next()?;
        let y: i64 = y.parse().ok()?;
        let m: i64 = m.parse().ok()?;
        let d: i64 = d.parse().ok()?;
        if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
            return None;
        }
        return Some(days_from_civil(y, m, d) * 86_400_000_000_000);
    }
    let v: i64 = s.parse().ok()?;
    // heuristic: anything past ~year 2300 in seconds is already nanoseconds
    if v > 10_000_000_000 {
        Some(v)
    } else {
        Some(v * 1_000_000_000)
    }
}

/// Parse one CSV bar line: `date,open,high,low,close,volume`.
///
/// Hand-rolled rather than depending on a CSV crate: the format is fixed, and this
/// stays allocation-free apart from the fields it must copy.
pub fn parse_bar_line(line: &str, symbol: SymbolId) -> Result<Bar, String> {
    let line = line.trim_end_matches(['\r', '\n']);
    if line.is_empty() {
        return Err("empty".into());
    }
    let mut it = line.split(',');
    let ts_s = it.next().ok_or("missing date")?;
    let o = it.next().ok_or("missing open")?;
    let h = it.next().ok_or("missing high")?;
    let l = it.next().ok_or("missing low")?;
    let c = it.next().ok_or("missing close")?;
    let v = it.next().unwrap_or("0");

    let ts = parse_ts(ts_s).ok_or_else(|| format!("bad date {:?}", ts_s))?;
    let f = |x: &str, what: &str| -> Result<f64, String> {
        x.trim()
            .parse::<f64>()
            .map_err(|_| format!("bad {} {:?}", what, x))
            .and_then(|v| {
                if v.is_finite() {
                    Ok(v)
                } else {
                    Err(format!("non-finite {}", what))
                }
            })
    };
    let bar = Bar {
        symbol,
        ts,
        open: f(o, "open")?,
        high: f(h, "high")?,
        low: f(l, "low")?,
        close: f(c, "close")?,
        volume: f(v, "volume").unwrap_or(0.0),
    };
    // A bar whose high/low do not contain its own prices is corrupt data, not a
    // surprise to be carried downstream.
    // Written as explicit comparisons rather than negated ones: on floats a negated
    // comparison also swallows NaN, and a NaN bar should be rejected as malformed
    // rather than fall through a range check that cannot decide.
    if !(bar.low.is_finite() && bar.high.is_finite()) || bar.low > bar.high {
        return Err(format!("low {} above high {}", bar.low, bar.high));
    }
    if bar.close < bar.low - 1e-9 || bar.close > bar.high + 1e-9 {
        return Err(format!(
            "close {} outside [{}, {}]",
            bar.close, bar.low, bar.high
        ));
    }
    Ok(bar)
}

/// A header line carries the column names rather than data.
fn looks_like_header(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    l.contains("date") && l.contains("close")
}

/// Reads bars from any `BufRead`. Skips a header line if present.
pub struct CsvFeed<R: BufRead> {
    reader: R,
    symbol: SymbolId,
    line_no: u64,
    started: bool,
    stats: FeedStats,
    t0: Option<Instant>,
}

impl<R: BufRead> CsvFeed<R> {
    pub fn new(reader: R, symbol: SymbolId) -> Self {
        CsvFeed {
            reader,
            symbol,
            line_no: 0,
            started: false,
            stats: FeedStats::default(),
            t0: None,
        }
    }

    pub fn stats(&self) -> FeedStats {
        let mut s = self.stats;
        if let Some(t0) = self.t0 {
            s.elapsed_ns = t0.elapsed().as_nanos();
        }
        s
    }
}

impl<R: BufRead> Feed for CsvFeed<R> {
    fn poll_batch(&mut self, max: usize, out: &mut Vec<Bar>) -> Result<usize, FeedError> {
        if self.t0.is_none() {
            self.t0 = Some(Instant::now());
        }
        let mut produced = 0;
        let mut line = String::with_capacity(128);
        while produced < max {
            line.clear();
            let n = self
                .reader
                .read_line(&mut line)
                .map_err(|e| FeedError::Io(e.to_string()))?;
            if n == 0 {
                break; // EOF
            }
            self.line_no += 1;
            self.stats.bytes += n as u64;
            if !self.started {
                self.started = true;
                if looks_like_header(&line) {
                    continue;
                }
            }
            match parse_bar_line(&line, self.symbol) {
                Ok(b) => {
                    out.push(b);
                    produced += 1;
                    self.stats.bars += 1;
                }
                Err(why) => {
                    self.stats.bad_lines += 1;
                    // A header or blank line is not an error worth stopping for.
                    if !why.starts_with("empty") && !why.starts_with("bad date") {
                        return Err(FeedError::Parse {
                            line: self.line_no,
                            why,
                        });
                    }
                }
            }
        }
        Ok(produced)
    }

    fn source_name(&self) -> &str {
        "csv"
    }
}

/// Deterministic in-memory replay. Used by tests and backtests so a run can be
/// reproduced exactly.
pub struct VecFeed {
    bars: Vec<Bar>,
    idx: usize,
}

impl VecFeed {
    pub fn new(bars: Vec<Bar>) -> Self {
        VecFeed { bars, idx: 0 }
    }
    pub fn total(&self) -> usize {
        self.bars.len()
    }
}

impl Feed for VecFeed {
    fn poll_batch(&mut self, max: usize, out: &mut Vec<Bar>) -> Result<usize, FeedError> {
        let remaining = self.bars.len() - self.idx;
        let n = remaining.min(max);
        out.extend_from_slice(&self.bars[self.idx..self.idx + n]);
        self.idx += n;
        Ok(n)
    }
    fn source_name(&self) -> &str {
        "replay"
    }
}

/// Runs an external process and parses its stdout as CSV bars.
///
/// This is the live path: point it at `curl` (or `wscat`, or anything that prints
/// OHLCV) and the venue-specific part of the system becomes one command line rather
/// than a TLS stack linked into the engine.
pub struct CommandFeed {
    child: Child,
    out: BufReader<ChildStdout>,
    symbol: SymbolId,
    line_no: u64,
    started: bool,
    stats: FeedStats,
    t0: Option<Instant>,
}

impl CommandFeed {
    pub fn spawn(program: &str, args: &[&str], symbol: SymbolId) -> Result<Self, FeedError> {
        let mut child = Command::new(program)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .stdin(Stdio::null())
            .spawn()
            .map_err(|e| FeedError::Spawn(format!("{}: {}", program, e)))?;
        let out = child
            .stdout
            .take()
            .ok_or_else(|| FeedError::Spawn("no stdout".into()))?;
        Ok(CommandFeed {
            child,
            out: BufReader::with_capacity(64 * 1024, out),
            symbol,
            line_no: 0,
            started: false,
            stats: FeedStats::default(),
            t0: None,
        })
    }

    pub fn stats(&self) -> FeedStats {
        let mut s = self.stats;
        if let Some(t0) = self.t0 {
            s.elapsed_ns = t0.elapsed().as_nanos();
        }
        s
    }
}

impl Feed for CommandFeed {
    fn poll_batch(&mut self, max: usize, out: &mut Vec<Bar>) -> Result<usize, FeedError> {
        if self.t0.is_none() {
            self.t0 = Some(Instant::now());
        }
        let mut produced = 0;
        let mut line = String::with_capacity(256);
        while produced < max {
            line.clear();
            let n = self
                .out
                .read_line(&mut line)
                .map_err(|e| FeedError::Io(e.to_string()))?;
            if n == 0 {
                break;
            }
            self.line_no += 1;
            self.stats.bytes += n as u64;
            if !self.started {
                self.started = true;
                if looks_like_header(&line) {
                    continue;
                }
            }
            if let Ok(b) = parse_bar_line(&line, self.symbol) {
                out.push(b);
                produced += 1;
                self.stats.bars += 1;
            } else {
                self.stats.bad_lines += 1;
            }
        }
        Ok(produced)
    }

    fn source_name(&self) -> &str {
        "command"
    }
}

impl Drop for CommandFeed {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Adapter that makes any `Read` a bar source, for tests and file ingestion.
pub fn feed_from_read<R: Read>(r: R, symbol: SymbolId) -> CsvFeed<BufReader<R>> {
    CsvFeed::new(BufReader::new(r), symbol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const CSV: &str = "date,open,high,low,close,volume\n\
2026-01-02,100.0,105.0,99.0,104.0,1000\n\
2026-01-03,104.0,108.0,103.0,107.5,1200\n\
2026-01-04,107.5,109.0,105.0,106.0,900\n";

    #[test]
    fn days_from_civil_matches_known_dates() {
        // every value here was cross-checked against Python's datetime, which is the
        // reference for calendar arithmetic
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1970, 1, 2), 1);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        assert_eq!(days_from_civil(2000, 1, 1), 10957);
        assert_eq!(days_from_civil(2024, 2, 29), 19782);
        assert_eq!(days_from_civil(2026, 1, 1), 20454);
        assert_eq!(days_from_civil(2026, 9, 30), 20726);
    }

    #[test]
    fn parse_ts_handles_dates_and_epochs() {
        assert_eq!(parse_ts("1970-01-01"), Some(0));
        assert_eq!(parse_ts("1970-01-02"), Some(86_400_000_000_000));
        assert_eq!(parse_ts("1700000000"), Some(1_700_000_000_000_000_000));
        assert_eq!(parse_ts("2026-01-02T09:30:00"), parse_ts("2026-01-02"));
        assert!(parse_ts("2026-13-01").is_none());
        assert!(parse_ts("nonsense").is_none());
    }

    #[test]
    fn parses_a_valid_bar() {
        let b = parse_bar_line("2026-01-02,100,105,99,104,1000", SymbolId(1)).unwrap();
        assert_eq!(b.open, 100.0);
        assert_eq!(b.high, 105.0);
        assert_eq!(b.low, 99.0);
        assert_eq!(b.close, 104.0);
        assert_eq!(b.volume, 1000.0);
    }

    #[test]
    fn rejects_a_close_outside_the_range() {
        // corrupt data must be refused, not carried into the model
        assert!(parse_bar_line("2026-01-02,100,105,99,200,1000", SymbolId(1)).is_err());
    }

    #[test]
    fn rejects_low_above_high() {
        assert!(parse_bar_line("2026-01-02,100,90,99,95,1000", SymbolId(1)).is_err());
    }

    #[test]
    fn rejects_short_lines() {
        assert!(parse_bar_line("2026-01-02,100,105", SymbolId(1)).is_err());
    }

    #[test]
    fn csv_feed_skips_the_header_and_reads_all_bars() {
        let mut f = feed_from_read(Cursor::new(CSV.as_bytes()), SymbolId(1));
        let bars = f.read_all(0).unwrap();
        assert_eq!(bars.len(), 3);
        assert_eq!(bars[0].close, 104.0);
        assert_eq!(bars[2].close, 106.0);
        assert_eq!(f.stats().bars, 3);
    }

    #[test]
    fn csv_feed_respects_the_batch_limit() {
        let mut f = feed_from_read(Cursor::new(CSV.as_bytes()), SymbolId(1));
        let mut buf = Vec::new();
        assert_eq!(f.poll_batch(2, &mut buf).unwrap(), 2);
        assert_eq!(buf.len(), 2);
        buf.clear();
        assert_eq!(f.poll_batch(2, &mut buf).unwrap(), 1);
        buf.clear();
        assert_eq!(f.poll_batch(2, &mut buf).unwrap(), 0);
    }

    #[test]
    fn csv_feed_tolerates_blank_lines() {
        let csv = "date,open,high,low,close,volume\n\n2026-01-02,100,105,99,104,1\n\n";
        let mut f = feed_from_read(Cursor::new(csv.as_bytes()), SymbolId(1));
        let bars = f.read_all(0).unwrap();
        assert_eq!(bars.len(), 1);
    }

    #[test]
    fn vec_feed_is_deterministic_and_repeatable() {
        let mk = || {
            vec![Bar {
                symbol: SymbolId(1),
                ts: 0,
                open: 1.0,
                high: 2.0,
                low: 0.5,
                close: 1.5,
                volume: 10.0,
            }]
        };
        let mut a = VecFeed::new(mk());
        let mut b = VecFeed::new(mk());
        assert_eq!(a.read_all(0).unwrap(), b.read_all(0).unwrap());
        assert_eq!(a.total(), 1);
    }

    #[test]
    fn stats_report_throughput_after_reading() {
        let mut f = feed_from_read(Cursor::new(CSV.as_bytes()), SymbolId(1));
        f.read_all(0).unwrap();
        let s = f.stats();
        assert!(s.bars == 3);
        assert!(s.bytes > 0);
        assert!(s.bars_per_sec() > 0.0);
    }

    #[test]
    fn command_feed_parses_a_real_process_output() {
        // The live path uses an external command. Proving it here means the live
        // wiring is the only untested part, not the parsing.
        #[cfg(windows)]
        let (prog, args): (&str, Vec<&str>) = (
            "cmd",
            vec![
                "/C",
                "echo date,open,high,low,close,volume& echo 2026-01-02,100,105,99,104,1000",
            ],
        );
        #[cfg(not(windows))]
        let (prog, args): (&str, Vec<&str>) = (
            "sh",
            vec![
                "-c",
                "printf 'date,open,high,low,close,volume\\n2026-01-02,100,105,99,104,1000\\n'",
            ],
        );
        let mut f = CommandFeed::spawn(prog, &args, SymbolId(1)).expect("spawn");
        let bars = f.read_all(0).unwrap();
        assert_eq!(bars.len(), 1, "got {:?}", bars);
        assert_eq!(bars[0].close, 104.0);
        assert_eq!(f.source_name(), "command");
    }

    #[test]
    fn command_feed_reports_a_bad_program() {
        let r = CommandFeed::spawn("definitely-not-a-program-xyz", &[], SymbolId(1));
        assert!(matches!(r, Err(FeedError::Spawn(_))));
    }
}
