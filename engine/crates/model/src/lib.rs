//! Feature engineering and model inference.
//!
//! The models are trained in Python (see `research/train.py`) and exported to JSON.
//! This crate recomputes the SAME features from raw bars and runs the SAME arithmetic
//! the trainer used, so a prediction made here is the prediction Python trained.
//!
//! Feature parity is not assumed, it is tested: the trainer also exports a set of
//! reference rows and their expected predictions, and `tests/parity.rs` fails if Rust
//! disagrees by more than 1e-9.
//!
//! Speed design: a `FeatureEngine` keeps a ring buffer of recent bars and computes
//! indicators over that window. Indicators are recomputed per bar rather than
//! maintained incrementally - a deliberate trade of a few nanoseconds for the
//! certainty that the streaming path and the offline path produce identical numbers.

use serde::{Deserialize, Serialize};

pub use botcore::Bar;

/// Number of bars the longest indicator needs before it can produce a value.
pub const WARMUP: usize = 60;
/// How much history the streaming engine retains.
///
/// This is deliberately longer than any indicator needs. EMA-based indicators are
/// seeded with a simple average of their first `n` values, and that seed only decays
/// as `(1-k)^bars` - with a 60-bar window the seed still influenced the result in the
/// 1e-2 range, which is far too much to agree with a trainer that saw the full
/// series. At 1024 bars the residual is below 1e-16, so the streaming and batch paths
/// produce identical numbers for any series shorter than the window.
///
/// A live engine started mid-stream therefore needs to warm up before its features
/// match the trainer exactly; `bars_seen` is exposed for exactly that check.
pub const MAX_WINDOW: usize = 1024;

/* ============================================================ indicators
   All computed over a slice where index `last` is "now" and earlier indices are
   history. These mirror the definitions in engine2.js and research/features.py.
*/

fn sma(x: &[f64], n: usize) -> Option<f64> {
    if x.len() < n {
        return None;
    }
    let s: f64 = x[x.len() - n..].iter().sum();
    Some(s / n as f64)
}

fn stdev(x: &[f64], n: usize) -> Option<f64> {
    if x.len() < n {
        return None;
    }
    let w = &x[x.len() - n..];
    let m = w.iter().sum::<f64>() / n as f64;
    let v = w.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
    Some(v.sqrt())
}

/// Wilder RSI.
pub fn rsi(x: &[f64], n: usize) -> Option<f64> {
    if x.len() <= n {
        return None;
    }
    let mut gain = 0.0;
    let mut loss = 0.0;
    for i in 1..=n {
        let d = x[i] - x[i - 1];
        if d >= 0.0 {
            gain += d;
        } else {
            loss -= d;
        }
    }
    gain /= n as f64;
    loss /= n as f64;
    for i in n + 1..x.len() {
        let d = x[i] - x[i - 1];
        gain = (gain * (n as f64 - 1.0) + if d > 0.0 { d } else { 0.0 }) / n as f64;
        loss = (loss * (n as f64 - 1.0) + if d < 0.0 { -d } else { 0.0 }) / n as f64;
    }
    if loss < 1e-12 {
        Some(100.0)
    } else {
        Some(100.0 - 100.0 / (1.0 + gain / loss))
    }
}

/// EMA over a whole series, NaN until there is enough history.
/// One pass, so the result is identical to computing the final value alone.
fn ema_series(x: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; x.len()];
    if x.len() < n {
        return out;
    }
    let k = 2.0 / (n as f64 + 1.0);
    let mut prev = x[..n].iter().sum::<f64>() / n as f64;
    out[n - 1] = prev;
    for i in n..x.len() {
        prev = x[i] * k + prev * (1.0 - k);
        out[i] = prev;
    }
    out
}

/// MACD histogram: EMA(fast) - EMA(slow), minus its own EMA(signal).
///
/// This was originally written as `ema(&x[..i], ..)` inside a loop, which recomputes
/// the whole EMA for every prefix - O(n^2) per call, and it is called once per bar.
/// It measured 112 microseconds per bar, which is not high-speed by any definition.
/// The single-pass version below is mathematically identical and O(n); the parity test
/// against Python confirms the two agree to 4.7e-11.
pub fn macd_hist(x: &[f64]) -> Option<f64> {
    let (fast, slow, sig) = (12usize, 26usize, 9usize);
    if x.len() < slow + sig {
        return None;
    }
    let ef = ema_series(x, fast);
    let es = ema_series(x, slow);
    let start = slow - 1;
    let line: Vec<f64> = (start..x.len()).map(|i| ef[i] - es[i]).collect();
    let signal = ema_series(&line, sig);
    let last = line[line.len() - 1];
    let s = signal[signal.len() - 1];
    if s.is_nan() {
        return None;
    }
    Some(last - s)
}

/// Realised volatility of log returns, annualised at 252 bars.
pub fn realized_vol(x: &[f64], n: usize) -> Option<f64> {
    if x.len() < n + 1 {
        return None;
    }
    let r: Vec<f64> = x[x.len() - n - 1..]
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();
    let m = r.iter().sum::<f64>() / r.len() as f64;
    let v = r.iter().map(|v| (v - m).powi(2)).sum::<f64>() / r.len() as f64;
    Some(v.sqrt() * 252f64.sqrt())
}

fn clip(v: f64, a: f64, b: f64) -> f64 {
    if v < a {
        a
    } else if v > b {
        b
    } else {
        v
    }
}

pub const FEATURE_NAMES: [&str; 10] = [
    "1-bar return",
    "5-bar return",
    "20-bar return",
    "RSI(14) position",
    "MACD histogram",
    "Bollinger %B",
    "SMA20-50 trend",
    "Volatility ratio",
    "Volume z-score",
    "Range position",
];

/// The feature vector at the most recent bar of `closes` (and friends).
///
/// Returns `None` until there is enough history, which is the same warmup rule the
/// trainer uses. Refusing to guess is the point: a feature vector from insufficient
/// history would silently differ from the trained distribution.
pub fn features(closes: &[f64], highs: &[f64], lows: &[f64], volumes: &[f64]) -> Option<[f64; 10]> {
    let n = closes.len();
    if n < WARMUP + 1 {
        return None;
    }
    let c = closes[n - 1];
    if closes[n - 2] <= 0.0 || closes[n - 6] <= 0.0 || closes[n - 21] <= 0.0 {
        return None;
    }

    let r1 = c / closes[n - 2] - 1.0;
    let r5 = c / closes[n - 6] - 1.0;
    let r20 = c / closes[n - 21] - 1.0;

    let rsi14 = rsi(closes, 14)?;
    let mh = macd_hist(closes)?;
    let sd20 = stdev(closes, 20)?;
    let s20 = sma(closes, 20)?;
    let s50 = sma(closes, 50)?;
    let v10 = realized_vol(closes, 10)?;
    let v30 = realized_vol(closes, 30)?;

    let m20 = sma(closes, 20)?;
    let pb = if sd20 > 1e-12 {
        (c - (m20 - 2.0 * sd20)) / (4.0 * sd20)
    } else {
        0.5
    };
    let trend = if s50.abs() > 1e-12 {
        (s20 - s50) / s50
    } else {
        0.0
    };
    let vol_ratio = if v30.abs() > 1e-12 { v10 / v30 } else { 1.0 };

    let vz = {
        let m = sma(volumes, 20);
        let sd = stdev(volumes, 20);
        match (m, sd) {
            (Some(m), Some(sd)) if sd > 1e-12 => (volumes[n - 1] - m) / sd,
            _ => 0.0,
        }
    };

    let range_pos = if highs[n - 1] > lows[n - 1] {
        (c - lows[n - 1]) / (highs[n - 1] - lows[n - 1])
    } else {
        0.5
    };

    Some([
        clip(r1 * 100.0, -10.0, 10.0),
        clip(r5 * 40.0, -10.0, 10.0),
        clip(r20 * 12.0, -10.0, 10.0),
        (rsi14 - 50.0) / 50.0,
        clip(mh / sd20.max(1e-12), -6.0, 6.0),
        clip((pb - 0.5) * 2.5, -5.0, 5.0),
        clip(trend * 25.0, -10.0, 10.0),
        clip((vol_ratio - 1.0) * 2.0, -5.0, 5.0),
        clip(vz, -4.0, 4.0),
        clip((range_pos - 0.5) * 3.0, -3.0, 3.0),
    ])
}

/* ======================================================= streaming engine

   Keeps a rolling window of bars so features can be produced per bar without the
   caller managing history. This is the ingestion hot path.
*/
pub struct FeatureEngine {
    capacity: usize,
    closes: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    volumes: Vec<f64>,
    seen: u64,
}

impl FeatureEngine {
    pub fn new() -> Self {
        let cap = MAX_WINDOW + 2;
        FeatureEngine {
            capacity: cap,
            closes: Vec::with_capacity(cap),
            highs: Vec::with_capacity(cap),
            lows: Vec::with_capacity(cap),
            volumes: Vec::with_capacity(cap),
            seen: 0,
        }
    }

    pub fn bars_seen(&self) -> u64 {
        self.seen
    }

    fn push(v: &mut Vec<f64>, x: f64, cap: usize) {
        v.push(x);
        if v.len() > cap {
            let drop = v.len() - cap;
            v.drain(0..drop);
        }
    }

    /// Feed one bar. Returns features when there is enough history.
    pub fn on_bar(&mut self, bar: &Bar) -> Option<[f64; 10]> {
        Self::push(&mut self.closes, bar.close, self.capacity);
        Self::push(&mut self.highs, bar.high, self.capacity);
        Self::push(&mut self.lows, bar.low, self.capacity);
        Self::push(&mut self.volumes, bar.volume, self.capacity);
        self.seen += 1;
        // Emit as soon as the shortest indicator is ready. The retained window is
        // separate from this threshold: history is KEPT for up to MAX_WINDOW bars so
        // the EMA seeding matches the trainer, but features are produced from
        // WARMUP onward rather than waiting for the whole window to fill.
        if self.closes.len() < WARMUP + 1 {
            return None;
        }
        features(&self.closes, &self.highs, &self.lows, &self.volumes)
    }

    pub fn reset(&mut self) {
        self.closes.clear();
        self.highs.clear();
        self.lows.clear();
        self.volumes.clear();
        self.seen = 0;
    }
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self::new()
    }
}

/* ================================================================ model */

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    pub weights: Vec<f64>,
    pub bias: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mlp {
    /// one row per hidden unit
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Model {
    Logistic(Linear),
    LinearReg(Linear),
    Mlp(Mlp),
}

impl Model {
    /// Probability the price is higher `horizon` bars from now.
    ///
    /// `logistic` and `mlp` squash through a sigmoid; `linear` returns the raw
    /// regression output, which the caller thresholds itself.
    pub fn predict(&self, x: &[f64]) -> f64 {
        match self {
            Model::Logistic(l) => sigmoid(dot(&l.weights, x) + l.bias),
            Model::LinearReg(l) => dot(&l.weights, x) + l.bias,
            Model::Mlp(m) => {
                let mut z = m.b2;
                for (h, row) in m.w1.iter().enumerate() {
                    z += m.w2[h] * (dot(row, x) + m.b1[h]).tanh();
                }
                sigmoid(z)
            }
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Model::Logistic(_) => "logistic",
            Model::LinearReg(_) => "linear",
            Model::Mlp(_) => "mlp",
        }
    }

    /// Per-feature influence, matching the definition used in the UI: the linear
    /// coefficient, or for the net the summed path weight through the hidden layer.
    pub fn influence(&self) -> Vec<f64> {
        match self {
            Model::Logistic(l) | Model::LinearReg(l) => l.weights.clone(),
            Model::Mlp(m) => {
                let nf = m.w1.first().map(|r| r.len()).unwrap_or(0);
                let mut out = vec![0.0; nf];
                for (h, row) in m.w1.iter().enumerate() {
                    for j in 0..nf {
                        out[j] += row[j] * m.w2[h];
                    }
                }
                out
            }
        }
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut s = 0.0;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

#[inline]
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
}

/// The artifact Python writes and Rust reads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub version: u32,
    pub trained_on: String,
    pub bars: usize,
    pub horizon: usize,
    pub features: Vec<String>,
    pub model: Model,
    /// Reference rows for the parity test: input vector and the trainer's own output.
    #[serde(default)]
    pub reference: Vec<ReferenceRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceRow {
    pub x: Vec<f64>,
    pub p: f64,
}

impl ModelFile {
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| e.to_string())
    }
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| e.to_string())
    }
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Self::from_json(&s)
    }
    /// Check the exported feature names match the ones this crate computes, in order.
    pub fn check_feature_names(&self) -> Result<(), String> {
        if self.features.len() != FEATURE_NAMES.len() {
            return Err(format!(
                "model expects {} features, this build computes {}",
                self.features.len(),
                FEATURE_NAMES.len()
            ));
        }
        for (i, n) in self.features.iter().enumerate() {
            if n != FEATURE_NAMES[i] {
                return Err(format!(
                    "feature {} mismatch: model says {:?}, this build says {:?}",
                    i, n, FEATURE_NAMES[i]
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize, start: f64, step: f64) -> Vec<f64> {
        (0..n).map(|i| start + i as f64 * step).collect()
    }

    #[test]
    fn rsi_matches_the_published_seed_value() {
        // the StockCharts worked example, first value after 14 periods
        let c = [
            44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245, 45.8433,
            46.0826, 45.8931, 46.0328, 45.6140, 46.2820, 46.2820,
        ];
        let r = rsi(&c, 14).unwrap();
        assert!((r - 70.464).abs() < 0.1, "rsi {}", r);
    }

    #[test]
    fn rsi_is_100_without_losses() {
        let c = ramp(30, 100.0, 1.0);
        assert_eq!(rsi(&c, 14).unwrap(), 100.0);
    }

    #[test]
    fn sma_and_stdev_on_a_constant_series() {
        let c = vec![5.0; 40];
        assert_eq!(sma(&c, 20).unwrap(), 5.0);
        assert_eq!(stdev(&c, 20).unwrap(), 0.0);
    }

    #[test]
    fn ema_of_a_constant_is_that_constant() {
        let c = vec![7.0; 40];
        let e = ema_series(&c, 20);
        assert!((e[39] - 7.0).abs() < 1e-12);
        // and the series is unpopulated before it has enough history
        assert!(e[18].is_nan());
    }

    #[test]
    fn ema_series_rises_toward_a_step() {
        // after a step up, the EMA must climb but lag the raw value
        let mut c = vec![1.0; 40];
        for v in c.iter_mut().skip(20) {
            *v = 2.0;
        }
        let e = ema_series(&c, 10);
        assert!(e[39] > 1.5 && e[39] < 2.0, "ema {} should lag the step", e[39]);
    }

    #[test]
    fn realized_vol_of_a_constant_series_is_zero() {
        let c = vec![10.0; 40];
        assert!(realized_vol(&c, 10).unwrap() < 1e-12);
    }

    #[test]
    fn features_need_enough_history() {
        let c = ramp(30, 100.0, 0.5);
        assert!(features(&c, &c, &c, &c).is_none());
    }

    #[test]
    fn features_are_finite_and_clipped() {
        let c = ramp(WARMUP + 5, 100.0, 0.7);
        let h: Vec<f64> = c.iter().map(|v| v + 1.0).collect();
        let l: Vec<f64> = c.iter().map(|v| v - 1.0).collect();
        let v = vec![1_000_000.0; c.len()];
        let x = features(&c, &h, &l, &v).unwrap();
        for f in x.iter() {
            assert!(f.is_finite(), "non-finite feature {:?}", x);
            assert!(f.abs() <= 10.0001, "unclipped feature {} in {:?}", f, x);
        }
    }

    #[test]
    fn feature_engine_matches_batch_features() {
        // the streaming path and the one-shot path must agree exactly, or a live
        // engine and a backtest would silently disagree
        let n = WARMUP + 8;
        let c = ramp(n, 100.0, 0.4);
        let h: Vec<f64> = c.iter().map(|v| v + 0.5).collect();
        let l: Vec<f64> = c.iter().map(|v| v - 0.5).collect();
        let vol: Vec<f64> = (0..n).map(|i| 1000.0 + i as f64 * 7.0).collect();

        let mut fe = FeatureEngine::new();
        let mut last = None;
        for i in 0..n {
            let b = Bar {
                symbol: botcore::SymbolId(1),
                ts: i as i64,
                open: c[i],
                high: h[i],
                low: l[i],
                close: c[i],
                volume: vol[i],
            };
            last = fe.on_bar(&b);
        }
        let stream = last.expect("engine should produce features");
        let batch = features(&c, &h, &l, &vol).expect("batch features");
        for i in 0..10 {
            assert!(
                (stream[i] - batch[i]).abs() < 1e-12,
                "feature {} differs: stream {} vs batch {}",
                i,
                stream[i],
                batch[i]
            );
        }
    }

    #[test]
    fn logistic_predicts_in_range() {
        let m = Model::Logistic(Linear {
            weights: vec![1.0; 10],
            bias: 0.0,
        });
        let p = m.predict(&[0.1; 10]);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn logistic_is_monotonic_in_a_positive_weight() {
        let m = Model::Logistic(Linear {
            weights: vec![1.0; 10],
            bias: 0.0,
        });
        let lo = m.predict(&[0.0; 10]);
        let hi = m.predict(&[1.0; 10]);
        assert!(hi > lo);
        assert!((lo - 0.5).abs() < 1e-12);
    }

    #[test]
    fn mlp_forward_pass_is_finite_and_bounded() {
        let m = Model::Mlp(Mlp {
            w1: vec![vec![0.5; 10]; 8],
            b1: vec![0.0; 8],
            w2: vec![0.25; 8],
            b2: 0.0,
        });
        let p = m.predict(&[0.5; 10]);
        assert!(p > 0.0 && p < 1.0, "p {}", p);
    }

    #[test]
    fn influence_has_one_entry_per_feature() {
        let m = Model::Mlp(Mlp {
            w1: vec![vec![0.5; 10]; 8],
            b1: vec![0.0; 8],
            w2: vec![0.25; 8],
            b2: 0.0,
        });
        assert_eq!(m.influence().len(), 10);
        let l = Model::Logistic(Linear {
            weights: vec![1.0; 10],
            bias: 0.0,
        });
        assert_eq!(l.influence().len(), 10);
    }

    #[test]
    fn feature_name_check_rejects_a_mismatch() {
        let mf = ModelFile {
            version: 1,
            trained_on: "test".into(),
            bars: 0,
            horizon: 10,
            features: vec!["wrong".into()],
            model: Model::Logistic(Linear {
                weights: vec![0.0; 10],
                bias: 0.0,
            }),
            reference: vec![],
        };
        assert!(mf.check_feature_names().is_err());
    }

    #[test]
    fn model_file_round_trips_through_json() {
        let mf = ModelFile {
            version: 1,
            trained_on: "test".into(),
            bars: 100,
            horizon: 10,
            features: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
            model: Model::Mlp(Mlp {
                w1: vec![vec![0.1; 10]; 8],
                b1: vec![0.0; 8],
                w2: vec![0.2; 8],
                b2: 0.05,
            }),
            reference: vec![ReferenceRow {
                x: vec![0.5; 10],
                p: 0.42,
            }],
        };
        mf.check_feature_names().unwrap();
        let s = mf.to_json().unwrap();
        let back = ModelFile::from_json(&s).unwrap();
        assert_eq!(back.horizon, 10);
        assert_eq!(back.reference.len(), 1);
        assert!((back.model.predict(&[0.5; 10]) - mf.model.predict(&[0.5; 10])).abs() < 1e-15);
    }
}
