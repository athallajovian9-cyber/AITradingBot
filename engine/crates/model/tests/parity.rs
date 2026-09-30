//! Python/Rust parity.
//!
//! Feature engineering exists twice: once in `research/train.py` and once in
//! `crates/model`. That duplication is a real risk - if the two drift, Rust quietly
//! feeds the model inputs it was never trained on, and every prediction is wrong
//! while every test still passes.
//!
//! This test is the guard. Python exports the bars it used, the features it computed
//! and the probabilities it produced. Rust recomputes both from the same bars and
//! fails on any disagreement.
//!
//! Skipped with a clear message if the training artifacts are absent, so a fresh
//! clone can still build and test the engine before running the trainer.

use model::{FeatureEngine, Model, ModelFile, FEATURE_NAMES};
use serde::Deserialize;

#[derive(Deserialize)]
struct ParityBar {
    close: f64,
    high: f64,
    low: f64,
    open: f64,
    volume: f64,
}

#[derive(Deserialize)]
struct ParityRow {
    bar_index: usize,
    x: Vec<f64>,
    p: f64,
}

#[derive(Deserialize)]
struct ParityFile {
    symbol: String,
    bars: Vec<ParityBar>,
    rows: Vec<ParityRow>,
    model: Model,
    features: Vec<String>,
}

fn parity_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("models")
        .join("parity.json")
}

#[test]
fn rust_reproduces_pythons_features_and_predictions() {
    let path = parity_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "SKIPPED: {} not found. Run `python3 research/train.py` first.",
            path.display()
        );
        return;
    };
    let pf: ParityFile = serde_json::from_str(&text).expect("parity.json should parse");

    // The feature definitions must match before anything else is worth comparing.
    assert_eq!(
        pf.features.len(),
        FEATURE_NAMES.len(),
        "feature count differs between the trainer and this build"
    );
    for (i, n) in pf.features.iter().enumerate() {
        assert_eq!(
            n, FEATURE_NAMES[i],
            "feature {} is {:?} in the trainer but {:?} here",
            i, n, FEATURE_NAMES[i]
        );
    }

    assert!(
        pf.rows.len() > 20,
        "parity file has too few rows to be meaningful: {}",
        pf.rows.len()
    );

    // Replay the identical stream: feed bar 0 upward and read what the engine emits.
    // Because the parity file holds the full series, the EMA seeding matches Python's
    // exactly rather than approximating it.
    let mut fe = FeatureEngine::new();
    let mut cursor = 0usize;
    let mut latest: Option<[f64; 10]> = None;
    let mut feature_worst = 0.0f64;
    let mut pred_worst = 0.0f64;
    let mut checked = 0usize;

    for row in &pf.rows {
        while cursor <= row.bar_index {
            let b = &pf.bars[cursor];
            let bar = botcore::Bar {
                symbol: botcore::SymbolId(1),
                ts: cursor as i64,
                open: b.open,
                high: b.high,
                low: b.low,
                close: b.close,
                volume: b.volume,
            };
            latest = fe.on_bar(&bar);
            cursor += 1;
        }

        let x = latest.unwrap_or_else(|| {
            panic!(
                "no features at bar {} after {} bars - the window is too short",
                row.bar_index,
                fe.bars_seen()
            )
        });

        assert_eq!(x.len(), row.x.len(), "feature vector length mismatch");
        for j in 0..x.len() {
            let d = (x[j] - row.x[j]).abs();
            if d > feature_worst {
                feature_worst = d;
            }
        }

        let p = pf.model.predict(&x);
        let d = (p - row.p).abs();
        if d > pred_worst {
            pred_worst = d;
        }
        checked += 1;
    }

    eprintln!(
        "  parity: {} on {} rows across {} bars - worst feature delta {:.3e}, worst probability delta {:.3e}",
        pf.symbol,
        checked,
        pf.bars.len(),
        feature_worst,
        pred_worst
    );

    assert!(
        feature_worst < 1e-9,
        "features disagree with Python by {:.3e} - the two implementations have drifted",
        feature_worst
    );
    assert!(
        pred_worst < 1e-9,
        "predictions disagree with Python by {:.3e}",
        pred_worst
    );
}

#[test]
fn every_exported_model_loads_and_predicts() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("models");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        eprintln!("SKIPPED: {} not found", dir.display());
        return;
    };
    let mut seen = 0;
    for e in rd.flatten() {
        let p = e.path();
        let name = p
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if !name.ends_with(".json") || name.starts_with('_') || name == "parity.json" {
            continue;
        }
        let mf = ModelFile::load(&p).expect("model file should load");
        mf.check_feature_names()
            .expect("feature names should match");
        assert!(mf.bars > 0, "{} says it trained on 0 bars", name);
        assert!(
            mf.horizon > 0 && mf.horizon < 100,
            "{} has an implausible horizon {}",
            name,
            mf.horizon
        );
        // a prediction on a neutral vector must be a usable probability
        let x = vec![0.0; mf.features.len()];
        let p = mf.model.predict(&x);
        assert!(p.is_finite(), "{} produced a non-finite prediction", name);
        if mf.model.kind() != "linearreg" {
            assert!((0.0..=1.0).contains(&p), "{} predicted {}", name, p);
        }
        assert_eq!(mf.model.influence().len(), mf.features.len());
        seen += 1;
    }
    eprintln!("  loaded and predicted from {} exported models", seen);
}
