"""
Trains the models and exports everything Rust needs.

This is the Python half of the split: research and training here, inference and
execution in Rust. Three artifacts come out of a run:

  engine/data/<SYM>.csv     OHLCV bars, for the Rust feed to ingest
  engine/models/<SYM>.json  the trained model, with feature names and weights
  engine/models/parity.json bars plus Python's own features and predictions

The third one is the important one. Feature engineering exists twice - once here and
once in Rust - and the parity file is how that duplication is kept honest: the Rust
test recomputes features from the same bars and fails if it disagrees.

Dependencies: numpy only. Deliberately not torch/sklearn - the models here are small
enough that hand-written gradient descent is clearer, faster to run and easier to
verify than a framework, and it keeps the toolchain to what is already installed.

RUN
    python3 research/train.py            # all symbols
    python3 research/train.py SPY        # one symbol
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_JSON = os.path.join(HERE, "market2.json")
OUT_DATA = os.path.join(ROOT, "engine", "data")
OUT_MODELS = os.path.join(ROOT, "engine", "models")

FEATURE_NAMES = [
    "1-bar return", "5-bar return", "20-bar return", "RSI(14) position",
    "MACD histogram", "Bollinger %B", "SMA20-50 trend", "Volatility ratio",
    "Volume z-score", "Range position",
]
WARMUP = 60
HORIZON = 10           # bars ahead the label looks
TRAIN_FRAC = 0.7       # last 30% is never trained on


# ------------------------------------------------------------------ features
def sma(x, n):
    """Rolling mean; index i uses x[i-n+1 .. i]. NaN before that."""
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def stdev(x, n):
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))
    m = (c[n:] - c[:-n]) / n
    v = (c2[n:] - c2[:-n]) / n - m * m
    out[n - 1:] = np.sqrt(np.maximum(v, 0.0))
    return out


def ema_series(x, n):
    """EMA seeded with the simple mean of the first n values, then recursive.
    Matches the Rust implementation in crates/model exactly."""
    k = 2.0 / (n + 1.0)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    prev = float(np.mean(x[:n]))
    out[n - 1] = prev
    for i in range(n, len(x)):
        prev = x[i] * k + prev * (1.0 - k)
        out[i] = prev
    return out


def rsi_series(x, n=14):
    """Wilder RSI. Uses Wilder smoothing, converging from a simple-average seed."""
    out = np.full(len(x), np.nan)
    if len(x) <= n:
        return out
    d = np.diff(x)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    g = float(np.mean(gain[:n]))
    l = float(np.mean(loss[:n]))
    out[n] = 100.0 if l < 1e-12 else 100.0 - 100.0 / (1.0 + g / l)
    for i in range(n, len(d)):
        g = (g * (n - 1) + gain[i]) / n
        l = (l * (n - 1) + loss[i]) / n
        out[i + 1] = 100.0 if l < 1e-12 else 100.0 - 100.0 / (1.0 + g / l)
    return out


def macd_line_series(c):
    """EMA(12) - EMA(26), defined from index 25 onward."""
    ef, es = ema_series(c, 12), ema_series(c, 26)
    return ef - es


def macd_hist(c):
    line = macd_line_series(c)
    valid = line[25:]
    if len(valid) < 9:
        return np.full(len(c), np.nan)
    sig = ema_series(valid, 9)
    hist = np.full(len(c), np.nan)
    hist[25:] = valid - sig
    return hist


def realized_vol(c, n):
    """Annualised stdev of log returns over a window, matching the Rust version."""
    out = np.full(len(c), np.nan)
    lr = np.full(len(c), np.nan)
    lr[1:] = np.log(c[1:] / c[:-1])
    for i in range(n, len(c)):
        w = lr[i - n + 1:i + 1]
        if np.any(np.isnan(w)):
            continue
        m = w.mean()
        out[i] = np.sqrt(((w - m) ** 2).mean()) * np.sqrt(252.0)
    return out


def volume_z(v, n=20):
    m, s = sma(v, n), stdev(v, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (v - m) / s
    z[s <= 1e-12] = 0.0
    return z


def clip(v, a, b):
    return np.clip(v, a, b)


def build_features(o, h, l, c, v):
    """The ten features, one row per usable bar. NaN rows are dropped."""
    n = len(c)
    rsi = rsi_series(c, 14)
    mh = macd_hist(c)
    s20, s50 = sma(c, 20), sma(c, 50)
    sd20 = stdev(c, 20)
    m20 = sma(c, 20)
    v10, v30 = realized_vol(c, 10), realized_vol(c, 30)
    vz = volume_z(v, 20)

    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.full(n, np.nan); r1[1:] = c[1:] / c[:-1] - 1.0
        r5 = np.full(n, np.nan); r5[5:] = c[5:] / c[:-5] - 1.0
        r20 = np.full(n, np.nan); r20[20:] = c[20:] / c[:-20] - 1.0
        pb = (c - (m20 - 2 * sd20)) / (4 * sd20)
        trend = (s20 - s50) / np.abs(s50)
        vol_ratio = v10 / v30
        rng = np.where(h > l, (c - l) / (h - l), 0.5)

    X = np.column_stack([
        clip(r1 * 100.0, -10, 10),
        clip(r5 * 40.0, -10, 10),
        clip(r20 * 12.0, -10, 10),
        (rsi - 50.0) / 50.0,
        clip(mh / np.maximum(sd20, 1e-12), -6, 6),
        clip((pb - 0.5) * 2.5, -5, 5),
        clip(trend * 25.0, -10, 10),
        clip((vol_ratio - 1.0) * 2.0, -5, 5),
        clip(vz, -4, 4),
        clip((rng - 0.5) * 3.0, -3, 3),
    ])
    # NaN anywhere means the row is unusable
    good = ~np.isnan(X).any(axis=1) & ~np.isnan(c)
    good[:WARMUP + 1] = False
    return X, good


def labels(c, idx, horizon):
    """1 if price is higher `horizon` bars later. Only defined where that bar exists."""
    ok = idx + horizon < len(c)
    y = (c[idx[ok] + horizon] > c[idx[ok]]).astype(np.float64)
    return y, ok


# -------------------------------------------------------------------- models
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_logistic(X, y, epochs=400, lr=0.05, l2=5e-4, seed=1):
    """Full-batch gradient descent. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        err = p - y
        w -= lr * (X.T @ err / len(y) + l2 * w)
        b -= lr * float(err.mean())
    return {"kind": "logistic", "weights": w, "bias": b}


def train_linear(X, y, epochs=400, lr=0.05, l2=5e-4, seed=2):
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        err = (X @ w + b) - y
        w -= lr * (X.T @ err / len(y) + l2 * w)
        b -= lr * float(err.mean())
    return {"kind": "linearreg", "weights": w, "bias": b}


def train_mlp(X, y, hidden=8, epochs=1500, lr=0.05, l2=5e-4, seed=3):
    """One hidden layer with tanh and a sigmoid output, trained by backprop.
    The forward pass mirrors Model::predict in Rust exactly."""
    rng = np.random.default_rng(seed)
    nf = X.shape[1]
    w1 = (rng.random((hidden, nf)) - 0.5) * (2.0 / np.sqrt(nf))
    b1 = np.zeros(hidden)
    w2 = (rng.random(hidden) - 0.5) * (2.0 / np.sqrt(hidden))
    b2 = 0.0
    for _ in range(epochs):
        z1 = X @ w1.T + b1
        a1 = np.tanh(z1)
        z2 = a1 @ w2 + b2
        p = sigmoid(z2)
        d2 = (p - y)
        gw2 = a1.T @ d2 / len(y) + l2 * w2
        gb2 = float(d2.mean())
        da1 = np.outer(d2, w2) * (1 - a1 ** 2)
        gw1 = da1.T @ X / len(y) + l2 * w1
        gb1 = da1.mean(axis=0)
        w1 -= lr * gw1
        b1 -= lr * gb1
        w2 -= lr * gw2
        b2 -= lr * gb2
    return {"kind": "mlp", "w1": w1, "b1": b1, "w2": w2, "b2": b2}


def predict(m, X):
    if m["kind"] in ("logistic", "linearreg"):
        z = X @ np.asarray(m["weights"]) + m["bias"]
        return sigmoid(z) if m["kind"] == "logistic" else z
    a1 = np.tanh(X @ np.asarray(m["w1"]).T + np.asarray(m["b1"]))
    z2 = a1 @ np.asarray(m["w2"]) + m["b2"]
    return sigmoid(z2)


# ---------------------------------------------------------------------- main
def run(sym, blob):
    o = np.asarray(blob["open"], float)
    h = np.asarray(blob["high"], float)
    l = np.asarray(blob["low"], float)
    c = np.asarray(blob["closes"], float)
    v = np.asarray(blob["volume"], float)
    dates = blob["dates"]

    # ---- CSV for the Rust feed ----
    os.makedirs(OUT_DATA, exist_ok=True)
    csv_path = os.path.join(OUT_DATA, f"{sym}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("date,open,high,low,close,volume\n")
        for i in range(len(c)):
            f.write(f"{dates[i]},{o[i]:.4f},{h[i]:.4f},{l[i]:.4f},{c[i]:.4f},{v[i]:.0f}\n")

    X, good = build_features(o, h, l, c, v)
    idx = np.where(good)[0]
    y, ok = labels(c, idx, HORIZON)
    idx = idx[ok]
    Xg = X[idx]

    if len(Xg) < 120:
        print(f"  {sym:<9} only {len(Xg)} usable rows, skipped")
        return None

    cut = int(len(Xg) * TRAIN_FRAC)
    Xtr, ytr = Xg[:cut], y[:cut]

    models = {
        "logistic": train_logistic(Xtr, ytr),
        "linear": train_linear(Xtr, ytr),
        "mlp": train_mlp(Xtr, ytr),
    }

    # in-sample / out-of-sample accuracy, so overfitting is visible in the training log
    report = {}
    for name, m in models.items():
        ptr = predict(m, Xtr)
        pte = predict(m, Xg[cut:])
        acc = lambda p, yy: float(((p > 0.5) == (yy > 0.5)).mean()) if len(yy) else float("nan")
        report[name] = (acc(ptr, ytr), acc(pte, y[cut:]))

    # ---- export: primary model is the logistic one (fastest, most stable) ----
    os.makedirs(OUT_MODELS, exist_ok=True)
    primary = models["logistic"]
    mf = {
        "version": 1,
        "trained_on": f"{sym} {dates[0]}..{dates[-1]}",
        "bars": int(len(c)),
        "horizon": HORIZON,
        "features": FEATURE_NAMES,
        "model": {
            "kind": primary["kind"],
            "weights": [float(x) for x in primary["weights"]],
            "bias": float(primary["bias"]),
        },
    }
    with open(os.path.join(OUT_MODELS, f"{sym}.json"), "w") as f:
        json.dump(mf, f, indent=1)

    # ---- parity file: bars, Python's features, and Python's predictions ----
    # The FULL series from bar 0, indexed absolutely. EMA-based indicators are seeded
    # from the start of the available data, so a parity check that began part-way
    # through would compare two different seedings and report a false mismatch.
    start = 0
    bars = [{
        "ts": dates[i], "open": float(o[i]), "high": float(h[i]),
        "low": float(l[i]), "close": float(c[i]), "volume": float(v[i]),
    } for i in range(start, len(c))]

    parity_rows = []
    for k, i in enumerate(idx):
        if k % 7:            # a sample is enough, and keeps the file small
            continue
        parity_rows.append({
            "bar_index": int(i),
            "x": [float(t) for t in X[i]],
            "p": float(predict(primary, X[i:i + 1])[0]),
        })

    with open(os.path.join(OUT_MODELS, "parity.json"), "w") as f:
        json.dump({"symbol": sym, "horizon": HORIZON, "bars": bars,
                   "rows": parity_rows,
                   "model": mf["model"],
                   "features": FEATURE_NAMES}, f, indent=1)

    return {
        "sym": sym, "rows": len(Xg), "train": cut, "test": len(Xg) - cut,
        "report": report, "parity": len(parity_rows),
        "balance": float(y.mean()),
    }


def main():
    blob = json.load(open(DATA_JSON))
    want = sys.argv[1:] or list(blob.keys())
    print(f"  training on {len(want)} symbols, horizon {HORIZON}, warmup {WARMUP}")
    print()
    print(f"  {'symbol':<9}{'rows':>6}{'train':>7}{'test':>6}  "
          f"{'logistic tr/te':>18}{'mlp tr/te':>18}{'up-bias':>9}")
    out = {}
    for sym in want:
        if sym not in blob:
            print(f"  {sym}: not in market2.json")
            continue
        r = run(sym, blob[sym])
        if not r:
            continue
        out[sym] = r
        lg = r["report"]["logistic"]
        ml = r["report"]["mlp"]
        print(f"  {sym:<9}{r['rows']:>6}{r['train']:>7}{r['test']:>6}  "
              f"{lg[0]*100:>8.1f}%{lg[1]*100:>8.1f}%"
              f"{ml[0]*100:>8.1f}%{ml[1]*100:>8.1f}%"
              f"{r['balance']*100:>8.1f}%")
    print()
    print("  a model whose test accuracy is far below its train accuracy is overfit")
    json.dump(out, open(os.path.join(OUT_MODELS, "_training_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
