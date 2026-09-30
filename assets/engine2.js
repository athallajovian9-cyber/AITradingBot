/* =====================================================================
   ENGINE v2 — advanced, and honest about it.

   Everything here is a real implementation: real indicators, a neural net
   with backprop, walk-forward validation with an embargo, a cost model,
   volatility targeting, ATR stops, Monte Carlo and a full metric suite.

   The rules, unchanged from v1 and enforced by tests:
     1. No lookahead anywhere. Every model is fit only on data whose
        outcome has already happened.
     2. Costs are always charged.
     3. The benchmark is always computed on the same capital and path.
     4. Anything the code cannot know (e.g. OHLC when only closes exist)
        degrades explicitly rather than being invented.
   ===================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ENGINE = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const EPS = 1e-9;
  const clip = (v, a, b) => v < a ? a : (v > b ? b : v);
  const isNum = v => typeof v === 'number' && isFinite(v);

  /* ================================================== deterministic RNG
     A real RNG with a seed, so Monte Carlo results are reproducible and a
     reported confidence interval can be re-checked. Math.random() cannot. */
  function rng(seed) {
    let s = (seed >>> 0) || 1;
    return function () {
      s ^= s << 13; s >>>= 0;
      s ^= s >> 17;
      s ^= s << 5;  s >>>= 0;
      return s / 4294967296;
    };
  }

  /* =========================================================== indicators */
  function sma(x, n, i) {
    if (i + 1 < n) return null;
    let s = 0;
    for (let k = i - n + 1; k <= i; k++) s += x[k];
    return s / n;
  }

  function emaSeries(x, n) {
    const k = 2 / (n + 1), out = new Array(x.length).fill(null);
    let prev = null;
    for (let i = 0; i < x.length; i++) {
      if (i + 1 < n) continue;
      if (prev === null) {
        let s = 0;
        for (let j = i - n + 1; j <= i; j++) s += x[j];
        prev = s / n;
      } else prev = x[i] * k + prev * (1 - k);
      out[i] = prev;
    }
    return out;
  }

  /* Wilder RSI. Reproduces the published worked example (see tests). */
  function rsiSeries(x, n) {
    const out = new Array(x.length).fill(null);
    if (x.length <= n) return out;
    let g = 0, l = 0;
    for (let i = 1; i <= n; i++) { const d = x[i] - x[i - 1]; if (d >= 0) g += d; else l -= d; }
    g /= n; l /= n;
    out[n] = l < EPS ? 100 : 100 - 100 / (1 + g / l);
    for (let i = n + 1; i < x.length; i++) {
      const d = x[i] - x[i - 1];
      g = (g * (n - 1) + (d > 0 ? d : 0)) / n;
      l = (l * (n - 1) + (d < 0 ? -d : 0)) / n;
      out[i] = l < EPS ? 100 : 100 - 100 / (1 + g / l);
    }
    return out;
  }

  function macdSeries(x, f, s, sig) {
    const ef = emaSeries(x, f), es = emaSeries(x, s);
    const line = x.map((_, i) => (ef[i] != null && es[i] != null) ? ef[i] - es[i] : null);
    const filled = line.map(v => v == null ? 0 : v);
    const sigl = emaSeries(filled, sig);
    const hist = line.map((v, i) => (v != null && sigl[i] != null) ? v - sigl[i] : null);
    return { line, signal: sigl, hist };
  }

  /* Wilder's Average True Range. Needs real high/low; returns nulls without
     them rather than substituting a close-to-close range and calling it ATR. */
  function atrSeries(high, low, close, n) {
    const out = new Array(close.length).fill(null);
    if (!high || !low) return out;
    const tr = new Array(close.length).fill(null);
    for (let i = 1; i < close.length; i++) {
      const a = high[i] - low[i];
      const b = Math.abs(high[i] - close[i - 1]);
      const c = Math.abs(low[i] - close[i - 1]);
      tr[i] = Math.max(a, b, c);
    }
    if (close.length <= n + 1) return out;
    let acc = 0;
    for (let i = 1; i <= n; i++) acc += tr[i];
    let prev = acc / n;
    out[n] = prev;
    for (let i = n + 1; i < close.length; i++) {
      prev = (prev * (n - 1) + tr[i]) / n;
      out[i] = prev;
    }
    return out;
  }

  function stdev(x, n, i) {
    if (i + 1 < n) return null;
    let m = 0;
    for (let k = i - n + 1; k <= i; k++) m += x[k];
    m /= n;
    let v = 0;
    for (let k = i - n + 1; k <= i; k++) v += (x[k] - m) ** 2;
    return Math.sqrt(v / n);
  }

  /* Bollinger %B: where price sits inside the band, 0 = lower, 1 = upper. */
  function percentB(close, n, k) {
    const out = new Array(close.length).fill(null);
    for (let i = 0; i < close.length; i++) {
      const m = sma(close, n, i), sd = stdev(close, n, i);
      if (m == null || sd == null || sd < EPS) continue;
      out[i] = (close[i] - (m - k * sd)) / (2 * k * sd);
    }
    return out;
  }

  /* Realised volatility of log returns, annualised. */
  function realizedVol(close, n) {
    const out = new Array(close.length).fill(null);
    const r = logReturns(close);
    for (let i = 0; i < close.length; i++) {
      if (i + 1 < n || r[i] == null) continue;
      let m = 0, cnt = 0;
      for (let k = i - n + 1; k <= i; k++) { if (r[k] != null) { m += r[k]; cnt++; } }
      if (!cnt) continue;
      m /= cnt;
      let v = 0;
      for (let k = i - n + 1; k <= i; k++) if (r[k] != null) v += (r[k] - m) ** 2;
      out[i] = Math.sqrt(v / cnt) * Math.sqrt(252);
    }
    return out;
  }

  function logReturns(close) {
    const out = new Array(close.length).fill(null);
    for (let i = 1; i < close.length; i++)
      if (close[i - 1] > 0) out[i] = Math.log(close[i] / close[i - 1]);
    return out;
  }

  /* Volume z-score: is today's activity unusual for this asset? */
  function volumeZ(volume, n) {
    const out = new Array((volume || []).length).fill(null);
    if (!volume) return out;
    for (let i = 0; i < volume.length; i++) {
      const m = sma(volume, n, i), sd = stdev(volume, n, i);
      if (m == null || sd == null || sd < EPS) continue;
      out[i] = (volume[i] - m) / sd;
    }
    return out;
  }

  /* ====================================================== feature builder

     Every feature at bar i uses bars <= i only. The names are returned with
     the matrix so the UI can show which inputs the model leaned on.
  */
  const FEATURE_NAMES = [
    '1-bar return', '5-bar return', '20-bar return',
    'RSI(14) position', 'MACD histogram', 'Bollinger %B',
    'SMA20-50 trend', 'Volatility ratio', 'Volume z-score', 'Range position',
  ];

  function buildFeatures(d) {
    const c = d.closes, n = c.length;
    const rsi = rsiSeries(c, 14);
    const macd = macdSeries(c, 12, 26, 9);
    const pb = percentB(c, 20, 2);
    const s20 = c.map((_, i) => sma(c, 20, i));
    const s50 = c.map((_, i) => sma(c, 50, i));
    const vol10 = realizedVol(c, 10), vol30 = realizedVol(c, 30);
    const vz = volumeZ(d.volume, 20);
    const hasHL = !!(d.high && d.low);

    const rows = [];               // { i, x: [], names }
    const raw = { atr: atrSeries(d.high, d.low, c, 14) };

    for (let i = 0; i < n; i++) {
      if (i < 60) continue;
      if (rsi[i] == null || macd.hist[i] == null || pb[i] == null) continue;
      if (s20[i] == null || s50[i] == null || vol10[i] == null || vol30[i] == null) continue;
      if (c[i - 1] <= 0 || c[i - 5] <= 0 || c[i - 20] <= 0) continue;

      const r1 = c[i] / c[i - 1] - 1;
      const r5 = c[i] / c[i - 5] - 1;
      const r20 = c[i] / c[i - 20] - 1;
      const sd20 = stdev(c, 20, i) || 1;
      const trend = (s20[i] - s50[i]) / (s50[i] || 1);
      const volRatio = (vol10[i] || 0) / (vol30[i] || 1);
      const rangePos = hasHL && d.high[i] > d.low[i]
        ? (c[i] - d.low[i]) / (d.high[i] - d.low[i]) : 0.5;

      // scaled so no single input dominates the gradient
      const x = [
        clip(r1 * 100, -10, 10),
        clip(r5 * 40, -10, 10),
        clip(r20 * 12, -10, 10),
        (rsi[i] - 50) / 50,
        clip(macd.hist[i] / sd20, -6, 6),
        clip((pb[i] - 0.5) * 2.5, -5, 5),
        clip(trend * 25, -10, 10),
        clip((volRatio - 1) * 2, -5, 5),
        clip((vz[i] == null ? 0 : vz[i]), -4, 4),
        clip((rangePos - 0.5) * 3, -3, 3),
      ];
      rows.push({ i, x });
    }
    return { rows, names: FEATURE_NAMES, atr: raw.atr, hasHL, rsi, macd, pb, vol10, vol30 };
  }

  /* =============================================================== models

     All models stream: they are updated one bar at a time and are only ever
     asked to predict a bar whose outcome is not yet known.
  */

  function makeLogistic(nF, lr, l2) {
    return { kind: 'logistic', w: new Array(nF).fill(0), b: 0,
             lr: lr || 0.05, l2: l2 == null ? 0.0005 : l2, n: 0, loss: 0 };
  }

  function makeLinear(nF, lr, l2) {
    return { kind: 'linear', w: new Array(nF).fill(0), b: 0,
             lr: lr || 0.01, l2: l2 == null ? 0.0005 : l2, n: 0, loss: 0 };
  }

  /* A genuine feed-forward net: one hidden layer, tanh, sigmoid output, and
     backprop. Small enough to stream online, real enough to have weights that
     mean something - which is the difference between this and Math.sin(). */
  function makeMLP(nF, hidden, lr, l2) {
    const H = hidden || 8;
    const r = rng(7);
    const W1 = [], b1 = new Array(H).fill(0);
    for (let h = 0; h < H; h++) {
      W1.push(new Array(nF).fill(0).map(() => (r() - 0.5) * (2 / Math.sqrt(nF))));
    }
    const W2 = new Array(H).fill(0).map(() => (r() - 0.5) * (2 / Math.sqrt(H)));
    return { kind: 'mlp', H, W1, b1, W2, b2: 0,
             lr: lr == null ? 0.02 : lr, l2: l2 == null ? 0.0005 : l2, n: 0, loss: 0 };
  }

  function forward(m, x) {
    if (m.kind === 'mlp') {
      const a = new Array(m.H), z = new Array(m.H);
      for (let h = 0; h < m.H; h++) {
        let s = m.b1[h];
        for (let j = 0; j < x.length; j++) s += m.W1[h][j] * x[j];
        z[h] = s; a[h] = Math.tanh(s);
      }
      let o = m.b2;
      for (let h = 0; h < m.H; h++) o += m.W2[h] * a[h];
      return { p: 1 / (1 + Math.exp(-clip(o, -30, 30))), a, z };
    }
    let s = m.b;
    for (let j = 0; j < x.length; j++) s += m.w[j] * x[j];
    return { p: 1 / (1 + Math.exp(-clip(s, -30, 30))), raw: s };
  }

  function predict(m, x) { return forward(m, x).p; }

  function learnOne(m, x, y, sampleW) {
    const sw = sampleW == null ? 1 : sampleW;
    const f = forward(m, x);
    const p = f.p;
    const d = (p - y) * sw;
    if (m.kind === 'mlp') {
      // output layer
      const dOut = d;
      const g2 = new Array(m.H);
      for (let h = 0; h < m.H; h++) {
        g2[h] = dOut * f.a[h];
        m.W2[h] -= m.lr * (g2[h] + m.l2 * m.W2[h]);
      }
      m.b2 -= m.lr * dOut;
      // hidden layer
      for (let h = 0; h < m.H; h++) {
        const dh = dOut * m.W2[h] * (1 - f.a[h] * f.a[h]);   // tanh'
        for (let j = 0; j < x.length; j++)
          m.W1[h][j] -= m.lr * (dh * x[j] + m.l2 * m.W1[h][j]);
        m.b1[h] -= m.lr * dh;
      }
    } else {
      for (let j = 0; j < x.length; j++)
        m.w[j] -= m.lr * (d * x[j] + m.l2 * m.w[j]);
      m.b -= m.lr * d;
    }
    m.n++;
    m.loss += -(y * Math.log(p + EPS) + (1 - y) * Math.log(1 - p + EPS));
  }

  function modelMeanLoss(m) { return m.n ? m.loss / m.n : NaN; }

  /* Feature influence for display: for linear models it is the coefficient,
     for the net it is the summed absolute contribution through the hidden
     layer, normalised so the bars are comparable. Reported, not invented. */
  function featureInfluence(m) {
    if (!m) return [];
    const nF = m.kind === 'mlp' ? m.W1[0].length : m.w.length;
    const out = new Array(nF).fill(0);
    if (m.kind === 'mlp') {
      for (let j = 0; j < nF; j++)
        for (let h = 0; h < m.H; h++) out[j] += m.W1[h][j] * m.W2[h];
    } else {
      for (let j = 0; j < nF; j++) out[j] = m.w[j];
    }
    return out;
  }

  function makeModel(kind, nF, lr) {
    if (kind === 'mlp') return makeMLP(nF, 8, lr == null ? 0.02 : lr);
    if (kind === 'linear') return makeLinear(nF, lr == null ? 0.02 : lr);
    return makeLogistic(nF, lr == null ? 0.05 : lr);
  }

  /* ========================================================== cost model */
  function makeCost(cfg) {
    const commission = (cfg.commissionBps == null ? 10 : cfg.commissionBps) / 10000;
    const spread = (cfg.spreadBps == null ? 5 : cfg.spreadBps) / 10000;
    const slip = (cfg.slippageBps == null ? 5 : cfg.slippageBps) / 10000;
    const impact = (cfg.impactBps == null ? 0 : cfg.impactBps) / 10000;
    return {
      oneWay: commission / 2 + spread / 2 + slip,
      commission, spread, slip, impact,
      // market impact scales with how much of the book you take
      forOrder(notional, liquidity) {
        const base = commission + spread / 2 + slip;
        if (impact <= 0 || !liquidity || liquidity <= 0) return base;
        return base + impact * Math.sqrt(notional / liquidity);
      },
    };
  }

  /* ======================================================= position sizing */
  function sizePosition(mode, cfg, ctx) {
    if (mode === 'voltarget') {
      const target = cfg.targetVol || 0.15;
      const v = ctx.vol || target;
      return clip(target / Math.max(v, 0.02), 0, cfg.maxLeverage || 1);
    }
    if (mode === 'kelly') {
      // fraction = edge / odds, halved for safety, floored at zero
      const p = ctx.winProb, b = ctx.payoff || 1;
      if (!isNum(p) || b <= 0) return 0;
      const k = (p * (b + 1) - 1) / b;
      return clip(k * (cfg.kellyFraction || 0.5), 0, cfg.maxLeverage || 1);
    }
    return clip(cfg.fraction == null ? 1 : cfg.fraction, 0, cfg.maxLeverage || 1);
  }

  /* ================================================== core bar-by-bar loop

     One engine, used by both the single backtest and the walk-forward, so a
     walk-forward cannot accidentally use different rules from a plain run.
  */
  function runSeries(d, cfg, sink) {
    const c = d.closes, n = c.length;
    const f = buildFeatures(d);
    const cost = makeCost(cfg);
    const mode = cfg.model || 'logistic';
    const lr = cfg.lr == null ? 'auto' : cfg.lr;
    const horizon = Math.max(1, cfg.horizon == null ? 10 : cfg.horizon);
    const thresh = cfg.thresh == null ? 0.55 : cfg.thresh;
    const allowShort = !!cfg.allowShort;
    const sizing = cfg.sizing || 'full';
    const stopAtr = cfg.stopAtr == null ? 0 : cfg.stopAtr;
    const trailAtr = cfg.trailAtr == null ? 0 : cfg.trailAtr;
    const takeProfitAtr = cfg.tpAtr == null ? 0 : cfg.tpAtr;
    const maxHold = cfg.maxHold == null ? 0 : cfg.maxHold;
    const warmBars = cfg.warmBars == null ? 60 : cfg.warmBars;

    const startCash = cfg.cash == null ? 10000 : cfg.cash;
    let cash = startCash, units = 0, side = 0;      // side: 0 flat, +1 long, -1 short
    let entryPx = 0, entryBar = -1, entryEquity = startCash, highWater = 0, lowWater = 0;
    const equity = new Array(n).fill(startCash);
    const pending = [];                              // samples awaiting a label
    const trades = [];
    const influenceAt = [];
    let fees = 0, tradesWon = 0, tradesLost = 0, grossWin = 0, grossLoss = 0;
    let barsInMarket = 0, turnover = 0;

    // the model is (re)created here so a walk-forward window starts clean
    let model = null;
    const infl = [];
    const featureNames = f.names;

    const positions = f.rows.map(r => r.i);
    const rowByBar = {};
    for (const r of f.rows) rowByBar[r.i] = r;

    for (let i = 0; i < n; i++) {
      // ---- 1. learn from any sample whose horizon has elapsed ----
      while (pending.length && i - pending[0].i >= horizon) {
        const s = pending.shift();
        const y = c[s.i + horizon] > c[s.i] ? 1 : 0;
        learnOne(model, s.x, y);
      }

      const px = c[i];
      const row = rowByBar[i];
      const atr = f.atr[i];
      const vol = f.vol30[i];

      // ---- 2. exits first (a stop must be able to fire) ----
      if (side !== 0 && row) {
        const held = i - entryBar;
        const stopPx = stopAtr > 0 && atr ? (side > 0 ? entryPx - stopAtr * atr
                                                       : entryPx + stopAtr * atr) : null;
        const trailPx = trailAtr > 0 && atr
          ? (side > 0 ? highWater - trailAtr * atr : lowWater + trailAtr * atr) : null;
        const tpPx = takeProfitAtr > 0 && atr
          ? (side > 0 ? entryPx + takeProfitAtr * atr : entryPx - takeProfitAtr * atr) : null;
        let exit = false, why = '';
        // intrabar: use the bar's low/high to see whether a stop was touched
        if (side > 0) {
          if (stopPx != null && d.low && d.low[i] <= stopPx) { exit = true; why = 'stop'; }
          else if (trailPx != null && d.low && d.low[i] <= trailPx) { exit = true; why = 'trail'; }
          else if (tpPx != null && d.high && d.high[i] >= tpPx) { exit = true; why = 'target'; }
        } else {
          if (stopPx != null && d.high && d.high[i] >= stopPx) { exit = true; why = 'stop'; }
          else if (trailPx != null && d.high && d.high[i] >= trailPx) { exit = true; why = 'trail'; }
          else if (tpPx != null && d.low && d.low[i] <= tpPx) { exit = true; why = 'target'; }
        }
        if (!exit && maxHold > 0 && held >= maxHold) { exit = true; why = 'time'; }
        if (!exit) {
          // signal flip also closes
          const p = predict(model, row.x);
          const want = p > thresh ? 1 : (allowShort && p < 1 - thresh ? -1 : 0);
          if (want !== side) { exit = true; why = 'signal'; }
        }
        if (exit) {
          const rate = cost.oneWay;
          const fill = side > 0 ? px * (1 - rate) : px * (1 + rate);
          fees += Math.abs(units * px) * rate;
          const pnl = side * units * (fill - entryPx);   // side-aware; fills carry cost
          if (pnl > 0) { tradesWon++; grossWin += pnl; } else { tradesLost++; grossLoss += -pnl; }
          trades.push({ entry: entryBar, exit: i, pnl, side, why,
                        ret: entryEquity > 0 ? pnl / entryEquity : 0, held });
          cash = entryEquity + pnl;
          units = 0; side = 0; entryPx = 0; highWater = 0; lowWater = 0;
        }
      }

      // ---- 3. entries ----
      if (side === 0 && row && i >= warmBars) {
        const p = predict(model, row.x);
        let want = 0;
        if (p > thresh) want = 1;
        else if (allowShort && p < 1 - thresh) want = -1;
        if (want !== 0) {
          const fraction = sizePosition(sizing, cfg, { vol, winProb: p, payoff: 1 });
          const notional = cash * fraction;
          if (notional > startCash * 0.005) {
            const rate = cost.oneWay;
            const fill = want > 0 ? px * (1 + rate) : px * (1 - rate);
            units = notional / fill;
            side = want; entryPx = fill; entryBar = i;
            entryEquity = cash;                  // account value at entry
            highWater = px; lowWater = px;
            fees += notional * rate;
            turnover += notional;
          }
        }
      }

      // ---- 4. queue the sample for learning (label not yet known) ----
      if (row && i + horizon < n) {
        pending.push({ x: row.x, i });
        // the model must exist before learning; create on first sight
      }
      if (!model) model = makeModel(mode, featureNames.length, lr === 'auto' ? undefined : lr);

      // ---- 5. mark to market ----
      equity[i] = side === 0 ? cash : entryEquity + side * units * (px - entryPx);
      if (side !== 0) {
        barsInMarket++;
        highWater = Math.max(highWater, px);
        lowWater = lowWater === 0 ? px : Math.min(lowWater, px);
      }
      if (i % 25 === 0 && model) infl.push({ i, v: featureInfluence(model) });

      if (sink) sink(i, equity[i], side);
    }

    if (model) {
      const d2 = featureInfluence(model);
      influenceAt.push({ i: n - 1, v: d2 });
    }

    // benchmark on identical capital from the same start bar
    const benchStart = c[Math.min(warmBars, n - 1)];
    const benchUnits = startCash / (benchStart * (1 + cost.oneWay));
    const bench = c.map((v, i) => i < warmBars ? startCash : benchUnits * v);

    return {
      equity, bench, trades, fees, turnover, barsInMarket,
      wins: tradesWon, losses: tradesLost, grossWin, grossLoss,
      influence: influenceAt, featureNames, finalWeights: model ? featureInfluence(model) : [],
      model, n, startCash, warmBars,
      hasHL: f.hasHL,
    };
  }

  /* ============================================================= metrics */
  function metrics(res) {
    const eq = res.equity, b = res.bench, n = eq.length;
    const start = res.startCash;
    const final = eq[n - 1];
    const totalRet = final / start - 1;
    const benchRet = b[n - 1] / start - 1;

    const rets = [], brets = [];
    for (let i = res.warmBars + 1; i < n; i++) {
      rets.push(eq[i - 1] > 0 ? eq[i] / eq[i - 1] - 1 : 0);
      brets.push(b[i - 1] > 0 ? b[i] / b[i - 1] - 1 : 0);
    }
    const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
    const sd = a => {
      if (a.length < 2) return 0;
      const m = mean(a);
      return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1));
    };
    const mu = mean(rets), sg = sd(rets);
    const down = rets.filter(r => r < 0);
    const dsd = sd(down);
    const ann = Math.sqrt(252);

    let peak = start, maxDD = 0, ddStart = -1, ddDur = 0, worstDur = 0;
    for (let i = 0; i < n; i++) {
      if (eq[i] > peak) { peak = eq[i]; if (ddStart >= 0) { worstDur = Math.max(worstDur, i - ddStart); ddStart = -1; } }
      else if (ddStart < 0 && eq[i] < peak) ddStart = i;
      const d = eq[i] / peak - 1;
      if (d < maxDD) maxDD = d;
    }
    if (ddStart >= 0) worstDur = Math.max(worstDur, n - ddStart);

    // beta / alpha against the benchmark, on the same window
    const bm = mean(brets);
    let cov = 0, varb = 0;
    for (let i = 0; i < rets.length; i++) { cov += (rets[i] - mu) * (brets[i] - bm); varb += (brets[i] - bm) ** 2; }
    const beta = varb > EPS ? cov / varb : 0;
    const alphaAnn = (mu - beta * bm) * 252;
    const activeRets = rets.map((r, i) => r - brets[i]);
    const te = sd(activeRets);
    const informationRatio = te > EPS ? (mean(activeRets) / te) * ann : 0;

    const sorted = rets.slice().sort((a, b2) => a - b2);
    const var95 = sorted.length ? sorted[Math.floor(sorted.length * 0.05)] : 0;

    const years = rets.length / 252;
    const cagr = years > 0 && final > 0 ? Math.pow(final / start, 1 / years) - 1 : 0;
    const sdDown = dsd > EPS ? dsd : 0;
    const sortino = sdDown > EPS ? (mu / sdDown) * ann : 0;
    const calmar = maxDD < -EPS ? cagr / Math.abs(maxDD) : 0;
    const profitFactor = res.grossLoss > EPS ? res.grossWin / res.grossLoss : (res.grossWin > 0 ? Infinity : 0);
    const closed = res.wins + res.losses;

    return {
      final, totalRet, benchRet, excess: totalRet - benchRet,
      beatBench: totalRet > benchRet,
      cagr, sharpe: sg > EPS ? (mu / sg) * ann : 0, sortino, calmar,
      maxDD, maxDDDuration: worstDur,
      beta, alphaAnn, informationRatio, trackingError: te * ann,
      profitFactor, hitRate: closed ? res.wins / closed : 0,
      avgWin: res.wins ? res.grossWin / res.wins : 0,
      avgLoss: res.losses ? res.grossLoss / res.losses : 0,
      trades: res.trades.length, fees: res.fees, turnover: res.turnover,
      exposure: res.barsInMarket / Math.max(1, rets.length),
      var95, bestDay: sorted.length ? sorted[sorted.length - 1] : 0,
      worstDay: sorted.length ? sorted[0] : 0,
      volatility: sg * ann, days: rets.length,
      openPosition: false,
    };
  }

  /* ==================================================== single backtest */
  function backtest(d, cfg) {
    const res = runSeries(d, cfg || {}, null);
    const m = metrics(res);
    return Object.assign(m, {
      equity: res.equity, bench: res.bench, trades: res.trades,
      finalWeights: res.finalWeights, featureNames: res.featureNames,
      influence: res.influence, n: res.n, hasHL: res.hasHL,
      benchFinal: res.bench[res.n - 1],
    });
  }

  /* ==================================================== walk-forward test

     The honest way to validate a strategy: repeatedly fit on a window of the
     past, then trade the NEXT window untouched, and roll forward. The equity
     curve this produces is entirely out-of-sample, which is the only kind of
     result that means anything.

     An embargo of `horizon` bars sits between train and test, so a label
     whose outcome lands in the test window is never trained on.
  */
  function walkForward(d, cfg) {
    cfg = cfg || {};
    const c = d.closes, n = c.length;
    const trainBars = cfg.trainBars || 250;
    const testBars = cfg.testBars || 63;
    const horizon = cfg.horizon == null ? 10 : cfg.horizon;
    const cost = makeCost(cfg);
    const startCash = cfg.cash == null ? 10000 : cfg.cash;

    const features = buildFeatures(d);
    const rowByBar = {};
    for (const r of features.rows) rowByBar[r.i] = r;

    let cash = startCash, units = 0, side = 0, entryPx = 0, entryBar = -1;
    let entryEquity = startCash, highWater = 0, lowWater = 0;
    const equity = new Array(n).fill(startCash);
    const trades = [];
    const windows = [];
    let fees = 0, wins = 0, losses = 0, grossWin = 0, grossLoss = 0, inMkt = 0;

    const thresh = cfg.thresh == null ? 0.55 : cfg.thresh;
    const allowShort = !!cfg.allowShort;
    const sizing = cfg.sizing || 'full';
    const maxHold = cfg.maxHold == null ? 0 : cfg.maxHold;
    const stopAtr = cfg.stopAtr == null ? 0 : cfg.stopAtr;

    for (let t0 = cfg.warmBars || 60; t0 + trainBars + horizon < n; t0 += testBars) {
      const trainEnd = t0 + trainBars;
      const embargoEnd = Math.min(n, trainEnd + horizon);
      const testEnd = Math.min(n, embargoEnd + testBars);

      // ---- fit on the training window only ----
      const model = makeModel(cfg.model || 'logistic', features.names.length,
                              cfg.lr === 'auto' ? undefined : cfg.lr);
      for (let i = t0; i + horizon < trainEnd; i++) {
        const r = rowByBar[i];
        if (!r) continue;
        const y = c[i + horizon] > c[i] ? 1 : 0;
        learnOne(model, r.x, y);
      }

      let acc = 0, tot = 0;
      for (let i = trainEnd; i < trainEnd + 20 && i + horizon < n; i++) {
        const r = rowByBar[i];
        if (!r) continue;
        const y = c[i + horizon] > c[i] ? 1 : 0;
        if ((predict(model, r.x) > 0.5 ? 1 : 0) === y) acc++;
        tot++;
      }
      windows.push({ from: t0, trainEnd, embargoEnd, testEnd,
                     trainAcc: tot ? acc / tot : NaN });

      // ---- trade the test window with that frozen model ----
      for (let i = embargoEnd; i < testEnd; i++) {
        const px = c[i], r = rowByBar[i], atr = features.atr[i];

        if (side !== 0) {
          const stopPx = stopAtr > 0 && atr
            ? (side > 0 ? entryPx - stopAtr * atr : entryPx + stopAtr * atr) : null;
          let exit = false, why = '';
          if (side > 0 && stopPx != null && d.low && d.low[i] <= stopPx) { exit = true; why = 'stop'; }
          if (side < 0 && stopPx != null && d.high && d.high[i] >= stopPx) { exit = true; why = 'stop'; }
          if (!exit && maxHold > 0 && i - entryBar >= maxHold) { exit = true; why = 'time'; }
          if (!exit) {
            const p = predict(model, r ? r.x : new Array(features.names.length).fill(0));
            const want = p > thresh ? 1 : (allowShort && p < 1 - thresh ? -1 : 0);
            if (want !== side) { exit = true; why = 'signal'; }
          }
          if (exit) {
            const rate = cost.oneWay;
            const fill = side > 0 ? px * (1 - rate) : px * (1 + rate);
            fees += Math.abs(units * px) * rate;
            const pnl = side * units * (fill - entryPx);
            if (pnl > 0) { wins++; grossWin += pnl; } else { losses++; grossLoss += -pnl; }
            trades.push({ entry: entryBar, exit: i, pnl, side, why });
            cash = entryEquity + pnl;
            units = 0; side = 0;
          }
        }
        if (side === 0 && r) {
          const p = predict(model, r.x);
          let want = p > thresh ? 1 : (allowShort && p < 1 - thresh ? -1 : 0);
          if (want !== 0) {
            const frac = sizePosition(sizing, cfg, { vol: features.vol30[i], winProb: p, payoff: 1 });
            const notional = cash * frac;
            if (notional > startCash * 0.005) {
              const rate = cost.oneWay;
              const fill = want > 0 ? px * (1 + rate) : px * (1 - rate);
              fees += notional * rate;
              units = notional / fill; side = want; entryPx = fill;
              entryBar = i; entryEquity = cash; highWater = px; lowWater = px;
            }
          }
        }
        if (side !== 0) inMkt++;
        equity[i] = side === 0 ? cash : entryEquity + side * units * (px - entryPx);
      }
      // carry equity flat across gaps
      for (let i = 0; i < n; i++) if (!isFinite(equity[i])) equity[i] = cash;
    }

    // Outside a test window no strategy is running, so carry the last known equity
    // forward. Without this the curve drops to the starting cash in the gaps and the
    // walk-forward looks like it keeps getting reset.
    const inTest = new Array(n).fill(false);
    for (const w of windows) for (let i = w.embargoEnd; i < w.testEnd; i++) inTest[i] = true;
    let carry = startCash;
    for (let i = 0; i < n; i++) {
      if (inTest[i]) carry = equity[i];
      else equity[i] = carry;
    }

    const benchStart = c[Math.min(cfg.warmBars || 60, n - 1)];
    const bu = startCash / (benchStart * (1 + cost.oneWay));
    const bench = c.map((v, i) => i < (cfg.warmBars || 60) ? startCash : bu * v);

    const res = { equity, bench, trades, fees, turnover: 0, barsInMarket: inMkt,
                  wins, losses, grossWin, grossLoss, influence: [],
                  featureNames: features.names, finalWeights: [], n, startCash,
                  warmBars: cfg.warmBars || 60, hasHL: features.hasHL };
    const m = metrics(res);
    return Object.assign(m, { equity, bench, trades, windows, n, res,
                              oosBars: windows.reduce((a, w) => a + (w.testEnd - w.embargoEnd), 0) });
  }

  /* ====================================================== Monte Carlo

     Bootstraps the strategy's own daily returns to show how much of a result
     could be luck. Resamples with replacement, compounding in the same order
     statistics a real account would experience. Reproducible from a seed.
  */
  function monteCarlo(res, runs, seed) {
    runs = runs || 2000;
    const r = rng(seed == null ? 12345 : seed);
    const eq = res.equity, n = eq.length;
    const rets = [];
    for (let i = res.warmBars + 1; i < n; i++)
      rets.push(eq[i - 1] > 0 ? eq[i] / eq[i - 1] - 1 : 0);
    if (rets.length < 20) return null;

    const start = res.startCash;
    const finals = [], maxDDs = [];
    for (let run = 0; run < runs; run++) {
      let v = start, peak = start, mdd = 0;
      for (let k = 0; k < rets.length; k++) {
        v *= (1 + rets[Math.floor(r() * rets.length)]);
        if (v > peak) peak = v;
        const d = v / peak - 1;
        if (d < mdd) mdd = d;
      }
      finals.push(v); maxDDs.push(mdd);
    }
    finals.sort((a, b) => a - b);
    maxDDs.sort((a, b) => a - b);
    const q = (arr, p) => arr[clampIdx(Math.floor(p * (arr.length - 1)), arr.length)];
    const actual = eq[n - 1];

    // where does the real result sit inside its own resampled distribution?
    let below = 0;
    for (const f of finals) if (f < actual) below++;

    return {
      runs, actual,
      p05: q(finals, 0.05), p25: q(finals, 0.25), median: q(finals, 0.5),
      p75: q(finals, 0.75), p95: q(finals, 0.95),
      mean: finals.reduce((a, b) => a + b, 0) / finals.length,
      ddMedian: q(maxDDs, 0.5), ddP05: q(maxDDs, 0.05),
      percentileOfActual: below / finals.length,
      probProfit: finals.filter(f => f > start).length / finals.length,
    };
  }
  function clampIdx(i, len) { return i < 0 ? 0 : (i >= len ? len - 1 : i); }

  /* ========================================================== optimizer

     Grid search for the best in-sample setting, then the same setting is
     reported out-of-sample. The GAP between them is the number that matters:
     a large gap is overfitting, and it is shown rather than hidden.
  */
  function optimize(d, cfg, grid) {
    grid = grid || {};
    const models = grid.models || ['logistic', 'mlp'];
    const horizons = grid.horizons || [5, 10, 20];
    const lrs = grid.lrs || [0.01, 0.05];
    const threshes = grid.threshes || [0.52, 0.55, 0.60];
    const n = d.closes.length;
    const cut = grid.split == null ? Math.floor(n * 0.6) : grid.split;
    const train = sliceData(d, 0, cut);
    const test = sliceData(d, cut - 60 > 0 ? cut - 60 : 0, n);   // 60 bars of warmup

    const rows = [];
    for (const model of models)
      for (const horizon of horizons)
        for (const lr of lrs)
          for (const thresh of threshes) {
            const c = Object.assign({}, cfg, { model, horizon, lr, thresh });
            const a = backtest(train, c);
            const b = backtest(test, c);
            rows.push({ model, horizon, lr, thresh,
                        inReturn: a.totalRet, outReturn: b.totalRet,
                        inExcess: a.excess, outExcess: b.excess,
                        inSharpe: a.sharpe, outSharpe: b.sharpe,
                        gap: a.excess - b.excess,
                        outBeat: b.beatBench });
          }
    rows.sort((a, b) => b.inExcess - a.inExcess);
    const best = rows[0];
    const robust = rows.slice().sort((a, b) => a.gap - b.gap)[0];
    const outBeats = rows.filter(r => r.outBeat).length;
    return { rows, best, robust, outBeats, total: rows.length };
  }

  function sliceData(d, from, to) {
    const o = { closes: d.closes.slice(from, to) };
    if (d.high) o.high = d.high.slice(from, to);
    if (d.low) o.low = d.low.slice(from, to);
    if (d.volume) o.volume = d.volume.slice(from, to);
    if (d.dates) o.dates = d.dates.slice(from, to);
    o.name = d.name; o.source = d.source;
    return o;
  }

  /* ============================================================ portfolio

     Runs the strategy independently on each asset and combines the equity
     curves at equal weight. Genuinely diversified in the sense that matters:
     the assets are different, and the drawdown of the combination is not the
     drawdown of any one of them.
  */
  function portfolio(assets, cfg) {
    const per = (cfg.cash == null ? 10000 : cfg.cash) / Math.max(1, assets.length);
    const runs = [];
    for (const a of assets) {
      const c = Object.assign({}, cfg, { cash: per });
      runs.push({ name: a.name || a.symbol || '?', res: runSeries(a, c, null) });
    }
    if (!runs.length) return null;
    const n = runs[0].res.equity.length;
    const equity = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (const r of runs) equity[i] += r.res.equity[Math.min(i, r.res.equity.length - 1)];

    // benchmark: equal-weight buy and hold of the same assets
    const bench = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (const r of runs) bench[i] += r.res.bench[Math.min(i, r.res.bench.length - 1)];

    const trades = [], agg = { wins: 0, losses: 0, grossWin: 0, grossLoss: 0, fees: 0, inMkt: 0 };
    for (const r of runs) {
      for (const t of r.res.trades) trades.push(t);
      agg.wins += r.res.wins; agg.losses += r.res.losses;
      agg.grossWin += r.res.grossWin; agg.grossLoss += r.res.grossLoss;
      agg.fees += r.res.fees; agg.inMkt += r.res.barsInMarket;
    }
    const total = cfg.cash == null ? 10000 : cfg.cash;
    const res = { equity, bench, trades, fees: agg.fees, turnover: 0,
                  barsInMarket: agg.inMkt, wins: agg.wins, losses: agg.losses,
                  grossWin: agg.grossWin, grossLoss: agg.grossLoss,
                  influence: [], featureNames: runs[0].res.featureNames,
                  finalWeights: [], n, startCash: total, warmBars: cfg.warmBars || 60,
                  hasHL: runs[0].res.hasHL };
    const m = metrics(res);
    return Object.assign(m, { equity, bench, trades, n, parts: runs.map(r => ({
      name: r.name, totalRet: r.res.equity[n - 1] / per - 1,
      benchRet: r.res.bench[n - 1] / per - 1 })) });
  }

  return {
    // indicators
    sma, emaSeries, rsiSeries, macdSeries, atrSeries, stdev, percentB,
    realizedVol, logReturns, volumeZ,
    // models
    makeLogistic, makeLinear, makeMLP, makeModel, predict, learnOne,
    modelMeanLoss, featureInfluence,
    // engine
    buildFeatures, FEATURE_NAMES, makeCost, sizePosition,
    backtest, walkForward, monteCarlo, optimize, portfolio, metrics, runSeries,
    rng, sliceData,
  };
}));
