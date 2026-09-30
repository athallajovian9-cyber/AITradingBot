/* =====================================================================
   ENGINE — real indicators, a real learner, an honest backtest.

   Written once and run in both Node (to test) and the browser (to play),
   so the tested code is the shipping code.

   Three rules this file follows, because the thing it replaces broke all three:
     1. No lookahead. A decision at bar t uses bars <= t and nothing else.
     2. Costs are charged. Spread and commission on every fill.
     3. The benchmark is always computed, so skill is provable or absent.
   ===================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ENGINE = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* ---------------------------------------------------------- indicators */

  function sma(xs, n, i) {
    if (i + 1 < n) return null;
    let s = 0;
    for (let k = i - n + 1; k <= i; k++) s += xs[k];
    return s / n;
  }

  function emaSeries(xs, n) {
    const k = 2 / (n + 1), out = new Array(xs.length).fill(null);
    let prev = null;
    for (let i = 0; i < xs.length; i++) {
      if (i + 1 < n) continue;
      if (prev === null) {
        let s = 0;
        for (let j = i - n + 1; j <= i; j++) s += xs[j];
        prev = s / n;
      } else {
        prev = xs[i] * k + prev * (1 - k);
      }
      out[i] = prev;
    }
    return out;
  }

  /* Wilder's RSI. Uses Wilder smoothing (not a plain average), which is what
     every charting package means by "RSI 14". */
  function rsiSeries(xs, n) {
    const out = new Array(xs.length).fill(null);
    if (xs.length <= n) return out;
    let gain = 0, loss = 0;
    for (let i = 1; i <= n; i++) {
      const d = xs[i] - xs[i - 1];
      if (d >= 0) gain += d; else loss -= d;
    }
    gain /= n; loss /= n;
    out[n] = loss === 0 ? 100 : 100 - 100 / (1 + gain / loss);
    for (let i = n + 1; i < xs.length; i++) {
      const d = xs[i] - xs[i - 1];
      const g = d > 0 ? d : 0, l = d < 0 ? -d : 0;
      gain = (gain * (n - 1) + g) / n;
      loss = (loss * (n - 1) + l) / n;
      out[i] = loss === 0 ? 100 : 100 - 100 / (1 + gain / loss);
    }
    return out;
  }

  function macdSeries(xs, fast, slow, signal) {
    const ef = emaSeries(xs, fast), es = emaSeries(xs, slow);
    const line = xs.map((_, i) =>
      (ef[i] != null && es[i] != null) ? ef[i] - es[i] : null);
    const valid = line.map(v => v == null ? 0 : v);
    const sig = emaSeries(valid, signal);
    const hist = line.map((v, i) => (v != null && sig[i] != null) ? v - sig[i] : null);
    return { line, signal: sig, hist };
  }

  function stdev(xs, n, i) {
    if (i + 1 < n) return null;
    let m = 0;
    for (let k = i - n + 1; k <= i; k++) m += xs[k];
    m /= n;
    let v = 0;
    for (let k = i - n + 1; k <= i; k++) v += (xs[k] - m) ** 2;
    return Math.sqrt(v / n);
  }

  /* --------------------------------------------------------- the learner

     Online logistic regression. This is genuinely a learning algorithm: it has
     weights, it computes an error against the realised outcome, and it moves the
     weights down the gradient every bar. The weights are exposed so they can be
     watched changing - which is the honest version of "AI weighting".
  */
  function makeModel(nFeatures, lr) {
    return { w: new Array(nFeatures).fill(0), b: 0, lr: lr, n: 0, lossSum: 0 };
  }

  function predict(m, x) {
    let z = m.b;
    for (let i = 0; i < x.length; i++) z += m.w[i] * x[i];
    return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
  }

  function learnOne(m, x, y) {
    const p = predict(m, x);
    const err = p - y;                    // gradient of log loss
    for (let i = 0; i < x.length; i++) m.w[i] -= m.lr * err * x[i];
    m.b -= m.lr * err;
    m.n++;
    m.lossSum += -(y * Math.log(p + 1e-9) + (1 - y) * Math.log(1 - p + 1e-9));
    return err;
  }

  function meanLoss(m) { return m.n ? m.lossSum / m.n : NaN; }

  /* --------------------------------------------------------- the backtest

     strategy: 'learner' | 'momentum' | 'meanrev' | 'buyhold' | 'flat'
     Every strategy gets the same price path, the same costs, and the same
     starting cash. The decision for bar t is made from data up to t, and the
     fill happens at the CLOSE of t - never at a price the strategy could not
     have known.
  */
  function backtest(closes, opts) {
    opts = opts || {};
    const strat = opts.strategy || 'learner';
    const feeBps = opts.feeBps == null ? 10 : opts.feeBps;    // 10 bps = 0.10%
    const slipBps = opts.slipBps == null ? 5 : opts.slipBps;
    const lr = opts.lr == null ? 0.05 : opts.lr;
    const thresh = opts.thresh == null ? 0.55 : opts.thresh;
    const warm = opts.warm == null ? 60 : opts.warm;
    const cash0 = opts.cash == null ? 10000 : opts.cash;

    const n = closes.length;
    const rsi14 = rsiSeries(closes, 14);
    const macd = macdSeries(closes, 12, 26, 9);
    const sma20 = closes.map((_, i) => sma(closes, 20, i));
    const sma50 = closes.map((_, i) => sma(closes, 50, i));
    const sd20 = closes.map((_, i) => stdev(closes, 20, i));

    const h = Math.max(1, opts.horizon == null ? 10 : opts.horizon);
    const model = makeModel(5, lr);
    const cost = (feeBps + slipBps) / 10000;
    const pending = [];        // samples awaiting a knowable label: {x, i}

    let cash = cash0, units = 0;
    const equity = new Array(n).fill(cash0);
    const weights = [];
    const trades = [];
    let wins = 0, losses = 0, feesPaid = 0, entry = null;

    const inPos = () => units > 1e-12;

    for (let i = 0; i < n; i++) {
      const px = closes[i];

      // ---- learn FIRST, only from samples whose outcome has now happened ----
      while (pending.length && i - pending[0].i >= h) {
        const smp = pending.shift();
        const y = closes[smp.i + h] > closes[smp.i] ? 1 : 0;
        learnOne(model, smp.x, y);
      }

      if (i >= warm && rsi14[i] != null && macd.hist[i] != null &&
          sma20[i] != null && sma50[i] != null && sd20[i] != null) {

        // ---- features: all from bars <= i ----
        const ret1 = (closes[i] - closes[i - 1]) / closes[i - 1];
        const ret5 = (closes[i] - closes[i - 5]) / closes[i - 5];
        const rsiF = (rsi14[i] - 50) / 50;
        const macdF = macd.hist[i] / (sd20[i] || 1);
        const trendF = (sma20[i] - sma50[i]) / (sma50[i] || 1);
        const x = [ret1 * 50, ret5 * 20, rsiF, macdF, trendF * 20];

        let wantLong;
        if (strat === 'learner') {
          wantLong = predict(model, x) > thresh;
        } else if (strat === 'momentum') {
          wantLong = trendF > 0 && ret5 > 0;
        } else if (strat === 'meanrev') {
          wantLong = rsi14[i] < 35;
        } else if (strat === 'buyhold') {
          wantLong = true;
        } else {
          wantLong = false;
        }

        // ---- act, then LEARN from what actually happened ----
        if (wantLong && !inPos()) {
          const spend = cash;
          if (spend > cash0 * 0.01) {
            const fill = px * (1 + cost);
            units = spend / fill;
            feesPaid += spend * cost;
            entry = { px: px, i: i };
            cash = 0;
          }
        } else if (!wantLong && inPos()) {
          const fill = px * (1 - cost);
          const gross = units * fill;
          feesPaid += units * px * cost;
          const pnl = gross - (entry ? entry.px * units : 0);
          if (pnl > 0) wins++; else losses++;
          trades.push({ in: entry ? entry.i : i, out: i, pnl: pnl,
                        ret: entry ? (px / entry.px - 1) : 0 });
          cash = gross; units = 0; entry = null;
        }

        // Queue the sample. Its label is decided by price at i+h, so it is not
        // learned until then - the delay above is what keeps this honest.
        if (i + h < n) pending.push({ x: x, i: i });
        if (i % 20 === 0) weights.push({ i: i, w: model.w.slice(), b: model.b });
      }

      equity[i] = cash + units * px;
    }

    // Deliberately NOT force-liquidated. Selling on the last bar would charge an
    // exit fee on one arbitrary bar and invent a trade that was never taken, which
    // is how a backtest quietly stops matching a run cut short. An open position is
    // marked to market instead, so equity is continuous and comparable.
    const openPos = inPos();

    // ---- benchmark on the SAME path, with the same entry cost ----
    const bench = new Array(n).fill(cash0);
    const bhUnits = cash0 / (closes[warm] * (1 + cost));
    for (let i = 0; i < n; i++) {
      bench[i] = i < warm ? cash0 : bhUnits * closes[i];
    }

    // ---- metrics ----
    const finalEq = equity[n - 1];
    const totalRet = finalEq / cash0 - 1;
    const benchRet = bench[n - 1] / cash0 - 1;
    let peak = cash0, maxDD = 0;
    for (const e of equity) {
      if (e > peak) peak = e;
      const dd = e / peak - 1;
      if (dd < maxDD) maxDD = dd;
    }
    const rets = [];
    for (let i = warm + 1; i < n; i++) rets.push(equity[i] / equity[i - 1] - 1);
    const mean = rets.reduce((a, b) => a + b, 0) / (rets.length || 1);
    const varr = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / (rets.length || 1);
    const sd = Math.sqrt(varr);
    const ann = Math.sqrt(365);
    const sharpe = sd > 0 ? (mean / sd) * ann : 0;

    return {
      equity, bench, weights, openPos,
      final: finalEq, benchFinal: bench[n - 1],
      totalRet, benchRet,
      excess: totalRet - benchRet,
      beatBench: totalRet > benchRet,
      trades, wins, losses,
      hitRate: (wins + losses) ? wins / (wins + losses) : 0,
      feesPaid, maxDD, sharpe, exposures: n, horizon: h,
      finalLoss: meanLoss(model), finalWeights: model.w.slice(), n: n
    };
  }

  return {
    sma, emaSeries, rsiSeries, macdSeries, stdev,
    makeModel, predict, learnOne, meanLoss,
    backtest
  };
}));
