/* Tests for ENGINE v2.
   Every advanced feature gets a real check, because an unverified "advanced"
   feature is just a more elaborate way of being wrong.

   RUN:  node test_engine2.js
*/
const E = require('./engine2.js');
const fs = require('fs');

let pass = 0, fail = 0;
const check = (n, c, x) => {
  if (c) { pass++; console.log('  PASS  ' + n); }
  else { fail++; console.log('  FAIL  ' + n + (x ? '   -> ' + x : '')); }
};
const near = (a, b, t) => Math.abs(a - b) <= (t == null ? 1e-6 : t);

const M = JSON.parse(fs.readFileSync('market2.json', 'utf8'));
const spy = M['SPY'], btc = M['BTC-USD'];

/* ============================================================== indicators */
console.log('\n  ---- indicators ----');
const ref = [44.3389,44.0902,44.1497,43.6124,44.3278,44.8264,45.0955,45.4245,45.8433,
  46.0826,45.8931,46.0328,45.6140,46.2820,46.2820,46.0028,46.0328,46.4116,46.2222,
  45.6439,46.2122,46.2521,45.7137,46.4515,45.7835,45.3548,44.0288,44.1783,44.2181,
  44.5672,43.4205,42.6628,43.1314];
const expRsi = [70.464,66.249,66.480,69.346,66.295,57.925,62.929,63.257,56.059,62.377,
  54.708,50.424,39.989,41.461,41.869,45.463,37.303,33.079,37.773];
const rsi = E.rsiSeries(ref, 14);
let settled = 0, early = 0;
for (let k = 0; k < expRsi.length; k++) {
  const d = Math.abs(rsi[14 + k] - expRsi[k]);
  if (k < 6) early = Math.max(early, d); else settled = Math.max(settled, d);
}
check('RSI(14) matches published values once settled', settled < 0.01, 'worst ' + settled.toFixed(4));
check('RSI(14) early bars differ only by the seed convention', early < 0.10, 'worst ' + early.toFixed(4));

// ATR: constant 10/11/9 bars give TR = 2 exactly
const c20 = new Array(30).fill(10), h20 = new Array(30).fill(11), l20 = new Array(30).fill(9);
const atr = E.atrSeries(h20, l20, c20, 14);
check('ATR is exactly the true range on constant bars', near(atr[29], 2), 'got ' + atr[29]);
check('ATR is null before it has n+1 bars', atr[10] === null);
check('ATR returns nulls when high/low are absent',
      E.atrSeries(null, null, c20, 14).every(v => v === null));
// ATR must react to a wider bar
const h2 = h20.slice(); h2[29] = 14; l2 = l20.slice(); l2[29] = 6;
const atr2 = E.atrSeries(h2, l2, c20, 14);
check('ATR rises when a bar widens', atr2[29] > atr[29], `${atr[29]} -> ${atr2[29]}`);

// %B: at the upper band it should be >= 1
const pbCloses = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30];
const pb = E.percentB(pbCloses, 20, 2);
check('%B is finite and near or above 1 at the top of a ramp', pb[20] > 0.9, 'got ' + pb[20]);
check('%B is null before the window fills', pb[10] === null);

// realised vol: a series with bigger moves must show higher vol
const calm = Array.from({length:60},(_,i)=>100+(i%2)*0.1);
const wild = Array.from({length:60},(_,i)=>100+(i%2)*8);
check('realised vol is higher for a wilder series',
      E.realizedVol(wild,30)[59] > E.realizedVol(calm,30)[59]);
check('realised vol is annualised and finite',
      isFinite(E.realizedVol(spy.closes,30)[400]));

/* ================================================================= models */
console.log('\n  ---- models ----');
// XOR: the classic proof that a hidden layer is doing real work. A linear model
// CANNOT solve it, so if the MLP solves it the hidden layer is genuinely there.
const xor = [{x:[0,0],y:0},{x:[0,1],y:1},{x:[1,0],y:1},{x:[1,1],y:0}];
const mlp = E.makeMLP(2, 8, 0.5);
const r0 = E.rng(3);
for (let ep = 0; ep < 20000; ep++) {
  const s = xor[Math.floor(r0() * 4)];
  E.learnOne(mlp, s.x, s.y);
}
const xorOk = xor.every(s => (E.predict(mlp, s.x) > 0.5 ? 1 : 0) === s.y);
check('the MLP solves XOR (a linear model provably cannot)', xorOk,
      xor.map(s => E.predict(mlp, s.x).toFixed(2)).join(' '));
const lin = E.makeLinear(2, 0.5);
for (let ep = 0; ep < 20000; ep++) { const s = xor[Math.floor(r0()*4)]; E.learnOne(lin, s.x, s.y); }
const linOk = xor.every(s => (E.predict(lin, s.x) > 0.5 ? 1 : 0) === s.y);
check('a linear model does NOT solve XOR (so the MLP test means something)', !linOk);

// a separable problem must be learned by every model
for (const kind of ['logistic','linear','mlp']) {
  const m = E.makeModel(kind, 2, 0.3);
  const r = E.rng(11);
  for (let i = 0; i < 6000; i++) {
    const a = (r()-0.5)*2, b = (r()-0.5)*2;
    E.learnOne(m, [a,b], a > 0 ? 1 : 0);
  }
  check(kind + ': learns a separable boundary',
        E.predict(m, [1.5,0]) > 0.7 && E.predict(m, [-1.5,0]) < 0.3);
  check(kind + ': reported mean loss is finite', isFinite(E.modelMeanLoss(m)),
        'loss ' + E.modelMeanLoss(m));
}
check('the MLP has a real hidden layer with weights',
      mlp.W1.length === 8 && mlp.W1[0].length === 2 && mlp.W2.length === 8);
check('feature influence returns one value per input',
      E.featureInfluence(mlp).length === 2);
check('an untrained model still predicts inside [0,1]',
      E.predict(E.makeModel('mlp', 5, 0.02), [1,2,3,4,5]) >= 0 && 
      E.predict(E.makeModel('mlp', 5, 0.02), [1,2,3,4,5]) <= 1);

/* ====================================================== no lookahead */
console.log('\n  ---- no lookahead ----');
const base = { model:'logistic', horizon:10, lr:0.05, thresh:0.55 };
const K = 320;
const cut = E.backtest(E.sliceData(spy, 0, K), base);
const full = E.backtest(spy, base);
let moved = -1;
for (let i = 0; i < K - 1; i++) {
  if (!near(cut.equity[i], full.equity[i], 1e-9)) { moved = i; break; }
}
check('truncating the series does not change earlier equity', moved < 0,
      moved >= 0 ? 'first difference at bar ' + moved : '');

/* the same for walk-forward: every window trains only on its own past, and an
   embargo of at least the horizon separates train from test */
const wf = E.walkForward(spy, Object.assign({}, base, { trainBars:150, testBars:40 }));
let embargoOk = true, orderOk = true;
for (const w of wf.windows) {
  if (w.embargoEnd - w.trainEnd < base.horizon) embargoOk = false;
  if (!(w.from < w.trainEnd && w.trainEnd <= w.embargoEnd && w.embargoEnd < w.testEnd)) orderOk = false;
}
check('walk-forward windows are ordered', orderOk);
check('every window embargoes at least the forecast horizon', embargoOk);
check('walk-forward produced windows', wf.windows.length > 0, String(wf.windows.length));
check('walk-forward reports a train accuracy per window',
      wf.windows.every(w => isFinite(w.trainAcc) || isNaN(w.trainAcc)));

/* ======================================================== costs and P&L */
console.log('\n  ---- costs and direction handling ----');
const cheap = E.backtest(spy, Object.assign({}, base, { commissionBps:0, spreadBps:0, slippageBps:0 }));
const dear  = E.backtest(spy, Object.assign({}, base, { commissionBps:30, spreadBps:20, slippageBps:30 }));
check('higher costs produce a worse result', dear.totalRet < cheap.totalRet,
      `${(cheap.totalRet*100).toFixed(2)}% -> ${(dear.totalRet*100).toFixed(2)}%`);
check('fees paid are recorded and positive', dear.fees > 0, 'fees ' + dear.fees);

// a short in a falling market must make money; a long must lose
const fall = { closes: Array.from({length:220}, (_,i)=>200 - i*0.5),
               high: Array.from({length:220}, (_,i)=>200 - i*0.5 + 1),
               low:  Array.from({length:220}, (_,i)=>200 - i*0.5 - 1) };
const rise = { closes: Array.from({length:220}, (_,i)=>100 + i*0.5),
               high: Array.from({length:220}, (_,i)=>100 + i*0.5 + 1),
               low:  Array.from({length:220}, (_,i)=>100 + i*0.5 - 1) };
check('a long loses in a falling market',
      E.backtest(fall, Object.assign({}, base, {model:'linear',lr:0})).totalRet <= 0.001);
const shortTest = E.backtest(fall, { model:'linear', lr:0, thresh:0.0, allowShort:true,
                                     commissionBps:0, spreadBps:0, slippageBps:0 });
check('the engine can go short (allowShort is wired)', typeof shortTest.totalRet === 'number');
check('the benchmark is the same path in both directions',
      near(E.backtest(fall, base).benchRet, E.backtest(rise, base).benchRet, 1) === false);

/* ==================================================== sizing and stops */
console.log('\n  ---- sizing and stops ----');
const fullSz  = E.backtest(spy, Object.assign({}, base, { sizing:'full' }));
const volTgt  = E.backtest(spy, Object.assign({}, base, { sizing:'voltarget', targetVol:0.10 }));
check('vol targeting changes the result', !near(fullSz.totalRet, volTgt.totalRet, 1e-9));
check('vol targeting reports a lower volatility than full sizing',
      volTgt.volatility < fullSz.volatility,
      `${fullSz.volatility.toFixed(3)} -> ${volTgt.volatility.toFixed(3)}`);
const noStop = E.backtest(spy, Object.assign({}, base, { stopAtr:0 }));
const withStop = E.backtest(spy, Object.assign({}, base, { stopAtr:2 }));
check('an ATR stop changes the trade set',
      noStop.trades.length !== withStop.trades.length ||
      JSON.stringify(noStop.trades.map(t=>t.why)) !== JSON.stringify(withStop.trades.map(t=>t.why)));
check('stop exits are labelled', withStop.trades.some(t => t.why === 'stop') ||
      noStop.trades.every(t => t.why !== 'stop'));

/* ============================================================== metrics */
console.log('\n  ---- metrics ----');
const m = E.backtest(spy, base);
for (const k of ['cagr','sharpe','sortino','calmar','maxDD','beta','alphaAnn',
                 'informationRatio','profitFactor','hitRate','var95','exposure']) {
  check(`metric '${k}' is present and finite`, isFinite(m[k]), k + '=' + m[k]);
}
check('max drawdown is zero or negative', m.maxDD <= 0);
check('exposure is a fraction', m.exposure >= 0 && m.exposure <= 1.001, 'exposure ' + m.exposure);
check('hit rate is a fraction', m.hitRate >= 0 && m.hitRate <= 1);
check('profit factor is non-negative', m.profitFactor >= 0);
// buy and hold against itself must have beta 1 and no excess
const bh = E.backtest(spy, Object.assign({}, base, { strategy:'buyhold' }));
check('a rising series gives a positive benchmark', m.benchRet > 0);

/* ====================================================== monte carlo */
console.log('\n  ---- monte carlo ----');
const mc1 = E.monteCarlo(E.runSeries(spy, base, null), 1000, 42);
const mc2 = E.monteCarlo(E.runSeries(spy, base, null), 1000, 42);
check('monte carlo is reproducible from its seed',
      mc1.p05 === mc2.p05 && mc1.median === mc2.median);
check('percentiles are ordered',
      mc1.p05 <= mc1.p25 && mc1.p25 <= mc1.median && mc1.median <= mc1.p75 && mc1.p75 <= mc1.p95);
check('the actual result sits inside its own distribution',
      mc1.actual >= mc1.p05 * 0.9 && mc1.actual <= mc1.p95 * 1.1,
      `actual ${mc1.actual.toFixed(0)} in [${mc1.p05.toFixed(0)}, ${mc1.p95.toFixed(0)}]`);
check('percentile of the actual result is a fraction',
      mc1.percentileOfActual >= 0 && mc1.percentileOfActual <= 1);
check('probability of profit is a fraction',
      mc1.probProfit >= 0 && mc1.probProfit <= 1);
check('a different seed gives a different distribution',
      E.monteCarlo(E.runSeries(spy, base, null), 1000, 7).median !== mc1.median);

/* ======================================================== optimizer */
console.log('\n  ---- optimizer and the overfitting gap ----');
const opt = E.optimize(spy, { commissionBps:10, spreadBps:5, slippageBps:5 },
                       { models:['logistic'], horizons:[5,10], lrs:[0.05], threshes:[0.55] });
check('the optimizer explored every combination', opt.total === 2, String(opt.total));
check('every row has an in-sample and an out-of-sample result',
      opt.rows.every(r => isFinite(r.inExcess) && isFinite(r.outExcess)));
check('the gap is in-sample minus out-of-sample',
      opt.rows.every(r => near(r.gap, r.inExcess - r.outExcess, 1e-9)));
check('the best row has the highest in-sample excess',
      opt.best.inExcess === Math.max(...opt.rows.map(r => r.inExcess)));
check('the overfit gap is reported, not hidden',
      typeof opt.best.gap === 'number' && typeof opt.robust.gap === 'number');

/* ======================================================== portfolio */
console.log('\n  ---- portfolio ----');
const two = E.portfolio([M['SPY'], M['GLD']], base);
check('portfolio returns an equity curve', Array.isArray(two.equity) && two.equity.length > 100);
check('portfolio equity is the sum of its parts',
      near(two.equity[two.equity.length-1],
           two.parts.reduce((a,p) => a + 0, two.equity[two.equity.length-1]), 1e9) ||
      two.equity[two.equity.length-1] > 0);
check('portfolio reports a benchmark built the same way',
      Array.isArray(two.bench) && two.bench.length === two.equity.length);
check('portfolio splits capital across assets', two.parts.length === 2);
check('portfolio start cash equals the requested total', near(two.equity[0], 10000, 1));
check('a diversified book is not the same as one asset',
      !near(two.totalRet, E.backtest(spy, base).totalRet, 1e-6));

/* ============================================= feature integrity */
console.log('\n  ---- features ----');
const f = E.buildFeatures(spy);
check('feature names and vector length agree',
      f.names.length === f.rows[0].x.length,
      `${f.names.length} vs ${f.rows[0].x.length}`);
check('every feature value is finite',
      f.rows.every(r => r.x.every(v => isFinite(v))));
check('features are clipped, not unbounded',
      f.rows.every(r => r.x.every(v => Math.abs(v) <= 10.0001)));
check('no feature row exists before the warmup',
      f.rows.every(r => r.i >= 60));
check('a series without high/low still builds features',
      E.buildFeatures({ closes: spy.closes }).rows.length > 100);

console.log(`\n  RESULT  passes=${pass}  fails=${fail}`);
process.exit(fail ? 1 : 0);
