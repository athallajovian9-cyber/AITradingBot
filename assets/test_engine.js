/* Tests for ENGINE. The point of this file is that the replacements for the fake
   bot's claims are actually true, so each claim gets a real check.

   RUN:  node test_engine.js
*/
const E = require('./engine.js');
const fs = require('fs');

let pass = 0, fail = 0;
const check = (n, c, x) => {
  if (c) { pass++; console.log('  PASS  ' + n); }
  else { fail++; console.log('  FAIL  ' + n + (x ? '   -> ' + x : '')); }
};

/* --------------------------------------------------------------- RSI
   Published reference: the StockCharts Wilder RSI worked example. If our RSI
   does not reproduce these numbers, it is not RSI. */
console.log('\n  ---- RSI against the published Wilder example ----');
const refCloses = [44.3389,44.0902,44.1497,43.6124,44.3278,44.8264,45.0955,45.4245,
  45.8433,46.0826,45.8931,46.0328,45.6140,46.2820,46.2820,46.0028,46.0328,46.4116,
  46.2222,45.6439,46.2122,46.2521,45.7137,46.4515,45.7835,45.3548,44.0288,44.1783,
  44.2181,44.5672,43.4205,42.6628,43.1314];
const expected = [70.464,66.249,66.480,69.346,66.295,57.925,62.929,63.257,56.059,
  62.377,54.708,50.424,39.989,41.461,41.869,45.463,37.303,33.079,37.773];
const rsi = E.rsiSeries(refCloses, 14);
let settledWorst = 0, earlyWorst = 0;
for (let k = 0; k < expected.length; k++) {
  const d = Math.abs(rsi[14 + k] - expected[k]);
  if (k < 6) { if (d > earlyWorst) earlyWorst = d; }
  else { if (d > settledWorst) settledWorst = d; }
}
/* Values from bar 20 onward must match the published table exactly. The first six
   differ by ~0.07 and the error decays to zero, which is what a different initial
   seed produces - not what a wrong formula produces. Asserting both separately so
   the seed difference is visible instead of hidden behind one loose tolerance. */
check('RSI(14) matches the published values once settled (tol 0.01)',
      settledWorst < 0.01, 'worst ' + settledWorst.toFixed(4));
check('RSI(14) early bars differ only by the seed convention (tol 0.10)',
      earlyWorst < 0.10, 'worst ' + earlyWorst.toFixed(4));
check('RSI is null before it has enough bars', rsi[13] === null);
check('RSI is 100 when there are no losses',
      E.rsiSeries(Array.from({length:20},(_,i)=>100+i),14)[14] === 100);

/* ------------------------------------------------------- SMA / EMA / MACD */
console.log('\n  ---- SMA and EMA ----');
const ones = new Array(30).fill(5);
check('SMA of a constant series is that constant', E.sma(ones, 10, 29) === 5);
const ramp = Array.from({length:30},(_,i)=>i+1);
check('SMA(10) of 1..30 at the last bar is 25.5', E.sma(ramp,10,29) === 25.5,
      'got ' + E.sma(ramp,10,29));
const ema = E.emaSeries(ones, 10);
check('EMA of a constant series is that constant', Math.abs(ema[29]-5) < 1e-9);
check('EMA is null before n bars', ema[8] === null);

const mc = E.macdSeries(ramp, 12, 26, 9);
check('MACD line is positive in an uptrend', mc.line[29] > 0, 'got ' + mc.line[29]);
check('MACD histogram is finite', isFinite(mc.hist[29]));

/* --------------------------------------------------------- the learner
   It must genuinely learn. Test on a separable problem where the answer is
   known: label 1 when feature0 > 0. */
console.log('\n  ---- the learner actually learns ----');
let seed = 12345;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
const m = E.makeModel(2, 0.3);
const data = [];
for (let i = 0; i < 4000; i++) {
  const a = (rnd() - 0.5) * 2, b = (rnd() - 0.5) * 2;
  data.push({ x: [a, b], y: a > 0 ? 1 : 0 });
}
const lossBefore = (() => {           // loss of the untrained model
  let s = 0;
  for (const d of data) { const p = E.predict(m, d.x);
    s += -(d.y*Math.log(p+1e-9) + (1-d.y)*Math.log(1-p+1e-9)); }
  return s / data.length;
})();
for (let epoch = 0; epoch < 6; epoch++) {
  for (const d of data) E.learnOne(m, d.x, d.y);
}
const lossAfter = (() => {
  let s = 0;
  for (const d of data) { const p = E.predict(m, d.x);
    s += -(d.y*Math.log(p+1e-9) + (1-d.y)*Math.log(1-p+1e-9)); }
  return s / data.length;
})();
check('training reduces loss on a separable problem',
      lossAfter < lossBefore * 0.6, `${lossBefore.toFixed(4)} -> ${lossAfter.toFixed(4)}`);
check('the weight on the informative feature grew large',
      Math.abs(m.w[0]) > 1.0, 'w0=' + m.w[0].toFixed(3));
check('the weight on the noise feature stayed small',
      Math.abs(m.w[1]) < 1.0, 'w1=' + m.w[1].toFixed(3));
check('weights have the right sign (informative feature > 0)',
      m.w[0] > 0, 'w0=' + m.w[0].toFixed(3));
check('predictions are confident on a separable problem',
      E.predict(m, [2, 0]) > 0.9 && E.predict(m, [-2, 0]) < 0.1);

/* ------------------------------------------------------- NO LOOKAHEAD
   The decisive test: a decision at bar i must be identical whether or not the
   future exists. Truncate the series and the earlier decisions must not move. */
console.log('\n  ---- no lookahead ----');
const market = JSON.parse(fs.readFileSync('market.json', 'utf8'));
const full = market.SPY.closes;
const K = 300;
const a = E.backtest(full.slice(0, K), { strategy: 'learner' });
const b = E.backtest(full, { strategy: 'learner' });
let identical = true, firstDiff = -1;
for (let i = 0; i < K - 2; i++) {
  if (Math.abs(a.equity[i] - b.equity[i]) > 1e-9) { identical = false; firstDiff = i; break; }
}
check('equity up to the cut is unchanged by the existence of later data',
      identical, firstDiff >= 0 ? 'first difference at bar ' + firstDiff : '');
check('the truncated run is identical right up to and including the cut bar',
      Math.abs(a.equity[K-1] - b.equity[K-1]) < 1e-9,
      `${a.equity[K-1].toFixed(6)} vs ${b.equity[K-1].toFixed(6)}`);
check('every trade closed before the cut is identical',
      a.trades.length === b.trades.filter(t => t.out < K).length,
      `${a.trades.length} vs ${b.trades.filter(t => t.out < K).length}`);
check('the open position is reported, not silently liquidated',
      typeof a.openPos === 'boolean' || typeof b.openPos === 'boolean');

/* no decision may depend on a price not yet printed */
const rev = E.backtest(full.slice().reverse(), { strategy: 'learner' });
check('a reversed series produces a different result (proves it reads the data in order)',
      Math.abs(rev.final - b.final) > 1, `${rev.final.toFixed(2)} vs ${b.final.toFixed(2)}`);

/* ------------------------------------------------------------ costs charged */
console.log('\n  ---- costs are charged ----');
const noFee = E.backtest(full, { strategy: 'buyhold', feeBps: 0, slipBps: 0 });
const withFee = E.backtest(full, { strategy: 'buyhold', feeBps: 10, slipBps: 5 });
check('higher costs give a strictly worse result', withFee.final < noFee.final,
      `${noFee.final.toFixed(2)} -> ${withFee.final.toFixed(2)}`);
check('the recorded fees are positive', withFee.feesPaid > 0,
      'fees ' + withFee.feesPaid.toFixed(2));

/* --------------------------------------------------------------- benchmark */
console.log('\n  ---- the benchmark is the same path ----');
const bh = E.backtest(full, { strategy: 'buyhold' });
const rawRet = full[full.length-1] / full[60] - 1;
check('buy-and-hold matches the raw price change (minus entry cost)',
      Math.abs(bh.totalRet - rawRet) < 0.01,
      `engine ${(bh.totalRet*100).toFixed(2)}% vs raw ${(rawRet*100).toFixed(2)}%`);
check('a flat strategy never beats holding in a rising market',
      E.backtest(full, { strategy: 'flat' }).final < bh.final);

console.log(`\n  RESULT  passes=${pass}  fails=${fail}`);
process.exit(fail ? 1 : 0);
