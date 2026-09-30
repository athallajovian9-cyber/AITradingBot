const E = require('./engine.js');
const m = JSON.parse(require('fs').readFileSync('market.json', 'utf8'));
const pct = x => ((x * 100 >= 0 ? '+' : '') + (x * 100).toFixed(1) + '%');

console.log('  ==========================================================================');
console.log('  DOES A LONGER FORECAST HORIZON HELP?  (real data, costs charged)');
console.log('  ==========================================================================');
console.log('');

for (const h of [1, 5, 10, 20, 40]) {
  let beats = 0, sumEx = 0;
  const cells = [];
  for (const k of ['SPY', 'NVDA', 'AAPL', 'BTC']) {
    const r = E.backtest(m[k].closes, { strategy: 'learner', lr: 0.02, thresh: 0.55, horizon: h });
    if (r.beatBench) beats++;
    sumEx += r.excess;
    cells.push((k + ' ' + pct(r.excess)).padEnd(15));
  }
  console.log('  horizon ' + String(h).padStart(2) + '   ' + cells.join('') +
    '  beats ' + beats + '/4   mean ' + pct(sumEx / 4));
}

console.log('');
console.log('  ==========================================================================');
console.log('  BEST HORIZON IN DETAIL');
console.log('  ==========================================================================');
console.log('  series    bot      hold     excess    hit   trades   maxDD   fees');
let beats = 0;
for (const k of ['SPY', 'NVDA', 'AAPL', 'BTC']) {
  const r = E.backtest(m[k].closes, { strategy: 'learner', lr: 0.02, thresh: 0.55, horizon: 20 });
  if (r.beatBench) beats++;
  console.log('  ' + k.padEnd(9) +
    pct(r.totalRet).padStart(8) + pct(r.benchRet).padStart(9) +
    pct(r.excess).padStart(10) +
    (r.hitRate * 100).toFixed(0).padStart(6) + '%' +
    String(r.trades.length).padStart(8) +
    pct(r.maxDD).padStart(9) +
    ('$' + r.feesPaid.toFixed(0)).padStart(8));
}
console.log('');
console.log('  learner (h=20) beats buy-and-hold in ' + beats + '/4');

console.log('');
console.log('  ==========================================================================');
console.log('  AND THE FAIR TEST: out-of-sample. Tune on the first half,');
console.log('  then run untouched on the second half.');
console.log('  ==========================================================================');
console.log('  series    in-sample best lr   out-of-sample excess');
for (const k of ['SPY', 'NVDA', 'AAPL', 'BTC']) {
  const c = m[k].closes;
  const cut = Math.floor(c.length / 2);
  const train = c.slice(0, cut), test = c.slice(cut);
  let bestLr = null, bestEx = -Infinity;
  for (const lr of [0.005, 0.01, 0.02, 0.05, 0.1]) {
    const r = E.backtest(train, { strategy: 'learner', lr, thresh: 0.55, horizon: 20 });
    if (r.excess > bestEx) { bestEx = r.excess; bestLr = lr; }
  }
  const oos = E.backtest(test, { strategy: 'learner', lr: bestLr, thresh: 0.55, horizon: 20 });
  console.log('  ' + k.padEnd(9) + String(bestLr).padStart(14) +
    '        ' + pct(oos.excess).padStart(10) +
    (oos.beatBench ? '   BEAT' : '   lost to hold'));
}
