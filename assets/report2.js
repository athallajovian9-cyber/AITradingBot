const E = require('./engine2.js');
const M = JSON.parse(require('fs').readFileSync('market2.json', 'utf8'));
const pct = x => ((x*100 >= 0 ? '+' : '') + (x*100).toFixed(1) + '%');
const cfg = { model:'logistic', horizon:10, lr:0.05, thresh:0.55,
              commissionBps:10, spreadBps:5, slippageBps:5, warmBars:60,
              trainBars:220, testBars:50 };
const syms = Object.keys(M);

console.log('  =============================================================================');
console.log('  WALK-FORWARD: every bar traded with a model fitted only on the past');
console.log('  =============================================================================');
console.log('  asset       in-sample   OUT-OF-SAMPLE   hold      excess    windows   trades');
let beats = 0;
for (const s of syms){
  const w = E.walkForward(M[s], cfg);
  if (w.beatBench) beats++;
  console.log('  ' + s.padEnd(11) +
    pct(w.benchRet).padStart(9) + '   ' + pct(w.totalRet).padStart(12) +
    '   ' + pct(w.benchRet).padStart(8) + '  ' + pct(w.excess).padStart(8) +
    '   ' + String(w.windows.length).padStart(6) + '   ' + String(w.trades.length).padStart(6));
}
console.log('  ' + '-'.repeat(75));
console.log('  walk-forward beat buy-and-hold in ' + beats + '/' + syms.length + ' assets');

console.log('');
console.log('  =============================================================================');
console.log('  PORTFOLIO: several assets at once, capital split equally');
console.log('  =============================================================================');
for (const basket of [['SPY'], ['SPY','GLD'], ['SPY','TLT'], ['SPY','GLD','TLT'],
                      ['SPY','QQQ','GLD','TLT'], ['SPY','QQQ','GLD','TLT','BTC-USD'],
                      syms]){
  const assets = basket.map(s => Object.assign({symbol:s}, M[s]));
  const p = E.portfolio(assets, cfg);
  console.log('  ' + (basket.join('+')).padEnd(34) +
    'bot ' + pct(p.totalRet).padStart(8) +
    '  hold ' + pct(p.benchRet).padStart(8) +
    '  excess ' + pct(p.excess).padStart(8) +
    '  maxDD ' + pct(p.maxDD).padStart(7) +
    '  Sharpe ' + p.sharpe.toFixed(2).padStart(5));
}

console.log('');
console.log('  =============================================================================');
console.log('  THE OVERFITTING GAP: best in-sample setting vs the same setting out-of-sample');
console.log('  =============================================================================');
console.log('  asset      best setting              in-sample   out-sample     gap');
for (const s of syms){
  const r = E.optimize(M[s], cfg, { models:['logistic','mlp'], horizons:[5,10,20,40],
                                     lrs:[0.01,0.05,0.2], threshes:[0.52,0.55,0.60] });
  const b = r.best;
  console.log('  ' + s.padEnd(11) +
    (b.model + ' h' + b.horizon + ' lr' + b.lr + ' th' + b.thresh).padEnd(24) +
    pct(b.inExcess).padStart(10) + '   ' + pct(b.outExcess).padStart(10) +
    '  ' + pct(b.gap).padStart(8) +
    '   (' + r.outBeats + '/' + r.total + ' beat out-of-sample)');
}

console.log('');
console.log('  =============================================================================');
console.log('  MONTE CARLO: how much of a result is just the order things happened in');
console.log('  =============================================================================');
for (const s of ['SPY','NVDA','BTC-USD']){
  const res = E.runSeries(M[s], Object.assign({}, cfg, {model:'logistic'}));
  const mc = E.monteCarlo(res, 4000, 20260930);
  console.log('  ' + s.padEnd(10) + 'actual ' + ('$'+Math.round(mc.actual)).padStart(9) +
    '  p5 ' + ('$'+Math.round(mc.p05)).padStart(9) +
    '  median ' + ('$'+Math.round(mc.median)).padStart(9) +
    '  p95 ' + ('$'+Math.round(mc.p95)).padStart(9) +
    '  P(profit) ' + (mc.probProfit*100).toFixed(0) + '%');
}
