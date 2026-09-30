/* Verifies the BUILT advanced page.

   The six tabs each call a different engine entry point. This runs all of them
   against every embedded asset, so a tab cannot ship broken without a test failing.

   RUN:  node test_page2.js
*/
const fs = require('fs');
const vm = require('vm');
const path = require('path');

const PAGE = path.join(__dirname, '..', 'AITradingBot.html');
const html = fs.readFileSync(PAGE, 'utf8');
const ref = require('./engine2.js');
const market = JSON.parse(fs.readFileSync(path.join(__dirname, 'market2.json'), 'utf8'));

let pass = 0, fail = 0;
const check = (n, c, x) => {
  if (c) { pass++; console.log('  PASS  ' + n); }
  else { fail++; console.log('  FAIL  ' + n + (x ? '   -> ' + x : '')); }
};

/* -------------------------------------------------- offline single file */
console.log('\n  ---- the page stands alone ----');
check('no external script/style references',
      !/(?:src|href)\s*=\s*["\'](?:https?:)?\/\//.test(html));
check('no CDN anywhere', !/cdn\.|jsdelivr|unpkg|googleapis/.test(html));
check('no <link> tags', !/<link/i.test(html));
check('charts are hand-drawn, no chart library',
      /getContext\('2d'\)/.test(html) && !/new\s+Chart\s*\(/.test(html));

/* ------------------------------------------------------ data fidelity */
console.log('\n  ---- the embedded data ----');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
check('three script blocks', scripts.length === 3, String(scripts.length));
const blk = scripts[1];
const body = blk.slice(blk.indexOf('const DATA = ') + 'const DATA = '.length).trim().replace(/;$/, '');
let emb = null;
try { emb = JSON.parse(body); } catch (e) {}
check('the embedded data parses', emb !== null);
check('identical to market2.json', JSON.stringify(emb) === JSON.stringify(market));
check('10 assets present', emb && Object.keys(emb).length === 10);
check('every asset carries full OHLCV', emb && Object.values(emb).every(v =>
      ['open','high','low','closes','volume'].every(k => Array.isArray(v[k]) && v[k].length)));
check('all OHLCV arrays are the same length', emb && Object.values(emb).every(v =>
      v.dates.length === v.closes.length && v.high.length === v.closes.length &&
      v.low.length === v.closes.length && v.volume.length === v.closes.length));
check('every bar encloses its own close', emb && Object.values(emb).every(v =>
      v.closes.every((c,i) => v.low[i] <= c + 1e-6 && c <= v.high[i] + 1e-6)));
check('no zero or negative prices', emb && Object.values(emb).every(v =>
      v.closes.every(c => c > 0)));
check('sources are recorded', emb && Object.values(emb).every(v => !!v.source));

/* ------------------------------------------- the page runs the real engine */
console.log('\n  ---- the page runs the engine I tested ----');
const box = { console };
box.self = box; box.window = box; box.globalThis = box;
vm.createContext(box);
vm.runInContext(scripts[0] + '\n' + scripts[1] + '\n; __o = { E: self.ENGINE, D: DATA };', box);
const E = box.__o.E, D = box.__o.D;
check('ENGINE is exposed in a browser-like context', !!(E && E.backtest && E.walkForward));
check('DATA is exposed', !!(D && D['SPY']));

const cfg = { model:'logistic', horizon:10, lr:0.05, thresh:0.55,
              commissionBps:10, spreadBps:5, slippageBps:5, warmBars:60 };
let same = 0, tot = 0;
for (const k of Object.keys(D)) {
  const a = E.backtest(D[k], cfg);
  const b = ref.backtest(market[k], cfg);
  tot++;
  if (Math.abs(a.totalRet - b.totalRet) < 1e-9 && a.trades.length === b.trades.length) same++;
}
check('identical to the tested engine on every asset', same === tot, `${same}/${tot}`);

/* ------------------------------------------- every tab's entry point works */
console.log('\n  ---- every tab runs on every asset ----');
const syms = Object.keys(D);
const results = {};

for (const k of syms) {
  const r = E.backtest(D[k], cfg);
  results[k] = r;
  if (!['totalRet','benchRet','excess','sharpe','sortino','calmar','maxDD',
        'beta','alphaAnn','informationRatio','profitFactor','hitRate','var95',
        'exposure','volatility','cagr'].every(f => isFinite(r[f]))) {
    check(k + ': backtest metrics finite', false, JSON.stringify(r).slice(0, 90));
  }
}
check('Backtest tab: all metrics finite on all 10 assets',
      Object.values(results).every(r => isFinite(r.totalRet) && isFinite(r.sharpe) &&
        isFinite(r.maxDD) && isFinite(r.volatility)));

let wfOk = true, wfDetail = '';
for (const k of syms) {
  const w = E.walkForward(D[k], Object.assign({}, cfg, { trainBars:200, testBars:50 }));
  if (!w || !w.windows.length || !isFinite(w.totalRet) || !isFinite(w.excess)) {
    wfOk = false; wfDetail = k; break;
  }
  if (!w.windows.every(x => x.embargoEnd - x.trainEnd >= cfg.horizon)) {
    wfOk = false; wfDetail = k + ' embargo';
  }
}
check('Walk-forward tab: every asset produces valid, embargoed windows', wfOk, wfDetail);

let pfOk = true, pfDetail = '';
for (const combo of [['SPY','GLD'], ['SPY','TLT','GLD'], ['AAPL','NVDA','BTC-USD','TLT'],
                     ['QQQ','MSFT','TSLA','ETH-USD','GLD']]) {
  const assets = combo.map(s => Object.assign({ symbol:s }, D[s]));
  const p = E.portfolio(assets, cfg);
  if (!p || !isFinite(p.totalRet) || !isFinite(p.benchRet) || p.parts.length !== combo.length) {
    pfOk = false; pfDetail = combo.join('+'); break;
  }
}
check('Portfolio tab: valid results for 4 different baskets', pfOk, pfDetail);

const opt = E.optimize(D['SPY'], cfg, { models:['logistic','mlp'], horizons:[5,20],
                                         lrs:[0.05], threshes:[0.55,0.60] });
check('Optimise tab: explores the full grid', opt.total === 8, String(opt.total));
check('Optimise tab: every row has in and out-of-sample numbers',
      opt.rows.every(r => isFinite(r.inExcess) && isFinite(r.outExcess) && isFinite(r.gap)));
check('Optimise tab: reports the overfitting gap', typeof opt.best.gap === 'number');

const mc = E.monteCarlo(E.runSeries(D['SPY'], cfg, null), 1500, 20260930);
check('Monte Carlo tab: produces a distribution', !!mc && mc.runs === 1500);
check('Monte Carlo tab: percentiles ordered and finite',
      mc.p05 <= mc.median && mc.median <= mc.p95 && isFinite(mc.actual));
check('Monte Carlo tab: reproducible', (() => {
  const again = E.monteCarlo(E.runSeries(D['SPY'], cfg, null), 1500, 20260930);
  return again.median === mc.median;
})());

let modOk = true;
for (const k of ['SPY','BTC-USD']) {
  for (const model of ['logistic','linear','mlp']) {
    const r = E.backtest(D[k], Object.assign({}, cfg, { model, lr: model==='mlp'?0.02:cfg.lr }));
    if (!isFinite(r.totalRet) || !r.finalWeights.length) { modOk = false; }
  }
}
check('Models tab: all three models run on stocks and crypto', modOk);

/* ------------------------------------------------ no lookahead, in the page */
console.log('\n  ---- the page keeps the guarantees ----');
const K = 320;
const cut = E.backtest(E.sliceData(D['SPY'], 0, K), cfg);
const full = E.backtest(D['SPY'], cfg);
let moved = -1;
for (let i = 0; i < K - 1; i++) if (Math.abs(cut.equity[i] - full.equity[i]) > 1e-9) { moved = i; break; }
check('no lookahead in the built page', moved < 0, moved >= 0 ? 'bar ' + moved : '');
check('costs are charged in the built page',
      E.backtest(D['SPY'], cfg).fees > 0);
check('a higher cost setting makes the result worse',
      E.backtest(D['SPY'], Object.assign({}, cfg, { commissionBps:40, spreadBps:40, slippageBps:40 }))
        .totalRet < full.totalRet);

/* ------------------------------------------------ honesty in the wording */
console.log('\n  ---- the page does not overclaim ----');
const text = html.replace(/<script>[\s\S]*?<\/script>/g, '')
  .replace(/&amp;/g, '&').replace(/&mdash;/g, '-').replace(/&nbsp;/g, ' ')
  .toLowerCase();
check('says it measures the past rather than predicting', text.includes('measures the past'));
check('names the benchmark in the interface', text.includes('buy & hold') || text.includes('buy-and-hold'));
check('mentions costs on every fill', text.includes('every fill'));
check('states that out-of-sample is the honest number',
      text.includes('out-of-sample') && text.includes('honest'));
check('mentions overfitting where the optimiser is', text.includes('overfitting'));
check('does not promise profit', !/guaranteed|risk-free|will make you/.test(text));

console.log(`\n  RESULT  passes=${pass}  fails=${fail}`);
process.exit(fail ? 1 : 0);
