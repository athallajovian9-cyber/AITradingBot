/* Verifies the BUILT page, not the sources.

   A build step that inlines things can silently drop, reorder or corrupt them, and
   the result still looks like a working page. So this loads the shipped file the way
   a browser would and checks it against the sources it was built from.

   RUN:  node test_page.js
*/
const fs = require('fs');
const vm = require('vm');
const path = require('path');

const PAGE = path.join(__dirname, '..', 'AITradingBot.html');
const html = fs.readFileSync(PAGE, 'utf8');
const ref = require('./engine.js');
const market = JSON.parse(fs.readFileSync(path.join(__dirname, 'market.json'), 'utf8'));

let pass = 0, fail = 0;
const check = (n, c, x) => {
  if (c) { pass++; console.log('  PASS  ' + n); }
  else { fail++; console.log('  FAIL  ' + n + (x ? '   -> ' + x : '')); }
};

/* ------------------------------------------------- offline, single file */
console.log('\n  ---- the page stands alone ----');
const ext = html.match(/(?:src|href)\s*=\s*["'](https?:)?\/\/[^"']+/g) || [];
check('no external script/style references', ext.length === 0, ext.join(', '));
check('no CDN dependency at all', !/cdn\.|jsdelivr|unpkg|googleapis/.test(html));
check('no <link> tags', !/<link/i.test(html));
check('the chart is drawn on canvas, not by a library',
      /getContext\('2d'\)/.test(html) && !/new\s+Chart\s*\(/.test(html));

/* ------------------------------------------------------- data fidelity */
console.log('\n  ---- the embedded data is the data that was fetched ----');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
check('three script blocks (engine, data, app)', scripts.length === 3, String(scripts.length));

const blk = scripts[1];
const body = blk.slice(blk.indexOf('const DATA = ') + 'const DATA = '.length).trim().replace(/;$/, '');
let embedded = null;
try { embedded = JSON.parse(body); } catch (e) { /* reported below */ }
check('the embedded data parses as JSON', embedded !== null);
check('embedded data is identical to market.json',
      JSON.stringify(embedded) === JSON.stringify(market));
check('four real series are present',
      embedded && ['BTC','AAPL','SPY','NVDA'].every(k => k in embedded));
check('every series has dates and closes of equal length',
      embedded && Object.values(embedded).every(v => v.dates.length === v.closes.length));
check('every series is long enough to be meaningful',
      embedded && Object.values(embedded).every(v => v.closes.length >= 300));
check('no price is zero or negative',
      embedded && Object.values(embedded).every(v => v.closes.every(c => c > 0)));
check('sources are recorded (not passed off as live)',
      embedded && Object.values(embedded).every(v => !!v.source));

/* ------------------------------------------- the page runs the real engine */
console.log('\n  ---- the page runs the engine I tested ----');
const box = { console };
box.self = box; box.window = box; box.globalThis = box;
vm.createContext(box);
vm.runInContext(scripts[0] + '\n' + scripts[1] + '\n; __o = { E: self.ENGINE, D: DATA };', box);
const E = box.__o.E, D = box.__o.D;
check('ENGINE is exposed in a browser-like context', !!(E && E.backtest));
check('DATA is exposed', !!(D && D.SPY));

let same = 0, tot = 0;
for (const k of ['SPY','NVDA','AAPL','BTC']) {
  const opts = { strategy:'learner', lr:0.02, thresh:0.55, horizon:20 };
  const a = E.backtest(D[k].closes, opts);
  const b = ref.backtest(market[k].closes, opts);
  tot++;
  if (Math.abs(a.final - b.final) < 1e-6 && a.trades.length === b.trades.length) same++;
}
check('identical results to the tested engine on every series', same === tot,
      `${same}/${tot}`);

/* ------------------------------------------------ the honesty guarantees */
console.log('\n  ---- the guarantees the old page broke ----');
const opts = { strategy:'learner', lr:0.02, thresh:0.55, horizon:20 };
for (const k of ['SPY','AAPL']) {
  const r = E.backtest(D[k].closes, opts);
  check(`${k}: a benchmark is always returned`, Array.isArray(r.bench) &&
        r.bench.length === D[k].closes.length);
  check(`${k}: costs are charged`, r.feesPaid > 0, 'fees ' + r.feesPaid);
  check(`${k}: the result says whether it beat the benchmark`,
        typeof r.beatBench === 'boolean');
  check(`${k}: max drawdown is reported`, r.maxDD <= 0);
  check(`${k}: hit rate is reported`, r.hitRate >= 0 && r.hitRate <= 1);
}
// truncation = no lookahead, the single most important property
const c = D.SPY.closes, K = 300;
const cut = E.backtest(c.slice(0, K), opts), full = E.backtest(c, opts);
let ok = true;
for (let i = 0; i < K; i++) if (Math.abs(cut.equity[i] - full.equity[i]) > 1e-9) { ok = false; break; }
check('no lookahead: past results do not change when the future exists', ok);

/* --------------------------------- the app must not overclaim in its text */
console.log('\n  ---- the page does not overclaim ----');
const text = html.replace(/<script>[\s\S]*?<\/script>/g, '').toLowerCase();
check('the word "neural" is not used', !text.includes('neural'));
check('it does not promise profit', !/guarantee|risk-free|will make you/.test(text));
check('it states that it measures the past, not predicts',
      text.includes('measures the past'));
check('the benchmark is named in the interface', text.includes('buy'));
check('it mentions costs on every fill', text.includes('every fill'));

console.log(`\n  RESULT  passes=${pass}  fails=${fail}`);
process.exit(fail ? 1 : 0);
