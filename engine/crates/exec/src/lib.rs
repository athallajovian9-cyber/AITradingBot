//! Order execution: risk checks, order lifecycle, a deterministic matching engine,
//! and a router that ties them together with measured latency.
//!
//! # Two rules this crate is built around
//!
//! 1. **No lookahead.** A market order sent after bar `t` closes fills at the OPEN of
//!    bar `t + latency`, never at a price that was already visible when the decision
//!    was made. A backtest that fills at the deciding bar's close is fiction.
//! 2. **Latency is measured, not claimed.** Every order records how many nanoseconds
//!    passed inside the engine. `Metrics` reports real percentiles from those samples.
//!
//! # Live venues
//!
//! `Exchange` is a trait. `SimExchange` implements it deterministically and is what
//! the tests exercise. A real venue plugs in behind the same trait; the exchange
//! endpoints are unreachable from the machine this was built on, so no live adapter
//! is shipped pretending to be verified.

use std::collections::HashMap;
use std::time::Instant;

pub use botcore::{
    Account, Bar, Fill, Liquidity, Order, OrderId, OrderState, OrderType, Px, RejectReason, Side,
    SymbolId, TimeInForce, Ts,
};

/* ================================================================= risk */

#[derive(Debug, Clone, Copy)]
pub struct RiskLimits {
    /// Absolute size of a single order.
    pub max_order_qty: f64,
    /// Absolute net position allowed in one symbol.
    pub max_position_qty: f64,
    /// Gross exposure across all symbols.
    pub max_gross_notional: f64,
    /// Reject a limit order priced further than this fraction from the last trade.
    pub fat_finger_pct: f64,
    /// Stop trading after losing this fraction of the starting equity in a day.
    pub max_daily_loss_pct: f64,
    /// Master switch. When true, every order is rejected.
    pub kill_switch: bool,
    /// Buying power as a multiple of cash. 1.0 means no leverage.
    pub leverage: f64,
}

impl Default for RiskLimits {
    fn default() -> Self {
        RiskLimits {
            max_order_qty: 1e9,
            max_position_qty: 1e9,
            max_gross_notional: 1e12,
            fat_finger_pct: 0.25,
            max_daily_loss_pct: 1.0,
            kill_switch: false,
            leverage: 1.0,
        }
    }
}

/// Reasons a risk check can refuse an order, with the count of each.
#[derive(Debug, Clone, Default)]
pub struct RiskStats {
    pub checked: u64,
    pub rejected: u64,
    pub by_reason: HashMap<&'static str, u64>,
}

pub struct RiskEngine {
    pub limits: RiskLimits,
    pub stats: RiskStats,
    pub day_start_equity: f64,
}

impl RiskEngine {
    pub fn new(limits: RiskLimits, day_start_equity: f64) -> Self {
        RiskEngine {
            limits,
            stats: RiskStats::default(),
            day_start_equity,
        }
    }

    pub fn kill(&mut self) {
        self.limits.kill_switch = true;
        self.stats
            .by_reason
            .entry("kill_switch")
            .and_modify(|v| *v += 1)
            .or_insert(1);
    }

    fn refuse(&mut self, r: RejectReason) -> Result<(), RejectReason> {
        self.stats.rejected += 1;
        *self.stats.by_reason.entry(r.as_str()).or_insert(0) += 1;
        Err(r)
    }

    /// Pre-trade checks. Runs before anything reaches a venue.
    pub fn check(
        &mut self,
        order: &Order,
        account: &Account,
        last_px: Option<Px>,
        equity: f64,
    ) -> Result<(), RejectReason> {
        self.stats.checked += 1;

        if self.limits.kill_switch {
            return self.refuse(RejectReason::KillSwitch);
        }
        if !order.qty.is_finite() || order.qty <= 0.0 {
            return self.refuse(RejectReason::InvalidQuantity);
        }
        if order.qty > self.limits.max_order_qty {
            return self.refuse(RejectReason::MaxOrderQty);
        }

        // Daily loss circuit breaker.
        if self.day_start_equity > 0.0 {
            let dd = 1.0 - equity / self.day_start_equity;
            if dd > self.limits.max_daily_loss_pct {
                return self.refuse(RejectReason::DailyLossLimit);
            }
        }

        // Position limit applies to the RESULTING position, not the order alone.
        let cur = account.position_qty(order.symbol);
        let delta = order.side.signed(order.qty);
        if (cur + delta).abs() > self.limits.max_position_qty {
            return self.refuse(RejectReason::MaxPosition);
        }

        // Gross exposure after the order.
        let px = match last_px {
            Some(p) if p > 0.0 => p,
            _ => 0.0,
        };
        if px > 0.0 {
            let mut gross = 0.0;
            for p in &account.positions {
                gross += (p.qty * px).abs();
            }
            gross += order.qty * px;
            if gross > self.limits.max_gross_notional {
                return self.refuse(RejectReason::MaxNotional);
            }

            // Fat finger: a limit far away from the market is almost always a typo.
            if let OrderType::Limit(lp) = order.order_type {
                let away = (lp / px - 1.0).abs();
                if away > self.limits.fat_finger_pct {
                    return self.refuse(RejectReason::FatFinger);
                }
            }

            // Needs buying power.
            if order.side == Side::Buy {
                let cost = order.qty * px;
                if cost > account.cash * self.limits.leverage + 1e-9 {
                    return self.refuse(RejectReason::InsufficientCash);
                }
            }
        }

        Ok(())
    }
}

/* ============================================================ exchange */

/// What a venue produces.
#[derive(Debug, Clone)]
pub enum Event {
    Accepted(OrderId, Ts),
    Fill(Fill),
    Rejected(OrderId, RejectReason),
    Cancelled(OrderId),
}

/// Anything that can accept and fill orders.
pub trait Exchange {
    fn submit(&mut self, order: Order) -> OrderId;
    fn cancel(&mut self, id: OrderId) -> bool;
    /// Advance the venue to a new bar. Fills generate events.
    fn on_bar(&mut self, bar: &Bar);
    fn drain_events(&mut self, out: &mut Vec<Event>);
    fn last_price(&self, s: SymbolId) -> Option<Px>;
    fn name(&self) -> &str;
}

#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    /// Commission in basis points of notional, charged on each fill.
    pub fee_bps: f64,
    /// Adverse price movement applied to every fill, in basis points.
    pub slippage_bps: f64,
    /// How many bars pass before a market order reaches the venue.
    pub latency_bars: u32,
    /// Fraction of the order that fills at each opportunity (1.0 = all at once).
    pub fill_ratio: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            fee_bps: 5.0,
            slippage_bps: 2.0,
            latency_bars: 1,
            fill_ratio: 1.0,
        }
    }
}

struct Resting {
    order: Order,
    remaining: f64,
    arrive_bar: u64,
}

/// Deterministic matching engine.
///
/// Market orders wait `latency_bars` and then fill at that bar's OPEN, so the price
/// used is one the strategy could not have seen when it decided. Limit orders rest
/// and fill only if a later bar trades through them.
pub struct SimExchange {
    cfg: SimConfig,
    bar_index: u64,
    last: HashMap<SymbolId, Px>,
    open: HashMap<SymbolId, Px>,
    working: Vec<Resting>,
    events: Vec<Event>,
    next_id: u64,
    pub fills: u64,
    pub rejected: u64,
}

impl SimExchange {
    pub fn new(cfg: SimConfig) -> Self {
        SimExchange {
            cfg,
            bar_index: 0,
            last: HashMap::new(),
            open: HashMap::new(),
            working: Vec::new(),
            events: Vec::new(),
            next_id: 1,
            fills: 0,
            rejected: 0,
        }
    }

    fn fee(&self, notional: f64) -> f64 {
        notional * self.cfg.fee_bps / 10_000.0
    }

    /// Raw fill price before slippage: the reference price for this side.
    fn base_px(&self, side: Side, open_px: Px) -> Px {
        let slip = self.cfg.slippage_bps / 10_000.0;
        match side {
            // buying pushes the price up, selling pushes it down
            Side::Buy => open_px * (1.0 + slip),
            Side::Sell => open_px * (1.0 - slip),
        }
    }

    fn emit_fill(&mut self, r: &mut Resting, qty: f64, px: Px, liq: Liquidity, ts: Ts) {
        let notional = px * qty;
        let fill = Fill {
            order_id: r.order.id,
            symbol: r.order.symbol,
            side: r.order.side,
            px,
            qty,
            fee: self.fee(notional),
            ts,
            liquidity: liq,
        };
        r.remaining -= qty;
        self.fills += 1;
        self.events.push(Event::Fill(fill));
        if r.remaining <= 1e-12 {
            self.events.push(Event::Accepted(r.order.id, ts)); // lifecycle marker
        }
    }
}

impl Exchange for SimExchange {
    fn submit(&mut self, order: Order) -> OrderId {
        let id = order.id;
        self.next_id = self.next_id.max(id.0 + 1);
        self.working.push(Resting {
            order,
            remaining: order.qty,
            arrive_bar: self.bar_index + self.cfg.latency_bars as u64,
        });
        id
    }

    fn cancel(&mut self, id: OrderId) -> bool {
        if let Some(i) = self.working.iter().position(|r| r.order.id == id) {
            let r = self.working.remove(i);
            self.events.push(Event::Cancelled(r.order.id));
            true
        } else {
            false
        }
    }

    fn on_bar(&mut self, bar: &Bar) {
        self.bar_index += 1;
        self.last.insert(bar.symbol, bar.close);
        self.open.insert(bar.symbol, bar.open);

        let mut done: Vec<usize> = Vec::new();
        for i in 0..self.working.len() {
            let remaining = self.working[i].remaining;
            if remaining <= 1e-12 {
                done.push(i);
                continue;
            }
            // not yet arrived
            if self.bar_index < self.working[i].arrive_bar {
                continue;
            }
            let side = self.working[i].order.side;
            let sym = self.working[i].order.symbol;
            if bar.symbol != sym {
                continue;
            }
            let otype = self.working[i].order.order_type;

            match otype {
                OrderType::Market => {
                    let Some(&open_px) = self.open.get(&sym) else {
                        continue;
                    };
                    let px = self.base_px(side, open_px);
                    let qty = remaining * self.cfg.fill_ratio.clamp(1e-9, 1.0);
                    let mut r = self.working.remove(i);
                    self.emit_fill(&mut r, qty, px, Liquidity::Taker, bar.ts);
                    if r.remaining > 1e-12 {
                        match r.order.tif {
                            // IOC/FOK do not rest: whatever did not fill is cancelled
                            TimeInForce::Ioc | TimeInForce::Fok => {
                                self.events.push(Event::Cancelled(r.order.id));
                            }
                            TimeInForce::Gtc => {
                                r.arrive_bar = self.bar_index + 1;
                                self.working.push(r);
                            }
                        }
                    }
                    return; // indices shifted; resume on the next bar
                }
                OrderType::Limit(limit_px) => {
                    // A buy limit fills only if the bar traded at or below it.
                    let touched = match side {
                        Side::Buy => bar.low <= limit_px,
                        Side::Sell => bar.high >= limit_px,
                    };
                    if !touched {
                        continue;
                    }
                    let qty = remaining * self.cfg.fill_ratio.clamp(1e-9, 1.0);
                    let mut r = self.working.remove(i);
                    // Maker fills get the limit price, not the open, and no slippage:
                    // the order was resting, so the venue owed it that price.
                    let notional = limit_px * qty;
                    let fill = Fill {
                        order_id: r.order.id,
                        symbol: sym,
                        side,
                        px: limit_px,
                        qty,
                        fee: notional * self.cfg.fee_bps / 10_000.0 * 0.5,
                        ts: bar.ts,
                        liquidity: Liquidity::Maker,
                    };
                    r.remaining -= qty;
                    self.fills += 1;
                    self.events.push(Event::Fill(fill));
                    if r.remaining > 1e-12 && r.order.tif == TimeInForce::Gtc {
                        r.arrive_bar = self.bar_index + 1;
                        self.working.push(r);
                    } else if r.remaining > 1e-12 {
                        self.events.push(Event::Cancelled(r.order.id));
                    }
                    return;
                }
            }
        }
        for i in done.into_iter().rev() {
            self.working.remove(i);
        }
    }

    fn drain_events(&mut self, out: &mut Vec<Event>) {
        out.append(&mut self.events);
    }

    fn last_price(&self, s: SymbolId) -> Option<Px> {
        self.last.get(&s).copied()
    }

    fn name(&self) -> &str {
        "sim"
    }
}

/* ============================================================ engine */

/// Latency samples, in nanoseconds, for the synchronous part of order handling.
#[derive(Debug, Default, Clone)]
pub struct LatencyStats {
    samples: Vec<u64>,
}

impl LatencyStats {
    pub fn record(&mut self, ns: u64) {
        self.samples.push(ns);
    }
    pub fn count(&self) -> usize {
        self.samples.len()
    }
    pub fn sum_ns(&self) -> u64 {
        self.samples.iter().sum()
    }
    pub fn mean_ns(&self) -> f64 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.sum_ns() as f64 / self.samples.len() as f64
        }
    }
    /// Nearest-rank percentile.
    pub fn percentile_ns(&self, p: f64) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }
        let mut v = self.samples.clone();
        v.sort_unstable();
        let idx = ((p / 100.0) * (v.len() as f64 - 1.0)).round() as usize;
        v[idx.min(v.len() - 1)]
    }
    pub fn max_ns(&self) -> u64 {
        self.samples.iter().copied().max().unwrap_or(0)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExecMetrics {
    pub orders_sent: u64,
    pub orders_rejected: u64,
    pub fills: u64,
    pub cancelled: u64,
    pub latency: LatencyStats,
    pub rejects_by_reason: HashMap<&'static str, u64>,
}

/// Routes orders through risk and into a venue, and keeps the account in step.
pub struct ExecutionEngine<E: Exchange> {
    pub exchange: E,
    pub risk: RiskEngine,
    pub account: Account,
    pub metrics: ExecMetrics,
    pub is_rejected: Vec<(OrderId, RejectReason)>,
    next_id: u64,
    scratch: Vec<Event>,
}

impl<E: Exchange> ExecutionEngine<E> {
    pub fn new(exchange: E, risk: RiskEngine, account: Account) -> Self {
        ExecutionEngine {
            exchange,
            risk,
            account,
            metrics: ExecMetrics::default(),
            is_rejected: Vec::new(),
            next_id: 1,
            scratch: Vec::new(),
        }
    }

    pub fn next_order_id(&mut self) -> OrderId {
        let id = OrderId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Build, risk-check and send an order. Returns the id, or why it was refused.
    ///
    /// The timing wraps only the synchronous engine work - risk check plus venue
    /// submit - which is the part this engine owns. Network time would be measured
    /// separately and is not claimed here.
    pub fn send(
        &mut self,
        symbol: SymbolId,
        side: Side,
        qty: f64,
        order_type: OrderType,
        tif: TimeInForce,
        ts: Ts,
    ) -> Result<OrderId, RejectReason> {
        let t0 = Instant::now();
        let id = self.next_order_id();
        let order = Order {
            id,
            symbol,
            side,
            qty,
            order_type,
            tif,
            ts,
        };

        let last = self.exchange.last_price(symbol);
        let equity = self
            .account
            .equity(&|s| self.exchange.last_price(s).unwrap_or(0.0));

        if let Err(r) = self.risk.check(&order, &self.account, last, equity) {
            self.metrics.orders_rejected += 1;
            *self
                .metrics
                .rejects_by_reason
                .entry(r.as_str())
                .or_insert(0) += 1;
            self.is_rejected.push((id, r));
            self.metrics.latency.record(t0.elapsed().as_nanos() as u64);
            return Err(r);
        }

        self.exchange.submit(order);
        self.metrics.orders_sent += 1;
        self.metrics.latency.record(t0.elapsed().as_nanos() as u64);
        Ok(id)
    }

    /// Advance the market by one bar and settle every fill the venue produced.
    pub fn on_bar(&mut self, bar: &Bar) {
        self.exchange.on_bar(bar);
        self.settle();
    }

    /// Drain venue events into the account.
    pub fn settle(&mut self) {
        self.scratch.clear();
        self.exchange.drain_events(&mut self.scratch);
        let mut drained = Vec::new();
        std::mem::swap(&mut drained, &mut self.scratch);
        for ev in drained.iter() {
            match ev {
                Event::Fill(f) => {
                    self.account.apply(f);
                    self.metrics.fills += 1;
                }
                Event::Rejected(_, r) => {
                    self.metrics.orders_rejected += 1;
                    *self
                        .metrics
                        .rejects_by_reason
                        .entry(r.as_str())
                        .or_insert(0) += 1;
                }
                Event::Cancelled(_) => {
                    self.metrics.cancelled += 1;
                }
                Event::Accepted(_, _) => {}
            }
        }
        self.scratch = drained;
    }

    pub fn kill(&mut self) {
        self.risk.kill();
    }

    /// Mark-to-market equity using venue prices.
    pub fn equity(&self) -> f64 {
        self.account
            .equity(&|s| self.exchange.last_price(s).unwrap_or(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: SymbolId = SymbolId(1);

    fn bar(ts: i64, o: f64, h: f64, l: f64, c: f64) -> Bar {
        Bar {
            symbol: S,
            ts,
            open: o,
            high: h,
            low: l,
            close: c,
            volume: 1000.0,
        }
    }

    fn engine(cfg: SimConfig) -> ExecutionEngine<SimExchange> {
        let acct = Account::new(100_000.0, &[S]);
        ExecutionEngine::new(
            SimExchange::new(cfg),
            RiskEngine::new(RiskLimits::default(), 100_000.0),
            acct,
        )
    }

    fn zerocost() -> SimConfig {
        SimConfig {
            fee_bps: 0.0,
            slippage_bps: 0.0,
            latency_bars: 1,
            fill_ratio: 1.0,
        }
    }

    #[test]
    fn a_market_order_fills_at_the_NEXT_bar_open_not_the_deciding_close() {
        // the no-lookahead rule, tested directly
        let mut e = engine(zerocost());
        e.on_bar(&bar(1, 100.0, 101.0, 99.0, 100.0));
        e.send(S, Side::Buy, 10.0, OrderType::Market, TimeInForce::Gtc, 1)
            .unwrap();
        assert_eq!(e.metrics.fills, 0, "must not fill on the deciding bar");
        // next bar opens at 110 - the strategy never saw that price when it decided
        e.on_bar(&bar(2, 110.0, 112.0, 109.0, 111.0));
        assert_eq!(e.metrics.fills, 1);
        assert!((e.account.position_qty(S) - 10.0).abs() < 1e-9);
        let p = e.account.position(S).unwrap();
        assert!((p.avg_px - 110.0).abs() < 1e-9, "avg {}", p.avg_px);
    }

    #[test]
    fn latency_bars_are_respected() {
        let mut cfg = zerocost();
        cfg.latency_bars = 3;
        let mut e = engine(cfg);
        e.on_bar(&bar(1, 100.0, 100.0, 100.0, 100.0));
        e.send(S, Side::Buy, 5.0, OrderType::Market, TimeInForce::Gtc, 1)
            .unwrap();
        for i in 2..=3 {
            e.on_bar(&bar(i, 100.0 + i as f64, 100.0, 100.0, 100.0));
            assert_eq!(e.metrics.fills, 0, "filled too early at bar {}", i);
        }
        e.on_bar(&bar(4, 104.0, 104.0, 104.0, 104.0));
        assert_eq!(e.metrics.fills, 1);
        assert!((e.account.position(S).unwrap().avg_px - 104.0).abs() < 1e-9);
    }

    #[test]
    fn slippage_moves_the_fill_against_the_taker() {
        let mut cfg = zerocost();
        cfg.slippage_bps = 100.0; // 1%
        let mut e = engine(cfg);
        e.on_bar(&bar(1, 100.0, 100.0, 100.0, 100.0));
        e.send(S, Side::Buy, 1.0, OrderType::Market, TimeInForce::Gtc, 1)
            .unwrap();
        e.on_bar(&bar(2, 200.0, 200.0, 200.0, 200.0));
        let px = e.account.position(S).unwrap().avg_px;
        assert!((px - 202.0).abs() < 1e-9, "buy should pay up, got {}", px);
    }

    #[test]
    fn commission_is_charged_on_the_fill() {
        let mut cfg = zerocost();
        cfg.fee_bps = 10.0; // 0.1%
        let mut e = engine(cfg);
        e.on_bar(&bar(1, 100.0, 100.0, 100.0, 100.0));
        e.send(S, Side::Buy, 10.0, OrderType::Market, TimeInForce::Gtc, 1)
            .unwrap();
        e.on_bar(&bar(2, 100.0, 100.0, 100.0, 100.0));
        // notional 1000, fee 1.0
        assert!(
            (e.account.fees_paid - 1.0).abs() < 1e-9,
            "fees {}",
            e.account.fees_paid
        );
    }

    #[test]
    fn a_buy_limit_rests_until_price_trades_through_it() {
        let mut e = engine(zerocost());
        e.on_bar(&bar(1, 100.0, 101.0, 99.0, 100.0));
        e.send(
            S,
            Side::Buy,
            5.0,
            OrderType::Limit(95.0),
            TimeInForce::Gtc,
            1,
        )
        .unwrap();
        e.on_bar(&bar(2, 100.0, 101.0, 98.0, 99.0));
        assert_eq!(
            e.metrics.fills, 0,
            "must not fill when price stays above the limit"
        );
        e.on_bar(&bar(3, 98.0, 99.0, 94.0, 95.0));
        assert_eq!(e.metrics.fills, 1);
        let p = e.account.position(S).unwrap();
        assert!(
            (p.avg_px - 95.0).abs() < 1e-9,
            "maker should get its price, got {}",
            p.avg_px
        );
    }

    #[test]
    fn a_sell_limit_waits_for_a_high_above_it() {
        let mut e = engine(zerocost());
        e.on_bar(&bar(1, 100.0, 100.0, 100.0, 100.0));
        e.send(
            S,
            Side::Sell,
            5.0,
            OrderType::Limit(110.0),
            TimeInForce::Gtc,
            1,
        )
        .unwrap();
        e.on_bar(&bar(2, 100.0, 105.0, 99.0, 104.0));
        assert_eq!(e.metrics.fills, 0);
        e.on_bar(&bar(3, 104.0, 112.0, 103.0, 111.0));
        assert_eq!(e.metrics.fills, 1);
        assert!(e.account.position_qty(S) < 0.0);
    }

    #[test]
    fn a_missing_limit_can_be_cancelled() {
        let mut e = engine(zerocost());
        e.on_bar(&bar(1, 100.0, 100.0, 100.0, 100.0));
        // 80 is inside the fat-finger band (20% away) but far below the market,
        // so it rests unfilled until a bar trades through it
        let id = e
            .send(
                S,
                Side::Buy,
                5.0,
                OrderType::Limit(80.0),
                TimeInForce::Gtc,
                1,
            )
            .unwrap();
        assert!(e.exchange.cancel(id));
        // this bar would have filled it: low 40 is through the 80 limit
        e.on_bar(&bar(2, 100.0, 100.0, 40.0, 45.0));
        assert_eq!(e.metrics.fills, 0, "cancelled order must never fill");
        assert_eq!(e.metrics.cancelled, 1);
    }

    #[test]
    fn ioc_does_not_rest() {
        let mut cfg = zerocost();
        cfg.fill_ratio = 0.5; // only half fills immediately
        let mut e = engine(cfg);
        e.on_bar(&bar(1, 100.0, 100.0, 100.0, 100.0));
        e.send(S, Side::Buy, 10.0, OrderType::Market, TimeInForce::Ioc, 1)
            .unwrap();
        e.on_bar(&bar(2, 100.0, 100.0, 100.0, 100.0));
        assert!((e.account.position_qty(S) - 5.0).abs() < 1e-9);
        assert!(
            e.metrics.cancelled >= 1,
            "the unfilled remainder is cancelled"
        );
        // and nothing more arrives on later bars
        e.on_bar(&bar(3, 100.0, 100.0, 100.0, 100.0));
        assert!((e.account.position_qty(S) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn risk_rejects_an_oversized_order() {
        let mut e = engine(zerocost());
        e.risk.limits.max_order_qty = 100.0;
        let r = e.send(S, Side::Buy, 1000.0, OrderType::Market, TimeInForce::Gtc, 1);
        assert_eq!(r, Err(RejectReason::MaxOrderQty));
        assert_eq!(e.metrics.orders_sent, 0);
        assert_eq!(e.metrics.orders_rejected, 1);
        assert_eq!(
            *e.metrics.rejects_by_reason.get("max_order_qty").unwrap(),
            1
        );
    }

    #[test]
    fn risk_rejects_a_position_that_would_breach_the_limit() {
        let mut e = engine(zerocost());
        e.on_bar(&bar(1, 100.0, 100.0, 100.0, 100.0));
        e.risk.limits.max_position_qty = 15.0;
        e.send(S, Side::Buy, 10.0, OrderType::Market, TimeInForce::Gtc, 1)
            .unwrap();
        e.on_bar(&bar(2, 100.0, 100.0, 100.0, 100.0));
        assert_eq!(e.account.position_qty(S), 10.0);
        // another 10 would make 20, over the 15 limit
        let r = e.send(S, Side::Buy, 10.0, OrderType::Market, TimeInForce::Gtc, 2);
        assert_eq!(r, Err(RejectReason::MaxPosition));
    }

    #[test]
    fn risk_rejects_a_fat_finger_limit() {
        let mut e = engine(zerocost());
        e.on_bar(&bar(1, 100.0, 100.0, 100.0, 100.0));
        // 50% away from the market, limit is 25%
        let r = e.send(
            S,
            Side::Buy,
            1.0,
            OrderType::Limit(150.0),
            TimeInForce::Gtc,
            1,
        );
        assert_eq!(r, Err(RejectReason::FatFinger));
    }

    #[test]
    fn risk_rejects_when_cash_is_short() {
        let mut e = engine(zerocost());
        e.on_bar(&bar(1, 100.0, 100.0, 100.0, 100.0));
        // 100k cash, try to buy 2000 at 100 = 200k
        let r = e.send(S, Side::Buy, 2000.0, OrderType::Market, TimeInForce::Gtc, 1);
        assert_eq!(r, Err(RejectReason::InsufficientCash));
    }

    #[test]
    fn the_kill_switch_stops_everything() {
        let mut e = engine(zerocost());
        e.on_bar(&bar(1, 100.0, 100.0, 100.0, 100.0));
        e.kill();
        let r = e.send(S, Side::Buy, 1.0, OrderType::Market, TimeInForce::Gtc, 1);
        assert_eq!(r, Err(RejectReason::KillSwitch));
        assert_eq!(e.metrics.orders_sent, 0);
    }

    #[test]
    fn the_daily_loss_limit_refuses_new_orders() {
        let mut e = engine(zerocost());
        e.risk.day_start_equity = 100_000.0;
        e.risk.limits.max_daily_loss_pct = 0.05;
        e.on_bar(&bar(1, 100.0, 100.0, 100.0, 100.0));
        // go heavily long, then the market halves: equity falls well past the 5% limit
        e.send(S, Side::Buy, 900.0, OrderType::Market, TimeInForce::Gtc, 1)
            .unwrap();
        e.on_bar(&bar(2, 100.0, 100.0, 100.0, 100.0));
        e.on_bar(&bar(3, 50.0, 50.0, 50.0, 50.0));
        let dd = 1.0 - e.equity() / 100_000.0;
        assert!(dd > 0.05, "drawdown should exceed the limit, got {}", dd);
        let r = e.send(S, Side::Buy, 1.0, OrderType::Market, TimeInForce::Gtc, 3);
        assert_eq!(r, Err(RejectReason::DailyLossLimit));
    }

    #[test]
    fn latency_is_measured_and_ordered() {
        let mut e = engine(zerocost());
        e.on_bar(&bar(1, 100.0, 100.0, 100.0, 100.0));
        for i in 0..200 {
            let _ = e.send(S, Side::Buy, 1.0, OrderType::Market, TimeInForce::Gtc, i);
        }
        let l = &e.metrics.latency;
        assert_eq!(l.count(), 200);
        assert!(l.percentile_ns(50.0) <= l.percentile_ns(99.0));
        assert!(l.percentile_ns(99.0) <= l.max_ns());
        assert!(l.mean_ns() > 0.0);
        let _ = e.metrics.latency.sum_ns();
    }

    #[test]
    fn a_round_trip_moves_cash_by_the_expected_amount() {
        let mut e = engine(zerocost());
        e.on_bar(&bar(1, 100.0, 100.0, 100.0, 100.0));
        e.send(S, Side::Buy, 10.0, OrderType::Market, TimeInForce::Gtc, 1)
            .unwrap();
        e.on_bar(&bar(2, 100.0, 100.0, 100.0, 100.0));
        e.send(S, Side::Sell, 10.0, OrderType::Market, TimeInForce::Gtc, 2)
            .unwrap();
        e.on_bar(&bar(3, 120.0, 120.0, 120.0, 120.0));
        assert!(e.account.position(S).unwrap().is_flat());
        // bought at 100, sold at 120 -> +200 on 10 units
        assert!(
            (e.equity() - 100_200.0).abs() < 1e-6,
            "equity {}",
            e.equity()
        );
        assert!((e.account.position(S).unwrap().realized - 200.0).abs() < 1e-9);
    }

    #[test]
    fn costs_reduce_a_round_trip() {
        let mk = |cfg: SimConfig| {
            let mut e = engine(cfg);
            e.on_bar(&bar(1, 100.0, 100.0, 100.0, 100.0));
            e.send(S, Side::Buy, 10.0, OrderType::Market, TimeInForce::Gtc, 1)
                .unwrap();
            e.on_bar(&bar(2, 100.0, 100.0, 100.0, 100.0));
            e.send(S, Side::Sell, 10.0, OrderType::Market, TimeInForce::Gtc, 2)
                .unwrap();
            e.on_bar(&bar(3, 100.0, 100.0, 100.0, 100.0));
            e.equity()
        };
        let free = mk(zerocost());
        let mut dear = zerocost();
        dear.fee_bps = 20.0;
        dear.slippage_bps = 20.0;
        let costed = mk(dear);
        assert!(costed < free, "{} should be less than {}", costed, free);
    }
}
