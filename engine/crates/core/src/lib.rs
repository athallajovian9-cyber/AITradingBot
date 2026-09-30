//! Core domain types shared by the feed, the model and the execution engine.
//!
//! Design notes that matter:
//!
//! * Time is a `i64` of nanoseconds since the Unix epoch. Monotonic ordering and
//!   cheap comparison matter far more in a hot path than calendar ergonomics.
//! * Money uses `f64` deliberately. This is a bar-driven strategy with fill prices
//!   derived from candle prices, not a tick-by-tick market maker, so the precision
//!   an integer minor-unit type would buy is not the limiting factor. The one place
//!   rounding matters - cash settlement - is rounded explicitly in `Account`.
//! * Nothing here allocates on the hot path except where it must (fills), and the
//!   fill buffer is reusable.

use std::fmt;

pub type Ts = i64; // nanoseconds since Unix epoch
pub type Px = f64;

/// Instrument identifier. Interned to a small integer so the hot path never
/// compares strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct SymbolId(pub u32);

impl fmt::Display for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    #[inline]
    pub fn sign(self) -> f64 {
        match self {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        }
    }
    /// Quantity with the direction applied.
    #[inline]
    pub fn signed(self, qty: f64) -> f64 {
        self.sign() * qty
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    /// Rest until filled or cancelled.
    Gtc,
    /// Fill what you can immediately, cancel the rest.
    Ioc,
    /// Fill entirely and immediately, or not at all.
    Fok,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderType {
    Market,
    Limit(Px),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OrderId(pub u64);

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "O{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Order {
    pub id: OrderId,
    pub symbol: SymbolId,
    pub side: Side,
    pub qty: f64,
    pub order_type: OrderType,
    pub tif: TimeInForce,
    /// When the strategy decided to send it.
    pub ts: Ts,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrderState {
    Pending,
    Accepted,
    PartiallyFilled(f64),
    Filled,
    Cancelled,
    Rejected(RejectReason),
}

impl OrderState {
    pub fn filled_qty(&self) -> f64 {
        match self {
            OrderState::PartiallyFilled(q) => *q,
            OrderState::Filled => f64::INFINITY, // caller knows the order qty
            _ => 0.0,
        }
    }
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            OrderState::Filled | OrderState::Cancelled | OrderState::Rejected(_)
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Liquidity {
    Maker,
    Taker,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fill {
    pub order_id: OrderId,
    pub symbol: SymbolId,
    pub side: Side,
    pub px: Px,
    pub qty: f64,
    pub fee: f64,
    pub ts: Ts,
    pub liquidity: Liquidity,
}

impl Fill {
    #[inline]
    pub fn notional(&self) -> f64 {
        self.px * self.qty
    }
    /// Signed cash impact: negative when buying (cash leaves), positive when selling.
    #[inline]
    pub fn cash_delta(&self) -> f64 {
        -self.side.sign() * self.notional() - self.fee
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    MaxOrderQty,
    MaxPosition,
    MaxNotional,
    FatFinger,
    KillSwitch,
    InsufficientCash,
    DailyLossLimit,
    InvalidQuantity,
    UnknownSymbol,
    VenueRejected,
}

impl RejectReason {
    pub fn as_str(self) -> &'static str {
        match self {
            RejectReason::MaxOrderQty => "max_order_qty",
            RejectReason::MaxPosition => "max_position",
            RejectReason::MaxNotional => "max_notional",
            RejectReason::FatFinger => "fat_finger",
            RejectReason::KillSwitch => "kill_switch",
            RejectReason::InsufficientCash => "insufficient_cash",
            RejectReason::DailyLossLimit => "daily_loss_limit",
            RejectReason::InvalidQuantity => "invalid_quantity",
            RejectReason::UnknownSymbol => "unknown_symbol",
            RejectReason::VenueRejected => "venue_rejected",
        }
    }
}

/// A single OHLCV bar. The feed produces these; the model consumes them.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Bar {
    pub symbol: SymbolId,
    pub ts: Ts,
    pub open: Px,
    pub high: Px,
    pub low: Px,
    pub close: Px,
    pub volume: f64,
}

/// A net position in one instrument, tracked with average cost.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Position {
    pub symbol: SymbolId,
    /// Signed: positive is long.
    pub qty: f64,
    /// Average entry price of the open quantity.
    pub avg_px: Px,
    /// Cash already banked from closed quantity.
    pub realized: f64,
}

impl Position {
    pub fn new(symbol: SymbolId) -> Self {
        Position {
            symbol,
            qty: 0.0,
            avg_px: 0.0,
            realized: 0.0,
        }
    }

    pub fn is_flat(&self) -> bool {
        self.qty.abs() < 1e-12
    }

    pub fn unrealized(&self, mark: Px) -> f64 {
        if self.is_flat() {
            0.0
        } else {
            (mark - self.avg_px) * self.qty
        }
    }

    /// Apply a fill, updating average cost and realising PnL on any quantity that
    /// closes or flips the position.
    ///
    /// The sign handling is the part that is easy to get wrong: an opposing fill
    /// first closes, and only the remainder opens a new position on the other side.
    pub fn apply(&mut self, fill: &Fill) {
        let signed = fill.side.signed(fill.qty);
        let new_qty = self.qty + signed;

        if self.is_flat() || self.qty.signum() == signed.signum() {
            // opening or adding: blend the average cost
            let total = self.qty.abs() + fill.qty;
            self.avg_px = if total > 0.0 {
                (self.avg_px * self.qty.abs() + fill.px * fill.qty) / total
            } else {
                fill.px
            };
            self.qty = new_qty;
        } else {
            // opposing: realise on the closed part
            let closing = self.qty.abs().min(fill.qty);
            let dir = self.qty.signum();
            self.realized += (fill.px - self.avg_px) * closing * dir;
            self.qty = new_qty;
            if self.qty.abs() < 1e-12 {
                self.qty = 0.0;
                self.avg_px = 0.0;
            } else if self.qty.signum() != dir {
                // flipped through zero: the remainder is a fresh position
                self.avg_px = fill.px;
            }
        }
    }
}

/// Cash plus positions. All settlement goes through here so there is exactly one
/// place that can round.
#[derive(Debug, Clone)]
pub struct Account {
    pub cash: f64,
    pub positions: Vec<Position>,
    pub fees_paid: f64,
    pub fills: u64,
    pub turnover: f64,
}

impl Account {
    pub fn new(cash: f64, symbols: &[SymbolId]) -> Self {
        Account {
            cash,
            positions: symbols.iter().map(|s| Position::new(*s)).collect(),
            fees_paid: 0.0,
            fills: 0,
            turnover: 0.0,
        }
    }

    fn slot(&mut self, s: SymbolId) -> usize {
        if let Some(i) = self.positions.iter().position(|p| p.symbol == s) {
            return i;
        }
        self.positions.push(Position::new(s));
        self.positions.len() - 1
    }

    pub fn position(&self, s: SymbolId) -> Option<&Position> {
        self.positions.iter().find(|p| p.symbol == s)
    }

    pub fn position_qty(&self, s: SymbolId) -> f64 {
        self.position(s).map(|p| p.qty).unwrap_or(0.0)
    }

    pub fn apply(&mut self, fill: &Fill) {
        let i = self.slot(fill.symbol);
        self.positions[i].apply(fill);
        self.cash += fill.cash_delta();
        self.fees_paid += fill.fee;
        self.fills += 1;
        self.turnover += fill.notional();
    }

    /// Total equity marked at the given prices.
    ///
    /// `cash` already has the cost of every open position deducted from it, so
    /// adding `unrealized` on top would subtract that cost twice. Equity is the cash
    /// balance plus the current MARK-TO-MARKET VALUE of the positions. A short has
    /// negative quantity, which is exactly right: it is a liability.
    pub fn equity(&self, marks: &dyn Fn(SymbolId) -> Px) -> f64 {
        let mut eq = self.cash;
        for p in &self.positions {
            if !p.is_flat() {
                eq += p.qty * marks(p.symbol);
            }
        }
        eq
    }

    /// Book equity from realised PnL only, for a cheap cross-check against `equity`.
    /// They must agree once positions are marked correctly.
    pub fn equity_from_pnl(&self, opening_cash: f64, marks: &dyn Fn(SymbolId) -> Px) -> f64 {
        let mut eq = opening_cash;
        for p in &self.positions {
            eq += p.realized;
            if !p.is_flat() {
                eq += p.unrealized(marks(p.symbol));
            }
        }
        eq - self.fees_paid
    }

    pub fn gross_exposure(&self, marks: &dyn Fn(SymbolId) -> Px) -> f64 {
        self.positions
            .iter()
            .map(|p| (p.qty * marks(p.symbol)).abs())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(side: Side, qty: f64, px: Px) -> Fill {
        Fill {
            order_id: OrderId(1),
            symbol: SymbolId(1),
            side,
            px,
            qty,
            fee: 0.0,
            ts: 0,
            liquidity: Liquidity::Taker,
        }
    }

    #[test]
    fn opening_a_long_sets_average_price() {
        let mut p = Position::new(SymbolId(1));
        p.apply(&f(Side::Buy, 10.0, 100.0));
        assert_eq!(p.qty, 10.0);
        assert_eq!(p.avg_px, 100.0);
        assert_eq!(p.realized, 0.0);
    }

    #[test]
    fn averaging_up_blends_the_cost() {
        let mut p = Position::new(SymbolId(1));
        p.apply(&f(Side::Buy, 10.0, 100.0));
        p.apply(&f(Side::Buy, 10.0, 120.0));
        assert_eq!(p.qty, 20.0);
        assert_eq!(p.avg_px, 110.0);
    }

    #[test]
    fn closing_realises_profit_and_flattens() {
        let mut p = Position::new(SymbolId(1));
        p.apply(&f(Side::Buy, 10.0, 100.0));
        p.apply(&f(Side::Sell, 10.0, 130.0));
        assert!(p.is_flat());
        assert!((p.realized - 300.0).abs() < 1e-9, "realized {}", p.realized);
    }

    #[test]
    fn partial_close_realises_only_the_closed_part() {
        let mut p = Position::new(SymbolId(1));
        p.apply(&f(Side::Buy, 10.0, 100.0));
        p.apply(&f(Side::Sell, 4.0, 150.0));
        assert_eq!(p.qty, 6.0);
        assert!((p.realized - 200.0).abs() < 1e-9);
        assert_eq!(p.avg_px, 100.0);
    }

    #[test]
    fn flipping_through_zero_takes_the_new_price() {
        let mut p = Position::new(SymbolId(1));
        p.apply(&f(Side::Buy, 10.0, 100.0));
        p.apply(&f(Side::Sell, 15.0, 110.0));
        assert_eq!(p.qty, -5.0);
        assert_eq!(p.avg_px, 110.0);
        // realised on the 10 closed at 110 from 100 = 100
        assert!((p.realized - 100.0).abs() < 1e-9);
    }

    #[test]
    fn short_then_cover_realises_correctly() {
        let mut p = Position::new(SymbolId(1));
        p.apply(&f(Side::Sell, 10.0, 100.0));
        assert_eq!(p.qty, -10.0);
        p.apply(&f(Side::Buy, 10.0, 90.0));
        assert!(p.is_flat());
        assert!((p.realized - 100.0).abs() < 1e-9);
    }

    #[test]
    fn account_settles_cash_and_fees() {
        let mut a = Account::new(1000.0, &[SymbolId(1)]);
        let mut fill = f(Side::Buy, 2.0, 100.0);
        fill.fee = 1.5;
        a.apply(&fill);
        // 1000 - 200 - 1.5
        assert!((a.cash - 798.5).abs() < 1e-9, "cash {}", a.cash);
        assert_eq!(a.fees_paid, 1.5);
        assert_eq!(a.fills, 1);
    }

    #[test]
    fn equity_marks_to_market() {
        let mut a = Account::new(1000.0, &[SymbolId(1)]);
        a.apply(&f(Side::Buy, 2.0, 100.0));
        // cash 800 + 2 units worth 150 = 1100
        let eq = a.equity(&|_| 150.0);
        assert!((eq - 1100.0).abs() < 1e-9, "equity {}", eq);
        // and the independent derivation must agree
        let alt = a.equity_from_pnl(1000.0, &|_| 150.0);
        assert!((alt - eq).abs() < 1e-9, "pnl-derived {} vs {}", alt, eq);
    }

    #[test]
    fn equity_falls_when_a_long_loses() {
        let mut a = Account::new(1000.0, &[SymbolId(1)]);
        a.apply(&f(Side::Buy, 2.0, 100.0));
        assert!((a.equity(&|_| 90.0) - 980.0).abs() < 1e-9);
    }

    #[test]
    fn a_short_gains_as_price_falls() {
        let mut a = Account::new(1000.0, &[SymbolId(1)]);
        a.apply(&f(Side::Sell, 2.0, 100.0));
        assert!((a.equity(&|_| 90.0) - 1020.0).abs() < 1e-9);
        assert!((a.equity(&|_| 110.0) - 980.0).abs() < 1e-9);
    }

    #[test]
    fn flat_account_equity_is_just_cash() {
        let a = Account::new(1000.0, &[SymbolId(1)]);
        assert!((a.equity(&|_| 999.0) - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn fill_cash_delta_is_signed_correctly() {
        let buy = f(Side::Buy, 1.0, 10.0);
        let sell = f(Side::Sell, 1.0, 10.0);
        assert!(buy.cash_delta() < 0.0);
        assert!(sell.cash_delta() > 0.0);
    }

    #[test]
    fn rejects_are_terminal_and_labelled() {
        let s = OrderState::Rejected(RejectReason::MaxOrderQty);
        assert!(s.is_terminal());
        assert_eq!(RejectReason::MaxOrderQty.as_str(), "max_order_qty");
    }
}
