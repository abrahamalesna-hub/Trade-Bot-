#!/usr/bin/env python3
"""Virtual swing setup management for GitHub Actions/Telegram research alerts.

NO real brokerage positions or executions. Only use completed OHLC bars. SQLite
state lives in each original scanner's own existing cache database. Missing
cache history can mean missing exits: this is NEVER protective order execution.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from datetime import date, datetime, timedelta
from typing import Callable, Any

import pandas as pd

LOG = logging.getLogger('swing-exits')


def init_exit_tables(db: sqlite3.Connection) -> None:
    db.execute("""CREATE TABLE IF NOT EXISTS virtual_swing_setups (
        source TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        entry_marker TEXT NOT NULL,
        entry_reference REAL NOT NULL,
        initial_stop REAL NOT NULL,
        active_stop REAL NOT NULL,
        target1 REAL NOT NULL,
        target2 REAL NOT NULL,
        stage INTEGER NOT NULL DEFAULT 0,
        last_checked TEXT NOT NULL,
        final_checkpoint TEXT NOT NULL,
        time_reviewed INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (source, symbol, side))""")
    db.commit()


def _load(db: sqlite3.Connection, source: str, symbol: str) -> dict[str, Any] | None:
    cur = db.execute('SELECT * FROM virtual_swing_setups WHERE source=? AND symbol=?',
                     (source, symbol))
    names = [x[0] for x in cur.description]
    rows = cur.fetchall()
    if len(rows) > 1:
        LOG.warning('%s has multiple virtual directions; managing earliest', symbol)
    return dict(zip(names, rows[0])) if rows else None


def count_open(db: sqlite3.Connection, source: str) -> int:
    return db.execute('SELECT COUNT(*) FROM virtual_swing_setups WHERE source=?',
                      (source,)).fetchone()[0]


def open_virtual(db: sqlite3.Connection, source: str, sig, marker: str) -> bool:
    """Follow only delivered CONFIRMED alerts. Does not imply a real fill."""
    if _load(db, source, sig.symbol) is not None:
        LOG.info('%s already has a tracked virtual swing setup: not replacing stop/targets', sig.symbol)
        return False
    side = 'LONG' if sig.direction.startswith('BUY') else 'SHORT'
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in
           (sig.price, sig.stop, sig.target1, sig.target2)):
        raise ValueError('invalid swing reference price/levels')
    if side == 'LONG' and not sig.stop < sig.price < sig.target1 < sig.target2:
        raise ValueError('invalid bullish swing levels')
    if side == 'SHORT' and not sig.stop > sig.price > sig.target1 > sig.target2:
        raise ValueError('invalid bearish swing levels')
    db.execute('''INSERT INTO virtual_swing_setups
        (source, symbol, side, entry_marker, entry_reference, initial_stop,
         active_stop, target1, target2, stage, last_checked, final_checkpoint)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)''',
         (source, sig.symbol, side, marker, float(sig.price), float(sig.stop),
          float(sig.stop), float(sig.target1), float(sig.target2), marker,
          sig.final_checkpoint.isoformat()))
    db.commit()
    LOG.info('%s: tracking hypothetical %s swing setup from %s, stop %.2f / targets %.2f, %.2f',
             sig.symbol, side, marker, sig.stop, sig.target1, sig.target2)
    return True


def _save(db: sqlite3.Connection, p: dict[str, Any]) -> None:
    db.execute('''UPDATE virtual_swing_setups
                  SET active_stop=?, stage=?, last_checked=?, time_reviewed=?
                  WHERE source=? AND symbol=? AND side=?''',
               (p['active_stop'], p['stage'], p['last_checked'], p['time_reviewed'],
                p['source'], p['symbol'], p['side']))
    db.commit()


def _close(db: sqlite3.Connection, p: dict[str, Any]) -> None:
    db.execute('DELETE FROM virtual_swing_setups WHERE source=? AND symbol=? AND side=?',
               (p['source'], p['symbol'], p['side']))
    db.commit()


def _fmt(p: dict, what: str, marker: str, observed: float, note: str,
         ended: bool) -> str:
    is_intra = p['source'] == 'intraday'
    label = 'INTRADAY SWING' if is_intra else 'END-OF-DAY SWING'
    return (f"⚠️ VIRTUAL {label} {what} | {p['symbol']} {p['side']}\n"
            f"Signal reference: ${p['entry_reference']:.2f} (NOT an executed trade)\n"
            f"Latest observed bar: {marker}\n"
            f"Observed bar close: ${observed:.2f} (IEX data; NOT executable quote)\n"
            f"Current stop reference: ${p['active_stop']:.2f}\n"
            f"Illustrative T1/T2: ${p['target1']:.2f} / ${p['target2']:.2f}\n"
            f"Reason: {note}\n"
            + ('Hypothetical setup closed in tracker.\n' if ended else
               'Hypothetical setup stays tracked for future exit checks.\n')
            + f"Chart: https://www.tradingview.com/symbols/{p['symbol']}/\n"
            + 'Research alert ONLY, not an execution or proof of a fill. '
              'Confirm current brokerage quotes and manage actual stops at your broker.')


def _step(db: sqlite3.Connection, p: dict[str, Any], *, marker: str,
          observed: float, high: float, low: float, reversal: bool,
          time_review: bool, send: Callable[[str], object],
          daily_atr: float | None = None) -> tuple[bool, int]:
    """Evaluate a fully observed bar. Conservative if stop and target both touched.

    Returns (still_active, notifications_sent). State changes are committed only
    after a notification succeeded; when no message is needed, progress is saved.
    """
    if marker <= p['last_checked']:
        return True, 0
    long = p['side'] == 'LONG'
    stop = float(p['active_stop'])
    t1, t2 = float(p['target1']), float(p['target2'])
    stop_hit = low <= stop if long else high >= stop
    t2_hit = high >= t2 if long else low <= t2
    t1_hit = high >= t1 if long else low <= t1
    notifications = 0

    def notify(reason: str, note: str, *, ended: bool) -> None:
        nonlocal notifications
        # Alert side effect first: failed delivery must not advance the stored state.
        result = send(_fmt(p, reason, marker, observed, note, ended))
        if result is False:
            raise RuntimeError('Telegram did not confirm exit alert')
        notifications += 1

    if stop_hit:
        note = ('Observed OHLC touched the virtual stop reference. '
                'If stop and target were both inside this same bar, order is unknown; '
                'the stop has been prioritized conservatively.')
        notify('EXIT — STOP REFERENCE TOUCHED', note, ended=True)
        _close(db, p)
        return False, notifications

    if t2_hit:
        notify('EXIT — FINAL TARGET REACHED',
               'Observed OHLC touched illustrative 3R target; this is not a verified sell fill.',
               ended=True)
        _close(db, p)
        return False, notifications

    if reversal:
        notify('EXIT — TREND REVERSAL',
               'Observed completed bars breached the technical reversal rule.',
               ended=True)
        _close(db, p)
        return False, notifications

    if t1_hit and p['stage'] == 0:
        notify('PARTIAL PROFIT WATCH — TARGET 1',
               'Observed OHLC touched illustrative 2R. Consider partial profit-taking '
               'and reassessing risk; this is NOT a fill.', ended=False)
        p['stage'] = 1
        # Breakeven is for the hypothetical reference, not an actual stop order.
        p['active_stop'] = (max(stop, float(p['entry_reference'])) if long
                            else min(stop, float(p['entry_reference'])))

    # Only move trailing reference after evaluating this bar: never retroactive.
    if p['stage'] >= 1 and daily_atr is not None and daily_atr > 0:
        proposed = (observed - 1.5 * daily_atr if long
                    else observed + 1.5 * daily_atr)
        if math.isfinite(proposed):
            p['active_stop'] = (max(p['active_stop'], proposed) if long
                                else min(p['active_stop'], proposed))

    if time_review and not p['time_reviewed']:
        notify('TIME-BASED EXIT REVIEW',
               f"30-exchange-session planning horizon reached ({p['final_checkpoint']}). "
               'Review whether the setup remains valid; no automatic liquidation.',
               ended=False)
        p['time_reviewed'] = 1

    p['last_checked'] = marker
    _save(db, p)
    return True, notifications


def review_eod(db: sqlite3.Connection, symbol: str, daily: pd.DataFrame,
               indicators: pd.DataFrame, send: Callable[[str], object]) -> int:
    """Review unseen fully completed daily OHLC bars, including stop gap risk."""
    p = _load(db, 'eod', symbol)
    if p is None:
        return 0
    count = 0
    for _, row in indicators.iterrows():
        marker = str(row['session'])
        if marker <= p['last_checked']:
            continue
        close = float(row['close'])
        e20, e50, rsi = float(row['ema20']), float(row['ema50']), float(row['rsi'])
        rev = bool(math.isfinite(e20) and math.isfinite(rsi) and
                   ((p['side'] == 'LONG' and ((close < e20 and rsi < 48) or e20 < e50)) or
                    (p['side'] == 'SHORT' and ((close > e20 and rsi > 52) or e20 > e50))))
        atr = float(row['atr']) if pd.notna(row['atr']) else None
        active, n = _step(db, p, marker=marker, observed=close,
                          high=float(row['high']), low=float(row['low']),
                          reversal=rev,
                          time_review=marker >= p['final_checkpoint'],
                          daily_atr=atr, send=send)
        count += n
        if not active:
            break
    return count


def review_intraday(db: sqlite3.Connection, symbol: str,
                    bars: pd.DataFrame, daily: pd.DataFrame, now: datetime,
                    send: Callable[[str], object]) -> int:
    """Check every completed five-minute bar since prior successful review.

    Today's data may not cover missed *previous* sessions, so this is an
    advisory tracker only and not adequate stop-loss protection.
    """
    p = _load(db, 'intraday', symbol)
    if p is None or bars.empty or daily.empty:
        return 0
    # Prior completed daily ATR and EMA20; no use of unfinished daily candle.
    e20 = float(daily['ema20'].iloc[-1])
    atr = float(daily['atr'].iloc[-1])
    if not (math.isfinite(e20) and math.isfinite(atr) and atr > 0):
        return 0
    ct = 0
    prior_close = None
    for t, row in bars.iterrows():
        bar_end = t.to_pydatetime() + timedelta(minutes=5)
        marker = bar_end.isoformat()
        price = float(row['close'])
        # Require 2 successive observed closes across the daily reference.
        reversal = bool(prior_close is not None and
                        ((p['side'] == 'LONG' and
                          prior_close < e20 - 0.2*atr and price < e20 - 0.2*atr) or
                         (p['side'] == 'SHORT' and
                          prior_close > e20 + 0.2*atr and price > e20 + 0.2*atr)))
        prior_close = price
        if marker <= p['last_checked']:
            continue
        active, n = _step(db, p, marker=marker, observed=price,
                          high=float(row['high']), low=float(row['low']),
                          reversal=reversal,
                          time_review=bar_end.astimezone(now.tzinfo).date().isoformat() >= p['final_checkpoint'],
                          send=send)
        ct += n
        if not active:
            break
    return ct
