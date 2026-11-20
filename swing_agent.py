#!/usr/bin/env python3
"""End-of-day swing-setup alert scanner; NO trade execution.

Requires the day trading agent.py and its requirements.txt in the same folder.
Runs once after each NYSE session closes. All dates are planning checkpoints,
not predictions of when any price will be reached.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from agent import ET, UTC, NYSE, Settings, api_get, make_http, send_telegram
from swing_exit_tracking import init_exit_tables, open_virtual, review_eod, count_open

LOG = logging.getLogger("swing-scanner")


@dataclass(frozen=True)
class SwingSignal:
    symbol: str
    direction: str
    pattern: str
    session: date
    price: float
    rsi: float
    rvol: float
    atr: float
    ema20: float
    ema50: float
    ema200: float
    stop: float
    target1: float
    target2: float
    first_checkpoint: date
    final_checkpoint: date
    feed: str


def most_recent_completed_session(now: datetime) -> date | None:
    """Wait at least 45 minutes after NYSE close, including early closes."""
    now = now.astimezone(UTC)
    sessions = NYSE.sessions_in_range(
        (now - timedelta(days=21)).date(), now.astimezone(ET).date()
    )
    complete = [s.date() for s in sessions
                if NYSE.session_close(s).to_pydatetime() + timedelta(minutes=45) <= now]
    return complete[-1] if complete else None


def trading_day_after(session: date, count: int) -> date:
    """Return date of the n'th future NYSE session, skipping holidays."""
    if count < 1:
        raise ValueError("count must be positive")
    candidates = NYSE.sessions_in_range(session + timedelta(days=1),
                                         session + timedelta(days=count * 3 + 30))
    if len(candidates) < count:
        raise ValueError("No exchange sessions for requested horizon")
    return candidates[count - 1].date()


def fetch_daily_bars(http: requests.Session, cfg: Settings, symbol: str, now: datetime) -> pd.DataFrame:
    """Fetch 560 calendar days of historical split-adjusted daily IEX bars."""
    body = api_get(http, cfg, f"/v2/stocks/{quote(symbol, safe='')}/bars", {
        "timeframe": "1Day", "feed": cfg.feed, "adjustment": "split",
        "start": (now - timedelta(days=560)).isoformat().replace("+00:00", "Z"),
        "end": now.isoformat().replace("+00:00", "Z"),
        "limit": 1000, "sort": "asc",
    })
    values = body.get("bars", [])
    if not values:
        return pd.DataFrame()
    df = pd.DataFrame(values)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.sort_values("t").drop_duplicates("t").set_index("t")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume"})
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    # Daily Alpaca timestamps refer to the session's date in New York.
    df["session"] = [ts.astimezone(ET).date() for ts in df.index.to_pydatetime()]
    return df


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ema20"] = x["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    x["ema200"] = x["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    delta = x["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    x["rsi"] = 100 - 100 / (1 + rs)
    x.loc[(loss == 0) & (gain > 0), "rsi"] = 100.0
    x.loc[(loss == 0) & (gain == 0), "rsi"] = 50.0
    pc = x["close"].shift(1)
    ranges = pd.concat((x["high"] - x["low"], (x["high"] - pc).abs(),
                        (x["low"] - pc).abs()), axis=1)
    x["atr"] = ranges.max(axis=1).rolling(14, min_periods=14).mean()
    x["rvol"] = x["volume"] / x["volume"].shift(1).rolling(20, min_periods=20).mean().replace(0, float("nan"))
    x["prev20_high"] = x["high"].shift(1).rolling(20, min_periods=20).max()
    x["prev20_low"] = x["low"].shift(1).rolling(20, min_periods=20).min()
    return x


def assess_swing(symbol: str, daily: pd.DataFrame, completed_session: date,
                 feed: str = "iex") -> SwingSignal | None:
    """Screen breakouts or EMA20 pullback-reclaims within the EMA50/200 trend."""
    if len(daily) < 215 or daily.iloc[-1]["session"] != completed_session:
        return None
    df = indicators(daily)
    last, prev, earlier = df.iloc[-1], df.iloc[-2], df.iloc[-6]
    needed = ("ema20", "ema50", "ema200", "rsi", "atr", "rvol", "prev20_high", "prev20_low")
    if any(pd.isna(last[k]) or not math.isfinite(float(last[k])) for k in needed):
        return None
    price = float(last["close"])
    atr = float(last["atr"])
    if price <= 1 or atr <= 0 or atr / price > 0.15:
        return None

    long_trend = (price > last["ema20"] > last["ema50"] > last["ema200"]
                  and last["ema50"] > earlier["ema50"] and 52 <= last["rsi"] <= 74)
    short_trend = (price < last["ema20"] < last["ema50"] < last["ema200"]
                   and last["ema50"] < earlier["ema50"] and 26 <= last["rsi"] <= 48)
    vol = float(last["rvol"])
    long_breakout = (price > last["prev20_high"] and
                     float(prev["close"]) <= float(prev["prev20_high"]) and vol >= 1.15)
    short_breakdown = (price < last["prev20_low"] and
                       float(prev["close"]) >= float(prev["prev20_low"]) and vol >= 1.15)
    long_reclaim = (float(prev["close"]) <= float(prev["ema20"]) and
                    price > float(last["ema20"]) and vol >= 0.9)
    short_reject = (float(prev["close"]) >= float(prev["ema20"]) and
                    price < float(last["ema20"]) and vol >= 0.9)

    if long_trend and (long_breakout or long_reclaim):
        direction = "BUY / BULLISH"
        pattern = "20-day breakout" if long_breakout else "20-EMA reclaim"
        stop = min(price - 1.25 * atr, float(df["low"].iloc[-10:].min()) - 0.25 * atr)
        risk = price - stop
        target1, target2 = price + 2 * risk, price + 3 * risk
    elif short_trend and (short_breakdown or short_reject):
        direction = "SELL / BEARISH"
        pattern = "20-day breakdown" if short_breakdown else "20-EMA rejection"
        stop = max(price + 1.25 * atr, float(df["high"].iloc[-10:].max()) + 0.25 * atr)
        risk = stop - price
        target1, target2 = price - 2 * risk, price - 3 * risk
        if target2 <= 0:
            return None
    else:
        return None
    # Reject extremely wide technical stops. No trade sizing or order placement.
    if risk <= 0 or risk / price > 0.10:
        return None
    return SwingSignal(
        symbol=symbol, direction=direction, pattern=pattern, session=completed_session,
        price=price, rsi=float(last["rsi"]), rvol=vol, atr=atr,
        ema20=float(last["ema20"]), ema50=float(last["ema50"]),
        ema200=float(last["ema200"]), stop=stop, target1=target1,
        target2=target2,
        first_checkpoint=trading_day_after(completed_session, 10),
        final_checkpoint=trading_day_after(completed_session, 30), feed=feed,
    )


def message_for_signal(s: SwingSignal) -> str:
    emoji = "🟢" if s.direction.startswith("BUY") else "🔴"
    return (f"{emoji} SWING {s.direction} | {s.symbol}\n"
            f"Pattern: {s.pattern} | Session: {s.session.isoformat()} ET\n"
            f"Reference entry: ${s.price:.2f} (daily close, NOT live quote)\n"
            f"EMA20/50/200: ${s.ema20:.2f} / ${s.ema50:.2f} / ${s.ema200:.2f}\n"
            f"RSI14: {s.rsi:.1f} | Relative daily volume: {s.rvol:.2f}x ({s.feed.upper()} only)\n"
            f"Invalidation/stop reference: ${s.stop:.2f}\n"
            f"Illustrative target 1 (2R): ${s.target1:.2f} — review {s.first_checkpoint}\n"
            f"Illustrative target 2 (3R): ${s.target2:.2f} — review {s.final_checkpoint}\n"
            "Estimated planning horizon: 2–6 weeks (10–30 exchange sessions).\n"
            "DATES ARE REVIEW CHECKPOINTS, NOT PRICE REACH FORECASTS.\n"
            f"Chart: https://www.tradingview.com/symbols/{s.symbol}/\n"
            "Research/watchlist signal only. Validate current price, liquidity, news and risk before trading.")


def new_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE IF NOT EXISTS swing_sent (
        symbol TEXT NOT NULL, direction TEXT NOT NULL,
        session TEXT NOT NULL, sent_at TEXT NOT NULL,
        PRIMARY KEY(symbol, direction, session))""")
    db.commit()
    return db


def is_sent(db: sqlite3.Connection, s: SwingSignal) -> bool:
    return db.execute("SELECT 1 FROM swing_sent WHERE symbol=? AND direction=? AND session=?",
                      (s.symbol, s.direction, s.session.isoformat())).fetchone() is not None


def record_sent(db: sqlite3.Connection, s: SwingSignal, now: datetime) -> None:
    db.execute("INSERT OR IGNORE INTO swing_sent VALUES (?, ?, ?, ?)",
               (s.symbol, s.direction, s.session.isoformat(), now.astimezone(UTC).isoformat()))
    db.commit()


def scan(http: requests.Session, cfg: Settings, db: sqlite3.Connection,
         now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    expected = most_recent_completed_session(now)
    if expected is None:
        LOG.info("No completed NYSE session in lookback")
        return 0
    # Only send immediately after close on an exchange session; avoids old weekend signals.
    if now.astimezone(ET).date() != expected:
        LOG.info("Today is not a just-completed NYSE session; no stale alerts")
        return 0

    init_exit_tables(db)
    n, exits, good, errors = 0, 0, 0, 0
    LOG.info("EOD virtual swing setups loaded: %d", count_open(db, 'eod'))
    for symbol in cfg.watchlist:
        try:
            daily = fetch_daily_bars(http, cfg, symbol, now)
            if daily.empty:
                LOG.warning("%s: no daily bars", symbol)
                errors += 1
                continue
            if daily.iloc[-1]["session"] != expected:
                LOG.warning("%s: stale daily bar dated %s; need %s", symbol,
                            daily.iloc[-1]["session"], expected)
                errors += 1
                continue
            # Exclude any provisional daily candle before checking exit triggers.
            daily = daily.loc[daily["session"] <= expected].copy()
            good += 1
            computed = indicators(daily)
            exits += review_eod(
                db, symbol, daily, computed,
                lambda message: send_telegram(http, cfg, message))
            signal = assess_swing(symbol, daily, expected, cfg.feed)
            if signal and not is_sent(db, signal):
                send_telegram(http, cfg, message_for_signal(signal))
                record_sent(db, signal, now)
                # Open a virtual setup ONLY after a delivered confirmed entry signal.
                open_virtual(db, 'eod', signal, signal.session.isoformat())
                n += 1
                LOG.info("Sent %s %s swing signal", symbol, signal.direction)
        except requests.HTTPError as exc:
            errors += 1
            LOG.error("%s data HTTP %s", symbol,
                      exc.response.status_code if exc.response is not None else "unknown")
        except Exception:
            errors += 1
            LOG.exception("%s swing scan failed", symbol)
    if good == 0:
        raise RuntimeError("No symbols produced fresh daily data; check Alpaca keys/feed and API logs")
    LOG.info("Swing scan completed for %s: %d fresh symbols, %d errors, %d ENTRY alerts, %d EXIT alerts, %d virtual setups remaining",
             expected, good, errors, n, exits, count_open(db, 'eod'))
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily swing scanner: signals only, no orders")
    parser.add_argument("--once", action="store_true", help="Scan completed NYSE day once")
    parser.add_argument("--test-alert", action="store_true", help="Telegram connectivity test only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Settings.read()
    if not cfg.tg_token or not cfg.tg_chat_id:
        parser.error("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    if not args.test_alert and not (cfg.key and cfg.secret):
        parser.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
    if cfg.feed not in ("iex", "sip"):
        parser.error("DATA_FEED must be iex or sip")
    http = make_http()
    try:
        if args.test_alert:
            send_telegram(http, cfg, "✅ TEST ONLY — Swing trade scanner connected. No real trade setup.")
        else:
            db = new_db(Path(os.getenv("SWING_DB_PATH", "swing_alerts.db")))
            try:
                scan(http, cfg, db)
            finally:
                db.close()
    finally:
        http.close()


if __name__ == "__main__":
    main()
