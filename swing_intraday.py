#!/usr/bin/env python3
"""Swing-trade watch scanner that evaluates *completed 5-minute IEX bars* intraday.

Keeps historical trend context from the preceding completed NYSE daily session.
Sends Telegram watch signals only; never places trades. Requires agent.py,
swing_agent.py, and requirements.txt from the existing GitHub repository.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from agent import ET, UTC, NYSE, Settings, api_get, make_http, send_telegram
from swing_agent import (SwingSignal, fetch_daily_bars, indicators,
                         most_recent_completed_session, new_db, is_sent,
                         record_sent, trading_day_after)

LOG = logging.getLogger("swing-intraday")
BAR_INTERVAL = timedelta(minutes=5)
FRESHNESS_LIMIT = timedelta(minutes=20)


def scan_window(now: datetime):
    """Only attempt scans during an NYSE session, after 3 completed candles."""
    now = now.astimezone(UTC)
    session = now.astimezone(ET).date()
    if not NYSE.is_session(session):
        return None
    open_at = NYSE.session_open(session).to_pydatetime()
    close_at = NYSE.session_close(session).to_pydatetime()
    if not (open_at + timedelta(minutes=16) <= now < close_at):
        return None
    return session, open_at, close_at


def fetch_intraday_bars(http: requests.Session, cfg: Settings, sym: str,
                        now: datetime, open_at: datetime, close_at: datetime) -> pd.DataFrame:
    """Only use IEX 5-minute bars that have fully closed and are within RTH."""
    response = api_get(http, cfg, f"/v2/stocks/{quote(sym, safe='')}/bars", {
        "timeframe": "5Min", "feed": cfg.feed, "adjustment": "split",
        "start": open_at.isoformat().replace("+00:00", "Z"),
        "end": min(now, close_at).isoformat().replace("+00:00", "Z"),
        "limit": 1000, "sort": "asc",
    })
    vals = response.get("bars", [])
    if not vals:
        return pd.DataFrame()
    df = pd.DataFrame(vals)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.sort_values("t").drop_duplicates("t").set_index("t")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    # A few seconds of grace to avoid using provider's unfinished candle.
    eligible = ((df.index >= pd.Timestamp(open_at)) &
                (df.index < pd.Timestamp(close_at)) &
                (df.index + pd.Timedelta(minutes=5) <= pd.Timestamp(now - timedelta(seconds=20))))
    return df.loc[eligible].copy()


def assess_intraday_swing(sym: str, historical: pd.DataFrame, intraday: pd.DataFrame,
                          session, now: datetime, feed: str = "iex") -> SwingSignal | None:
    """Prior-day EMA20/50/200 trend + 2 confirmed 5-minute breakout/reclaim bars.

    Only uses *completed* prior daily candles; today's daily candle is not used.
    This is a provisional swing entry watch, not a prediction or trade order.
    """
    if len(historical) < 215 or len(intraday) < 2:
        return None
    if historical.iloc[-1]["session"] >= session:
        return None
    prior = indicators(historical)
    last_daily, week_ago = prior.iloc[-1], prior.iloc[-6]
    need = ("ema20", "ema50", "ema200", "rsi", "atr")
    if any(pd.isna(last_daily[x]) or not math.isfinite(float(last_daily[x])) for x in need):
        return None
    close1, close2 = intraday.iloc[-2], intraday.iloc[-1]
    last_bar_start = intraday.index[-1].to_pydatetime()
    if (now.astimezone(UTC) - (last_bar_start + BAR_INTERVAL) > FRESHNESS_LIMIT or
        intraday.index[-1] - intraday.index[-2] != pd.Timedelta(minutes=5) or
        last_bar_start.astimezone(ET).date() != session or
        min(float(close1["volume"]), float(close2["volume"])) <= 0):
        return None
    price = float(close2["close"])
    atr = float(last_daily["atr"])
    prev_close = float(last_daily["close"])
    ema20, ema50, ema200 = (float(last_daily[x]) for x in ("ema20", "ema50", "ema200"))
    rsi = float(last_daily["rsi"])
    if price <= 1 or atr <= 0 or atr / price > 0.15:
        return None

    high20 = float(historical["high"].tail(20).max())
    low20 = float(historical["low"].tail(20).min())
    long_trend = (ema20 > ema50 > ema200 and
                  ema50 > float(week_ago["ema50"]) and 50 <= rsi <= 75)
    short_trend = (ema20 < ema50 < ema200 and
                   ema50 < float(week_ago["ema50"]) and 25 <= rsi <= 50)
    long_break = (prev_close <= high20 and
                  float(close1["close"]) > high20 and price > high20)
    short_break = (prev_close >= low20 and
                   float(close1["close"]) < low20 and price < low20)
    long_reclaim = (prev_close <= ema20 and
                    float(close1["close"]) > ema20 and price > ema20)
    short_reject = (prev_close >= ema20 and
                    float(close1["close"]) < ema20 and price < ema20)

    if long_trend and (long_break or long_reclaim):
        direction, pattern = "BUY / BULLISH", ("Intraday 20-day breakout" if long_break
                                                   else "Intraday 20-EMA reclaim")
        stop = min(price - 1.25 * atr, float(historical["low"].tail(10).min()) - 0.25 * atr)
        risk = price - stop
        target1, target2 = price + 2*risk, price + 3*risk
    elif short_trend and (short_break or short_reject):
        direction, pattern = "SELL / BEARISH", ("Intraday 20-day breakdown" if short_break
                                                    else "Intraday 20-EMA rejection")
        stop = max(price + 1.25 * atr, float(historical["high"].tail(10).max()) + 0.25 * atr)
        risk = stop - price
        target1, target2 = price - 2*risk, price - 3*risk
        if target2 <= 0:
            return None
    else:
        return None
    if risk <= 0 or risk / price > 0.10:
        return None
    return SwingSignal(
        symbol=sym, direction=direction, pattern=pattern,
        session=session, price=price, rsi=rsi, rvol=float("nan"), atr=atr,
        ema20=ema20, ema50=ema50, ema200=ema200, stop=stop,
        target1=target1, target2=target2,
        first_checkpoint=trading_day_after(session, 10),
        final_checkpoint=trading_day_after(session, 30), feed=feed,
    )


def alert_message(s: SwingSignal, last_bar_start: datetime) -> str:
    icon = "🟢" if s.direction.startswith("BUY") else "🔴"
    observed = (last_bar_start + BAR_INTERVAL).astimezone(ET).strftime("%Y-%m-%d %I:%M %p ET")
    return (f"{icon} INTRADAY SWING WATCH — {s.symbol} {s.direction}\n"
            f"Setup: {s.pattern}\n"
            f"Last completed 5-min bar: {observed}\n"
            f"Indicative IEX reference price: ${s.price:.2f} (NOT executable quote)\n"
            f"Prior-day EMA20/50/200: ${s.ema20:.2f} / ${s.ema50:.2f} / ${s.ema200:.2f}\n"
            f"Prior-day RSI14: {s.rsi:.1f} | Prior-day ATR14: ${s.atr:.2f}\n"
            f"Stop/invalidation reference: ${s.stop:.2f}\n"
            f"Illustrative target 1 (2R): ${s.target1:.2f} | review {s.first_checkpoint}\n"
            f"Illustrative target 2 (3R): ${s.target2:.2f} | review {s.final_checkpoint}\n"
            "Review dates are checkpoints (10/30 trading sessions), NOT arrival predictions.\n"
            f"Chart: https://www.tradingview.com/symbols/{s.symbol}/\n"
            "IEX is a single-exchange feed; intraday breakout is provisional. "
            "Check consolidated price, liquidity, news and risk before trading. No orders placed.")


def scan(http: requests.Session, cfg: Settings, db, now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    window = scan_window(now)
    if window is None:
        LOG.info("Outside active NYSE intraday swing scan window — skipped")
        return 0
    today, open_at, close_at = window
    completed = most_recent_completed_session(now)
    if completed is None or completed >= today:
        LOG.info("No prior completed NYSE session available — skipped")
        return 0
    good, errors, alerts = 0, 0, 0
    for sym in cfg.watchlist:
        try:
            daily = fetch_daily_bars(http, cfg, sym, now)
            if daily.empty:
                LOG.warning("%s no historical daily bars", sym)
                errors += 1
                continue
            # The provider might return a partial daily candle for today.
            daily = daily.loc[daily["session"] <= completed].copy()
            if daily.empty or daily.iloc[-1]["session"] != completed:
                LOG.warning("%s stale daily history; need %s", sym, completed)
                errors += 1
                continue
            bars = fetch_intraday_bars(http, cfg, sym, now, open_at, close_at)
            if bars.empty or len(bars) < 2:
                LOG.warning("%s no recent completed intraday bars", sym)
                errors += 1
                continue
            if now - (bars.index[-1].to_pydatetime() + BAR_INTERVAL) > FRESHNESS_LIMIT:
                LOG.warning("%s intraday data stale at %s", sym, bars.index[-1])
                errors += 1
                continue
            good += 1
            signal = assess_intraday_swing(sym, daily, bars, today, now, cfg.feed)
            if signal is not None and not is_sent(db, signal):
                send_telegram(http, cfg, alert_message(signal, bars.index[-1].to_pydatetime()))
                record_sent(db, signal, now)
                alerts += 1
                LOG.info("Sent %s %s intraday swing watch", sym, signal.direction)
        except requests.HTTPError as exc:
            errors += 1
            LOG.error("%s Alpaca HTTP %s", sym,
                      exc.response.status_code if exc.response is not None else "unknown")
        except Exception:
            errors += 1
            LOG.exception("%s intraday swing scan failed", sym)
    if not good:
        raise RuntimeError("No fresh Alpaca data verified for any symbol; check feed, keys and logs")
    LOG.info("Intraday swing scan complete: %d valid symbols, %d errors, %d new alerts",
             good, errors, alerts)
    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description="Intraday swing watch signals; no execution")
    parser.add_argument("--once", action="store_true", help="Run one market-hour scan")
    parser.add_argument("--test-alert", action="store_true", help="Send Telegram test only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Settings.read()
    if not (cfg.tg_token and cfg.tg_chat_id):
        parser.error("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    if not args.test_alert and not (cfg.key and cfg.secret):
        parser.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
    if cfg.feed not in ("iex", "sip"):
        parser.error("DATA_FEED must be iex or sip")
    http = make_http()
    try:
        if args.test_alert:
            send_telegram(http, cfg, "✅ TEST ONLY — INTRADAY swing watch connected; no market scan.")
        else:
            db = new_db(Path(os.getenv("SWING_DB_PATH", "swing_intraday_sent.db")))
            try:
                scan(http, cfg, db)
            finally:
                db.close()
    finally:
        http.close()


if __name__ == "__main__":
    main()
