#!/usr/bin/env python3
"""Day-trading signal WATCH bot. Sends alerts; NEVER places orders."""
from __future__ import annotations

import argparse
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time as dtime
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import exchange_calendars as xcals
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
ALPACA = "https://data.alpaca.markets"
NYSE = xcals.get_calendar("XNYS")
LOG = logging.getLogger("day-trading-agent")
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


@dataclass(frozen=True)
class Settings:
    key: str
    secret: str
    tg_token: str
    tg_chat_id: str
    feed: str
    watchlist: tuple[str, ...]
    scan_top_active: int
    poll_seconds: int
    min_rvol: float
    cooldown_minutes: int
    db_path: Path

    @classmethod
    def read(cls) -> "Settings":
        load_dotenv()
        watch = tuple(dict.fromkeys(s.strip().upper() for s in
                                        os.getenv("WATCHLIST", "SPY,QQQ,IWM,NVDA,TSLA,AMD,AMZN,PLTR,HOOD,AAPL,MSFT,META")
                                        .split(",") if SYMBOL_RE.match(s.strip().upper())))
        return cls(
            key=os.getenv("ALPACA_API_KEY", "").strip(),
            secret=os.getenv("ALPACA_SECRET_KEY", "").strip(),
            tg_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            tg_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            feed=os.getenv("DATA_FEED", "iex").strip().lower(),
            watchlist=watch,
            scan_top_active=max(0, min(20, int(os.getenv("TOP_ACTIVE", "10")))),
            poll_seconds=max(30, int(os.getenv("POLL_SECONDS", "60"))),
            min_rvol=max(1, float(os.getenv("MIN_RVOL", "1.3"))),
            cooldown_minutes=max(0, int(os.getenv("COOLDOWN_MINUTES", "45"))),
            db_path=Path(os.getenv("DB_PATH", "alerts.db")),
        )


def make_http() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.7, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "POST"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def api_headers(s: Settings) -> dict[str, str]:
    return {"APCA-API-KEY-ID": s.key, "APCA-API-SECRET-KEY": s.secret}


def api_get(http: requests.Session, cfg: Settings, path: str, params: dict) -> dict:
    r = http.get(ALPACA + path, headers=api_headers(cfg), params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def trading_window(now: datetime) -> Optional[tuple[datetime, datetime]]:
    """Calendar handles holidays and early closes. Returns exchange open & close (UTC)."""
    d = now.astimezone(ET).date()
    if not NYSE.is_session(d):
        return None
    o = NYSE.session_open(d).to_pydatetime()
    c = NYSE.session_close(d).to_pydatetime()
    return o, c


def fetch_bars(http: requests.Session, cfg: Settings, sym: str, now: datetime) -> pd.DataFrame:
    """Seven-day lookback warms EMA/RSI/ATR. Only consume completed 5-minute bars."""
    d = api_get(http, cfg, f"/v2/stocks/{quote(sym, safe='')}/bars", {
        "timeframe": "5Min", "feed": cfg.feed, "adjustment": "split",
        "start": (now - timedelta(days=7)).isoformat().replace("+00:00", "Z"),
        "end": now.isoformat().replace("+00:00", "Z"),
        "limit": 10000,
        "sort": "asc",
    })
    vals = d.get("bars", [])
    if not vals:
        return pd.DataFrame()
    df = pd.DataFrame(vals)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.sort_values("t").drop_duplicates("t").set_index("t")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    # VWAP means regular-session VWAP: omit premarket/after-hours candles on all days.
    exchange_time = df.index.tz_convert(ET).time
    rth = [(t >= dtime(9, 30) and t < dtime(16, 0)) for t in exchange_time]
    df = df.loc[rth]
    # Bars can be provisional in a provider response. Wait 20 seconds beyond close.
    cutoff = now - timedelta(seconds=20)
    return df[df.index + pd.Timedelta(minutes=5) <= cutoff].copy()


def with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """EMA(9/21), Wilder RSI(14), session VWAP, prior-20-bar relative volume, ATR(14)."""
    if df.empty:
        return df.copy()
    x = df.copy()
    x["ema9"] = x["close"].ewm(span=9, adjust=False, min_periods=9).mean()
    x["ema21"] = x["close"].ewm(span=21, adjust=False, min_periods=21).mean()
    delta = x["close"].diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ag = up.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    al = down.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = ag / al.replace(0, float("nan"))
    x["rsi"] = 100 - 100 / (1 + rs)
    x.loc[(al == 0) & (ag > 0), "rsi"] = 100.0
    x.loc[(al == 0) & (ag == 0), "rsi"] = 50.0
    pc = x["close"].shift(1)
    tr = pd.concat([x["high"] - x["low"], (x["high"] - pc).abs(),
                    (x["low"] - pc).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14, min_periods=14).mean()
    x["rvol"] = x["volume"] / x["volume"].shift(1).rolling(20, min_periods=20).mean().replace(0, float("nan"))
    # NYSE trading date (not UTC date) is used for VWAP reset.
    sessions = x.index.tz_convert(ET).date
    tpv = ((x["high"] + x["low"] + x["close"]) / 3) * x["volume"]
    x["vwap"] = tpv.groupby(sessions).cumsum() / x["volume"].groupby(sessions).cumsum().replace(0, float("nan"))
    return x


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: str
    bar_time: datetime
    price: float
    ema9: float
    ema21: float
    vwap: float
    rsi: float
    rvol: float
    atr: float
    stop: float
    target: float
    feed: str


def assess(symbol: str, df: pd.DataFrame, now: datetime, cfg: Settings) -> Optional[Signal]:
    if len(df) < 35:
        return None
    df = with_indicators(df)
    # 3 consecutive closed 5-minute bars required, to exclude sparse IEX series.
    last = df.iloc[-1]
    prev = df.iloc[-2]
    t = df.index[-1].to_pydatetime()
    if (now - (t + timedelta(minutes=5))) > timedelta(minutes=10):
        return None
    if not (df.index[-1] - df.index[-2] == pd.Timedelta(minutes=5) and
            df.index[-2] - df.index[-3] == pd.Timedelta(minutes=5)):
        return None
    day = now.astimezone(ET).date()
    if t.astimezone(ET).date() != day or df.index[-2].tz_convert(ET).date() != day:
        return None
    for f in ("ema9", "ema21", "vwap", "rsi", "rvol", "atr"):
        if pd.isna(last[f]):
            return None
    # A new setup occurs only as all factors transition to alignment.
    def aligned(row: pd.Series, side: str) -> bool:
        if side == "BUY":
            return bool(row["ema9"] > row["ema21"] and row["close"] > row["vwap"]
                        and 55 <= row["rsi"] <= 80 and row["rvol"] >= cfg.min_rvol)
        return bool(row["ema9"] < row["ema21"] and row["close"] < row["vwap"]
                    and 20 <= row["rsi"] <= 45 and row["rvol"] >= cfg.min_rvol)
    side = next((s for s in ("BUY", "SELL/BEARISH") if aligned(last, s) and not aligned(prev, s)), None)
    if not side:
        return None
    price = float(last["close"])
    dist = max(1.5 * float(last["atr"]), price * 0.001)
    if side == "BUY":
        stop, target = price - dist, price + 2 * dist
    else:
        stop, target = price + dist, max(0, price - 2 * dist)
    return Signal(symbol, side, t, price, float(last["ema9"]), float(last["ema21"]),
                  float(last["vwap"]), float(last["rsi"]), float(last["rvol"]),
                  float(last["atr"]), stop, target, cfg.feed)


def create_db(cfg: Settings) -> sqlite3.Connection:
    db = sqlite3.connect(cfg.db_path)
    db.execute("""CREATE TABLE IF NOT EXISTS alerts (
       symbol TEXT NOT NULL, side TEXT NOT NULL, bar_time TEXT NOT NULL,
       sent_at TEXT NOT NULL, PRIMARY KEY(symbol, side, bar_time))""")
    db.commit()
    return db


def is_duplicate(db: sqlite3.Connection, sig: Signal, cfg: Settings) -> bool:
    last = db.execute("SELECT sent_at FROM alerts WHERE symbol=? AND side=? ORDER BY sent_at DESC LIMIT 1",
                      (sig.symbol, sig.side)).fetchone()
    if not last:
        return False
    ts = datetime.fromisoformat(last[0])
    return (datetime.now(UTC) - ts) < timedelta(minutes=cfg.cooldown_minutes)


def mark_sent(db: sqlite3.Connection, sig: Signal) -> None:
    db.execute("INSERT OR IGNORE INTO alerts(symbol, side, bar_time, sent_at) VALUES (?, ?, ?, ?)",
               (sig.symbol, sig.side, sig.bar_time.isoformat(), datetime.now(UTC).isoformat()))
    db.commit()


def format_alert(sig: Signal) -> str:
    et = sig.bar_time.astimezone(ET)
    tag = "🟢" if sig.side == "BUY" else "🔴"
    return (f"{tag} {sig.side} SETUP | {sig.symbol}\n"
            f"Closed 5-min bar: {et:%Y-%m-%d %I:%M %p} ET (start time)\n"
            f"Reference price: ${sig.price:,.2f}\n"
            f"EMA9 / EMA21: {sig.ema9:.2f} / {sig.ema21:.2f}\n"
            f"Session VWAP: ${sig.vwap:.2f} | RSI14: {sig.rsi:.1f}\n"
            f"Relative volume: {sig.rvol:.2f}x | ATR14: ${sig.atr:.2f}\n"
            f"Indicative invalidation/stop: ${sig.stop:.2f}\n"
            f"Illustrative 2R target: ${sig.target:.2f}\n"
            f"Feed: Alpaca {sig.feed.upper()} | {sig.symbol} chart: https://www.tradingview.com/symbols/{sig.symbol}/\n"
            "RESEARCH SIGNAL ONLY. Prices can change; not a trade order. Confirm on your broker's live chart.")


def send_telegram(http: requests.Session, cfg: Settings, message: str) -> bool:
    if not (cfg.tg_token and cfg.tg_chat_id):
        print("\n" + message + "\n", flush=True)
        LOG.warning("Telegram not configured; alert printed to console only")
        return True
    res = http.post(f"https://api.telegram.org/bot{cfg.tg_token}/sendMessage",
                    json={"chat_id": cfg.tg_chat_id, "text": message,
                          "disable_web_page_preview": True}, timeout=15)
    res.raise_for_status()
    body = res.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram reported failure: {body.get('description', 'unknown')}")
    return True


def active_symbols(http: requests.Session, cfg: Settings) -> list[str]:
    symbols = list(cfg.watchlist)
    if not cfg.scan_top_active:
        return symbols
    try:
        resp = api_get(http, cfg, "/v1beta1/screener/stocks/most-actives",
                       {"by": "volume", "top": cfg.scan_top_active})
        for item in resp.get("most_actives", []):
            symbol = item.get("symbol", "").upper()
            if SYMBOL_RE.fullmatch(symbol) and symbol not in symbols:
                symbols.append(symbol)
    except requests.HTTPError as exc:
        LOG.warning("Top-active screener unavailable (%s); using watchlist", exc.response.status_code if exc.response is not None else str(exc))
    except Exception as exc:
        LOG.warning("Top-active screener failed (%s); using watchlist", exc)
    return symbols[:35]


def run_scan(http: requests.Session, cfg: Settings, db: sqlite3.Connection,
             now: Optional[datetime] = None) -> int:
    now = now or datetime.now(UTC)
    window = trading_window(now)
    if window is None:
        LOG.info("NYSE closed: holiday or weekend")
        return 0
    op, cl = window
    if not (op + timedelta(minutes=15) <= now <= cl - timedelta(minutes=10)):
        LOG.info("Outside active intraday scan window: %s", now.astimezone(ET))
        return 0
    count = 0
    successes = 0
    evaluable = 0
    failures = 0
    symbols = active_symbols(http, cfg)
    for sym in symbols:
        try:
            bars = fetch_bars(http, cfg, sym, now)
            successes += 1
            if bars.empty:
                LOG.warning("%s: API responded, but no completed regular-session 5-minute bars were returned", sym)
                continue
            last_bar = bars.index[-1].to_pydatetime()
            minutes_since_close = (now - (last_bar + timedelta(minutes=5))).total_seconds() / 60
            last_three_contiguous = len(bars) >= 3 and (
                bars.index[-1] - bars.index[-2] == pd.Timedelta(minutes=5) and
                bars.index[-2] - bars.index[-3] == pd.Timedelta(minutes=5))
            same_session = last_bar.astimezone(ET).date() == now.astimezone(ET).date()
            usable = (len(bars) >= 35 and 0 <= minutes_since_close <= 10 and
                      last_three_contiguous and same_session)
            LOG.info("%s: %d completed bars; latest %s ET; %.1f min since close; %s",
                     sym, len(bars), last_bar.astimezone(ET).strftime("%I:%M %p"),
                     minutes_since_close, "data eligible" if usable else "data stale/sparse/insufficient")
            if usable:
                evaluable += 1
            sig = assess(sym, bars, now, cfg)
            if sig and not is_duplicate(db, sig, cfg):
                msg = format_alert(sig)
                if send_telegram(http, cfg, msg):
                    mark_sent(db, sig)
                    count += 1
                    LOG.info("Sent %s %s setup", sym, sig.side)
        except requests.HTTPError as exc:
            failures += 1
            LOG.warning("%s request failed: HTTP %s", sym, exc.response.status_code if exc.response is not None else "?")
        except Exception:
            failures += 1
            LOG.exception("%s scan failed", sym)
    LOG.info("DATA HEALTH: %d/%d symbols fetched, %d had fresh continuous bars, %d errors; %d new alert(s)",
             successes, len(symbols), evaluable, failures, count)
    if successes == 0:
        raise RuntimeError("No market data could be fetched. Check Alpaca credentials, feed and API errors above.")
    if evaluable == 0:
        raise RuntimeError("No symbols had enough fresh continuous bars to evaluate. Check IEX data coverage / timestamps above.")
    LOG.info("Scan complete: %d new alert(s)", count)
    return count


def main() -> None:
    p = argparse.ArgumentParser(description="5-minute day-trading alert bot (never trades)")
    p.add_argument("--once", action="store_true", help="Run one scan now, then exit")
    p.add_argument("--test-alert", action="store_true", help="Send an obviously labeled connection test")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Settings.read()
    if args.test_alert and not (cfg.tg_token and cfg.tg_chat_id):
        p.error("--test-alert requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    if not args.test_alert and not (cfg.key and cfg.secret):
        p.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
    if cfg.feed not in ("iex", "sip"):
        p.error("DATA_FEED must be 'iex' or 'sip'")
    http = make_http()
    if args.test_alert:
        send_telegram(http, cfg, "✅ TEST ONLY — Trading signal bot is connected. This is NOT a buy or sell alert.")
        print("Test sent")
        return
    db = create_db(cfg)
    try:
        if args.once:
            run_scan(http, cfg, db)
        else:
            LOG.info("Agent started. Monitoring %s and up to %d most-active stocks; feed=%s",
                     ",".join(cfg.watchlist), cfg.scan_top_active, cfg.feed)
            while True:
                try:
                    run_scan(http, cfg, db)
                except Exception:
                    LOG.exception("Scan cycle failed; trying again on next poll")
                time.sleep(cfg.poll_seconds)
    except KeyboardInterrupt:
        LOG.info("Stopped")
    finally:
        db.close()
        http.close()


if __name__ == "__main__":
    main()
