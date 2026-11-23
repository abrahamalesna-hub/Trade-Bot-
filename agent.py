#!/usr/bin/env python3
"""Day-trading alert bot with quiet CONFIRMED digests and virtual exit tracking. NEVER trades."""
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
    max_entries_per_day: int
    db_path: Path

    @classmethod
    def read(cls) -> "Settings":
        load_dotenv()
        watch = tuple(dict.fromkeys(s.strip().upper() for s in
                                        os.getenv("WATCHLIST", "SPY,QQQ,IWM,DIA,NVDA,TSLA,AMD,AMZN,PLTR,HOOD,AAPL,MSFT,META,GOOGL,AVGO,MU,SMCI,CRWD,COIN,MSTR,NFLX,SOFI,SHOP,UBER")
                                        .split(",") if SYMBOL_RE.match(s.strip().upper())))
        return cls(
            key=os.getenv("ALPACA_API_KEY", "").strip(),
            secret=os.getenv("ALPACA_SECRET_KEY", "").strip(),
            tg_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            tg_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            feed=os.getenv("DATA_FEED", "iex").strip().lower(),
            watchlist=watch,
            scan_top_active=max(0, min(20, int(os.getenv("TOP_ACTIVE", "0")))),
            poll_seconds=max(30, int(os.getenv("POLL_SECONDS", "60"))),
            min_rvol=max(1, float(os.getenv("MIN_RVOL", "1.6"))),
            cooldown_minutes=max(0, int(os.getenv("COOLDOWN_MINUTES", "180"))),
            max_entries_per_day=max(0, int(os.getenv("MAX_ENTRIES_PER_DAY", "2"))),
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
    level: str = "CONFIRMED"
    matched: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()


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


def assess_levels(symbol: str, bars: pd.DataFrame, now: datetime,
                  cfg: Settings) -> tuple[list[Signal], str]:
    """WATCH = 3/4, CONFIRMED = 4/4. Ranking is not a profit guarantee."""
    if len(bars) < 35:
        return [], "insufficient completed candles"
    if (bars.index[-1] - bars.index[-2] != pd.Timedelta(minutes=5) or
        bars.index[-2] - bars.index[-3] != pd.Timedelta(minutes=5)):
        return [], "latest 3 candles are not consecutive"
    t = bars.index[-1].to_pydatetime()
    age = now - (t + timedelta(minutes=5))
    if not (timedelta(0) <= age <= timedelta(minutes=10)):
        return [], f"stale/latest bar age {age.total_seconds() / 60:.1f} minutes"
    if t.astimezone(ET).date() != now.astimezone(ET).date():
        return [], "latest candle belongs to a prior NYSE trading date"
    x = with_indicators(bars)
    last = x.iloc[-1]
    required = ("ema9", "ema21", "vwap", "rsi", "rvol", "atr")
    if any(pd.isna(last[k]) for k in required):
        return [], "indicator warmup/validity incomplete"
    price = float(last["close"])
    atr = float(last["atr"])
    if price <= 0 or not pd.notna(atr) or atr <= 0:
        return [], "nonpositive price/ATR"
    rsi = float(last["rsi"])
    rvol = float(last["rvol"])
    sides = (
        ("BUY", {"EMA": last["ema9"] > last["ema21"],
                 "VWAP": price > last["vwap"], "RSI": 55 <= rsi <= 80,
                 "RVOL": rvol >= cfg.min_rvol}),
        ("SELL/BEARISH", {"EMA": last["ema9"] < last["ema21"],
                          "VWAP": price < last["vwap"], "RSI": 20 <= rsi <= 45,
                          "RVOL": rvol >= cfg.min_rvol}),
    )
    diagnostics = []
    out: list[Signal] = []
    for side, tests in sides:
        passed = tuple(k for k, ok in tests.items() if bool(ok))
        failed = tuple(k for k, ok in tests.items() if not bool(ok))
        diagnostics.append(f"{side} {len(passed)}/4; missing: {', '.join(failed) if failed else 'none'}")
        if len(passed) < 3:
            continue
        level = "CONFIRMED" if len(passed) == 4 else "WATCH"
        dist = max(1.5 * atr, price * 0.001)
        stop = price - dist if side == "BUY" else price + dist
        target = price + 2 * dist if side == "BUY" else max(0.0, price - 2 * dist)
        if target <= 0:
            continue
        out.append(Signal(symbol, side, t, price, float(last["ema9"]), float(last["ema21"]),
                          float(last["vwap"]), rsi, rvol, atr, stop, target, cfg.feed,
                          level=level, matched=passed, missing=failed))
    return out, " | ".join(diagnostics)


def create_db(cfg: Settings) -> sqlite3.Connection:
    db = sqlite3.connect(cfg.db_path)
    db.execute("""CREATE TABLE IF NOT EXISTS alerts (
       symbol TEXT NOT NULL, side TEXT NOT NULL, bar_time TEXT NOT NULL,
       sent_at TEXT NOT NULL, PRIMARY KEY(symbol, side, bar_time))""")
    db.execute("""CREATE TABLE IF NOT EXISTS tracked_setups (
       symbol TEXT PRIMARY KEY, side TEXT NOT NULL, entry_bar TEXT NOT NULL,
       entry_reference REAL NOT NULL, stop REAL NOT NULL, target REAL NOT NULL,
       last_checked_bar TEXT NOT NULL, feed TEXT NOT NULL)""")
    db.commit()
    return db


def _storage_side(sig: Signal) -> str:
    return f"{sig.level}:{sig.side}"


def is_duplicate(db: sqlite3.Connection, sig: Signal, cfg: Settings) -> bool:
    """One alert per symbol / direction / tier / NYSE session (same schema as old bot)."""
    rows = db.execute("SELECT bar_time FROM alerts WHERE symbol=? AND side=? ORDER BY sent_at DESC LIMIT 10",
                      (sig.symbol, _storage_side(sig))).fetchall()
    day = sig.bar_time.astimezone(ET).date()
    return any(datetime.fromisoformat(row[0]).astimezone(ET).date() == day for row in rows)


def mark_sent(db: sqlite3.Connection, sig: Signal) -> None:
    db.execute("INSERT OR IGNORE INTO alerts(symbol, side, bar_time, sent_at) VALUES (?, ?, ?, ?)",
               (sig.symbol, _storage_side(sig), sig.bar_time.isoformat(), datetime.now(UTC).isoformat()))
    db.commit()


def entries_sent_today(db: sqlite3.Connection, now: datetime) -> int:
    """Counts previously delivered CONFIRMED entries using persisted NYSE-day bar timestamps."""
    day = now.astimezone(ET).date()
    rows = db.execute("SELECT bar_time FROM alerts WHERE side LIKE 'CONFIRMED:%'").fetchall()
    count = 0
    for (bar,) in rows:
        try:
            if datetime.fromisoformat(bar).astimezone(ET).date() == day:
                count += 1
        except (ValueError, TypeError):
            LOG.warning("Ignoring invalid historical timestamp: %s", bar)
    return count


def format_alert(sig: Signal) -> str:
    """Original detailed formatter preserved for debugging or manual review."""
    closed = (sig.bar_time + timedelta(minutes=5)).astimezone(ET)
    icon = "ð¢" if sig.side == "BUY" else "ð´"
    tier_note = ("Early watch: 3/4 checks align; not a confirmed entry."
                 if sig.level == "WATCH" else
                 "Rule-confirmed: 4/4 technical checks align; not a guaranteed trade.")
    return (f"{icon} {sig.level} {sig.side} SETUP | {sig.symbol}\n"
            f"{tier_note}\n"
            f"Completed 5-min candle: {closed:%Y-%m-%d %I:%M %p} ET\n"
            f"IEX reference price: ${sig.price:,.2f} (not executable quote)\n"
            f"Checks met: {', '.join(sig.matched) if sig.matched else 'all four'}\n"
            f"Checks missing: {', '.join(sig.missing) if sig.missing else 'none'}\n"
            f"EMA9 / EMA21: {sig.ema9:.2f} / {sig.ema21:.2f}\n"
            f"Session VWAP: ${sig.vwap:.2f} | RSI14: {sig.rsi:.1f}\n"
            f"Relative volume: {sig.rvol:.2f}x | ATR14: ${sig.atr:.2f}\n"
            f"Illustrative stop reference: ${sig.stop:.2f}\n"
            f"Illustrative 2R target: ${sig.target:.2f}\n"
            f"Alpaca {sig.feed.upper()} single-exchange feed; volume may not represent all US trading.\n"
            f"Chart: https://www.tradingview.com/symbols/{sig.symbol}/\n"
            "Research watch only. Verify news, liquidity, current price and risks. No orders placed.")




@dataclass(frozen=True)
class SwingContext:
    """Daily-chart context ONLY; not a separately generated swing-trade entry."""
    classification: str
    as_of: Optional[str] = None
    explanation: str = ""


def fetch_daily_bars(http: requests.Session, cfg: Settings, sym: str,
                     now: datetime) -> pd.DataFrame:
    """Daily bars for a separate swing trend check; EXCLUDES today's incomplete bar."""
    d = api_get(http, cfg, f"/v2/stocks/{quote(sym, safe='')}/bars", {
        "timeframe": "1Day", "feed": cfg.feed, "adjustment": "split",
        "start": (now - timedelta(days=280)).isoformat().replace("+00:00", "Z"),
        "end": now.isoformat().replace("+00:00", "Z"),
        "limit": 10000, "sort": "asc",
    })
    vals = d.get("bars", [])
    if not vals:
        return pd.DataFrame()
    df = pd.DataFrame(vals)
    if not all(c in df.columns for c in ("t", "c")):
        return pd.DataFrame()
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["close"] = pd.to_numeric(df["c"], errors="coerce")
    df = df.dropna(subset=["t", "close"]).sort_values("t").drop_duplicates("t").set_index("t")
    # A daily candle that belongs to the current NYSE date may still be forming.
    et_day = now.astimezone(ET).date()
    df = df.loc[[d < et_day for d in df.index.tz_convert(ET).date]]
    return df[["close"]].copy()


def assess_swing_context(bars: pd.DataFrame, side: str,
                         now: datetime) -> SwingContext:
    """Trend alignment is not a swing entry signal or a holding instruction."""
    if bars.empty or len(bars) < 70:
        return SwingContext("UNAVAILABLE", explanation="Insufficient completed daily candles")
    last_day = bars.index[-1].tz_convert(ET).date()
    if (now.astimezone(ET).date() - last_day).days > 7:
        return SwingContext("UNAVAILABLE", explanation="Daily data too old")
    closes = pd.to_numeric(bars["close"], errors="coerce").dropna()
    if len(closes) < 70 or (closes <= 0).any():
        return SwingContext("UNAVAILABLE", explanation="Invalid daily close series")
    e10 = closes.ewm(span=10, adjust=False, min_periods=10).mean()
    e20 = closes.ewm(span=20, adjust=False, min_periods=20).mean()
    e50 = closes.ewm(span=50, adjust=False, min_periods=50).mean()
    bullish = side == "BUY"
    if bullish:
        short = (closes.iloc[-1] > e10.iloc[-1] > e20.iloc[-1]
                 and e10.iloc[-1] > e10.iloc[-6])
        longer = (closes.iloc[-1] > e20.iloc[-1] > e50.iloc[-1]
                  and e50.iloc[-1] > e50.iloc[-11])
    else:
        short = (closes.iloc[-1] < e10.iloc[-1] < e20.iloc[-1]
                 and e10.iloc[-1] < e10.iloc[-6])
        longer = (closes.iloc[-1] < e20.iloc[-1] < e50.iloc[-1]
                  and e50.iloc[-1] < e50.iloc[-11])
    if longer:
        label = "LONGER-TERM SWING BIAS (2â8 weeks)"
        note = "daily 20/50 EMA trend aligned" + ("; short-term aligned too" if short else "; short-term not aligned")
    elif short:
        label = "SHORT-TERM SWING BIAS (2â10 trading days)"
        note = "daily 10/20 EMA trend aligned; longer trend unconfirmed"
    else:
        label = "NO SAME-DIRECTION SWING CONFIRMATION"
        note = "daily-chart filters not aligned with intraday direction"
    return SwingContext(label, as_of=last_day.isoformat(), explanation=note)


def format_entry_digest(chosen: list[Signal], now: datetime, candidates: int,
                        contexts: Optional[dict[str, SwingContext]] = None) -> str:
    """A single succinct Telegram notification with no more than 2 ranked setups by default."""
    timestamp = now.astimezone(ET)
    contexts = contexts or {}
    lines = [f"ð DAY-TRADE TOP {len(chosen)} | {timestamp:%b %d, %I:%M %p} ET",
             f"CONFIRMED (4/4) only Â· {candidates} qualifying Â· ranked by RVOL"]
    for i, s in enumerate(chosen, 1):
        icon = "ð¢" if s.side == "BUY" else "ð´"
        close_time = (s.bar_time + timedelta(minutes=5)).astimezone(ET)
        lines.extend([
            f"{i}. {icon} {s.symbol} {s.side} | ${s.price:.2f} reference",
            f"   5m close {close_time:%I:%M %p} ET Â· RVOL {s.rvol:.1f}x Â· RSI {s.rsi:.0f}",
            f"   Type: DAY TRADE Â· 5-minute setup Â· same-day exit",
            f"   Stop ref ${s.stop:.2f} Â· 2R target ${s.target:.2f}",
            f"   Daily-chart context: {contexts.get(s.symbol, SwingContext('NOT CHECKED')).classification}",
        ])
        ctx = contexts.get(s.symbol)
        if ctx is not None and ctx.as_of:
            lines.append(f"   Daily bar {ctx.as_of} Â· {ctx.explanation}")
    lines.append("Swing bias is trend context ONLY, not a swing entry/exit signal. Day-trade tracker ends same session.")
    lines.append("Research only; Alpaca reference prices may be incomplete/stale. Verify Merrill live quotes. No orders.")
    return "\n".join(lines)


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


@dataclass(frozen=True)
class TrackedSetup:
    """A hypothetical setup from a CONFIRMED alert, NOT an executed position."""
    symbol: str
    side: str
    entry_bar: datetime
    entry_reference: float
    stop: float
    target: float
    last_checked_bar: datetime
    feed: str


@dataclass(frozen=True)
class ExitNotice:
    symbol: str
    side: str
    time: datetime
    reference_price: Optional[float]
    reason: str
    entry_reference: float
    stop: float
    target: float
    feed: str


def load_setup(db: sqlite3.Connection, symbol: str) -> Optional[TrackedSetup]:
    row = db.execute("""SELECT symbol,side,entry_bar,entry_reference,stop,target,
                              last_checked_bar,feed FROM tracked_setups WHERE symbol=?""",
                     (symbol,)).fetchone()
    if not row:
        return None
    return TrackedSetup(row[0], row[1], datetime.fromisoformat(row[2]),
                        float(row[3]), float(row[4]), float(row[5]),
                        datetime.fromisoformat(row[6]), row[7])


def register_confirmed_setup(db: sqlite3.Connection, sig: Signal) -> bool:
    """Track ONLY a confirmed alert and only if the ticker has no open virtual setup."""
    if sig.level != "CONFIRMED":
        return False
    existing = load_setup(db, sig.symbol)
    if existing is not None:
        LOG.info("%s: a virtual setup is already being tracked; no new tracking opened", sig.symbol)
        return False
    db.execute("""INSERT INTO tracked_setups
        (symbol,side,entry_bar,entry_reference,stop,target,last_checked_bar,feed)
        VALUES (?,?,?,?,?,?,?,?)""", (sig.symbol, sig.side, sig.bar_time.isoformat(),
                                      sig.price, sig.stop, sig.target,
                                      sig.bar_time.isoformat(), sig.feed))
    db.commit()
    LOG.info("%s: tracking a hypothetical %s setup from reference %.2f (stop %.2f target %.2f)",
             sig.symbol, sig.side, sig.price, sig.stop, sig.target)
    return True


def delete_setup(db: sqlite3.Connection, symbol: str) -> None:
    db.execute("DELETE FROM tracked_setups WHERE symbol=?", (symbol,))
    db.commit()


def advance_setup(db: sqlite3.Connection, symbol: str, bar_time: datetime) -> None:
    db.execute("UPDATE tracked_setups SET last_checked_bar=? WHERE symbol=?",
               (bar_time.isoformat(), symbol))
    db.commit()


def _notice(tracked: TrackedSetup, start: datetime, reference: Optional[float],
            reason: str) -> ExitNotice:
    return ExitNotice(tracked.symbol, tracked.side, start, reference, reason,
                      tracked.entry_reference, tracked.stop, tracked.target, tracked.feed)


def evaluate_exit(tracked: TrackedSetup, bars: pd.DataFrame,
                  now: datetime, session_close: datetime) -> tuple[Optional[ExitNotice], Optional[datetime]]:
    """Evaluate only NEW, CLOSED 5-minute bars after a hypothetical entry.

    Stop/target collisions in the same bar resolve conservatively as STOP FIRST.
    Exit is a *signal*, never evidence that a broker order executed at that price.
    """
    current_session = now.astimezone(ET).date()
    if tracked.entry_bar.astimezone(ET).date() != current_session:
        return _notice(tracked, now, None, "TRACKING EXPIRED â previous session, fill/exit unknown"), None
    if bars.empty:
        return None, None
    recent = bars.loc[bars.index > tracked.last_checked_bar]
    recent = recent.loc[recent.index > tracked.entry_bar]
    if recent.empty:
        last = bars.index[-1].to_pydatetime()
        fresh = timedelta(0) <= now - (last + timedelta(minutes=5)) <= timedelta(minutes=10)
        if (now >= session_close - timedelta(minutes=12) and fresh and
                last > tracked.entry_bar and last.astimezone(ET).date() == current_session):
            return _notice(tracked, last, float(bars.iloc[-1]["close"]),
                           "END-OF-DAY time exit reference"), last
        return None, None
    x = with_indicators(bars)
    last_checked = None
    for ts, bar in x.loc[recent.index].iterrows():
        t = ts.to_pydatetime()
        if t.astimezone(ET).date() != current_session:
            continue
        last_checked = t
        is_long = tracked.side == "BUY"
        stop_hit = (float(bar["low"]) <= tracked.stop if is_long else
                    float(bar["high"]) >= tracked.stop)
        target_hit = (float(bar["high"]) >= tracked.target if is_long else
                      float(bar["low"]) <= tracked.target)
        if stop_hit:
            return _notice(tracked, t, tracked.stop, "STOP reference touched"), last_checked
        if target_hit:
            return _notice(tracked, t, tracked.target, "2R TARGET reference touched"), last_checked
        if all(pd.notna(bar[f]) for f in ("ema9", "ema21", "vwap", "rsi")):
            close = float(bar["close"])
            if is_long:
                reversal = ((float(bar["ema9"]) < float(bar["ema21"])) or (
                    close < float(bar["ema9"]) and close < float(bar["vwap"]) and float(bar["rsi"]) < 50))
            else:
                reversal = ((float(bar["ema9"]) > float(bar["ema21"])) or (
                    close > float(bar["ema9"]) and close > float(bar["vwap"]) and float(bar["rsi"]) > 50))
            if reversal:
                return _notice(tracked, t, close, "MOMENTUM / TREND invalidation"), last_checked
    if last_checked is not None and now >= session_close - timedelta(minutes=12):
        return _notice(tracked, last_checked, float(x.loc[last_checked, "close"]),
                       "END-OF-DAY time exit reference"), last_checked
    return None, last_checked


def format_exit_notice(n: ExitNotice) -> str:
    clock = n.time.astimezone(ET)
    when = (n.time + timedelta(minutes=5)).astimezone(ET) if n.reference_price is not None else clock
    side_label = "BULLISH/LONG" if n.side == "BUY" else "BEARISH/SHORT"
    value = (f"Reference exit: ${n.reference_price:,.2f} (NOT a fill quote)" if
             n.reference_price is not None else "Exit price unknown â check your brokerage")
    return (f"ðª DAY-TRADE VIRTUAL EXIT WATCH | {n.symbol} | {side_label}\n"
            f"Reason: {n.reason}\n"
            f"5-min reference time: {when:%Y-%m-%d %I:%M %p} ET\n"
            f"Original alert reference: ${n.entry_reference:,.2f}\n"
            f"{value}\n"
            f"Stop / target references: ${n.stop:,.2f} / ${n.target:,.2f}\n"
            f"Source: Alpaca {n.feed.upper()} (single-exchange if IEX).\n"
            "This tracks a hypothetical CONFIRMED signal, not your actual brokerage position. "
            "Price thresholds may be crossed without an executable fill. "
            "No orders are placed; verify actual holdings and current quotes.")


def check_and_alert_exit(http: requests.Session, cfg: Settings,
                         db: sqlite3.Connection, sym: str, bars: pd.DataFrame,
                         now: datetime, close: datetime) -> bool:
    tracked = load_setup(db, sym)
    if tracked is None:
        return False
    notice, last = evaluate_exit(tracked, bars, now, close)
    if notice:
        # A failed Telegram send must NOT clear the tracked setup.
        if send_telegram(http, cfg, format_exit_notice(notice)):
            delete_setup(db, sym)
            LOG.info("%s virtual setup exit alert: %s", sym, notice.reason)
            return True
    elif last is not None:
        advance_setup(db, sym, last)
    return False

def run_scan(http: requests.Session, cfg: Settings, db: sqlite3.Connection,
             now: Optional[datetime] = None) -> int:
    now = now or datetime.now(UTC)
    window = trading_window(now)
    if window is None:
        LOG.info("NYSE closed: holiday or weekend")
        return 0
    op, cl = window
    if not (op + timedelta(minutes=15) <= now <= cl - timedelta(minutes=5)):
        LOG.info("Outside active intraday scan window: %s", now.astimezone(ET))
        return 0

    exit_alerts = 0
    successes = 0
    evaluable = 0
    failures = 0
    candidates: list[Signal] = []
    symbols = active_symbols(http, cfg)
    virtual_count = db.execute("SELECT COUNT(*) FROM tracked_setups").fetchone()[0]
    LOG.info("Tracked hypothetical setups loaded from previous runs: %d", virtual_count)

    # IMPORTANT: ALWAYS check exits, even when today's entry-alert budget is exhausted.
    for sym in symbols:
        try:
            bars = fetch_bars(http, cfg, sym, now)
            successes += 1
            if bars.empty:
                LOG.warning("%s: no completed regular-session 5-minute bars", sym)
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
                if check_and_alert_exit(http, cfg, db, sym, bars, now, cl):
                    exit_alerts += 1
                    # Avoid a contradictory entry immediately after an exit on this scan.
                    continue

            # WATCH (3/4 criteria) signals are logged but never Telegram-notified.
            signals, diagnostic = assess_levels(sym, bars, now, cfg)
            LOG.info("%s criteria: %s", sym, diagnostic)
            if now >= cl - timedelta(minutes=30):
                continue  # Don't open new virtual day trades near the close.
            for sig in signals:
                if sig.level != "CONFIRMED":
                    continue
                if is_duplicate(db, sig, cfg):
                    LOG.info("%s %s %s suppressed: already alerted this session",
                             sym, sig.level, sig.side)
                    continue
                if load_setup(db, sig.symbol) is not None:
                    LOG.info("%s skipped: an earlier hypothetical setup is still tracked", sig.symbol)
                    continue
                candidates.append(sig)
        except requests.HTTPError as exc:
            failures += 1
            LOG.warning("%s request failed: HTTP %s", sym,
                        exc.response.status_code if exc.response is not None else "?")
        except Exception:
            failures += 1
            LOG.exception("%s scan failed", sym)

    LOG.info("DATA HEALTH: %d/%d symbols fetched, %d had fresh continuous bars, %d errors",
             successes, len(symbols), evaluable, failures)
    if successes == 0:
        raise RuntimeError("No market data could be fetched. Check Alpaca credentials, feed and API errors above.")
    if evaluable == 0:
        raise RuntimeError("No symbols had enough fresh continuous bars to evaluate. Check IEX data coverage / timestamps above.")

    used_today = entries_sent_today(db, now)
    remaining = max(0, cfg.max_entries_per_day - used_today)
    # RVOL prioritizes stronger participation; it is NOT a calibrated profitability score.
    ranked = sorted(candidates, key=lambda s: (
        -s.rvol, -abs(s.ema9 - s.ema21) / max(s.price, 0.01), s.symbol))
    chosen = ranked[:remaining]

    # Check daily swing context only for chosen entries (at most 2 by default).
    # Failed/insufficient daily data is labeled UNAVAILABLE, never guessed.
    swing_context: dict[str, SwingContext] = {}
    for sig in chosen:
        try:
            day_bars = fetch_daily_bars(http, cfg, sig.symbol, now)
            swing_context[sig.symbol] = assess_swing_context(day_bars, sig.side, now)
        except Exception as exc:
            LOG.warning("%s: daily trend context unavailable: %s", sig.symbol, exc)
            swing_context[sig.symbol] = SwingContext("UNAVAILABLE", explanation="Daily data request failed")

    if chosen:
        # Send only ONE concise message and record entries only after Telegram accepts it.
        if send_telegram(http, cfg, format_entry_digest(chosen, now, len(candidates), swing_context)):
            for sig in chosen:
                mark_sent(db, sig)
                register_confirmed_setup(db, sig)
            LOG.info("Quiet digest sent: %d candidates, %d entries in a single message", len(candidates), len(chosen))
    else:
        LOG.info("No new entries: %d confirmed candidates; daily budget %d/%d",
                 len(candidates), used_today, cfg.max_entries_per_day)
    LOG.info("Scan complete: %d new confirmed entries (%d used today), %d virtual exit alert(s)",
             len(chosen), used_today + len(chosen), exit_alerts)
    return len(chosen)


def main() -> None:
    p = argparse.ArgumentParser(description="Quiet 5-minute day-trading alert bot (never trades)")
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
        try:
            send_telegram(http, cfg, "â TEST ONLY â Quiet day-trading bot connected. No trade signal; no orders placed.")
        finally:
            http.close()
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
