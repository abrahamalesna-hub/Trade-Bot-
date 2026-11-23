#!/usr/bin/env python3
"""Research-only swing stock + optional contract scanner. NO brokerage orders.

Entry ideas are built from completed daily stock bars and a fresh underlying trade.
Separate horizons: SHORT (2-10 trading days) and LONGER (2-8 weeks).
Telegram: at most one new recommendation per horizon per NYSE day, one digest.
State records hypothetical stock-reference setups, NOT positions/fills.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from datetime import date, datetime, timedelta, timezone, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import exchange_calendars as xcals
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UTC = timezone.utc
ET = ZoneInfo('America/New_York')
NYSE = xcals.get_calendar('XNYS')
DATA = 'https://data.alpaca.markets'
STATE_PATH = Path(os.getenv('SWING_OPTIONS_STATE', 'swing_options_state.json'))
WATCHLIST = ('SPY,QQQ,IWM,DIA,NVDA,TSLA,AMD,AMZN,PLTR,HOOD,AAPL,MSFT,META,GOOGL,'
             'AVGO,MU,SMCI,CRWD,COIN,MSTR,NFLX,SOFI,SHOP,UBER')
LOG = logging.getLogger('swing-options')


def timestamp(value):
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).astimezone(UTC)
    except (ValueError, TypeError, AttributeError):
        return None


def check_positive(value):
    try:
        f = float(value)
        return f if math.isfinite(f) and f > 0 else None
    except (ValueError, TypeError):
        return None


def new_session():
    key = os.getenv('ALPACA_API_KEY', '').strip()
    secret = os.getenv('ALPACA_SECRET_KEY', '').strip()
    if not key or not secret:
        raise RuntimeError('Missing ALPACA_API_KEY/ALPACA_SECRET_KEY')
    s = requests.Session()
    s.headers.update({'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret})
    retry = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=['GET'], respect_retry_after_header=True)
    s.mount('https://', HTTPAdapter(max_retries=retry))
    return s


def api_get(s, path, params):
    r = s.get(DATA + path, params=params, timeout=22)
    r.raise_for_status()
    return r.json()


def telegram(message, preview=False):
    if preview:
        print(message, flush=True)
        return
    token, chat = os.getenv('TELEGRAM_BOT_TOKEN'), os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat:
        raise RuntimeError('Missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID')
    r = requests.post(f'https://api.telegram.org/bot{token}/sendMessage',
                      json={'chat_id': chat, 'text': message, 'disable_web_page_preview': True}, timeout=20)
    r.raise_for_status()
    if r.json().get('ok') is not True:
        raise RuntimeError('Telegram did not report successful delivery')


def load_state():
    if not STATE_PATH.exists():
        return {'sent': {}, 'tracked': {}}
    try:
        value = json.loads(STATE_PATH.read_text())
        if not isinstance(value, dict) or not all(isinstance(value.get(k), dict) for k in ('sent', 'tracked')):
            raise ValueError('Malformed state')
        return value
    except (ValueError, OSError):
        raise RuntimeError(f'{STATE_PATH} is corrupt. Refusing to reset deduplication silently.')


def save_state(value):
    temp = STATE_PATH.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True))
    temp.replace(STATE_PATH)


def completed_daily(s, sym, now, feed):
    # Request ample history; ignore today's incomplete daily bar and never use future data.
    result = api_get(s, f'/v2/stocks/{sym}/bars', {
        'timeframe': '1Day', 'start': (now.date() - timedelta(days=300)).isoformat(),
        'end': (now.date() + timedelta(days=1)).isoformat(), 'limit': 1000,
        'feed': feed, 'adjustment': 'split', 'sort': 'asc'})
    entries = []
    for raw in result.get('bars', []):
        t = timestamp(raw.get('t'))
        if not t or t.astimezone(ET).date() >= now.astimezone(ET).date():
            continue
        try:
            a = [float(raw[x]) for x in ('o', 'h', 'l', 'c', 'v')]
        except (TypeError, ValueError, KeyError):
            continue
        if any(not math.isfinite(x) for x in a) or min(a[:4]) <= 0 or a[4] < 0:
            continue
        entries.append((t.astimezone(ET).date(), *a))
    if len(entries) < 125:
        return pd.DataFrame()
    df = pd.DataFrame(entries, columns=['day', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates('day', keep='last').sort_values('day').reset_index(drop=True)
    # Require recent completed session; a long weekend or holiday is permissible.
    prior = NYSE.previous_session(now.astimezone(ET).date())
    if df.iloc[-1]['day'] != prior.date():
        LOG.warning('%s latest completed daily bar %s differs from previous session %s', sym, df.iloc[-1]['day'], prior)
        return pd.DataFrame()
    return df


def latest_stock_trade(s, sym, now, feed):
    raw = api_get(s, f'/v2/stocks/{sym}/trades/latest', {'feed': feed}).get('trade', {})
    price = check_positive(raw.get('p'))
    t = timestamp(raw.get('t'))
    if not price or not t:
        return None
    age = (now - t).total_seconds()/60
    if not (-1 <= age <= 15) or t.astimezone(ET).date() != now.astimezone(ET).date():
        return None
    return {'price': price, 'time': t.isoformat(), 'age': age}


def technical_ideas(sym, df):
    """Strict daily breakout or pullback trigger, with trend filters; not forecasts."""
    if len(df) < 125:
        return []
    x = df.copy()
    for n in (10, 20, 50, 100):
        x[f'e{n}'] = x['close'].ewm(span=n, adjust=False, min_periods=n).mean()
    d = x['close'].diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    down = (-d).clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    x['rsi'] = 100 - 100/(1 + up / down.replace(0, float('nan')))
    x.loc[(down == 0) & (up > 0), 'rsi'] = 100.0
    tr = pd.concat([x.high-x.low, (x.high-x.close.shift(1)).abs(),
                    (x.low-x.close.shift(1)).abs()], axis=1).max(axis=1)
    x['atr'] = tr.rolling(14, min_periods=14).mean()
    a, b = x.iloc[-1], x.iloc[-2]
    out = []
    if not all(math.isfinite(float(a[z])) for z in ('e10', 'e20', 'e50', 'e100', 'rsi', 'atr')):
        return out
    for horizon in ('SHORT', 'LONGER'):
        for side in ('BULLISH', 'BEARISH'):
            bull = side == 'BULLISH'
            trend = (a.close > a.e10 > a.e20 and a.e10 > x.iloc[-4].e10) if horizon == 'SHORT' else (a.close > a.e20 > a.e50 > a.e100 and a.e50 > x.iloc[-11].e50)
            if not bull:
                trend = (a.close < a.e10 < a.e20 and a.e10 < x.iloc[-4].e10) if horizon == 'SHORT' else (a.close < a.e20 < a.e50 < a.e100 and a.e50 < x.iloc[-11].e50)
            in_rsi = (53 <= a.rsi <= 74 if bull else 26 <= a.rsi <= 47) if horizon == 'SHORT' else (50 <= a.rsi <= 72 if bull else 28 <= a.rsi <= 50)
            if not (trend and in_rsi):
                continue
            lookback = 10 if horizon == 'SHORT' else 25
            prior_high = float(x['high'].iloc[-lookback-1:-1].max())
            prior_low = float(x['low'].iloc[-lookback-1:-1].min())
            breakout = (a.close > prior_high if bull else a.close < prior_low)
            ema = a.e10 if horizon == 'SHORT' else a.e20
            # A bounce: prior session traded into the average, then closed back
            # above/below it, with a favorable close-to-close move.
            bounce = (ema*0.965 <= a.low <= ema*1.007 and a.close > ema and a.close > b.close) if bull else (ema*0.993 <= a.high <= ema*1.035 and a.close < ema and a.close < b.close)
            if not (breakout or bounce):
                continue
            pattern = 'DAILY BREAKOUT' if breakout else 'EMA PULLBACK/BOUNCE'
            # Recent closed-bar stop/target levels are recalculated from CURRENT stock price below.
            out.append({'symbol': sym, 'horizon': horizon, 'side': side,
                        'pattern': pattern, 'closed_date': str(a.day),
                        'signal_close': float(a.close), 'atr': float(a.atr),
                        'rsi': float(a.rsi), 'ema10': float(a.e10),
                        'ema20': float(a.e20), 'ema50': float(a.e50),
                        'score': (2 if breakout else 1) + abs(float(a.rsi)-50)/25})
    return out


def qualify_current(idea, trade):
    # Do not send yesterday's breakout as actionable if it has moved too far.
    if not trade:
        return None
    idea = dict(idea)
    price = trade['price']
    move = (price / idea['signal_close'] - 1)
    if abs(move) > 0.025:
        return None
    # Preserve original bullish/bearish directional invalidation.
    if idea['side'] == 'BULLISH' and price < idea['ema20']:
        return None
    if idea['side'] == 'BEARISH' and price > idea['ema20']:
        return None
    mult = 1.6 if idea['horizon'] == 'SHORT' else 2.1
    risk = mult * idea['atr']
    stop = price - risk if idea['side'] == 'BULLISH' else price + risk
    target = price + 2*risk if idea['side'] == 'BULLISH' else price - 2*risk
    # Do not promote extreme-risk ATR setups to concise top-pick alerts.
    if stop <= 0 or target <= 0 or (100 * risk / price) > (9 if idea['horizon'] == 'SHORT' else 15):
        return None
    idea.update(price=price, trade_time=trade['time'], stop=stop, target=target,
                risk_pct=100*risk/price, option=None)
    return idea


def contract_metadata(contract, sym):
    match = re.fullmatch(re.escape(sym) + r'(\d{6})([CP])(\d{8})', contract)
    if not match:
        return None
    ymd, cp, strike = match.groups()
    try:
        expiry = datetime.strptime(ymd, '%y%m%d').date()
    except ValueError:
        return None
    return expiry, 'CALL' if cp == 'C' else 'PUT', int(strike)/1000


def option_quote(snap, now, feed):
    q = snap.get('latestQuote') or {}
    bid, ask = check_positive(q.get('bp')), check_positive(q.get('ap'))
    t = timestamp(q.get('t'))
    if not bid or not ask or ask <= bid or not t:
        return None
    age = (now-t).total_seconds()/60
    max_age = 8 if feed == 'opra' else 35
    if not (-1 <= age <= max_age):
        return None
    mid = (ask+bid)/2
    spread = (ask-bid)/mid
    if spread > float(os.getenv('MAX_OPTION_SPREAD', '0.12')):
        return None
    greeks = snap.get('greeks') or {}
    try:
        delta, iv = float(greeks.get('delta')), float(snap.get('impliedVolatility'))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(z) for z in (delta, iv)) or not (0.10 <= iv <= 1.8):
        return None
    if not (0.40 <= abs(delta) <= 0.70):
        return None
    if not (0.50 <= ask <= float(os.getenv('MAX_OPTION_ASK', '35'))):
        return None
    return dict(bid=bid, ask=ask, spread=spread, delta=delta, iv=iv,
                quote_time=t.isoformat(), quote_age_min=age)


def option_for(s, idea, now, feed):
    d = now.astimezone(ET).date()
    first, last, ideal = (21, 45, 30) if idea['horizon']=='SHORT' else (60, 120, 85)
    typ = 'call' if idea['side']=='BULLISH' else 'put'
    params = {'feed': feed, 'type': typ, 'root_symbol': idea['symbol'],
              'expiration_date_gte': str(d + timedelta(days=first)),
              'expiration_date_lte': str(d + timedelta(days=last)),
              'strike_price_gte': round(idea['price']*0.90, 2),
              'strike_price_lte': round(idea['price']*1.10, 2), 'limit': 1000}
    best = None
    token = None
    for _ in range(5):
        if token:
            params['page_token'] = token
        resp = api_get(s, f"/v1beta1/options/snapshots/{idea['symbol']}", params)
        for contract, snap in (resp.get('snapshots') or {}).items():
            meta = contract_metadata(contract, idea['symbol'])
            if not meta:
                continue
            expiry, kind, strike = meta
            if kind != typ.upper():
                continue
            dte = (expiry-d).days
            if not first <= dte <= last or not (idea['price']*0.90 <= strike <= idea['price']*1.10):
                continue
            q = option_quote(snap, now, feed)
            if not q or ((kind=='CALL' and q['delta'] <= 0) or (kind=='PUT' and q['delta'] >= 0)):
                continue
            rank = (abs(abs(q['delta'])-0.55)*3 + abs(dte-ideal)/50 + q['spread']*4)
            o = {'contract': contract, 'kind': kind, 'strike': strike,
                 'expiration': str(expiry), 'dte': dte, 'rank': rank, **q}
            if best is None or rank < best['rank']:
                best = o
        token = resp.get('next_page_token')
        if not token:
            break
    if token:
        LOG.warning('%s options chain pagination truncated; no exhaustive guarantee', idea['symbol'])
    return best


def key(idea):
    return f"{idea['horizon']}|{idea['symbol']}|{idea['side']}"


def already_notified(state, idea, day):
    sent = state['sent']
    horizon = idea['horizon']
    # Exactly one NEW idea per horizon per day (not counting exits).
    if any(s.get('day') == str(day) and s.get('horizon') == horizon for s in sent.values()):
        return True
    recorded = sent.get(key(idea))
    if recorded:
        ago = (day - date.fromisoformat(recorded['day'])).days
        if ago < (7 if horizon == 'SHORT' else 21):
            return True
    # Do not track conflicting bullish/bearish hypothetical setups for one ticker/horizon.
    return any(t['symbol'] == idea['symbol'] and t['horizon'] == horizon
               for t in state['tracked'].values())


def mark_entry(state, idea, day):
    k = key(idea)
    state['sent'][k] = {'day': str(day), 'horizon': idea['horizon']}
    state['tracked'][k] = {
        'symbol': idea['symbol'], 'side': idea['side'], 'horizon': idea['horizon'],
        'day': str(day), 'entry': idea['price'], 'stop': idea['stop'],
        'target': idea['target'], 'last_daily': idea['closed_date']}


def check_exit(state, idea_key, tracked, history, live, day):
    """Return an EXIT WATCH only if reference thresholds or time rules trigger."""
    if history.empty:
        return None
    entry_date = date.fromisoformat(tracked['day'])
    new = history.loc[history.day > entry_date]
    bullish = tracked['side'] == 'BULLISH'
    for _, b in new.iterrows():
        low, high = float(b.low), float(b.high)
        stop_hit = low <= tracked['stop'] if bullish else high >= tracked['stop']
        target_hit = high >= tracked['target'] if bullish else low <= tracked['target']
        # If both stop and target were reached during one bar, ordering unknown.
        if stop_hit and target_hit:
            return 'BOTH stop and target touched same session; order unknown, check actual fill'
        if stop_hit:
            return 'Stock STOP reference touched on completed daily bar'
        if target_hit:
            return 'Stock 2R TARGET reference touched on completed daily bar'
    if live:
        price = live['price']
        if (price <= tracked['stop'] if bullish else price >= tracked['stop']):
            return 'Current stock trade crossed STOP reference'
        if (price >= tracked['target'] if bullish else price <= tracked['target']):
            return 'Current stock trade crossed 2R TARGET reference'
    # A completed daily close through the relevant EMA also invalidates the trend.
    if not new.empty and len(history) >= 30:
        closes = history['close'].astype(float)
        n = 10 if tracked['horizon'] == 'SHORT' else 20
        trend_ema = closes.ewm(span=n, adjust=False, min_periods=n).mean()
        last_close = float(closes.iloc[-1])
        threshold = float(trend_ema.iloc[-1])
        if (last_close < threshold if bullish else last_close > threshold):
            return f'Completed daily close invalidated {n}-day EMA trend'
    # Age in calendar days; conservative reminder, not automatic close.
    if (day-entry_date).days >= (16 if tracked['horizon']=='SHORT' else 60):
        return 'Time horizon reached; REVIEW / EXIT WATCH'
    return None


def format_exit(tracked, reason, live, now):
    value = f"${live['price']:.2f}" if live else 'unavailable'
    return (f"🚨 SWING EXIT/REVIEW | {tracked['horizon']} | {tracked['symbol']} {tracked['side']}\n"
            f"Reason: {reason}\nEntry reference ${tracked['entry']:.2f}; stop ${tracked['stop']:.2f}; "
            f"target ${tracked['target']:.2f}\nStock reference now: {value}\n"
            f"Checked: {now.astimezone(ET):%b %d %I:%M %p} ET\n"
            "Virtual alert only; NOT your Merrill position or a verified fill. Check your order/quotes.")


def format_entry(idea, feed):
    horizon = 'SHORT-TERM SWING (2–10 trading days)' if idea['horizon']=='SHORT' else 'LONGER-TERM SWING (2–8 weeks)'
    action = 'BUY / BULLISH' if idea['side']=='BULLISH' else 'BEARISH / consider PUT'
    lines = [f"{horizon} | {idea['symbol']} | {action}",
             f"Setup: {idea['pattern']} on {idea['closed_date']} completed daily bar",
             f"Stock reference ${idea['price']:.2f} | RSI {idea['rsi']:.0f} | 20EMA ${idea['ema20']:.2f}",
             f"Stock stop ${idea['stop']:.2f} | 2R target ${idea['target']:.2f} | indicative risk {idea['risk_pct']:.1f}%",
             f"Stock trade time {timestamp(idea['trade_time']).astimezone(ET):%b %d %I:%M %p} ET"]
    o = idea.get('option')
    if o:
        provenance = 'OPRA market quote' if feed == 'opra' else 'INDICATIVE (not a firm/executable quote)'
        lines.extend([
            f"OPTION CANDIDATE: BUY TO OPEN {o['kind']} ${o['strike']:g} exp {o['expiration']} ({o['dte']} DTE)",
            f"Exact OCC symbol: {o['contract']}",
            f"Bid ${o['bid']:.2f} / ask ${o['ask']:.2f} | est. ask cost ${o['ask']*100:.0f}/standard contract (verify multiplier)",
            f"Delta {o['delta']:.2f} | IV {o['iv']:.0%} | spread {o['spread']:.1%}",
            f"Quote {timestamp(o['quote_time']).astimezone(ET):%b %d %I:%M %p} ET | {provenance}",
            'OPTION EXIT: Monitor the underlying stock invalidation and premium; stock stop/target are NOT option stop prices.'
        ])
    else:
        lines.append('OPTION: No contract passed live data/liquidity filters; do not infer strike or premium.')
    lines.append('Verify earnings, live Merrill quotes, liquidity and firm preclearance. Research idea only; no orders.')
    return '\n'.join(lines)


def run_scan(s, now, preview=False):
    et = now.astimezone(ET)
    if not NYSE.is_session(et.date()) or not dtime(10, 0) <= et.time() <= dtime(15, 45):
        LOG.info('Outside stock market screening window; no scan')
        return
    feed = os.getenv('DATA_FEED', 'iex').lower()
    options_feed = os.getenv('OPTIONS_FEED', 'indicative').lower()
    if feed not in ('iex', 'sip') or options_feed not in ('indicative', 'opra'):
        raise RuntimeError('Invalid DATA_FEED or OPTIONS_FEED')
    state = load_state()
    symbols = list(dict.fromkeys(t.strip().upper() for t in os.getenv('WATCHLIST', WATCHLIST).split(',')
                                 if re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,9}', t.strip().upper())))
    if not symbols:
        raise RuntimeError('WATCHLIST is empty')
    candidates, history, live_by_symbol = [], {}, {}
    success = 0
    for sym in symbols:
        try:
            bars = completed_daily(s, sym, now, feed)
            if bars.empty:
                continue
            success += 1
            history[sym] = bars
            # Fetch fresh stock print only for potential entries or exit-watch setups.
            raw_ideas = technical_ideas(sym, bars)
            tracked_exists = any(v['symbol']==sym for v in state['tracked'].values())
            if raw_ideas or tracked_exists:
                live_by_symbol[sym] = latest_stock_trade(s, sym, now, feed)
            for idea in raw_ideas:
                qualified = qualify_current(idea, live_by_symbol.get(sym))
                if qualified:
                    candidates.append(qualified)
        except requests.HTTPError as e:
            LOG.warning('%s HTTP %s while scanning', sym, e.response.status_code if e.response else '?')
        except Exception:
            LOG.exception('%s failed', sym)
    if success == 0:
        raise RuntimeError('No valid fresh completed stock histories; check credentials, feed and logs')
    # Critical exit alerts first; do not deduct them from the entry cap.
    for k, tracked in list(state['tracked'].items()):
        bars = history.get(tracked['symbol'])
        if bars is None or bars.empty:
            LOG.warning('%s exit check skipped because no valid stock history', k)
            continue
        reason = check_exit(state, k, tracked, bars, live_by_symbol.get(tracked['symbol']), et.date())
        if reason:
            telegram(format_exit(tracked, reason, live_by_symbol.get(tracked['symbol']), now), preview=preview)
            if not preview:
                del state['tracked'][k]
                save_state(state)
    # Take top choice in each horizon; cap is per NYSE session, stored persistently.
    candidates.sort(key=lambda x: (x['horizon'] != 'SHORT', -x['score'], x['symbol']))
    picks = []
    for horizon in ('SHORT', 'LONGER'):
        eligible = [x for x in candidates if x['horizon']==horizon
                    and not already_notified(state, x, et.date())]
        if not eligible:
            continue
        chosen = eligible[0]
        try:
            chosen['option'] = option_for(s, chosen, now, options_feed)
        except requests.HTTPError as exc:
            LOG.warning('%s options data unavailable (HTTP %s); stock idea remains available',
                        chosen['symbol'], exc.response.status_code if exc.response else '?')
        except Exception as exc:
            LOG.warning('%s options unavailable: %s', chosen['symbol'], exc)
        picks.append(chosen)
    if not picks:
        LOG.info('No new qualifying short/longer swing recommendations today')
    else:
        body = (f"📌 SWING TRADE IDEAS — {et:%b %d, %Y %I:%M %p} ET\n"
                f"{len(picks)} new candidate(s) | Completed daily bars + fresh stock prices\n\n"
                + '\n\n────────────\n\n'.join(format_entry(x, options_feed) for x in picks))
        telegram(body, preview=preview)
        if not preview:
            for x in picks:
                mark_entry(state, x, et.date())
            save_state(state)
    LOG.info('%d successful data fetches; %d potential setups; %d new recommendations',
             success, len(candidates), len(picks))


def main():
    p = argparse.ArgumentParser(description='Manual stock and options swing recommendations; no execution')
    p.add_argument('--once', action='store_true')
    p.add_argument('--preview', action='store_true', help='Print instead of sending; does not update tracking state')
    p.add_argument('--test-alert', action='store_true', help='Telegram connection test')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.test_alert:
        telegram('✅ SWING + OPTIONS BOT TEST ONLY — Telegram connected. No trade or order.')
        return
    with new_session() as s:
        run_scan(s, datetime.now(UTC), preview=args.preview)


if __name__ == '__main__':
    main()
