import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from agent import Settings, assess, with_indicators, trading_window


class SignalTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings("x", "y", "", "", "iex", ("SPY",), 0, 60, 1.3, 45, Path(":memory:"))

    def candles(self, start="2026-11-18T14:30:00+00:00", n=50):
        idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
        close = [100 + i * .12 for i in range(n)]
        return pd.DataFrame({"open": close, "high": [c+.15 for c in close],
                             "low": [c-.15 for c in close], "close": close,
                             "volume": [100]*n}, index=idx)

    def test_indicator_warmup(self):
        df = with_indicators(self.candles())
        self.assertTrue(pd.notna(df.iloc[-1]["ema21"]))
        self.assertTrue(pd.notna(df.iloc[-1]["vwap"]))
        self.assertAlmostEqual(df.iloc[-1]["rvol"], 1.0)

    def test_no_signal_without_volume(self):
        df = self.candles()
        now = df.index[-1].to_pydatetime() + timedelta(minutes=5, seconds=25)
        self.assertIsNone(assess("SPY", df, now, self.cfg))

    def test_no_signal_stale(self):
        df = self.candles()
        now = df.index[-1].to_pydatetime() + timedelta(hours=1)
        self.assertIsNone(assess("SPY", df, now, self.cfg))

    def test_reject_missing_bars(self):
        df = self.candles()
        df = df.drop(df.index[-2])
        now = df.index[-1].to_pydatetime() + timedelta(minutes=5, seconds=25)
        self.assertIsNone(assess("SPY", df, now, self.cfg))

    def test_vwap_resets_for_new_session(self):
        yesterday = self.candles(start="2026-11-17T14:30:00+00:00", n=30)
        today = self.candles(start="2026-11-18T14:30:00+00:00", n=30)
        today[["open", "high", "low", "close"]] += 50
        out = with_indicators(pd.concat([yesterday, today]))
        self.assertGreater(out.iloc[-1]["vwap"], 145)

    def test_fresh_bullish_setup(self):
        idx = pd.date_range("2026-11-18T14:30:00Z", periods=60, freq="5min", tz="UTC")
        price, close = 100, []
        for i in range(len(idx)):
            price += .17 if i % 3 else -.10
            close.append(price)
        df = pd.DataFrame({"open": close, "high": [x + .1 for x in close],
                           "low": [x - .1 for x in close], "close": close,
                           "volume": [100] * 59 + [700]}, index=idx)
        now = idx[-1].to_pydatetime() + timedelta(minutes=5, seconds=30)
        signal = assess("SPY", df, now, self.cfg)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "BUY")
        self.assertLess(signal.stop, signal.price)

    def test_fresh_bearish_setup(self):
        idx = pd.date_range("2026-11-18T14:30:00Z", periods=60, freq="5min", tz="UTC")
        price, close = 100, []
        for i in range(len(idx)):
            price -= .17 if i % 3 else -.10
            close.append(price)
        df = pd.DataFrame({"open": close, "high": [x + .1 for x in close],
                           "low": [x - .1 for x in close], "close": close,
                           "volume": [100] * 59 + [700]}, index=idx)
        now = idx[-1].to_pydatetime() + timedelta(minutes=5, seconds=30)
        signal = assess("SPY", df, now, self.cfg)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "SELL/BEARISH")
        self.assertGreater(signal.stop, signal.price)

    def test_market_holiday_and_early_close(self):
        self.assertIsNone(trading_window(datetime(2026, 11, 26, 16, tzinfo=timezone.utc)))
        op, cl = trading_window(datetime(2026, 11, 27, 16, tzinfo=timezone.utc))
        self.assertEqual((cl - op).total_seconds() / 3600, 3.5)


if __name__ == "__main__":
    unittest.main()
