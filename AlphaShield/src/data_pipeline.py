"""Market data ingestion, memory buffering, and technical indicator generation."""

from __future__ import annotations

import asyncio
import enum
import math
import threading
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd


class ConnectionState(enum.Enum):
    """Explicit lifecycle states for the Alpaca stream consumer."""

    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    FATAL = "FATAL"


class MemoryBufferManager:
    """Thread-safe ring buffer enforcing a strict market bar schema."""

    REQUIRED_SCHEMA = {
        "symbol": str,
        "timestamp": pd.Timestamp,
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": int,
    }

    def __init__(self, max_bars: int = 200) -> None:
        if max_bars <= 0:
            raise ValueError("max_bars must be positive")
        self.max_bars = int(max_bars)
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=self.max_bars)
        self._lock = threading.Lock()
        self.tracking_index = 0

    def _validate_row(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(row, Mapping):
            raise TypeError("row must be a mapping")
        missing = set(self.REQUIRED_SCHEMA) - set(row)
        extra = set(row) - set(self.REQUIRED_SCHEMA)
        if missing or extra:
            raise ValueError(f"row schema mismatch; missing={missing}, extra={extra}")
        validated: Dict[str, Any] = {}
        for key, expected_type in self.REQUIRED_SCHEMA.items():
            value = row[key]
            if key == "volume":
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif expected_type is float:
                valid = isinstance(value, float) and math.isfinite(value)
            else:
                valid = isinstance(value, expected_type)
            if not valid:
                raise TypeError(f"{key} must be {expected_type.__name__}")
            validated[key] = value
        if validated["high"] < max(validated["open"], validated["close"], validated["low"]):
            raise ValueError("high must be greater than or equal to open, close, and low")
        if validated["low"] > min(validated["open"], validated["close"], validated["high"]):
            raise ValueError("low must be less than or equal to open, close, and high")
        return validated

    def append_bar(self, row: Mapping[str, Any]) -> None:
        validated = self._validate_row(row)
        with self._lock:
            self._buffer.append(validated)
            self.tracking_index += 1

    def extend(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            self.append_bar(row)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._buffer)

    def to_dataframe(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.snapshot())
        if frame.empty:
            return pd.DataFrame(columns=list(self.REQUIRED_SCHEMA))
        return frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


class AlpacaWebSocketConsumer:
    """Resilient asynchronous Alpaca IEX bar stream consumer."""

    BASE_DELAY = 2.0
    MULTIPLIER = 2.0
    MAX_DELAY = 300.0

    def __init__(self, api_key: str, secret_key: str, symbols: list, buffer_ref: MemoryBufferManager):
        if not api_key or not secret_key:
            raise ValueError("api_key and secret_key are required")
        if not symbols or not all(isinstance(symbol, str) for symbol in symbols):
            raise ValueError("symbols must be a non-empty list of strings")
        if not isinstance(buffer_ref, MemoryBufferManager):
            raise TypeError("buffer_ref must be a MemoryBufferManager")
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbols = symbols
        self.buffer_ref = buffer_ref
        self.state = ConnectionState.DISCONNECTED
        self._stream: Any = None
        self._attempt = 0
        self._stop_requested = False

    def _transition(self, state: ConnectionState) -> None:
        if self.state != state:
            print(f"AlphaShield stream state: {self.state.value} -> {state.value}")
            self.state = state

    def _retry_delay(self) -> float:
        return min(self.BASE_DELAY * (self.MULTIPLIER ** self._attempt), self.MAX_DELAY)

    async def connect_stream(self) -> None:
        while not self._stop_requested:
            try:
                self._transition(ConnectionState.CONNECTING if self._attempt == 0 else ConnectionState.RECONNECTING)
                try:
                    from alpaca.data.live import StockDataStream
                except Exception as exc:
                    raise RuntimeError("alpaca-py live stream dependency is unavailable") from exc
                self._stream = StockDataStream(self.api_key, self.secret_key, feed="iex")
                self._stream.subscribe_bars(self.on_bars_received, *self.symbols)
                self._transition(ConnectionState.CONNECTED)
                self._attempt = 0
                return
            except Exception as exc:
                print(f"AlphaShield stream connection failure: {exc}")
                self._transition(ConnectionState.DISCONNECTED)
                delay = self._retry_delay()
                self._attempt += 1
                await asyncio.sleep(delay)
        self._transition(ConnectionState.FATAL)

    async def listen_loop(self) -> None:
        if self._stream is None or self.state != ConnectionState.CONNECTED:
            await self.connect_stream()
        while not self._stop_requested and self._stream is not None:
            try:
                runner = getattr(self._stream, "run", None)
                if runner is None:
                    raise RuntimeError("Alpaca stream object does not expose run()")
                result = runner()
                if asyncio.iscoroutine(result):
                    await result
                return
            except Exception as exc:
                print(f"AlphaShield stream listen failure: {exc}")
                self._transition(ConnectionState.DISCONNECTED)
                delay = self._retry_delay()
                self._attempt += 1
                await asyncio.sleep(delay)
                await self.connect_stream()
        if self._stop_requested:
            self._transition(ConnectionState.DISCONNECTED)

    async def on_bars_received(self, bar) -> None:
        raw = {
            "symbol": str(getattr(bar, "symbol")),
            "timestamp": pd.Timestamp(getattr(bar, "timestamp")),
            "open": float(getattr(bar, "open")),
            "high": float(getattr(bar, "high")),
            "low": float(getattr(bar, "low")),
            "close": float(getattr(bar, "close")),
            "volume": int(getattr(bar, "volume")),
        }
        self.buffer_ref.append_bar(raw)

    def stop(self) -> None:
        self._stop_requested = True


class TechnicalIndicatorEngine:
    """Vectorized technical indicator calculator with anti-lookahead shifting."""

    REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}

    def __init__(self, ema_fast: int = 20, ema_slow: int = 50, rsi_period: int = 14, atr_period: int = 14) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.atr_period = atr_period

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLUMNS - set(raw.columns)
        if missing:
            raise ValueError(f"raw dataframe missing columns: {missing}")
        frame = raw.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp").set_index("timestamp")
        numeric_columns = ["open", "high", "low", "close", "volume"]
        frame[numeric_columns] = frame[numeric_columns].astype(float)

        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        previous_close = close.shift(1)

        indicators = pd.DataFrame(index=frame.index)
        indicators["open"] = frame["open"]
        indicators["high"] = high
        indicators["low"] = low
        indicators["close"] = close
        indicators["volume"] = frame["volume"]
        ema_20 = close.ewm(span=self.ema_fast, adjust=False, min_periods=1).mean()
        ema_50 = close.ewm(span=self.ema_slow, adjust=False, min_periods=1).mean()
        indicators["ema_20"] = ema_20
        indicators["ema_50"] = ema_50

        delta = close.diff()
        upward = delta.clip(lower=0.0)
        downward = (-delta).clip(lower=0.0)
        average_gain = upward.ewm(alpha=1 / self.rsi_period, adjust=False, min_periods=1).mean()
        average_loss = downward.ewm(alpha=1 / self.rsi_period, adjust=False, min_periods=1).mean()
        rs = average_gain.div(average_loss.replace(0.0, np.nan))
        neutral_rsi = pd.Series(np.where(average_gain > 0.0, 100.0, 50.0), index=frame.index)
        indicators["rsi_14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(neutral_rsi)

        true_range = pd.concat(
            [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr_14 = true_range.ewm(alpha=1 / self.atr_period, adjust=False, min_periods=1).mean()
        indicators["atr_14"] = atr_14
        indicators["return_1"] = close.pct_change()

        # --- Normalized / relative features (computed before the 1-period shift) ---
        safe_close = close.replace(0.0, np.nan)
        indicators["return_3"] = close.pct_change(3)
        indicators["return_5"] = close.pct_change(5)
        indicators["return_10"] = close.pct_change(10)
        indicators["return_20"] = close.pct_change(20)
        indicators["rolling_vol_20"] = close.pct_change().rolling(20, min_periods=2).std()
        indicators["close_to_ema20"] = close / ema_20.replace(0.0, np.nan) - 1.0
        indicators["close_to_ema50"] = close / ema_50.replace(0.0, np.nan) - 1.0
        indicators["atr_pct"] = atr_14 / safe_close
        vol_mean = frame["volume"].rolling(20, min_periods=2).mean()
        vol_std = frame["volume"].rolling(20, min_periods=2).std()
        indicators["volume_zscore"] = (frame["volume"] - vol_mean) / vol_std.replace(0.0, np.nan)

        unshifted_market_columns = ["open", "high", "low", "close", "volume"]
        feature_columns = indicators.columns.difference(unshifted_market_columns)
        indicators.loc[:, feature_columns] = indicators.loc[:, feature_columns].shift(1)
        indicators = indicators.ffill().bfill()
        indicators["rsi_14"] = indicators["rsi_14"].clip(0.0, 100.0)
        indicators = indicators.replace([np.inf, -np.inf], 0.0).ffill().bfill().fillna(0.0)
        return indicators
