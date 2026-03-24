"""Timeframe resampler — converts 1-minute bars to higher timeframes."""
import pandas as pd
import structlog
from typing import List
from ghost.modules.m01_data.models import Bar

logger = structlog.get_logger()

TIMEFRAME_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


class Resampler:
    """Resamples 1-minute OHLCV data to any higher timeframe."""

    @staticmethod
    def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample a 1m DataFrame to a higher timeframe.

        Args:
            df: DataFrame with ts_event, open, high, low, close, volume columns
            timeframe: Target timeframe ("5m", "15m", "1h", "4h", "1d")
        """
        if timeframe == "1m":
            return df.copy()

        freq = TIMEFRAME_MAP.get(timeframe)
        if freq is None:
            raise ValueError(f"Unknown timeframe: {timeframe}. Use: {list(TIMEFRAME_MAP.keys())}")

        df = df.copy()
        df["ts_event"] = pd.to_datetime(df["ts_event"])
        df = df.set_index("ts_event")

        resampled = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        resampled = resampled.reset_index()

        # Carry forward instrument/symbol if present
        for col in ["instrument", "symbol"]:
            if col in df.columns:
                resampled[col] = df[col].iloc[0] if not df.empty else ""

        logger.debug(
            "resampled",
            timeframe=timeframe,
            input_bars=len(df),
            output_bars=len(resampled),
        )
        return resampled

    @staticmethod
    def resample_to_bars(df: pd.DataFrame, timeframe: str) -> List[Bar]:
        """Resample and return as list of Bar objects."""
        resampled = Resampler.resample(df, timeframe)
        return [
            Bar(
                timestamp=row["ts_event"].timestamp(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            )
            for _, row in resampled.iterrows()
        ]
