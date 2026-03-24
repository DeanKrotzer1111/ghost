"""Databento data loader — reads zstd-compressed CSV OHLCV files into Bar objects."""
import os
import io
import glob
import zstandard as zstd
import pandas as pd
import structlog
from typing import Dict, List, Optional
from ghost.modules.m01_data.models import Bar

logger = structlog.get_logger()

# Map Databento contract roots to Ghost instrument names
CONTRACT_MAP = {
    "MNQ": "MNQ", "MES": "MES", "MGC": "MGC", "SIL": "SIL",
    "NQ": "NQ", "ES": "ES", "GC": "GC", "SI": "SI",
    "CL": "CL", "MCL": "MCL", "NG": "NG", "YM": "YM",
    "MYM": "MYM", "RTY": "RTY", "M2K": "M2K",
    "6E": "6E", "6B": "6B", "6J": "6J", "6A": "6A", "6C": "6C",
    "ZB": "ZB", "ZN": "ZN", "ZF": "ZF", "ZT": "ZT",
    "ZC": "ZC", "ZS": "ZS", "ZW": "ZW",
}


def _extract_root(symbol: str) -> str:
    """Extract instrument root from Databento symbol like MNQH6, MESH6, etc."""
    # Strip trailing contract month+year (last 2 chars like H6, M6, Z5)
    for length in range(4, 1, -1):
        root = symbol[:length]
        if root in CONTRACT_MAP:
            return CONTRACT_MAP[root]
    return symbol[:3]


class DatabentoLoader:
    """Loads Databento OHLCV CSV files (zstd-compressed) from a download directory."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._cache: Dict[str, pd.DataFrame] = {}

    def list_instruments(self) -> List[str]:
        """List unique instruments available in the data directory."""
        files = glob.glob(os.path.join(self.data_dir, "*.csv.zst"))
        instruments = set()
        for f in files:
            basename = os.path.basename(f)
            # Filename format: prefix.ohlcv-1m.SYMBOL.csv.zst — symbol is parts[-3]
            parts = basename.split(".")
            symbol_part = parts[-3] if len(parts) >= 4 else ""
            if "-" not in symbol_part and symbol_part:
                instruments.add(_extract_root(symbol_part))
        return sorted(instruments)

    def load_instrument(self, instrument: str, use_front_month: bool = True) -> pd.DataFrame:
        """Load all 1-minute bars for an instrument, concatenating across contract months.

        Uses front-month (highest volume) contracts by default.
        """
        if instrument in self._cache:
            return self._cache[instrument]

        files = glob.glob(os.path.join(self.data_dir, "*.csv.zst"))
        matching = []
        for f in files:
            basename = os.path.basename(f)
            parts = basename.split(".")
            if len(parts) < 4:
                continue
            symbol_part = parts[-3]
            # Skip spreads (contain dashes in the symbol portion)
            if "-" in symbol_part:
                continue
            if _extract_root(symbol_part) == instrument:
                matching.append(f)

        if not matching:
            logger.warning("no_data_files", instrument=instrument)
            return pd.DataFrame()

        dfs = []
        for f in matching:
            df = self._read_zstd_csv(f)
            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined["ts_event"] = pd.to_datetime(combined["ts_event"])
        combined = combined.sort_values("ts_event")

        # For overlapping contracts, keep the one with highest volume at each timestamp
        if use_front_month:
            combined = (
                combined.sort_values(["ts_event", "volume"], ascending=[True, False])
                .drop_duplicates(subset=["ts_event"], keep="first")
            )

        combined = combined.reset_index(drop=True)
        combined["instrument"] = instrument
        self._cache[instrument] = combined

        logger.info(
            "data.loaded",
            instrument=instrument,
            bars=len(combined),
            start=str(combined["ts_event"].iloc[0]),
            end=str(combined["ts_event"].iloc[-1]),
        )
        return combined

    def to_bars(self, df: pd.DataFrame) -> List[Bar]:
        """Convert DataFrame to list of Bar objects."""
        bars = []
        for _, row in df.iterrows():
            bars.append(Bar(
                timestamp=row["ts_event"].timestamp(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                symbol=str(row.get("symbol", row.get("instrument", ""))),
            ))
        return bars

    def _read_zstd_csv(self, filepath: str) -> Optional[pd.DataFrame]:
        """Read a single zstd-compressed CSV file using streaming decompression."""
        try:
            dctx = zstd.ZstdDecompressor()
            with open(filepath, "rb") as f:
                reader = dctx.stream_reader(f)
                text = io.TextIOWrapper(reader, encoding="utf-8")
                df = pd.read_csv(text)
            if df.empty:
                return None
            return df
        except Exception as e:
            logger.warning("file_read_error", file=filepath, error=str(e))
            return None
