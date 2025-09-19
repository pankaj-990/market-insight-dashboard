"""Data acquisition and caching helpers for symbol-agnostic workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pandas_ta as ta
import yfinance as yf

_PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
_INDICATOR_COLUMNS = ["EMA_30", "EMA_200", "RSI_14"]
_ALL_COLUMNS = _PRICE_COLUMNS + _INDICATOR_COLUMNS


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    return slug or "dataset"


def default_data_filename(ticker: str, interval: str | None = None) -> str:
    slug = _slugify(ticker)
    interval_slug = _slugify(interval) if interval else ""
    if interval_slug:
        return f"{slug}_{interval_slug}_ohlc.csv"
    return f"{slug}_ohlc.csv"


def default_data_path(ticker: str, interval: str | None = None) -> Path:
    return Path(default_data_filename(ticker, interval=interval))


@dataclass(slots=True)
class DataFetcher:
    """Download OHLC candles and attach common technical indicators."""

    ticker: str = "^NSEI"
    period: str = "1y"
    interval: str = "1d"

    def fetch(self) -> pd.DataFrame:
        """Return a DataFrame with OHLC data and EMA/RSI indicators."""
        data = yf.download(
            self.ticker,
            period=self.period,
            interval=self.interval,
            progress=False,
        )
        if data.empty:
            raise ValueError("No data returned for the requested period.")

        # yfinance sometimes delivers a column MultiIndex; flatten it for consistency.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = data[_PRICE_COLUMNS].copy()
        df.index.name = "Date"
        df["EMA_30"] = ta.ema(df["Close"], length=30)
        df["EMA_200"] = ta.ema(df["Close"], length=200)
        df["RSI_14"] = ta.rsi(df["Close"], length=14)

        return df

    def fetch_and_save(
        self, path: Path | str, decimal_places: Optional[int] = 2
    ) -> pd.DataFrame:
        """Fetch data, optionally round the result, and persist it to ``path``."""
        df = self.fetch()
        if decimal_places is not None:
            df = df.round(decimal_places)
        save_dataframe(df, path)
        return df


@dataclass(slots=True)
class DatasetRequest:
    """Parameters that describe a dataset download/cache request."""

    ticker: str = "^NSEI"
    period: str = "1y"
    interval: str = "1d"
    data_path: Path | str | None = None
    round_digits: Optional[int] = 2
    max_age_days: int = 3

    def resolved_path(self) -> Path:
        """Return the cache path as an expanded :class:`~pathlib.Path`."""
        if self.data_path is not None:
            return Path(self.data_path).expanduser()
        return default_data_path(self.ticker, interval=self.interval)

    def create_fetcher(self) -> DataFetcher:
        """Instantiate a :class:`DataFetcher` for the current request."""
        return DataFetcher(
            ticker=self.ticker,
            period=self.period,
            interval=self.interval,
        )


def ensure_dataset(
    request: DatasetRequest,
    *,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Return a dataset, refreshing the cache when required."""
    fetcher = request.create_fetcher()
    cache_path = request.resolved_path()

    if force_refresh:
        df = fetcher.fetch_and_save(cache_path, decimal_places=request.round_digits)
        return df, "freshly downloaded"

    try:
        df = load_cached_data(cache_path)
    except FileNotFoundError:
        df = fetcher.fetch_and_save(cache_path, decimal_places=request.round_digits)
        return df, "freshly downloaded"

    if is_stale(df.index[-1], request.max_age_days):
        df = fetcher.fetch_and_save(cache_path, decimal_places=request.round_digits)
        return df, "refreshed"

    return df, "cached"


def is_stale(last_timestamp: pd.Timestamp | datetime, max_age_days: int) -> bool:
    """Return ``True`` when the cached data is older than permitted."""
    if max_age_days < 0:
        return False

    if hasattr(last_timestamp, "to_pydatetime"):
        last_date = last_timestamp.to_pydatetime().date()
    else:
        last_date = last_timestamp.date()  # type: ignore[union-attr]

    today = datetime.utcnow().date()
    age_days = (today - last_date).days
    return age_days > max_age_days


def save_dataframe(df: pd.DataFrame, path: Path | str) -> None:
    """Write the OHLC/indicator DataFrame to disk."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)


def load_cached_data(path: Path | str) -> pd.DataFrame:
    """Load a cached OHLC/indicator CSV produced by :class:`DataFetcher`."""
    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    except ValueError:
        # Older caches (especially intraday) may store the index under a different column.
        raw = pd.read_csv(csv_path)
        if raw.empty:
            raise
        date_column = None
        for candidate in ("Date", "Datetime", "date", "datetime", raw.columns[0]):
            if candidate in raw.columns:
                date_column = candidate
                break
        if date_column is None:
            raise
        if date_column != "Date":
            raw = raw.rename(columns={date_column: "Date"})
        raw["Date"] = pd.to_datetime(raw["Date"])
        df = raw.set_index("Date")

    missing_columns = [col for col in _ALL_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Data file is missing expected columns: " + ", ".join(missing_columns)
        )

    df.index.name = "Date"
    return df


__all__ = [
    "DataFetcher",
    "DatasetRequest",
    "ensure_dataset",
    "is_stale",
    "load_cached_data",
    "save_dataframe",
]
