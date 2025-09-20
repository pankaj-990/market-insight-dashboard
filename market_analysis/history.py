"""History persistence helpers for technical analysis runs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


@dataclass(slots=True)
class AnalysisHistory:
    """Lightweight history store (SQLite-backed with legacy JSON support)."""

    path: Path = Path("analysis_history.db")
    _mode: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = self.path.expanduser()
        suffix = self.path.suffix.lower()
        if suffix == ".json":
            self._mode = "json"
        else:
            self._mode = "sqlite"
            self._initialise_db()

    def load(self) -> List[Dict[str, Any]]:
        """Return previously recorded history entries."""
        if self._mode == "json":
            if not self.path.exists():
                return []
            try:
                return json.loads(self.path.read_text())
            except Exception:
                return []

        if not self.path.exists():
            return []

        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT payload FROM entries ORDER BY timestamp DESC"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def save(self, entries: Iterable[Dict[str, Any]]) -> None:
        """Persist ``entries`` to disk."""
        items = list(entries)
        if self._mode == "json":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(items, indent=2))
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialised = [self._serialise_entry(entry) for entry in items]
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM entries")
            conn.executemany(
                "INSERT INTO entries (cache_key, timestamp, payload) VALUES (?, ?, ?)",
                serialised,
            )

    def record(self, entry: Dict[str, Any]) -> None:
        """Add or replace a history entry and persist it."""
        if self._mode == "json":
            entries = self.load()
            cache_key = entry.get("cache_key")
            entries = [item for item in entries if item.get("cache_key") != cache_key]
            entries.append(entry)
            entries.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
            self.save(entries)
            return

        cache_key_str, timestamp, payload = self._serialise_entry(entry)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO entries (cache_key, timestamp, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    payload = excluded.payload
                """,
                (cache_key_str, timestamp, payload),
            )

    def find(self, cache_key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the entry that matches ``cache_key`` if present."""
        if self._mode == "json":
            for entry in self.load():
                if entry.get("cache_key") == cache_key:
                    return entry
            return None

        key_str = json.dumps(cache_key, sort_keys=True)
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT payload FROM entries WHERE cache_key = ?", (key_str,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def as_dataframe(
        self, entries: Optional[Iterable[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """Turn stored entries into a tabular summary for display."""
        items = list(entries) if entries is not None else self.load()
        if not items:
            return pd.DataFrame()
        records = []
        for entry in items:
            params = entry.get("params", {})
            cache_key = entry.get("cache_key", {})
            lower_interval = params.get("lower_interval") or cache_key.get(
                "lower_interval"
            )
            records.append(
                {
                    "Recorded": entry.get("timestamp"),
                    "Ticker": params.get("ticker"),
                    "Period": params.get("period"),
                    "Interval": params.get("interval"),
                    "Lower Interval": lower_interval,
                    "Lower Period": params.get("lower_period"),
                    "As Of": params.get("as_of") or cache_key.get("last_timestamp"),
                    "Last Candle": cache_key.get("last_timestamp"),
                    "Lower Last Candle": cache_key.get("lower_last_timestamp"),
                    "LLM": "Yes" if entry.get("llm_text") else "No",
                }
            )
        df = pd.DataFrame.from_records(records)
        if not df.empty:
            df = df.sort_values("Recorded", ascending=False)
        return df

    def _initialise_db(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    cache_key TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_entries_timestamp
                ON entries(timestamp DESC)
                """
            )
        self._maybe_import_legacy_json()

    def _maybe_import_legacy_json(self) -> None:
        legacy_path = self.path.with_suffix(".json")
        if not legacy_path.exists():
            return
        try:
            with sqlite3.connect(self.path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
                if count:
                    return
        except Exception:
            return
        try:
            entries = json.loads(legacy_path.read_text())
        except Exception:
            return
        if not entries:
            return
        self.save(entries)
        try:
            legacy_path.rename(legacy_path.with_suffix(".json.bak"))
        except OSError:
            pass

    def _serialise_entry(self, entry: Dict[str, Any]) -> tuple[str, str, str]:
        cache_key = entry.get("cache_key") or {}
        cache_key_str = json.dumps(cache_key, sort_keys=True)
        timestamp_value = entry.get("timestamp")
        if isinstance(timestamp_value, str):
            timestamp = timestamp_value
        elif timestamp_value is None:
            timestamp = ""
        else:
            timestamp = str(timestamp_value)
        payload = json.dumps(entry, separators=(",", ":"))
        return cache_key_str, timestamp, payload


def make_cache_key(
    params: Dict[str, Any],
    last_timestamp: str,
    lower_last_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a serialisable cache key for analysis results."""
    key = {
        "ticker": params.get("ticker"),
        "period": params.get("period"),
        "interval": params.get("interval"),
        "round_digits": params.get("round_digits"),
        "data_path": params.get("data_path"),
        "last_timestamp": last_timestamp,
    }

    if params.get("lower_interval"):
        key.update(
            {
                "lower_period": params.get("lower_period"),
                "lower_interval": params.get("lower_interval"),
                "lower_data_path": params.get("lower_data_path"),
                "lower_key_window": params.get("lower_key_window"),
                "lower_recent_rows": params.get("lower_recent_rows"),
                "lower_last_timestamp": lower_last_timestamp,
            }
        )

    return key


def make_entry_id(cache_key: Dict[str, Any]) -> str:
    """Return a deterministic identifier for a cached analysis entry."""

    return json.dumps(cache_key or {}, sort_keys=True)


__all__ = ["AnalysisHistory", "make_cache_key", "make_entry_id"]
