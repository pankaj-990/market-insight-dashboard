"""History persistence helpers for technical analysis runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


@dataclass(slots=True)
class AnalysisHistory:
    """Lightweight JSON-backed storage for analysis history."""

    path: Path = Path("analysis_history.json")

    def load(self) -> List[Dict[str, Any]]:
        """Return previously recorded history entries."""
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return []

    def save(self, entries: Iterable[Dict[str, Any]]) -> None:
        """Persist ``entries`` to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(list(entries), indent=2))

    def record(self, entry: Dict[str, Any]) -> None:
        """Add or replace a history entry and persist it."""
        entries = self.load()
        cache_key = entry.get("cache_key")
        entries = [item for item in entries if item.get("cache_key") != cache_key]
        entries.append(entry)
        entries.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        self.save(entries)

    def find(self, cache_key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the entry that matches ``cache_key`` if present."""
        for entry in self.load():
            if entry.get("cache_key") == cache_key:
                return entry
        return None

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


__all__ = ["AnalysisHistory", "make_cache_key"]
