"""Utilities for turning market data into structured text prompts."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Iterable

import pandas as pd

_REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "EMA_30",
    "EMA_200",
    "RSI_14",
]

_DEFAULT_RECENT_COLUMNS = ["Open", "High", "Low", "Close", "RSI_14"]
_MAX_RECENT_ROWS = 12


@dataclass(slots=True)
class _TimeframeSnapshot:
    label: str
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    latest_close: float
    recent_high: float
    recent_low: float
    ema_30: float
    ema_200: float
    rsi_14: float
    last_high: float
    last_low: float
    pivot_point: float
    resistance_1: float
    support_1: float
    resistance_2: float
    support_2: float
    recent_table: str
    key_window: int
    recent_rows: int


def build_technical_summary(
    df: pd.DataFrame,
    key_window: int = 30,
    recent_rows: int = 8,
) -> str:
    """Return the structured technical summary used to prompt the LLM."""
    snapshot = _build_snapshot(
        df, label="Primary", key_window=key_window, recent_rows=recent_rows
    )
    return _render_snapshot(snapshot, include_label=False)


def build_multi_timeframe_summary(
    higher_df: pd.DataFrame,
    lower_df: pd.DataFrame,
    *,
    higher_label: str = "Higher Timeframe",
    lower_label: str = "Lower Timeframe",
    higher_key_window: int = 30,
    lower_key_window: int = 30,
    higher_recent_rows: int = 8,
    lower_recent_rows: int = 12,
) -> str:
    """Return a summary that captures both higher and lower timeframe context."""

    higher_snapshot = _build_snapshot(
        higher_df,
        label=higher_label,
        key_window=higher_key_window,
        recent_rows=higher_recent_rows,
    )
    lower_snapshot = _build_snapshot(
        lower_df,
        label=lower_label,
        key_window=lower_key_window,
        recent_rows=lower_recent_rows,
    )

    alignment_section = _render_alignment_section(higher_snapshot, lower_snapshot)

    summary_parts = [
        "TECHNICAL ANALYSIS SUMMARY",
        _render_snapshot(higher_snapshot, include_label=True),
        _render_snapshot(lower_snapshot, include_label=True),
        alignment_section,
    ]

    return "\n\n".join(part.strip() for part in summary_parts if part).strip()


def _describe_rsi(value: float) -> str:
    if value > 70:
        return "Overbought"
    if value < 30:
        return "Oversold"
    return "Neutral"


def _qualify_comparison(
    value: float, reference: float, positive: str, negative: str
) -> str:
    return positive if value > reference else negative


def _build_snapshot(
    df: pd.DataFrame,
    *,
    label: str,
    key_window: int,
    recent_rows: int,
) -> _TimeframeSnapshot:
    if df.empty:
        raise ValueError("DataFrame is empty; cannot build summary.")

    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("DataFrame missing required columns: " + ", ".join(missing))

    recent_rows = max(1, min(recent_rows, _MAX_RECENT_ROWS, len(df)))

    latest_close = df["Close"].iloc[-1]
    recent_high = df["High"].tail(key_window).max()
    recent_low = df["Low"].tail(key_window).min()
    ema_30 = df["EMA_30"].iloc[-1]
    ema_200 = df["EMA_200"].iloc[-1]
    rsi_14 = df["RSI_14"].iloc[-1]

    last_high = df["High"].iloc[-1]
    last_low = df["Low"].iloc[-1]
    pivot_point = (last_high + last_low + latest_close) / 3
    resistance_1 = (2 * pivot_point) - last_low
    support_1 = (2 * pivot_point) - last_high
    resistance_2 = pivot_point + (last_high - last_low)
    support_2 = pivot_point - (last_high - last_low)

    recent_table = _format_recent_table(
        df,
        rows=recent_rows,
        columns=_DEFAULT_RECENT_COLUMNS,
    )

    return _TimeframeSnapshot(
        label=label,
        data_start=df.index[0],
        data_end=df.index[-1],
        latest_close=latest_close,
        recent_high=recent_high,
        recent_low=recent_low,
        ema_30=ema_30,
        ema_200=ema_200,
        rsi_14=rsi_14,
        last_high=last_high,
        last_low=last_low,
        pivot_point=pivot_point,
        resistance_1=resistance_1,
        support_1=support_1,
        resistance_2=resistance_2,
        support_2=support_2,
        recent_table=recent_table,
        key_window=key_window,
        recent_rows=recent_rows,
    )


def _render_snapshot(snapshot: _TimeframeSnapshot, *, include_label: bool) -> str:
    rsi_state = _describe_rsi(snapshot.rsi_14)
    price_vs_ema30 = _qualify_comparison(
        snapshot.latest_close, snapshot.ema_30, "Bullish", "Bearish"
    )
    price_vs_ema200 = _qualify_comparison(
        snapshot.latest_close, snapshot.ema_200, "Bullish", "Bearish"
    )
    ema_cross = _qualify_comparison(
        snapshot.ema_30, snapshot.ema_200, "Golden", "Death"
    )

    lines: list[str] = []
    if include_label:
        lines.append(snapshot.label)
    lines.append(
        f"Data Period: {snapshot.data_start.date()} to {snapshot.data_end.date()}"
    )
    lines.append(f"Latest Data: {snapshot.data_end.date()}")
    lines.append("")
    lines.append("KEY LEVELS:")
    lines.append(f"- Latest Close: {snapshot.latest_close:.2f}")
    lines.append(f"- Recent High ({snapshot.key_window}): {snapshot.recent_high:.2f}")
    lines.append(f"- Recent Low ({snapshot.key_window}): {snapshot.recent_low:.2f}")
    lines.append(f"- Key Support: ~{snapshot.recent_low:.2f}")
    lines.append(f"- Key Resistance: ~{snapshot.recent_high:.2f}")
    lines.append("")
    lines.append("KEY MOVING AVERAGES:")
    lines.append(f"- Price vs EMA 30: {price_vs_ema30}")
    lines.append(f"- Price vs EMA 200: {price_vs_ema200}")
    lines.append(f"- EMA 30 vs EMA 200: {ema_cross} Cross")
    lines.append(f"- EMA 30 Value: {snapshot.ema_30:.2f}")
    lines.append(f"- EMA 200 Value: {snapshot.ema_200:.2f}")
    lines.append("")
    lines.append("SUPPORT / RESISTANCE CUES:")
    lines.append(f"- Pivot Point: {snapshot.pivot_point:.2f}")
    lines.append(f"- R1 / S1: {snapshot.resistance_1:.2f} / {snapshot.support_1:.2f}")
    lines.append(f"- R2 / S2: {snapshot.resistance_2:.2f} / {snapshot.support_2:.2f}")
    lines.append("")
    lines.append("MOMENTUM & OSCILLATORS:")
    lines.append(f"- RSI 14: {snapshot.rsi_14:.2f} ({rsi_state})")
    lines.append("")
    lines.append(f"RECENT PRICE ACTION (Last {snapshot.recent_rows} candles):")
    lines.append(snapshot.recent_table)

    summary = "\n".join(lines).strip()
    if include_label:
        return summary
    return "\n".join(["TECHNICAL ANALYSIS SUMMARY", summary]).strip()


def _format_recent_table(df: pd.DataFrame, *, rows: int, columns: Iterable[str]) -> str:
    if rows <= 0 or df.empty:
        return "No recent candles available."

    cols = [col for col in columns if col in df.columns]
    if not cols:
        return "No recent candles available."

    trimmed = df.tail(rows)
    header = ["Date"] + cols
    lines = [" | ".join(header)]

    for index, row in trimmed.iterrows():
        ts_text = _format_timestamp(index)
        values = [_format_number(row.get(col)) for col in cols]
        lines.append(" | ".join([ts_text] + values))

    return "\n".join(lines)


def _format_timestamp(value: pd.Timestamp) -> str:
    if not isinstance(value, pd.Timestamp):
        return str(value)
    ts = value.tz_convert(None) if value.tzinfo else value
    if ts.hour or ts.minute:
        return ts.strftime("%Y-%m-%d %H:%M")
    return ts.strftime("%Y-%m-%d")


def _format_number(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.2f}"
    return str(value)


def _render_alignment_section(
    higher: _TimeframeSnapshot, lower: _TimeframeSnapshot
) -> str:
    higher_trend = _qualify_comparison(
        higher.latest_close, higher.ema_30, "Bullish", "Bearish"
    )
    lower_trend = _qualify_comparison(
        lower.latest_close, lower.ema_30, "Bullish", "Bearish"
    )

    higher_delta = higher.ema_30 - higher.ema_200
    lower_delta = lower.ema_30 - lower.ema_200
    ema_alignment = (
        "Aligned"
        if (higher_delta >= 0 and lower_delta >= 0)
        or (higher_delta <= 0 and lower_delta <= 0)
        else "Divergent"
    )

    rsi_gap = abs(lower.rsi_14 - higher.rsi_14)
    if rsi_gap <= 2:
        rsi_alignment = "Momentum in sync"
    elif lower.rsi_14 > higher.rsi_14:
        rsi_alignment = "Lower TF leading"
    else:
        rsi_alignment = "Lower TF lagging"

    confluence_text = _analyse_confluence(higher, lower)

    section = dedent(
        f"""
        MULTI-TIMEFRAME ALIGNMENT:
        - Trend: {higher.label} {higher_trend} / {lower.label} {lower_trend}
        - EMA: {ema_alignment} (HTF Δ={higher_delta:.2f}, LTF Δ={lower_delta:.2f})
        - RSI: HTF {higher.rsi_14:.2f} / LTF {lower.rsi_14:.2f} -> {rsi_alignment}
        - Confluence: {confluence_text}
        """
    ).strip()

    return section


def _analyse_confluence(higher: _TimeframeSnapshot, lower: _TimeframeSnapshot) -> str:
    higher_levels = {higher.pivot_point, higher.recent_high, higher.recent_low}
    lower_levels = {lower.pivot_point, lower.recent_high, lower.recent_low}
    tolerance = max(
        abs(higher.latest_close) * 0.005,  # 0.5% of price as tolerance
        0.01,
    )

    matches = []
    for h_level in higher_levels:
        for l_level in lower_levels:
            if abs(h_level - l_level) <= tolerance:
                matches.append((h_level + l_level) / 2)

    if not matches:
        return "No immediate overlap; watch how LTF develops near HTF key zones."

    unique_levels = sorted({round(level, 2) for level in matches})
    joined = ", ".join(str(level) for level in unique_levels)
    return f"Confluence near {joined}"


__all__ = ["build_technical_summary", "build_multi_timeframe_summary"]
