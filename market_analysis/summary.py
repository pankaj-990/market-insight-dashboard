"""Utilities for turning market data into structured text prompts."""

from __future__ import annotations

from textwrap import dedent

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


def build_technical_summary(
    df: pd.DataFrame,
    key_window: int = 30,
    recent_rows: int = 10,
) -> str:
    """Return the structured technical summary used to prompt the LLM."""
    if df.empty:
        raise ValueError("DataFrame is empty; cannot build summary.")

    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("DataFrame missing required columns: " + ", ".join(missing))

    latest_close = df["Close"].iloc[-1]
    recent_high = df["High"].tail(key_window).max()
    recent_low = df["Low"].tail(key_window).min()
    ema_30 = df["EMA_30"].iloc[-1]
    ema_200 = df["EMA_200"].iloc[-1]
    rsi_14 = df["RSI_14"].iloc[-1]

    # Classic floor-trader pivot levels derived from the latest candle.
    last_high = df["High"].iloc[-1]
    last_low = df["Low"].iloc[-1]
    pivot_point = (last_high + last_low + latest_close) / 3
    resistance_1 = (2 * pivot_point) - last_low
    support_1 = (2 * pivot_point) - last_high
    resistance_2 = pivot_point + (last_high - last_low)
    support_2 = pivot_point - (last_high - last_low)

    rsi_state = _describe_rsi(rsi_14)
    price_vs_ema30 = _qualify_comparison(latest_close, ema_30, "Bullish", "Bearish")
    price_vs_ema200 = _qualify_comparison(latest_close, ema_200, "Bullish", "Bearish")
    ema_cross = _qualify_comparison(ema_30, ema_200, "Golden", "Death")

    recent_table = (
        df[["Open", "High", "Low", "Close", "RSI_14"]].tail(recent_rows).to_string()
    )

    summary = dedent(
        f"""
        TECHNICAL ANALYSIS SUMMARY
        Data Period: {df.index[0].date()} to {df.index[-1].date()}
        Latest Data: {df.index[-1].date()}

        KEY LEVELS:
        - Latest Close: {latest_close}
        - Recent High ({key_window} periods): {recent_high}
        - Recent Low ({key_window} periods): {recent_low}
        - Key Support: ~{recent_low}
        - Key Resistance: ~{recent_high}

        KEY MOVING AVERAGES:
        - Price vs EMA 30: {price_vs_ema30}
        - Price vs EMA 200: {price_vs_ema200}
        - EMA 30 vs EMA 200: {ema_cross} Cross
        - EMA 30 Value: {ema_30}
        - EMA 200 Value: {ema_200}

        SUPPORT / RESISTANCE CUES:
        - Pivot Point (H+L+C)/3: {pivot_point:.2f}
        - R1 / S1: {resistance_1:.2f} / {support_1:.2f}
        - R2 / S2: {resistance_2:.2f} / {support_2:.2f}

        MOMENTUM & OSCILLATORS:
        - RSI 14: {rsi_14} ({rsi_state})


        RECENT PRICE ACTION (Last {recent_rows} candles):
        {recent_table}
        """
    ).strip()

    return summary


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


__all__ = ["build_technical_summary"]
