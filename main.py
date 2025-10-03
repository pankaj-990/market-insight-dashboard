"""Streamlit UI for the market technical analysis workflow."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_analysis import (DEFAULT_FEATURE_COLUMNS, SUPPORTED_NORMALIZATIONS,
                             AnalysisHistory, DatasetRequest, MotifQueryResult,
                             build_multi_timeframe_summary,
                             build_technical_summary, ensure_dataset,
                             format_prompt, generate_llm_analysis,
                             generate_motif_matches_from_dataframe,
                             make_cache_key, make_entry_id)
from market_analysis.data import default_data_path
from market_analysis.llm import LLMConfig

st.set_page_config(
    page_title="Technical Analysis Dashboard",
    page_icon="📈",
    layout="wide",
)

HISTORY = AnalysisHistory(Path("analysis_history.db"))


PERIOD_OPTIONS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"]
INTERVAL_OPTIONS = ["1d", "1wk", "1mo"]
LOWER_INTERVAL_OPTIONS = ["30m", "1h", "2h", "4h", "1d"]
LOWER_PERIOD_SUGGESTIONS = ["30d", "60d", "90d", "180d", "1y"]

SUMMARY_HEADING_RE = re.compile(r"^[A-Z0-9 /()_-]+:?$")


WEEKLY_EMA_TICKERS: list[tuple[str, str]] = [
    ("niftybees.ns", "NiftyBeES"),
    ("bankbees.ns", "NiftyBankBeES"),
    ("goldbees.ns", "GoldBeES"),
    ("silverbees.ns", "SilverBeES"),
    ("hngsngbees.ns", "Hang Seng BeES"),
    ("mon100.ns", "NASDAQ 100"),
]
WEEKLY_EMA_PERIOD = "5y"
WEEKLY_EMA_INTERVAL = "1wk"


def _load_weekly_ema_dataset(ticker: str) -> tuple[pd.DataFrame, str]:
    """Fetch cached weekly OHLC data with EMA indicators for the given ticker."""

    return _load_dataset_cached(
        ticker,
        WEEKLY_EMA_PERIOD,
        WEEKLY_EMA_INTERVAL,
        None,
        2,
        1,
    )


def _build_weekly_ema_chart(df: pd.DataFrame, *, title: str) -> go.Figure:
    """Return a Plotly figure with weekly close and 30-EMA lines."""

    trimmed = df[["Close", "EMA_30"]].dropna(subset=["Close"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trimmed.index,
            y=trimmed["Close"],
            mode="lines",
            name="Close",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trimmed.index,
            y=trimmed["EMA_30"],
            mode="lines",
            name="30 EMA",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=50, r=20, b=40, l=60),
    )
    return fig


@st.cache_data(show_spinner=False)
def _load_dataset_cached(
    ticker: str,
    period: str,
    interval: str,
    data_path: Optional[str],
    round_digits: Optional[int],
    max_age_days: int,
) -> tuple[pd.DataFrame, str]:
    """Shared cache for dataset loading to avoid repeated disk I/O."""
    request = DatasetRequest(
        ticker=ticker,
        period=period,
        interval=interval,
        data_path=data_path,
        round_digits=round_digits,
        max_age_days=max_age_days,
    )
    return ensure_dataset(request, force_refresh=False)


def _safe_index(options: list[str], value: str, fallback: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return fallback


def _summary_to_markdown(summary: str) -> str:
    """Render the plain-text technical summary as lightweight Markdown."""

    def flush(
        blocks: list[tuple[str | None, list[str]]],
        buffer: list[str],
        block_kind: list[str | None],
    ) -> None:
        if buffer:
            blocks.append((block_kind[0], buffer.copy()))
            buffer.clear()
            block_kind[0] = None

    lines = summary.splitlines()
    blocks: list[tuple[str | None, list[str]]] = []
    buffer: list[str] = []
    block_kind: list[str | None] = [None]

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush(blocks, buffer, block_kind)
            continue

        if (
            SUMMARY_HEADING_RE.match(stripped) and len(stripped.split()) <= 7
        ) or stripped.endswith(":"):
            flush(blocks, buffer, block_kind)
            heading_text = stripped.rstrip(":")
            if heading_text:
                blocks.append(("heading", [heading_text]))
            continue

        if stripped.startswith("- "):
            if block_kind[0] != "list":
                flush(blocks, buffer, block_kind)
                block_kind[0] = "list"
            buffer.append(stripped[2:].strip())
            continue

        if "|" in stripped:
            if block_kind[0] != "table":
                flush(blocks, buffer, block_kind)
                block_kind[0] = "table"
            cells = [cell.strip() for cell in stripped.split("|")]
            buffer.append("|".join(cells))
            continue

        if block_kind[0] != "text":
            flush(blocks, buffer, block_kind)
            block_kind[0] = "text"
        buffer.append(stripped)

    flush(blocks, buffer, block_kind)

    markdown_parts: list[str] = []
    for kind, block in blocks:
        if not block:
            continue
        if kind == "heading":
            markdown_parts.append(f"### {block[0]}")
        elif kind == "list":
            markdown_parts.extend(f"- {item}" for item in block)
        elif kind == "table":
            rows = [row.split("|") for row in block]
            header = rows[0]
            markdown_parts.append("| " + " | ".join(header) + " |")
            markdown_parts.append("| " + " | ".join(["---"] * len(header)) + " |")
            for row in rows[1:]:
                padded = row + [""] * (len(header) - len(row))
                markdown_parts.append("| " + " | ".join(padded[: len(header)]) + " |")
        else:
            markdown_parts.append(" ".join(block))

    return "\n\n".join(part for part in markdown_parts if part.strip())


@dataclass(slots=True)
class AnalysisParams:
    """User-selected configuration captured from the sidebar form."""

    ticker: str = "^NSEI"
    period: str = "5y"
    interval: str = "1d"
    include_lower: bool = True
    lower_period: str = "60d"
    lower_interval: str = "1h"
    lower_key_window: int = 30
    lower_recent_rows: int = 30
    round_digits: int = 2
    max_age_days: int = 3
    force_refresh: bool = False
    call_llm: bool = True
    show_prompt: bool = False
    motif_enabled: bool = True
    motif_backend: str = "faiss"
    motif_window_size: int = 30
    motif_top_k: int = 5
    motif_features: str = ",".join(DEFAULT_FEATURE_COLUMNS)
    motif_normalization: str = "zscore"
    motif_default_regime: str = "auto"
    motif_filter_ticker: str = ""
    motif_filter_timeframe: str = ""
    motif_filter_regime: str = ""
    motif_persist_dir: str = ""
    motif_collection: str = "motifs"

    @property
    def data_path(self) -> Path:
        return default_data_path(self.ticker, self.interval)

    @property
    def lower_data_path(self) -> Path:
        return default_data_path(self.ticker, self.lower_interval)

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "AnalysisParams":
        """Hydrate an ``AnalysisParams`` instance from session state."""
        return cls(
            ticker=state.get("ticker", "^NSEI"),
            period=state.get("period", "1y"),
            interval=state.get("interval", "1d"),
            include_lower=bool(state.get("include_lower", True)),
            lower_period=state.get("lower_period", "60d"),
            lower_interval=state.get("lower_interval", "1h"),
            lower_key_window=int(state.get("lower_key_window", 30)),
            lower_recent_rows=int(state.get("lower_recent_rows", 30)),
            round_digits=int(state.get("round_digits", 2)),
            max_age_days=int(state.get("max_age_days", 3)),
            force_refresh=bool(state.get("force_refresh", False)),
            call_llm=bool(state.get("call_llm", True)),
            show_prompt=bool(state.get("show_prompt", False)),
            motif_enabled=bool(state.get("motif_enabled", False)),
            motif_backend=str(state.get("motif_backend", "faiss")),
            motif_window_size=int(state.get("motif_window_size", 30)),
            motif_top_k=int(state.get("motif_top_k", 5)),
            motif_features=str(
                state.get("motif_features", ",".join(DEFAULT_FEATURE_COLUMNS))
            ),
            motif_normalization=str(state.get("motif_normalization", "zscore")),
            motif_default_regime=str(state.get("motif_default_regime", "auto")),
            motif_filter_ticker=str(state.get("motif_filter_ticker", "")),
            motif_filter_timeframe=str(state.get("motif_filter_timeframe", "")),
            motif_filter_regime=str(state.get("motif_filter_regime", "")),
            motif_persist_dir=str(state.get("motif_persist_dir", "")),
            motif_collection=str(state.get("motif_collection", "motifs")),
        )

    def to_state(self) -> Dict[str, Any]:
        """Serialise the params for storage in ``st.session_state``."""
        return {
            "ticker": self.ticker,
            "period": self.period,
            "interval": self.interval,
            "include_lower": self.include_lower,
            "lower_period": self.lower_period,
            "lower_interval": self.lower_interval,
            "lower_key_window": self.lower_key_window,
            "lower_recent_rows": self.lower_recent_rows,
            "round_digits": self.round_digits,
            "max_age_days": self.max_age_days,
            "data_path": str(self.data_path),
            "force_refresh": self.force_refresh,
            "call_llm": self.call_llm,
            "show_prompt": self.show_prompt,
            "motif_enabled": self.motif_enabled,
            "motif_backend": self.motif_backend,
            "motif_window_size": self.motif_window_size,
            "motif_top_k": self.motif_top_k,
            "motif_features": self.motif_features,
            "motif_normalization": self.motif_normalization,
            "motif_default_regime": self.motif_default_regime,
            "motif_filter_ticker": self.motif_filter_ticker,
            "motif_filter_timeframe": self.motif_filter_timeframe,
            "motif_filter_regime": self.motif_filter_regime,
            "motif_persist_dir": self.motif_persist_dir,
            "motif_collection": self.motif_collection,
        }

    def higher_dataset_request(self) -> DatasetRequest:
        """Translate the primary timeframe into a :class:`DatasetRequest`."""
        return DatasetRequest(
            ticker=self.ticker,
            period=self.period,
            interval=self.interval,
            round_digits=self.round_digits,
            max_age_days=self.max_age_days,
        )

    def lower_dataset_request(self) -> Optional[DatasetRequest]:
        """Return a :class:`DatasetRequest` for the lower timeframe when enabled."""
        if not self.include_lower:
            return None
        interval = (self.lower_interval or "").strip()
        if not interval or interval.lower() == "none":
            return None
        period = (self.lower_period or self.period).strip()
        return DatasetRequest(
            ticker=self.ticker,
            period=period,
            interval=interval,
            round_digits=self.round_digits,
            max_age_days=self.max_age_days,
        )

    def cache_payload(self) -> Dict[str, Any]:
        """Payload used when building a cache key."""
        payload = {
            "ticker": self.ticker,
            "period": self.period,
            "interval": self.interval,
            "round_digits": self.round_digits,
            "data_path": str(self.data_path),
        }
        lower_request = self.lower_dataset_request()
        if lower_request is not None:
            payload.update(
                {
                    "lower_period": lower_request.period,
                    "lower_interval": lower_request.interval,
                    "lower_data_path": str(self.lower_data_path),
                    "lower_key_window": self.lower_key_window,
                    "lower_recent_rows": self.lower_recent_rows,
                }
            )
        return payload

    def history_payload(self) -> Dict[str, Any]:
        """Payload stored alongside history entries."""
        payload = self.cache_payload()
        payload.update(
            {
                "max_age_days": self.max_age_days,
                "force_refresh": self.force_refresh,
                "motif_enabled": self.motif_enabled,
                "motif_backend": self.motif_backend,
                "motif_window_size": self.motif_window_size,
                "motif_top_k": self.motif_top_k,
                "motif_features": self.motif_features,
                "motif_normalization": self.motif_normalization,
                "motif_default_regime": self.motif_default_regime,
                "motif_filter_ticker": self.motif_filter_ticker,
                "motif_filter_timeframe": self.motif_filter_timeframe,
                "motif_filter_regime": self.motif_filter_regime,
                "motif_persist_dir": self.motif_persist_dir,
                "motif_collection": self.motif_collection,
            }
        )
        return payload

    def motif_feature_list(self) -> list[str]:
        features = [
            item.strip() for item in self.motif_features.split(",") if item.strip()
        ]
        if not features:
            return list(DEFAULT_FEATURE_COLUMNS)
        return list(dict.fromkeys(features))

    def motif_metadata_filter(self) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        if self.motif_filter_ticker.strip():
            filters["ticker"] = self.motif_filter_ticker.strip()
        if self.motif_filter_timeframe.strip():
            filters["timeframe"] = self.motif_filter_timeframe.strip()
        if self.motif_filter_regime.strip():
            regimes = [
                item.strip()
                for item in self.motif_filter_regime.split(",")
                if item.strip()
            ]
            if len(regimes) == 1:
                filters["regime"] = regimes[0]
            elif regimes:
                filters["regime"] = regimes
        return filters

    def motif_normalization_mode(self) -> str:
        value = (self.motif_normalization or "").strip().lower()
        if value not in SUPPORTED_NORMALIZATIONS:
            return "zscore"
        return value

    def motif_persist_directory(self) -> Optional[str]:
        directory = self.motif_persist_dir.strip()
        return directory or None


def _load_default_params() -> AnalysisParams:
    stored = st.session_state.get("analysis_params")
    if isinstance(stored, dict):
        try:
            return AnalysisParams.from_state(stored)
        except Exception:
            pass
    return AnalysisParams()


def render_sidebar(defaults: AnalysisParams) -> tuple[AnalysisParams, bool]:
    """Render the configuration sidebar and return the captured params."""
    with st.sidebar:
        st.header("Configuration")
        with st.form("analysis-controls"):
            ticker = st.text_input("Ticker", value=defaults.ticker)
            period = st.selectbox(
                "Period",
                PERIOD_OPTIONS,
                index=_safe_index(PERIOD_OPTIONS, defaults.period, 3),
            )
            interval = st.selectbox(
                "Interval",
                INTERVAL_OPTIONS,
                index=_safe_index(INTERVAL_OPTIONS, defaults.interval),
            )
            round_digits = int(
                st.number_input(
                    "Round digits",
                    min_value=0,
                    max_value=6,
                    value=int(defaults.round_digits),
                    help="Controls rounding applied when caching new data.",
                )
            )
            max_age_days = int(
                st.number_input(
                    "Max cache age (days)",
                    min_value=-1,
                    max_value=30,
                    value=int(defaults.max_age_days),
                    help="Set -1 to skip freshness checks.",
                )
            )
            st.caption(f"Cache file: {default_data_path(ticker, interval)}")
            include_lower = st.checkbox(
                "Include lower timeframe analysis",
                value=defaults.include_lower,
                help="Adds a secondary intraday timeframe (e.g. hourly) to the summary.",
            )
            lower_interval = defaults.lower_interval
            lower_period = defaults.lower_period
            lower_key_window = defaults.lower_key_window
            lower_recent_rows = defaults.lower_recent_rows
            if include_lower:
                lower_interval_options = list(
                    dict.fromkeys([defaults.lower_interval] + LOWER_INTERVAL_OPTIONS)
                )
                lower_period = st.text_input(
                    "Lower period",
                    value=str(defaults.lower_period),
                    help="Lower timeframe lookback (e.g. 30d, 60d, 1y).",
                )
                lower_interval = st.selectbox(
                    "Lower interval",
                    lower_interval_options,
                    index=_safe_index(lower_interval_options, defaults.lower_interval),
                )

                # lower_key_window = int(
                #     st.number_input(
                #         "Lower key window (candles)",
                #         min_value=10,
                #         max_value=200,
                #         value=int(defaults.lower_key_window),
                #         help="Number of lower timeframe candles considered for highs and lows (flattened at 12-row summary).",
                #     )
                # )
                # lower_recent_rows = int(
                #     st.number_input(
                #         "Lower recent rows",
                #         min_value=10,
                #         max_value=60,
                #         value=int(defaults.lower_recent_rows),
                #         help="Rows from the lower timeframe included in the summary table (trimmed to 12 for the prompt).",
                #     )
                # )
                st.caption(
                    f"Lower cache file: {default_data_path(ticker, lower_interval)}"
                )
            else:
                st.caption("Lower timeframe disabled for this run.")
            call_llm = st.checkbox(
                "Generate LLM analysis",
                value=defaults.call_llm,
                help="Requires OpenRouter API key.",
            )
            motif_enabled = st.checkbox(
                "Enable price pattern",
                value=defaults.motif_enabled,
                help="Index rolling windows and surface the closest historical motifs for the latest candle.",
            )
            force_refresh = st.checkbox(
                "Force data refresh",
                value=defaults.force_refresh,
                help="Ignore cached CSV and refetch.",
            )
            show_prompt = st.checkbox(
                "Show LLM prompt",
                value=defaults.show_prompt,
                help="Display the composed system + user messages sent to the model.",
            )

            motif_backend = defaults.motif_backend
            motif_window_size = defaults.motif_window_size
            motif_top_k = defaults.motif_top_k
            motif_features = defaults.motif_features
            motif_normalization = defaults.motif_normalization
            motif_default_regime = defaults.motif_default_regime
            stored_filter_ticker = defaults.motif_filter_ticker.strip()
            previous_ticker = defaults.ticker.strip()
            if stored_filter_ticker:
                if (
                    previous_ticker
                    and stored_filter_ticker.lower() == previous_ticker.lower()
                ):
                    motif_filter_ticker = ticker
                else:
                    motif_filter_ticker = stored_filter_ticker
            else:
                motif_filter_ticker = ticker
            motif_filter_timeframe = defaults.motif_filter_timeframe or interval
            motif_filter_regime = defaults.motif_filter_regime
            motif_persist_dir = defaults.motif_persist_dir
            motif_collection = defaults.motif_collection

            # Keeping this disabled and using defaults for now
            if motif_enabled and False:
                with st.expander("Motif settings", expanded=True):
                    if False:
                        motif_backend = st.selectbox(
                            "Vector store backend",
                            ("faiss", "chroma"),
                            index=_safe_index(
                                ["faiss", "chroma"], defaults.motif_backend, 0
                            ),
                        )
                        motif_window_size = int(
                            st.number_input(
                                "Window size (candles)",
                                min_value=5,
                                max_value=500,
                                value=int(defaults.motif_window_size),
                                help="Number of sequential candles included in each motif window.",
                            )
                        )
                        motif_top_k = int(
                            st.number_input(
                                "Top K motifs",
                                min_value=1,
                                max_value=50,
                                value=int(defaults.motif_top_k),
                                help="Number of closest motifs to retrieve for the latest window.",
                            )
                        )
                        motif_features = st.text_input(
                            "Feature columns",
                            value=defaults.motif_features,
                            help="Comma-separated columns used to build motif feature vectors.",
                        )
                        normalization_options = list(SUPPORTED_NORMALIZATIONS)
                        motif_normalization = st.selectbox(
                            "Window normalisation",
                            normalization_options,
                            index=_safe_index(
                                normalization_options, defaults.motif_normalization, 1
                            ),
                            help="Normalisation applied across columns inside each window.",
                        )
                        motif_default_regime = st.text_input(
                            "Default regime label",
                            value=defaults.motif_default_regime,
                            help="Fallback regime label when automatic inference is unavailable (use 'auto' to infer).",
                        )
                        motif_filter_ticker = st.text_input(
                            "Filter ticker",
                            value=defaults.motif_filter_ticker or ticker,
                            help="Limit motif results to this ticker metadata (leave empty to use the analysed ticker).",
                        )
                        motif_filter_timeframe = st.text_input(
                            "Filter timeframe",
                            value=defaults.motif_filter_timeframe or interval,
                            help="Limit motif results to this timeframe metadata (leave empty to use the analysed interval).",
                        )
                        motif_filter_regime = st.text_input(
                            "Filter regime(s)",
                            value=defaults.motif_filter_regime,
                            help="Optional comma-separated list of regimes to include (matches metadata labels).",
                        )
                        if motif_backend == "chroma":
                            motif_persist_dir = st.text_input(
                                "Chroma persist directory",
                                value=defaults.motif_persist_dir,
                                help="Optional directory for persisting the Chroma collection.",
                            )
                            motif_collection = st.text_input(
                                "Chroma collection name",
                                value=defaults.motif_collection,
                            )
            submitted = st.form_submit_button("Run analysis")

    params = AnalysisParams(
        ticker=ticker,
        period=period,
        interval=interval,
        include_lower=include_lower,
        lower_period=lower_period,
        lower_interval=lower_interval,
        lower_key_window=lower_key_window,
        lower_recent_rows=lower_recent_rows,
        round_digits=round_digits,
        max_age_days=max_age_days,
        force_refresh=force_refresh,
        call_llm=call_llm,
        show_prompt=show_prompt,
        motif_enabled=motif_enabled,
        motif_backend=motif_backend,
        motif_window_size=motif_window_size,
        motif_top_k=motif_top_k,
        motif_features=motif_features,
        motif_normalization=motif_normalization,
        motif_default_regime=motif_default_regime,
        motif_filter_ticker=motif_filter_ticker or "",
        motif_filter_timeframe=motif_filter_timeframe or "",
        motif_filter_regime=motif_filter_regime,
        motif_persist_dir=motif_persist_dir,
        motif_collection=motif_collection,
    )
    return params, submitted


def handle_submission(
    params: AnalysisParams, history_entries: list[Dict[str, Any]]
) -> None:
    """Execute the end-to-end analysis workflow when the form is submitted."""
    lower_request = params.lower_dataset_request()
    higher_request = params.higher_dataset_request()

    if params.force_refresh:
        _load_dataset_cached.clear()
    with st.spinner("Loading market data..."):
        try:
            if params.force_refresh:
                higher_df, higher_source = ensure_dataset(
                    higher_request, force_refresh=True
                )
            else:
                higher_df, higher_source = _load_dataset_cached(
                    higher_request.ticker,
                    higher_request.period,
                    higher_request.interval,
                    str(higher_request.resolved_path()),
                    higher_request.round_digits,
                    higher_request.max_age_days,
                )
        except Exception as exc:
            _store_error(f"Unable to load data: {exc}")
            return

        lower_df: Optional[pd.DataFrame] = None
        lower_source: Optional[str] = None
        if lower_request is not None:
            try:
                if params.force_refresh:
                    lower_df, lower_source = ensure_dataset(
                        lower_request, force_refresh=True
                    )
                else:
                    lower_df, lower_source = _load_dataset_cached(
                        lower_request.ticker,
                        lower_request.period,
                        lower_request.interval,
                        str(lower_request.resolved_path()),
                        lower_request.round_digits,
                        lower_request.max_age_days,
                    )
            except Exception as exc:
                _store_error(f"Unable to load lower timeframe data: {exc}")
                return

    higher_last_timestamp = higher_df.index[-1].to_pydatetime().isoformat()
    lower_last_timestamp = (
        lower_df.index[-1].to_pydatetime().isoformat() if lower_df is not None else None
    )
    cache_key = make_cache_key(
        params.cache_payload(), higher_last_timestamp, lower_last_timestamp
    )
    existing_entry = None
    if not params.force_refresh:
        existing_entry = _lookup_history_entry(history_entries, cache_key)

    reused_summary = False
    reused_llm = False

    if existing_entry:
        technical_summary = existing_entry["technical_summary"]
        reused_summary = True
    else:
        try:
            if lower_df is not None and lower_request is not None:
                technical_summary = build_multi_timeframe_summary(
                    higher_df,
                    lower_df,
                    higher_label=f"Higher Timeframe ({params.interval})",
                    lower_label=f"Lower Timeframe ({lower_request.interval})",
                    lower_key_window=params.lower_key_window,
                    lower_recent_rows=params.lower_recent_rows,
                )
            else:
                technical_summary = build_technical_summary(higher_df)
        except Exception as exc:
            _store_error(f"Failed to build technical summary: {exc}")
            return

    prompt_text = format_prompt(technical_summary) if params.show_prompt else None

    llm_text: Optional[str] = None
    llm_error: Optional[str] = None
    llm_tables: Optional[Dict[str, Any]] = None

    if params.call_llm:
        if (
            existing_entry
            and existing_entry.get("llm_text")
            and not existing_entry.get("llm_error")
        ):
            llm_text = existing_entry["llm_text"]
            llm_error = existing_entry.get("llm_error")
            llm_tables = existing_entry.get("llm_tables")
            reused_llm = True
        else:
            llm_text, llm_tables, llm_error = _run_llm(technical_summary)
    elif existing_entry:
        llm_text = existing_entry.get("llm_text")
        llm_error = existing_entry.get("llm_error")
        llm_tables = existing_entry.get("llm_tables")

    cache_entry_id = make_entry_id(cache_key)

    motif_result: Optional[MotifQueryResult] = None
    motif_error: Optional[str] = None
    if params.motif_enabled:
        try:
            motif_result = generate_motif_matches_from_dataframe(
                higher_df,
                window_size=params.motif_window_size,
                feature_columns=params.motif_feature_list(),
                ticker=params.ticker,
                timeframe=params.interval,
                top_k=params.motif_top_k,
                backend=params.motif_backend,
                default_regime=params.motif_default_regime,
                normalization=(
                    None
                    if params.motif_normalization_mode() == "none"
                    else params.motif_normalization_mode()
                ),
                metadata_filter=(params.motif_metadata_filter() or None),
                persist_directory=(
                    params.motif_persist_directory()
                    if params.motif_backend == "chroma"
                    else None
                ),
                collection_name=(
                    params.motif_collection
                    if params.motif_backend == "chroma"
                    else None
                ),
            )
        except Exception as exc:
            motif_error = str(exc)
            motif_result = None

    motif_history_payload: Optional[Dict[str, Any]] = None
    if params.motif_enabled:
        motif_history_payload = {
            "enabled": True,
            "error": motif_error,
            "settings": {
                "backend": params.motif_backend,
                "window_size": params.motif_window_size,
                "top_k": params.motif_top_k,
                "features": params.motif_feature_list(),
                "normalization": params.motif_normalization_mode(),
            },
        }
        if motif_result is not None:
            motif_history_payload["query"] = motif_result.query_metadata.as_dict()
            motif_history_payload["matches"] = [
                {
                    "motif_id": match.motif_id,
                    "score": match.score,
                    "metadata": dict(match.metadata),
                }
                for match in motif_result.matches
            ]

    data_sources: Dict[str, Dict[str, str]] = {
        "higher": {
            "interval": params.interval,
            "source": higher_source,
            "path": str(params.data_path),
            "last_timestamp": higher_last_timestamp,
        }
    }
    if lower_df is not None and lower_request is not None and lower_source:
        data_sources["lower"] = {
            "interval": lower_request.interval,
            "source": lower_source,
            "path": str(params.lower_data_path),
            "last_timestamp": lower_last_timestamp or "",
        }

    history_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "cache_key": cache_key,
        "params": params.history_payload(),
        "data_source": data_sources,
        "technical_summary": technical_summary,
        "llm_text": llm_text,
        "llm_error": llm_error,
        "llm_tables": llm_tables,
        "entry_id": cache_entry_id,
    }
    if motif_history_payload is not None:
        history_entry["motif"] = motif_history_payload
    HISTORY.record(history_entry)

    result_payload = {
        "higher_df": higher_df,
        "lower_df": lower_df,
        "data_sources": data_sources,
        "technical_summary": technical_summary,
        "llm_text": llm_text,
        "llm_error": llm_error,
        "llm_tables": llm_tables,
        "call_llm": params.call_llm,
        "prompt_text": prompt_text,
        "show_prompt": params.show_prompt,
        "params": params.history_payload(),
        "cache_key": cache_key,
        "reused_summary": reused_summary,
        "reused_llm": reused_llm,
        "entry_id": cache_entry_id,
        "higher_last_timestamp": higher_last_timestamp,
        "lower_last_timestamp": lower_last_timestamp,
        "motif_enabled": params.motif_enabled,
        "motif_result": motif_result,
        "motif_error": motif_error,
        "motif_summary": motif_history_payload,
    }
    _store_result(result_payload, params)


def _lookup_history_entry(
    entries: list[Dict[str, Any]], cache_key: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    for entry in entries:
        if entry.get("cache_key") == cache_key:
            return entry
    return None


def _run_llm(
    technical_summary: str,
) -> tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """Execute the LLM analysis, handling missing configuration gracefully."""
    try:
        config = LLMConfig.from_env()
    except RuntimeError as exc:
        return None, None, str(exc)
    except Exception as exc:
        return None, None, f"Failed to initialise LLM: {exc}"

    try:
        with st.spinner("Generating LLM-backed narrative..."):
            analysis = generate_llm_analysis(technical_summary, config)
    except Exception as exc:
        return None, None, f"The LLM request failed: {exc}"
    return analysis.markdown, analysis.tables, None


def _store_result(result: Dict[str, Any], params: AnalysisParams) -> None:
    st.session_state["analysis_result"] = result
    st.session_state.pop("analysis_error", None)
    st.session_state["analysis_params"] = params.to_state()


def _store_error(message: str) -> None:
    st.session_state["analysis_error"] = message
    st.session_state.pop("analysis_result", None)


def _get_cached_result() -> Optional[Dict[str, Any]]:
    return st.session_state.get("analysis_result")


def _plot_candlestick(df: pd.DataFrame) -> None:
    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA_30"],
            mode="lines",
            name="EMA 30",
            line={"color": "#1f77b4"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA_200"],
            mode="lines",
            name="EMA 200",
            line={"color": "#ff7f0e"},
        )
    )
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(figure, width="stretch")


def main() -> None:
    st.title("Technical Analysis Dashboard")
    st.caption(
        "Interactively fetch market data, review technical summaries, and (optionally) "
        "generate LLM-backed insights."
    )

    history_entries = HISTORY.load()
    defaults = _load_default_params()
    params, submitted = render_sidebar(defaults)

    if submitted:
        handle_submission(params, history_entries)
        history_entries = HISTORY.load()

    error_message = st.session_state.get("analysis_error")
    if error_message:
        st.error(error_message)
        return

    result = _get_cached_result()
    if not result:
        st.info(
            "Configure parameters in the sidebar and click *Run analysis* to begin."
        )
        history_df = HISTORY.as_dataframe(history_entries)
        if not history_df.empty:
            st.subheader("Previous Analyses")
            st.dataframe(history_df, width="stretch")
        return

    higher_df = result["higher_df"]
    lower_df = result.get("lower_df")
    round_digits = result["params"]["round_digits"]
    data_sources = result.get("data_sources", {})
    higher_info = data_sources.get("higher", {})
    lower_info = data_sources.get("lower")
    motif_enabled = bool(result.get("motif_enabled"))
    motif_result: Optional[MotifQueryResult] = result.get("motif_result")
    motif_error = result.get("motif_error")
    motif_summary = result.get("motif_summary") or {}

    ticker_label = result["params"].get("ticker", "?")
    higher_message = (
        f"Using {higher_info.get('source', result.get('data_source', 'cached'))} data for `{ticker_label}` "
        f"({higher_info.get('interval', result['params'].get('interval', 'primary'))}) "
        f"from {higher_info.get('path', result['params'].get('data_path', 'cache'))}"
    )
    if lower_info:
        lower_message = (
            f" | Lower timeframe {lower_info.get('interval')} sourced "
            f"from {lower_info.get('path')} ({lower_info.get('source')})"
        )
    else:
        lower_message = ""
    st.success(higher_message + lower_message)

    status_notes = []
    if result.get("reused_summary"):
        status_notes.append("Reused cached technical summary")
    if result.get("reused_llm"):
        status_notes.append("Reused cached LLM analysis")
    if status_notes:
        st.caption(" | ".join(status_notes))

    latest_close = higher_df["Close"].iloc[-1]
    ema_30 = higher_df["EMA_30"].iloc[-1]
    ema_200 = higher_df["EMA_200"].iloc[-1]
    rsi_14 = higher_df["RSI_14"].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close", _format_metric(latest_close, round_digits))
    col2.metric("EMA 30", _format_metric(ema_30, round_digits))
    col3.metric("EMA 200", _format_metric(ema_200, round_digits))
    col4.metric("RSI 14", _format_metric(rsi_14, round_digits))

    if lower_df is not None and not lower_df.empty:
        lower_interval_label = (
            lower_info.get("interval", result["params"].get("lower_interval", "lower"))
            if lower_info
            else result["params"].get("lower_interval", "lower")
        )
        lower_close = lower_df["Close"].iloc[-1]
        lower_ema30 = lower_df["EMA_30"].iloc[-1]
        lower_ema200 = lower_df["EMA_200"].iloc[-1]
        lower_rsi = lower_df["RSI_14"].iloc[-1]
        lcol1, lcol2, lcol3, lcol4 = st.columns(4)
        lcol1.metric(
            f"Lower Close ({lower_interval_label})",
            _format_metric(lower_close, round_digits),
        )
        lcol2.metric(
            "Lower EMA 30",
            _format_metric(lower_ema30, round_digits),
        )
        lcol3.metric(
            "Lower EMA 200",
            _format_metric(lower_ema200, round_digits),
        )
        lcol4.metric(
            "Lower RSI 14",
            _format_metric(lower_rsi, round_digits),
        )

    summary_tab, data_tab, llm_tab, motif_tab, weekly_tab, history_tab = st.tabs(
        [
            "Technical Summary",
            "Recent Candles",
            "LLM Analysis",
            "Price Pattern Matches",
            "Weekly 30-EMA Overview",
            "History",
        ]
    )

    with summary_tab:
        formatted_summary = _summary_to_markdown(result["technical_summary"])
        st.markdown(formatted_summary)
        with st.expander("Raw summary (text)"):
            st.code(result["technical_summary"], language="text")
        st.download_button(
            "Download summary (.txt)",
            result["technical_summary"],
            file_name=f"technical_summary_{result['params']['ticker']}.txt",
        )
        if result.get("prompt_text"):
            with st.expander("LLM Prompt"):
                st.code(result["prompt_text"], language="text")
                st.download_button(
                    "Download prompt (.txt)",
                    result["prompt_text"],
                    file_name=f"llm_prompt_{result['params']['ticker']}.txt",
                    key="download-prompt",
                )

    with data_tab:
        higher_interval_label = higher_info.get(
            "interval", result["params"].get("interval", "primary")
        )
        st.subheader(f"Higher Timeframe Candlestick ({higher_interval_label})")
        _plot_candlestick(higher_df)
        st.divider()
        st.subheader(f"Higher Timeframe Latest Candles ({higher_interval_label})")
        st.dataframe(higher_df.tail(20))
        st.download_button(
            "Download higher timeframe (.csv)",
            higher_df.to_csv().encode("utf-8"),
            file_name=f"ohlc_{result['params']['ticker']}_{higher_interval_label}.csv",
            key="download-higher-data",
        )
        if lower_df is not None and not lower_df.empty:
            lower_interval_label = (
                lower_info.get(
                    "interval", result["params"].get("lower_interval", "lower")
                )
                if lower_info
                else result["params"].get("lower_interval", "lower")
            )
            st.divider()
            st.subheader(f"Lower Timeframe Candlestick ({lower_interval_label})")
            _plot_candlestick(lower_df)
            st.divider()
            st.subheader(f"Lower Timeframe Latest Candles ({lower_interval_label})")
            st.dataframe(lower_df.tail(60))
            st.download_button(
                "Download lower timeframe (.csv)",
                lower_df.to_csv().encode("utf-8"),
                file_name=f"ohlc_{result['params']['ticker']}_{lower_interval_label}.csv",
                key="download-lower-data",
            )

    with llm_tab:
        st.subheader("Narrative Insights")
        if not result["call_llm"]:
            st.info("LLM analysis was not requested for the latest run.")
        elif result.get("llm_text"):
            st.markdown(result["llm_text"])

        elif result.get("llm_tables"):
            tables = result.get("llm_tables") or {}
            # section_titles = {
            #     "overall_trend": "Overall Trend Assessment",
            #     "key_evidence": "Key Technical Evidence",
            #     "candlestick_patterns": "Candlestick Pattern Analysis",
            #     "chart_patterns": "Chart Pattern Analysis",
            #     "trade_plan": "Trade Plan Outline",
            # }
            # displayed = False
            # for key, title in section_titles.items():
            #     section = tables.get(key)
            #     if not section:
            #         continue
            #     headers = section.get("headers") or []
            #     rows = section.get("rows") or []
            #     if not headers:
            #         continue
            #     st.subheader(title)
            #     st.table(pd.DataFrame(rows, columns=headers))
            #     displayed = True
            # if displayed:
            #     st.divider()
            st.json(tables)
            st.download_button(
                "Download analysis (.json)",
                json.dumps(tables, indent=2),
                file_name=f"llm_analysis_{result['params']['ticker']}.json",
                mime="application/json",
                key="download-llm-json",
            )
        elif result.get("llm_text"):
            st.code(result["llm_text"], language="text")
        elif result["llm_error"]:
            st.warning(result["llm_error"])
        else:
            st.info("LLM analysis not available.")

        if result.get("llm_error") and result.get("llm_text"):
            st.caption("Warning: LLM reported an issue while generating this output.")

    with motif_tab:
        st.subheader("Nearest Matches")
        if not motif_enabled:
            st.info("Motif retrieval was not enabled for the latest analysis.")
        elif motif_error:
            st.warning(f"Motif retrieval failed: {motif_error}")
        elif motif_result is None:
            st.info(
                "Motif results are not available. Adjust the settings and rerun the analysis."
            )
        else:
            backend_label = motif_result.backend.upper()
            feature_list = ", ".join(motif_result.feature_columns)
            st.caption(
                f"Backend: {backend_label} | Window size: {motif_result.window_size} | "
                f"Features: {feature_list} | Normalisation: {motif_result.normalization}"
            )
            if motif_result.skipped_windows:
                st.caption(
                    f"Skipped {motif_result.skipped_windows} window(s) due to missing data or normalisation issues."
                )
            if motif_result.filters:
                filter_text = ", ".join(
                    f"{key}={value if not isinstance(value, list) else '/'.join(map(str, value))}"
                    for key, value in motif_result.filters.items()
                )
                st.caption(f"Applied filters: {filter_text}")

            query_metadata = motif_result.query_metadata.as_dict()
            st.markdown(
                f"**Query window:** {query_metadata.get('start_date')} → {query_metadata.get('end_date')} "
                f"(`regime={query_metadata.get('regime')}`)"
            )

            if not motif_result.matches:
                st.info("No motifs matched the configured filters.")
            else:
                rows = []
                score_label = (
                    "Distance" if motif_result.backend == "faiss" else "Similarity"
                )
                for idx, match in enumerate(motif_result.matches, start=1):
                    metadata = match.metadata
                    rows.append(
                        {
                            "Rank": idx,
                            "Ticker": metadata.get("ticker"),
                            "Timeframe": metadata.get("timeframe"),
                            "Start": metadata.get("start_date"),
                            "End": metadata.get("end_date"),
                            "Regime": metadata.get("regime"),
                            score_label: match.score,
                            "Window Index": metadata.get("window_index"),
                            "ID": match.motif_id,
                        }
                    )
                matches_df = pd.DataFrame(rows)
                st.dataframe(matches_df, width="stretch")
                csv_bytes = matches_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download price pattern matches (.csv)",
                    data=csv_bytes,
                    file_name=f"motif_matches_{result['params'].get('ticker', 'instrument')}.csv",
                    key="download-motifs",
                )
                with st.expander("Query metadata"):
                    st.json(query_metadata)
                with st.expander("Match metadata (raw)"):
                    st.json(motif_summary.get("matches", []))

    with weekly_tab:
        st.subheader("Weekly 30-EMA Overview")
        st.caption("Weekly closes with a 30-period EMA from Yahoo Finance data.")

        weekly_results: list[tuple[str, str, pd.DataFrame, str]] = []
        for ticker, label in WEEKLY_EMA_TICKERS:
            try:
                dataset, freshness = _load_weekly_ema_dataset(ticker)
            except Exception as exc:  # pragma: no cover - UI surface
                st.warning(f"Unable to load {ticker.upper()}: {exc}")
                continue

            if dataset.empty:
                st.info(f"No weekly data available for {label} ({ticker.upper()}).")
                continue

            weekly_results.append((ticker, label, dataset, freshness))

        if not weekly_results:
            st.info("Weekly datasets are unavailable right now. Try refreshing later.")
        else:
            cols: list[Any] = []
            for idx, (ticker, label, dataset, freshness) in enumerate(weekly_results):
                if idx % 2 == 0:
                    cols = st.columns(2)
                col = cols[idx % 2]
                chart_title = f"{label} ({ticker.upper()})"
                col.plotly_chart(
                    _build_weekly_ema_chart(dataset, title=chart_title),
                    use_container_width=True,
                )
                col.caption(f"Source: Yahoo Finance • Cache: {freshness}")

    with history_tab:
        st.subheader("Saved Analyses")
        history_df = HISTORY.as_dataframe(history_entries)
        if history_df.empty:
            st.info("Run the analysis to build your history.")
        else:
            st.dataframe(history_df, width="stretch")

    st.warning(
        "This application provides educational market analysis. It is not investment, "
        "trading, or financial advice. Review all insights independently before "
        "making decisions."
    )


def _format_metric(value: float, digits: int) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


if __name__ == "__main__":
    main()
