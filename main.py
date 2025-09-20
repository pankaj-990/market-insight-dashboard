"""Streamlit UI for the market technical analysis workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_analysis import (
    AnalysisHistory,
    DatasetRequest,
    build_multi_timeframe_summary,
    build_technical_summary,
    create_playbook_builder,
    ensure_dataset,
    format_prompt,
    generate_llm_analysis,
    make_cache_key,
    make_entry_id,
)
from market_analysis.data import default_data_path
from market_analysis.llm import LLMConfig

st.set_page_config(
    page_title="Market Technical Analysis Dashboard",
    page_icon="📈",
    layout="wide",
)

HISTORY = AnalysisHistory(Path("analysis_history.db"))


PERIOD_OPTIONS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"]
INTERVAL_OPTIONS = ["1d", "1wk", "1mo"]
LOWER_INTERVAL_OPTIONS = ["30m", "1h", "2h", "4h", "1d"]
LOWER_PERIOD_SUGGESTIONS = ["30d", "60d", "90d", "180d", "1y"]

SUMMARY_HEADING_RE = re.compile(r"^[A-Z0-9 /()_-]+:?$")


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


def _render_playbook_markdown(plan_text: str) -> None:
    """Render the LLM playbook output with nicer Streamlit formatting."""
    lines = plan_text.splitlines()
    sections: list[tuple[str, str]] = []
    current_title: Optional[str] = None
    buffer: list[str] = []

    def _flush() -> None:
        nonlocal buffer, current_title, sections
        if current_title is None:
            return
        content = "\n".join(buffer).strip()
        sections.append((current_title, content))
        buffer = []

    for raw_line in lines:
        line = raw_line.rstrip()
        if line.startswith("## "):
            _flush()
            current_title = line[3:].strip()
        else:
            if current_title is None and line:
                current_title = "Playbook"
            buffer.append(line)
    _flush()

    if not sections:
        st.markdown(plan_text)
        return

    for title, content in sections:
        if not title:
            title = "Playbook"
        st.markdown(f"### {title}")
        if content:
            st.markdown(content)
        else:
            st.info("No guidance provided for this section.")


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
    period: str = "1y"
    interval: str = "1d"
    include_lower: bool = True
    lower_period: str = "60d"
    lower_interval: str = "1h"
    lower_key_window: int = 30
    lower_recent_rows: int = 20
    round_digits: int = 2
    max_age_days: int = 3
    force_refresh: bool = False
    call_llm: bool = True
    show_prompt: bool = False

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
            lower_recent_rows=int(state.get("lower_recent_rows", 20)),
            round_digits=int(state.get("round_digits", 2)),
            max_age_days=int(state.get("max_age_days", 3)),
            force_refresh=bool(state.get("force_refresh", False)),
            call_llm=bool(state.get("call_llm", True)),
            show_prompt=bool(state.get("show_prompt", False)),
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
            }
        )
        return payload


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
                lower_interval = st.selectbox(
                    "Lower interval",
                    lower_interval_options,
                    index=_safe_index(lower_interval_options, defaults.lower_interval),
                )
                lower_period = st.text_input(
                    "Lower period",
                    value=str(defaults.lower_period),
                    help="Lower timeframe lookback (e.g. 30d, 60d, 1y).",
                )
                lower_key_window = int(
                    st.number_input(
                        "Lower key window (candles)",
                        min_value=10,
                        max_value=200,
                        value=int(defaults.lower_key_window),
                        help="Number of lower timeframe candles considered for highs and lows (flattened at 12-row summary).",
                    )
                )
                lower_recent_rows = int(
                    st.number_input(
                        "Lower recent rows",
                        min_value=10,
                        max_value=60,
                        value=int(defaults.lower_recent_rows),
                        help="Rows from the lower timeframe included in the summary table (trimmed to 12 for the prompt).",
                    )
                )
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
    reused_playbook = False
    playbook_payload: Optional[Dict[str, Any]] = None
    playbook_error: Optional[str] = None

    if existing_entry:
        technical_summary = existing_entry["technical_summary"]
        reused_summary = True
        playbook_payload = existing_entry.get("playbook")
        reused_playbook = bool(playbook_payload)
        playbook_error = existing_entry.get("playbook_error")
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

    playbook_builder = None

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

    if not reused_playbook:
        playbook_builder = create_playbook_builder()
        if playbook_builder:
            for warning in (
                getattr(playbook_builder, "embedding_warning", None),
                getattr(playbook_builder, "index_warning", None),
            ):
                if warning:
                    playbook_error = (
                        warning
                        if not playbook_error
                        else f"{warning}; {playbook_error}"
                    )
            try:
                playbook_result = playbook_builder.generate_playbook(
                    technical_summary=technical_summary,
                    params=params.history_payload(),
                    cache_key=cache_key,
                )
                playbook_payload = playbook_result.to_payload()
            except Exception as exc:
                playbook_error = f"Unable to build playbook: {exc}"
        elif existing_entry and not playbook_payload:
            playbook_error = existing_entry.get("playbook_error")

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
        "playbook": playbook_payload,
        "playbook_error": playbook_error,
    }
    HISTORY.record(history_entry)

    if playbook_builder:
        try:
            playbook_builder.upsert_history_entry(history_entry)
        except Exception as exc:
            message = f"Failed to persist playbook entry: {exc}"
            playbook_error = (
                f"{playbook_error}; {message}" if playbook_error else message
            )

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
        "reused_playbook": reused_playbook,
        "playbook": playbook_payload,
        "playbook_error": playbook_error,
        "entry_id": cache_entry_id,
        "higher_last_timestamp": higher_last_timestamp,
        "lower_last_timestamp": lower_last_timestamp,
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
    st.plotly_chart(figure, use_container_width=True)


def main() -> None:
    st.title("Market Technical Analysis Dashboard")
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
            st.dataframe(history_df, use_container_width=True)
        return

    higher_df = result["higher_df"]
    lower_df = result.get("lower_df")
    round_digits = result["params"]["round_digits"]
    data_sources = result.get("data_sources", {})
    higher_info = data_sources.get("higher", {})
    lower_info = data_sources.get("lower")

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
    if result.get("reused_playbook"):
        status_notes.append("Reused cached playbook")
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

    summary_tab, data_tab, llm_tab, playbook_tab, history_tab = st.tabs(
        [
            "Technical Summary",
            "Recent Candles",
            "LLM Analysis",
            "Playbook Insights",
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

    with playbook_tab:
        playbook_info = result.get("playbook")
        playbook_error_msg = result.get("playbook_error")
        if playbook_error_msg:
            st.warning(playbook_error_msg)
        elif not playbook_info:
            st.info(
                "Playbook suggestions will appear once enough historical analyses are "
                "indexed and embeddings are configured."
            )
        else:
            plan_text = playbook_info.get("plan")
            cases = playbook_info.get("cases", [])
            if plan_text:
                _render_playbook_markdown(plan_text)
                st.download_button(
                    "Download playbook (.md)",
                    plan_text,
                    file_name=f"playbook_{result['params']['ticker']}.md",
                    key="download-playbook",
                )
            else:
                st.info("No comparable historical cases were available for this run.")
            if cases:
                case_rows = []
                for case in cases:
                    similarity_value = case.get("similarity")
                    case_rows.append(
                        {
                            "Case": f"#{case['rank']}",
                            "Ticker": case.get("ticker"),
                            "Period": case.get("period"),
                            "Interval": case.get("interval"),
                            "Lower Interval": case.get("lower_interval"),
                            "Recorded": case.get("recorded"),
                            "Last Candle": case.get("last_timestamp"),
                            "Lower Last Candle": case.get("lower_last_timestamp"),
                            "Similarity": (
                                f"{similarity_value:.3f}"
                                if isinstance(similarity_value, (int, float))
                                else "—"
                            ),
                            "Snippet": case.get("summary_snippet"),
                        }
                    )
                cases_df = pd.DataFrame(case_rows)
                st.dataframe(cases_df, use_container_width=True)
                st.caption(
                    "Similarity scores derive from vector distance (lower is closer)."
                )
            if result.get("reused_playbook"):
                st.caption("Reused cached playbook insights.")

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
        elif result["llm_text"]:
            tables = result.get("llm_tables") or {}
            section_titles = {
                "overall_trend": "Overall Trend Assessment",
                "key_evidence": "Key Technical Evidence",
                "candlestick_patterns": "Candlestick Pattern Analysis",
                "chart_patterns": "Chart Pattern Analysis",
                "trade_plan": "Trade Plan Outline",
            }
            rendered = False
            for key, title in section_titles.items():
                section = tables.get(key)
                if not section:
                    continue
                headers = section.get("headers") or []
                rows = section.get("rows") or []
                if not headers:
                    continue
                st.markdown(f"**{title}**")
                st.table(pd.DataFrame(rows, columns=headers))
                rendered = True
            if not rendered:
                st.markdown(result["llm_text"])
            else:
                with st.expander("Markdown output"):
                    st.markdown(result["llm_text"])
        elif result["llm_error"]:
            st.warning(result["llm_error"])
        else:
            st.info("LLM analysis not available.")

        if result.get("llm_error") and result.get("llm_text"):
            st.caption("Warning: LLM reported an issue while generating this output.")

    with history_tab:
        st.subheader("Saved Analyses")
        history_df = HISTORY.as_dataframe(history_entries)
        if history_df.empty:
            st.info("Run the analysis to build your history.")
        else:
            st.dataframe(history_df, use_container_width=True)

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
