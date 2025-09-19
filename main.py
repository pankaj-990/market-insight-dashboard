"""Streamlit UI for the market technical analysis workflow."""

from __future__ import annotations

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

HISTORY = AnalysisHistory(Path("analysis_history.json"))


PERIOD_OPTIONS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"]
INTERVAL_OPTIONS = ["1d", "1wk", "1mo"]


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


@dataclass(slots=True)
class AnalysisParams:
    """User-selected configuration captured from the sidebar form."""

    ticker: str = "^NSEI"
    period: str = "1y"
    interval: str = "1d"
    round_digits: int = 2
    max_age_days: int = 3
    force_refresh: bool = False
    call_llm: bool = True
    show_prompt: bool = False

    @property
    def data_path(self) -> Path:
        return default_data_path(self.ticker)

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "AnalysisParams":
        """Hydrate an ``AnalysisParams`` instance from session state."""
        return cls(
            ticker=state.get("ticker", "^NSEI"),
            period=state.get("period", "1y"),
            interval=state.get("interval", "1d"),
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
            "round_digits": self.round_digits,
            "max_age_days": self.max_age_days,
            "data_path": str(self.data_path),
            "force_refresh": self.force_refresh,
            "call_llm": self.call_llm,
            "show_prompt": self.show_prompt,
        }

    def dataset_request(self) -> DatasetRequest:
        """Translate the params into a :class:`DatasetRequest`."""
        return DatasetRequest(
            ticker=self.ticker,
            period=self.period,
            interval=self.interval,
            round_digits=self.round_digits,
            max_age_days=self.max_age_days,
        )

    def cache_payload(self) -> Dict[str, Any]:
        """Payload used when building a cache key."""
        return {
            "ticker": self.ticker,
            "period": self.period,
            "interval": self.interval,
            "round_digits": self.round_digits,
            "data_path": str(self.data_path),
        }

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
            st.caption(f"Cache file: {default_data_path(ticker)}")
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
    with st.spinner("Loading market data..."):
        try:
            df, data_source = ensure_dataset(
                params.dataset_request(), force_refresh=params.force_refresh
            )
        except Exception as exc:
            _store_error(f"Unable to load data: {exc}")
            return

    last_timestamp = df.index[-1].to_pydatetime().isoformat()
    cache_key = make_cache_key(params.cache_payload(), last_timestamp)
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
        if existing_entry.get("playbook"):
            playbook_payload = existing_entry.get("playbook")
            reused_playbook = True
        if existing_entry.get("playbook_error"):
            playbook_error = existing_entry.get("playbook_error")
    else:
        try:
            technical_summary = build_technical_summary(df)
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
            if playbook_builder.embedding_warning and playbook_error is None:
                playbook_error = playbook_builder.embedding_warning
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

    history_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "cache_key": cache_key,
        "params": params.history_payload(),
        "data_source": data_source,
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
            # We already surface the generation error above; indexing failures are non-fatal.
            message = f"Failed to persist playbook entry: {exc}"
            playbook_error = (
                f"{playbook_error}; {message}" if playbook_error else message
            )

    result_payload = {
        "df": df,
        "data_source": data_source,
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

    df = result["df"]
    round_digits = result["params"]["round_digits"]

    st.success(
        f"Using {result['data_source']} data for `{result['params']['ticker']}` "
        f"from {result['params']['data_path']}"
    )

    status_notes = []
    if result.get("reused_summary"):
        status_notes.append("Reused cached technical summary")
    if result.get("reused_llm"):
        status_notes.append("Reused cached LLM analysis")
    if result.get("reused_playbook"):
        status_notes.append("Reused cached playbook")
    if status_notes:
        st.caption(" | ".join(status_notes))

    latest_close = df["Close"].iloc[-1]
    ema_30 = df["EMA_30"].iloc[-1]
    ema_200 = df["EMA_200"].iloc[-1]
    rsi_14 = df["RSI_14"].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close", _format_metric(latest_close, round_digits))
    col2.metric("EMA 30", _format_metric(ema_30, round_digits))
    col3.metric("EMA 200", _format_metric(ema_200, round_digits))
    col4.metric("RSI 14", _format_metric(rsi_14, round_digits))

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
        st.subheader("Structured Summary")
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
        st.subheader("Strategy Playbook")
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
                            "Recorded": case.get("recorded"),
                            "Last Candle": case.get("last_timestamp"),
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
        st.subheader("Candlestick View")
        _plot_candlestick(df)
        st.divider()
        st.subheader("Latest Candles")
        st.dataframe(df.tail(20))
        st.download_button(
            "Download data (.csv)",
            df.to_csv().encode("utf-8"),
            file_name=f"ohlc_{result['params']['ticker']}.csv",
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


def _format_metric(value: float, digits: int) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


if __name__ == "__main__":
    main()
