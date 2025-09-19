"""CLI entry point for fetching OHLC data and generating LLM-backed analysis."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd

from market_analysis import (
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
from market_analysis.llm import LLMConfig


def _parse_date(value: str) -> date:
    """Return a ``date`` parsed from YYYY-MM-DD strings."""
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use the YYYY-MM-DD format."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Fetch OHLC data (if needed) and generate a technical analysis report."
    )
    parser.add_argument(
        "--ticker",
        default="^NSEI",
        help="Yahoo Finance ticker symbol (default: %(default)s)",
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="Lookback period accepted by yfinance (default: %(default)s)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Sampling interval accepted by yfinance (default: %(default)s)",
    )
    parser.add_argument(
        "--lower-interval",
        default="1h",
        help=(
            "Lower timeframe sampling interval (default: %(default)s). "
            "Set to 'none' to disable multi-timeframe analysis."
        ),
    )
    parser.add_argument(
        "--lower-period",
        default="60d",
        help=(
            "Lookback period for the lower timeframe (default: %(default)s). "
            "Ignored when --lower-interval is 'none'."
        ),
    )
    parser.add_argument(
        "--lower-key-window",
        default=30,
        type=int,
        help=(
            "Recent high/low window measured in lower timeframe candles "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--lower-recent-rows",
        default=20,
        type=int,
        help=(
            "Number of lower timeframe candles to include in the summary table "
            "(default: %(default)s, capped at 12 in the prompt)."
        ),
    )
    parser.add_argument(
        "--round-digits",
        default=2,
        type=int,
        help="Round prices/indicators to this many decimals (default: %(default)s)",
    )
    parser.add_argument(
        "--max-age-days",
        default=3,
        type=int,
        help="Maximum acceptable data age before forcing a refresh (default: %(default)s)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached data and fetch a fresh dataset.",
    )
    parser.add_argument(
        "--model",
        help="Override the OpenRouter model identifier.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override the LLM sampling temperature.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the technical summary instead of calling the LLM.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the full LLM prompt that will be sent to the model.",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_date,
        help="Analyse using data available up to this YYYY-MM-DD date.",
    )
    parser.add_argument(
        "--run-dates",
        nargs="+",
        type=_parse_date,
        help="Run analyses for one or more YYYY-MM-DD dates (space-separated).",
    )
    parser.add_argument(
        "--show-playbook",
        action="store_true",
        help="Display strategy playbook insights built via retrieval.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_parser().parse_args()


def run_cli(args: argparse.Namespace) -> None:
    """Execute the workflow according to the provided CLI arguments."""
    run_dates = _resolve_run_dates(args)

    for index, as_of in enumerate(run_dates):
        if index:
            print("\n" + "=" * 72 + "\n")
        try:
            _run_single_analysis(args, as_of)
        except Exception as exc:  # pragma: no cover - defensive
            label = as_of.isoformat() if as_of is not None else "latest"
            print(f"Failed to analyse {args.ticker} for {label}: {exc}")


def configure_llm(args: argparse.Namespace) -> LLMConfig:
    """Return an ``LLMConfig`` optionally overridden by CLI arguments."""
    config = LLMConfig.from_env()
    if args.model is not None:
        config = replace(config, model=args.model)
    if args.temperature is not None:
        config = replace(config, temperature=args.temperature)
    return config


def _print_prompt(summary: str) -> None:
    prompt_text = format_prompt(summary)
    border = "=" * 16
    prompt_lines = [
        f"{border} LLM PROMPT {border}",
        prompt_text,
        f"{border} END PROMPT {border}",
    ]
    print("\n" + "\n".join(prompt_lines) + "\n")


def _resolve_run_dates(args: argparse.Namespace) -> list[Optional[date]]:
    """Return a sequence of dates to analyse (None signifies latest)."""
    if args.run_dates:
        if args.as_of:
            print("Ignoring --as-of because --run-dates was provided.")
        return list(args.run_dates)
    if args.as_of:
        return [args.as_of]
    return [None]


def _run_single_analysis(args: argparse.Namespace, as_of: Optional[date]) -> None:
    """Execute a single analysis for ``as_of`` (or latest when ``None``)."""
    header_label = as_of.isoformat() if as_of is not None else "latest"
    print(f"=== Analysis for {args.ticker} (as of {header_label}) ===")

    def _trim_to_as_of(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        if as_of is None:
            return frame
        cutoff = pd.Timestamp(as_of)
        index_tz = getattr(frame.index, "tz", None)
        if index_tz is not None and cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(index_tz)
        trimmed = frame.loc[:cutoff]
        if trimmed.empty:
            raise ValueError(
                f"No {label} candles available on or before the requested as-of date. "
                f"Increase the --period for that timeframe or choose a later date."
            )
        return trimmed

    higher_request = DatasetRequest(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        round_digits=args.round_digits,
        max_age_days=args.max_age_days,
    )
    higher_df, higher_source = ensure_dataset(
        higher_request, force_refresh=args.force_refresh
    )
    higher_df = _trim_to_as_of(higher_df, "higher timeframe")

    lower_interval_value = (args.lower_interval or "").strip()
    use_lower = lower_interval_value and lower_interval_value.lower() != "none"
    lower_request: DatasetRequest | None = None
    lower_df: pd.DataFrame | None = None
    lower_source: str | None = None

    if use_lower:
        lower_period_value = (args.lower_period or args.period).strip()
        lower_request = DatasetRequest(
            ticker=args.ticker,
            period=lower_period_value,
            interval=lower_interval_value,
            round_digits=args.round_digits,
            max_age_days=args.max_age_days,
        )
        lower_df, lower_source = ensure_dataset(
            lower_request, force_refresh=args.force_refresh
        )
        lower_df = _trim_to_as_of(lower_df, "lower timeframe")

    print(
        f"Using {higher_source} data from {higher_request.resolved_path()} for interval {args.interval}"
    )
    if lower_df is not None and lower_source and lower_request is not None:
        print(
            f"Using {lower_source} data from {lower_request.resolved_path()} for interval {lower_request.interval}"
        )

    if lower_df is not None:
        summary = build_multi_timeframe_summary(
            higher_df,
            lower_df,
            higher_label=f"Higher Timeframe ({args.interval})",
            lower_label=f"Lower Timeframe ({lower_interval_value})",
            lower_key_window=args.lower_key_window,
            lower_recent_rows=args.lower_recent_rows,
        )
    else:
        summary = build_technical_summary(higher_df)
    llm_config: Optional[LLMConfig] = None
    llm_text: Optional[str] = None
    llm_tables: Optional[dict[str, dict[str, Any]]] = None
    if args.summary_only:
        print(summary)
        if args.show_prompt:
            _print_prompt(summary)
    else:
        if args.show_prompt:
            _print_prompt(summary)
        llm_config = configure_llm(args)
        llm_result = generate_llm_analysis(summary, llm_config)
        llm_text = llm_result.markdown
        llm_tables = llm_result.tables
        print(llm_text)

    params_payload = {
        "ticker": args.ticker,
        "period": args.period,
        "interval": args.interval,
        "round_digits": args.round_digits,
        "data_path": str(higher_request.resolved_path()),
        "max_age_days": args.max_age_days,
        "as_of": as_of.isoformat() if as_of is not None else None,
    }
    lower_last_timestamp: str | None = None
    if lower_df is not None and lower_request is not None:
        lower_last_timestamp = lower_df.index[-1].to_pydatetime().isoformat()
        params_payload.update(
            {
                "lower_period": lower_request.period,
                "lower_interval": lower_request.interval,
                "lower_data_path": str(lower_request.resolved_path()),
                "lower_key_window": args.lower_key_window,
                "lower_recent_rows": args.lower_recent_rows,
            }
        )
    higher_last_timestamp = higher_df.index[-1].to_pydatetime().isoformat()
    cache_key = make_cache_key(
        params_payload, higher_last_timestamp, lower_last_timestamp
    )
    timestamp = datetime.utcnow().isoformat()
    entry_id = make_entry_id(cache_key)

    data_source_payload: dict[str, dict[str, str]] = {
        "higher": {
            "interval": args.interval,
            "source": higher_source,
            "path": str(higher_request.resolved_path()),
        }
    }
    if lower_df is not None and lower_source and lower_request is not None:
        data_source_payload["lower"] = {
            "interval": lower_request.interval,
            "source": lower_source,
            "path": str(lower_request.resolved_path()),
        }

    history_entry = {
        "timestamp": timestamp,
        "cache_key": cache_key,
        "params": params_payload,
        "data_source": data_source_payload,
        "technical_summary": summary,
        "llm_text": llm_text,
        "llm_error": None,
        "entry_id": entry_id,
        "llm_tables": llm_tables,
    }

    playbook_builder = create_playbook_builder(llm_config)
    embedding_warning = None
    if playbook_builder is not None:
        embedding_warning = getattr(playbook_builder, "embedding_warning", None)
    playbook_result = None
    playbook_error: Optional[str] = None

    if playbook_builder:
        if args.show_playbook:
            try:
                playbook_result = playbook_builder.generate_playbook(
                    technical_summary=summary,
                    params=params_payload,
                    cache_key=cache_key,
                )
            except Exception as exc:  # pragma: no cover - defensive
                playbook_error = f"Playbook generation failed: {exc}"
        try:
            playbook_builder.upsert_history_entry(history_entry)
        except Exception as exc:  # pragma: no cover - defensive
            message = f"Playbook indexing failed: {exc}"
            playbook_error = (
                f"{playbook_error}; {message}" if playbook_error else message
            )
    elif args.show_playbook:
        playbook_error = (
            "Playbook unavailable: ensure OpenRouter credentials and embedding "
            "configuration are set."
        )

    if embedding_warning:
        playbook_error = (
            embedding_warning
            if not playbook_error
            else f"{embedding_warning}; {playbook_error}"
        )

    if args.show_playbook:
        if playbook_result and playbook_result.plan:
            print("\n=== STRATEGY PLAYBOOK ===\n")
            print(playbook_result.plan)
            if playbook_result.cases:
                print("\nReferenced historical cases:")
                for case in playbook_result.cases:
                    score_text = (
                        f"{case.similarity:.3f}" if case.similarity is not None else "—"
                    )
                    print(
                        f"  - Case {case.rank}: {case.ticker or 'N/A'} "
                        f"({case.period}/{case.interval}) recorded {case.recorded} "
                        f"score={score_text}"
                    )
        elif playbook_error:
            print(f"Playbook error: {playbook_error}")
        else:
            print("Playbook not available yet (insufficient indexed history).")


def main() -> None:
    """CLI entry point used by ``python -m`` or direct execution."""
    args = parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()
