"""CLI entry point for fetching OHLC data and generating LLM-backed analysis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from market_analysis import (DatasetRequest, build_multi_timeframe_summary,
                             build_technical_summary, ensure_dataset,
                             format_prompt, generate_llm_analysis)
from market_analysis.llm import LLMConfig
from market_analysis.motifs import (DEFAULT_FEATURE_COLUMNS,
                                    SUPPORTED_NORMALIZATIONS, MotifQueryResult,
                                    generate_motif_matches_from_dataframe)


@dataclass(slots=True)
class _LoadedTimeframes:
    """Pre-loaded dataset state reused across multiple CLI runs."""

    higher_request: DatasetRequest
    higher_df: pd.DataFrame
    higher_source: str
    lower_request: DatasetRequest | None
    lower_df: pd.DataFrame | None
    lower_source: str | None


@dataclass(slots=True)
class MotifCLIOptions:
    """Parsed motif retrieval options gathered from CLI arguments."""

    enabled: bool = False
    backend: str = "faiss"
    window_size: int = 30
    top_k: int = 5
    features: Sequence[str] = DEFAULT_FEATURE_COLUMNS
    normalization: str = "zscore"
    default_regime: str = "auto"
    filter_ticker: Optional[str] = None
    filter_timeframe: Optional[str] = None
    filter_regimes: Optional[Sequence[str]] = None
    persist_directory: Optional[str] = None
    collection_name: Optional[str] = None

    def effective_filters(self, ticker: str, timeframe: str) -> Dict[str, Any]:
        filters: Dict[str, Any] = {
            "ticker": self.filter_ticker or ticker,
            "timeframe": self.filter_timeframe or timeframe,
        }
        if self.filter_regimes:
            if len(self.filter_regimes) == 1:
                filters["regime"] = self.filter_regimes[0]
            else:
                filters["regime"] = list(self.filter_regimes)
        return filters


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

    motif = parser.add_argument_group("Motif retrieval")
    motif.add_argument(
        "--motif-enable",
        action="store_true",
        help="Index historical windows and retrieve nearest motifs for the latest window.",
    )
    motif.add_argument(
        "--motif-backend",
        choices=("faiss", "chroma"),
        default="faiss",
        help="Vector store backend to use when motif retrieval is enabled (default: %(default)s).",
    )
    motif.add_argument(
        "--motif-window-size",
        type=int,
        default=30,
        help="Number of candles per motif window (default: %(default)s).",
    )
    motif.add_argument(
        "--motif-top-k",
        type=int,
        default=5,
        help="Number of similar motifs to display (default: %(default)s).",
    )
    motif.add_argument(
        "--motif-features",
        default=",".join(DEFAULT_FEATURE_COLUMNS),
        help=(
            "Comma-separated list of features to include in motif vectors. "
            "Defaults to OHLC plus EMA/RSI indicators."
        ),
    )
    motif.add_argument(
        "--motif-normalization",
        choices=SUPPORTED_NORMALIZATIONS,
        default="zscore",
        help="Normalisation strategy applied within each window (default: %(default)s).",
    )
    motif.add_argument(
        "--motif-default-regime",
        default="auto",
        help="Regime label applied when automatic inference is unavailable (default: %(default)s).",
    )
    motif.add_argument(
        "--motif-filter-ticker",
        help="Restrict motif search to this ticker metadata (default: analysed ticker).",
    )
    motif.add_argument(
        "--motif-filter-timeframe",
        help="Restrict motif search to this timeframe metadata (default: analysed interval).",
    )
    motif.add_argument(
        "--motif-filter-regime",
        help="Restrict motif search to one or more regime labels (comma-separated).",
    )
    motif.add_argument(
        "--motif-persist-dir",
        help="Directory for persisting Chroma collections (ignored for FAISS).",
    )
    motif.add_argument(
        "--motif-collection",
        help="Chroma collection name (default: 'motifs').",
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_parser().parse_args()


def _parse_feature_list(value: str) -> Sequence[str]:
    if not value:
        return tuple()
    parts = [item.strip() for item in value.split(",")]
    return tuple(item for item in parts if item)


def _parse_regime_filter(value: Optional[str]) -> Optional[Sequence[str]]:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return None
    # Preserve input order while removing duplicates.
    unique: List[str] = list(dict.fromkeys(items))
    return tuple(unique)


def _parse_motif_options(args: argparse.Namespace) -> MotifCLIOptions:
    if not getattr(args, "motif_enable", False):
        return MotifCLIOptions(enabled=False)

    features = _parse_feature_list(getattr(args, "motif_features", ""))
    if not features:
        features = DEFAULT_FEATURE_COLUMNS

    normalization = getattr(args, "motif_normalization", "zscore") or "none"
    default_regime = getattr(args, "motif_default_regime", "auto") or "auto"

    options = MotifCLIOptions(
        enabled=True,
        backend=getattr(args, "motif_backend", "faiss"),
        window_size=max(2, int(getattr(args, "motif_window_size", 30))),
        top_k=max(1, int(getattr(args, "motif_top_k", 5))),
        features=features,
        normalization=normalization,
        default_regime=default_regime,
        filter_ticker=getattr(args, "motif_filter_ticker", None),
        filter_timeframe=getattr(args, "motif_filter_timeframe", None),
        filter_regimes=_parse_regime_filter(getattr(args, "motif_filter_regime", None)),
        persist_directory=getattr(args, "motif_persist_dir", None),
        collection_name=getattr(args, "motif_collection", None) or "motifs",
    )

    return options


def run_cli(args: argparse.Namespace) -> None:
    """Execute the workflow according to the provided CLI arguments."""
    run_dates = _resolve_run_dates(args)

    try:
        timeframes = _load_timeframes(args)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to load required datasets: {exc}")
        return

    motif_options = _parse_motif_options(args)

    print(
        "Using {source} data from {path} for interval {interval}".format(
            source=timeframes.higher_source,
            path=timeframes.higher_request.resolved_path(),
            interval=args.interval,
        )
    )
    if (
        timeframes.lower_df is not None
        and timeframes.lower_source
        and timeframes.lower_request is not None
    ):
        print(
            "Using {source} data from {path} for interval {interval}".format(
                source=timeframes.lower_source,
                path=timeframes.lower_request.resolved_path(),
                interval=timeframes.lower_request.interval,
            )
        )

    for index, as_of in enumerate(run_dates):
        if index:
            print("\n" + "=" * 72 + "\n")
        try:
            _run_single_analysis(args, as_of, timeframes, motif_options)
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


def _load_timeframes(args: argparse.Namespace) -> _LoadedTimeframes:
    """Fetch higher/lower timeframe datasets once per CLI invocation."""
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

    return _LoadedTimeframes(
        higher_request=higher_request,
        higher_df=higher_df,
        higher_source=higher_source,
        lower_request=lower_request,
        lower_df=lower_df,
        lower_source=lower_source,
    )


def _run_single_analysis(
    args: argparse.Namespace,
    as_of: Optional[date],
    timeframes: _LoadedTimeframes,
    motif_options: MotifCLIOptions,
) -> None:
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
        return trimmed.copy()

    higher_df = _trim_to_as_of(timeframes.higher_df, "higher timeframe")

    lower_request = timeframes.lower_request
    lower_df: pd.DataFrame | None = None
    lower_source = timeframes.lower_source
    if lower_request is not None and timeframes.lower_df is not None:
        lower_df = _trim_to_as_of(timeframes.lower_df, "lower timeframe")

    if lower_df is not None:
        summary = build_multi_timeframe_summary(
            higher_df,
            lower_df,
            higher_label=f"Higher Timeframe ({args.interval})",
            lower_label=f"Lower Timeframe ({lower_request.interval})",
            lower_key_window=args.lower_key_window,
            lower_recent_rows=args.lower_recent_rows,
        )
    else:
        summary = build_technical_summary(higher_df)
    llm_config: Optional[LLMConfig] = None
    llm_text: Optional[str] = None

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

        print(llm_text)

    if motif_options.enabled:
        try:
            _run_motif_cli(
                higher_df,
                ticker=args.ticker,
                timeframe=args.interval,
                options=motif_options,
            )
        except RuntimeError as exc:
            print(f"[Motifs] {exc}")

    params_payload = {
        "ticker": args.ticker,
        "period": args.period,
        "interval": args.interval,
        "round_digits": args.round_digits,
        "data_path": str(timeframes.higher_request.resolved_path()),
        "max_age_days": args.max_age_days,
        "as_of": as_of.isoformat() if as_of is not None else None,
    }

    if lower_df is not None and lower_request is not None:

        params_payload.update(
            {
                "lower_period": lower_request.period,
                "lower_interval": lower_request.interval,
                "lower_data_path": str(lower_request.resolved_path()),
                "lower_key_window": args.lower_key_window,
                "lower_recent_rows": args.lower_recent_rows,
            }
        )

    data_source_payload: dict[str, dict[str, str]] = {
        "higher": {
            "interval": args.interval,
            "source": timeframes.higher_source,
            "path": str(timeframes.higher_request.resolved_path()),
        }
    }
    if lower_df is not None and lower_source and lower_request is not None:
        data_source_payload["lower"] = {
            "interval": lower_request.interval,
            "source": lower_source,
            "path": str(lower_request.resolved_path()),
        }


def _run_motif_cli(
    higher_df: pd.DataFrame,
    *,
    ticker: str,
    timeframe: str,
    options: MotifCLIOptions,
) -> MotifQueryResult:
    normalization = options.normalization if options.normalization != "none" else None
    try:
        result = generate_motif_matches_from_dataframe(
            higher_df,
            window_size=options.window_size,
            feature_columns=options.features,
            ticker=ticker,
            timeframe=timeframe,
            top_k=options.top_k,
            backend=options.backend,
            default_regime=options.default_regime,
            normalization=normalization,
            metadata_filter=options.effective_filters(ticker, timeframe),
            persist_directory=(
                options.persist_directory if options.backend == "chroma" else None
            ),
            collection_name=(
                options.collection_name if options.backend == "chroma" else None
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(str(exc)) from exc

    _print_motif_result(result, options)
    return result


def _print_motif_result(result: MotifQueryResult, options: MotifCLIOptions) -> None:
    print(
        "\n=== Motif Retrieval ({backend}) ===".format(backend=options.backend.upper())
    )
    query_meta = result.query_metadata
    print(
        "Query window: {start} -> {end} | regime={regime}".format(
            start=query_meta.start_date,
            end=query_meta.end_date,
            regime=query_meta.regime,
        )
    )
    print(
        "Window size: {size} | Features: {features} | Normalization: {norm}".format(
            size=result.window_size,
            features=", ".join(result.feature_columns),
            norm=result.normalization,
        )
    )
    if result.skipped_windows:
        print(
            f"Skipped {result.skipped_windows} window(s) due to missing values or normalization issues."
        )
    print(f"Indexed windows: {result.indexed_count}")
    if result.filters:
        filter_parts = []
        for key, value in result.filters.items():
            if isinstance(value, list):
                rendered = "/".join(str(item) for item in value)
            else:
                rendered = str(value)
            filter_parts.append(f"{key}={rendered}")
        print("Filters: " + ", ".join(filter_parts))

    if not result.matches:
        print("No motifs matched the requested filters.")
        return

    score_label = "distance" if options.backend == "faiss" else "similarity"
    print(f"Top {len(result.matches)} motifs:")
    for idx, match in enumerate(result.matches, start=1):
        meta = match.metadata
        ticker_label = meta.get("ticker", "?")
        start_date = meta.get("start_date", "?")
        end_date = meta.get("end_date", "?")
        regime = meta.get("regime", "?")
        motif_id = match.motif_id
        print(
            "  {rank:>2}. {ticker} {start}->{end} | regime={regime} | {label}={score:.4f} | id={motif_id}".format(
                rank=idx,
                ticker=ticker_label,
                start=start_date,
                end=end_date,
                regime=regime,
                label=score_label,
                score=match.score,
                motif_id=motif_id,
            )
        )


def main() -> None:
    """CLI entry point used by ``python -m`` or direct execution."""
    args = parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()
