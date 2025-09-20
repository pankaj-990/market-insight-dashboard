# Market Technical Analysis Toolkit

A small toolkit for downloading OHLC data for any Yahoo Finance symbol, generating a structured technical summary, and (optionally) enriching it with an LLM-backed narrative. It ships with both a CLI workflow and an interactive Streamlit dashboard that share the same underlying services.

## Features
- Fetches OHLC candles from Yahoo Finance and enriches them with EMA/RSI indicators.
- Optional multi-timeframe workflow that blends higher timeframe context (e.g. daily) with intraday views (e.g. hourly).
- Builds a structured technical summary that is suitable for human review or prompting an LLM.
- Optional LLM integration via OpenRouter for narrative insights (CLI & Streamlit).
- Streamlit dashboard with candlestick visualisations, caching, history, and download helpers.
- Structured LLM analysis rendered as data tables for easier review.
- Reusable core library (`market_analysis`) for data access, caching, summaries, history persistence, and prompt generation.

## Getting Started
1. Create a virtual environment (recommended) and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Supply OpenRouter credentials if you intend to call the LLM:
   - Create `.streamlit/secrets.toml` (or use the Streamlit Cloud secrets UI) and set `OPENROUTER_API_KEY` (or `LLM_API_KEY`).
   - Optional overrides: `OPENROUTER_MODEL`, `OPENROUTER_TEMPERATURE`, `OPENROUTER_API_BASE`.
   - When running the CLI directly, exporting the same values as environment variables also works.

## Command-Line Workflow
```bash
python -m market_analysis.cli --ticker ^NSEI --period 1y --interval 1d --summary-only
```
Key flags:
- `--round-digits`: Rounding applied when caching new data.
- `--max-age-days`: Cache freshness window (`-1` disables the check).
- `--force-refresh`: Ignore cached data and refetch.
- `--lower-interval` / `--lower-period`: Enable multi-timeframe analysis (defaults to hourly over the last 60 days). Set `--lower-interval none` to disable.
- `--lower-key-window` / `--lower-recent-rows`: Control how many lower timeframe candles feed the summary (prompt output trimmed to ~12 rows).
- `--model` / `--temperature`: Override OpenRouter defaults.
- `--show-prompt`: Print the composed LLM prompt to stdout.
- `--as-of YYYY-MM-DD`: Analyse the state of the market as it stood on the specified date.
- `--run-dates YYYY-MM-DD [YYYY-MM-DD ...]`: Backfill multiple dates in one command.

## Streamlit Dashboard
```bash
streamlit run main.py
```
The dashboard lets you tweak fetch parameters, inspect the structured summary, download artefacts, and review prior analyses. Results are cached to `analysis_history.db` (alongside a Parquet file named after the ticker), so re-running with the same parameters skips unnecessary work.

## Architecture Overview
- `market_analysis/data.py`: Data fetching, indicator calculation, caching, and freshness checks.
- `market_analysis/summary.py`: Builds the structured technical summary used in prompts.
- `market_analysis/llm.py`: Prompt assembly and LangChain/OpenRouter integration.
- `market_analysis/history.py`: Lightweight SQLite-backed store for past analyses.
- `main.py`: Streamlit dashboard UI built on the shared services with additional UX helpers.
- `market_analysis/cli.py`: CLI entry point wiring arguments to the shared services.

## Analysis Flow
```mermaid
flowchart TD
    A[User opens Streamlit app] --> B[main()]
    B --> C[HISTORY.load()]
    B --> D[_load_default_params()]
    B --> E[render_sidebar()]
    E --> F{Run analysis submitted?}

    F -- No --> G[Show instructions & history table]
    F -- Yes --> H[handle_submission(params, history)]

    H --> I[lower_dataset_request / higher_dataset_request]
    I --> J{force_refresh?}
    J -- Yes --> K[ensure_dataset (higher & optional lower)]
    J -- No --> L[_load_dataset_cached()]
    K --> M[Dataframes + sources]
    L --> M

    M --> N[Build cache_key & lookup history]
    N --> O{Cached summary?}
    O -- Yes --> P[Reuse technical_summary]
    O -- No --> Q{Lower timeframe enabled?}
    Q -- Yes --> R[build_multi_timeframe_summary]
    Q -- No --> S[build_technical_summary]
    R --> T[technical_summary]
    S --> T
    P --> T

    T --> U{show_prompt enabled?}
    U -- Yes --> V[format_prompt preview]
    U -- No --> W[Skip prompt preview]
    V --> X[prompt_text]
    W --> X

    T --> Y{call_llm enabled?}
    Y -- No --> Z[Reuse prior LLM output if available]
    Y -- Yes --> AA{Cached LLM tables?}
    AA -- Yes --> AB[Reuse llm_text/tables]
    AA -- No --> AC[_run_llm()]
    AC --> AD{Credentials ok?}
    AD -- No --> AE[Record llm_error]
    AD -- Yes --> AF[generate_llm_analysis()]
    AF --> AG[markdown + structured tables]
    AB --> AH[llm_text/tables]
    Z --> AH
    AE --> AH
    AG --> AH

    X --> AI[Assemble result payload]
    AH --> AI
    M --> AI
    T --> AI

    AI --> AJ[HISTORY.record()]
    AJ --> AK[_store_result -> session_state]

    AK --> AL[main() renders metrics, charts, tables]
    AK --> AM[History updated for future reuse]
```

## Development Notes
- All modules are fully type-hinted; keep `from __future__ import annotations` when extending.
- `python -m compileall main.py market_analysis/cli.py market_analysis` is a quick syntax check.
- When adding new indicators or prompts, prefer extending the library modules (e.g. add to `DataFetcher` or create a new helper) so both the CLI and Streamlit front-ends benefit automatically.
