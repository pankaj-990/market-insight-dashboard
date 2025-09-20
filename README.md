# Market Technical Analysis Toolkit

A small toolkit for downloading OHLC data for any Yahoo Finance symbol, generating a structured technical summary, and (optionally) enriching it with an LLM-backed narrative. It ships with both a CLI workflow and an interactive Streamlit dashboard that share the same underlying services.

## Features
- Fetches OHLC candles from Yahoo Finance and enriches them with EMA/RSI indicators.
- Optional multi-timeframe workflow that blends higher timeframe context (e.g. daily) with intraday views (e.g. hourly).
- Builds a structured technical summary that is suitable for human review or prompting an LLM.
- Optional LLM integration via OpenRouter for narrative insights (CLI & Streamlit).
- Streamlit dashboard with candlestick visualisations, caching, history, and download helpers.
- Retrieval-augmented *Strategy Playbook* that surfaces comparable historical setups and suggests trades.
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
- `--show-playbook`: Display retrieval-backed playbook insights (requires embeddings + indexed history).
- `--as-of YYYY-MM-DD`: Analyse the state of the market as it stood on the specified date.
- `--run-dates YYYY-MM-DD [YYYY-MM-DD ...]`: Backfill multiple dates in one command (useful for seeding the playbook index).

## Streamlit Dashboard
```bash
streamlit run main.py
```
The dashboard lets you tweak fetch parameters, inspect the structured summary, download artefacts, and review prior analyses. Results are cached to `analysis_history.db` (alongside a Parquet file named after the ticker), so re-running with the same parameters skips unnecessary work.

### Strategy Playbook (RAG)
- Set `OPENROUTER_API_KEY` (or `LLM_API_KEY`) plus optional overrides `PLAYBOOK_EMBED_MODEL`, `PLAYBOOK_TOP_K`, `PLAYBOOK_INDEX_PATH`, and `PLAYBOOK_TEMPERATURE`.
- After a few analyses are stored, the Streamlit “Playbook Insights” tab (or CLI `--show-playbook`) retrieves similar historical cases and drafts trade ideas.
- Historical entries are automatically indexed into a FAISS vector store located at `playbook_index/` by default. Install `faiss-cpu` (already listed in `requirements.txt`) to enable the index. If remote embeddings are unavailable (e.g., OpenRouter does not support them), the app falls back to a lightweight hash-based embedding. You can force this with `PLAYBOOK_EMBED_BACKEND=hash` or stick with OpenAI-compatible embeddings via `PLAYBOOK_EMBED_BACKEND=openai` and a supported model (`PLAYBOOK_EMBED_MODEL`).

## Architecture Overview
- `market_analysis/data.py`: Data fetching, indicator calculation, caching, and freshness checks.
- `market_analysis/summary.py`: Builds the structured technical summary used in prompts.
- `market_analysis/llm.py`: Prompt assembly and LangChain/OpenRouter integration.
- `market_analysis/history.py`: Lightweight SQLite-backed store for past analyses.
- `market_analysis/playbook.py`: Retrieval-augmented indexing and playbook generation.
- `main.py`: Streamlit dashboard UI built on the shared services with additional UX helpers.
- `market_analysis/cli.py`: CLI entry point wiring arguments to the shared services.

## Development Notes
- All modules are fully type-hinted; keep `from __future__ import annotations` when extending.
- `python -m compileall main.py market_analysis/cli.py market_analysis` is a quick syntax check.
- When adding new indicators or prompts, prefer extending the library modules (e.g. add to `DataFetcher` or create a new helper) so both the CLI and Streamlit front-ends benefit automatically.
