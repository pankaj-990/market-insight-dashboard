# Project Documentation

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Market Analysis Guide](#2-market-analysis-guide)
- [3. Code Walkthrough & Reference](#3-code-walkthrough--reference)
- [4. Conclusion](#4-conclusion)



## 1. Introduction
# Market Technical Analysis Toolkit

A small toolkit for downloading OHLC data for any Yahoo Finance symbol, generating a structured technical summary, and (optionally) enriching it with an LLM-backed narrative. It ships with both a CLI workflow and an interactive Streamlit dashboard that share the same underlying services.

## Features
- Fetches OHLC candles from Yahoo Finance and enriches them with EMA/RSI indicators.
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
- `--model` / `--temperature`: Override OpenRouter defaults.
- `--show-prompt`: Print the composed LLM prompt to stdout.
- `--as-of YYYY-MM-DD`: Analyse the state of the market as it stood on the specified date.

## Streamlit Dashboard
```bash
streamlit run main.py
```
The dashboard lets you tweak fetch parameters, inspect the structured summary, download artefacts, and review prior analyses. Results are cached to `analysis_history.db` (alongside a Parquet cache named after the ticker), so re-running with the same parameters skips unnecessary work.


## Architecture Overview
- `market_analysis/data.py`: Data fetching, indicator calculation, caching, and freshness checks.
- `market_analysis/summary.py`: Builds the structured technical summary used in prompts.
- `market_analysis/llm.py`: Prompt assembly and LangChain/OpenRouter integration.
- `market_analysis/history.py`: Lightweight SQLite-backed store for past analyses (with automatic migration from the legacy JSON file).
- `main.py`: Streamlit dashboard UI built on the shared services with additional UX helpers.
- `market_analysis/cli.py`: CLI entry point wiring arguments to the shared services.

## Development Notes
- All modules are fully type-hinted; keep `from __future__ import annotations` when extending.
- `python -m compileall main.py market_analysis/cli.py market_analysis` is a quick syntax check.
- When adding new indicators or prompts, prefer extending the library modules (e.g. add to `DataFetcher` or create a new helper) so both the CLI and Streamlit front-ends benefit automatically.


---

## 2. Market Analysis Guide

This guide serves two roles:

1. A narrative documentation of the **Market Technical Analysis Toolkit** contained in this repository.
2. A self-contained learning resource that illustrates how modern **LLM** and **LangChain** patterns are applied in a production-style project.

The material is organised so you can both understand the codebase and adapt the ideas to your own AI workflows.

---

## 1. Project Overview

The toolkit automates the lifecycle of generating technical market insights:

1. **Fetch** OHLC candles for any Yahoo Finance symbol.
2. **Enrich** the dataset with common indicators (EMA, RSI) and cache the results on disk.
3. **Summarise** the market structure using rule-based heuristics.
4. **Analyse** the summary with an LLM to produce trader-friendly tables.
6. **Serve** results through both a CLI and an interactive Streamlit dashboard.

The default ticker is `^NSEI`, but the workflow is symbol-agnostic—just point it at another ticker.

### Key Features

- **Shared core library** (`market_analysis`) consumed by CLI (`market_analysis/cli.py`) and Streamlit app (`main.py`).
- **Structured LLM output** delivered as markdown tables and machine-readable JSON.
- **History persistence** to avoid redundant work and to enable incremental learning.

---

## 2. Architecture Deep Dive

### 2.1 Modules at a Glance

| Layer | Module(s) | Responsibility |
|-------|-----------|----------------|
| Data acquisition | `market_analysis/data.py` | Download OHLC data, compute indicators, cache Parquet files (with CSV fallback), and enforce freshness rules. |
| Summaries | `market_analysis/summary.py` | Build deterministic technical narratives from indicator data. |
| LLM integration | `market_analysis/llm.py` | Configure models, craft prompts, parse structured outputs. |
| Interfaces | `main.py`, `market_analysis/cli.py` | Provide Streamlit dashboard and CLI automation. |

### 2.2 Data & Control Flow

1. **User Input** (CLI flags or Streamlit form) → `DatasetRequest` (ticker, period, interval, rounding, cache policy).
2. **Data Fetch** → `DataFetcher.fetch_and_save` materialises Parquet files (auto-named by ticker) and returns a Pandas DataFrame. Legacy CSV caches are detected automatically.
3. **Summary Generation** → `build_technical_summary` converts the DataFrame to a structured text synopsis.
4. **LLM Analysis** → `generate_llm_analysis` feeds the summary into a LangChain chain that returns five tables of insights.
5. **History Recording** → `AnalysisHistory.record` stores result payloads (summary, tables, metadata) in `analysis_history.db`.

```
```

### 2.3 Configuration & Persistence

- **Caching**: Parquet caches live next to the repository and are named automatically (`{ticker}_ohlc.parquet`). Legacy CSV files are still detected on load.
- **History Store**: `analysis_history.db` keeps every run, enabling reuse in both front-ends and the RAG index.

---

## 3. LangChain & LLM Patterns Illustrated

### 3.1 Prompt Engineering and Structured Output

The prompt scaffold lives in `market_analysis/llm.py`:

- `ChatPromptTemplate` stitches a system role (trading mentor) with a dynamic human message that includes the technical summary and `format_instructions` derived from a parser.
- `PydanticOutputParser` enforces a schema (`AnalysisTables`), guaranteeing five tables with explicit headers and rows.
- `generate_llm_analysis` constructs a runnable pipeline: `prompt | ChatOpenAI | parser`. The result is wrapped in `LLMAnalysisResult`, bundling both markdown and JSON-friendly dictionaries.

**Why it matters**: This approach demonstrates how to demand high-fidelity, machine-readable output from an LLM—essential for downstream rendering and validation.

### 3.2 Retrieval-Augmented Generation (RAG)


1. **Indexing**: Each history entry is rendered into a `Document` that includes summary text and metadata (ticker, period, last timestamp, etc.).
2. **Vector Store**: FAISS stores embeddings. The builder tries OpenAI/OpenRouter embeddings first and gracefully falls back to `HashingEmbeddings`—a deterministic, local alternative.
3. **Retrieval**: `similarity_search_with_score` fetches nearest cases (deduplicating the current entry).

This demonstrates a complete RAG loop—from ingestion to retrieval to synthesis—which you can adapt to other knowledge bases.

### 3.3 Runnable Pipelines & Reuse

Both CLI and Streamlit invoke the same library functions. This encapsulation makes it easy to:

- Swap LLM models by adjusting environment variables.
- Inject different embedding strategies without touching front-end code.
- Export the LangChain `Runnable` objects for testing or orchestration elsewhere.

---

## 4. Using the Toolkit as a Learning Playground

### 4.1 Quick Start

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=<your_key>
# Optional: force offline embeddings

# or launch the dashboard
streamlit run main.py
```

#### Historical Backfills


```bash
# Analyse the market as it looked on 2024-01-05

# Backfill multiple trade days in one shot (space-separated YYYY-MM-DD values)
python -m market_analysis.cli --ticker AAPL --run-dates 2024-01-05 2024-01-12 2024-01-19 --summary-only
```

The CLI trims the cached dataset to the requested date before building summaries, letting you reconstruct historical state and seed the retrieval index quickly.

### 4.2 Experiments to Try

1. **Prompt Iterations**: Modify the prompt (e.g., add volatility commentary) and observe how the structured tables change.
4. **Summary Heuristics**: Tweak `market_analysis/summary.py` to incorporate new indicators and watch the downstream LLM output adapt.
5. **History Analytics**: Analyse `analysis_history.db` to build dashboards or augment the retriever with outcome labels.

### 4.3 Extend the Project

- **Agentic Indicators**: Wrap indicator calculations as LangChain tools and let the LLM decide which to compute.
- **Session Memory**: Incorporate conversational memory in Streamlit so practitioners can interrogate results interactively.

Each extension builds upon LangChain primitives already showcased here.

---

## 5. Reference Appendix

### 5.1 Key Files

- `main.py` – CLI orchestration.
- `main.py` – Streamlit dashboard UI.
- `market_analysis/data.py` – Data acquisition and caching helpers.
- `market_analysis/summary.py` – Technical summary generation.
- `market_analysis/llm.py` – Prompt, structured output, LangChain chain definitions.
- `market_analysis/history.py` – JSON-backed history persistence.

### 5.2 Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | LLM authentication | _required_ |
| `OPENROUTER_MODEL` | Override chat model | `deepseek/deepseek-chat-v3.1:free` |
| `OPENROUTER_TEMPERATURE` | LLM sampling temperature | `0.1` |

---

## 6. Closing Thoughts

This repository exemplifies how to combine deterministic analytics with adaptive LLM reasoning using LangChain. By walking through data ingestion, prompt design, structured decoding, and retrieval, you gain a blueprint for building your own AI-powered trading copilot—or any domain-specific assistant that requires a blend of quantitative signals and narrative intelligence.

Use this document as both a reference manual for the project and a living notebook for LangChain and LLM experimentation. Happy building!


---

## 3. Code Walkthrough & Reference
# Market Technical Analysis Toolkit — Code Walkthrough & Reference (CLI + Streamlit, TA + LLM via OpenRouter/LangChain)

## Table of Contents
- [Assumptions](#assumptions)
- [1. Executive Overview](#1-executive-overview)
- [2. Quickstart](#2-quickstart)
- [3. Technical Analysis (Core Focus)](#3-technical-analysis-core-focus)
- [4. LLM Layer (Core Focus)](#4-llm-layer-core-focus)
- [5. CLI: Command & Flag Reference](#5-cli-command--flag-reference)
- [6. Streamlit App Guide](#6-streamlit-app-guide)
- [7. Module & API Reference](#7-module--api-reference)
- [8. Configuration & Environments](#8-configuration--environments)
- [9. Data & Storage](#9-data--storage)
- [10. Testing & Quality](#10-testing--quality)
- [11. Performance & Cost Footprint](#11-performance--cost-footprint)
- [12. Troubleshooting](#12-troubleshooting)
- [13. Roadmap / Extensibility](#13-roadmap--extensibility)
- [Optional Variables](#optional-variables)

## Assumptions
- No dedicated backtesting or risk/position-sizing modules exist in the current codebase; guidance below flags these gaps explicitly.
- Streamlit sessions run in a single-user context; multi-user synchronisation is out of scope.
- OpenRouter access is via `langchain_openai.ChatOpenAI`; API-compatible models must be provisioned by the operator.

---

## 1. Executive Overview

**Who it’s for**  
- Market analysts seeking rapid technical snapshots.  
- Quant/LLM engineers integrating TA signals with language workflows.  
- Traders wanting a reproducible TA + narrative pipeline.

**Typical workflows**
- **CLI batch run**: fetch data, print TA summary, optionally call LLM, seed retrieval index.  
- **Streamlit session**: adjust ticker/timeframe, visualise candlesticks, inspect structured LLM tables, download artefacts, revisit history.

**Architecture (ASCII)**

```
           +------------------+
           |  CLI (main.py)   |
           +--------+---------+
                    |
                    | args / flags
                    v
+-------------+   +-------------------+   +--------------------+   +------------------+
| Data source |-->| DataFetcher / Parquet |-->| Summary Builder    |-->| LangChain LLM    |
| (yfinance)  |   | cache (market_…)  |   | (build_technical…) |   | (ChatOpenAI,     |
+-------------+   +---------+---------+   +---------+----------+   | Pydantic parser) |
                                |                     |             +------------------+
                                | history entries     |
                                v                     |
                         +--------------+             |
                         | Analysis     |<------------+
                         | History JSON |
                         +------+-------+
                                |
                                v
                        +---------------+
                        | (FAISS index, |
                        | ChatOpenAI)   |
                        +-------+-------+
                                |
                 +--------------+---------------+
                 |                              |
        +--------v---------+          +---------v--------+
        | Streamlit UI     |          | CLI output (md)  |
        | (visuals, state) |          +------------------+
        +------------------+
```

**Core components**
- Technical Analysis: `market_analysis/data.py`, `market_analysis/summary.py`

---

## 2. Quickstart
**Prerequisites**
- Python 3.11+ (pinned dependencies in `requirements.txt` target CPython 3.11/3.12).
- `pip` or `uv` (Poetry not configured, but easy to add).
- Optional: FAISS CPU wheels (already listed), OpenRouter API credentials.

**Installation**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Environment variables**

| Variable | Required | Default | Purpose | Consumed in |
| --- | --- | --- | --- | --- |
| `OPENROUTER_API_KEY` / `LLM_API_KEY` | For LLM | None | API key for OpenRouter-compatible ChatOpenAI calls | `market_analysis/llm.py:LLMConfig.from_env` |
| `OPENROUTER_MODEL` | Optional | `deepseek/deepseek-chat-v3.1:free` | Default chat model routing | same as above |
| `OPENROUTER_TEMPERATURE` | Optional | `0.1` | Sample temperature override | same |
| `OPENROUTER_API_BASE` | Optional | `https://openrouter.ai/api/v1` | Endpoint override | same |

Provide them via Streamlit secrets (preferred) or standard environment variables.

**Run commands**
- CLI (module style): `python -m main --ticker ^NSEI --summary-only`
- Streamlit: `streamlit run main.py -- --ticker ^NSEI` (use `--` to pass custom Streamlit args if needed)

**Minimal end-to-end demo**
```bash
# 1. Fetch & cache NSEI data, print TA summary only
python -m main --ticker ^NSEI --period 6mo --interval 1d --summary-only

# Expected stdout (abridged):
# === Analysis for ^NSEI (as of latest) ===
# Using cached data from NSEI_ohlc.parquet
# TECHNICAL ANALYSIS SUMMARY
# ...
# RSI 14: 62.67 (Neutral)
# RECENT PRICE ACTION (Last 7 candles):
# Date        Open    High ...


# 3. Launch dashboard
streamlit run main.py
```

---

## 3. Technical Analysis (Core Focus)

### Indicator implementations
| Component | Module.Path | Params | Returns | Used In |
| --- | --- | --- | --- | --- |
| EMA 30 | `market_analysis.data:DataFetcher.fetch` | `length=30` | `pd.Series` appended as `EMA_30` | `build_technical_summary`, Streamlit metrics, CLI output |
| EMA 200 | same | `length=200` | `pd.Series` (`EMA_200`) | same |
| RSI 14 | same | `length=14` | `pd.Series` (`RSI_14`) | Summary narrative, Streamlit metrics, LLM prompt |

- Each indicator is computed via `pandas_ta` vectorised routines (`O(n)` over candles).
- Additional indicators require extending `DataFetcher.fetch`.

### Strategy blocks
- Entry/exit/position logic is descriptive (LLM markdown) rather than executable.

### Risk & position sizing

### Backtest pipeline
- Absent. Extend by introducing dedicated modules (e.g., `backtest.py`) that consume cached DataFrames.

### Data model
- OHLC schema: Index = `Date` (timezone preserved if source supplies), columns `Open`, `High`, `Low`, `Close`, `EMA_30`, `EMA_200`, `RSI_14`.
- Column enforcement: `load_cached_data` validates presence of `_ALL_COLUMNS` and raises if missing.
- Missing data: relies on pandas/yfinance output; downstream summary will raise `ValueError` on absent columns.

### Performance notes
- All indicator calculations are vectorised.
- Parquet caching avoids redundant network calls (`ensure_dataset` controls freshness with `max_age_days` and `is_stale`).
- Streamlit keeps DataFrame in session state to prevent recomputation within a run.

---

## 4. LLM Layer (Core Focus)

### OpenRouter integration
- `market_analysis.llm:LLMConfig.from_env` loads credentials; missing API key raises `RuntimeError`.
- `ChatOpenAI` initialised with `model`, `temperature`, `openai_api_key`, `openai_api_base`.
- Retry/backoff handled inside LangChain/OpenAI client; adjust via environment if needed.
- Cost awareness: defaults to a free-tier DeepSeek model; switching to paid SKUs requires manual rate monitoring.
- Rate limits: rely on OpenRouter responses; `_run_llm` surfaces exceptions to Streamlit UI.

### LangChain components
| Component | Location | Usage |
| --- | --- | --- |
| `PydanticOutputParser` | `market_analysis.llm:_create_parser` | Enforces schema for five analysis tables |
| `Runnable` pipeline | `market_analysis.llm:generate_llm_analysis` | Builds prompt → ChatOpenAI → parser internally |
| `Embeddings` (`OpenAIEmbeddings`, `HashingEmbeddings`) | same | Document embeddings; hash fallback for offline use |
| Retrievers | FAISS via `similarity_search_with_score` | Pull top-k historical cases |
| Tools/Agents | None; logic is direct chain execution |
| Memory | Not used; session caching handled manually |
| Callbacks/Tracing | Not implemented (LangSmith/OTEL not wired) |
| Evaluation | No automated eval harness; rely on manual inspection/historical reuse |

### Prompt design & guardrails
- System prompt enforces “expert financial technical analyst” persona.
- Human prompt injects structured TA summary and required JSON schema instructions.
- `PydanticOutputParser` rejects malformed outputs (LangChain handles).
- `format_prompt` exposes composed prompt for audit (`--show-prompt`, Streamlit expander).

- Embedding backend gracefully downgrades to hashing with warning captured in UI/CLI.

---

## 5. CLI: Command & Flag Reference

**Executable**: `python -m main` (entry: `main.py:main`)

**Usage**
```
```

**Flags**

| Flag | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--ticker` | str | `^NSEI` | No | Yahoo Finance symbol |
| `--period` | str | `1y` | No | yfinance period (e.g., `6mo`, `max`) |
| `--interval` | str | `1d` | No | Candle interval (`1d`, `1wk`, etc.) |
| `--round-digits` | int | `2` | No | Rounding before Parquet write |
| `--max-age-days` | int | `3` | No | Cache freshness; `-1` disables |
| `--force-refresh` | bool | False | No | Always refetch |
| `--model` | str | None | No | Override LLM model |
| `--temperature` | float | None | No | Override LLM temperature |
| `--summary-only` | bool | False | No | Skip LLM call |
| `--show-prompt` | bool | False | No | Print composed LLM messages |
| `--as-of` | date | None | No | Analyse candles up to date |
| `--run-dates` | [date] | None | No | Multiple analyses per invocation |

**I/O**
- Input: network call to yfinance unless cached Parquet present; secrets loaded from Streamlit or env vars.
- Output: stdout (plain text + markdown tables); exit code `0` on success, `1` on unexpected exception (captured per run date).
- Artefacts: Parquet caches named `<ticker>_ohlc.parquet`, history entry appended.

**Examples**
1. Compute RSI/EMA summary only: `python -m main --ticker AAPL --summary-only`
2. Backfill multiple historical sessions: `python -m main --ticker BTC-USD --run-dates 2024-01-05 2024-02-02`
3. Force refresh & inspect prompt: `python -m main --ticker ^NSEBANK --force-refresh --show-prompt`
5. Swap model & temperature: `python -m main --model openrouter/anthropic/claude-3.5-sonnet --temperature 0.3`

---

## 6. Streamlit App Guide

**Entry**: `main.py:main`

| Page/Tab | Purpose |
| --- | --- |
| Top metrics | Latest `Close`, `EMA_30`, `EMA_200`, `RSI_14` |

**Sidebar form (`render_sidebar`)**
- Inputs: `ticker`, `period`, `interval`, plus lower timeframe toggles (`include_lower`, `lower_interval`, `lower_period`, window sizes), rounding, cache age, LLM toggles.
- Submission writes `AnalysisParams` to session state (`analysis_params` key).

**State flow**
1. `AnalysisParams.higher_dataset_request` / `.lower_dataset_request` translate form to `DatasetRequest` objects.
3. Results cached under `analysis_result`; errors under `analysis_error`.
4. `AnalysisHistory` persists to `analysis_history.db` (with automatic import from the legacy JSON file on first run).

**Interactive elements**
- Candlestick plot via Plotly (`_plot_candlestick`).

**LLM panes**
- `LLM Analysis` tab shows structured tables when `llm_tables` present; fallback to markdown narrative.
- Prompt expander enabled when `show_prompt` toggled.

**Caching**
- `ensure_dataset` prevents redundant downloads.
- `HISTORY.record` ensures deduplicated entries (overwriting by `entry_id`).
- Reuse flags (`reused_summary`, etc.) displayed via `st.caption`.

---

## 7. Module & API Reference

### `market_analysis/data.py`
```
DataFetcher.fetch()
 ├─ yfinance.download()
 ├─ pandas ta.ema(), ta.rsi()
 └─ returns pd.DataFrame with OHLC + indicators
ensure_dataset()
 ├─ DatasetRequest.create_fetcher()
 ├─ load_cached_data()
 ├─ DataFetcher.fetch_and_save()
 └─ is_stale()
```

- `default_data_filename(ticker: str, interval: str | None = None) -> str`  
  Summary: slugifies ticker and optional interval → `{slug}_{interval}_ohlc.parquet`.  
  Returns: filename. No side effects. Used by `default_data_path`.

- `default_data_path(ticker: str, interval: str | None = None) -> Path`  
  Summary: wraps filename in `Path`. Used by `AnalysisParams.data_path`.

- `DataFetcher.fetch(self) -> pd.DataFrame`  
  Params: dataclass fields (`ticker`, `period`, `interval`).  
  Returns: OHLC DataFrame with EMA/RSI columns.  
  Raises: `ValueError` (empty dataset).  
  Side effects: network call to Yahoo Finance.  
  Used in: `DataFetcher.fetch_and_save`, `ensure_dataset`.

- `DataFetcher.fetch_and_save(self, path, decimal_places=None) -> pd.DataFrame`  
  Side effects: writes Parquet via `save_dataframe`.  
  Used in: `ensure_dataset`.

- `DatasetRequest.resolved_path(self) -> Path`  
  Returns: expanded cache path.  
  Used in: CLI/Streamlit to reference Parquet caches.

- `DatasetRequest.create_fetcher(self) -> DataFetcher`  
  Used in: `ensure_dataset`.

- `ensure_dataset(request, force_refresh=False) -> tuple[pd.DataFrame, str]`  
  Summary: returns DataFrame plus source label (`freshly downloaded`, `refreshed`, `cached`).  
  Side effects: may write Parquet.  
  Used in: CLI `_run_single_analysis`, Streamlit `handle_submission`.

- `is_stale(last_timestamp, max_age_days) -> bool`  
  Used in: `ensure_dataset`.

- `save_dataframe(df, path)` and `load_cached_data(path)`  
  Side effects: disk IO.  
  `load_cached_data` raises `FileNotFoundError` or `ValueError` if columns missing.  
  Used in: CLI/Streamlit fetch path.

**Core hot path**: `ensure_dataset`.

### `market_analysis/summary.py`
```
build_technical_summary()
 ├─ _describe_rsi()
 └─ _qualify_comparison()
```

- `build_technical_summary(df, key_window=20, recent_rows=7) -> str`  
  Summary: Formats TA narrative with levels/momentum tables.  
  Raises: `ValueError` when DataFrame empty or missing required columns.  

- `_describe_rsi(value)` / `_qualify_comparison(value, ref, positive, negative)`  
  Internal helpers (not exported).

**Core hot path**: `build_technical_summary`.

### `market_analysis/llm.py`
```
generate_llm_analysis()
 ├─ _build_prompt()
 ├─ ChatOpenAI()
 └─ PydanticOutputParser()
```

- `TableSection`, `AnalysisTables` (`pydantic.BaseModel`)  
  Structured schema for LLM outputs.

- `LLMAnalysisResult` dataclass (`markdown: str`, `tables: dict`)  
  Returned to CLI/Streamlit.

- `LLMConfig.from_env() -> LLMConfig`  
  Raises: `RuntimeError` if API key missing.  

- `format_prompt(technical_summary: str) -> str`  
  Returns: multi-message prompt for audit.  
  Used in: CLI `_print_prompt`, Streamlit prompt display.

- `generate_llm_analysis(technical_summary, config=None) -> LLMAnalysisResult`  
  Side effects: remote LLM call.  
  Raises: propogates LangChain exceptions.  
  Used in: CLI `_run_single_analysis`, Streamlit `_run_llm`.

**Core hot path**: `generate_llm_analysis`.

### `market_analysis/history.py`
```
AnalysisHistory.record()
 ├─ load()
 ├─ prune duplicates
 └─ save()
```

- `AnalysisHistory.load() -> list[dict]`  
  Returns entries or empty list.

- `AnalysisHistory.save(entries)`  
  Side effects: write JSON.

- `AnalysisHistory.record(entry)`  
  Deduplicates by `cache_key`, sorts by timestamp descending.  
  Used in: Streamlit `handle_submission`.

- `AnalysisHistory.find(cache_key)`  
  Used in: (currently unused) but available.

- `AnalysisHistory.as_dataframe(entries=None) -> pd.DataFrame`  
  Used in: Streamlit tabs.

- `make_cache_key(params, last_timestamp, lower_last_timestamp=None) -> dict`  

**Core hot path**: `AnalysisHistory.record`.

```
 ├─ _load_store()/FAISS
 ├─ similarity_search_with_score()
 ├─ _format_cases_for_prompt()
 └─ ChatOpenAI.invoke()
```

- `make_entry_id(cache_key) -> str`  
  Deterministic JSON string; used as vector-store metadata key.

  Raises: `ValueError` on invalid backend.  

- `HashingEmbeddings`  
  Deterministic fallback embeddings; side-effect free.

  Checks embeddings; stores `embedding_warning` if fallback invoked.

  Side effects: create/update FAISS index on disk.


  Returns `None` if configuration missing.


### `main.py`
```
main()
 └─ run_cli(parse_args())
      └─ _run_single_analysis()
           ├─ ensure_dataset()
           ├─ build_technical_summary()
           ├─ generate_llm_analysis()
```

- `build_parser() -> argparse.ArgumentParser`  
  Mutates CLI interface.

- `parse_args() -> argparse.Namespace`  
  Wraps `build_parser`.

- `run_cli(args)`  
  Iterates over requested dates; prints output or error per run.

- `configure_llm(args) -> LLMConfig`  
  Overrides model/temperature.

- `_resolve_run_dates(args)`  
  Handles `--run-dates` vs `--as-of`.

- `_run_single_analysis(args, as_of)`  
  Core CLI workflow (see diagram).  

- `_print_prompt(summary)`  
  Pretty-prints prompt.

- `main()`  
  Entry point (used by `python -m main`).

### `main.py`
```
main()
 ├─ render_sidebar()
 ├─ handle_submission()
 │    ├─ ensure_dataset()
 │    ├─ build_technical_summary()
 │    ├─ _run_llm()
 └─ render tabs/plots/downloads
```

Key public-ish helpers (`st` prefix indicates Streamlit coupling):

- `AnalysisParams` dataclass with `from_state`, `to_state`, `higher_dataset_request`, `lower_dataset_request`, `cache_payload`, `history_payload`, `data_path`.
- `render_sidebar(defaults) -> (AnalysisParams, bool)`  
  Builds the sidebar form.

- `handle_submission(params, history_entries)`  
  Primary orchestration (mirrors CLI with caching reuse).

- `_run_llm(technical_summary)`  
  Wraps `LLMConfig.from_env` + `generate_llm_analysis` for Streamlit context.

- `_plot_candlestick(df)`  
  Plotly candlestick overlay with EMA traces.

- `main()`  
  Registers UI layout and tab content.

**Core hot paths**: `handle_submission`, `main`.

---

## 8. Configuration & Environments
- **Hierarchy**: Streamlit secrets → environment variables → CLI flags / Streamlit form values.
- **Secrets**: Store credentials in `.streamlit/secrets.toml` (local) or the Streamlit Cloud UI; rotate by updating secrets and restarting/deploying.

---

## 9. Data & Storage
- **Parquet caches**: `default_data_path(ticker, interval)` → e.g., `NSEI_1d_ohlc.parquet`. Rounded per `round_digits`.
- **Temporary artifacts**: Streamlit downloads produced on demand (not persisted).
- **Data sources**: Yahoo Finance via `yfinance.download`. Legacy sample CSVs (`NSEI_ohlc.csv`, `NSEBANK_ohlc.csv`) remain for demos and are auto-detected alongside the new Parquet caches.

---

## 10. Testing & Quality
- No automated tests present. Recommended additions:
  - Unit tests for `DataFetcher.ensure_dataset` (mock yfinance).
  - Unit tests for `build_technical_summary` edge cases.
  - Integration tests for `generate_llm_analysis` using mocked ChatOpenAI.
- Suggested command: `pytest` (once tests added).  
- Linting: `black`, `isort` available via `requirements.txt`.  
- CI: Not configured; GitHub Actions or similar recommended.

---

## 11. Performance & Cost Footprint
- **TA pipeline**: dominated by pandas operations; typical 1-year daily history (<300 rows) executes in milliseconds.
- **LLM calls**:
  - Default model `deepseek/deepseek-chat-v3.1:free` (no token charges; rate-limited).
  - Example prompt tokens ~1.2k; response ~500 tokens → ensure OpenRouter quotas support this.
- **Invalidation**: 
- Cache refresh triggered by `max_age_days` or `--force-refresh`.

| Task | Estimated Tokens | Est. Cost (USD) | Notes |
| --- | --- | --- | --- |
| `generate_llm_analysis` (default model) | ~1.7k | $0 (free tier) | Check OpenRouter policy; swap model to adjust |

---

## 12. Troubleshooting
- **Missing OpenRouter key**: CLI prints exception; Streamlit shows warning in LLM tab. Set `OPENROUTER_API_KEY`.
- **Rate limits / timeouts**: `_run_llm` surfaces `The LLM request failed: …`; re-run later or change model/temperature.
- **Streamlit port in use**: `streamlit run main.py --server.port 8502`.
- **pdflatex missing**: Not required (no PDF exports).
- **macOS Gatekeeper**: For `faiss-cpu`, use Python wheels (already in requirements).

---

## 13. Roadmap / Extensibility
1. **Add a new indicator / strategy**  
   - Extend `DataFetcher.fetch` to compute and append new columns.  
   - Update `_REQUIRED_COLUMNS` & summary narrative.  
   - Surface metrics in Streamlit metrics + CLI.

2. **Add a LangChain tool/agent**  
   - Wrap indicator calculators as LangChain Tools; introduce an Agent in `market_analysis/llm.py` or new module.  
   - Manage tool routing via `RunnableSequence` or `AgentExecutor`.

3. **Swap LLMs**  
   - Pass `--model` in CLI or set `OPENROUTER_MODEL`.  
   - For Streamlit, set env var before launch; `AnalysisParams` currently doesn’t expose model override—extend form if needed.

4. **Plug in new data source/broker**  
   - Implement alternate `DataFetcher` (e.g., broker SDK).  
   - Adjust `DatasetRequest.create_fetcher` to select backend based on ticker prefix or config.  
   - Ensure consistent column schema to keep summary builder/LLM stable.

5. **Backtesting module** (future work)  
   - Leverage cached Parquet + indicators.  
   - Persist results to history for RAG context.

---

## Optional Variables
- `{{SUPPORTED_SYMBOLS}}`: Symbol-agnostic; defaults to `^NSEI` but accepts any Yahoo Finance ticker.
- `{{DEFAULT_TIMEFRAMES}}`: CLI defaults (`period=1y`, `interval=1d`); Streamlit options `["1d", "1wk", "1mo"]`.
- `{{OHLC_STORE}}`: `<ticker>_ohlc.parquet` in repository root (configurable via `DatasetRequest.data_path`).
- `{{OPENROUTER_MODELS}}`: Default `deepseek/deepseek-chat-v3.1:free`; override via `OPENROUTER_MODEL` or CLI `--model`.
- `{{AGENT_TYPE}}`: Not implemented (direct chain usage).
- `{{BACKTEST_WINDOWS}}`: Not implemented; summary uses `key_window=20`, `recent_rows=7`.
- `{{METRICS}}`: Summary shows Latest Close, EMA30/EMA200, RSI14; LLM tables expand on trend/evidence.


---

## 4. Conclusion
This document consolidates the project README, market analysis methodologies, and technical code walkthrough into one cohesive reference. 
Use it as a single source of truth for understanding, running, and extending the project.
