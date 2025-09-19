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
- `--model` / `--temperature`: Override OpenRouter defaults.
- `--show-prompt`: Print the composed LLM prompt to stdout.
- `--show-playbook`: Display retrieval-backed playbook insights (requires embeddings + indexed history).
- `--as-of YYYY-MM-DD`: Analyse the state of the market as it stood on the specified date.
- `--run-dates YYYY-MM-DD [YYYY-MM-DD ...]`: Backfill multiple dates in one command (useful for seeding the playbook index).

## Streamlit Dashboard
```bash
streamlit run main.py
```
The dashboard lets you tweak fetch parameters, inspect the structured summary, download artefacts, and review prior analyses. Results are cached to `analysis_history.json` (alongside the cached CSV named after the ticker), so re-running with the same parameters skips unnecessary work.

### Strategy Playbook (RAG)
- Set `OPENROUTER_API_KEY` (or `LLM_API_KEY`) plus optional overrides `PLAYBOOK_EMBED_MODEL`, `PLAYBOOK_TOP_K`, `PLAYBOOK_INDEX_PATH`, and `PLAYBOOK_TEMPERATURE`.
- After a few analyses are stored, the Streamlit “Playbook Insights” tab (or CLI `--show-playbook`) retrieves similar historical cases and drafts trade ideas.
- Historical entries are automatically indexed into a FAISS vector store located at `playbook_index/` by default. Install `faiss-cpu` (already listed in `requirements.txt`) to enable the index. If remote embeddings are unavailable (e.g., OpenRouter does not support them), the app falls back to a lightweight hash-based embedding. You can force this with `PLAYBOOK_EMBED_BACKEND=hash` or stick with OpenAI-compatible embeddings via `PLAYBOOK_EMBED_BACKEND=openai` and a supported model (`PLAYBOOK_EMBED_MODEL`).

## Architecture Overview
- `market_analysis/data.py`: Data fetching, indicator calculation, caching, and freshness checks.
- `market_analysis/summary.py`: Builds the structured technical summary used in prompts.
- `market_analysis/llm.py`: Prompt assembly and LangChain/OpenRouter integration.
- `market_analysis/history.py`: Minimal JSON-backed store for past analyses.
- `market_analysis/playbook.py`: Retrieval-augmented indexing and playbook generation.
- `main.py`: Streamlit dashboard UI built on the shared services with additional UX helpers.
- `market_analysis/cli.py`: CLI entry point wiring arguments to the shared services.

## Development Notes
- All modules are fully type-hinted; keep `from __future__ import annotations` when extending.
- `python -m compileall main.py market_analysis/cli.py market_analysis` is a quick syntax check.
- When adding new indicators or prompts, prefer extending the library modules (e.g. add to `DataFetcher` or create a new helper) so both the CLI and Streamlit front-ends benefit automatically.


---

## 2. Market Analysis Guide
# Market Analysis Playbook: Architecture & LangChain Handbook

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
5. **Retrieve** similar historical scenarios (RAG) and assemble a strategy playbook.
6. **Serve** results through both a CLI and an interactive Streamlit dashboard.

The default ticker is `^NSEI`, but the workflow is symbol-agnostic—just point it at another ticker.

### Key Features

- **Shared core library** (`market_analysis`) consumed by CLI (`market_analysis/cli.py`) and Streamlit app (`main.py`).
- **Structured LLM output** delivered as markdown tables and machine-readable JSON.
- **Retrieval-Augmented Playbook** that learns from prior runs via a FAISS vector store, with automatic hashing-based embeddings when remote embeddings are unavailable.
- **History persistence** to avoid redundant work and to enable incremental learning.

---

## 2. Architecture Deep Dive

### 2.1 Modules at a Glance

| Layer | Module(s) | Responsibility |
|-------|-----------|----------------|
| Data acquisition | `market_analysis/data.py` | Download OHLC data, compute indicators, cache CSVs, and enforce freshness rules. |
| Summaries | `market_analysis/summary.py` | Build deterministic technical narratives from indicator data. |
| LLM integration | `market_analysis/llm.py` | Configure models, craft prompts, parse structured outputs. |
| Retrieval & history | `market_analysis/history.py`, `market_analysis/playbook.py` | Persist analysis runs, manage FAISS index, build strategy playbooks. |
| Interfaces | `main.py`, `market_analysis/cli.py` | Provide Streamlit dashboard and CLI automation. |

### 2.2 Data & Control Flow

1. **User Input** (CLI flags or Streamlit form) → `DatasetRequest` (ticker, period, interval, rounding, cache policy).
2. **Data Fetch** → `DataFetcher.fetch_and_save` materialises CSVs (auto-named by ticker) and returns a Pandas DataFrame.
3. **Summary Generation** → `build_technical_summary` converts the DataFrame to a structured text synopsis.
4. **LLM Analysis** → `generate_llm_analysis` feeds the summary into a LangChain chain that returns five tables of insights.
5. **History Recording** → `AnalysisHistory.record` stores result payloads (summary, tables, metadata) in `analysis_history.json`.
6. **Playbook RAG** → `PlaybookBuilder` indexes each run in FAISS (OpenAI embeddings or deterministic hashing). A new run retrieves similar entries and prompts the LLM to assemble a strategy playbook.
7. **Presentation** → The CLI prints markdown; Streamlit renders metric widgets, tables, and playbook insights.

```
Inputs ──▶ DataFetcher ─▶ Summary ─▶ LLM (LangChain) ─▶ History ─▶ FAISS Playbook ─▶ UI/CLI
```

### 2.3 Configuration & Persistence

- **Environment Variables**: Configure LLMs (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_TEMPERATURE`), playbook embeddings (`PLAYBOOK_EMBED_MODEL`, `PLAYBOOK_EMBED_BACKEND`, `PLAYBOOK_EMBED_DIM`, `PLAYBOOK_TOP_K`), and storage locations (`PLAYBOOK_INDEX_PATH`).
- **Caching**: The CSV cache lives next to the repository and is named automatically (`{ticker}_ohlc.csv`).
- **History Store**: `analysis_history.json` keeps every run, enabling reuse in both front-ends and the RAG index.
- **Vector Store**: `playbook_index/` holds FAISS data; if remote embeddings fail, a hash-based embedding maintains functionality offline.

---

## 3. LangChain & LLM Patterns Illustrated

### 3.1 Prompt Engineering and Structured Output

The prompt scaffold lives in `market_analysis/llm.py`:

- `ChatPromptTemplate` stitches a system role (trading mentor) with a dynamic human message that includes the technical summary and `format_instructions` derived from a parser.
- `PydanticOutputParser` enforces a schema (`AnalysisTables`), guaranteeing five tables with explicit headers and rows.
- `generate_llm_analysis` constructs a runnable pipeline: `prompt | ChatOpenAI | parser`. The result is wrapped in `LLMAnalysisResult`, bundling both markdown and JSON-friendly dictionaries.

**Why it matters**: This approach demonstrates how to demand high-fidelity, machine-readable output from an LLM—essential for downstream rendering and validation.

### 3.2 Retrieval-Augmented Generation (RAG)

`market_analysis/playbook.py` shows the retrieval stack:

1. **Indexing**: Each history entry is rendered into a `Document` that includes summary text and metadata (ticker, period, last timestamp, etc.).
2. **Vector Store**: FAISS stores embeddings. The builder tries OpenAI/OpenRouter embeddings first and gracefully falls back to `HashingEmbeddings`—a deterministic, local alternative.
3. **Retrieval**: `similarity_search_with_score` fetches nearest cases (deduplicating the current entry).
4. **Playbook Prompt**: A LangChain prompt merges the current summary with retrieved case digests, asking the LLM to produce three sections (Historical Patterns, Strategy Recommendations, Risk Watchlist).

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
export PLAYBOOK_EMBED_BACKEND=hash

python -m market_analysis.cli --ticker AAPL --show-playbook
# or launch the dashboard
streamlit run main.py
```

#### Historical Backfills

Populate the playbook with past regimes by replaying analyses for earlier dates:

```bash
# Analyse the market as it looked on 2024-01-05
python -m market_analysis.cli --ticker AAPL --as-of 2024-01-05 --show-playbook

# Backfill multiple trade days in one shot (space-separated YYYY-MM-DD values)
python -m market_analysis.cli --ticker AAPL --run-dates 2024-01-05 2024-01-12 2024-01-19 --summary-only
```

The CLI trims the cached dataset to the requested date before building summaries, letting you reconstruct historical state and seed the retrieval index quickly.

### 4.2 Experiments to Try

1. **Prompt Iterations**: Modify the prompt (e.g., add volatility commentary) and observe how the structured tables change.
2. **Embedding Backends**: Compare OpenAI embeddings vs. hash embeddings by toggling `PLAYBOOK_EMBED_BACKEND`.
3. **FAISS Vector Store**: Inspect `playbook_index/` to understand how LangChain serialises vector stores.
4. **Summary Heuristics**: Tweak `market_analysis/summary.py` to incorporate new indicators and watch the downstream LLM output adapt.
5. **History Analytics**: Analyse `analysis_history.json` to build dashboards or augment the retriever with outcome labels.

### 4.3 Extend the Project

- **Add Fundamentals**: Introduce a fundamentals retriever (news, earnings) and feed it into the playbook prompt.
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
- `market_analysis/playbook.py` – FAISS indexing, fallback embeddings, RAG playbook generation.
- `market_analysis/history.py` – JSON-backed history persistence.

### 5.2 Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | LLM authentication | _required_ |
| `OPENROUTER_MODEL` | Override chat model | `deepseek/deepseek-chat-v3.1:free` |
| `OPENROUTER_TEMPERATURE` | LLM sampling temperature | `0.1` |
| `PLAYBOOK_EMBED_MODEL` | Remote embedding model | `text-embedding-3-small` |
| `PLAYBOOK_EMBED_BACKEND` | `auto`, `openai`, or `hash` | `auto` |
| `PLAYBOOK_EMBED_DIM` | Dimension for hash embeddings | `512` |
| `PLAYBOOK_TOP_K` | Number of historical cases | `3` |
| `PLAYBOOK_CASE_SNIPPET` | Snippet length (chars) | `600` |
| `PLAYBOOK_INDEX_PATH` | FAISS index location | `playbook_index/` |

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
A shared Python library (`market_analysis/`) powers both a CLI (`market_analysis/cli.py`) and a Streamlit dashboard (`main.py`). The workflow fetches Yahoo Finance OHLC data, derives EMA/RSI indicators, constructs a deterministic technical narrative, and optionally enriches it with an OpenRouter-backed LLM plus a retrieval-augmented “strategy playbook”.

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
| Data source |-->| DataFetcher / CSV |-->| Summary Builder    |-->| LangChain LLM    |
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
                        | Playbook RAG  |
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
- LLM & LangChain: `market_analysis/llm.py`, `market_analysis/playbook.py`
- OpenRouter integration: `LLMConfig` (`market_analysis/llm.py`), `PlaybookBuilder`
- Storage: CSV caches, `analysis_history.json`, `playbook_index/`
- Configuration: Streamlit secrets (OpenRouter, playbook embeddings), CLI flags, Streamlit session state

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
| `PLAYBOOK_INDEX_PATH` | Optional | `playbook_index` | FAISS storage directory | `market_analysis/playbook.py:PlaybookConfig.from_env` |
| `PLAYBOOK_EMBED_MODEL` | Optional | `text-embedding-3-small` | Embedding SKU for OpenAI-compatible backend | same |
| `PLAYBOOK_EMBED_BACKEND` | Optional | `auto` | `auto`/`openai`/`hash` embedding selection | same |
| `PLAYBOOK_EMBED_DIM` | Optional | `512` | Hash embedding dimension | same |
| `PLAYBOOK_TOP_K` | Optional | `3` | # of similar cases to surface | same |
| `PLAYBOOK_TEMPERATURE` | Optional | `0.0` | LLM temperature for playbook generation | same |
| `PLAYBOOK_CASE_SNIPPET` | Optional | `600` | Characters retained per retrieved case | same |

Provide them via Streamlit secrets (preferred) or standard environment variables.

**Run commands**
- CLI (module style): `python -m main --ticker ^NSEI --summary-only`
- CLI (file style): `python -m market_analysis.cli --ticker AAPL --show-playbook`
- Streamlit: `streamlit run main.py -- --ticker ^NSEI` (use `--` to pass custom Streamlit args if needed)

**Minimal end-to-end demo**
```bash
# 1. Fetch & cache NSEI data, print TA summary only
python -m main --ticker ^NSEI --period 6mo --interval 1d --summary-only

# Expected stdout (abridged):
# === Analysis for ^NSEI (as of latest) ===
# Using cached data from NSEI_ohlc.csv
# TECHNICAL ANALYSIS SUMMARY
# ...
# RSI 14: 62.67 (Neutral)
# RECENT PRICE ACTION (Last 7 candles):
# Date        Open    High ...

# 2. Include LLM narrative & playbook (requires API key & embeddings)
python -m main --ticker ^NSEI --show-prompt --show-playbook

# 3. Launch dashboard
streamlit run main.py
```
When the Streamlit app finishes its first run, expect `analysis_history.json` to contain a structured payload and `playbook_index/` to materialise (if embeddings succeed).

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
- No deterministic backtest/strategy classes exist. Strategy insights come from `PlaybookBuilder.generate_playbook` (LLM-driven using retrieved history cases).
- Entry/exit/position logic is descriptive (LLM markdown) rather than executable.

### Risk & position sizing
- Not modelled in code; playbook LLM output may include guidance but nothing is enforced programmatically.

### Backtest pipeline
- Absent. Extend by introducing dedicated modules (e.g., `backtest.py`) that consume cached DataFrames.

### Data model
- OHLC schema: Index = `Date` (timezone preserved if source supplies), columns `Open`, `High`, `Low`, `Close`, `EMA_30`, `EMA_200`, `RSI_14`.
- Column enforcement: `load_cached_data` validates presence of `_ALL_COLUMNS` and raises if missing.
- Missing data: relies on pandas/yfinance output; downstream summary will raise `ValueError` on absent columns.

### Performance notes
- All indicator calculations are vectorised.
- CSV caching avoids redundant network calls (`ensure_dataset` controls freshness with `max_age_days` and `is_stale`).
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
| `ChatPromptTemplate` | `market_analysis.llm:_build_prompt`; `market_analysis.playbook:_PLAYBOOK_PROMPT` | Formats structured system/human messages |
| `PydanticOutputParser` | `market_analysis.llm:_create_parser` | Enforces schema for five analysis tables |
| `Runnable` pipeline | `market_analysis.llm:generate_llm_analysis` | Builds prompt → ChatOpenAI → parser internally |
| `ChatOpenAI` | `market_analysis.llm`, `market_analysis.playbook` | OpenRouter-compatible chat model |
| `FAISS` | `market_analysis.playbook:PlaybookBuilder` | Vector store for historical summaries |
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

### Playbook generation
- Uses RAG: retrieved case summaries + current summary pass through `_PLAYBOOK_PROMPT`.
- Embedding backend gracefully downgrades to hashing with warning captured in UI/CLI.

---

## 5. CLI: Command & Flag Reference

**Executable**: `python -m main` (entry: `main.py:main`)

**Usage**
```
python -m main --ticker NIFTY --period 1y --interval 1d --show-prompt --show-playbook
```

**Flags**

| Flag | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--ticker` | str | `^NSEI` | No | Yahoo Finance symbol |
| `--period` | str | `1y` | No | yfinance period (e.g., `6mo`, `max`) |
| `--interval` | str | `1d` | No | Candle interval (`1d`, `1wk`, etc.) |
| `--round-digits` | int | `2` | No | Rounding before CSV write |
| `--max-age-days` | int | `3` | No | Cache freshness; `-1` disables |
| `--force-refresh` | bool | False | No | Always refetch |
| `--model` | str | None | No | Override LLM model |
| `--temperature` | float | None | No | Override LLM temperature |
| `--summary-only` | bool | False | No | Skip LLM call |
| `--show-prompt` | bool | False | No | Print composed LLM messages |
| `--as-of` | date | None | No | Analyse candles up to date |
| `--run-dates` | [date] | None | No | Multiple analyses per invocation |
| `--show-playbook` | bool | False | No | Display RAG playbook insights |

**I/O**
- Input: network call to yfinance unless cached CSV present; secrets loaded from Streamlit or env vars.
- Output: stdout (plain text + markdown tables); exit code `0` on success, `1` on unexpected exception (captured per run date).
- Artefacts: CSV caches named `<ticker>_ohlc.csv`, history entry appended.

**Examples**
1. Compute RSI/EMA summary only: `python -m main --ticker AAPL --summary-only`
2. Backfill multiple historical sessions: `python -m main --ticker BTC-USD --run-dates 2024-01-05 2024-02-02`
3. Force refresh & inspect prompt: `python -m main --ticker ^NSEBANK --force-refresh --show-prompt`
4. Generate playbook (requires history & embeddings): `python -m main --ticker ^NSEI --show-playbook`
5. Swap model & temperature: `python -m main --model openrouter/anthropic/claude-3.5-sonnet --temperature 0.3`

---

## 6. Streamlit App Guide

**Entry**: `main.py:main`

| Page/Tab | Purpose |
| --- | --- |
| Top metrics | Latest `Close`, `EMA_30`, `EMA_200`, `RSI_14` |
| Tabs: `Technical Summary`, `Playbook Insights`, `Recent Candles`, `LLM Analysis`, `History` | Mirrors CLI artefacts with downloads & tables |

**Sidebar form (`render_sidebar`)**
- Inputs: `ticker`, `period`, `interval`, plus lower timeframe toggles (`include_lower`, `lower_interval`, `lower_period`, window sizes), rounding, cache age, LLM toggles.
- Submission writes `AnalysisParams` to session state (`analysis_params` key).

**State flow**
1. `AnalysisParams.higher_dataset_request` / `.lower_dataset_request` translate form to `DatasetRequest` objects.
2. `handle_submission` orchestrates fetch → summary → LLM → playbook → history record.
3. Results cached under `analysis_result`; errors under `analysis_error`.
4. `AnalysisHistory` persists to `analysis_history.json`.

**Interactive elements**
- Candlestick plot via Plotly (`_plot_candlestick`).
- Data tables: last 20 candles, playbook case DataFrame, historical runs using `st.dataframe`.
- Downloads: summary `.txt`, playbook `.md`, prompt `.txt`, OHLC `.csv`.

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
  Summary: slugifies ticker and optional interval → `{slug}_{interval}_ohlc.csv`.  
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
  Side effects: writes CSV via `save_dataframe`.  
  Used in: `ensure_dataset`.

- `DatasetRequest.resolved_path(self) -> Path`  
  Returns: expanded cache path.  
  Used in: CLI/Streamlit to reference CSV.

- `DatasetRequest.create_fetcher(self) -> DataFetcher`  
  Used in: `ensure_dataset`.

- `ensure_dataset(request, force_refresh=False) -> tuple[pd.DataFrame, str]`  
  Summary: returns DataFrame plus source label (`freshly downloaded`, `refreshed`, `cached`).  
  Side effects: may write CSV.  
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
  Used in: CLI, Streamlit, playbook index.

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
  Used in: CLI `configure_llm`, Streamlit `_run_llm`, `create_playbook_builder`.

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
  Deterministic key used across CLI/Streamlit/Playbook.

**Core hot path**: `AnalysisHistory.record`.

### `market_analysis/playbook.py`
```
PlaybookBuilder.generate_playbook()
 ├─ _load_store()/FAISS
 ├─ similarity_search_with_score()
 ├─ _format_cases_for_prompt()
 └─ ChatOpenAI.invoke()
```

- `make_entry_id(cache_key) -> str`  
  Deterministic JSON string; used as vector-store metadata key.

- `PlaybookConfig.from_env() -> PlaybookConfig`  
  Raises: `ValueError` on invalid backend.  
  Used in: `create_playbook_builder`.

- `HashingEmbeddings`  
  Deterministic fallback embeddings; side-effect free.

- `PlaybookBuilder.__init__(config, llm_config)`  
  Checks embeddings; stores `embedding_warning` if fallback invoked.

- `PlaybookBuilder.upsert_history_entry(entry)`  
  Side effects: create/update FAISS index on disk.

- `PlaybookBuilder.generate_playbook(technical_summary, params, cache_key)`  
  Returns `PlaybookResult(plan, cases)`; handles empty store.

- `create_playbook_builder(llm_config=None) -> Optional[PlaybookBuilder]`  
  Returns `None` if configuration missing.

**Core hot paths**: `PlaybookBuilder.upsert_history_entry`, `generate_playbook`.

### `main.py`
```
main()
 └─ run_cli(parse_args())
      └─ _run_single_analysis()
           ├─ ensure_dataset()
           ├─ build_technical_summary()
           ├─ generate_llm_analysis()
           └─ PlaybookBuilder.*
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
  Side effects: print, LLM call, playbook indexing.

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
 │    └─ PlaybookBuilder.*
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
- **Profiles**: No explicit dev/staging/prod toggles; mimic via separate secrets files or environment-specific `PLAYBOOK_INDEX_PATH` values.
- **Secrets**: Store credentials in `.streamlit/secrets.toml` (local) or the Streamlit Cloud UI; rotate by updating secrets and restarting/deploying.
- **Local dev tips**: Set `PLAYBOOK_EMBED_BACKEND=hash` to avoid remote embedding dependencies; adjust `max_age_days=-1` to keep offline caches fresh indefinitely.

---

## 9. Data & Storage
- **CSV caches**: `default_data_path(ticker, interval)` → e.g., `NSEI_1d_ohlc.csv`. Rounded per `round_digits`.
- **History JSON**: `analysis_history.json` (see example entry committed). Stores summary, LLM outputs, playbook payloads.
- **Vector store**: `playbook_index/` containing `index.faiss`, `index.pkl`. Persisted via `FAISS.save_local`.
- **Temporary artifacts**: Streamlit downloads produced on demand (not persisted).
- **Data sources**: Yahoo Finance via `yfinance.download`. Sample offline CSVs included (`NSEI_ohlc.csv`, `NSEBANK_ohlc.csv`) for demos.

---

## 10. Testing & Quality
- No automated tests present. Recommended additions:
  - Unit tests for `DataFetcher.ensure_dataset` (mock yfinance).
  - Unit tests for `build_technical_summary` edge cases.
  - Integration tests for `generate_llm_analysis` using mocked ChatOpenAI.
  - Golden tests for playbook retrieval with deterministic hash embeddings.
- Suggested command: `pytest` (once tests added).  
- Linting: `black`, `isort` available via `requirements.txt`.  
- CI: Not configured; GitHub Actions or similar recommended.

---

## 11. Performance & Cost Footprint
- **TA pipeline**: dominated by pandas operations; typical 1-year daily history (<300 rows) executes in milliseconds.
- **LLM calls**:
  - Default model `deepseek/deepseek-chat-v3.1:free` (no token charges; rate-limited).
  - Example prompt tokens ~1.2k; response ~500 tokens → ensure OpenRouter quotas support this.
- **Playbook retrieval**: FAISS search over small dataset (<1k entries) ~milliseconds; remote embedding fallback adds latency if `PLAYBOOK_EMBED_BACKEND=openai`.
- **Caching**: `ensure_dataset` prevents repeated downloads; `HISTORY.record` enables reusing LLM/playbook outputs (flags displayed in Streamlit).
- **Invalidation**: 
  - CSV refresh triggered by `max_age_days` or `--force-refresh`.
  - Playbook index updated on each history record.

| Task | Estimated Tokens | Est. Cost (USD) | Notes |
| --- | --- | --- | --- |
| `generate_llm_analysis` (default model) | ~1.7k | $0 (free tier) | Check OpenRouter policy; swap model to adjust |
| Playbook plan generation | ~1.2k | Model-dependent | Controlled by `PLAYBOOK_TEMPERATURE` |

---

## 12. Troubleshooting
- **Missing OpenRouter key**: CLI prints exception; Streamlit shows warning in LLM tab. Set `OPENROUTER_API_KEY`.
- **Playbook unavailable**: Requires prior history + embeddings. Check `playbook_error` message (hash fallback warning). Set `PLAYBOOK_EMBED_BACKEND=hash` to silence.
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
   - Leverage cached CSV + indicators.  
   - Persist results to history for RAG context.

---

## Optional Variables
- `{{SUPPORTED_SYMBOLS}}`: Symbol-agnostic; defaults to `^NSEI` but accepts any Yahoo Finance ticker.
- `{{DEFAULT_TIMEFRAMES}}`: CLI defaults (`period=1y`, `interval=1d`); Streamlit options `["1d", "1wk", "1mo"]`.
- `{{OHLC_STORE}}`: `<ticker>_ohlc.csv` in repository root (configurable via `DatasetRequest.data_path`).
- `{{OPENROUTER_MODELS}}`: Default `deepseek/deepseek-chat-v3.1:free`; override via `OPENROUTER_MODEL` or CLI `--model`.
- `{{AGENT_TYPE}}`: Not implemented (direct chain usage).
- `{{VECTOR_STORE}}`: FAISS index stored under `playbook_index/`.
- `{{BACKTEST_WINDOWS}}`: Not implemented; summary uses `key_window=20`, `recent_rows=7`.
- `{{METRICS}}`: Summary shows Latest Close, EMA30/EMA200, RSI14; LLM tables expand on trend/evidence.


---

## 4. Conclusion
This document consolidates the project README, market analysis methodologies, and technical code walkthrough into one cohesive reference. 
Use it as a single source of truth for understanding, running, and extending the project.
