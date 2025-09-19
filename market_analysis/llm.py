"""LLM integration helpers for turning summaries into narrative analysis."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, Mapping, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .settings import get_setting

_DEFAULT_MODEL = "deepseek/deepseek-chat-v3.1:free"
_DEFAULT_TEMPERATURE = 0.1
_DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
_SYSTEM_PROMPT = dedent(
    """
    You are an expert financial technical analyst. Deliver decisive, action-oriented trading
    insights strictly from the provided data. Always choose a directional bias (bullish, bearish,
    or sideways) and state the key invalidation level that would negate it. Avoid vague hedging
    language such as "maybe", "possibly", "might", or "could consider"—use direct present-tense
    statements and keep each table cell concise (<= 12 words).
    """
).strip()


class TableSection(BaseModel):
    """Tabular data returned by the LLM for a particular section."""

    headers: list[str] = Field(..., min_length=1)
    rows: list[list[str]] = Field(default_factory=list)


class AnalysisTables(BaseModel):
    """Structured representation of the five analysis tables."""

    overall_trend: TableSection
    key_evidence: TableSection
    candlestick_patterns: TableSection
    chart_patterns: TableSection
    trade_plan: TableSection


@dataclass(slots=True)
class LLMAnalysisResult:
    """Return object containing markdown and structured tables."""

    markdown: str
    tables: Dict[str, Dict[str, Any]]


def _create_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=AnalysisTables)


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            (
                "human",
                dedent(
                    """
                    Current technical summary:
                    {technical_summary}

                    Please analyse the summary and populate every table in the JSON schema below.

                    Requirements:
                    - State a clear directional bias and preferred trade posture.
                    - Highlight the most decisive evidence first; cite the exact price/indicator level.
                    - Include an invalidation trigger or stop reference wherever risk is discussed.
                    - Keep wording crisp (<= 12 words per table cell) and avoid filler.
                    - Assume the trade setup targets a 2 to 30 week holding window.

                    {format_instructions}
                    """
                ).strip(),
            ),
        ]
    )


def _table_to_markdown(title: str, section: TableSection) -> str:
    headers = section.headers
    if not headers:
        return f"## {title}\nNo data provided."
    header_row = " | ".join(headers)
    separator = " | ".join(["---"] * len(headers))
    body_rows = [" | ".join(row) for row in section.rows] or [
        " | ".join(["—"] * len(headers))
    ]
    table_text = "\n".join(
        [f"| {header_row} |", f"| {separator} |"] + [f"| {row} |" for row in body_rows]
    )
    return f"## {title}\n{table_text}"


def _tables_to_markdown(tables: AnalysisTables) -> str:
    sections = [
        ("Overall Trend Assessment", tables.overall_trend),
        ("Key Technical Evidence", tables.key_evidence),
        ("Candlestick Pattern Analysis", tables.candlestick_patterns),
        ("Chart Pattern Analysis", tables.chart_patterns),
        ("Trade Plan Outline", tables.trade_plan),
    ]
    return "\n\n".join(
        _table_to_markdown(title, section) for title, section in sections
    )


@dataclass(slots=True)
class LLMConfig:
    """Runtime configuration for the LLM-backed analysis chain."""

    model: str = _DEFAULT_MODEL
    temperature: float = _DEFAULT_TEMPERATURE
    api_key: str | None = None
    api_base: str = _DEFAULT_API_BASE

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load API credentials using Streamlit secrets or environment variables."""
        api_key = get_setting("OPENROUTER_API_KEY") or get_setting("LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OpenRouter API key. Set OPENROUTER_API_KEY (or LLM_API_KEY) via Streamlit secrets or environment variables."
            )

        model = get_setting("OPENROUTER_MODEL") or _DEFAULT_MODEL
        temperature_value = get_setting("OPENROUTER_TEMPERATURE")
        api_base = get_setting("OPENROUTER_API_BASE") or _DEFAULT_API_BASE

        temperature = (
            float(temperature_value)
            if temperature_value is not None
            else _DEFAULT_TEMPERATURE
        )

        return cls(
            model=model, temperature=temperature, api_key=api_key, api_base=api_base
        )


def format_prompt(technical_summary: str) -> str:
    """Render the chat prompt that will be submitted to the LLM."""
    parser = _create_parser()
    prompt = _build_prompt()
    formatted = prompt.format_prompt(
        technical_summary=technical_summary,
        format_instructions=parser.get_format_instructions(),
    ).to_messages()
    lines = []
    for message in formatted:
        role = getattr(message, "type", getattr(message, "role", "message"))
        lines.append(f"[{role.upper()}]\n{message.content}")
    return "\n\n".join(lines)


def generate_llm_analysis(
    technical_summary: str, config: Optional[LLMConfig] = None
) -> LLMAnalysisResult:
    """Invoke the analysis chain and return markdown plus structured tables."""
    cfg = config or LLMConfig.from_env()
    parser = _create_parser()
    prompt = _build_prompt().partial(
        format_instructions=parser.get_format_instructions()
    )
    llm = ChatOpenAI(
        model=cfg.model,
        temperature=cfg.temperature,
        openai_api_key=cfg.api_key,
        openai_api_base=cfg.api_base,
    )
    chain = prompt | llm | parser
    tables: AnalysisTables = chain.invoke({"technical_summary": technical_summary})
    markdown = _tables_to_markdown(tables)
    return LLMAnalysisResult(
        markdown=markdown,
        tables={
            "overall_trend": tables.overall_trend.model_dump(),
            "key_evidence": tables.key_evidence.model_dump(),
            "candlestick_patterns": tables.candlestick_patterns.model_dump(),
            "chart_patterns": tables.chart_patterns.model_dump(),
            "trade_plan": tables.trade_plan.model_dump(),
        },
    )


__all__ = [
    "LLMConfig",
    "create_analysis_chain",
    "generate_llm_analysis",
    "format_prompt",
    "LLMAnalysisResult",
]
