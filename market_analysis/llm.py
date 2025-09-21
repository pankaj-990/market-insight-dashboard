"""LLM integration helpers for turning summaries into narrative analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
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


def _coerce_message_text(message: Any) -> Optional[str]:
    """Extract the textual payload from an LLM message.

    Some OpenAI-compatible models (via OpenRouter) return tool/function calls or
    multi-part content blocks instead of a simple string. We defensively probe
    the common locations for the serialized JSON payload that our parser
    expects.
    """

    if message is None:
        return None

    if isinstance(message, str):
        text_value = message.strip()
        return text_value or None

    if isinstance(message, BaseMessage):
        content = message.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            # Some providers emit a list of text blocks for multi-part replies.
            texts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_value = str(block.get("text", "")).strip()
                    if text_value:
                        texts.append(text_value)
                elif isinstance(block, str) and block.strip():
                    texts.append(block.strip())
            if texts:
                return "\n".join(texts)

        additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
        function_call = additional_kwargs.get("function_call")
        if isinstance(function_call, dict):
            arguments = function_call.get("arguments")
            if isinstance(arguments, str) and arguments.strip():
                return arguments.strip()

        tool_calls = additional_kwargs.get("tool_calls") or []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function_details = call.get("function") or {}
            arguments = function_details.get("arguments")
            if isinstance(arguments, str) and arguments.strip():
                return arguments.strip()

        response_metadata = getattr(message, "response_metadata", {}) or {}
        metadata_message = response_metadata.get("message")
        if isinstance(metadata_message, str) and metadata_message.strip():
            return metadata_message.strip()

    text_fallback = str(message).strip()
    return text_fallback or None


_TABLE_KEYS = (
    "overall_trend",
    "key_evidence",
    "candlestick_patterns",
    "chart_patterns",
    "trade_plan",
)


def _normalise_tables_payload(raw_payload: str) -> str:
    """Repair common JSON-wrapping mistakes before handing off to the parser."""

    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError:
        return raw_payload

    collected: Dict[str, Any] = {}

    def _collect(obj: Any) -> None:
        if isinstance(obj, dict):
            for key in _TABLE_KEYS:
                if key in obj and key not in collected:
                    collected[key] = obj[key]
            for value in obj.values():
                _collect(value)
        elif isinstance(obj, list):
            for item in obj:
                _collect(item)

    _collect(parsed)

    if not all(key in collected for key in _TABLE_KEYS):
        return raw_payload

    def _coerce_section(section: Any) -> Optional[Dict[str, Any]]:
        if isinstance(section, dict):
            return section if {"headers", "rows"} <= section.keys() else None
        if isinstance(section, list):
            for item in section:
                if isinstance(item, dict) and {"headers", "rows"} <= item.keys():
                    return item
        return None

    normalised: Dict[str, Dict[str, Any]] = {}
    for key in _TABLE_KEYS:
        section = _coerce_section(collected[key])
        if section is None:
            return raw_payload
        normalised[key] = section

    try:
        return json.dumps(normalised)
    except (TypeError, ValueError):
        return raw_payload


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
    message = (prompt | llm).invoke({"technical_summary": technical_summary})

    raw_payload = _coerce_message_text(message)
    if not raw_payload:
        raise RuntimeError(
            "LLM response did not contain any parsable content. "
            "Verify that the selected model returns text responses or disable function calling."
        )

    normalised_payload = _normalise_tables_payload(raw_payload)

    try:
        tables: AnalysisTables = parser.parse(normalised_payload)
    except OutputParserException as exc:
        snippet = raw_payload[:500]
        raise RuntimeError(
            "Failed to parse structured analysis tables from the LLM response. "
            "Inspect the model output (truncated below) and adjust the prompt/model if needed:\n"
            f"{snippet}"
        ) from exc

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
    "generate_llm_analysis",
    "format_prompt",
    "LLMAnalysisResult",
]
