"""Simplified helpers for generating LLM-backed analysis tables."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from .settings import get_setting

_DEFAULT_MODEL = "deepseek/deepseek-chat-v3.1:free"
_DEFAULT_TEMPERATURE = 0.1
_DEFAULT_API_BASE = "https://openrouter.ai/api/v1"

_SYSTEM_PROMPT = dedent(
    """
    You are an expert financial technical analyst. Use the supplied summary only and
    state a decisive directional view with a clear invalidation level. Avoid hedging
    language and keep every bullet short (<= 12 words).
    """
).strip()

_USER_PROMPT = dedent(
    """
    Technical summary:
    {technical_summary}

    Produce a JSON object with exactly these keys:
    - overall_trend
    - key_evidence
    - candlestick_patterns
    - chart_patterns
    - trade_plan

    Each value must be an object with:
    - "headers": an array of short column titles (1-4 words each)
    - "rows": an array of rows, where every row is an array of strings equal in length
      to the headers. Keep values concise and action-focused.

    Rules:
    - Include at least one row per table.
    - Use precise price/indicator levels whenever referenced.
    - Do not add extra keys or commentary.
    - Respond with JSON only (no markdown fences).
    """
).strip()

_SECTION_TITLES = {
    "overall_trend": "Overall Trend Assessment",
    "key_evidence": "Key Technical Evidence",
    "candlestick_patterns": "Candlestick Pattern Analysis",
    "chart_patterns": "Chart Pattern Analysis",
    "trade_plan": "Trade Plan Outline",
}

_EXPECTED_TABLE_KEYS = tuple(_SECTION_TITLES.keys())
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass(slots=True)
class LLMAnalysisResult:
    """Return object containing markdown and structured tables."""

    markdown: str
    tables: Dict[str, Dict[str, Any]]


@dataclass(slots=True)
class LLMConfig:
    """Runtime configuration for the LLM-backed analysis chain."""

    model: str = _DEFAULT_MODEL
    temperature: float = _DEFAULT_TEMPERATURE
    api_key: Optional[str] = None
    api_base: str = _DEFAULT_API_BASE

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = get_setting("OPENROUTER_API_KEY") or get_setting("LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OpenRouter API key. Set OPENROUTER_API_KEY (or LLM_API_KEY)."
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


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", _USER_PROMPT),
        ]
    )


def format_prompt(technical_summary: str) -> str:
    """Render the chat prompt that will be submitted to the LLM."""
    prompt = _build_prompt()
    messages = prompt.format_messages(technical_summary=technical_summary)
    rendered: List[str] = []
    for message in messages:
        role = getattr(message, "type", getattr(message, "role", "message"))
        rendered.append(f"[{role.upper()}]\n{message.content}")
    return "\n\n".join(rendered)


def _coerce_message_text(message: Any) -> Optional[str]:
    if message is None:
        return None

    if isinstance(message, BaseMessage):
        content = message.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_value = str(block.get("text", "")).strip()
                    if text_value:
                        parts.append(text_value)
                elif isinstance(block, str) and block.strip():
                    parts.append(block.strip())
            if parts:
                return "\n".join(parts)

        additional = getattr(message, "additional_kwargs", {}) or {}
        function_call = additional.get("function_call")
        if isinstance(function_call, dict):
            arguments = function_call.get("arguments")
            if isinstance(arguments, str) and arguments.strip():
                return arguments.strip()

        tool_calls = additional.get("tool_calls") or []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            arguments = fn.get("arguments")
            if isinstance(arguments, str) and arguments.strip():
                return arguments.strip()

    if isinstance(message, str) and message.strip():
        return message.strip()

    text_fallback = str(message).strip()
    return text_fallback or None


def _repair_json_text(text: str) -> str:
    repaired = text
    repaired = re.sub(r']\s*"', "]", repaired)
    repaired = re.sub(r'}\s*"', "}", repaired)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired.strip()


def _extract_json_text(raw_text: str) -> str:
    candidates: List[str] = []
    stripped = raw_text.strip()
    if stripped:
        candidates.append(stripped)

    for match in _JSON_BLOCK_RE.finditer(raw_text):
        snippet = match.group(1).strip()
        if snippet:
            candidates.append(snippet)

    brace_start = stripped.find("{")
    brace_end = stripped.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidates.append(stripped[brace_start : brace_end + 1])

    ordered: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)

    for candidate in ordered:
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return candidate

    for candidate in ordered:
        repaired = _repair_json_text(candidate)
        if not repaired or repaired in seen:
            continue
        try:
            json.loads(repaired)
        except json.JSONDecodeError:
            continue
        return repaired

    raise ValueError("LLM response does not contain valid JSON content.")


def _normalise_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.strip().split())


def _ensure_rows(headers: List[str], rows: Any) -> List[List[str]]:
    if not isinstance(rows, list):
        rows_iterable: List[List[str]] = []
    else:
        rows_iterable = []
        for row in rows:
            if not isinstance(row, list):
                continue
            values = [_normalise_cell(cell) for cell in row]
            if not any(values):
                continue
            if len(values) < len(headers):
                values.extend([""] * (len(headers) - len(values)))
            elif len(values) > len(headers):
                values = values[: len(headers)]
            rows_iterable.append(values)

    if not rows_iterable:
        rows_iterable.append(["—"] * len(headers))
    return rows_iterable


def _parse_tables(raw_text: str) -> Dict[str, Dict[str, Any]]:
    json_text = _extract_json_text(raw_text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Failed to decode JSON from LLM response.") from exc

    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object.")

    tables: Dict[str, Dict[str, Any]] = {}
    for key in _EXPECTED_TABLE_KEYS:
        section = payload.get(key)
        if not isinstance(section, dict):
            raise ValueError(f"Missing or invalid table '{key}'.")

        headers_raw = section.get("headers", [])
        if not isinstance(headers_raw, list):
            raise ValueError(f"Headers for '{key}' must be an array of strings.")
        headers = [
            _normalise_cell(header) for header in headers_raw if _normalise_cell(header)
        ]
        if not headers:
            raise ValueError(f"Table '{key}' must include at least one header.")

        rows_raw = section.get("rows", [])
        tables[key] = {
            "headers": headers,
            "rows": _ensure_rows(headers, rows_raw),
        }

    return tables


def _table_to_markdown(title: str, section: Dict[str, Any]) -> str:
    headers: List[str] = section.get("headers", [])
    rows: List[List[str]] = section.get("rows", [])
    header_row = " | ".join(headers)
    separator = " | ".join(["---"] * len(headers))
    body = [" | ".join(row) for row in rows]
    if not body:
        body = [" | ".join(["—"] * len(headers))]
    table_lines = [f"| {header_row} |", f"| {separator} |"]
    table_lines.extend(f"| {row} |" for row in body)
    return f"## {title}\n" + "\n".join(table_lines)


def _tables_to_markdown(tables: Dict[str, Dict[str, Any]]) -> str:
    sections = []
    for key in _EXPECTED_TABLE_KEYS:
        section = tables.get(key, {"headers": [], "rows": []})
        if not section.get("headers"):
            section = {
                "headers": ["Info"],
                "rows": [["No data provided."]],
            }
        sections.append(_table_to_markdown(_SECTION_TITLES[key], section))
    return "\n\n".join(sections)


def generate_llm_analysis(
    technical_summary: str, config: Optional[LLMConfig] = None
) -> LLMAnalysisResult:
    cfg = config or LLMConfig.from_env()
    llm = ChatOpenAI(
        model=cfg.model,
        temperature=cfg.temperature,
        openai_api_key=cfg.api_key,
        openai_api_base=cfg.api_base,
    )

    prompt = _build_prompt()
    messages = prompt.format_messages(technical_summary=technical_summary)
    message = llm.invoke(messages)
    raw_payload = _coerce_message_text(message)
    if not raw_payload:
        raise RuntimeError("LLM response did not contain any text content.")

    try:
        tables = _parse_tables(raw_payload)
    except ValueError as exc:
        snippet = raw_payload.strip().replace("\n", " ")[:400]
        raise RuntimeError(
            "Failed to parse structured analysis tables from the LLM response. "
            f"Captured snippet: {snippet}"
        ) from exc

    markdown = _tables_to_markdown(tables)
    return LLMAnalysisResult(markdown=markdown, tables=tables)


__all__ = [
    "LLMConfig",
    "generate_llm_analysis",
    "format_prompt",
    "LLMAnalysisResult",
]
