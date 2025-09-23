"""Minimal, readable helpers to get JSON from an LLM and render markdown.

Design goals:
- Keep the prompt structure.
- Ask the LLM for a single JSON object with 5 keys.
- Extract JSON with a tiny, understandable parser.
- Convert to markdown with straightforward code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .settings import get_setting

# -----------------------------
# Constants
# -----------------------------
_DEFAULT_MODEL = "deepseek/deepseek-chat-v3.1:free"
_DEFAULT_TEMPERATURE = 0.1
_DEFAULT_API_BASE = "https://openrouter.ai/api/v1"

_SYSTEM_PROMPT = dedent(
    """
You are an expert financial technical analyst.
Use only the supplied summary.
Provide a decisive positional view (2–30 weeks holding).
State clear directional bias, invalidation, and trade plan.
Avoid hedging language.
Output only JSON.
    """
).strip()

_USER_PROMPT = dedent(
    """
    Technical summary:
    {technical_summary}

Structure the JSON with exactly these top-level keys:
\t•\ttechnical_evidence_matrix
\t•\tprobability_outlook
\t•\ttrade_plan
\t•\texpert_observations
\t•\tconclusion

Rules for each section:
\t1.\ttechnical_evidence_matrix
\t•\tObject with "headers": ["Factor","Daily Chart","Hourly Chart"]
\t•\t"rows": each row array with factor and concise evidence
\t•\tFactors: Trend, Support Zones, Resistance Zones, Moving Averages, RSI (14), Patterns
\t2.\tprobability_outlook
\t•\tObject with "headers": ["Scenario","Probability","Trigger Levels","Notes"]
\t•\t"rows": 3 scenarios (sideways consolidation, bullish continuation, bearish breakdown)
\t3.\ttrade_plan
\t•\tObject with "headers": ["Action","Entry Zone","Stop Loss","Target","Notes"]
\t•\t"rows": include safer long, aggressive breakout, partial booking, SL trail
\t4.\texpert_observations
\t•\tArray of 3 short strings with evidence-based insights
\t5.\tconclusion
\t•\tArray of ≤4 short bullets, decisive view with clear entry/avoid instructions

Do not add commentary or extra keys.
Do not use markdown or text outside JSON.
    """
).strip()

_SECTION_TITLES: Dict[str, str] = {
    "technical_evidence_matrix": "Technical Evidence Matrix",
    "probability_outlook": "Probability Outlook",
    "trade_plan": "Trade Plan",
    "expert_observations": "Expert Observations",
    "conclusion": "Conclusion",
}

_EXPECTED_KEYS = tuple(_SECTION_TITLES.keys())

# -----------------------------
# Public API
# -----------------------------


@dataclass(slots=True)
class LLMAnalysisResult:
    markdown: str
    tables: Dict[str, Any]


@dataclass(slots=True)
class LLMConfig:
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


def generate_llm_analysis(
    technical_summary: str, config: Optional[LLMConfig] = None
) -> LLMAnalysisResult:
    """Send the prompt to the LLM, parse its JSON, and return markdown + tables.

    The code path is short and readable:
      1) Build prompt and call the model
      2) Extract the first {...} JSON block from the text
      3) Coerce the payload into the 5 sections we expect
      4) Render markdown
    """
    cfg = config or LLMConfig.from_env()
    llm = ChatOpenAI(
        model=cfg.model,
        temperature=cfg.temperature,
        openai_api_key=cfg.api_key,
        openai_api_base=cfg.api_base,
    )

    prompt = _build_prompt()
    messages = prompt.format_messages(technical_summary=technical_summary)
    resp = llm.invoke(messages)

    # Pull text content in the simplest possible way
    raw_text = _message_text(resp)
    if not raw_text:
        raise RuntimeError("LLM response did not contain any text content.")

    payload = _first_json_object(raw_text)
    tables = _coerce_sections(payload)
    markdown = _render_markdown(tables)
    return LLMAnalysisResult(markdown=markdown, tables=tables)


# -----------------------------
# Prompt helpers
# -----------------------------


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", _USER_PROMPT),
        ]
    )


def format_prompt(technical_summary: str) -> str:
    prompt = _build_prompt()
    messages = prompt.format_messages(technical_summary=technical_summary)
    return "\n\n".join(
        f"[{getattr(m, 'type', getattr(m, 'role', 'message')).upper()}]\n{m.content}"
        for m in messages
    )


# -----------------------------
# Step 1: get plain text from the LLM message
# -----------------------------


def _message_text(message: Any) -> str:
    """Return message.content if it's a string; otherwise stringify.
    This keeps it very simple.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    # Fallback: stringified object
    return (str(content) if content is not None else str(message) or "").strip()


# -----------------------------
# Step 2: extract a JSON object from raw text
# -----------------------------

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


def _first_json_object(text: str) -> Dict[str, Any]:
    """Find the first {...} block and parse it. Apply minimal repairs.

    We purposely keep this tiny and understandable:
      - Try fenced ```json blocks first
      - Else slice from first '{' to last '}'
      - Minimal repairs: normalize smart quotes/dashes and remove trailing commas
    """
    # 1) Try fenced blocks
    m = _JSON_FENCE.search(text)
    candidate: Optional[str] = m.group(1) if m else None

    # 2) Otherwise take widest brace slice
    if not candidate:
        s = text.find("{")
        e = text.rfind("}")
        candidate = text[s : e + 1] if (s != -1 and e != -1 and e > s) else None

    if not candidate:
        raise ValueError("No JSON object found in LLM response.")

    def _repairs(s: str) -> str:
        s = (
            s.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
        s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        s = (
            s.replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("–", "-")
            .replace("—", "-")
        )
        s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
        return s

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        candidate = _repairs(candidate)
        return json.loads(candidate)


# -----------------------------
# Step 3: coerce to expected sections
# -----------------------------


def _norm(x: Any) -> str:
    return " ".join(("" if x is None else str(x)).strip().split())


def _coerce_sections(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # Tables
    for key in ("technical_evidence_matrix", "probability_outlook", "trade_plan"):
        sec = payload.get(key)
        if not isinstance(sec, dict):
            out[key] = {"headers": ["Info"], "rows": [["No data provided."]]}
            continue
        headers = [h for h in map(_norm, sec.get("headers", [])) if h]
        if not headers:
            headers = ["Info"]
        rows_in = sec.get("rows", [])
        rows: List[List[str]] = []
        if isinstance(rows_in, list):
            for row in rows_in:
                if isinstance(row, list):
                    vals = [
                        _norm(v)
                        for v in (row + [""] * (len(headers) - len(row)))[
                            : len(headers)
                        ]
                    ]
                    if any(vals):
                        rows.append(vals)
        if not rows:
            rows = [["No data provided."] + [""] * (len(headers) - 1)]
        out[key] = {"headers": headers, "rows": rows}

    # Lists
    for key in ("expert_observations", "conclusion"):
        sec = payload.get(key)
        items: List[str] = []
        if isinstance(sec, list):
            items = [i for i in map(_norm, sec) if i]
        elif isinstance(sec, dict) and isinstance(sec.get("items"), list):
            items = [i for i in map(_norm, sec.get("items", [])) if i]
        if not items:
            items = ["No data provided."]
        out[key] = {"items": items}

    return out


# -----------------------------
# Step 4: markdown rendering
# -----------------------------


def _render_table(title: str, table: Dict[str, Any]) -> str:
    headers: List[str] = table.get("headers", [])
    rows: List[List[str]] = table.get("rows", [])
    if not headers:
        headers = ["Info"]
    if not rows:
        rows = [["—"]]
    head = " | ".join(headers)
    sep = " | ".join(["---"] * len(headers))
    body = [" | ".join(r) for r in rows]
    lines = [f"## {title}", f"| {head} |", f"| {sep} |"] + [f"| {b} |" for b in body]
    return "\n".join(lines)


def _render_list(title: str, items: List[str]) -> str:
    if not items:
        items = ["No data provided."]
    bullets = "\n".join(f"- {i}" for i in items)
    return f"## {title}\n{bullets}"


def _render_markdown(tables: Dict[str, Any]) -> str:
    parts: List[str] = []
    # Tables
    for key in ("technical_evidence_matrix", "probability_outlook", "trade_plan"):
        parts.append(_render_table(_SECTION_TITLES[key], tables.get(key, {})))
    # Lists
    for key in ("expert_observations", "conclusion"):
        items = tables.get(key, {}).get("items", [])
        parts.append(_render_list(_SECTION_TITLES[key], items))
    return "\n\n".join(parts)


__all__ = [
    "LLMConfig",
    "LLMAnalysisResult",
    "format_prompt",
    "generate_llm_analysis",
]
