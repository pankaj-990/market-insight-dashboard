"""Strategy playbook builder powered by retrieval-augmented generation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .llm import LLMConfig
from .settings import get_setting


def make_entry_id(cache_key: Mapping[str, Any]) -> str:
    """Return a deterministic identifier for a cached analysis entry."""
    return json.dumps(cache_key, sort_keys=True)


@dataclass(slots=True)
class PlaybookConfig:
    """Configuration for the strategy playbook index and generation."""

    index_path: Path
    embedding_model: str
    embedding_backend: str
    embedding_dim: int
    top_k: int
    temperature: float
    case_summary_chars: int

    @classmethod
    def from_env(cls) -> "PlaybookConfig":
        """Load configuration using Streamlit secrets or environment variables (with defaults)."""
        index_path = Path(get_setting("PLAYBOOK_INDEX_PATH") or "playbook_index")
        embedding_model = (
            get_setting("PLAYBOOK_EMBED_MODEL") or "text-embedding-3-small"
        )
        embedding_backend = (get_setting("PLAYBOOK_EMBED_BACKEND") or "auto").lower()
        embedding_dim_raw = get_setting("PLAYBOOK_EMBED_DIM")
        top_k_raw = get_setting("PLAYBOOK_TOP_K")
        temperature_raw = get_setting("PLAYBOOK_TEMPERATURE")
        case_summary_chars_raw = get_setting("PLAYBOOK_CASE_SNIPPET")

        embedding_dim = int(embedding_dim_raw) if embedding_dim_raw is not None else 512
        top_k = int(top_k_raw) if top_k_raw is not None else 3
        temperature = float(temperature_raw) if temperature_raw is not None else 0.0
        case_summary_chars = (
            int(case_summary_chars_raw) if case_summary_chars_raw is not None else 600
        )
        return cls(
            index_path=index_path,
            embedding_model=embedding_model,
            embedding_backend=embedding_backend,
            embedding_dim=max(32, embedding_dim),
            top_k=max(1, top_k),
            temperature=temperature,
            case_summary_chars=max(120, case_summary_chars),
        )


@dataclass(slots=True)
class PlaybookCase:
    """A retrieved historical case used to enrich the playbook."""

    rank: int
    entry_id: str
    ticker: str | None
    period: str | None
    interval: str | None
    recorded: str | None
    last_timestamp: str | None
    similarity: float | None
    summary_snippet: str
    data_source: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "entry_id": self.entry_id,
            "ticker": self.ticker,
            "period": self.period,
            "interval": self.interval,
            "recorded": self.recorded,
            "last_timestamp": self.last_timestamp,
            "similarity": self.similarity,
            "summary_snippet": self.summary_snippet,
            "data_source": self.data_source,
        }


class HashingEmbeddings(Embeddings):
    """Lightweight, deterministic embedding implementation based on token hashing."""

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = max(32, dimension)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in self._tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass(slots=True)
class PlaybookResult:
    """Result returned by the playbook builder."""

    plan: Optional[str]
    cases: Sequence[PlaybookCase]

    def to_payload(self) -> dict[str, Any]:
        return {
            "plan": self.plan,
            "cases": [case.to_payload() for case in self.cases],
        }


_PLAYBOOK_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent(
                """
                You are a senior quantitative trading mentor. Combine the current technical
                summary with historically similar cases to propose a concise strategy
                playbook. Reference the retrieved case numbers where useful.
                """
            ).strip(),
        ),
        (
            "human",
            dedent(
                """
                Current technical summary:
                {current_summary}

                Relevant historical cases:
                {historical_cases}

                Produce a playbook with three markdown sections:
                1. "Historical Patterns" – 3-5 bullet points referencing the case numbers and
                   highlighting parallels and divergences.
                2. "Strategy Recommendations" – actionable trade ideas (entry, stops, targets)
                   influenced by those cases.
                3. "Risk Watchlist" – key risks or invalidation cues to monitor.

                Keep each bullet concise (<=2 lines). If there are no suitable cases, explain
                that the playbook is unavailable.
                """
            ).strip(),
        ),
    ]
)


class PlaybookBuilder:
    """Manage retrieval storage and playbook generation."""

    def __init__(self, config: PlaybookConfig, llm_config: LLMConfig):
        self.config = config
        self.llm_config = llm_config
        self._embeddings, warning = self._initialise_embeddings(config, llm_config)
        self.embedding_warning = warning
        self._llm = ChatOpenAI(
            model=llm_config.model,
            temperature=config.temperature,
            openai_api_key=llm_config.api_key,
            openai_api_base=llm_config.api_base,
        )
        self._store: Optional[FAISS] = None

    def _initialise_embeddings(
        self, config: PlaybookConfig, llm_config: LLMConfig
    ) -> Tuple[Embeddings, Optional[str]]:
        backend = config.embedding_backend

        def _hash_backend(
            message: Optional[str] = None,
        ) -> Tuple[Embeddings, Optional[str]]:
            return HashingEmbeddings(dimension=config.embedding_dim), message

        if backend in {"auto", "openai"}:
            try:
                embeddings = OpenAIEmbeddings(
                    model=config.embedding_model,
                    openai_api_key=llm_config.api_key,
                    openai_api_base=llm_config.api_base,
                )
                # Lightweight connectivity check to fail fast when embeddings aren't supported.
                embeddings.embed_query("playbook-initialisation-check")
                return embeddings, None
            except Exception as exc:
                if backend == "openai":
                    raise
                warning = (
                    f"Remote embeddings unavailable ({exc}). Falling back to hash-based embeddings. "
                    "Set PLAYBOOK_EMBED_BACKEND=hash to silence this warning or "
                    "PLAYBOOK_EMBED_BACKEND=openai after configuring a compatible embedding endpoint."
                )
                return _hash_backend(warning)

        if backend == "hash":
            return _hash_backend(None)

        raise ValueError(
            "Unsupported PLAYBOOK_EMBED_BACKEND value. Use 'auto', 'openai', or 'hash'."
        )

    def _load_store(self) -> Optional[FAISS]:
        if self._store is not None:
            return self._store
        if self.config.index_path.exists():
            self._store = FAISS.load_local(
                str(self.config.index_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        return self._store

    def _save_store(self, store: FAISS) -> None:
        self.config.index_path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(self.config.index_path))

    def _render_document(
        self, summary: str, llm_text: Optional[str], entry: Mapping[str, Any]
    ) -> str:
        params = entry.get("params", {})
        lines = [
            f"Ticker: {params.get('ticker', 'unknown')}",
            f"Period: {params.get('period', 'unknown')} | Interval: {params.get('interval', 'unknown')}",
            f"Recorded: {entry.get('timestamp', 'unknown')}",
            "",
            "Technical Summary:",
            summary.strip(),
        ]
        if llm_text:
            lines.extend(["", "LLM Narrative:", llm_text.strip()])
        notes = entry.get("notes")
        if notes:
            lines.extend(["", "Analyst Notes:", str(notes).strip()])
        return "\n".join(lines)

    def upsert_history_entry(self, entry: Mapping[str, Any]) -> None:
        """Add a history entry to the vector index for future retrieval."""
        summary = entry.get("technical_summary")
        if not summary:
            return
        llm_text = entry.get("llm_text")
        cache_key = entry.get("cache_key", {})
        entry_id = entry.get("entry_id") or make_entry_id(cache_key)
        params = entry.get("params", {})
        metadata = {
            "entry_id": entry_id,
            "ticker": params.get("ticker"),
            "period": params.get("period"),
            "interval": params.get("interval"),
            "recorded": entry.get("timestamp"),
            "last_timestamp": cache_key.get("last_timestamp"),
            "max_age_days": params.get("max_age_days"),
            "data_source": entry.get("data_source"),
            "data_path": params.get("data_path"),
            "summary": summary,
            "has_llm": bool(llm_text),
        }
        document = Document(
            page_content=self._render_document(summary, llm_text, entry),
            metadata=metadata,
        )
        store = self._load_store()
        try:
            if store is None:
                store = FAISS.from_documents([document], self._embeddings)
            else:
                store.add_documents([document])
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to embed playbook entry: {exc}") from exc
        self._save_store(store)
        self._store = store

    def _format_cases_for_prompt(self, cases: Sequence[PlaybookCase]) -> str:
        if not cases:
            return "No comparable historical cases were retrieved."
        blocks = []
        for case in cases:
            details = [
                f"Case {case.rank}",
                f"Ticker: {case.ticker or 'unknown'}",
            ]
            if case.period or case.interval:
                details.append(
                    "Window: " + " / ".join(filter(None, [case.period, case.interval]))
                )
            if case.recorded:
                details.append(f"Recorded: {case.recorded}")
            if case.last_timestamp:
                details.append(f"Last candle: {case.last_timestamp}")
            if case.similarity is not None:
                details.append(f"Similarity score: {case.similarity:.3f}")
            blocks.append(
                " | ".join(details) + f"\nSummary snippet:\n{case.summary_snippet}"
            )
        return "\n\n".join(blocks)

    def generate_playbook(
        self,
        *,
        technical_summary: str,
        params: Mapping[str, Any],
        cache_key: Mapping[str, Any],
    ) -> PlaybookResult:
        """Retrieve similar cases and craft a strategy playbook."""
        store = self._load_store()
        if store is None:
            return PlaybookResult(plan=None, cases=[])

        entry_id = make_entry_id(cache_key)
        raw_results = store.similarity_search_with_score(
            technical_summary, k=max(self.config.top_k * 2, self.config.top_k)
        )
        cases: list[PlaybookCase] = []
        for doc, score in raw_results:
            metadata = doc.metadata or {}
            if metadata.get("entry_id") == entry_id:
                continue
            summary = metadata.get("summary") or doc.page_content
            snippet = summary.strip().replace("\n\n", "\n")
            snippet = snippet[: self.config.case_summary_chars].strip()
            cases.append(
                PlaybookCase(
                    rank=len(cases) + 1,
                    entry_id=metadata.get("entry_id", ""),
                    ticker=metadata.get("ticker"),
                    period=metadata.get("period"),
                    interval=metadata.get("interval"),
                    recorded=metadata.get("recorded"),
                    last_timestamp=metadata.get("last_timestamp"),
                    similarity=float(score) if score is not None else None,
                    summary_snippet=snippet,
                    data_source=metadata.get("data_source"),
                )
            )
            if len(cases) >= self.config.top_k:
                break

        if not cases:
            return PlaybookResult(plan=None, cases=[])

        prompt_inputs = {
            "current_summary": technical_summary.strip(),
            "historical_cases": self._format_cases_for_prompt(cases),
        }
        prompt = _PLAYBOOK_PROMPT.format_prompt(**prompt_inputs)
        plan = self._llm.invoke(prompt.to_messages())
        plan_text = plan.content if hasattr(plan, "content") else str(plan)
        return PlaybookResult(plan=plan_text, cases=cases)


def create_playbook_builder(
    llm_config: Optional[LLMConfig] = None,
) -> Optional[PlaybookBuilder]:
    """Return a ready-to-use playbook builder when configuration is present."""
    try:
        config = PlaybookConfig.from_env()
    except Exception:
        return None

    try:
        resolved_llm_config = llm_config or LLMConfig.from_env()
    except Exception:
        return None

    try:
        return PlaybookBuilder(config=config, llm_config=resolved_llm_config)
    except Exception:
        return None


__all__ = [
    "PlaybookBuilder",
    "PlaybookCase",
    "PlaybookConfig",
    "PlaybookResult",
    "create_playbook_builder",
    "make_entry_id",
]
