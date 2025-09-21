"""Motif retrieval utilities built on LangChain vector stores."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Mapping, Optional, Protocol,
                    Sequence)

import numpy as np
import pandas as pd

DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = (
    "Open",
    "High",
    "Low",
    "Close",
    "EMA_30",
    "EMA_200",
    "RSI_14",
)

SUPPORTED_NORMALIZATIONS: tuple[str, ...] = ("none", "zscore", "minmax", "first")

try:  # LangChain >= 0.0.335 splits community integrations
    from langchain_community.vectorstores import FAISS, Chroma
except ImportError:  # pragma: no cover - fallback for older LangChain versions
    from langchain.vectorstores import FAISS, Chroma  # type: ignore

try:  # LangChain >= 0.1
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.docstore.document import Document  # type: ignore

try:
    from langchain_community.docstore.in_memory import InMemoryDocstore
except ImportError:  # pragma: no cover
    InMemoryDocstore = None  # type: ignore


class WindowEmbedder(Protocol):
    """Lightweight protocol for transforming OHLC windows into embeddings."""

    def embed(self, windows: Sequence[Sequence[float]]) -> List[List[float]]:
        """Return embeddings for the supplied batch of numeric windows."""

    def embed_one(self, window: Sequence[float]) -> List[float]:
        """Return the embedding for a single numeric window."""


@dataclass(slots=True)
class IdentityWindowEmbedder:
    """Default embedder that keeps numeric feature vectors unchanged."""

    dtype: Any = np.float32

    def embed(self, windows: Sequence[Sequence[float]]) -> List[List[float]]:
        return [self._as_list(window) for window in windows]

    def embed_one(self, window: Sequence[float]) -> List[float]:
        return self._as_list(window)

    def _as_list(self, window: Sequence[float]) -> List[float]:
        array = np.asarray(window, dtype=self.dtype)
        if array.ndim != 1:
            raise ValueError("Each window must be a 1D feature vector.")
        return array.astype(np.float32).tolist()


@dataclass(slots=True)
class MotifMetadata:
    """Metadata stored alongside each motif embedding."""

    ticker: str
    timeframe: str
    start_date: str
    end_date: str
    regime: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "regime": self.regime,
        }
        if self.extra:
            data.update(self.extra)
        return data

    def describe(self) -> str:
        """Compact human readable summary for diagnostic purposes."""
        return (
            f"{self.ticker} {self.timeframe} {self.start_date}->{self.end_date}"
            f" [{self.regime}]"
        )


@dataclass(slots=True)
class MotifMatch:
    """Container holding a similarity score and associated motif metadata."""

    motif_id: str
    score: float
    metadata: Mapping[str, Any]


@dataclass(slots=True)
class MotifQueryResult:
    """Summary of a motif retrieval query executed against recent windows."""

    matches: List[MotifMatch]
    query_metadata: MotifMetadata
    query_vector: List[float]
    indexed_count: int
    window_size: int
    feature_columns: Sequence[str]
    backend: str
    normalization: str = "none"
    skipped_windows: int = 0
    filters: Dict[str, Any] = field(default_factory=dict)


class MotifVectorStore:
    """Shared interface for motif indexing and retrieval."""

    def __init__(
        self,
        backend: str,
        vectorstore: Any,
        *,
        embedder: Optional[WindowEmbedder] = None,
    ) -> None:
        self.backend = backend
        self.vectorstore = vectorstore
        self.embedder = embedder or IdentityWindowEmbedder()

    @classmethod
    def faiss(
        cls,
        dimension: int,
        *,
        embedder: Optional[WindowEmbedder] = None,
    ) -> "MotifVectorStore":
        """Create an empty FAISS-backed motif vector store."""

        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency missing at runtime
            raise RuntimeError(
                "faiss-cpu is required for FAISS-backed motif retrieval."
            ) from exc

        if InMemoryDocstore is None:  # pragma: no cover - defensive guard
            raise RuntimeError(
                "InMemoryDocstore is unavailable; verify LangChain installation."
            )

        index = faiss.IndexFlatL2(dimension)
        docstore = InMemoryDocstore({})
        faiss_store = FAISS(
            embedding_function=None,  # type: ignore[arg-type]
            index=index,
            docstore=docstore,
            index_to_docstore_id={},
        )
        return cls("faiss", faiss_store, embedder=embedder)

    @classmethod
    def chroma(
        cls,
        *,
        collection_name: str = "motifs",
        persist_directory: Optional[str] = None,
        embedder: Optional[WindowEmbedder] = None,
        **client_kwargs: Any,
    ) -> "MotifVectorStore":
        """Create a Chroma-backed motif vector store."""

        chroma_store = Chroma(
            collection_name=collection_name,
            embedding_function=None,  # type: ignore[arg-type]
            persist_directory=persist_directory,
            **client_kwargs,
        )
        return cls("chroma", chroma_store, embedder=embedder)

    def index_windows(
        self,
        windows: Sequence[Sequence[float]],
        metadatas: Sequence[MotifMetadata],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Insert motif vectors and metadata into the underlying store."""

        if len(windows) != len(metadatas):
            raise ValueError("windows and metadatas must have identical lengths")

        embeddings = self.embedder.embed(windows)
        resolved_ids = (
            list(ids) if ids is not None else [str(uuid.uuid4()) for _ in embeddings]
        )
        metadata_dicts = []
        documents = []
        for meta, doc_id in zip(metadatas, resolved_ids):
            payload = meta.as_dict()
            payload["id"] = doc_id
            metadata_dicts.append(payload)
            documents.append(
                Document(page_content=json.dumps(payload), metadata=dict(payload))
            )

        if self.backend == "faiss":
            return self._faiss_add(embeddings, documents, metadata_dicts, resolved_ids)
        if self.backend == "chroma":
            return self._chroma_add(embeddings, documents, metadata_dicts, resolved_ids)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def _faiss_add(
        self,
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[Document],
        metadatas: Sequence[Mapping[str, Any]],
        ids: Sequence[str],
    ) -> List[str]:
        if not hasattr(self.vectorstore, "index"):
            raise RuntimeError("FAISS vector store is improperly configured.")

        # Convert to float32 arrays and add to FAISS index.
        vectors = np.asarray(embeddings, dtype=np.float32)
        previous_total = self.vectorstore.index.ntotal
        self.vectorstore.index.add(vectors)

        # Persist documents and metadata inside the docstore mapping.
        for position, (doc, metadata, doc_id) in enumerate(
            zip(documents, metadatas, ids)
        ):
            doc.metadata = dict(metadata)
            doc.metadata["_docstore_id"] = doc_id
            self.vectorstore.docstore.add({doc_id: doc})
            self.vectorstore.index_to_docstore_id[previous_total + position] = doc_id

        return list(ids)

    def _chroma_add(
        self,
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[Document],
        metadatas: Sequence[Mapping[str, Any]],
        ids: Sequence[str],
    ) -> List[str]:
        self.vectorstore._collection.add(  # type: ignore[attr-defined]
            embeddings=embeddings,
            metadatas=[dict(meta) for meta in metadatas],
            ids=list(ids),
            documents=[doc.page_content for doc in documents],
        )
        return list(ids)

    def query_nearest_neighbors(
        self,
        window: Sequence[float],
        *,
        top_k: int = 5,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        search_k: Optional[int] = None,
    ) -> List[MotifMatch]:
        """Return top-k similar motifs, optionally constrained by metadata."""

        embedding = self.embedder.embed_one(window)
        k = top_k if search_k is None else max(top_k, search_k)

        results = self._similarity_search(
            embedding, k=k, metadata_filter=metadata_filter
        )

        matches: List[MotifMatch] = []
        for doc, score, doc_id in results:
            if metadata_filter and not _metadata_matches(doc.metadata, metadata_filter):
                continue
            matches.append(
                MotifMatch(motif_id=doc_id, score=score, metadata=doc.metadata)
            )
            if len(matches) >= top_k:
                break

        return matches

    def _similarity_search(
        self,
        embedding: Sequence[float],
        *,
        k: int,
        metadata_filter: Optional[Mapping[str, Any]] = None,
    ) -> List[tuple[Document, float, str]]:
        if self.backend == "faiss":
            raw = _run_faiss_similarity(self.vectorstore, embedding, k)
        else:
            if metadata_filter:
                raw = self.vectorstore.similarity_search_by_vector_with_relevance_scores(  # type: ignore[attr-defined]
                    embedding,
                    k=k,
                    filter=dict(metadata_filter),
                )
            else:
                raw = self.vectorstore.similarity_search_by_vector_with_relevance_scores(  # type: ignore[attr-defined]
                    embedding,
                    k=k,
                )

        results: List[tuple[Document, float, str]] = []
        for item in raw:
            if isinstance(item, tuple) and len(item) == 3:
                doc, score, doc_id = item
            elif isinstance(item, tuple) and len(item) == 2:
                doc, score = item
                doc_id = getattr(doc, "id", getattr(doc, "metadata", {}).get("id", ""))
            else:
                doc, score = item, 0.0
                doc_id = getattr(doc, "id", "")
            if not doc_id:
                doc_id = _resolve_doc_id(self.backend, self.vectorstore, doc)
            results.append((doc, float(score), doc_id))
        return results


def _resolve_doc_id(backend: str, vectorstore: Any, doc: Document) -> str:
    if backend == "faiss":
        # When FAISS returns documents, the docstore id is stored inside metadata.
        if "_docstore_id" in doc.metadata:
            return str(doc.metadata["_docstore_id"])
    return doc.metadata.get("id", doc.page_content)


def _metadata_matches(
    metadata: Mapping[str, Any],
    criteria: Mapping[str, Any],
) -> bool:
    for key, expected in criteria.items():
        value = metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if value not in expected:
                return False
        elif value != expected:
            return False
    return True


def dataframe_to_motif_windows(
    frame: pd.DataFrame,
    *,
    window_size: int,
    feature_columns: Optional[Sequence[str]] = None,
    ticker: str,
    timeframe: str,
    default_regime: str = "unspecified",
    regime_resolver: Optional[Callable[[pd.DataFrame], str]] = None,
    normalization: Optional[str] = None,
    extra_metadata: Optional[Callable[[pd.DataFrame], Dict[str, Any]]] = None,
) -> tuple[List[List[float]], List[MotifMetadata], int, List[str]]:
    """Convert a price/indicator dataframe into motif vectors and metadata."""

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    if len(frame) < window_size:
        return [], [], 0, []

    resolved_columns = _resolve_feature_columns(frame, feature_columns)
    if not resolved_columns:
        raise ValueError("No numeric feature columns available for motif windowing.")

    normalization_mode = normalization or "none"
    if normalization_mode not in SUPPORTED_NORMALIZATIONS:
        raise ValueError(
            "Unsupported normalization strategy. Expected one of 'none', 'zscore', "
            "'minmax', or 'first'."
        )

    resolver = regime_resolver
    fallback_regime = default_regime
    if resolver is None and default_regime.lower() == "auto":
        resolver = _default_regime_resolver
        fallback_regime = "unspecified"

    sorted_frame = frame.sort_index()
    selected = sorted_frame[resolved_columns]

    vectors: List[List[float]] = []
    metadatas: List[MotifMetadata] = []
    skipped = 0

    for start in range(0, len(selected) - window_size + 1):
        window = selected.iloc[start : start + window_size]
        if window.isna().values.any():
            skipped += 1
            continue

        values = window.to_numpy(dtype=np.float32, copy=True)
        normalised = _normalize_values(values, normalization_mode)
        if not np.isfinite(normalised).all():
            skipped += 1
            continue

        vector = normalised.reshape(-1).tolist()
        start_timestamp = _format_timestamp(window.index[0])
        end_timestamp = _format_timestamp(window.index[-1])

        regime = fallback_regime
        if resolver is not None:
            try:
                inferred = resolver(window)
            except Exception:
                inferred = fallback_regime
            if inferred:
                regime = inferred

        extras: Dict[str, Any] = {
            "window_index": start,
            "window_size": window_size,
            "features": ",".join(resolved_columns),
            "normalization": normalization_mode,
        }
        if extra_metadata is not None:
            try:
                extras.update(extra_metadata(window))
            except Exception:
                pass

        metadatas.append(
            MotifMetadata(
                ticker=ticker,
                timeframe=timeframe,
                start_date=start_timestamp,
                end_date=end_timestamp,
                regime=regime or fallback_regime,
                extra=extras,
            )
        )
        vectors.append(vector)

    return vectors, metadatas, skipped, list(resolved_columns)


def generate_motif_matches_from_dataframe(
    frame: pd.DataFrame,
    *,
    window_size: int,
    feature_columns: Optional[Sequence[str]] = None,
    ticker: str,
    timeframe: str,
    top_k: int = 5,
    backend: str = "faiss",
    default_regime: str = "unspecified",
    regime_resolver: Optional[Callable[[pd.DataFrame], str]] = None,
    normalization: Optional[str] = None,
    metadata_filter: Optional[Mapping[str, Any]] = None,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedder: Optional[WindowEmbedder] = None,
) -> MotifQueryResult:
    """Build a motif index from ``frame`` and retrieve similar motifs for the latest window."""

    vectors, metadatas, skipped, columns = dataframe_to_motif_windows(
        frame,
        window_size=window_size,
        feature_columns=feature_columns,
        ticker=ticker,
        timeframe=timeframe,
        default_regime=default_regime,
        regime_resolver=regime_resolver,
        normalization=normalization,
    )

    if not vectors or not metadatas:
        raise ValueError(
            "Unable to build motif windows. Ensure the dataset contains enough "
            "rows without missing values for the requested window size."
        )
    if len(vectors) < 2:
        raise ValueError(
            "At least two motif windows are required (one for indexing, one for querying)."
        )

    query_vector = vectors[-1]
    query_metadata = metadatas[-1]
    index_vectors = vectors[:-1]
    index_metadatas = metadatas[:-1]

    if backend.lower() == "faiss":
        store = MotifVectorStore.faiss(len(query_vector), embedder=embedder)
    elif backend.lower() == "chroma":
        store = MotifVectorStore.chroma(
            collection_name=collection_name or "motifs",
            persist_directory=persist_directory,
            embedder=embedder,
        )
    else:
        raise ValueError("backend must be either 'faiss' or 'chroma'")

    store.index_windows(index_vectors, index_metadatas)

    filters: Dict[str, Any] = {
        "ticker": query_metadata.ticker,
        "timeframe": query_metadata.timeframe,
    }
    if metadata_filter:
        for key, value in metadata_filter.items():
            if value is None:
                filters.pop(key, None)
            else:
                filters[key] = value

    matches = store.query_nearest_neighbors(
        query_vector,
        top_k=top_k,
        metadata_filter=filters,
    )

    if backend.lower() == "chroma" and persist_directory:
        persist_fn = getattr(store.vectorstore, "persist", None)
        if callable(persist_fn):
            persist_fn()

    return MotifQueryResult(
        matches=matches,
        query_metadata=query_metadata,
        query_vector=list(query_vector),
        indexed_count=len(index_vectors),
        window_size=window_size,
        feature_columns=columns,
        backend=backend,
        normalization=(normalization or "none"),
        skipped_windows=skipped,
        filters=filters,
    )


def _resolve_feature_columns(
    frame: pd.DataFrame, feature_columns: Optional[Sequence[str]]
) -> List[str]:
    if feature_columns:
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            raise KeyError(
                f"Missing feature columns for motif extraction: {', '.join(missing)}"
            )
        resolved = [
            column
            for column in feature_columns
            if pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not resolved:
            raise ValueError(
                "None of the requested feature columns are numeric; unable to build motifs."
            )
        return resolved

    return [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]


def _normalize_values(values: np.ndarray, mode: str) -> np.ndarray:
    if mode in {"none", "", None}:
        return values

    normalised = np.array(values, dtype=np.float32, copy=True)
    if mode == "zscore":
        mean = normalised.mean(axis=0)
        std = normalised.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        normalised = (normalised - mean) / std
    elif mode == "minmax":
        min_vals = normalised.min(axis=0)
        ranges = normalised.max(axis=0) - min_vals
        ranges = np.where(ranges == 0, 1.0, ranges)
        normalised = (normalised - min_vals) / ranges
    elif mode == "first":
        baseline = normalised[0]
        denom = np.where(baseline == 0, 1.0, baseline)
        normalised = (normalised - baseline) / denom
    else:
        raise ValueError(f"Unsupported normalization strategy: {mode}")
    return normalised


def _format_timestamp(value: Any) -> str:
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()  # type: ignore[call-arg]
        except TypeError:
            pass
    return str(value)


def _default_regime_resolver(window: pd.DataFrame) -> str:
    close = window.get("Close")
    if close is None or close.empty:
        return "unspecified"

    returns = close.pct_change().dropna()
    volatility = float(returns.std()) if not returns.empty else 0.0
    if volatility < 0.005:
        vol_label = "low_vol"
    elif volatility < 0.015:
        vol_label = "medium_vol"
    else:
        vol_label = "high_vol"

    trend_label = "sideways"
    ema_200 = window.get("EMA_200")
    ema_30 = window.get("EMA_30")
    last_close = float(close.iloc[-1])
    first_close = float(close.iloc[0])

    if ema_200 is not None and not ema_200.isna().any():
        trend_label = (
            "uptrend" if last_close >= float(ema_200.iloc[-1]) else "downtrend"
        )
    elif ema_30 is not None and not ema_30.isna().any():
        trend_label = "uptrend" if last_close >= float(ema_30.iloc[-1]) else "downtrend"
    else:
        if last_close > first_close * 1.01:
            trend_label = "uptrend"
        elif last_close < first_close * 0.99:
            trend_label = "downtrend"

    return f"{trend_label}_{vol_label}"


def _run_faiss_similarity(vectorstore: Any, embedding: Sequence[float], k: int):
    """Execute FAISS similarity search across LangChain API variants."""

    candidates = []
    for method_name in (
        "similarity_search_with_score_by_vector",
        "similarity_search_by_vector_with_score",
        "similarity_search_by_vector_with_relevance_scores",
    ):
        method = getattr(vectorstore, method_name, None)
        if callable(method):
            candidates = method(embedding, k=k)
            break
    else:
        method = getattr(vectorstore, "similarity_search_by_vector", None)
        if callable(method):
            docs = method(embedding, k=k)
            candidates = [(doc, 0.0) for doc in docs]
        else:
            raise AttributeError(
                "FAISS vector store is missing similarity search helpers expected by LangChain."
            )

    return candidates


def retrieve_top_k_for_today(
    motif_store: MotifVectorStore,
    today_window: Sequence[float],
    *,
    ticker: str,
    timeframe: str,
    regime: Optional[str] = None,
    k: int = 5,
) -> List[MotifMatch]:
    """Convenience wrapper to fetch today's top-K motifs with optional filters."""

    filters: Dict[str, Any] = {
        "ticker": ticker,
        "timeframe": timeframe,
    }
    if regime is not None:
        filters["regime"] = regime

    return motif_store.query_nearest_neighbors(
        today_window,
        top_k=k,
        metadata_filter=filters,
    )


__all__ = [
    "IdentityWindowEmbedder",
    "MotifMatch",
    "MotifMetadata",
    "MotifVectorStore",
    "MotifQueryResult",
    "WindowEmbedder",
    "DEFAULT_FEATURE_COLUMNS",
    "SUPPORTED_NORMALIZATIONS",
    "dataframe_to_motif_windows",
    "generate_motif_matches_from_dataframe",
    "retrieve_top_k_for_today",
]
