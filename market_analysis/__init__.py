"""Reusable helpers for market technical analysis workflows."""

from .data import DataFetcher, DatasetRequest, ensure_dataset, is_stale
from .history import AnalysisHistory, make_cache_key, make_entry_id
from .llm import (LLMAnalysisResult, LLMConfig, format_prompt,
                  generate_llm_analysis)
from .motifs import (DEFAULT_FEATURE_COLUMNS, SUPPORTED_NORMALIZATIONS,
                     IdentityWindowEmbedder, MotifMatch, MotifMetadata,
                     MotifQueryResult, MotifVectorStore,
                     dataframe_to_motif_windows,
                     generate_motif_matches_from_dataframe,
                     retrieve_top_k_for_today)
from .summary import build_multi_timeframe_summary, build_technical_summary

__all__ = [
    "DataFetcher",
    "DatasetRequest",
    "ensure_dataset",
    "is_stale",
    "AnalysisHistory",
    "make_cache_key",
    "LLMConfig",
    "format_prompt",
    "generate_llm_analysis",
    "LLMAnalysisResult",
    "build_technical_summary",
    "build_multi_timeframe_summary",
    "make_entry_id",
    "MotifMetadata",
    "MotifMatch",
    "MotifVectorStore",
    "IdentityWindowEmbedder",
    "MotifQueryResult",
    "DEFAULT_FEATURE_COLUMNS",
    "SUPPORTED_NORMALIZATIONS",
    "dataframe_to_motif_windows",
    "generate_motif_matches_from_dataframe",
    "retrieve_top_k_for_today",
]
